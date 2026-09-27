import json
import sqlite3
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

from nexus.agent.client import PostToolExecutionError
from nexus.agent.session import AgentSession, SessionState
from nexus.core.deadlines import VIETNAM_TIMEZONE, format_deadline
from nexus.storage.sqlite_db import create_task, initialize_database, list_tasks, set_task_deadline


def tool_response(name, arguments):
    return {
        "choices": [{
            "finish_reason": "tool_calls",
            "message": {
                "content": None,
                "tool_calls": [{
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(arguments)},
                }],
            },
        }]
    }


class DeadlineSessionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.database_path = Path(self.directory.name) / "tasks.db"
        initialize_database(self.database_path)
        create_task(self.database_path, "mua sữa")
        self.reference = datetime(2026, 9, 26, 23, 59, tzinfo=VIETNAM_TIMEZONE)
        self.session = AgentSession(self.database_path, clock=lambda: self.reference)

    def test_direct_deadline_runs_model_policy_tool_and_formatter(self):
        result = self.session.run_turn(
            "Đặt hạn việc 1 lúc 8 giờ sáng mai.",
            Mock(return_value=tool_response("set_task_deadline", {"id": 1, "when": "8 giờ sáng mai"})),
        )
        self.assertEqual(result["status"], "executed")
        self.assertEqual(result["source"], "model")
        self.assertEqual(self.session.state, SessionState.IDLE)
        self.assertEqual(format_deadline(list_tasks(self.database_path)[0].due_at), "27/09/2026 08:00")

    def test_missing_id_keeps_raw_time_and_original_reference(self):
        later = datetime(2026, 9, 27, 0, 1, tzinfo=VIETNAM_TIMEZONE)
        clock = Mock(side_effect=[self.reference, later])
        session = AgentSession(self.database_path, clock=clock)
        first = session.run_turn("Đặt hạn lúc 8 giờ sáng mai.", Mock(side_effect=AssertionError))
        self.assertEqual(first["status"], "needs_clarification")
        self.assertEqual(session.state, SessionState.AWAITING_DEADLINE_ID)
        self.assertEqual(session.pending_deadline_text, "8 giờ sáng mai")
        second = session.run_turn("#1", Mock(side_effect=AssertionError))
        self.assertEqual(second["status"], "executed")
        self.assertEqual(format_deadline(list_tasks(self.database_path)[0].due_at), "27/09/2026 08:00")

    def test_missing_id_then_missing_time_then_value(self):
        first = self.session.run_turn("Đặt hạn", Mock(side_effect=AssertionError))
        self.assertEqual(first["reply"], "Bạn muốn đặt hạn cho việc có ID nào?")
        second = self.session.run_turn("việc 1", Mock(side_effect=AssertionError))
        self.assertEqual(self.session.state, SessionState.AWAITING_DEADLINE_TEXT)
        self.assertEqual(second["reply"], "Bạn muốn đặt thời hạn khi nào? Ví dụ: 8 giờ sáng mai.")
        third = self.session.run_turn("8 giờ sáng mai", Mock(side_effect=AssertionError))
        self.assertEqual(third["status"], "executed")
        self.assertEqual(self.session.state, SessionState.IDLE)

    def test_missing_or_invalid_time_stays_pending(self):
        first = self.session.run_turn("Đặt hạn việc 1.", Mock(side_effect=AssertionError))
        self.assertEqual(first["status"], "needs_clarification")
        self.assertEqual(self.session.pending_deadline_id, 1)
        invalid = self.session.run_turn("tuần sau", Mock(side_effect=AssertionError))
        self.assertEqual(invalid["status"], "needs_clarification")
        self.assertEqual(self.session.state, SessionState.AWAITING_DEADLINE_TEXT)
        done = self.session.run_turn("27/09/2026 08:00", Mock(side_effect=AssertionError))
        self.assertEqual(done["status"], "executed")

    def test_cancel_and_new_command_replace_pending(self):
        self.session.run_turn("Đặt hạn việc 1", Mock(side_effect=AssertionError))
        cancelled = self.session.run_turn("hủy", Mock(side_effect=AssertionError))
        self.assertEqual(cancelled["status"], "cancelled")
        self.assertIsNone(list_tasks(self.database_path)[0].due_at)

        self.session.run_turn("Đặt hạn việc 1", Mock(side_effect=AssertionError))
        listed = self.session.run_turn(
            "Xem danh sách",
            Mock(return_value=tool_response("list_tasks", {})),
        )
        self.assertEqual(listed["status"], "executed")
        self.assertEqual(self.session.state, SessionState.IDLE)

    def test_database_error_keeps_pending_and_formatter_error_clears_it(self):
        self.session.run_turn("Đặt hạn việc 1", Mock(side_effect=AssertionError))
        with patch("nexus.agent.session.execute_tool", side_effect=sqlite3.OperationalError("locked")):
            with self.assertRaises(sqlite3.OperationalError):
                self.session.run_turn("8 giờ sáng mai", Mock(side_effect=AssertionError))
        self.assertEqual(self.session.state, SessionState.AWAITING_DEADLINE_TEXT)
        self.assertIsNone(list_tasks(self.database_path)[0].due_at)

        with patch("nexus.agent.session.format_tool_result", side_effect=ValueError("broken")):
            with self.assertRaises(PostToolExecutionError):
                self.session.run_turn("8 giờ sáng mai", Mock(side_effect=AssertionError))
        self.assertEqual(self.session.state, SessionState.IDLE)
        self.assertIsNotNone(list_tasks(self.database_path)[0].due_at)

    def test_invalid_saved_time_asks_for_time_after_id(self):
        first = self.session.run_turn(
            "Đặt hạn lúc tuần sau", Mock(side_effect=AssertionError)
        )
        self.assertEqual(first["status"], "needs_clarification")
        self.assertEqual(self.session.state, SessionState.AWAITING_DEADLINE_ID)
        second = self.session.run_turn("1", Mock(side_effect=AssertionError))
        self.assertEqual(second["status"], "needs_clarification")
        self.assertEqual(self.session.state, SessionState.AWAITING_DEADLINE_TEXT)
        third = self.session.run_turn(
            "27/09/2026 08:00", Mock(side_effect=AssertionError)
        )
        self.assertEqual(third["status"], "executed")

    def test_delete_snapshot_and_stale_reply_include_deadline(self):
        set_task_deadline(self.database_path, 1, 1790470800)
        first = self.session.run_turn(
            "Xóa việc 1",
            Mock(return_value=tool_response("delete_task", {"id": 1})),
        )
        self.assertIn("hạn 27/09/2026 08:00", first["reply"])
        set_task_deadline(self.database_path, 1, 1790562600)
        stale = self.session.run_turn("có", Mock(side_effect=AssertionError))
        self.assertEqual(stale["status"], "needs_confirmation")
        self.assertIn("hạn 28/09/2026 09:30", stale["reply"])
        self.assertEqual(self.session.state, SessionState.AWAITING_DELETE_CONFIRMATION)

    def test_sessions_do_not_share_pending_deadline(self):
        other = AgentSession(self.database_path, clock=lambda: self.reference)
        self.session.run_turn("Đặt hạn việc 1", Mock(side_effect=AssertionError))
        response = {
            "choices": [{"finish_reason": "stop", "message": {"content": "1"}}]
        }
        result = other.run_turn("8 giờ sáng mai", Mock(return_value=response))
        self.assertEqual(result["status"], "no_tool")
        self.assertEqual(other.state, SessionState.IDLE)
        self.assertEqual(self.session.state, SessionState.AWAITING_DEADLINE_TEXT)


if __name__ == "__main__":
    unittest.main()
