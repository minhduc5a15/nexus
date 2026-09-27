import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock

from nexus.agent.session import AgentSession, SessionState
from nexus.core.deadlines import VIETNAM_TIMEZONE
from nexus.storage.sqlite_db import create_task, initialize_database, set_task_deadline


def tool_response(name, arguments):
    return {
        "choices": [{
            "finish_reason": "tool_calls",
            "message": {
                "content": None,
                "tool_calls": [{
                    "id": "due-1",
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(arguments),
                    },
                }],
            },
        }]
    }


class DeadlineQuerySessionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.database_path = Path(self.directory.name) / "tasks.db"
        initialize_database(self.database_path)
        task = create_task(self.database_path, "mua sữa")
        self.reference = datetime(2026, 9, 27, 10, 0, tzinfo=VIETNAM_TIMEZONE)
        due_at = int(
            datetime(2026, 9, 27, 8, 0, tzinfo=VIETNAM_TIMEZONE).timestamp()
        )
        set_task_deadline(self.database_path, task.id, due_at)
        self.session = AgentSession(
            self.database_path, prompt_version="v13", clock=lambda: self.reference
        )

    def test_direct_query_runs_model_policy_tool_and_formatter(self):
        result = self.session.run_turn(
            "Xem việc đến hạn hôm nay.",
            Mock(return_value=tool_response(
                "list_tasks_by_deadline", {"scope": "today"}
            )),
        )
        self.assertEqual(result["status"], "executed")
        self.assertEqual(result["source"], "model")
        self.assertEqual(result["authorized_calls"][0]["reason"], "explicit_deadline_query")
        self.assertIn("[1] [ ] mua sữa", result["reply"])
        self.assertEqual(self.session.state, SessionState.IDLE)

    def test_missing_scope_is_remembered_and_followup_does_not_call_model(self):
        first = self.session.run_turn(
            "Xem việc theo hạn", Mock(side_effect=AssertionError)
        )
        self.assertEqual(first["status"], "needs_clarification")
        self.assertEqual(self.session.state, SessionState.AWAITING_DEADLINE_SCOPE)
        invalid = self.session.run_turn(
            "tuần này", Mock(side_effect=AssertionError)
        )
        self.assertEqual(invalid["status"], "needs_clarification")
        self.assertEqual(self.session.state, SessionState.AWAITING_DEADLINE_SCOPE)
        done = self.session.run_turn(
            "hôm nay", Mock(side_effect=AssertionError)
        )
        self.assertEqual(done["status"], "executed")
        self.assertEqual(done["source"], "session_continuation")
        self.assertEqual(done["calls"][0]["arguments"], {"scope": "today"})
        self.assertEqual(self.session.state, SessionState.IDLE)

    def test_cancel_and_new_command_replace_pending(self):
        self.session.run_turn("Xem việc theo hạn", Mock(side_effect=AssertionError))
        cancelled = self.session.run_turn("hủy", Mock(side_effect=AssertionError))
        self.assertEqual(cancelled["status"], "cancelled")
        self.assertEqual(self.session.state, SessionState.IDLE)

        self.session.run_turn("Xem việc theo hạn", Mock(side_effect=AssertionError))
        listed = self.session.run_turn(
            "Xem danh sách",
            Mock(return_value=tool_response("list_tasks", {})),
        )
        self.assertEqual(listed["status"], "executed")
        self.assertEqual(self.session.state, SessionState.IDLE)

    def test_sessions_do_not_share_pending_scope(self):
        other = AgentSession(
            self.database_path, prompt_version="v13", clock=lambda: self.reference
        )
        self.session.run_turn("Xem việc theo hạn", Mock(side_effect=AssertionError))
        self.assertEqual(self.session.state, SessionState.AWAITING_DEADLINE_SCOPE)
        self.assertEqual(other.state, SessionState.IDLE)
        result = other.run_turn("hôm nay", Mock(return_value={
            "choices": [{"finish_reason": "stop", "message": {"content": "?"}}]
        }))
        self.assertEqual(result["status"], "no_tool")
        self.assertEqual(self.session.state, SessionState.AWAITING_DEADLINE_SCOPE)


if __name__ == "__main__":
    unittest.main()
