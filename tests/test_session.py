"""Stateful CREATE clarification without expanding the tool scope."""

import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from nexus.agent.client import PostToolExecutionError
from nexus.agent.session import AgentSession
from nexus.storage.sqlite_db import create_task, initialize_database, list_tasks


def tool_response(name, arguments):
    return {"choices": [{"finish_reason": "tool_calls", "message": {
        "role": "assistant", "content": None,
        "tool_calls": [{"type": "function", "id": "call-1", "function": {
            "name": name, "arguments": arguments,
        }}],
    }}]}


class AgentSessionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.database_path = Path(self.directory.name) / "tasks.db"
        initialize_database(self.database_path)
        self.session = AgentSession(self.database_path)

    def test_missing_create_then_multiline_content_uses_prior_authorization(self):
        generate = Mock(side_effect=AssertionError("model must not be called"))
        first = self.session.run_turn("Thêm việc.", generate)
        self.assertEqual(first["status"], "needs_clarification")
        self.assertTrue(self.session.pending_create)
        second = self.session.run_turn("mua sữa\ngọi mẹ", generate)
        self.assertEqual(second["status"], "executed")
        self.assertEqual(second["source"], "session_continuation")
        self.assertEqual(second["proposed_calls"], [])
        self.assertEqual(second["authorized_calls"][0]["reason"], "session_continuation")
        self.assertEqual([t.content for t in list_tasks(self.database_path)],
                         ["mua sữa", "gọi mẹ"])
        self.assertFalse(self.session.pending_create)
        generate.assert_not_called()

    def test_blank_followup_keeps_pending_and_cancel_clears_it(self):
        generate = Mock(side_effect=AssertionError("model must not be called"))
        self.session.run_turn("Ghi việc:", generate)
        blank = self.session.run_turn(" \n ", generate)
        self.assertEqual(blank["status"], "needs_clarification")
        self.assertTrue(self.session.pending_create)
        cancelled = self.session.run_turn("Thôi.", generate)
        self.assertEqual(cancelled["status"], "cancelled")
        self.assertFalse(self.session.pending_create)
        self.assertEqual(list_tasks(self.database_path), [])
        generate.assert_not_called()

    def test_new_list_command_replaces_pending_request(self):
        create_task(self.database_path, "việc cũ")
        self.session.run_turn("Thêm việc", Mock(side_effect=AssertionError))
        generate = Mock(return_value=tool_response("list_tasks", "{}"))
        result = self.session.run_turn("Xem danh sách", generate)
        self.assertEqual(result["source"], "model")
        self.assertEqual(result["status"], "executed")
        self.assertIn("việc cũ", result["reply"])
        self.assertFalse(self.session.pending_create)
        generate.assert_called_once()
        self.assertEqual([t.content for t in list_tasks(self.database_path)], ["việc cũ"])

    def test_list_framing_rejected_by_policy_is_not_saved_as_followup(self):
        self.session.run_turn("Thêm việc", Mock(side_effect=AssertionError))
        generate = Mock(return_value=tool_response("list_tasks", "{}"))
        result = self.session.run_turn("Xem danh sách việc đã thêm", generate)
        self.assertEqual(result["status"], "rejected")
        self.assertFalse(self.session.pending_create)
        self.assertEqual(list_tasks(self.database_path), [])
        generate.assert_called_once()

    def test_negated_command_cancels_pending_without_saving(self):
        self.session.run_turn("Thêm việc", Mock(side_effect=AssertionError))
        generate = Mock(side_effect=AssertionError("model must not be called"))
        result = self.session.run_turn("Đừng thêm việc nữa", generate)
        self.assertEqual(result["status"], "cancelled")
        self.assertFalse(self.session.pending_create)
        self.assertEqual(list_tasks(self.database_path), [])
        generate.assert_not_called()

    def test_new_create_command_replaces_pending_request(self):
        self.session.run_turn("Thêm việc", Mock(side_effect=AssertionError))
        generate = Mock(return_value=tool_response("create_task", '{"content":"mua táo"}'))
        result = self.session.run_turn("Thêm việc: mua táo", generate)
        self.assertEqual(result["source"], "model")
        self.assertEqual(result["status"], "executed")
        self.assertFalse(self.session.pending_create)
        self.assertEqual([t.content for t in list_tasks(self.database_path)], ["mua táo"])

    def test_new_create_framing_rejected_by_policy_is_not_saved_as_followup(self):
        self.session.run_turn("Thêm việc", Mock(side_effect=AssertionError))
        generate = Mock(return_value=tool_response("create_task", '{"content":"mua sữa"}'))
        result = self.session.run_turn("Thêm việc ví dụ mua sữa", generate)
        self.assertEqual(result["status"], "rejected")
        self.assertFalse(self.session.pending_create)
        self.assertEqual(list_tasks(self.database_path), [])
        generate.assert_called_once()

    def test_unsupported_command_replaces_pending_without_saving(self):
        self.session.run_turn("Thêm việc", Mock(side_effect=AssertionError))
        generate = Mock(side_effect=AssertionError("model must not be called"))
        result = self.session.run_turn("Xóa task cũ", generate)
        self.assertEqual(result["status"], "rejected")
        self.assertIn("chỉ hỗ trợ", result["reply"])
        self.assertFalse(self.session.pending_create)
        self.assertEqual(list_tasks(self.database_path), [])
        generate.assert_not_called()

    def test_task_text_that_starts_with_other_verbs_is_saved(self):
        generate = Mock(side_effect=AssertionError("model must not be called"))
        for content in ("sửa xe", "xóa file nháp", "đừng quên mua sữa", "xem phim"):
            with self.subTest(content=content):
                self.session.run_turn("Thêm việc", generate)
                result = self.session.run_turn(content, generate)
                self.assertEqual(result["status"], "executed")
        self.assertEqual([t.content for t in list_tasks(self.database_path)],
                         ["sửa xe", "xóa file nháp", "đừng quên mua sữa", "xem phim"])
        generate.assert_not_called()

    def test_database_failure_rolls_back_and_keeps_pending_for_explicit_retry(self):
        self.session.run_turn("Thêm việc", Mock(side_effect=AssertionError))
        with sqlite3.connect(self.database_path) as connection:
            connection.execute("""
                CREATE TRIGGER fail_second BEFORE INSERT ON tasks
                WHEN NEW.content = 'lỗi'
                BEGIN SELECT RAISE(ABORT, 'simulated failure'); END
            """)
        with self.assertRaises(sqlite3.IntegrityError):
            self.session.run_turn("việc mới\nlỗi", Mock(side_effect=AssertionError))
        self.assertTrue(self.session.pending_create)
        self.assertEqual(list_tasks(self.database_path), [])
        retried = self.session.run_turn("việc mới", Mock(side_effect=AssertionError))
        self.assertEqual(retried["status"], "executed")
        self.assertEqual([t.content for t in list_tasks(self.database_path)], ["việc mới"])

    def test_formatter_failure_after_commit_clears_pending(self):
        self.session.run_turn("Thêm việc", Mock(side_effect=AssertionError))
        with patch("nexus.agent.session.format_tool_result", side_effect=ValueError("broken")):
            with self.assertRaises(PostToolExecutionError) as raised:
                self.session.run_turn("mua sữa", Mock(side_effect=AssertionError))
        self.assertEqual(raised.exception.stage, "response_formatting")
        self.assertEqual(len(raised.exception.executed_calls), 1)
        self.assertFalse(self.session.pending_create)
        self.assertEqual([t.content for t in list_tasks(self.database_path)], ["mua sữa"])

    def test_pending_state_is_local_to_one_session(self):
        self.session.run_turn("Thêm việc", Mock(side_effect=AssertionError))
        other = AgentSession(self.database_path)
        direct = other.run_turn("mua sữa", Mock(return_value={"choices": [{
            "finish_reason": "stop", "message": {"role": "assistant", "content": "Đã thêm"},
        }]}))
        self.assertEqual(direct["status"], "no_tool")
        self.assertFalse(other.pending_create)
        self.assertTrue(self.session.pending_create)
        self.assertEqual(list_tasks(self.database_path), [])


if __name__ == "__main__":
    unittest.main()
