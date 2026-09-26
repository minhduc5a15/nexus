"""Stateful CREATE clarification without expanding the tool scope."""

import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from nexus.agent.client import PostToolExecutionError
from nexus.agent.session import AgentSession, SessionState
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
        self.assertEqual(first["session_state_before"], "idle")
        self.assertEqual(first["session_state_after"], "awaiting_create_content")
        self.assertTrue(self.session.pending_create)
        second = self.session.run_turn("mua sữa\ngọi mẹ", generate)
        self.assertEqual(second["status"], "executed")
        self.assertEqual(second["session_state_before"], "awaiting_create_content")
        self.assertEqual(second["session_state_after"], "idle")
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
        self.assertEqual(blank["session_state_before"], "awaiting_create_content")
        self.assertEqual(blank["session_state_after"], "awaiting_create_content")
        self.assertTrue(self.session.pending_create)
        cancelled = self.session.run_turn("Thôi.", generate)
        self.assertEqual(cancelled["status"], "cancelled")
        self.assertEqual(cancelled["session_state_before"], "awaiting_create_content")
        self.assertEqual(cancelled["session_state_after"], "idle")
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
        self.assertEqual(result["session_state_before"], "awaiting_create_content")
        self.assertEqual(result["session_state_after"], "idle")
        generate.assert_called_once()
        self.assertEqual([t.content for t in list_tasks(self.database_path)], ["việc cũ"])

    def test_natural_list_question_replaces_pending_instead_of_becoming_task(self):
        create_task(self.database_path, "ăn cơm lúc 8 giờ sáng")
        self.session.run_turn("Thêm việc giúp tôi", Mock(side_effect=AssertionError))
        generate = Mock(return_value=tool_response("list_tasks", "{}"))

        result = self.session.run_turn(
            "Hiện tại tôi đang có những việc gì nhỉ?", generate
        )

        self.assertEqual(result["source"], "model")
        self.assertEqual(result["status"], "executed")
        self.assertIn("ăn cơm lúc 8 giờ sáng", result["reply"])
        self.assertFalse(self.session.pending_create)
        generate.assert_called_once()
        self.assertEqual(
            [task.content for task in list_tasks(self.database_path)],
            ["ăn cơm lúc 8 giờ sáng"],
        )

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
        result = self.session.run_turn("Sắp xếp task cũ", generate)
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
        self.assertEqual(self.session.state, SessionState.AWAITING_CREATE_CONTENT)
        self.assertTrue(self.session.pending_create)
        self.assertEqual(list_tasks(self.database_path), [])
        retried = self.session.run_turn("việc mới", Mock(side_effect=AssertionError))
        self.assertEqual(retried["status"], "executed")
        self.assertEqual(retried["session_state_before"], "awaiting_create_content")
        self.assertEqual(retried["session_state_after"], "idle")
        self.assertEqual([t.content for t in list_tasks(self.database_path)], ["việc mới"])

    def test_formatter_failure_after_commit_clears_pending(self):
        self.session.run_turn("Thêm việc", Mock(side_effect=AssertionError))
        with patch("nexus.agent.session.format_tool_result", side_effect=ValueError("broken")):
            with self.assertRaises(PostToolExecutionError) as raised:
                self.session.run_turn("mua sữa", Mock(side_effect=AssertionError))
        self.assertEqual(raised.exception.stage, "response_formatting")
        self.assertEqual(len(raised.exception.executed_calls), 1)
        self.assertFalse(self.session.pending_create)
        self.assertEqual(self.session.state, SessionState.IDLE)
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
        self.assertEqual(other.state, SessionState.IDLE)
        self.assertEqual(self.session.state, SessionState.AWAITING_CREATE_CONTENT)
        self.assertEqual(list_tasks(self.database_path), [])

    def test_direct_complete_runs_model_proposal_for_exact_id(self):
        task = create_task(self.database_path, "mua sữa")
        generate = Mock(return_value=tool_response("complete_task", '{"id":1}'))
        result = self.session.run_turn("Hoàn thành việc 1.", generate)
        self.assertEqual(result["status"], "executed")
        self.assertEqual(result["calls"][0]["result"]["status"], "completed")
        self.assertTrue(list_tasks(self.database_path)[0].completed)
        self.assertEqual(task.id, 1)

    def test_missing_complete_id_then_supported_followup_forms(self):
        generate = Mock(side_effect=AssertionError("model must not be called"))
        for reply in ("1", "#2", "việc 3", "task 4"):
            create_task(self.database_path, f"việc {reply}")
            first = self.session.run_turn("Hoàn thành việc", generate)
            self.assertEqual(first["status"], "needs_clarification")
            self.assertEqual(first["reply"], "Bạn muốn hoàn thành việc có ID nào?")
            self.assertEqual(self.session.state, SessionState.AWAITING_COMPLETE_ID)
            second = self.session.run_turn(reply, generate)
            self.assertEqual(second["status"], "executed")
            self.assertEqual(second["source"], "session_continuation")
            self.assertEqual(second["authorized_calls"][0]["reason"], "session_continuation")
            self.assertEqual(self.session.state, SessionState.IDLE)
        self.assertTrue(all(task.completed for task in list_tasks(self.database_path)))
        generate.assert_not_called()

    def test_multiple_ids_asks_for_one_and_invalid_followup_keeps_pending(self):
        create_task(self.database_path, "a")
        create_task(self.database_path, "b")
        generate = Mock(side_effect=AssertionError("model must not be called"))
        first = self.session.run_turn("Hoàn thành việc 1 và 2", generate)
        self.assertEqual(first["status"], "needs_clarification")
        self.assertEqual(self.session.state, SessionState.AWAITING_COMPLETE_ID)
        invalid = self.session.run_turn("việc đầu tiên", generate)
        self.assertEqual(invalid["status"], "needs_clarification")
        self.assertEqual(self.session.state, SessionState.AWAITING_COMPLETE_ID)
        done = self.session.run_turn("#2", generate)
        self.assertEqual(done["status"], "executed")
        self.assertEqual([task.completed for task in list_tasks(self.database_path)], [False, True])

    def test_cancel_and_new_commands_replace_pending_complete(self):
        generate = Mock(side_effect=AssertionError("model must not be called"))
        self.session.run_turn("Hoàn thành task", generate)
        cancelled = self.session.run_turn("hủy", generate)
        self.assertEqual(cancelled["status"], "cancelled")
        self.assertEqual(self.session.state, SessionState.IDLE)

        self.session.run_turn("Hoàn thành task", generate)
        listed = Mock(return_value=tool_response("list_tasks", "{}"))
        result = self.session.run_turn("Xem danh sách", listed)
        self.assertEqual(result["status"], "executed")
        self.assertEqual(self.session.state, SessionState.IDLE)

        self.session.run_turn("Hoàn thành task", generate)
        created = Mock(return_value=tool_response("create_task", '{"content":"mua táo"}'))
        result = self.session.run_turn("Thêm việc: mua táo", created)
        self.assertEqual(result["status"], "executed")
        self.assertEqual(self.session.state, SessionState.IDLE)

    def test_complete_database_error_keeps_pending_but_formatter_error_clears_it(self):
        create_task(self.database_path, "mua sữa")
        self.session.run_turn("Hoàn thành việc", Mock(side_effect=AssertionError))
        with patch(
            "nexus.agent.session.execute_tool",
            side_effect=sqlite3.OperationalError("before commit"),
        ):
            with self.assertRaises(sqlite3.OperationalError):
                self.session.run_turn("1", Mock(side_effect=AssertionError))
        self.assertEqual(self.session.state, SessionState.AWAITING_COMPLETE_ID)
        self.assertFalse(list_tasks(self.database_path)[0].completed)

        with patch(
            "nexus.agent.session.format_tool_result", side_effect=ValueError("after commit")
        ):
            with self.assertRaises(PostToolExecutionError):
                self.session.run_turn("1", Mock(side_effect=AssertionError))
        self.assertEqual(self.session.state, SessionState.IDLE)
        self.assertTrue(list_tasks(self.database_path)[0].completed)

    def test_complete_pending_state_is_not_shared(self):
        create_task(self.database_path, "mua sữa")
        self.session.run_turn("Hoàn thành việc", Mock(side_effect=AssertionError))
        other = AgentSession(self.database_path)
        response = {"choices": [{
            "finish_reason": "stop",
            "message": {"role": "assistant", "content": "Không rõ."},
        }]}
        result = other.run_turn("1", Mock(return_value=response))
        self.assertEqual(result["status"], "no_tool")
        self.assertEqual(other.state, SessionState.IDLE)
        self.assertEqual(self.session.state, SessionState.AWAITING_COMPLETE_ID)
        self.assertFalse(list_tasks(self.database_path)[0].completed)

    def test_two_complete_calls_are_rejected_before_any_update(self):
        create_task(self.database_path, "một")
        create_task(self.database_path, "hai")
        response = {"choices": [{
            "finish_reason": "tool_calls",
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"type": "function", "id": "a", "function": {
                        "name": "complete_task", "arguments": '{"id":1}'
                    }},
                    {"type": "function", "id": "b", "function": {
                        "name": "complete_task", "arguments": '{"id":2}'
                    }},
                ],
            },
        }]}
        result = self.session.run_turn(
            "Hoàn thành việc 1.", Mock(return_value=response)
        )
        self.assertEqual(result["status"], "rejected")
        self.assertEqual(result["calls"], [])
        self.assertEqual([task.completed for task in list_tasks(self.database_path)], [False, False])

    def test_direct_edit_runs_model_proposal_and_preserves_completion(self):
        task = create_task(self.database_path, "mua sữa")
        from nexus.storage.sqlite_db import complete_task
        complete_task(self.database_path, task.id)
        generate = Mock(return_value=tool_response(
            "update_task", '{"id":1,"content":"mua sữa không đường"}'
        ))
        result = self.session.run_turn(
            "Sửa việc 1 thành mua sữa không đường", generate
        )
        self.assertEqual(result["status"], "executed")
        self.assertEqual(result["calls"][0]["result"]["status"], "updated")
        self.assertEqual(list_tasks(self.database_path)[0].content, "mua sữa không đường")
        self.assertTrue(list_tasks(self.database_path)[0].completed)

    def test_edit_missing_content_uses_free_followup(self):
        create_task(self.database_path, "mua sữa")
        generate = Mock(side_effect=AssertionError("model must not be called"))
        first = self.session.run_turn("Sửa việc 1", generate)
        self.assertEqual(first["status"], "needs_clarification")
        self.assertEqual(self.session.state, SessionState.AWAITING_EDIT_CONTENT)
        self.assertEqual(self.session.pending_edit_id, 1)
        self.assertIn("việc 1", first["reply"])
        second = self.session.run_turn("mua sữa không đường", generate)
        self.assertEqual(second["status"], "executed")
        self.assertEqual(second["source"], "session_continuation")
        self.assertEqual(self.session.state, SessionState.IDLE)
        self.assertEqual(list_tasks(self.database_path)[0].content, "mua sữa không đường")
        generate.assert_not_called()

    def test_edit_missing_id_preserves_content_then_updates_selected_id(self):
        create_task(self.database_path, "một")
        create_task(self.database_path, "hai")
        generate = Mock(side_effect=AssertionError("model must not be called"))
        first = self.session.run_turn("Sửa việc thành nội dung mới", generate)
        self.assertEqual(self.session.state, SessionState.AWAITING_EDIT_ID)
        self.assertEqual(self.session.pending_edit_content, "nội dung mới")
        self.assertEqual(first["reply"], "Bạn muốn sửa việc có ID nào?")
        second = self.session.run_turn("#2", generate)
        self.assertEqual(second["status"], "executed")
        self.assertEqual(
            [task.content for task in list_tasks(self.database_path)],
            ["một", "nội dung mới"],
        )

    def test_edit_missing_both_collects_id_then_content(self):
        create_task(self.database_path, "cũ")
        generate = Mock(side_effect=AssertionError("model must not be called"))
        first = self.session.run_turn("Sửa việc", generate)
        self.assertEqual(first["session_state_after"], "awaiting_edit_id")
        second = self.session.run_turn("việc 1", generate)
        self.assertEqual(second["status"], "needs_clarification")
        self.assertEqual(second["session_state_after"], "awaiting_edit_content")
        self.assertIn("việc 1", second["reply"])
        third = self.session.run_turn("nội dung mới", generate)
        self.assertEqual(third["status"], "executed")
        self.assertEqual(list_tasks(self.database_path)[0].content, "nội dung mới")

    def test_edit_multiple_ids_asks_for_one_and_cancel_clears_fields(self):
        create_task(self.database_path, "một")
        create_task(self.database_path, "hai")
        generate = Mock(side_effect=AssertionError("model must not be called"))
        first = self.session.run_turn("Sửa việc 1 và 2 thành mới", generate)
        self.assertEqual(first["status"], "needs_clarification")
        self.assertEqual(self.session.state, SessionState.AWAITING_EDIT_ID)
        self.assertEqual(self.session.pending_edit_content, "mới")
        cancelled = self.session.run_turn("thôi", generate)
        self.assertEqual(cancelled["status"], "cancelled")
        self.assertEqual(self.session.state, SessionState.IDLE)
        self.assertIsNone(self.session.pending_edit_id)
        self.assertIsNone(self.session.pending_edit_content)
        self.assertEqual([task.content for task in list_tasks(self.database_path)], ["một", "hai"])

    def test_new_list_replaces_pending_edit(self):
        create_task(self.database_path, "cũ")
        self.session.run_turn("Sửa việc 1", Mock(side_effect=AssertionError))
        generate = Mock(return_value=tool_response("list_tasks", "{}"))
        result = self.session.run_turn("Xem danh sách", generate)
        self.assertEqual(result["status"], "executed")
        self.assertEqual(self.session.state, SessionState.IDLE)
        self.assertEqual(list_tasks(self.database_path)[0].content, "cũ")

    def test_edit_database_error_keeps_pending_and_formatter_error_clears_it(self):
        create_task(self.database_path, "cũ")
        self.session.run_turn("Sửa việc 1", Mock(side_effect=AssertionError))
        with patch(
            "nexus.agent.session.execute_tool",
            side_effect=sqlite3.OperationalError("before commit"),
        ):
            with self.assertRaises(sqlite3.OperationalError):
                self.session.run_turn("mới", Mock(side_effect=AssertionError))
        self.assertEqual(self.session.state, SessionState.AWAITING_EDIT_CONTENT)
        self.assertEqual(self.session.pending_edit_id, 1)
        self.assertEqual(list_tasks(self.database_path)[0].content, "cũ")

        with patch(
            "nexus.agent.session.format_tool_result", side_effect=ValueError("after commit")
        ):
            with self.assertRaises(PostToolExecutionError):
                self.session.run_turn("mới", Mock(side_effect=AssertionError))
        self.assertEqual(self.session.state, SessionState.IDLE)
        self.assertEqual(list_tasks(self.database_path)[0].content, "mới")

    def test_edit_pending_state_is_not_shared(self):
        create_task(self.database_path, "cũ")
        self.session.run_turn("Sửa việc 1", Mock(side_effect=AssertionError))
        other = AgentSession(self.database_path)
        response = {"choices": [{
            "finish_reason": "stop", "message": {"role": "assistant", "content": "Không rõ"}
        }]}
        result = other.run_turn("mới", Mock(return_value=response))
        self.assertEqual(result["status"], "no_tool")
        self.assertEqual(other.state, SessionState.IDLE)
        self.assertEqual(self.session.state, SessionState.AWAITING_EDIT_CONTENT)
        self.assertEqual(list_tasks(self.database_path)[0].content, "cũ")

    def test_delete_direct_request_requires_confirmation_then_deletes(self):
        task = create_task(self.database_path, "mua sữa")
        generate = Mock(return_value=tool_response("delete_task", '{"id":1}'))
        first = self.session.run_turn("Xóa việc 1.", generate)
        self.assertEqual(first["status"], "needs_confirmation")
        self.assertEqual(first["calls"], [])
        self.assertEqual(first["session_state_after"], "awaiting_delete_confirmation")
        self.assertEqual(self.session.pending_delete_task, task)
        self.assertEqual(list_tasks(self.database_path), [task])

        second = self.session.run_turn(
            "Đồng ý.", Mock(side_effect=AssertionError("model must not be called"))
        )
        self.assertEqual(second["status"], "executed")
        self.assertEqual(second["calls"][0]["result"]["status"], "deleted")
        self.assertEqual(second["session_state_after"], "idle")
        self.assertEqual(list_tasks(self.database_path), [])
        generate.assert_called_once()

    def test_delete_confirmation_rejects_free_text_and_accepts_cancel_words(self):
        task = create_task(self.database_path, "mua sữa")
        self.session.run_turn(
            "Xóa việc 1", Mock(return_value=tool_response("delete_task", '{"id":1}'))
        )
        generate = Mock(side_effect=AssertionError("model must not be called"))
        repeated = self.session.run_turn("chắc vậy", generate)
        self.assertEqual(repeated["status"], "needs_confirmation")
        self.assertEqual(self.session.pending_delete_task, task)
        self.assertEqual(list_tasks(self.database_path), [task])
        cancelled = self.session.run_turn("không", generate)
        self.assertEqual(cancelled["status"], "cancelled")
        self.assertEqual(self.session.state, SessionState.IDLE)
        self.assertEqual(list_tasks(self.database_path), [task])
        generate.assert_not_called()

    def test_delete_missing_or_multiple_id_collects_one_id_before_confirmation(self):
        task = create_task(self.database_path, "mua sữa")
        generate = Mock(side_effect=AssertionError("model must not be called"))
        first = self.session.run_turn("Xóa việc 1 và 2", generate)
        self.assertEqual(first["status"], "needs_clarification")
        self.assertEqual(self.session.state, SessionState.AWAITING_DELETE_ID)
        second = self.session.run_turn("#1", generate)
        self.assertEqual(second["status"], "needs_confirmation")
        self.assertEqual(self.session.pending_delete_task, task)
        self.assertEqual(list_tasks(self.database_path), [task])
        generate.assert_not_called()

    def test_delete_missing_id_reports_not_found_without_confirmation(self):
        generate = Mock(side_effect=AssertionError("model must not be called"))
        self.session.run_turn("Xóa việc", generate)
        result = self.session.run_turn("99", generate)
        self.assertEqual(result["status"], "not_found")
        self.assertEqual(self.session.state, SessionState.IDLE)
        self.assertIsNone(self.session.pending_delete_task)
        generate.assert_not_called()

    def test_delete_stale_snapshot_requires_confirmation_again(self):
        from nexus.storage.sqlite_db import update_task
        task = create_task(self.database_path, "cũ")
        self.session.run_turn(
            "Xóa việc 1", Mock(return_value=tool_response("delete_task", '{"id":1}'))
        )
        update_task(self.database_path, task.id, "mới")
        stale = self.session.run_turn(
            "có", Mock(side_effect=AssertionError("model must not be called"))
        )
        self.assertEqual(stale["status"], "needs_confirmation")
        self.assertEqual(self.session.pending_delete_task.content, "mới")
        self.assertEqual(list_tasks(self.database_path)[0].content, "mới")
        deleted = self.session.run_turn(
            "xác nhận", Mock(side_effect=AssertionError("model must not be called"))
        )
        self.assertEqual(deleted["status"], "executed")
        self.assertEqual(list_tasks(self.database_path), [])

    def test_delete_disappeared_task_reports_not_found(self):
        from nexus.storage.sqlite_db import delete_task
        task = create_task(self.database_path, "cũ")
        self.session.run_turn(
            "Xóa việc 1", Mock(return_value=tool_response("delete_task", '{"id":1}'))
        )
        delete_task(self.database_path, task.id)
        result = self.session.run_turn(
            "xóa", Mock(side_effect=AssertionError("model must not be called"))
        )
        self.assertEqual(result["status"], "not_found")
        self.assertEqual(self.session.state, SessionState.IDLE)

    def test_delete_database_error_keeps_pending_formatter_error_clears_it(self):
        task = create_task(self.database_path, "cũ")
        response = Mock(return_value=tool_response("delete_task", '{"id":1}'))
        self.session.run_turn("Xóa việc 1", response)
        with patch(
            "nexus.agent.session.execute_tool",
            side_effect=sqlite3.OperationalError("before commit"),
        ):
            with self.assertRaises(sqlite3.OperationalError):
                self.session.run_turn("có", Mock(side_effect=AssertionError))
        self.assertEqual(self.session.state, SessionState.AWAITING_DELETE_CONFIRMATION)
        self.assertEqual(list_tasks(self.database_path), [task])

        with patch(
            "nexus.agent.session.format_tool_result",
            side_effect=ValueError("after commit"),
        ):
            with self.assertRaises(PostToolExecutionError):
                self.session.run_turn("có", Mock(side_effect=AssertionError))
        self.assertEqual(self.session.state, SessionState.IDLE)
        self.assertEqual(list_tasks(self.database_path), [])

    def test_new_list_replaces_pending_delete_and_sessions_do_not_share_it(self):
        task = create_task(self.database_path, "cũ")
        self.session.run_turn(
            "Xóa việc 1", Mock(return_value=tool_response("delete_task", '{"id":1}'))
        )
        other = AgentSession(self.database_path)
        self.assertEqual(other.state, SessionState.IDLE)
        self.assertIsNone(other.pending_delete_task)

        result = self.session.run_turn(
            "Xem danh sách", Mock(return_value=tool_response("list_tasks", '{}'))
        )
        self.assertEqual(result["status"], "executed")
        self.assertEqual(self.session.state, SessionState.IDLE)
        self.assertEqual(list_tasks(self.database_path), [task])


if __name__ == "__main__":
    unittest.main()
