import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from nexus.agent.client import PostToolExecutionError
from nexus.agent.routing import ToolRoutingMode
from nexus.cli.main import default_database_path, main, print_tool_results
from nexus.storage.sqlite_db import create_task, initialize_database, list_tasks


class CliTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.database_path = Path(self.directory.name) / "tasks.db"

    def run_cli(self, *arguments, content=None):
        env = os.environ.copy()
        env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
        return subprocess.run(
            [
                sys.executable,
                "-B",
                "-m",
                "nexus.cli.main",
                "--db",
                str(self.database_path),
                *arguments,
            ],
            input=content,
            text=True,
            capture_output=True,
            cwd=self.directory.name,
            timeout=10,
            env=env,
        )

    def test_add_then_list_in_separate_process(self):
        content = "  mua sữa, gọi mẹ và học Python  "
        added = self.run_cli("add", content)
        self.assertEqual(added.returncode, 0, added.stderr)
        listed = self.run_cli("list")
        self.assertEqual(listed.returncode, 0, listed.stderr)
        self.assertEqual(listed.stdout, f"[1] [ ] {content}\n")

    def test_stdin_splits_lines_preserves_duplicates_and_ignores_blanks(self):
        added = self.run_cli(
            "add", content="mua sữa\r\n\r\n \t \nhọc Python\nmua sữa\n"
        )
        self.assertEqual(added.returncode, 0, added.stderr)
        listed = self.run_cli("list")
        self.assertEqual(listed.returncode, 0, listed.stderr)
        self.assertEqual(
            listed.stdout,
            "[1] [ ] mua sữa\n[2] [ ] học Python\n[3] [ ] mua sữa\n",
        )

    def test_empty_argument_is_not_replaced_with_stdin(self):
        added = self.run_cli("add", "", content="không được thêm")
        self.assertEqual(added.returncode, 0, added.stderr)
        self.assertEqual(added.stdout, "Không có nội dung để thêm.\n")
        listed = self.run_cli("list")
        self.assertEqual(listed.returncode, 0, listed.stderr)
        self.assertEqual(listed.stdout, "Danh sách trống.\n")

    def test_complete_command_success_idempotency_not_found_and_validation(self):
        self.assertEqual(self.run_cli("add", "mua sữa").returncode, 0)

        completed = self.run_cli("complete", "1")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout, "Đã hoàn thành [1] mua sữa\n")
        self.assertEqual(self.run_cli("list").stdout, "[1] [x] mua sữa\n")

        repeated = self.run_cli("complete", "1")
        self.assertEqual(repeated.returncode, 0, repeated.stderr)
        self.assertIn("đã hoàn thành trước đó", repeated.stdout)

        missing = self.run_cli("complete", "999")
        self.assertEqual(missing.returncode, 1)
        self.assertEqual(missing.stdout, "Không tìm thấy việc có ID 999.\n")

        for invalid in ("0", "-1", "abc"):
            with self.subTest(invalid=invalid):
                result = self.run_cli("complete", invalid)
                self.assertEqual(result.returncode, 2)
                self.assertIn("ID phải là số nguyên dương", result.stderr)

    def test_edit_command_updated_unchanged_not_found_and_validation(self):
        self.assertEqual(self.run_cli("add", "mua sữa").returncode, 0)
        self.assertEqual(self.run_cli("complete", "1").returncode, 0)

        updated = self.run_cli("edit", "1", "mua sữa không đường")
        self.assertEqual(updated.returncode, 0, updated.stderr)
        self.assertEqual(updated.stdout, "Đã sửa [1] thành: mua sữa không đường\n")
        self.assertEqual(
            self.run_cli("list").stdout, "[1] [x] mua sữa không đường\n"
        )

        unchanged = self.run_cli("edit", "1", "mua sữa không đường")
        self.assertEqual(unchanged.returncode, 0, unchanged.stderr)
        self.assertIn("đã có nội dung này", unchanged.stdout)

        missing = self.run_cli("edit", "999", "không tồn tại")
        self.assertEqual(missing.returncode, 1)
        self.assertEqual(missing.stdout, "Không tìm thấy việc có ID 999.\n")

        for arguments in (("0", "mới"), ("abc", "mới"), ("1", "")):
            with self.subTest(arguments=arguments):
                result = self.run_cli("edit", *arguments)
                self.assertEqual(result.returncode, 2)

    def test_database_error_returns_failure_without_success_message(self):
        self.database_path = Path(self.directory.name) / "missing" / "tasks.db"
        result = self.run_cli("add", "mua sữa")
        self.assertEqual(result.returncode, 1)
        self.assertIn("Lỗi:", result.stderr)
        self.assertEqual(result.stdout, "")

    def test_default_database_path_uses_xdg_data_home(self):
        data_home = Path(self.directory.name) / "data"
        with patch.dict(os.environ, {"XDG_DATA_HOME": str(data_home)}):
            self.assertEqual(default_database_path(), data_home / "nexus" / "nexus.db")

    def test_default_database_parent_is_created(self):
        data_home = Path(self.directory.name) / "new-data-directory"
        stdout = io.StringIO()
        argv = ["nexus", "add", "mua sữa"]
        with (
            patch.dict(os.environ, {"XDG_DATA_HOME": str(data_home)}),
            patch.object(sys, "argv", argv),
            contextlib.redirect_stdout(stdout),
        ):
            result = main()

        database_path = data_home / "nexus" / "nexus.db"
        self.assertEqual(result, 0)
        self.assertTrue(database_path.is_file())
        self.assertEqual(
            [task.content for task in list_tasks(database_path)], ["mua sữa"]
        )

    def test_ask_create_success_prints_only_single_reply_without_duplicate(self):
        def success_turn(database_path, *_args, **_kwargs):
            task = create_task(database_path, "mua sữa")
            return {
                "calls": [
                    {
                        "name": "create_task",
                        "arguments": {"content": "mua sữa"},
                        "result": {"tasks": [{"id": task.id, "content": task.content, "completed": task.completed}]},
                    }
                ],
                "proposed_calls": [],
                "authorized_calls": [],
                "rejected_calls": [],
                "reply": f"Đã thêm [{task.id}] {task.content}",
            }

        stdout = io.StringIO()
        stderr = io.StringIO()
        argv = ["nexus", "--db", str(self.database_path), "ask", "Thêm mua sữa"]
        with (
            patch.object(sys, "argv", argv),
            patch(
                "nexus.agent.client.run_turn", side_effect=success_turn
            ) as mocked_turn,
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
        ):
            result = main()

        self.assertEqual(result, 0)
        mocked_turn.assert_called_once()
        self.assertEqual(stdout.getvalue(), "\nAI: Đã thêm [1] mua sữa\n")
        self.assertNotIn("Đã thêm qua AI", stdout.getvalue())
        self.assertEqual(
            [task.content for task in list_tasks(self.database_path)], ["mua sữa"]
        )

    def test_ask_passes_prompt_version_and_explains_stateless_confirmation(self):
        turn = {
            "calls": [],
            "proposed_calls": [{"name": "delete_task", "arguments": {"id": 1}}],
            "authorized_calls": [],
            "rejected_calls": [],
            "status": "needs_confirmation",
            "reply": "Bạn có chắc muốn xóa [1] mua sữa?",
        }
        stdout = io.StringIO()
        stderr = io.StringIO()
        argv = [
            "nexus", "--db", str(self.database_path), "ask",
            "--prompt-version", "v11", "--model", "custom-model",
            "--tool-routing", "classified", "Xóa việc 1",
        ]
        with (
            patch.object(sys, "argv", argv),
            patch("nexus.agent.client.run_turn", return_value=turn) as mocked_turn,
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
        ):
            result = main()

        self.assertEqual(result, 0)
        self.assertEqual(mocked_turn.call_args.kwargs["prompt_version"], "v11")
        self.assertEqual(
            mocked_turn.call_args.kwargs["settings"]["model"], "custom-model"
        )
        self.assertEqual(
            mocked_turn.call_args.kwargs["tool_routing"],
            ToolRoutingMode.CLASSIFIED,
        )
        self.assertIn("Bạn có chắc muốn xóa", stdout.getvalue())
        self.assertIn("ask không giữ session", stdout.getvalue())

    def test_ask_list_success_prints_only_formatter_output_without_stats(self):
        initialize_database(self.database_path)
        create_task(self.database_path, "mua sữa")

        def success_turn(database_path, *_args, **_kwargs):
            tasks = list_tasks(database_path)
            return {
                "calls": [
                    {
                        "name": "list_tasks",
                        "arguments": {},
                        "result": {
                            "tasks": [{"id": t.id, "content": t.content, "completed": t.completed} for t in tasks]
                        },
                    }
                ],
                "proposed_calls": [],
                "authorized_calls": [],
                "rejected_calls": [],
                "reply": "Danh sách hiện có 1 việc:\n[1] [ ] mua sữa",
            }

        stdout = io.StringIO()
        stderr = io.StringIO()
        argv = ["nexus", "--db", str(self.database_path), "ask", "Xem việc"]
        with (
            patch.object(sys, "argv", argv),
            patch(
                "nexus.agent.client.run_turn", side_effect=success_turn
            ) as mocked_turn,
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
        ):
            result = main()

        self.assertEqual(result, 0)
        mocked_turn.assert_called_once()
        self.assertEqual(
            stdout.getvalue(), "\nAI: Danh sách hiện có 1 việc:\n[1] [ ] mua sữa\n"
        )
        self.assertNotIn("AI đã xem", stdout.getvalue())

    def test_ask_does_not_print_model_success_claim_without_tool(self):
        stdout = io.StringIO()
        stderr = io.StringIO()
        argv = ["nexus", "--db", str(self.database_path), "ask", "Thêm việc: mua sữa"]
        fake_response = {"choices": [{"finish_reason": "stop", "message": {
            "role": "assistant", "content": "Đã thêm mua sữa.",
        }}]}
        with (
            patch.object(sys, "argv", argv),
            patch("nexus.agent.client.chat", return_value=fake_response) as mocked_chat,
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
        ):
            result = main()

        self.assertEqual(result, 0)
        mocked_chat.assert_called_once()
        self.assertEqual(
            stdout.getvalue(), "\nAI: Không có thao tác nào được thực hiện.\n"
        )
        self.assertEqual(list_tasks(self.database_path), [])

    def test_ask_reports_committed_task_and_stderr_warning_on_response_formatting_error(
        self,
    ):
        def failed_turn(database_path, *_args, **_kwargs):
            task = create_task(database_path, "mua sữa")
            calls = [
                {
                    "name": "create_task",
                    "arguments": {"content": "mua sữa"},
                    "result": {"tasks": [{"id": task.id, "content": task.content, "completed": task.completed}]},
                }
            ]
            raise PostToolExecutionError(
                "response_formatting", calls, RuntimeError("formatting error")
            )

        stdout = io.StringIO()
        stderr = io.StringIO()
        argv = ["nexus", "--db", str(self.database_path), "ask", "Thêm mua sữa"]
        with (
            patch.object(sys, "argv", argv),
            patch(
                "nexus.agent.client.run_turn", side_effect=failed_turn
            ) as mocked_turn,
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
        ):
            result = main()

        self.assertEqual(result, 1)
        mocked_turn.assert_called_once()
        self.assertIn("Đã thêm qua AI [1] mua sữa", stdout.getvalue())
        self.assertIn(
            "Chương trình không định dạng được câu trả lời, nhưng các thao tác được liệt kê phía trên đã hoàn tất. Chương trình không tự thử lại.",
            stderr.getvalue(),
        )
        self.assertIn("Chi tiết: formatting error", stderr.getvalue())
        # Verify database not written twice
        self.assertEqual(
            [task.content for task in list_tasks(self.database_path)], ["mua sữa"]
        )

    def test_print_tool_results_reports_a_committed_delete(self):
        stdout = io.StringIO()
        calls = [{
            "name": "delete_task",
            "arguments": {"id": 3},
            "result": {
                "status": "deleted",
                "task": {"id": 3, "content": "bản nháp", "completed": False},
            },
        }]
        with contextlib.redirect_stdout(stdout):
            saved_count = print_tool_results(calls)
        self.assertEqual(saved_count, 0)
        self.assertEqual(stdout.getvalue(), "Đã xóa [3] bản nháp\n")

    def test_chat_keeps_one_session_for_clarification_create_and_list(self):
        list_response = {"choices": [{"finish_reason": "tool_calls", "message": {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "call-list",
                "type": "function",
                "function": {"name": "list_tasks", "arguments": "{}"},
            }],
        }}]}
        stdin = io.StringIO(
            "Thêm việc\nmua sữa\nThêm việc giúp tôi\n"
            "Hiện tại tôi đang có những việc gì nhỉ?\n/exit\n"
        )
        stdout = io.StringIO()
        stderr = io.StringIO()
        argv = [
            "nexus", "--db", str(self.database_path), "chat",
            "--model", "custom-model", "--tool-routing", "classified",
        ]

        with (
            patch.object(sys, "argv", argv),
            patch.object(sys, "stdin", stdin),
            patch("nexus.agent.client.chat", return_value=list_response) as mocked_chat,
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
        ):
            result = main()

        self.assertEqual(result, 0)
        mocked_chat.assert_called_once()
        self.assertEqual(mocked_chat.call_args.args[0]["model"], "custom-model")
        self.assertEqual(
            [tool["function"]["name"] for tool in mocked_chat.call_args.args[0]["tools"]],
            ["list_tasks"],
        )
        self.assertEqual(
            stdout.getvalue(),
            "NEXUS: Bạn muốn thêm việc gì?\n"
            "NEXUS: Đã thêm [1] mua sữa\n"
            "NEXUS: Bạn muốn thêm việc gì?\n"
            "NEXUS: Danh sách hiện có 1 việc:\n"
            "[1] [ ] mua sữa\n",
        )
        self.assertEqual(stderr.getvalue(), "")
        self.assertEqual(
            [task.content for task in list_tasks(self.database_path)], ["mua sữa"]
        )

    def test_chat_exit_is_local_and_does_not_call_model(self):
        stdin = io.StringIO("/exit\n")
        stdout = io.StringIO()
        stderr = io.StringIO()
        argv = ["nexus", "--db", str(self.database_path), "chat"]

        with (
            patch.object(sys, "argv", argv),
            patch.object(sys, "stdin", stdin),
            patch("nexus.agent.client.chat") as mocked_chat,
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
        ):
            result = main()

        self.assertEqual(result, 0)
        mocked_chat.assert_not_called()
        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(stderr.getvalue(), "")
        self.assertEqual(list_tasks(self.database_path), [])

    def test_chat_reports_continuation_commit_without_claiming_model_did_it(self):
        stdin = io.StringIO("Thêm việc\nmua sữa\n/exit\n")
        stdout = io.StringIO()
        stderr = io.StringIO()
        argv = ["nexus", "--db", str(self.database_path), "chat", "--trace"]

        with (
            patch.object(sys, "argv", argv),
            patch.object(sys, "stdin", stdin),
            patch("nexus.agent.client.chat") as mocked_chat,
            patch(
                "nexus.agent.session.format_tool_result",
                side_effect=ValueError("formatting error"),
            ),
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
        ):
            result = main()

        self.assertEqual(result, 1)
        mocked_chat.assert_not_called()
        self.assertIn("NEXUS: Bạn muốn thêm việc gì?", stdout.getvalue())
        self.assertIn("Đã thêm [1] mua sữa", stdout.getvalue())
        self.assertNotIn("qua AI", stdout.getvalue())
        self.assertIn("đã hoàn tất", stderr.getvalue())
        self.assertIn('"status": "error_after_execution"', stderr.getvalue())
        self.assertIn('"stage": "response_formatting"', stderr.getvalue())
        self.assertEqual(
            [task.content for task in list_tasks(self.database_path)], ["mua sữa"]
        )

    def test_chat_trace_explains_session_model_policy_tool_and_database(self):
        list_response = {"choices": [{"finish_reason": "tool_calls", "message": {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "call-list",
                "type": "function",
                "function": {"name": "list_tasks", "arguments": "{}"},
            }],
        }}]}
        stdin = io.StringIO("Thêm việc\nmua sữa\nXem danh sách\n/exit\n")
        stdout = io.StringIO()
        stderr = io.StringIO()
        argv = ["nexus", "--db", str(self.database_path), "chat", "--trace"]

        with (
            patch.object(sys, "argv", argv),
            patch.object(sys, "stdin", stdin),
            patch("nexus.agent.client.chat", return_value=list_response) as mocked_chat,
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
        ):
            result = main()

        self.assertEqual(result, 0)
        mocked_chat.assert_called_once()
        self.assertIn("NEXUS: Đã thêm [1] mua sữa", stdout.getvalue())

        blocks = stderr.getvalue().split("=== TRACE TURN ")[1:]
        self.assertEqual(len(blocks), 3)
        traces = []
        for block in blocks:
            _, json_and_end = block.split(" ===\n", 1)
            raw_json, _ = json_and_end.split("\n=== END TRACE ===", 1)
            traces.append(json.loads(raw_json))

        first, continuation, listed = traces
        self.assertEqual(first["runtime"]["status"], "needs_clarification")
        self.assertEqual(first["runtime"]["source"], "session")
        self.assertFalse(first["model"]["called"])
        self.assertFalse(first["session"]["pending_create_before"])
        self.assertTrue(first["session"]["pending_create_after"])
        self.assertEqual(first["session"]["state_before"], "idle")
        self.assertEqual(
            first["session"]["state_after"], "awaiting_create_content"
        )
        self.assertEqual(first["database"], {"before": [], "after": []})

        self.assertEqual(continuation["runtime"]["source"], "session_continuation")
        self.assertEqual(
            continuation["session"],
            {
                "state_before": "awaiting_create_content",
                "state_after": "idle",
                "pending_create_before": True,
                "pending_create_after": False,
                "pending_complete_before": False,
                "pending_complete_after": False,
                "pending_edit_id_before": None,
                "pending_edit_id_after": None,
                "pending_edit_content_before": None,
                "pending_edit_content_after": None,
                "pending_deadline_id_before": None,
                "pending_deadline_id_after": None,
                "pending_deadline_text_before": None,
                "pending_deadline_text_after": None,
                "pending_deadline_scope_before": False,
                "pending_deadline_scope_after": False,
                "pending_deadline_reference_before": None,
                "pending_deadline_reference_after": None,
                "pending_delete_before": None,
                "pending_delete_after": None,
            },
        )
        self.assertFalse(continuation["model"]["called"])
        self.assertEqual(
            continuation["authorized_calls"][0]["reason"],
            "session_continuation",
        )
        self.assertEqual(continuation["database"]["before"], [])
        self.assertEqual(
            continuation["database"]["after"],
            [{"id": 1, "content": "mua sữa", "completed": False, "due_at": None}],
        )

        self.assertEqual(listed["runtime"]["source"], "model")
        self.assertEqual(listed["session"]["state_before"], "idle")
        self.assertEqual(listed["session"]["state_after"], "idle")
        self.assertTrue(listed["model"]["called"])
        self.assertEqual(
            listed["model"]["request"]["messages"][-1]["content"],
            "Xem danh sách",
        )
        self.assertEqual(listed["proposed_calls"][0]["name"], "list_tasks")
        self.assertEqual(listed["contract_validation"]["status"], "passed")
        self.assertEqual(listed["contract_validation"]["proposal_count"], 1)
        self.assertEqual(
            listed["authorized_calls"][0]["reason"], "explicit_list"
        )
        self.assertEqual(listed["executed_calls"][0]["name"], "list_tasks")
        self.assertEqual(listed["database"]["before"], listed["database"]["after"])
        self.assertEqual(listed["error"], None)

    def test_delete_command_is_direct_and_does_not_reuse_id(self):
        self.assertEqual(self.run_cli("add", "mua sữa").returncode, 0)
        deleted = self.run_cli("delete", "1")
        self.assertEqual(deleted.returncode, 0, deleted.stderr)
        self.assertEqual(deleted.stdout, "Đã xóa [1] mua sữa\n")
        self.assertEqual(self.run_cli("list").stdout, "Danh sách trống.\n")

        self.assertEqual(self.run_cli("add", "gọi mẹ").stdout, "Đã thêm [2] gọi mẹ\n")
        missing = self.run_cli("delete", "999")
        self.assertEqual(missing.returncode, 1)
        self.assertEqual(missing.stdout, "Không tìm thấy việc có ID 999.\n")
        for invalid in ("0", "-1", "abc"):
            with self.subTest(invalid=invalid):
                self.assertEqual(self.run_cli("delete", invalid).returncode, 2)


if __name__ == "__main__":
    unittest.main()
