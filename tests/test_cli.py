import contextlib
import io
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from nexus.agent.client import PostToolExecutionError
from nexus.cli.main import default_database_path, main
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
        self.assertEqual(listed.stdout, f"[1] {content}\n")

    def test_stdin_splits_lines_preserves_duplicates_and_ignores_blanks(self):
        added = self.run_cli(
            "add", content="mua sữa\r\n\r\n \t \nhọc Python\nmua sữa\n"
        )
        self.assertEqual(added.returncode, 0, added.stderr)
        listed = self.run_cli("list")
        self.assertEqual(listed.returncode, 0, listed.stderr)
        self.assertEqual(listed.stdout, "[1] mua sữa\n[2] học Python\n[3] mua sữa\n")

    def test_empty_argument_is_not_replaced_with_stdin(self):
        added = self.run_cli("add", "", content="không được thêm")
        self.assertEqual(added.returncode, 0, added.stderr)
        self.assertEqual(added.stdout, "Không có nội dung để thêm.\n")
        listed = self.run_cli("list")
        self.assertEqual(listed.returncode, 0, listed.stderr)
        self.assertEqual(listed.stdout, "Danh sách trống.\n")

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
                        "result": {"tasks": [{"id": task.id, "content": task.content}]},
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
                            "tasks": [{"id": t.id, "content": t.content} for t in tasks]
                        },
                    }
                ],
                "proposed_calls": [],
                "authorized_calls": [],
                "rejected_calls": [],
                "reply": "Danh sách hiện có 1 việc:\n[1] mua sữa",
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
            stdout.getvalue(), "\nAI: Danh sách hiện có 1 việc:\n[1] mua sữa\n"
        )
        self.assertNotIn("AI đã xem", stdout.getvalue())

    def test_ask_direct_reply_prints_model_reply_once(self):
        def direct_reply_turn(*_args, **_kwargs):
            return {
                "calls": [],
                "proposed_calls": [],
                "authorized_calls": [],
                "rejected_calls": [],
                "reply": "Xin chào! Tôi có thể giúp gì cho bạn?",
            }

        stdout = io.StringIO()
        stderr = io.StringIO()
        argv = ["nexus", "--db", str(self.database_path), "ask", "Chào bạn"]
        with (
            patch.object(sys, "argv", argv),
            patch(
                "nexus.agent.client.run_turn", side_effect=direct_reply_turn
            ) as mocked_turn,
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
        ):
            result = main()

        self.assertEqual(result, 0)
        mocked_turn.assert_called_once()
        self.assertEqual(
            stdout.getvalue(), "\nAI: Xin chào! Tôi có thể giúp gì cho bạn?\n"
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
                    "result": {"tasks": [{"id": task.id, "content": task.content}]},
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


if __name__ == "__main__":
    unittest.main()
