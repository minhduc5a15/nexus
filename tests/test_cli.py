import subprocess
import sys
import tempfile
import unittest
from pathlib import Path




class CliTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.database_path = Path(self.directory.name) / "tasks.db"

    def run_cli(self, *arguments, content=None):
        import os
        env = os.environ.copy()
        env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
        return subprocess.run(
            [sys.executable, "-B", "-m", "nexus.cli.main", "--db", str(self.database_path),
             *arguments],
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


if __name__ == "__main__":
    unittest.main()
