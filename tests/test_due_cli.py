import os
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, time
from pathlib import Path

from nexus.core.deadlines import VIETNAM_TIMEZONE, vietnam_now
from nexus.storage.sqlite_db import (
    complete_task,
    create_task,
    initialize_database,
    set_task_deadline,
)


class DeadlineQueryCliTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.database_path = Path(self.directory.name) / "tasks.db"
        initialize_database(self.database_path)

    def run_cli(self, *arguments):
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
            text=True,
            capture_output=True,
            cwd=self.directory.name,
            timeout=10,
            env=env,
        )

    def test_due_today_prints_only_incomplete_tasks_with_deadline(self):
        today = vietnam_now().date()
        due_at = int(datetime.combine(today, time(8), tzinfo=VIETNAM_TIMEZONE).timestamp())
        wanted = create_task(self.database_path, "mua sữa")
        set_task_deadline(self.database_path, wanted.id, due_at)
        done = create_task(self.database_path, "đã xong")
        set_task_deadline(self.database_path, done.id, due_at)
        complete_task(self.database_path, done.id)
        create_task(self.database_path, "không có hạn")

        result = self.run_cli("due", "today")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Có 1 việc chưa hoàn thành đến hạn hôm nay:", result.stdout)
        self.assertIn("[1] [ ] mua sữa", result.stdout)
        self.assertNotIn("đã xong", result.stdout)
        self.assertNotIn("không có hạn", result.stdout)

    def test_empty_scope_and_invalid_scope_exit_codes(self):
        empty = self.run_cli("due", "tomorrow")
        self.assertEqual(empty.returncode, 0, empty.stderr)
        self.assertEqual(
            empty.stdout,
            "Không có việc chưa hoàn thành đến hạn ngày mai.\n",
        )
        invalid = self.run_cli("due", "week")
        self.assertEqual(invalid.returncode, 2)
        self.assertIn("invalid choice", invalid.stderr)


if __name__ == "__main__":
    unittest.main()
