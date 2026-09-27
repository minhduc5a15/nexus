import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class DeadlineCliTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.database_path = Path(self.directory.name) / "tasks.db"

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

    def test_deadline_set_update_unchanged_not_found_and_list(self):
        self.assertEqual(self.run_cli("add", "mua sữa").returncode, 0)
        set_result = self.run_cli("deadline", "1", "27/09/2026 08:00")
        self.assertEqual(set_result.returncode, 0, set_result.stderr)
        self.assertEqual(set_result.stdout, "Đã đặt hạn [1] vào 27/09/2026 08:00: mua sữa\n")
        self.assertEqual(
            self.run_cli("list").stdout,
            "[1] [ ] mua sữa — hạn 27/09/2026 08:00\n",
        )

        updated = self.run_cli("deadline", "1", "28/09/2026 09:30")
        self.assertEqual(updated.returncode, 0, updated.stderr)
        self.assertIn("Đã đổi hạn [1] thành 28/09/2026 09:30", updated.stdout)
        unchanged = self.run_cli("deadline", "1", "28/09/2026 09:30")
        self.assertEqual(unchanged.returncode, 0, unchanged.stderr)
        self.assertIn("đã có hạn 28/09/2026 09:30", unchanged.stdout)

        missing = self.run_cli("deadline", "99", "27/09/2026 08:00")
        self.assertEqual(missing.returncode, 1)
        self.assertEqual(missing.stdout, "Không tìm thấy việc có ID 99.\n")

    def test_invalid_id_or_time_returns_exit_two_without_mutation(self):
        self.assertEqual(self.run_cli("add", "mua sữa").returncode, 0)
        for arguments in (
            ("0", "27/09/2026 08:00"),
            ("abc", "27/09/2026 08:00"),
            ("1", "tuần sau"),
            ("1", "31/02/2026 08:00"),
        ):
            with self.subTest(arguments=arguments):
                result = self.run_cli("deadline", *arguments)
                self.assertEqual(result.returncode, 2)
        self.assertEqual(self.run_cli("list").stdout, "[1] [ ] mua sữa\n")


if __name__ == "__main__":
    unittest.main()
