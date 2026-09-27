import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from nexus.agent.client import run_turn
from nexus.core.deadlines import VIETNAM_TIMEZONE, format_deadline
from nexus.storage.sqlite_db import create_task, initialize_database, list_tasks


def tool_response(arguments):
    return {
        "choices": [{
            "finish_reason": "tool_calls",
            "message": {
                "content": None,
                "tool_calls": [{
                    "id": "deadline-1",
                    "type": "function",
                    "function": {
                        "name": "set_task_deadline",
                        "arguments": json.dumps(arguments, ensure_ascii=False),
                    },
                }],
            },
        }]
    }


class DeadlineRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.database_path = Path(self.directory.name) / "tasks.db"
        initialize_database(self.database_path)
        create_task(self.database_path, "mua sữa")
        self.reference = datetime(2026, 9, 26, 23, 59, tzinfo=VIETNAM_TIMEZONE)

    def run_deadline(self, prompt, arguments):
        return run_turn(
            self.database_path,
            prompt,
            lambda _payload: tool_response(arguments),
            prompt_version="v12",
            reference_time=self.reference,
        )

    def test_runtime_uses_one_reference_for_policy_and_execution(self):
        result = self.run_deadline(
            "Đặt hạn việc 1 lúc 8 giờ sáng mai.",
            {"id": 1, "when": "8 giờ sáng mai"},
        )
        self.assertEqual(result["status"], "executed")
        self.assertEqual(result["authorized_calls"][0]["reason"], "explicit_deadline")
        self.assertEqual(format_deadline(list_tasks(self.database_path)[0].due_at), "27/09/2026 08:00")

    def test_wrong_id_rewritten_or_partial_time_cannot_mutate(self):
        cases = (
            ({"id": 2, "when": "8 giờ sáng mai"}, "task_id_mismatch"),
            ({"id": 1, "when": "08:00 ngày mai"}, "content_not_grounded"),
            ({"id": 1, "when": "8 giờ sáng"}, "content_boundary_mismatch"),
        )
        prompt = "Đặt hạn việc 1 lúc 8 giờ sáng mai."
        for arguments, reason in cases:
            with self.subTest(arguments=arguments):
                result = self.run_deadline(prompt, arguments)
                self.assertEqual(result["status"], "rejected")
                self.assertEqual(result["rejected_calls"][0]["reason"], reason)
                self.assertIsNone(list_tasks(self.database_path)[0].due_at)

    def test_missing_data_and_invalid_time_return_clarification(self):
        cases = (
            ("Đặt hạn lúc 8 giờ sáng mai.", {"id": 1, "when": "8 giờ sáng mai"}, "missing_deadline_id"),
            ("Đặt hạn việc 1.", {"id": 1, "when": "8 giờ sáng mai"}, "missing_deadline_time"),
            ("Đặt hạn việc 1 lúc tuần sau.", {"id": 1, "when": "tuần sau"}, "invalid_deadline_time"),
        )
        for prompt, arguments, reason in cases:
            with self.subTest(prompt=prompt):
                result = self.run_deadline(prompt, arguments)
                self.assertEqual(result["status"], "needs_clarification")
                self.assertEqual(result["rejected_calls"][0]["reason"], reason)
                self.assertIsNone(list_tasks(self.database_path)[0].due_at)


if __name__ == "__main__":
    unittest.main()
