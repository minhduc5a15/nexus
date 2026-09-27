import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from nexus.agent.responses import format_tool_result
from nexus.agent.tools import execute_tool
from nexus.core.deadlines import VIETNAM_TIMEZONE
from nexus.storage.sqlite_db import create_task, initialize_database, list_tasks


class DeadlineToolAndFormatterTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.database_path = Path(self.directory.name) / "tasks.db"
        initialize_database(self.database_path)
        self.task = create_task(self.database_path, "mua sữa")
        self.reference = datetime(2026, 9, 26, 12, 0, tzinfo=VIETNAM_TIMEZONE)

    def test_tool_sets_deadline_and_returns_strict_real_task(self):
        result = execute_tool(
            self.database_path,
            "set_task_deadline",
            {"id": self.task.id, "when": "8 giờ sáng mai"},
            reference_time=self.reference,
        )
        self.assertEqual(result["status"], "set")
        self.assertEqual(
            result["task"],
            {
                "id": self.task.id,
                "content": "mua sữa",
                "completed": False,
                "due_at": 1790470800,
            },
        )
        self.assertEqual(list_tasks(self.database_path)[0].due_at, 1790470800)
        self.assertEqual(
            format_tool_result("set_task_deadline", result),
            "Đã đặt hạn [1] vào 27/09/2026 08:00: mua sữa",
        )

    def test_tool_schema_is_strict_and_boolean_is_not_integer(self):
        invalid = (
            None,
            {"id": True, "when": "8 giờ sáng mai"},
            {"id": 1, "when": "8 giờ sáng mai", "due_at": 1},
            {"id": 1, "when": ""},
            {"id": 1, "when": "8 giờ\nmai"},
        )
        for arguments in invalid:
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                execute_tool(
                    self.database_path,
                    "set_task_deadline",
                    arguments,
                    reference_time=self.reference,
                )
        self.assertIsNone(list_tasks(self.database_path)[0].due_at)

    def test_formatter_covers_all_statuses_and_list_deadline(self):
        task = {
            "id": 1,
            "content": "mua sữa",
            "completed": False,
            "due_at": 1790470800,
        }
        self.assertEqual(
            format_tool_result("set_task_deadline", {"status": "updated", "task": task}),
            "Đã đổi hạn [1] thành 27/09/2026 08:00: mua sữa",
        )
        self.assertEqual(
            format_tool_result("set_task_deadline", {"status": "unchanged", "task": task}),
            "Việc [1] đã có hạn 27/09/2026 08:00: mua sữa",
        )
        self.assertEqual(
            format_tool_result("set_task_deadline", {"status": "not_found", "task": None}),
            "Không tìm thấy việc có ID đã yêu cầu.",
        )
        self.assertEqual(
            format_tool_result("list_tasks", {"tasks": [task]}),
            "Danh sách hiện có 1 việc:\n[1] [ ] mua sữa — hạn 27/09/2026 08:00",
        )

    def test_formatter_rejects_invalid_deadline_result(self):
        base = {"id": 1, "content": "mua sữa", "completed": False, "due_at": 1790470800}
        invalid = (
            {"status": "set", "task": {**base, "due_at": None}},
            {"status": "set", "task": {**base, "due_at": True}},
            {"status": "not_found", "task": base},
            {"status": "unknown", "task": base},
            {"status": "set", "task": base, "extra": True},
        )
        for result in invalid:
            with self.subTest(result=result), self.assertRaises(ValueError):
                format_tool_result("set_task_deadline", result)


if __name__ == "__main__":
    unittest.main()
