import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from nexus.agent.responses import format_tool_result
from nexus.agent.tools import execute_tool
from nexus.core.deadlines import VIETNAM_TIMEZONE
from nexus.storage.sqlite_db import (
    complete_task,
    create_task,
    initialize_database,
    set_task_deadline,
)


class DeadlineQueryToolTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.database_path = Path(self.directory.name) / "tasks.db"
        initialize_database(self.database_path)
        self.reference = datetime(2026, 9, 27, 10, 0, tzinfo=VIETNAM_TIMEZONE)
        self.due_at = int(
            datetime(2026, 9, 27, 8, 0, tzinfo=VIETNAM_TIMEZONE).timestamp()
        )
        task = create_task(self.database_path, "mua sữa")
        set_task_deadline(self.database_path, task.id, self.due_at)
        done = create_task(self.database_path, "đã xong")
        set_task_deadline(self.database_path, done.id, self.due_at)
        complete_task(self.database_path, done.id)

    def test_tool_returns_strict_scope_and_real_filtered_tasks(self):
        result = execute_tool(
            self.database_path,
            "list_tasks_by_deadline",
            {"scope": "today"},
            reference_time=self.reference,
        )
        self.assertEqual(result["scope"], "today")
        self.assertEqual([task["content"] for task in result["tasks"]], ["mua sữa"])
        self.assertIn("Có 1 việc chưa hoàn thành đến hạn hôm nay:", format_tool_result(
            "list_tasks_by_deadline", result
        ))
        self.assertIn("[1] [ ] mua sữa — hạn 27/09/2026 08:00", format_tool_result(
            "list_tasks_by_deadline", result
        ))

    def test_tool_and_formatter_reject_invalid_contracts(self):
        for arguments in ({}, {"scope": "week"}, {"scope": True}, {"scope": "today", "x": 1}):
            with self.subTest(arguments=arguments):
                with self.assertRaises(ValueError):
                    execute_tool(
                        self.database_path,
                        "list_tasks_by_deadline",
                        arguments,
                        reference_time=self.reference,
                    )
        with self.assertRaises(ValueError):
            format_tool_result(
                "list_tasks_by_deadline",
                {"scope": "today", "tasks": [{
                    "id": 1,
                    "content": "x",
                    "completed": True,
                    "due_at": self.due_at,
                }]},
            )

    def test_empty_formatter_names_the_scope(self):
        self.assertEqual(
            format_tool_result(
                "list_tasks_by_deadline", {"scope": "overdue", "tasks": []}
            ),
            "Không có việc chưa hoàn thành quá hạn.",
        )


if __name__ == "__main__":
    unittest.main()
