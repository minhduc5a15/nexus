import json
import tempfile
import unittest
from pathlib import Path

from nexus.storage.sqlite_db import initialize_database, list_tasks
from nexus.agent.tools import execute_tool


class TaskToolTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.database_path = Path(self.directory.name) / "tasks.db"
        initialize_database(self.database_path)

    def test_create_and_list_return_json_serializable_storage_results(self):
        content = "  mua sữa, gọi mẹ  "
        first = execute_tool(self.database_path, "create_task", {"content": content})
        second = execute_tool(self.database_path, "create_task", {"content": content})
        self.assertNotEqual(first["tasks"][0]["id"], second["tasks"][0]["id"])
        self.assertEqual(first["tasks"][0]["content"], content)
        result = execute_tool(self.database_path, "list_tasks", {})
        self.assertEqual(result, {"tasks": [first["tasks"][0], second["tasks"][0]]})
        self.assertEqual(json.loads(json.dumps(result)), result)

    def test_create_multiple_tasks_with_newlines(self):
        content = "mua sữa\ngọi mẹ\n\n  làm bài tập  "
        result = execute_tool(self.database_path, "create_task", {"content": content})
        self.assertEqual(len(result["tasks"]), 3)
        self.assertEqual(result["tasks"][0]["content"], "mua sữa")
        self.assertEqual(result["tasks"][1]["content"], "gọi mẹ")
        self.assertEqual(result["tasks"][2]["content"], "  làm bài tập  ")
        list_result = execute_tool(self.database_path, "list_tasks", {})
        self.assertEqual(list_result["tasks"], result["tasks"])

    def test_invalid_calls_leave_existing_data_unchanged(self):
        execute_tool(self.database_path, "create_task", {"content": "giữ nguyên"})
        before = list_tasks(self.database_path)
        cases = [
            ("delete_task", {"id": 1}),
            ("create_task", '{"content": "mua sữa"}'),
            ("create_task", []),
            ("create_task", {}),
            ("create_task", {"content": "mua sữa", "database_path": "/tmp/other.db"}),
            ("create_task", {"content": None}),
            ("create_task", {"content": 42}),
            ("create_task", {"content": " \t"}),
            ("create_task", {"content": "\n  \n\r"}),
            ("list_tasks", {"limit": 1}),
        ]
        for name, arguments in cases:
            with self.subTest(name=name, arguments=arguments):
                with self.assertRaises(ValueError):
                    execute_tool(self.database_path, name, arguments)
                self.assertEqual(list_tasks(self.database_path), before)

    def test_database_failure_does_not_return_a_success_result(self):
        import sqlite3

        invalid_path = Path(self.directory.name) / "missing" / "tasks.db"
        with self.assertRaises(sqlite3.Error):
            execute_tool(invalid_path, "create_task", {"content": "mua sữa"})


if __name__ == "__main__":
    unittest.main()
