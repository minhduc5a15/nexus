import json
import sqlite3
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
        self.assertIs(first["tasks"][0]["completed"], False)
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
            ("complete_task", {}),
            ("complete_task", {"id": True}),
            ("complete_task", {"id": 0}),
            ("complete_task", {"id": "1"}),
            ("complete_task", {"id": 1, "content": "giữ nguyên"}),
            ("update_task", {}),
            ("update_task", {"id": True, "content": "mới"}),
            ("update_task", {"id": 1, "content": ""}),
            ("update_task", {"id": 1, "content": "a\nb"}),
            ("update_task", {"id": 1, "content": "mới", "extra": 1}),
        ]
        for name, arguments in cases:
            with self.subTest(name=name, arguments=arguments):
                with self.assertRaises(ValueError):
                    execute_tool(self.database_path, name, arguments)
                self.assertEqual(list_tasks(self.database_path), before)

    def test_database_failure_does_not_return_a_success_result(self):
        invalid_path = Path(self.directory.name) / "missing" / "tasks.db"
        with self.assertRaises(sqlite3.Error):
            execute_tool(invalid_path, "create_task", {"content": "mua sữa"})

    def test_multiline_create_rolls_back_all_lines_when_one_insert_fails(self):
        execute_tool(self.database_path, "create_task", {"content": "việc cũ"})
        with sqlite3.connect(self.database_path) as connection:
            connection.execute("""
                CREATE TRIGGER fail_second BEFORE INSERT ON tasks
                WHEN NEW.content = 'lỗi'
                BEGIN SELECT RAISE(ABORT, 'simulated insert failure'); END
            """)
        before = list_tasks(self.database_path)
        with self.assertRaises(sqlite3.IntegrityError):
            execute_tool(self.database_path, "create_task", {"content": "việc mới\nlỗi"})
        self.assertEqual(list_tasks(self.database_path), before)

    def test_complete_result_contract_and_list_state(self):
        created = execute_tool(
            self.database_path, "create_task", {"content": "mua sữa"}
        )["tasks"][0]
        task_id = created["id"]
        first = execute_tool(self.database_path, "complete_task", {"id": task_id})
        self.assertEqual(
            first,
            {
                "status": "completed",
                "task": {"id": task_id, "content": "mua sữa", "completed": True, "due_at": None},
            },
        )
        second = execute_tool(self.database_path, "complete_task", {"id": task_id})
        self.assertEqual(second["status"], "already_completed")
        self.assertEqual(second["task"], first["task"])
        self.assertEqual(
            execute_tool(self.database_path, "complete_task", {"id": 999}),
            {"status": "not_found", "task": None},
        )
        self.assertEqual(execute_tool(self.database_path, "list_tasks", {})["tasks"], [first["task"]])

    def test_update_result_contract_and_preserves_completed(self):
        created = execute_tool(
            self.database_path, "create_task", {"content": "mua sữa"}
        )["tasks"][0]
        execute_tool(self.database_path, "complete_task", {"id": created["id"]})
        updated = execute_tool(
            self.database_path,
            "update_task",
            {"id": created["id"], "content": "mua sữa không đường"},
        )
        self.assertEqual(
            updated,
            {
                "status": "updated",
                "task": {
                    "id": created["id"],
                    "content": "mua sữa không đường",
                    "completed": True,
                    "due_at": None,
                },
            },
        )
        self.assertEqual(
            execute_tool(
                self.database_path,
                "update_task",
                {"id": created["id"], "content": "mua sữa không đường"},
            )["status"],
            "unchanged",
        )
        self.assertEqual(
            execute_tool(
                self.database_path,
                "update_task",
                {"id": 999, "content": "không tồn tại"},
            ),
            {"status": "not_found", "task": None},
        )

    def test_delete_requires_internal_confirmation_snapshot(self):
        from nexus.core.models import Task
        created = execute_tool(
            self.database_path, "create_task", {"content": "mua sữa"}
        )["tasks"][0]
        task = Task(**created)
        with self.assertRaisesRegex(ValueError, "confirmed task snapshot"):
            execute_tool(self.database_path, "delete_task", {"id": task.id})
        self.assertEqual(len(list_tasks(self.database_path)), 1)

        result = execute_tool(
            self.database_path,
            "delete_task",
            {"id": task.id},
            confirmed_task=task,
        )
        self.assertEqual(result, {"status": "deleted", "task": created})
        self.assertEqual(list_tasks(self.database_path), [])

    def test_delete_stale_snapshot_cannot_remove_changed_task(self):
        from nexus.core.models import Task
        from nexus.storage.sqlite_db import update_task
        created = execute_tool(
            self.database_path, "create_task", {"content": "cũ"}
        )["tasks"][0]
        snapshot = Task(**created)
        update_task(self.database_path, snapshot.id, "mới")
        result = execute_tool(
            self.database_path,
            "delete_task",
            {"id": snapshot.id},
            confirmed_task=snapshot,
        )
        self.assertEqual(result["status"], "stale")
        self.assertEqual(result["task"]["content"], "mới")
        self.assertEqual(list_tasks(self.database_path)[0].content, "mới")


if __name__ == "__main__":
    unittest.main()
