import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from nexus.core.models import CompletionStatus, Task, UpdateStatus
from nexus.storage.sqlite_db import (
    DatabaseSchemaError,
    SCHEMA_VERSION,
    complete_task,
    create_task,
    create_tasks,
    initialize_database,
    list_tasks,
    update_task,
)


class TaskStoreTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.database_path = Path(self.directory.name) / "tasks.db"
        initialize_database(self.database_path)

    def test_new_database_is_empty(self):
        self.assertEqual(list_tasks(self.database_path), [])
        with sqlite3.connect(self.database_path) as connection:
            self.assertEqual(
                connection.execute("PRAGMA user_version").fetchone()[0],
                SCHEMA_VERSION,
            )
            columns = connection.execute("PRAGMA table_info(tasks)").fetchall()
        self.assertEqual([column[1] for column in columns], ["id", "content", "completed"])

    def test_duplicate_content_creates_distinct_tasks(self):
        first = create_task(self.database_path, "mua sữa")
        second = create_task(self.database_path, "mua sữa")
        self.assertNotEqual(first.id, second.id)
        self.assertEqual(list_tasks(self.database_path), [first, second])

    def test_content_is_preserved_and_survives_reopening(self):
        content = "  đi ăn lúc 8:00; 'học Python'  "
        task = create_task(self.database_path, content)
        # Each API call opens and closes its own connection. Initializing again
        # simulates startup without clearing previously saved tasks.
        initialize_database(self.database_path)
        self.assertEqual(list_tasks(self.database_path), [task])
        self.assertEqual(task.content, content)
        self.assertFalse(task.completed)
        # Read independently to verify the stored value, not just API output.
        connection = sqlite3.connect(self.database_path)
        try:
            self.assertEqual(
                connection.execute("SELECT content FROM tasks").fetchone()[0],
                content,
            )
        finally:
            connection.close()

    def test_blank_input_does_not_create_tasks(self):
        for content in ("", " ", "\t", "\u00a0"):
            with self.subTest(content=content):
                self.assertIsNone(create_task(self.database_path, content))
        self.assertEqual(list_tasks(self.database_path), [])

    def test_migrates_legacy_database_once_and_preserves_tasks(self):
        legacy = Path(self.directory.name) / "legacy.db"
        with sqlite3.connect(legacy) as connection:
            connection.execute(
                "CREATE TABLE tasks (id INTEGER PRIMARY KEY, content TEXT NOT NULL)"
            )
            connection.executemany(
                "INSERT INTO tasks (content) VALUES (?)", [("mua sữa",), ("gọi mẹ",)]
            )

        initialize_database(legacy)
        self.assertEqual(
            list_tasks(legacy),
            [Task(1, "mua sữa", False), Task(2, "gọi mẹ", False)],
        )
        initialize_database(legacy)
        with sqlite3.connect(legacy) as connection:
            self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], 1)
            names = [row[1] for row in connection.execute("PRAGMA table_info(tasks)")]
        self.assertEqual(names.count("completed"), 1)

    def test_rejects_newer_or_unrecognized_schema(self):
        newer = Path(self.directory.name) / "newer.db"
        with sqlite3.connect(newer) as connection:
            connection.execute("PRAGMA user_version = 99")
        with self.assertRaisesRegex(DatabaseSchemaError, "newer"):
            initialize_database(newer)

        unknown = Path(self.directory.name) / "unknown.db"
        with sqlite3.connect(unknown) as connection:
            connection.execute("CREATE TABLE tasks (id INTEGER PRIMARY KEY, title TEXT)")
        with self.assertRaisesRegex(DatabaseSchemaError, "Unrecognized"):
            initialize_database(unknown)

        missing_check = Path(self.directory.name) / "missing-check.db"
        with sqlite3.connect(missing_check) as connection:
            connection.execute(
                """CREATE TABLE tasks (
                    id INTEGER PRIMARY KEY,
                    content TEXT NOT NULL,
                    completed INTEGER NOT NULL DEFAULT 0
                )"""
            )
            connection.execute("PRAGMA user_version = 1")
        with self.assertRaisesRegex(DatabaseSchemaError, "Unrecognized"):
            initialize_database(missing_check)

        wrong_object = Path(self.directory.name) / "view.db"
        with sqlite3.connect(wrong_object) as connection:
            connection.execute("CREATE VIEW tasks AS SELECT 1 AS id, 'x' AS content")
        with self.assertRaisesRegex(DatabaseSchemaError, "Unrecognized"):
            initialize_database(wrong_object)

    def test_failed_migration_rolls_back_schema_and_version(self):
        legacy = Path(self.directory.name) / "rollback.db"
        with sqlite3.connect(legacy) as connection:
            connection.execute(
                "CREATE TABLE tasks (id INTEGER PRIMARY KEY, content TEXT NOT NULL)"
            )
            connection.execute("INSERT INTO tasks (content) VALUES ('giữ nguyên')")
        migration_connection = sqlite3.connect(legacy, isolation_level=None)

        def deny_version_write(action, arg1, arg2, _database, _trigger):
            if action == sqlite3.SQLITE_PRAGMA and arg1 == "user_version" and arg2:
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK

        migration_connection.set_authorizer(deny_version_write)
        with patch(
            "nexus.storage.sqlite_db.sqlite3.connect",
            return_value=migration_connection,
        ):
            with self.assertRaises(sqlite3.DatabaseError):
                initialize_database(legacy)
        with sqlite3.connect(legacy) as connection:
            self.assertEqual(
                [row[1] for row in connection.execute("PRAGMA table_info(tasks)")],
                ["id", "content"],
            )
            self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], 0)

    def test_complete_success_idempotency_and_not_found(self):
        created = create_task(self.database_path, "mua sữa")
        first = complete_task(self.database_path, created.id)
        self.assertEqual(first.status, CompletionStatus.COMPLETED)
        self.assertEqual(first.task, Task(created.id, "mua sữa", True))

        second = complete_task(self.database_path, created.id)
        self.assertEqual(second.status, CompletionStatus.ALREADY_COMPLETED)
        self.assertEqual(second.task, first.task)

        missing = complete_task(self.database_path, 999)
        self.assertEqual(missing.status, CompletionStatus.NOT_FOUND)
        self.assertIsNone(missing.task)

    def test_complete_rejects_non_positive_and_boolean_ids(self):
        for task_id in (True, False, 0, -1, 1.0, "1", None):
            with self.subTest(task_id=task_id), self.assertRaises(ValueError):
                complete_task(self.database_path, task_id)

    def test_complete_rolls_back_when_update_fails(self):
        task = create_task(self.database_path, "mua sữa")
        with sqlite3.connect(self.database_path) as connection:
            connection.execute("""
                CREATE TRIGGER fail_complete BEFORE UPDATE OF completed ON tasks
                WHEN NEW.completed = 1
                BEGIN SELECT RAISE(ABORT, 'failure'); END
            """)
        with self.assertRaises(sqlite3.IntegrityError):
            complete_task(self.database_path, task.id)
        self.assertEqual(list_tasks(self.database_path), [task])

    def test_update_success_unchanged_not_found_and_preserves_completion(self):
        task = create_task(self.database_path, "mua sữa")
        complete_task(self.database_path, task.id)

        updated = update_task(self.database_path, task.id, "mua sữa không đường")
        self.assertEqual(updated.status, UpdateStatus.UPDATED)
        self.assertEqual(updated.task, Task(task.id, "mua sữa không đường", True))

        unchanged = update_task(self.database_path, task.id, "mua sữa không đường")
        self.assertEqual(unchanged.status, UpdateStatus.UNCHANGED)
        self.assertEqual(unchanged.task, updated.task)

        missing = update_task(self.database_path, 999, "không tồn tại")
        self.assertEqual(missing.status, UpdateStatus.NOT_FOUND)
        self.assertIsNone(missing.task)

    def test_update_validates_id_and_one_line_content(self):
        task = create_task(self.database_path, "giữ nguyên")
        cases = [
            (True, "mới"), (0, "mới"), (-1, "mới"), ("1", "mới"),
            (task.id, ""), (task.id, "   "), (task.id, "a\nb"),
            (task.id, 123),
        ]
        for task_id, content in cases:
            with self.subTest(task_id=task_id, content=content), self.assertRaises(ValueError):
                update_task(self.database_path, task_id, content)
        self.assertEqual(list_tasks(self.database_path), [task])

    def test_update_rolls_back_when_database_rejects_change(self):
        task = create_task(self.database_path, "giữ nguyên")
        with sqlite3.connect(self.database_path) as connection:
            connection.execute("""
                CREATE TRIGGER fail_content_update BEFORE UPDATE OF content ON tasks
                BEGIN SELECT RAISE(ABORT, 'failure'); END
            """)
        with self.assertRaises(sqlite3.IntegrityError):
            update_task(self.database_path, task.id, "nội dung mới")
        self.assertEqual(list_tasks(self.database_path), [task])

    def test_create_tasks_rolls_back_all_rows(self):
        with sqlite3.connect(self.database_path) as connection:
            connection.execute("""
                CREATE TRIGGER fail_insert BEFORE INSERT ON tasks
                WHEN NEW.content = 'lỗi'
                BEGIN SELECT RAISE(ABORT, 'failure'); END
            """)
        with self.assertRaises(sqlite3.IntegrityError):
            create_tasks(self.database_path, ["việc mới", "lỗi"])
        self.assertEqual(list_tasks(self.database_path), [])

    def test_multiline_input_must_be_split_before_storage(self):
        for content in ("mua sữa\nhọc Python", "mua sữa\r\nhọc Python"):
            with self.subTest(content=content):
                with self.assertRaises(ValueError):
                    create_task(self.database_path, content)
        self.assertEqual(list_tasks(self.database_path), [])


if __name__ == "__main__":
    unittest.main()
