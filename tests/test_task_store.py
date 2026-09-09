import sqlite3
import tempfile
import unittest
from pathlib import Path

from nexus.storage.sqlite_db import create_task, initialize_database, list_tasks


class TaskStoreTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.database_path = Path(self.directory.name) / "tasks.db"
        initialize_database(self.database_path)

    def test_new_database_is_empty(self):
        self.assertEqual(list_tasks(self.database_path), [])

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

    def test_multiline_input_must_be_split_before_storage(self):
        for content in ("mua sữa\nhọc Python", "mua sữa\r\nhọc Python"):
            with self.subTest(content=content):
                with self.assertRaises(ValueError):
                    create_task(self.database_path, content)
        self.assertEqual(list_tasks(self.database_path), [])


if __name__ == "__main__":
    unittest.main()
