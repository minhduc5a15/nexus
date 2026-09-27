import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from nexus.core.deadlines import VIETNAM_TIMEZONE
from nexus.core.models import DeadlineScope
from nexus.storage.sqlite_db import (
    complete_task,
    create_task,
    initialize_database,
    list_tasks_by_deadline,
    set_task_deadline,
)


def timestamp(day, hour=0, minute=0):
    return int(datetime(2026, 9, day, hour, minute, tzinfo=VIETNAM_TIMEZONE).timestamp())


class DeadlineQueryStorageTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.database_path = Path(self.directory.name) / "tasks.db"
        initialize_database(self.database_path)
        self.reference = datetime(2026, 9, 27, 10, 0, tzinfo=VIETNAM_TIMEZONE)

    def add(self, content, due_at=None, completed=False):
        task = create_task(self.database_path, content)
        if due_at is not None:
            set_task_deadline(self.database_path, task.id, due_at)
        if completed:
            complete_task(self.database_path, task.id)
        return task

    def test_three_scopes_filter_and_order_tasks(self):
        today_past = self.add("hôm nay đã trễ", timestamp(27, 8))
        today_future = self.add("hôm nay sắp tới", timestamp(27, 12))
        tomorrow = self.add("ngày mai", timestamp(28, 9))
        old = self.add("hôm qua", timestamp(26, 22))
        self.add("đã xong", timestamp(27, 7), completed=True)
        self.add("không có hạn")
        same_time = self.add("cùng giờ", timestamp(27, 12))

        self.assertEqual(
            [task.id for task in list_tasks_by_deadline(
                self.database_path, DeadlineScope.TODAY, now=self.reference
            )],
            [today_past.id, today_future.id, same_time.id],
        )
        self.assertEqual(
            [task.id for task in list_tasks_by_deadline(
                self.database_path, DeadlineScope.TOMORROW, now=self.reference
            )],
            [tomorrow.id],
        )
        self.assertEqual(
            [task.id for task in list_tasks_by_deadline(
                self.database_path, DeadlineScope.OVERDUE, now=self.reference
            )],
            [old.id, today_past.id],
        )

    def test_day_boundaries_and_exact_now(self):
        start = self.add("đầu ngày", timestamp(27, 0))
        exact_now = self.add("đúng hiện tại", timestamp(27, 10))
        last_second = self.add("cuối ngày", timestamp(27, 23, 59) + 59)
        next_day = self.add("đầu ngày mai", timestamp(28, 0))

        today = list_tasks_by_deadline(
            self.database_path, DeadlineScope.TODAY, now=self.reference
        )
        self.assertEqual([task.id for task in today], [start.id, exact_now.id, last_second.id])
        overdue = list_tasks_by_deadline(
            self.database_path, DeadlineScope.OVERDUE, now=self.reference
        )
        self.assertEqual([task.id for task in overdue], [start.id])
        tomorrow = list_tasks_by_deadline(
            self.database_path, DeadlineScope.TOMORROW, now=self.reference
        )
        self.assertEqual([task.id for task in tomorrow], [next_day.id])

    def test_empty_result_and_invalid_inputs(self):
        self.assertEqual(
            list_tasks_by_deadline(
                self.database_path, DeadlineScope.TODAY, now=self.reference
            ),
            [],
        )
        with self.assertRaises(ValueError):
            list_tasks_by_deadline(self.database_path, "today", now=self.reference)
        with self.assertRaises(ValueError):
            list_tasks_by_deadline(
                self.database_path,
                DeadlineScope.TODAY,
                now=datetime(2026, 9, 27, 10, 0),
            )


if __name__ == "__main__":
    unittest.main()
