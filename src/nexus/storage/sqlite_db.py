"""Store tasks in a local SQLite database using only the standard library."""

import sqlite3
from contextlib import closing
from pathlib import Path

from nexus.core.models import Task


def initialize_database(database_path: str | Path) -> None:
    """Create the task table if it does not exist; preserve existing tasks."""
    with closing(sqlite3.connect(database_path)) as connection:
        with connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS tasks (
                    id INTEGER PRIMARY KEY,
                    content TEXT NOT NULL
                )
                """
            )


def create_task(database_path: str | Path, content: str) -> Task | None:
    """Save one nonblank line verbatim, or return None for blank input."""
    if not content.strip():
        return None
    if "\n" in content or "\r" in content:
        raise ValueError("create_task expects one line; split multiline input first")

    with closing(sqlite3.connect(database_path)) as connection:
        with connection:
            cursor = connection.execute(
                "INSERT INTO tasks (content) VALUES (?) RETURNING id",
                (content,),
            )
            task_id = cursor.fetchone()[0]
        return Task(id=task_id, content=content)


def list_tasks(database_path: str | Path) -> list[Task]:
    """Return all tasks ordered by their generated IDs."""
    with closing(sqlite3.connect(database_path)) as connection:
        rows = connection.execute(
            "SELECT id, content FROM tasks ORDER BY id"
        ).fetchall()
    return [Task(id=task_id, content=content) for task_id, content in rows]
