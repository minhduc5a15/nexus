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

    return create_tasks(database_path, [content])[0]


def create_tasks(database_path: str | Path, contents: list[str]) -> list[Task]:
    """Save all nonblank lines in one transaction, or save none on failure."""
    if not contents:
        return []
    for content in contents:
        if not isinstance(content, str) or not content.strip():
            raise ValueError("create_tasks expects nonblank strings")
        if "\n" in content or "\r" in content:
            raise ValueError("create_tasks expects one line per item")

    tasks = []
    with closing(sqlite3.connect(database_path)) as connection:
        with connection:
            for content in contents:
                cursor = connection.execute(
                    "INSERT INTO tasks (content) VALUES (?) RETURNING id",
                    (content,),
                )
                tasks.append(Task(id=cursor.fetchone()[0], content=content))
    return tasks


def list_tasks(database_path: str | Path) -> list[Task]:
    """Return all tasks ordered by their generated IDs."""
    with closing(sqlite3.connect(database_path)) as connection:
        rows = connection.execute(
            "SELECT id, content FROM tasks ORDER BY id"
        ).fetchall()
    return [Task(id=task_id, content=content) for task_id, content in rows]
