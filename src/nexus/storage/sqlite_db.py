"""Store tasks in a local SQLite database using only the standard library."""

import re
import sqlite3
from contextlib import closing
from pathlib import Path

from nexus.core.models import (
    CompletionResult,
    CompletionStatus,
    DeleteResult,
    DeleteStatus,
    Task,
    UpdateResult,
    UpdateStatus,
)


SCHEMA_VERSION = 2


class DatabaseSchemaError(RuntimeError):
    """The database version or task table is not understood by this build."""


def _task_columns(connection: sqlite3.Connection) -> list[tuple]:
    return connection.execute("PRAGMA table_info(tasks)").fetchall()


def _schema_kind(connection: sqlite3.Connection) -> str | None:
    entry = connection.execute(
        "SELECT type, sql FROM sqlite_master WHERE name = 'tasks'"
    ).fetchone()
    if entry is None:
        return None
    object_type, create_sql = entry
    if object_type != "table" or not isinstance(create_sql, str):
        return "unknown"

    columns = _task_columns(connection)
    names = [column[1] for column in columns]
    if names not in (["id", "content"], ["id", "content", "completed"]):
        return "unknown"

    by_name = {column[1]: column for column in columns}
    task_id = by_name["id"]
    content = by_name["content"]
    if task_id[2].upper() != "INTEGER" or task_id[5] != 1:
        return "unknown"
    if content[2].upper() != "TEXT" or content[3] != 1:
        return "unknown"
    if names == ["id", "content"]:
        return "legacy"

    completed = by_name["completed"]
    default = str(completed[4]).strip("()'") if completed[4] is not None else None
    if completed[2].upper() != "INTEGER" or completed[3] != 1 or default != "0":
        return "unknown"
    table_sql = re.sub(r"\s+", "", create_sql.lower())
    if "check(completedin(0,1))" not in table_sql:
        return "unknown"
    invalid = connection.execute(
        "SELECT 1 FROM tasks WHERE completed NOT IN (0, 1) LIMIT 1"
    ).fetchone()
    if invalid:
        return "unknown"
    return (
        "current"
        if "idintegerprimarykeyautoincrement" in table_sql
        else "version_1"
    )


def _create_current_tasks_table(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL,
            completed INTEGER NOT NULL DEFAULT 0
                CHECK(completed IN (0, 1))
        )
        """
    )


def _migrate_to_current(
    connection: sqlite3.Connection, *, has_completed: bool
) -> None:
    connection.execute("ALTER TABLE tasks RENAME TO tasks_before_v2")
    _create_current_tasks_table(connection)
    if has_completed:
        connection.execute(
            """INSERT INTO tasks (id, content, completed)
               SELECT id, content, completed FROM tasks_before_v2 ORDER BY id"""
        )
    else:
        connection.execute(
            """INSERT INTO tasks (id, content, completed)
               SELECT id, content, 0 FROM tasks_before_v2 ORDER BY id"""
        )
    connection.execute("DROP TABLE tasks_before_v2")


def initialize_database(database_path: str | Path) -> None:
    """Create or migrate the recognized task schema in one transaction."""
    with closing(sqlite3.connect(database_path, isolation_level=None)) as connection:
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        if version > SCHEMA_VERSION:
            raise DatabaseSchemaError(
                f"Database schema version {version} is newer than supported version "
                f"{SCHEMA_VERSION}"
            )

        kind = _schema_kind(connection)
        if kind == "unknown":
            raise DatabaseSchemaError("Unrecognized tasks table schema")
        if version == SCHEMA_VERSION:
            if kind != "current":
                raise DatabaseSchemaError(
                    f"Database declares schema version {version}, but its tasks table "
                    "does not match that version"
                )
            return
        if version == 1 and kind != "version_1":
            raise DatabaseSchemaError(
                "Database declares schema version 1, but its tasks table does not "
                "match that version"
            )
        if version not in (0, 1):
            raise DatabaseSchemaError(f"Unsupported database schema version {version}")
        if version == 0 and kind not in (None, "legacy", "version_1", "current"):
            raise DatabaseSchemaError("Unrecognized tasks table schema")

        connection.execute("BEGIN IMMEDIATE")
        try:
            if kind is None:
                _create_current_tasks_table(connection)
            elif kind == "legacy":
                _migrate_to_current(connection, has_completed=False)
            elif kind == "version_1":
                _migrate_to_current(connection, has_completed=True)
            # A current table at version 0 is a recognized interrupted upgrade.
            connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            connection.commit()
        except Exception:
            connection.rollback()
            raise


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
            "SELECT id, content, completed FROM tasks ORDER BY id"
        ).fetchall()
    return [
        Task(id=task_id, content=content, completed=bool(completed))
        for task_id, content, completed in rows
    ]


def complete_task(database_path: str | Path, task_id: int) -> CompletionResult:
    """Complete one task by ID and distinguish idempotency from absence."""
    if isinstance(task_id, bool) or not isinstance(task_id, int) or task_id <= 0:
        raise ValueError("task_id must be a positive integer")

    with closing(sqlite3.connect(database_path)) as connection:
        with connection:
            row = connection.execute(
                """UPDATE tasks SET completed = 1
                   WHERE id = ? AND completed = 0
                   RETURNING id, content, completed""",
                (task_id,),
            ).fetchone()
            if row is not None:
                task = Task(id=row[0], content=row[1], completed=bool(row[2]))
                return CompletionResult(CompletionStatus.COMPLETED, task)

            row = connection.execute(
                "SELECT id, content, completed FROM tasks WHERE id = ?", (task_id,)
            ).fetchone()
            if row is None:
                return CompletionResult(CompletionStatus.NOT_FOUND, None)
            task = Task(id=row[0], content=row[1], completed=bool(row[2]))
            return CompletionResult(CompletionStatus.ALREADY_COMPLETED, task)


def update_task(
    database_path: str | Path, task_id: int, content: str
) -> UpdateResult:
    """Replace one task's content while preserving its completion state."""
    if isinstance(task_id, bool) or not isinstance(task_id, int) or task_id <= 0:
        raise ValueError("task_id must be a positive integer")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("content must be a nonblank string")
    if "\n" in content or "\r" in content:
        raise ValueError("update_task expects one line")

    with closing(sqlite3.connect(database_path)) as connection:
        with connection:
            row = connection.execute(
                """UPDATE tasks SET content = ?
                   WHERE id = ? AND content <> ?
                   RETURNING id, content, completed""",
                (content, task_id, content),
            ).fetchone()
            if row is not None:
                task = Task(id=row[0], content=row[1], completed=bool(row[2]))
                return UpdateResult(UpdateStatus.UPDATED, task)

            row = connection.execute(
                "SELECT id, content, completed FROM tasks WHERE id = ?", (task_id,)
            ).fetchone()
            if row is None:
                return UpdateResult(UpdateStatus.NOT_FOUND, None)
            task = Task(id=row[0], content=row[1], completed=bool(row[2]))
            return UpdateResult(UpdateStatus.UNCHANGED, task)


def get_task(database_path: str | Path, task_id: int) -> Task | None:
    """Return one task by ID, or None when it does not exist."""
    if isinstance(task_id, bool) or not isinstance(task_id, int) or task_id <= 0:
        raise ValueError("task_id must be a positive integer")
    with closing(sqlite3.connect(database_path)) as connection:
        row = connection.execute(
            "SELECT id, content, completed FROM tasks WHERE id = ?", (task_id,)
        ).fetchone()
    if row is None:
        return None
    return Task(id=row[0], content=row[1], completed=bool(row[2]))


def delete_task(
    database_path: str | Path,
    task_id: int,
    *,
    expected_task: Task | None = None,
) -> DeleteResult:
    """Delete one task, optionally only when it still matches a snapshot."""
    if isinstance(task_id, bool) or not isinstance(task_id, int) or task_id <= 0:
        raise ValueError("task_id must be a positive integer")
    if expected_task is not None:
        if not isinstance(expected_task, Task) or expected_task.id != task_id:
            raise ValueError("expected_task must be a Task with the requested ID")

    with closing(sqlite3.connect(database_path)) as connection:
        with connection:
            if expected_task is None:
                row = connection.execute(
                    "DELETE FROM tasks WHERE id = ? RETURNING id, content, completed",
                    (task_id,),
                ).fetchone()
            else:
                row = connection.execute(
                    """DELETE FROM tasks
                       WHERE id = ? AND content = ? AND completed = ?
                       RETURNING id, content, completed""",
                    (task_id, expected_task.content, int(expected_task.completed)),
                ).fetchone()
            if row is not None:
                task = Task(id=row[0], content=row[1], completed=bool(row[2]))
                return DeleteResult(DeleteStatus.DELETED, task)

            current = connection.execute(
                "SELECT id, content, completed FROM tasks WHERE id = ?", (task_id,)
            ).fetchone()
            if current is None:
                return DeleteResult(DeleteStatus.NOT_FOUND, None)
            task = Task(
                id=current[0], content=current[1], completed=bool(current[2])
            )
            return DeleteResult(DeleteStatus.STALE, task)
