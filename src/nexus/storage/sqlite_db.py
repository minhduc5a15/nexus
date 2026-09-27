"""Store tasks in a local SQLite database using only the standard library."""

import re
import sqlite3
from contextlib import closing
from pathlib import Path

from nexus.core.models import (
    CompletionResult,
    CompletionStatus,
    DeadlineResult,
    DeadlineStatus,
    DeleteResult,
    DeleteStatus,
    Task,
    UpdateResult,
    UpdateStatus,
)


SCHEMA_VERSION = 3
_TASK_FIELDS = "id, content, completed, due_at"


class DatabaseSchemaError(RuntimeError):
    """The database version or task table is not understood by this build."""


def _task_columns(connection: sqlite3.Connection) -> list[tuple]:
    return connection.execute("PRAGMA table_info(tasks)").fetchall()


def _task_from_row(row: tuple) -> Task:
    return Task(
        id=row[0],
        content=row[1],
        completed=bool(row[2]),
        due_at=row[3],
    )


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
    recognized = (
        ["id", "content"],
        ["id", "content", "completed"],
        ["id", "content", "completed", "due_at"],
    )
    if names not in recognized:
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
    if connection.execute(
        "SELECT 1 FROM tasks WHERE completed NOT IN (0, 1) LIMIT 1"
    ).fetchone():
        return "unknown"

    autoincrement = "idintegerprimarykeyautoincrement" in table_sql
    if names == ["id", "content", "completed"]:
        return "version_2" if autoincrement else "version_1"

    due_at = by_name["due_at"]
    if due_at[2].upper() != "INTEGER" or due_at[3] != 0:
        return "unknown"
    if "check(due_atisnullortypeof(due_at)='integer')" not in table_sql:
        return "unknown"
    if connection.execute(
        "SELECT 1 FROM tasks WHERE due_at IS NOT NULL AND typeof(due_at) <> 'integer' LIMIT 1"
    ).fetchone():
        return "unknown"
    return "current" if autoincrement else "unknown"


def _create_current_tasks_table(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL,
            completed INTEGER NOT NULL DEFAULT 0
                CHECK(completed IN (0, 1)),
            due_at INTEGER
                CHECK(due_at IS NULL OR typeof(due_at) = 'integer')
        )
        """
    )


def _rebuild_to_current(
    connection: sqlite3.Connection, *, has_completed: bool
) -> None:
    connection.execute("ALTER TABLE tasks RENAME TO tasks_before_v3")
    _create_current_tasks_table(connection)
    if has_completed:
        connection.execute(
            """INSERT INTO tasks (id, content, completed, due_at)
               SELECT id, content, completed, NULL FROM tasks_before_v3 ORDER BY id"""
        )
    else:
        connection.execute(
            """INSERT INTO tasks (id, content, completed, due_at)
               SELECT id, content, 0, NULL FROM tasks_before_v3 ORDER BY id"""
        )
    connection.execute("DROP TABLE tasks_before_v3")


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
        expected_kinds = {1: "version_1", 2: "version_2", 3: "current"}
        if version in expected_kinds and kind != expected_kinds[version]:
            raise DatabaseSchemaError(
                f"Database declares schema version {version}, but its tasks table "
                "does not match that version"
            )
        if version not in (0, 1, 2, 3):
            raise DatabaseSchemaError(f"Unsupported database schema version {version}")
        if version == SCHEMA_VERSION:
            return
        if version == 0 and kind not in (
            None,
            "legacy",
            "version_1",
            "version_2",
            "current",
        ):
            raise DatabaseSchemaError("Unrecognized tasks table schema")

        connection.execute("BEGIN IMMEDIATE")
        try:
            if kind is None:
                _create_current_tasks_table(connection)
            elif kind == "legacy":
                _rebuild_to_current(connection, has_completed=False)
            elif kind == "version_1":
                _rebuild_to_current(connection, has_completed=True)
            elif kind == "version_2":
                connection.execute(
                    """ALTER TABLE tasks ADD COLUMN due_at INTEGER
                       CHECK(due_at IS NULL OR typeof(due_at) = 'integer')"""
                )
            # A current table at version 0 is a recognized interrupted upgrade.
            connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            connection.commit()
        except Exception:
            connection.rollback()
            raise


def create_task(database_path: str | Path, content: str) -> Task | None:
    """Save one nonblank line verbatim, or return None for blank input."""
    if not isinstance(content, str) or not content.strip():
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
                row = connection.execute(
                    f"INSERT INTO tasks (content) VALUES (?) RETURNING {_TASK_FIELDS}",
                    (content,),
                ).fetchone()
                tasks.append(_task_from_row(row))
    return tasks


def list_tasks(database_path: str | Path) -> list[Task]:
    """Return all tasks ordered by their generated IDs."""
    with closing(sqlite3.connect(database_path)) as connection:
        rows = connection.execute(
            f"SELECT {_TASK_FIELDS} FROM tasks ORDER BY id"
        ).fetchall()
    return [_task_from_row(row) for row in rows]


def complete_task(database_path: str | Path, task_id: int) -> CompletionResult:
    """Complete one task by ID and distinguish idempotency from absence."""
    _validate_task_id(task_id)
    with closing(sqlite3.connect(database_path)) as connection:
        with connection:
            row = connection.execute(
                f"""UPDATE tasks SET completed = 1
                    WHERE id = ? AND completed = 0
                    RETURNING {_TASK_FIELDS}""",
                (task_id,),
            ).fetchone()
            if row is not None:
                return CompletionResult(CompletionStatus.COMPLETED, _task_from_row(row))
            row = connection.execute(
                f"SELECT {_TASK_FIELDS} FROM tasks WHERE id = ?", (task_id,)
            ).fetchone()
            if row is None:
                return CompletionResult(CompletionStatus.NOT_FOUND, None)
            return CompletionResult(
                CompletionStatus.ALREADY_COMPLETED, _task_from_row(row)
            )


def update_task(
    database_path: str | Path, task_id: int, content: str
) -> UpdateResult:
    """Replace one task's content while preserving completion and deadline."""
    _validate_task_id(task_id)
    if not isinstance(content, str) or not content.strip():
        raise ValueError("content must be a nonblank string")
    if "\n" in content or "\r" in content:
        raise ValueError("update_task expects one line")

    with closing(sqlite3.connect(database_path)) as connection:
        with connection:
            row = connection.execute(
                f"""UPDATE tasks SET content = ?
                    WHERE id = ? AND content <> ?
                    RETURNING {_TASK_FIELDS}""",
                (content, task_id, content),
            ).fetchone()
            if row is not None:
                return UpdateResult(UpdateStatus.UPDATED, _task_from_row(row))
            row = connection.execute(
                f"SELECT {_TASK_FIELDS} FROM tasks WHERE id = ?", (task_id,)
            ).fetchone()
            if row is None:
                return UpdateResult(UpdateStatus.NOT_FOUND, None)
            return UpdateResult(UpdateStatus.UNCHANGED, _task_from_row(row))


def set_task_deadline(
    database_path: str | Path, task_id: int, due_at: int
) -> DeadlineResult:
    """Set or replace one task deadline using Unix seconds."""
    _validate_task_id(task_id)
    if isinstance(due_at, bool) or not isinstance(due_at, int):
        raise ValueError("due_at must be an integer Unix timestamp")

    with closing(sqlite3.connect(database_path, isolation_level=None)) as connection:
        connection.execute("BEGIN IMMEDIATE")
        try:
            row = connection.execute(
                f"SELECT {_TASK_FIELDS} FROM tasks WHERE id = ?", (task_id,)
            ).fetchone()
            if row is None:
                connection.commit()
                return DeadlineResult(DeadlineStatus.NOT_FOUND, None)
            current = _task_from_row(row)
            if current.due_at == due_at:
                connection.commit()
                return DeadlineResult(DeadlineStatus.UNCHANGED, current)
            status = DeadlineStatus.SET if current.due_at is None else DeadlineStatus.UPDATED
            updated_row = connection.execute(
                f"UPDATE tasks SET due_at = ? WHERE id = ? RETURNING {_TASK_FIELDS}",
                (due_at, task_id),
            ).fetchone()
            connection.commit()
            return DeadlineResult(status, _task_from_row(updated_row))
        except Exception:
            connection.rollback()
            raise


def get_task(database_path: str | Path, task_id: int) -> Task | None:
    """Return one task by ID, or None when it does not exist."""
    _validate_task_id(task_id)
    with closing(sqlite3.connect(database_path)) as connection:
        row = connection.execute(
            f"SELECT {_TASK_FIELDS} FROM tasks WHERE id = ?", (task_id,)
        ).fetchone()
    return None if row is None else _task_from_row(row)


def delete_task(
    database_path: str | Path,
    task_id: int,
    *,
    expected_task: Task | None = None,
) -> DeleteResult:
    """Delete one task, optionally only when it still matches a snapshot."""
    _validate_task_id(task_id)
    if expected_task is not None:
        if not isinstance(expected_task, Task) or expected_task.id != task_id:
            raise ValueError("expected_task must be a Task with the requested ID")

    with closing(sqlite3.connect(database_path)) as connection:
        with connection:
            if expected_task is None:
                row = connection.execute(
                    f"DELETE FROM tasks WHERE id = ? RETURNING {_TASK_FIELDS}",
                    (task_id,),
                ).fetchone()
            else:
                row = connection.execute(
                    f"""DELETE FROM tasks
                        WHERE id = ? AND content = ? AND completed = ? AND due_at IS ?
                        RETURNING {_TASK_FIELDS}""",
                    (
                        task_id,
                        expected_task.content,
                        int(expected_task.completed),
                        expected_task.due_at,
                    ),
                ).fetchone()
            if row is not None:
                return DeleteResult(DeleteStatus.DELETED, _task_from_row(row))

            current = connection.execute(
                f"SELECT {_TASK_FIELDS} FROM tasks WHERE id = ?", (task_id,)
            ).fetchone()
            if current is None:
                return DeleteResult(DeleteStatus.NOT_FOUND, None)
            return DeleteResult(DeleteStatus.STALE, _task_from_row(current))


def _validate_task_id(task_id: int) -> None:
    if isinstance(task_id, bool) or not isinstance(task_id, int) or task_id <= 0:
        raise ValueError("task_id must be a positive integer")
