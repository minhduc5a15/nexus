"""Tool descriptions and explicit dispatch for the future model adapter."""

from dataclasses import asdict
from pathlib import Path

from nexus.storage.sqlite_db import complete_task, create_tasks, list_tasks, update_task


# Internal tool descriptions. A provider adapter can wrap these as needed.
TOOL_DEFINITIONS = [
    {
        "name": "create_task",
        "description": (
            "Thêm một hoặc nhiều việc khi người dùng yêu cầu ghi lại việc cần làm. "
            "Nội dung có thể gồm nhiều dòng (mỗi dòng một việc). Giữ nguyên nội dung, "
            "không tự tách các hoạt động trong cùng dòng. Không gọi khi chỉ chào hỏi "
            "hoặc khi chưa rõ người dùng muốn thêm việc."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "minLength": 1,
                    "description": "Nội dung việc cần làm, có thể xuống dòng để thêm nhiều việc.",
                }
            },
            "required": ["content"],
            "additionalProperties": False,
        },
    },
    {
        "name": "list_tasks",
        "description": (
            "Xem toàn bộ danh sách và trạng thái khi người dùng yêu cầu xem các việc đã lưu."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
    {
        "name": "complete_task",
        "description": (
            "Đánh dấu đúng một việc là đã hoàn thành khi người dùng nêu rõ ID. "
            "Không tìm việc theo nội dung và không tự chọn ID."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "id": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "ID chính xác do người dùng nêu.",
                }
            },
            "required": ["id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "update_task",
        "description": (
            "Sửa nguyên văn nội dung của đúng một task khi người dùng nêu rõ ID "
            "và nội dung mới. Không tìm task theo nội dung và không viết lại nội dung."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "id": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "ID chính xác do người dùng nêu.",
                },
                "content": {
                    "type": "string",
                    "minLength": 1,
                    "description": "Nội dung mới nguyên văn, đúng một dòng.",
                },
            },
            "required": ["id", "content"],
            "additionalProperties": False,
        },
    },
]


def execute_tool(
    database_path: str | Path, name: str, arguments: object
) -> dict:
    """Validate one decoded call and execute it against an initialized database.

    Invalid calls raise ValueError before touching storage. Database errors
    propagate to the caller. This validates structure, not user intent.
    """
    if name not in ("create_task", "list_tasks", "complete_task", "update_task"):
        raise ValueError("Unknown tool")
    if not isinstance(arguments, dict):
        raise ValueError("Tool arguments must be an object")

    if name == "create_task":
        if set(arguments) != {"content"}:
            raise ValueError("create_task requires exactly one argument: content")
        content = arguments["content"]
        if not isinstance(content, str) or not content.strip():
            raise ValueError("content must be a nonblank string")
        
        lines = content.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        nonblank = [line for line in lines if line.strip()]
        if not nonblank:
            raise ValueError("content must contain at least one nonblank line")
        return {"tasks": [asdict(task) for task in create_tasks(database_path, nonblank)]}

    if name == "list_tasks":
        if arguments:
            raise ValueError("list_tasks does not accept arguments")
        return {"tasks": [asdict(task) for task in list_tasks(database_path)]}

    if name == "complete_task":
        if set(arguments) != {"id"}:
            raise ValueError("complete_task requires exactly one argument: id")
        task_id = arguments["id"]
        if isinstance(task_id, bool) or not isinstance(task_id, int) or task_id <= 0:
            raise ValueError("id must be a positive integer")
        completion = complete_task(database_path, task_id)
        return {
            "status": completion.status.value,
            "task": asdict(completion.task) if completion.task is not None else None,
        }

    if set(arguments) != {"id", "content"}:
        raise ValueError("update_task requires exactly two arguments: id and content")
    task_id = arguments["id"]
    content = arguments["content"]
    if isinstance(task_id, bool) or not isinstance(task_id, int) or task_id <= 0:
        raise ValueError("id must be a positive integer")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("content must be a nonblank string")
    if "\n" in content or "\r" in content:
        raise ValueError("content must contain exactly one line")
    updated = update_task(database_path, task_id, content)
    return {
        "status": updated.status.value,
        "task": asdict(updated.task) if updated.task is not None else None,
    }
