"""Tool descriptions and explicit dispatch for the future model adapter."""

from dataclasses import asdict
from pathlib import Path

from nexus.storage.sqlite_db import create_task, list_tasks


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
        "description": "Xem toàn bộ danh sách khi người dùng yêu cầu xem các việc đã lưu.",
        "parameters": {
            "type": "object",
            "properties": {},
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
    if name not in ("create_task", "list_tasks"):
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
        tasks = []
        for line in lines:
            if line.strip():
                task = create_task(database_path, line)
                if task:
                    tasks.append(asdict(task))
                    
        if not tasks:
            raise ValueError("content must contain at least one nonblank line")
            
        return {"tasks": tasks}

    if arguments:
        raise ValueError("list_tasks does not accept arguments")
    return {"tasks": [asdict(task) for task in list_tasks(database_path)]}
