"""Format tool execution results into deterministic user-facing Vietnamese confirmations."""

from typing import Any


def format_tool_result(name: str, result: Any) -> str:
    """Validate tool result structure and format deterministic response.

    Raises ValueError on any schema violation, unknown tool, or unhandled input.
    """
    if not isinstance(name, str) or name not in (
        "create_task",
        "list_tasks",
    ):
        raise ValueError(f"Unknown or unsupported tool: {name!r}")

    if not isinstance(result, dict) or isinstance(result, bool):
        raise ValueError("Result must be a dictionary")

    if set(result.keys()) != {"tasks"}:
        raise ValueError("Result must contain exactly the 'tasks' key")

    tasks = result["tasks"]
    if not isinstance(tasks, list):
        raise ValueError("'tasks' must be a list")

    if name == "create_task" and len(tasks) == 0:
        raise ValueError("create_task must not return an empty task list")

    for index, item in enumerate(tasks):
        if not isinstance(item, dict) or isinstance(item, bool):
            raise ValueError(f"Task at index {index} must be a dictionary")

        if set(item.keys()) != {"id", "content"}:
            raise ValueError(
                f"Task at index {index} must contain exactly 'id' and 'content' keys"
            )

        task_id = item["id"]
        if isinstance(task_id, bool) or not isinstance(task_id, int) or task_id <= 0:
            raise ValueError(
                f"Task id at index {index} must be a positive integer, got {task_id!r}"
            )

        content = item["content"]
        if not isinstance(content, str) or not content.strip():
            raise ValueError(
                f"Task content at index {index} must be a non-empty string, got {content!r}"
            )

    if name == "create_task":
        if len(tasks) == 1:
            task = tasks[0]
            return f"Đã thêm [{task['id']}] {task['content']}"
        lines = [f"Đã thêm {len(tasks)} việc:"]
        for task in tasks:
            lines.append(f"[{task['id']}] {task['content']}")
        return "\n".join(lines)

    if name == "list_tasks":
        if len(tasks) == 0:
            return "Danh sách trống."
        lines = [f"Danh sách hiện có {len(tasks)} việc:"]
        for task in tasks:
            lines.append(f"[{task['id']}] {task['content']}")
        return "\n".join(lines)

    raise ValueError(f"Unhandled tool: {name!r}")
