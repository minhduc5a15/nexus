"""Format tool execution results into deterministic user-facing Vietnamese confirmations."""

from typing import Any

from nexus.agent.policy import PolicyReason


def format_rejection(reason: PolicyReason) -> str:
    """Describe a rejected proposal without exposing model arguments."""
    if reason in (PolicyReason.UNSUPPORTED_ACTION, PolicyReason.UNSUPPORTED_TOOL):
        return "NEXUS hiện chỉ hỗ trợ thêm, xem, hoàn thành, sửa và xóa việc theo ID."
    if reason in (
        PolicyReason.CONTENT_NOT_GROUNDED,
        PolicyReason.CONTENT_BOUNDARY_MISMATCH,
    ):
        return "Tôi chưa lưu việc vì nội dung không khớp lời bạn. Hãy viết lại yêu cầu."
    return "Không có thao tác nào được thực hiện."


def format_tool_result(name: str, result: Any) -> str:
    """Validate tool result structure and format deterministic response.

    Raises ValueError on any schema violation, unknown tool, or unhandled input.
    """
    if not isinstance(name, str) or name not in (
        "create_task", "list_tasks", "complete_task", "update_task", "delete_task"
    ):
        raise ValueError(f"Unknown or unsupported tool: {name!r}")

    if not isinstance(result, dict) or isinstance(result, bool):
        raise ValueError("Result must be a dictionary")

    if name == "complete_task":
        if set(result.keys()) != {"status", "task"}:
            raise ValueError("Completion result must contain exactly 'status' and 'task'")
        status = result["status"]
        if status not in ("completed", "already_completed", "not_found"):
            raise ValueError(f"Unknown completion status: {status!r}")
        task = result["task"]
        if status == "not_found":
            if task is not None:
                raise ValueError("not_found completion must have a null task")
            return "Không tìm thấy việc có ID đã yêu cầu."
        _validate_task(task, "completion")
        if not task["completed"]:
            raise ValueError("Completed task result must have completed=true")
        if status == "completed":
            return f"Đã hoàn thành [{task['id']}] {task['content']}"
        return f"Việc [{task['id']}] đã hoàn thành trước đó: {task['content']}"

    if name == "delete_task":
        if set(result.keys()) != {"status", "task"}:
            raise ValueError("Delete result must contain exactly 'status' and 'task'")
        status = result["status"]
        if status not in ("deleted", "not_found", "stale"):
            raise ValueError(f"Unknown delete status: {status!r}")
        task = result["task"]
        if status == "not_found":
            if task is not None:
                raise ValueError("not_found delete must have a null task")
            return "Không tìm thấy việc có ID đã yêu cầu."
        _validate_task(task, "delete")
        if status == "deleted":
            return f"Đã xóa [{task['id']}] {task['content']}"
        marker = "x" if task["completed"] else " "
        return (
            f"Việc [{task['id']}] đã thay đổi thành [{marker}] {task['content']}. "
            "Hãy xác nhận lại nếu bạn vẫn muốn xóa."
        )

    if name == "update_task":
        if set(result.keys()) != {"status", "task"}:
            raise ValueError("Update result must contain exactly 'status' and 'task'")
        status = result["status"]
        if status not in ("updated", "unchanged", "not_found"):
            raise ValueError(f"Unknown update status: {status!r}")
        task = result["task"]
        if status == "not_found":
            if task is not None:
                raise ValueError("not_found update must have a null task")
            return "Không tìm thấy việc có ID đã yêu cầu."
        _validate_task(task, "update")
        if status == "updated":
            return f"Đã sửa [{task['id']}] thành: {task['content']}"
        return f"Việc [{task['id']}] đã có nội dung này: {task['content']}"

    if set(result.keys()) != {"tasks"}:
        raise ValueError("Result must contain exactly the 'tasks' key")

    tasks = result["tasks"]
    if not isinstance(tasks, list):
        raise ValueError("'tasks' must be a list")

    if name == "create_task" and len(tasks) == 0:
        raise ValueError("create_task must not return an empty task list")

    for index, item in enumerate(tasks):
        _validate_task(item, f"index {index}")

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
            marker = "x" if task["completed"] else " "
            lines.append(f"[{task['id']}] [{marker}] {task['content']}")
        return "\n".join(lines)

    raise ValueError(f"Unhandled tool: {name!r}")


def _validate_task(item: Any, location: str) -> None:
    if not isinstance(item, dict) or isinstance(item, bool):
        raise ValueError(f"Task at {location} must be a dictionary")
    if set(item.keys()) != {"id", "content", "completed"}:
        raise ValueError(
            f"Task at {location} must contain exactly 'id', 'content' and 'completed' keys"
        )
    task_id = item["id"]
    if isinstance(task_id, bool) or not isinstance(task_id, int) or task_id <= 0:
        raise ValueError(f"Task id at {location} must be a positive integer, got {task_id!r}")
    content = item["content"]
    if not isinstance(content, str) or not content.strip():
        raise ValueError(f"Task content at {location} must be a non-empty string, got {content!r}")
    if type(item["completed"]) is not bool:
        raise ValueError(f"Task completed at {location} must be a boolean")
