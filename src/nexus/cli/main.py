"""Command-line entry point for task commands and the stateful agent chat."""

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path

from nexus.core.deadlines import DeadlineParseError, format_deadline, parse_deadline
from nexus.storage.sqlite_db import (
    DatabaseSchemaError,
    complete_task,
    create_task,
    delete_task,
    initialize_database,
    list_tasks,
    set_task_deadline,
    update_task,
)


def default_database_path() -> Path:
    """Return a persistent user-data path without writing inside the package."""
    data_home = os.environ.get("XDG_DATA_HOME")
    base = Path(data_home).expanduser() if data_home else Path.home() / ".local" / "share"
    return base / "nexus" / "nexus.db"


def _positive_id(value: str) -> int:
    try:
        task_id = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("ID phải là số nguyên dương") from error
    if task_id <= 0:
        raise argparse.ArgumentTypeError("ID phải là số nguyên dương")
    return task_id


def _single_line_content(value: str) -> str:
    if not value.strip() or "\n" in value or "\r" in value:
        raise argparse.ArgumentTypeError("Nội dung mới phải là một dòng không trống")
    return value


def print_tool_results(
    calls: list[dict], *, create_prefix: str = "Đã thêm qua AI"
) -> int:
    """Print completed tool effects and return the number of created tasks."""
    saved_count = 0
    for call in calls:
        name = call.get("name")
        result = call.get("result")
        tasks = result.get("tasks", []) if isinstance(result, dict) else []
        if name == "create_task":
            for task in tasks:
                print(
                    f"{create_prefix} [{task['id']}] {task['content']}",
                    flush=True,
                )
                saved_count += 1
        elif name == "list_tasks":
            print(f"AI đã xem {len(tasks)} việc trong danh sách.", flush=True)
        elif name == "complete_task" and isinstance(result, dict):
            task = result.get("task")
            if isinstance(task, dict):
                prefix = (
                    "Đã hoàn thành"
                    if result.get("status") == "completed"
                    else "Đã hoàn thành từ trước"
                )
                print(f"{prefix} [{task['id']}] {task['content']}", flush=True)
        elif name == "update_task" and isinstance(result, dict):
            task = result.get("task")
            if isinstance(task, dict):
                prefix = "Đã sửa" if result.get("status") == "updated" else "Không đổi"
                print(f"{prefix} [{task['id']}] {task['content']}", flush=True)
        elif name == "delete_task" and isinstance(result, dict):
            task = result.get("task")
            if result.get("status") == "deleted" and isinstance(task, dict):
                print(f"Đã xóa [{task['id']}] {task['content']}", flush=True)
        elif name == "set_task_deadline" and isinstance(result, dict):
            task = result.get("task")
            if isinstance(task, dict) and task.get("due_at") is not None:
                shown = format_deadline(task["due_at"])
                print(
                    f"Đã đặt hạn [{task['id']}] {shown}: {task['content']}",
                    flush=True,
                )
    return saved_count


def _database_snapshot(database_path: Path) -> list[dict] | dict:
    """Return a JSON-safe diagnostic snapshot without breaking the chat."""
    try:
        return [
            {"id": task.id, "content": task.content, "completed": task.completed, "due_at": task.due_at}
            for task in list_tasks(database_path)
        ]
    except (sqlite3.Error, OSError) as error:
        return {"error": type(error).__name__, "message": str(error)}


def _print_chat_trace(record: dict) -> None:
    """Write one complete turn trace to stderr, separate from user output."""
    print(f"\n=== TRACE TURN {record['turn']} ===", file=sys.stderr)
    print(json.dumps(record, ensure_ascii=False, indent=2), file=sys.stderr)
    print("=== END TRACE ===", file=sys.stderr)


def run_chat(
    database_path: Path, *, trace: bool = False, prompt_version: str | None = None
) -> int:
    """Run one in-memory AgentSession until EOF or the local /exit command."""
    import urllib.error

    from nexus.agent.client import ENDPOINT, PostToolExecutionError, chat
    from nexus.agent.session import AgentSession

    session_kwargs = {} if prompt_version is None else {"prompt_version": prompt_version}
    session = AgentSession(database_path, **session_kwargs)
    interactive = sys.stdin.isatty()
    had_error = False
    turn_number = 0
    if interactive:
        print(
            "NEXUS chat đã bắt đầu. Mỗi dòng là một tin nhắn; "
            "gõ /exit hoặc nhấn Ctrl+D để thoát.",
            file=sys.stderr,
        )

    while True:
        try:
            if interactive:
                print("NEXUS > ", end="", flush=True)
            line = sys.stdin.readline()
        except KeyboardInterrupt:
            if interactive:
                print()
            return 130

        if line == "":
            if interactive:
                print()
            break

        prompt = line.rstrip("\r\n")
        if prompt.strip().lower() == "/exit":
            break

        turn_number += 1
        state_before = session.state.value
        pending_before = session.pending_create
        pending_complete_before = session.pending_complete
        pending_edit_id_before = session.pending_edit_id
        pending_edit_content_before = session.pending_edit_content
        pending_deadline_id_before = session.pending_deadline_id
        pending_deadline_text_before = session.pending_deadline_text
        pending_deadline_reference_before = (
            session.pending_deadline_reference.isoformat()
            if session.pending_deadline_reference is not None
            else None
        )
        pending_delete_before = (
            {
                "id": session.pending_delete_task.id,
                "content": session.pending_delete_task.content,
                "completed": session.pending_delete_task.completed,
                "due_at": session.pending_delete_task.due_at,
            }
            if session.pending_delete_task is not None
            else None
        )
        database_before = _database_snapshot(database_path) if trace else None
        model_exchange = {
            "called": False,
            "endpoint": ENDPOINT,
            "request": None,
            "response": None,
        }
        result = None
        trace_error = None

        def generate(payload: dict) -> dict:
            model_exchange["called"] = True
            model_exchange["request"] = payload
            response = chat(payload)
            model_exchange["response"] = response
            return response

        try:
            result = session.run_turn(prompt, generate)
            print(f"NEXUS: {result['reply']}", flush=True)
        except PostToolExecutionError as error:
            print_tool_results(error.executed_calls, create_prefix="Đã thêm")
            if error.stage == "response_formatting":
                stage_message = "Chương trình không định dạng được câu trả lời"
            else:
                stage_message = "NEXUS không hoàn tất toàn bộ các thao tác"
            print(
                f"{stage_message}, nhưng các thao tác được liệt kê phía trên "
                "đã hoàn tất. Chương trình không tự thử lại.",
                file=sys.stderr,
            )
            print(f"Chi tiết: {error.cause}", file=sys.stderr)
            trace_error = {
                "type": type(error).__name__,
                "stage": error.stage,
                "message": str(error),
                "cause": str(error.cause),
                "proposed_calls": error.proposed_calls,
                "authorized_calls": error.authorized_calls,
                "rejected_calls": error.rejected_calls,
                "executed_calls": error.executed_calls,
            }
            had_error = True
        except urllib.error.URLError as error:
            print(
                f"Lỗi kết nối tới AI (llama-server đã chạy chưa?): {error}",
                file=sys.stderr,
            )
            trace_error = {
                "type": type(error).__name__,
                "stage": "model_request",
                "message": str(error),
            }
            had_error = True
        except (sqlite3.Error, OSError, RuntimeError, ValueError) as error:
            print(f"Lỗi khi xử lý hội thoại: {error}", file=sys.stderr)
            trace_error = {
                "type": type(error).__name__,
                "stage": "turn_processing",
                "message": str(error),
            }
            had_error = True

        if trace:
            if result is not None:
                source = result.get("source")
                status = result.get("status")
                model_reply = result.get("model_reply")
                proposed_calls = result.get("proposed_calls", [])
                authorized_calls = result.get("authorized_calls", [])
                rejected_calls = result.get("rejected_calls", [])
                confirmation = result.get("confirmation")
                executed_calls = result.get("calls", [])
                application_reply = result.get("reply")
            else:
                source = (
                    "model"
                    if model_exchange["called"]
                    else "session_continuation"
                    if state_before != "idle"
                    else "session"
                )
                status = "error_after_execution" if isinstance(
                    trace_error, dict
                ) and trace_error.get("executed_calls") else "error"
                model_reply = None
                proposed_calls = (trace_error or {}).get("proposed_calls", [])
                authorized_calls = (trace_error or {}).get("authorized_calls", [])
                rejected_calls = (trace_error or {}).get("rejected_calls", [])
                confirmation = None
                executed_calls = (trace_error or {}).get("executed_calls", [])
                application_reply = None

            _print_chat_trace(
                {
                    "turn": turn_number,
                    "input": prompt,
                    "session": {
                        "state_before": state_before,
                        "state_after": session.state.value,
                        "pending_create_before": pending_before,
                        "pending_create_after": session.pending_create,
                        "pending_complete_before": pending_complete_before,
                        "pending_complete_after": session.pending_complete,
                        "pending_edit_id_before": pending_edit_id_before,
                        "pending_edit_id_after": session.pending_edit_id,
                        "pending_edit_content_before": pending_edit_content_before,
                        "pending_edit_content_after": session.pending_edit_content,
                        "pending_deadline_id_before": pending_deadline_id_before,
                        "pending_deadline_id_after": session.pending_deadline_id,
                        "pending_deadline_text_before": pending_deadline_text_before,
                        "pending_deadline_text_after": session.pending_deadline_text,
                        "pending_deadline_reference_before": pending_deadline_reference_before,
                        "pending_deadline_reference_after": (
                            session.pending_deadline_reference.isoformat()
                            if session.pending_deadline_reference is not None
                            else None
                        ),
                        "pending_delete_before": pending_delete_before,
                        "pending_delete_after": (
                            {
                                "id": session.pending_delete_task.id,
                                "content": session.pending_delete_task.content,
                                "completed": session.pending_delete_task.completed,
                                "due_at": session.pending_delete_task.due_at,
                            }
                            if session.pending_delete_task is not None
                            else None
                        ),
                    },
                    "model": model_exchange,
                    "runtime": {
                        "source": source,
                        "status": status,
                        "model_reply": model_reply,
                    },
                    "proposed_calls": proposed_calls,
                    "contract_validation": {
                        "max_tool_calls": 1,
                        "proposal_count": len(proposed_calls),
                        "status": (
                            "rejected"
                            if any(
                                call.get("reason") == "invalid_arguments"
                                for call in rejected_calls
                            )
                            else "passed"
                            if proposed_calls
                            else "not_applicable"
                        ),
                    },
                    "authorized_calls": authorized_calls,
                    "rejected_calls": rejected_calls,
                    "confirmation": confirmation,
                    "executed_calls": executed_calls,
                    "database": {
                        "before": database_before,
                        "after": _database_snapshot(database_path),
                    },
                    "application_reply": application_reply,
                    "error": trace_error,
                }
            )

    return 1 if had_error else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="NEXUS — ghi nhanh việc cần làm")
    parser.add_argument(
        "--db",
        type=Path,
        default=None,
        help="Đường dẫn database (mặc định: $XDG_DATA_HOME/nexus/nexus.db)",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    add_parser = commands.add_parser("add", help="Thêm việc, mỗi dòng một việc")
    add_parser.add_argument(
        "content", nargs="?", help="Nội dung; bỏ qua để đọc từ stdin đến EOF"
    )
    commands.add_parser("list", help="Xem các việc đã lưu")
    complete_parser = commands.add_parser(
        "complete", help="Đánh dấu một việc là đã hoàn thành theo ID"
    )
    complete_parser.add_argument("id", type=_positive_id, help="ID số nguyên dương")
    edit_parser = commands.add_parser("edit", help="Sửa nội dung một việc theo ID")
    edit_parser.add_argument("id", type=_positive_id, help="ID số nguyên dương")
    edit_parser.add_argument("content", type=_single_line_content, help="Nội dung mới")
    delete_parser = commands.add_parser("delete", help="Xóa vĩnh viễn một việc theo ID")
    delete_parser.add_argument("id", type=_positive_id, help="ID số nguyên dương")
    deadline_parser = commands.add_parser(
        "deadline", help="Đặt hoặc đổi thời hạn của một việc theo ID"
    )
    deadline_parser.add_argument("id", type=_positive_id, help="ID số nguyên dương")
    deadline_parser.add_argument(
        "when", help='Thời hạn, ví dụ "8 giờ sáng mai"'
    )
    ask_parser = commands.add_parser("ask", help="Ra lệnh bằng ngôn ngữ tự nhiên (cần chạy llama-server)")
    ask_parser.add_argument("prompt", nargs="?", help="Nội dung yêu cầu; bỏ qua để đọc từ stdin đến EOF")
    from nexus.agent.prompts import DEFAULT_PROMPT_VERSION, SYSTEM_PROMPTS
    ask_parser.add_argument(
        "--prompt-version",
        choices=sorted(SYSTEM_PROMPTS),
        default=DEFAULT_PROMPT_VERSION,
        help="Phiên bản system prompt cho agent",
    )
    chat_parser = commands.add_parser(
        "chat",
        help="Hội thoại nhiều lượt trong một session (cần chạy llama-server)",
    )
    chat_parser.add_argument(
        "--prompt-version",
        choices=sorted(SYSTEM_PROMPTS),
        default=DEFAULT_PROMPT_VERSION,
        help="Phiên bản system prompt cho agent",
    )
    chat_parser.add_argument(
        "--trace",
        action="store_true",
        help="Hiện trace model, policy, tool, session và SQLite cho từng lượt",
    )
    args = parser.parse_args()

    uses_default_database = args.db is None
    if uses_default_database:
        args.db = default_database_path()

    saved_count = 0
    try:
        if uses_default_database:
            args.db.parent.mkdir(parents=True, exist_ok=True)
        initialize_database(args.db)
        if args.command == "add":
            if args.content is None:
                if sys.stdin.isatty():
                    print(
                        "Nhập mỗi việc trên một dòng. Kết thúc bằng Ctrl+D "
                        "ở đầu dòng mới (Linux).",
                        file=sys.stderr,
                    )
                content = sys.stdin.read()
            else:
                content = args.content

            # Normalize line endings only; preserve spaces and other content.
            lines = content.replace("\r\n", "\n").replace("\r", "\n").split("\n")
            for line in lines:
                task = create_task(args.db, line)
                if task is not None:
                    saved_count += 1
                    print(f"Đã thêm [{task.id}] {task.content}")
            if saved_count == 0:
                print("Không có nội dung để thêm.")
        elif args.command == "complete":
            completion = complete_task(args.db, args.id)
            if completion.status.value == "not_found":
                print(f"Không tìm thấy việc có ID {args.id}.")
                return 1
            if completion.status.value == "completed":
                print(f"Đã hoàn thành [{completion.task.id}] {completion.task.content}")
            else:
                print(
                    f"Việc [{completion.task.id}] đã hoàn thành trước đó: "
                    f"{completion.task.content}"
                )
        elif args.command == "edit":
            updated = update_task(args.db, args.id, args.content)
            if updated.status.value == "not_found":
                print(f"Không tìm thấy việc có ID {args.id}.")
                return 1
            if updated.status.value == "updated":
                print(f"Đã sửa [{updated.task.id}] thành: {updated.task.content}")
            else:
                print(
                    f"Việc [{updated.task.id}] đã có nội dung này: "
                    f"{updated.task.content}"
                )
        elif args.command == "delete":
            deleted = delete_task(args.db, args.id)
            if deleted.status.value == "not_found":
                print(f"Không tìm thấy việc có ID {args.id}.")
                return 1
            print(f"Đã xóa [{deleted.task.id}] {deleted.task.content}")
        elif args.command == "deadline":
            try:
                due_at = parse_deadline(args.when)
            except DeadlineParseError as error:
                print(f"Thời hạn không hợp lệ: {error}", file=sys.stderr)
                return 2
            deadline = set_task_deadline(args.db, args.id, due_at)
            if deadline.status.value == "not_found":
                print(f"Không tìm thấy việc có ID {args.id}.")
                return 1
            shown = format_deadline(deadline.task.due_at)
            if deadline.status.value == "set":
                print(
                    f"Đã đặt hạn [{deadline.task.id}] vào {shown}: "
                    f"{deadline.task.content}"
                )
            elif deadline.status.value == "updated":
                print(
                    f"Đã đổi hạn [{deadline.task.id}] thành {shown}: "
                    f"{deadline.task.content}"
                )
            else:
                print(
                    f"Việc [{deadline.task.id}] đã có hạn {shown}: "
                    f"{deadline.task.content}"
                )
        elif args.command == "ask":
            from nexus.agent.client import PostToolExecutionError, chat, run_turn
            import urllib.error

            if args.prompt is None:
                if sys.stdin.isatty():
                    print(
                        "Nhập yêu cầu của bạn. Kết thúc bằng Ctrl+D ở đầu dòng mới.",
                        file=sys.stderr,
                    )
                prompt = sys.stdin.read()
            else:
                prompt = args.prompt

            if not prompt.strip():
                print("Không có yêu cầu nào.", file=sys.stderr)
                return 0

            print("Đang xử lý qua AI...", file=sys.stderr)
            try:
                result = run_turn(
                    args.db,
                    prompt,
                    chat,
                    prompt_version=args.prompt_version,
                )
                for call in result.get("calls", []):
                    if call.get("name") == "create_task":
                        call_result = call.get("result")
                        if isinstance(call_result, dict):
                            saved_count += len(call_result.get("tasks", []))
                print(f"\nAI: {result['reply']}")
                if result.get("status") == "needs_confirmation":
                    print(
                        "Lệnh ask không giữ session; hãy dùng nexus chat để xác nhận."
                    )
            except PostToolExecutionError as error:
                saved_count += print_tool_results(error.executed_calls)
                if error.stage == "response_formatting":
                    stage_message = "Chương trình không định dạng được câu trả lời"
                else:
                    stage_message = "AI không hoàn tất toàn bộ các thao tác"
                print(
                    f"{stage_message}, nhưng các thao tác được liệt kê phía trên "
                    "đã hoàn tất. Chương trình không tự thử lại.",
                    file=sys.stderr,
                )
                print(f"Chi tiết: {error.cause}", file=sys.stderr)
                return 1
            except urllib.error.URLError as error:
                print(f"Lỗi kết nối tới AI (llama-server đã chạy chưa?): {error}", file=sys.stderr)
                return 1
            except Exception as error:
                print(f"Lỗi khi xử lý qua AI: {error}", file=sys.stderr)
                return 1
        elif args.command == "chat":
            return run_chat(
                args.db, trace=args.trace, prompt_version=args.prompt_version
            )
        else:
            tasks = list_tasks(args.db)
            if not tasks:
                print("Danh sách trống.")
            for task in tasks:
                marker = "x" if task.completed else " "
                line = f"[{task.id}] [{marker}] {task.content}"
                if task.due_at is not None:
                    line += f" — hạn {format_deadline(task.due_at)}"
                print(line)
    except (sqlite3.Error, OSError, DatabaseSchemaError) as error:
        print(f"Lỗi: {error}", file=sys.stderr)
        if saved_count:
            print(
                f"Đã lưu {saved_count} việc trước khi xảy ra lỗi; "
                "hãy xem danh sách trước khi thử lại.",
                file=sys.stderr,
            )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
