"""Command-line entry point for adding and listing tasks."""

import argparse
import os
import sqlite3
import sys
from pathlib import Path

from nexus.storage.sqlite_db import create_task, initialize_database, list_tasks


def default_database_path() -> Path:
    """Return a persistent user-data path without writing inside the package."""
    data_home = os.environ.get("XDG_DATA_HOME")
    base = Path(data_home).expanduser() if data_home else Path.home() / ".local" / "share"
    return base / "nexus" / "nexus.db"


def print_tool_results(calls: list[dict]) -> int:
    """Print completed tool effects and return the number of created tasks."""
    saved_count = 0
    for call in calls:
        name = call.get("name")
        result = call.get("result")
        tasks = result.get("tasks", []) if isinstance(result, dict) else []
        if name == "create_task":
            for task in tasks:
                print(f"Đã thêm qua AI [{task['id']}] {task['content']}")
                saved_count += 1
        elif name == "list_tasks":
            print(f"AI đã xem {len(tasks)} việc trong danh sách.")
    return saved_count


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
    ask_parser = commands.add_parser("ask", help="Ra lệnh bằng ngôn ngữ tự nhiên (cần chạy llama-server)")
    ask_parser.add_argument("prompt", nargs="?", help="Nội dung yêu cầu; bỏ qua để đọc từ stdin đến EOF")
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
                result = run_turn(args.db, prompt, chat)
                for call in result.get("calls", []):
                    if call.get("name") == "create_task":
                        call_result = call.get("result")
                        if isinstance(call_result, dict):
                            saved_count += len(call_result.get("tasks", []))
                print(f"\nAI: {result['reply']}")
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
        else:
            tasks = list_tasks(args.db)
            if not tasks:
                print("Danh sách trống.")
            for task in tasks:
                print(f"[{task.id}] {task.content}")
    except (sqlite3.Error, OSError) as error:
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
