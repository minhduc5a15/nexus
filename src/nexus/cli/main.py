"""Command-line entry point for adding and listing tasks."""

import argparse
import sqlite3
import sys
from pathlib import Path

from nexus.storage.sqlite_db import create_task, initialize_database, list_tasks


def main() -> int:
    parser = argparse.ArgumentParser(description="NEXUS — ghi nhanh việc cần làm")
    parser.add_argument(
        "--db",
        type=Path,
        default=Path(__file__).resolve().with_name("nexus.db"),
        help="Đường dẫn database (mặc định: nexus.db cạnh cli.py)",
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

    saved_count = 0
    try:
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
            from nexus.agent.client import chat, run_turn
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
                    name = call["name"]
                    if name == "create_task":
                        tasks = call["result"].get("tasks", [])
                        for t in tasks:
                            print(f"Đã thêm qua AI [{t['id']}] {t['content']}")
                            saved_count += 1
                    elif name == "list_tasks":
                        tasks = call["result"].get("tasks", [])
                        print(f"AI đã xem {len(tasks)} việc trong danh sách.")
                
                print(f"\nAI: {result['reply']}")
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
