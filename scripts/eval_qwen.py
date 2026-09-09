"""Evaluate local Qwen3-1.7B via llama.cpp on task datasets."""

import argparse
import json
import sqlite3
import tempfile
import urllib.error
import urllib.request
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter, sleep

from nexus.agent.prompts import SYSTEM_PROMPTS
from nexus.agent.client import ENDPOINT, chat, response_diagnostics, run_turn
from nexus.storage.sqlite_db import create_task, initialize_database, list_tasks


CASES_PATH = Path(__file__).resolve().parents[1] / "evals" / "basic_tasks.json"


def evaluate_case(
    case: dict,
    generate,
    *,
    prompt_version: str = "v1",
    settings: dict | None = None,
) -> dict:
    observed_calls = []
    diagnostics = []
    request_count = 0

    def observe(payload):
        nonlocal request_count
        request_count += 1
        response = generate(payload)
        diagnostics.append(response_diagnostics(response))
        choices = response.get("choices", [])
        if choices:
            message = choices[0].get("message", {})
            for call in message.get("tool_calls") or []:
                function = call.get("function", {})
                args_raw = function.get("arguments", "{}")
                try:
                    args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                except (ValueError, TypeError):
                    args = args_raw
                observed_calls.append({"name": function.get("name"), "arguments": args})
        return response

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "eval.db"
        initialize_database(path)
        for content in case["initial_tasks"]:
            create_task(path, content)
        before = [asdict(task) for task in list_tasks(path)]
        started = perf_counter()
        result = None
        error = None
        try:
            result = run_turn(
                path,
                case["prompt"],
                observe,
                prompt_version=prompt_version,
                settings=settings,
            )
        except (RuntimeError, ValueError, sqlite3.Error, OSError) as failure:
            error = {"type": type(failure).__name__, "message": str(failure)}
        elapsed = perf_counter() - started
        after = [asdict(task) for task in list_tasks(path)]

    checks = {
        "calls_match": observed_calls == case["expected_calls"],
        "tasks_match": [task["content"] for task in after] == case["expected_tasks"],
        "existing_tasks_unchanged": after[: len(before)] == before,
    }
    status = "error" if error else ("pass" if all(checks.values()) else "fail")
    return {
        "id": case["id"],
        "prompt": case["prompt"],
        "expected_calls": case["expected_calls"],
        "expected_tasks": case["expected_tasks"],
        "observed_calls": observed_calls,
        "database_before": before,
        "database_after": after,
        "reply": result["reply"] if result else None,
        "reply_expectation": case["reply_expectation"],
        "reply_review": "pending_manual_review",
        "checks": checks,
        "status": status,
        "error": error,
        "response_diagnostics": diagnostics,
        "api_requests": request_count,
        "elapsed_seconds": round(elapsed, 3),
    }


def send_chat(endpoint: str, payload: dict, timeout: float = 120.0) -> dict:
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        try:
            details = json.load(error)
        except (ValueError, OSError):
            details = {}
        diagnostic = {"http_status": error.code, "details": details}
        raise RuntimeError(f"llama-server HTTP error: {json.dumps(diagnostic)}") from None
    except urllib.error.URLError as error:
        raise RuntimeError(f"Không kết nối được llama-server tại {endpoint}: {error.reason}") from None


def main() -> int:
    parser = argparse.ArgumentParser(description="Đánh giá Qwen3-1.7B trên bộ ca Todo")
    parser.add_argument("--output", type=Path, required=True, help="File JSON mới để lưu kết quả")
    parser.add_argument("--case", action="append", dest="case_ids", help="Chỉ chạy ID này; có thể lặp tùy chọn")
    parser.add_argument("--cases", type=Path, default=CASES_PATH, help="Dataset JSON")
    parser.add_argument("--endpoint", default=ENDPOINT, help="llama.cpp server endpoint")
    parser.add_argument("--prompt-version", choices=SYSTEM_PROMPTS, default="v1")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--request-interval", type=float, default=0.0, help="Khoảng nghỉ tối thiểu giữa các ca")
    args = parser.parse_args()

    if args.request_interval < 0:
        parser.error("--request-interval phải không âm")

    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    if args.case_ids:
        unknown = set(args.case_ids) - {case["id"] for case in cases}
        if unknown:
            parser.error(f"Case không tồn tại: {sorted(unknown)}")
        cases = [case for case in cases if case["id"] in args.case_ids]

    common_settings = {
        "model": "qwen3-1.7b-q8_0",
        "temperature": args.temperature,
        "top_p": 0.8,
        "top_k": 20,
        "min_p": 0,
        "max_tokens": 256,
        "chat_template_kwargs": {"enable_thinking": False},
    }

    report = {
        "model": "qwen3-1.7b-q8_0",
        "endpoint": args.endpoint,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "prompt_version": args.prompt_version,
        "system_prompt": SYSTEM_PROMPTS[args.prompt_version],
        "settings": common_settings,
        "request_interval_seconds": args.request_interval,
        "scope": "Tool calls and database effects; replies require manual review. One run per case.",
        "cases": [],
    }

    last_request_finished = None

    def paced_generate(payload):
        nonlocal last_request_finished
        if last_request_finished is not None and args.request_interval > 0:
            sleep(max(0, args.request_interval - (perf_counter() - last_request_finished)))
        try:
            return send_chat(args.endpoint, payload)
        finally:
            last_request_finished = perf_counter()

    with args.output.open("x", encoding="utf-8") as output:
        for case in cases:
            result = evaluate_case(
                case,
                paced_generate,
                prompt_version=args.prompt_version,
                settings=common_settings,
            )
            report["cases"].append(result)
            output.seek(0)
            json.dump(report, output, ensure_ascii=False, indent=2)
            output.truncate()
            output.flush()
            print(f"{result['id']}: {result['status']} ({result['elapsed_seconds']}s)", flush=True)
            if result["status"] == "error":
                break

    return 0 if len(report["cases"]) == len(cases) and all(
        case["status"] == "pass" for case in report["cases"]
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())

