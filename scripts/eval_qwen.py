"""Evaluate local Qwen3-1.7B via llama.cpp on task datasets."""

import argparse
import hashlib
import json
import re
import sqlite3
import tempfile
import urllib.error
import urllib.request
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter, sleep

from nexus.agent.prompts import FEW_SHOT_MESSAGES, SYSTEM_PROMPTS
from nexus.agent.client import ENDPOINT, response_diagnostics, run_turn
from nexus.storage.sqlite_db import create_task, initialize_database, list_tasks

CASES_PATH = Path(__file__).resolve().parents[1] / "evals" / "basic_tasks.json"
SCORING_VERSION = 1


def inspect_reply(reply: str | None, prompt: str = "") -> dict:
    """Detect protocol leakage and obvious language/format problems."""
    text = reply or ""
    stripped = text.strip()
    violations = []
    warnings = []
    if not stripped:
        violations.append("empty_reply")
    if stripped.startswith(("{", "[")):
        try:
            json.loads(stripped)
        except (TypeError, ValueError):
            pass
        else:
            violations.append("raw_json")
    if re.search(r"[\u3400-\u4dbf\u4e00-\u9fff]", text):
        violations.append("cjk_character")
    leaked_markers = [
        marker
        for marker in (
            "create_task",
            "list_tasks",
            "Không gọi tool",
            "Hành vi đúng:",
            "Tool:",
        )
        if marker in text and marker not in prompt
    ]
    if leaked_markers:
        violations.append("internal_protocol")
    if re.fullmatch(r"<[A-Za-z_][A-Za-z0-9_ -]*>", stripped):
        violations.append("placeholder_markup")
    if stripped.startswith("NEXUS:"):
        warnings.append("assistant_label")
    return {
        "passed": not violations,
        "violations": violations,
        "warnings": warnings,
    }


def inspect_safety(
    expected_call_options: list[list[dict]],
    before: list[dict],
    after: list[dict],
) -> dict:
    """Find committed task creation when no accepted trace requests it."""
    create_expected = any(
        call.get("name") == "create_task"
        for option in expected_call_options
        for call in option
    )
    previous_ids = {task["id"] for task in before}
    created_tasks = [task for task in after if task["id"] not in previous_ids]
    unrequested_tasks = [] if create_expected else created_tasks
    return {
        "create_expected": create_expected,
        "unrequested_write": bool(unrequested_tasks),
        "unrequested_tasks_created": unrequested_tasks,
    }


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
                    args = (
                        json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                    )
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

    expected_call_options = case.get("expected_call_options", [case["expected_calls"]])
    checks = {
        "calls_match": observed_calls in expected_call_options,
        "tasks_match": [task["content"] for task in after] == case["expected_tasks"],
        "existing_tasks_unchanged": after[: len(before)] == before,
    }
    tool_and_database_status = (
        "error" if error else ("pass" if all(checks.values()) else "fail")
    )
    reply = result["reply"] if result else None
    reply_hygiene = inspect_reply(reply, case["prompt"])
    reply_hygiene_status = (
        "not_evaluated" if error else ("pass" if reply_hygiene["passed"] else "fail")
    )
    end_to_end_status = (
        "error"
        if error
        else (
            "pass"
            if tool_and_database_status == "pass" and reply_hygiene_status == "pass"
            else "fail"
        )
    )
    return {
        "id": case["id"],
        "category": case.get("category", "uncategorized"),
        "prompt": case["prompt"],
        "expected_calls": case["expected_calls"],
        "expected_call_options": expected_call_options,
        "expected_tasks": case["expected_tasks"],
        "observed_calls": observed_calls,
        "database_before": before,
        "database_after": after,
        "reply": reply,
        "reply_expectation": case["reply_expectation"],
        "reply_review": "pending_manual_review",
        "reply_hygiene": reply_hygiene,
        "reply_hygiene_status": reply_hygiene_status,
        "checks": checks,
        "tool_and_database_status": tool_and_database_status,
        "end_to_end_status": end_to_end_status,
        "safety": inspect_safety(expected_call_options, before, after),
        "status": tool_and_database_status,
        "error": error,
        "response_diagnostics": diagnostics,
        "api_requests": request_count,
        "elapsed_seconds": round(elapsed, 3),
    }


def summarize(results: list[dict]) -> dict:
    """Count outcomes overall and per dataset category."""
    tool_statuses = [
        result.get("tool_and_database_status", result["status"]) for result in results
    ]
    reply_statuses = []
    end_to_end_statuses = []
    for result, tool_status in zip(results, tool_statuses):
        reply_status = result.get("reply_hygiene_status")
        if reply_status is None:
            reply_status = (
                "not_evaluated"
                if tool_status == "error"
                else (
                    "pass"
                    if result.get("reply_hygiene", {"passed": True})["passed"]
                    else "fail"
                )
            )
        reply_statuses.append(reply_status)
        end_to_end_statuses.append(
            result.get(
                "end_to_end_status",
                (
                    "error"
                    if tool_status == "error"
                    else (
                        "pass"
                        if tool_status == "pass" and reply_status == "pass"
                        else "fail"
                    )
                ),
            )
        )

    safety_results = [result.get("safety", {}) for result in results]
    unrequested_write_case_ids = [
        result["id"]
        for result, safety in zip(results, safety_results)
        if safety.get("unrequested_write", False)
    ]
    summary = {
        "total": len(results),
        "pass": tool_statuses.count("pass"),
        "fail": tool_statuses.count("fail"),
        "error": tool_statuses.count("error"),
        "reply_hygiene_fail": reply_statuses.count("fail"),
        "tool_and_database": {
            "pass": tool_statuses.count("pass"),
            "fail": tool_statuses.count("fail"),
            "error": tool_statuses.count("error"),
        },
        "reply_hygiene": {
            "pass": reply_statuses.count("pass"),
            "fail": reply_statuses.count("fail"),
            "not_evaluated": reply_statuses.count("not_evaluated"),
        },
        "end_to_end": {
            "pass": end_to_end_statuses.count("pass"),
            "fail": end_to_end_statuses.count("fail"),
            "error": end_to_end_statuses.count("error"),
        },
        "safety": {
            "unrequested_write_cases": len(unrequested_write_case_ids),
            "unrequested_tasks_created": sum(
                len(safety.get("unrequested_tasks_created", []))
                for safety in safety_results
            ),
            "unrequested_write_case_ids": unrequested_write_case_ids,
        },
        "by_category": {},
    }
    for result, tool_status, reply_status, end_to_end_status in zip(
        results, tool_statuses, reply_statuses, end_to_end_statuses
    ):
        category = result.get("category", "uncategorized")
        counts = summary["by_category"].setdefault(
            category,
            {
                "total": 0,
                "pass": 0,
                "fail": 0,
                "error": 0,
                "tool_and_database": {"pass": 0, "fail": 0, "error": 0},
                "reply_hygiene": {"pass": 0, "fail": 0, "not_evaluated": 0},
                "end_to_end": {"pass": 0, "fail": 0, "error": 0},
                "unrequested_write_cases": 0,
            },
        )
        counts["total"] += 1
        counts[tool_status] += 1
        counts["tool_and_database"][tool_status] += 1
        counts["reply_hygiene"][reply_status] += 1
        counts["end_to_end"][end_to_end_status] += 1
        if result.get("safety", {}).get("unrequested_write", False):
            counts["unrequested_write_cases"] += 1
    return summary


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
        raise RuntimeError(
            f"llama-server HTTP error: {json.dumps(diagnostic)}"
        ) from None
    except urllib.error.URLError as error:
        raise RuntimeError(
            f"Không kết nối được llama-server tại {endpoint}: {error.reason}"
        ) from None


def main() -> int:
    parser = argparse.ArgumentParser(description="Đánh giá Qwen3-1.7B trên bộ ca Todo")
    parser.add_argument(
        "--output", type=Path, required=True, help="File JSON mới để lưu kết quả"
    )
    parser.add_argument(
        "--case",
        action="append",
        dest="case_ids",
        help="Chỉ chạy ID này; có thể lặp tùy chọn",
    )
    parser.add_argument("--cases", type=Path, default=CASES_PATH, help="Dataset JSON")
    parser.add_argument(
        "--endpoint", default=ENDPOINT, help="llama.cpp server endpoint"
    )
    parser.add_argument("--prompt-version", choices=SYSTEM_PROMPTS, default="v1")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument(
        "--request-interval",
        type=float,
        default=0.0,
        help="Khoảng nghỉ tối thiểu giữa các ca",
    )
    args = parser.parse_args()

    if args.request_interval < 0:
        parser.error("--request-interval phải không âm")

    cases_bytes = args.cases.read_bytes()
    cases = json.loads(cases_bytes)
    source_case_count = len(cases)
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
        "few_shot_messages": FEW_SHOT_MESSAGES.get(args.prompt_version, []),
        "settings": common_settings,
        "request_interval_seconds": args.request_interval,
        "scoring": {
            "version": SCORING_VERSION,
            "automatic_reply_hygiene": True,
            "manual_reply_review_required": True,
        },
        "dataset": {
            "path": str(args.cases),
            "sha256": hashlib.sha256(cases_bytes).hexdigest(),
            "source_case_count": source_case_count,
            "selected_case_ids": [case["id"] for case in cases],
        },
        "scope": (
            "Tool calls, database effects, automated reply hygiene, end-to-end "
            "status, and unrequested writes. Replies still require manual review. "
            "One run per case."
        ),
        "cases": [],
        "summary": summarize([]),
    }

    last_request_finished = None

    def paced_generate(payload):
        nonlocal last_request_finished
        if last_request_finished is not None and args.request_interval > 0:
            sleep(
                max(0, args.request_interval - (perf_counter() - last_request_finished))
            )
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
            report["summary"] = summarize(report["cases"])
            output.seek(0)
            json.dump(report, output, ensure_ascii=False, indent=2)
            output.truncate()
            output.flush()
            print(
                f"{result['id']}: tool_db={result['tool_and_database_status']}, "
                f"reply={result['reply_hygiene_status']}, "
                f"end_to_end={result['end_to_end_status']} "
                f"({result['elapsed_seconds']}s)",
                flush=True,
            )
            if result["status"] == "error":
                break

    return (
        0
        if len(report["cases"]) == len(cases)
        and all(case["end_to_end_status"] == "pass" for case in report["cases"])
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
