"""Evaluate multi-turn AgentSession conversations and their state transitions."""

import argparse
import hashlib
import json
import sqlite3
import tempfile
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter, sleep

import nexus.agent.session as session_module
from nexus.agent.client import ENDPOINT, PostToolExecutionError, response_diagnostics
from nexus.agent.prompts import SYSTEM_PROMPTS
from nexus.agent.session import AgentSession
from nexus.storage.sqlite_db import create_task, initialize_database, list_tasks
from scripts.eval_qwen import send_chat


CASES_PATH = (
    Path(__file__).resolve().parents[1] / "evals" / "session_conversations_v1.json"
)
SCORING_VERSION = 1


def call_signatures(calls: list[dict]) -> list[dict]:
    """Keep the stable part of an executed or proposed tool trace."""
    return [
        {"name": call.get("name"), "arguments": call.get("arguments")}
        for call in calls
    ]


def _task_snapshot(database_path: Path) -> list[dict]:
    return [asdict(task) for task in list_tasks(database_path)]


def _scripted_response(specification: dict, sequence: int) -> dict:
    """Turn a compact dataset fixture into an OpenAI-compatible response."""
    if not isinstance(specification, dict):
        raise ValueError("scripted model response must be an object")
    if "calls" in specification:
        calls = specification["calls"]
        if not isinstance(calls, list):
            raise ValueError("scripted model calls must be a list")
        tool_calls = []
        for index, call in enumerate(calls, 1):
            tool_calls.append({
                "id": f"session-eval-{sequence}-{index}",
                "type": "function",
                "function": {
                    "name": call["name"],
                    "arguments": json.dumps(
                        call["arguments"], ensure_ascii=False, separators=(",", ":")
                    ),
                },
            })
        message = {"role": "assistant", "content": None, "tool_calls": tool_calls}
        finish_reason = "tool_calls"
    elif "reply" in specification:
        message = {"role": "assistant", "content": specification["reply"]}
        finish_reason = "stop"
    else:
        raise ValueError("scripted model response needs calls or reply")
    return {
        "model": "scripted-session-eval",
        "choices": [{"finish_reason": finish_reason, "message": message}],
    }


def _unexpected_tasks(actual: list[str], expected: list[str]) -> list[str]:
    remaining = Counter(expected)
    unexpected = []
    for content in actual:
        if remaining[content]:
            remaining[content] -= 1
        else:
            unexpected.append(content)
    return unexpected


def _matches_expected(actual, expected) -> bool:
    return actual in expected if isinstance(expected, list) else actual == expected


@contextmanager
def _inject_fault(database_path: Path, fault: str | None):
    """Inject only the two lifecycle faults named by the session contract."""
    trigger_created = False
    original_formatter = None
    if fault == "database_before_commit":
        with sqlite3.connect(database_path) as connection:
            connection.execute("""
                CREATE TRIGGER session_eval_fail_insert BEFORE INSERT ON tasks
                WHEN NEW.content = 'lỗi'
                BEGIN SELECT RAISE(ABORT, 'session eval failure'); END
            """)
        trigger_created = True
    elif fault == "formatter_after_commit":
        original_formatter = session_module.format_tool_result

        def fail_formatter(*_args, **_kwargs):
            raise ValueError("session eval formatter failure")

        session_module.format_tool_result = fail_formatter
    elif fault is not None:
        raise ValueError(f"unknown session eval fault: {fault}")

    try:
        yield
    finally:
        if trigger_created:
            with sqlite3.connect(database_path) as connection:
                connection.execute("DROP TRIGGER session_eval_fail_insert")
        if original_formatter is not None:
            session_module.format_tool_result = original_formatter


def _score_turn(
    expected: dict,
    *,
    state_before: str,
    state_after: str,
    status: str,
    model_called: bool,
    executed_calls: list[dict],
    database_before: list[dict],
    database_after: list[dict],
    reply: str | None,
    error: dict | None,
) -> tuple[dict, dict]:
    before_ids = {task["id"] for task in database_before}
    created = [
        task["content"] for task in database_after if task["id"] not in before_ids
    ]
    expected_created = expected.get("created_tasks", [])
    unrequested = _unexpected_tasks(created, expected_created)
    reply_text = reply or ""
    checks = {
        "state_before_match": state_before == expected["state_before"],
        "state_after_match": state_after == expected["state_after"],
        "status_match": _matches_expected(status, expected["status"]),
        "model_called_match": model_called == expected["model_called"],
        "executed_calls_match": call_signatures(executed_calls)
        == expected.get("executed_calls", []),
        "tasks_match": [task["content"] for task in database_after]
        == expected["tasks"],
        "existing_tasks_unchanged": database_after[: len(database_before)]
        == database_before,
        "reply_match": all(
            fragment in reply_text for fragment in expected.get("reply_contains", [])
        )
        and all(
            fragment not in reply_text
            for fragment in expected.get("reply_not_contains", [])
        ),
        "error_stage_match": (error or {}).get("stage")
        == expected.get("error_stage"),
        "no_unrequested_write": not unrequested,
    }
    safety = {
        "expected_created_tasks": expected_created,
        "created_tasks": created,
        "unrequested_write": bool(unrequested),
        "unrequested_tasks_created": unrequested,
    }
    return checks, safety


def evaluate_conversation(
    case: dict,
    generate=None,
    *,
    prompt_version: str = "v1",
    settings: dict | None = None,
) -> dict:
    """Run one isolated multi-turn case with scripted or live model responses."""
    with tempfile.TemporaryDirectory() as directory:
        database_path = Path(directory) / "session-eval.db"
        initialize_database(database_path)
        for content in case.get("initial_tasks", []):
            create_task(database_path, content)

        sessions: dict[str, AgentSession] = {}
        turn_results = []
        for turn_index, turn in enumerate(case["turns"], 1):
            session_id = turn.get("session", "default")
            session = sessions.setdefault(
                session_id,
                AgentSession(
                    database_path,
                    prompt_version=prompt_version,
                    settings=settings,
                ),
            )
            before = _task_snapshot(database_path)
            state_before = session.state.value
            model_called = False
            model_request = None
            model_response = None
            diagnostics = None
            scripted_model = turn.get("model")

            def observe(payload):
                nonlocal model_called, model_request, model_response, diagnostics
                model_called = True
                model_request = payload
                if generate is None:
                    if scripted_model is None:
                        raise RuntimeError("model was called without a scripted response")
                    model_response = _scripted_response(scripted_model, turn_index)
                else:
                    model_response = generate(payload)
                diagnostics = response_diagnostics(model_response)
                return model_response

            result = None
            error = None
            executed_calls = []
            proposed_calls = []
            authorized_calls = []
            rejected_calls = []
            started = perf_counter()
            try:
                with _inject_fault(database_path, turn.get("fault")):
                    result = session.run_turn(turn["prompt"], observe)
            except (RuntimeError, ValueError, sqlite3.Error, OSError) as failure:
                error = {
                    "type": type(failure).__name__,
                    "message": str(failure),
                    "stage": getattr(failure, "stage", None),
                }
                if isinstance(failure, PostToolExecutionError):
                    executed_calls = failure.executed_calls
                    proposed_calls = failure.proposed_calls
                    authorized_calls = failure.authorized_calls
                    rejected_calls = failure.rejected_calls

            if result is not None:
                status = result["status"]
                reply = result["reply"]
                executed_calls = result.get("calls", [])
                proposed_calls = result.get("proposed_calls", [])
                authorized_calls = result.get("authorized_calls", [])
                rejected_calls = result.get("rejected_calls", [])
            else:
                status = "error_after_execution" if executed_calls else "error"
                reply = None

            after = _task_snapshot(database_path)
            checks, safety = _score_turn(
                turn["expected"],
                state_before=state_before,
                state_after=session.state.value,
                status=status,
                model_called=model_called,
                executed_calls=executed_calls,
                database_before=before,
                database_after=after,
                reply=reply,
                error=error,
            )
            turn_results.append({
                "turn": turn_index,
                "session": session_id,
                "prompt": turn["prompt"],
                "expected": turn["expected"],
                "session_state_before": state_before,
                "session_state_after": session.state.value,
                "status": status,
                "model_called": model_called,
                "model_request": model_request,
                "model_response": model_response,
                "response_diagnostics": diagnostics,
                "proposed_calls": proposed_calls,
                "authorized_calls": authorized_calls,
                "rejected_calls": rejected_calls,
                "executed_calls": executed_calls,
                "database_before": before,
                "database_after": after,
                "reply": reply,
                "error": error,
                "checks": checks,
                "safety": safety,
                "result": "pass" if all(checks.values()) else "fail",
                "elapsed_seconds": round(perf_counter() - started, 3),
            })

        final_database = _task_snapshot(database_path)

    expected_final_tasks = case["expected_final_tasks"]
    final_tasks_match = (
        [task["content"] for task in final_database] == expected_final_tasks
    )
    return {
        "id": case["id"],
        "category": case.get("category", "uncategorized"),
        "initial_tasks": case.get("initial_tasks", []),
        "expected_final_tasks": expected_final_tasks,
        "database_after": final_database,
        "turns": turn_results,
        "checks": {"final_tasks_match": final_tasks_match},
        "unrequested_write": any(
            turn["safety"]["unrequested_write"] for turn in turn_results
        ),
        "status": (
            "pass"
            if final_tasks_match
            and all(turn["result"] == "pass" for turn in turn_results)
            else "fail"
        ),
    }


def summarize(results: list[dict]) -> dict:
    turns = [turn for case in results for turn in case["turns"]]
    check_groups = {
        "runtime": ("status_match", "model_called_match", "error_stage_match"),
        "state": ("state_before_match", "state_after_match"),
        "tool": ("executed_calls_match",),
        "database": (
            "tasks_match",
            "existing_tasks_unchanged",
            "no_unrequested_write",
        ),
        "reply": ("reply_match",),
    }
    summary = {
        "cases": {
            "total": len(results),
            "pass": sum(case["status"] == "pass" for case in results),
            "fail": sum(case["status"] == "fail" for case in results),
        },
        "turns": {
            "total": len(turns),
            "pass": sum(turn["result"] == "pass" for turn in turns),
            "fail": sum(turn["result"] == "fail" for turn in turns),
        },
        "checks": {},
        "safety": {
            "unrequested_write_turns": sum(
                turn["safety"]["unrequested_write"] for turn in turns
            ),
            "unrequested_write_case_ids": [
                case["id"] for case in results if case["unrequested_write"]
            ],
        },
        "by_category": {},
    }
    for group, names in check_groups.items():
        passed = sum(all(turn["checks"][name] for name in names) for turn in turns)
        summary["checks"][group] = {
            "pass": passed,
            "fail": len(turns) - passed,
        }
    for case in results:
        counts = summary["by_category"].setdefault(
            case["category"], {"total": 0, "pass": 0, "fail": 0}
        )
        counts["total"] += 1
        counts[case["status"]] += 1
    return summary


def _load_cases(path: Path) -> tuple[bytes, list[dict], dict]:
    source = path.read_bytes()
    document = json.loads(source)
    if isinstance(document, list):
        return source, document, {}
    if not isinstance(document, dict) or not isinstance(document.get("cases"), list):
        raise ValueError("session dataset must be a list or an object containing cases")
    metadata = {key: value for key, value in document.items() if key != "cases"}
    return source, document["cases"], metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", type=Path, default=CASES_PATH)
    parser.add_argument("--case", action="append", dest="case_ids")
    parser.add_argument("--mode", choices=("scripted", "live"), default="scripted")
    parser.add_argument("--endpoint", default=ENDPOINT)
    parser.add_argument("--prompt-version", choices=SYSTEM_PROMPTS, default="v1")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--request-interval", type=float, default=0.0)
    args = parser.parse_args()
    if args.request_interval < 0:
        parser.error("--request-interval phải không âm")

    dataset_bytes, cases, dataset_metadata = _load_cases(args.cases)
    source_case_count = len(cases)
    if args.case_ids:
        unknown = set(args.case_ids) - {case["id"] for case in cases}
        if unknown:
            parser.error(f"Case không tồn tại: {sorted(unknown)}")
        cases = [case for case in cases if case["id"] in args.case_ids]

    settings = {
        "model": "qwen3-1.7b-q8_0",
        "temperature": args.temperature,
        "top_p": 0.8,
        "top_k": 20,
        "min_p": 0,
        "max_tokens": 256,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    last_request_finished = None

    def live_generate(payload):
        nonlocal last_request_finished
        if last_request_finished is not None and args.request_interval:
            sleep(max(0, args.request_interval - (perf_counter() - last_request_finished)))
        try:
            return send_chat(args.endpoint, payload)
        finally:
            last_request_finished = perf_counter()

    report = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "model": settings["model"] if args.mode == "live" else "scripted",
        "endpoint": args.endpoint if args.mode == "live" else None,
        "prompt_version": args.prompt_version,
        "settings": settings if args.mode == "live" else None,
        "scoring_version": SCORING_VERSION,
        "dataset": {
            "path": str(args.cases),
            "sha256": hashlib.sha256(dataset_bytes).hexdigest(),
            "source_case_count": source_case_count,
            "selected_case_ids": [case["id"] for case in cases],
            "metadata": dataset_metadata,
        },
        "cases": [],
        "summary": summarize([]),
    }
    generator = live_generate if args.mode == "live" else None
    with args.output.open("x", encoding="utf-8") as output:
        for case in cases:
            result = evaluate_conversation(
                case,
                generator,
                prompt_version=args.prompt_version,
                settings=settings,
            )
            report["cases"].append(result)
            report["summary"] = summarize(report["cases"])
            output.seek(0)
            json.dump(report, output, ensure_ascii=False, indent=2)
            output.truncate()
            output.flush()
            print(
                f"{result['id']}: {result['status']} "
                f"({sum(turn['result'] == 'pass' for turn in result['turns'])}/"
                f"{len(result['turns'])} turns)"
            )

    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    return 0 if report["summary"]["cases"]["fail"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
