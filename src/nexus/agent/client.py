"""Probe a local llama.cpp server with text and one real SQLite tool round."""

import json
import os
import tempfile
import urllib.request
from copy import deepcopy
from dataclasses import asdict
from enum import Enum
from pathlib import Path
from time import perf_counter

from nexus.agent.prompts import DEFAULT_PROMPT_VERSION, FEW_SHOT_MESSAGES, SYSTEM_PROMPTS
from nexus.storage.sqlite_db import get_task, initialize_database, list_tasks
from nexus.agent.tools import TOOL_DEFINITIONS, execute_tool
from nexus.agent.policy import (
    PolicyResult,
    PolicyReason,
    edit_request_fields,
    policy_for_tool,
)
from nexus.agent.responses import format_rejection, format_tool_result

ENDPOINT = os.environ.get("QWEN_ENDPOINT", "http://127.0.0.1:8087/v1/chat/completions")


class TurnStatus(str, Enum):
    NO_TOOL = "no_tool"
    REJECTED = "rejected"
    NEEDS_CLARIFICATION = "needs_clarification"
    EXECUTED = "executed"
    CANCELLED = "cancelled"
    NEEDS_CONFIRMATION = "needs_confirmation"
    NOT_FOUND = "not_found"


class PostToolExecutionError(RuntimeError):
    """The model turn failed after one or more tools had already completed."""

    def __init__(
        self,
        stage: str,
        executed_calls: list[dict],
        cause: Exception,
        *,
        proposed_calls: list[dict] | None = None,
        authorized_calls: list[dict] | None = None,
        rejected_calls: list[dict] | None = None,
    ):
        super().__init__(
            f"Agent failed during {stage} after {len(executed_calls)} tool call(s): {cause}"
        )
        self.stage = stage
        self.executed_calls = executed_calls
        self.proposed_calls = proposed_calls or []
        self.authorized_calls = authorized_calls or []
        self.rejected_calls = rejected_calls or []
        self.cause = cause


def chat(payload: dict) -> dict:
    request = urllib.request.Request(
        ENDPOINT,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)


def complete_message(response: dict) -> dict:
    choices = response.get("choices", [])
    if not choices or choices[0].get("finish_reason") not in ("stop", "tool_calls"):
        raise RuntimeError("Local model response is incomplete")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise RuntimeError("Local model response has no valid message")
    return message


def response_diagnostics(response: dict) -> dict:
    """Extract execution metrics and token usage from local server response."""
    choices = response.get("choices", [])
    choice = choices[0] if choices else {}
    return {
        "finish_reason": choice.get("finish_reason"),
        "model": response.get("model"),
        "usage": response.get("usage"),
        "timings": response.get("timings"),
    }


def build_messages(prompt_version: str, prompt: str) -> list[dict]:
    """Build one request without sharing mutable few-shot messages."""
    return [
        {"role": "system", "content": SYSTEM_PROMPTS[prompt_version]},
        *deepcopy(FEW_SHOT_MESSAGES.get(prompt_version, [])),
        {"role": "user", "content": prompt},
    ]


def run_turn(
    database_path: Path,
    prompt: str,
    generate=chat,
    *,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
    settings: dict | None = None,
) -> dict:
    """Run one turn against the local model with at most one tool batch."""
    common = settings or {
        "model": "qwen3-1.7b-q8_0",
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "min_p": 0,
        "max_tokens": 256,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    payload = {
        **common,
        "messages": build_messages(prompt_version, prompt),
        "tools": [{"type": "function", "function": tool} for tool in TOOL_DEFINITIONS],
        "tool_choice": "auto",
    }
    first = generate(payload)
    message = complete_message(first)

    calls = []
    proposed_calls = []
    authorized_calls = []
    rejected_calls = []
    confirmation = None

    def turn_result(status: TurnStatus, reply: str) -> dict:
        model_reply = message.get("content")
        result = {
            "calls": calls,
            "proposed_calls": proposed_calls,
            "authorized_calls": authorized_calls,
            "rejected_calls": rejected_calls,
            "status": status.value,
            "model_reply": model_reply if isinstance(model_reply, str) else None,
            "reply": reply,
        }
        if confirmation is not None:
            result["confirmation"] = confirmation
        return result

    def no_call_result() -> dict:
        if first["choices"][0]["finish_reason"] == "tool_calls":
            rejected_calls.append(
                {
                    "name": None,
                    "arguments": None,
                    "result": PolicyResult.REJECT.value,
                    "reason": PolicyReason.INVALID_ARGUMENTS.value,
                }
            )
            return turn_result(TurnStatus.REJECTED, "Không có thao tác nào được thực hiện.")
        return turn_result(TurnStatus.NO_TOOL, "Không có thao tác nào được thực hiện.")

    raw_tool_calls = message.get("tool_calls")

    if raw_tool_calls is None:
        # No tool result exists, so a direct model reply cannot certify a write
        # or the contents of the saved list. Keep it in the trace only.
        return no_call_result()

    if not isinstance(raw_tool_calls, list):
        if isinstance(raw_tool_calls, dict):
            function = (
                raw_tool_calls.get("function")
                if isinstance(raw_tool_calls.get("function"), dict)
                else {}
            )
            name = function.get("name") or raw_tool_calls.get("name")
            raw_args = (
                function.get("arguments")
                if "arguments" in function
                else raw_tool_calls.get("arguments")
            )
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            except Exception:
                args = raw_args
        else:
            name = None
            args = raw_tool_calls

        proposed_calls.append({"name": name, "arguments": args})
        rejected_calls.append(
            {
                "name": name,
                "arguments": args,
                "result": PolicyResult.REJECT.value,
                "reason": PolicyReason.INVALID_ARGUMENTS.value,
            }
        )
        return turn_result(TurnStatus.REJECTED, "Không có thao tác nào được thực hiện.")

    tool_calls = raw_tool_calls

    if len(tool_calls) == 0:
        return no_call_result()

    if len(tool_calls) > 1:
        for call in tool_calls:
            function = call.get("function") if isinstance(call, dict) else None
            name = function.get("name") if isinstance(function, dict) else None
            raw_args = function.get("arguments") if isinstance(function, dict) else None
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            except Exception:
                args = raw_args
            proposed_calls.append({"name": name, "arguments": args})
            rejected_calls.append(
                {
                    "name": name,
                    "arguments": args,
                    "result": PolicyResult.REJECT.value,
                    "reason": PolicyReason.INVALID_ARGUMENTS.value,
                }
            )
        return turn_result(TurnStatus.REJECTED, "Không có thao tác nào được thực hiện.")

    call = tool_calls[0]
    call_id = call.get("id") if isinstance(call, dict) else None
    call_type = call.get("type") if isinstance(call, dict) else None
    function = call.get("function") if isinstance(call, dict) else None

    is_valid_envelope = (
        isinstance(call, dict)
        and isinstance(call_id, str)
        and bool(call_id.strip())
        and call_type == "function"
        and isinstance(function, dict)
        and "name" in function
        and "arguments" in function
        and isinstance(function["arguments"], str)
    )

    parsed_args = None
    if is_valid_envelope:
        try:
            parsed_args = json.loads(function["arguments"])
        except (ValueError, TypeError):
            is_valid_envelope = False

    if not is_valid_envelope:
        name = function.get("name") if isinstance(function, dict) else None
        raw_args = function.get("arguments") if isinstance(function, dict) else None
        proposed_calls.append({"name": name, "arguments": raw_args})
        rejected_calls.append(
            {
                "name": name,
                "arguments": raw_args,
                "result": PolicyResult.REJECT.value,
                "reason": PolicyReason.INVALID_ARGUMENTS.value,
            }
        )
        return turn_result(TurnStatus.REJECTED, "Không có thao tác nào được thực hiện.")

    name = function["name"]
    arguments = parsed_args
    proposed_calls.append({"name": name, "arguments": arguments})

    decision = policy_for_tool(prompt, name, arguments)

    if decision.result == PolicyResult.REQUIRES_CONFIRMATION:
        task = get_task(database_path, arguments["id"])
        authorized_calls.append(
            {
                "name": name,
                "arguments": arguments,
                "result": decision.result.value,
                "reason": decision.reason.value,
            }
        )
        if task is None:
            return turn_result(
                TurnStatus.NOT_FOUND, "Không tìm thấy việc có ID đã yêu cầu."
            )
        confirmation = {
            "name": name,
            "arguments": arguments,
            "task": asdict(task),
        }
        return turn_result(
            TurnStatus.NEEDS_CONFIRMATION,
            f"Bạn có chắc muốn xóa [{task.id}] {task.content}?",
        )

    if decision.result == PolicyResult.ALLOW:
        authorized_calls.append(
            {
                "name": name,
                "arguments": arguments,
                "result": decision.result.value,
                "reason": decision.reason.value,
            }
        )
        try:
            result = execute_tool(database_path, name, arguments)
            calls.append({"name": name, "arguments": arguments, "result": result})
        except Exception as error:
            if calls:
                raise PostToolExecutionError("tool_execution", calls, error) from error
            raise

        try:
            reply = format_tool_result(name, result)
        except Exception as error:
            raise PostToolExecutionError(
                "response_formatting",
                calls,
                error,
                proposed_calls=proposed_calls,
                authorized_calls=authorized_calls,
                rejected_calls=rejected_calls,
            ) from error

        return turn_result(TurnStatus.EXECUTED, reply)

    if decision.result == PolicyResult.NEEDS_CLARIFICATION:
        rejected_calls.append(
            {
                "name": name,
                "arguments": arguments,
                "result": decision.result.value,
                "reason": decision.reason.value,
            }
        )
        if name == "delete_task" and decision.reason in (
            PolicyReason.MISSING_DELETE_ID,
            PolicyReason.MULTIPLE_TASK_IDS,
        ):
            reply = "Bạn muốn xóa việc có ID nào?"
        elif name == "update_task" and decision.reason in (
            PolicyReason.MISSING_UPDATE_ID,
            PolicyReason.MULTIPLE_TASK_IDS,
        ):
            reply = "Bạn muốn sửa việc có ID nào?"
        elif name == "update_task" and decision.reason == PolicyReason.MISSING_UPDATE_CONTENT:
            task_id, _content = edit_request_fields(prompt)
            reply = f"Bạn muốn đổi nội dung việc {task_id} thành gì?"
        elif decision.reason in (
            PolicyReason.MISSING_TASK_ID,
            PolicyReason.MULTIPLE_TASK_IDS,
        ):
            reply = "Bạn muốn hoàn thành việc có ID nào?"
        else:
            reply = "Bạn muốn thêm việc gì?"
        return turn_result(TurnStatus.NEEDS_CLARIFICATION, reply)

    rejected_calls.append(
        {
            "name": name,
            "arguments": arguments,
            "result": decision.result.value,
            "reason": decision.reason.value,
        }
    )
    return turn_result(TurnStatus.REJECTED, format_rejection(decision.reason))


def run_probe(generate=chat) -> dict:
    """All mutations are confined to a disposable database."""
    common = {
        "model": "qwen3-1.7b-q8_0",
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "min_p": 0,
        "max_tokens": 256,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    started = perf_counter()
    greeting = generate(
        {
            **common,
            "messages": [
                {
                    "role": "user",
                    "content": "Chào bạn! Hãy trả lời bằng một câu tiếng Việt ngắn.",
                }
            ],
        }
    )
    greeting_message = complete_message(greeting)
    greeting_elapsed = perf_counter() - started
    if greeting_message.get("tool_calls"):
        raise RuntimeError("Unexpected tool call in the text-only probe")

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "probe.db"
        initialize_database(path)
        responses = []

        def observe(payload):
            resp = generate(payload)
            responses.append(resp)
            return resp

        started = perf_counter()
        turn = run_turn(path, "Thêm việc: mua sữa", observe, settings=common)
        tasks = [asdict(task) for task in list_tasks(path)]
        return {
            "endpoint": ENDPOINT,
            "settings": common,
            "greeting": {
                "reply": greeting_message.get("content"),
                "elapsed_seconds": round(greeting_elapsed, 3),
                "usage": greeting.get("usage"),
                "timings": greeting.get("timings"),
            },
            "tool_round": {
                "calls": turn["calls"],
                "reply": turn["reply"],
                "database_after": tasks,
                "elapsed_seconds": round(perf_counter() - started, 3),
                "responses": [
                    {"usage": r.get("usage"), "timings": r.get("timings")}
                    for r in responses
                ],
            },
            "tool_check_passed": len(turn["calls"]) == 1
            and turn["calls"][0]["name"] == "create_task"
            and turn["calls"][0]["arguments"] == {"content": "mua sữa"}
            and len(tasks) == 1
            and tasks[0]["content"] == "mua sữa",
        }


if __name__ == "__main__":
    result = run_probe()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["tool_check_passed"] else 1)
