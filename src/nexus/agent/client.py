"""Probe a local llama.cpp server with text and one real SQLite tool round."""

import json
import tempfile
import urllib.request
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

from nexus.agent.prompts import SYSTEM_PROMPTS
from nexus.storage.sqlite_db import initialize_database, list_tasks
from nexus.agent.tools import TOOL_DEFINITIONS, execute_tool


import os

ENDPOINT = os.environ.get("QWEN_ENDPOINT", "http://127.0.0.1:8087/v1/chat/completions")


def chat(payload: dict) -> dict:
    request = urllib.request.Request(
        ENDPOINT, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)


def complete_message(response: dict) -> dict:
    choices = response.get("choices", [])
    if not choices or choices[0].get("finish_reason") not in ("stop", "tool_calls"):
        raise RuntimeError("Local model response is incomplete; no tools executed")
    return choices[0]["message"]


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


def run_turn(database_path: Path, prompt: str, generate=chat, *, prompt_version: str = "v1", settings: dict | None = None) -> dict:
    """Run one turn against the local model with at most one tool batch."""
    common = settings or {
        "model": "qwen3-1.7b-q8_0", "temperature": 0.7, "top_p": 0.8,
        "top_k": 20, "min_p": 0, "max_tokens": 256,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    payload = {
        **common,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPTS[prompt_version]},
            {"role": "user", "content": prompt},
        ],
        "tools": [{"type": "function", "function": tool} for tool in TOOL_DEFINITIONS],
        "tool_choice": "auto",
    }
    first = generate(payload)
    message = complete_message(first)
    trace = []
    if message.get("tool_calls"):
        payload["messages"].append(message)
        for call in message["tool_calls"]:
            if call.get("type") != "function":
                raise ValueError("Only function tools are supported")
            function = call["function"]
            arguments = json.loads(function["arguments"])
            result = execute_tool(database_path, function["name"], arguments)
            trace.append({"name": function["name"], "arguments": arguments, "result": result})
            payload["messages"].append({
                "role": "tool", "tool_call_id": call["id"],
                "content": json.dumps(result, ensure_ascii=False),
            })
        payload["tool_choice"] = "none"
        final = generate(payload)
        message = complete_message(final)
        if message.get("tool_calls"):
            raise RuntimeError("Local model requested more tools than allowed")
    reply = message.get("content") or ""
    return {"calls": trace, "reply": reply}


def run_probe(generate=chat) -> dict:
    """All mutations are confined to a disposable database."""
    common = {
        "model": "qwen3-1.7b-q8_0", "temperature": 0.7, "top_p": 0.8,
        "top_k": 20, "min_p": 0, "max_tokens": 256,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    started = perf_counter()
    greeting = generate({**common, "messages": [
        {"role": "user", "content": "Chào bạn! Hãy trả lời bằng một câu tiếng Việt ngắn."}
    ]})
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
        turn = run_turn(path, "Thêm việc mua sữa vào danh sách giúp tôi.", observe, settings=common)
        tasks = [asdict(task) for task in list_tasks(path)]
        return {
            "endpoint": ENDPOINT, "settings": common,
            "greeting": {"reply": greeting_message.get("content"),
                         "elapsed_seconds": round(greeting_elapsed, 3),
                         "usage": greeting.get("usage"), "timings": greeting.get("timings")},
            "tool_round": {"calls": turn["calls"], "reply": turn["reply"],
                           "database_after": tasks,
                           "elapsed_seconds": round(perf_counter() - started, 3),
                           "responses": [{"usage": r.get("usage"), "timings": r.get("timings")} for r in responses]},
            "tool_check_passed": len(turn["calls"]) == 1 and turn["calls"][0]["name"] == "create_task"
                and turn["calls"][0]["arguments"] == {"content": "mua sữa"}
                and len(tasks) == 1 and tasks[0]["content"] == "mua sữa",
        }


if __name__ == "__main__":
    result = run_probe()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["tool_check_passed"] else 1)
