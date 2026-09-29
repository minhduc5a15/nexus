"""Score model proposals only: never authorize or execute a tool or open SQLite."""

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from time import perf_counter

from nexus.agent.client import (
    DEFAULT_MODEL_ID, ENDPOINT, build_model_request, default_model_settings,
)
from nexus.agent.prompts import SYSTEM_PROMPTS
from nexus.agent.routing import ToolRoutingMode
from nexus.agent.tools import TOOL_DEFINITIONS
from scripts.eval_qwen import send_chat

ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "evals/model_proposal_pilot_v1.json"
FIXTURES_PATH = ROOT / "evals/model_proposal_pilot_responses_v1.json"
SCORING_VERSION = "proposal-v1"
ACTION_TO_TOOL = {
    "create": "create_task", "list": "list_tasks",
    "complete": "complete_task", "edit": "update_task", "delete": "delete_task",
    "deadline": "set_task_deadline", "due_query": "list_tasks_by_deadline",
}
SCHEMAS = {tool["name"]: tool["parameters"] for tool in TOOL_DEFINITIONS}
FIELDS = ("id", "content", "when", "scope")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError(f"Non-standard JSON constant: {value}")


def strict_json(text):
    """Decode once, without repairing JSON or silently dropping duplicate keys."""
    return json.loads(text, object_pairs_hook=_unique_object, parse_constant=_invalid_constant)


def schema_errors(name, arguments):
    """Validate the object/integer/string subset used by the seven task schemas."""
    if not isinstance(name, str) or name not in SCHEMAS:
        return ["unknown_tool"]
    if not isinstance(arguments, dict):
        return ["arguments_not_object"]
    schema = SCHEMAS[name]
    properties = schema["properties"]
    errors = []
    for field in schema.get("required", []):
        if field not in arguments:
            errors.append(f"missing_argument:{field}")
    if schema.get("additionalProperties") is False:
        errors.extend(f"extra_argument:{key}" for key in arguments if key not in properties)
    for field, value in arguments.items():
        if field not in properties:
            continue
        definition = properties[field]
        kind = definition["type"]
        if kind == "integer":
            valid = type(value) is int and value >= definition.get("minimum", value)
        elif kind == "string":
            valid = isinstance(value, str) and len(value) >= definition.get("minLength", 0)
        else:
            raise ValueError(f"Unsupported evaluation schema type: {kind}")
        if not valid:
            errors.append(f"invalid_argument:{field}")
        elif "enum" in definition and value not in definition["enum"]:
            errors.append(f"invalid_enum:{field}")
    return errors


def exact_equal(left, right):
    """JSON equality that never treats True or 1.0 as the integer 1."""
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(exact_equal(left[k], right[k]) for k in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(exact_equal(a, b) for a, b in zip(left, right))
    return left == right


def validate_case(case):
    if not isinstance(case, dict) or not isinstance(case.get("expected"), dict):
        raise ValueError("Each case must be an object with an expected object")
    for key in ("id", "prompt", "family", "rationale"):
        if not isinstance(case.get(key), str) or not case[key].strip():
            raise ValueError(f"Case requires a non-empty {key}")
    expected = case.get("expected", {})
    behavior = expected.get("behavior")
    action = expected.get("action")
    calls = expected.get("accepted_calls")
    slots = expected.get("missing_slots")
    if behavior not in ("call", "clarify", "no_action"):
        raise ValueError(f"Invalid expected behavior in {case['id']}")
    if action is not None and action not in ACTION_TO_TOOL:
        raise ValueError(f"Invalid expected action in {case['id']}")
    if not isinstance(calls, list) or not isinstance(slots, list):
        raise ValueError("Expected accepted_calls and missing_slots must be lists")
    if any(not isinstance(slot, str) or not slot.strip() for slot in slots):
        raise ValueError("Missing slots must be non-empty strings")
    if (behavior == "clarify") != bool(slots):
        raise ValueError("Only clarification cases require missing_slots")
    if behavior == "call":
        if action is None or not calls:
            raise ValueError("Call cases require an action and accepted calls")
        for call in calls:
            if not isinstance(call, dict) or set(call) != {"name", "arguments"}:
                raise ValueError("Each accepted call is exactly name + arguments")
            if call["name"] != ACTION_TO_TOOL[action] or schema_errors(call["name"], call["arguments"]):
                raise ValueError(f"Invalid gold call in {case['id']}")
    elif calls:
        raise ValueError("Non-action cases cannot have accepted calls")


def load_cases(path):
    source = Path(path).read_bytes()
    document = strict_json(source)
    if not isinstance(document, dict) or document.get("schema_version") != 1 or not isinstance(document.get("cases"), list):
        raise ValueError("Expected a proposal dataset with schema_version=1 and cases")
    cases = document["cases"]
    if not cases:
        raise ValueError("Dataset cannot be empty")
    seen = set()
    for case in cases:
        validate_case(case)
        if case["id"] in seen:
            raise ValueError(f"Duplicate case ID: {case['id']}")
        seen.add(case["id"])
    return source, cases


def parse_proposal(response):
    """Preserve every call, including malformed calls, rather than salvage a subset."""
    calls, errors = [], []
    parsed = {"calls": calls, "errors": errors, "reply": None, "finish_reason": None}
    choices = response.get("choices") if isinstance(response, dict) else None
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        errors.append("invalid_response")
        return parsed
    choice = choices[0]
    finish = choice.get("finish_reason")
    parsed["finish_reason"] = finish
    if finish not in ("stop", "tool_calls"):
        errors.append("incomplete_response")
    message = choice.get("message")
    if not isinstance(message, dict) or message.get("role") != "assistant":
        errors.append("invalid_message")
        return parsed
    parsed["reply"] = message.get("content")
    if parsed["reply"] is not None and not isinstance(parsed["reply"], str):
        errors.append("invalid_reply_type")
    raw_calls = message.get("tool_calls")
    if raw_calls is None:
        if finish == "tool_calls":
            errors.append("missing_tool_calls")
        return parsed
    if not isinstance(raw_calls, list):
        errors.append("invalid_tool_calls")
        raw_calls = [raw_calls]
    if not raw_calls and finish == "tool_calls":
        errors.append("missing_tool_calls")
    if len(raw_calls) > 1:
        errors.append("multiple_calls")
    for index, raw in enumerate(raw_calls):
        function = raw.get("function") if isinstance(raw, dict) else None
        name = function.get("name") if isinstance(function, dict) else None
        arguments = function.get("arguments") if isinstance(function, dict) else None
        call_errors = []
        if not (
            isinstance(raw, dict) and raw.get("type") == "function"
            and isinstance(raw.get("id"), str) and raw["id"].strip()
            and isinstance(function, dict) and isinstance(name, str) and name
        ):
            call_errors.append("invalid_call_envelope")
        if isinstance(arguments, str):
            try:
                arguments = strict_json(arguments)
            except (ValueError, TypeError):
                call_errors.append("invalid_arguments_json")
        else:
            call_errors.append("invalid_arguments_encoding")
        call_errors.extend(schema_errors(name, arguments))
        calls.append({"name": name, "arguments": arguments, "errors": call_errors, "index": index})
    return parsed


def _text_differences(field, actual, targets, prompt):
    """Observable differences only; these tags are not inferred model intentions."""
    tags = [f"{field}_mismatch"]
    if not isinstance(actual, str):
        return tags
    if actual not in prompt:
        tags.append(f"{field}_not_verbatim")
    if actual and any(actual != target and actual in target for target in targets):
        tags.append(f"{field}_truncated")
    if any(target.rstrip('.?!') == actual and target != actual for target in targets):
        tags.append(f"{field}_punctuation_missing")
    if any(len(actual.splitlines()) != len(target.splitlines()) for target in targets):
        tags.append(f"{field}_line_count_changed")
    if any(
        actual.splitlines() != target.splitlines()
        and Counter(actual.splitlines()) == Counter(target.splitlines()) for target in targets
    ):
        tags.append(f"{field}_lines_reordered")
    return tags


def score_proposal(case, response):
    """Gold labels, never classifier or policy decisions, determine correctness."""
    validate_case(case)
    parsed = parse_proposal(response)
    calls = parsed["calls"]
    expected = case["expected"]
    wants_call = expected["behavior"] == "call"
    gold = expected["accepted_calls"]
    signatures = [{"name": c["name"], "arguments": c["arguments"]} for c in calls]
    errors = list(parsed["errors"])
    call_errors = [error for call in calls for error in call["errors"]]
    errors.extend(call_errors)
    fields = {field: None for field in FIELDS}
    tool_match = None
    if wants_call:
        tool_match = len(calls) == 1 and calls[0]["name"] == ACTION_TO_TOOL[expected["action"]]
        if not calls:
            errors.append("missing_call")
        elif any(call["name"] != ACTION_TO_TOOL[expected["action"]] for call in calls):
            errors.append("wrong_tool")
        for field in FIELDS:
            targets = [call["arguments"][field] for call in gold if field in call["arguments"]]
            if not targets:
                continue
            arguments = calls[0]["arguments"] if tool_match else None
            available = isinstance(arguments, dict) and field in arguments
            actual = arguments[field] if available else None
            fields[field] = bool(available) and any(exact_equal(actual, target) for target in targets)
            if not available:
                errors.append(f"{field}_missing" if isinstance(arguments, dict) else f"{field}_unavailable")
            elif not fields[field]:
                if field in ("content", "when"):
                    errors.extend(_text_differences(field, actual, targets, case["prompt"]))
                else:
                    errors.append(f"{field}_mismatch")
        exact = len(calls) == 1 and any(exact_equal(signatures[0], call) for call in gold)
    else:
        exact = not calls
        if calls:
            errors.append("unexpected_call")
            errors.append("call_during_clarification" if expected["behavior"] == "clarify" else "call_without_authorization")
    if not exact and not errors:
        errors.append("proposal_mismatch")
    passed = exact and not errors
    rubric = ["Không tự xác nhận thao tác đã thành công khi chưa có tool result."]
    if expected["behavior"] == "clarify":
        rubric.extend([
            "Hỏi đúng phần thiếu, không tự chọn giá trị: " + ", ".join(expected["missing_slots"]),
            "Chấp nhận cách diễn đạt tương đương; không chấm exact match câu trả lời.",
        ])
    return {
        "automatic_status": "pass" if passed else "fail",
        "checks": {
            "response_valid": not parsed["errors"],
            "arguments_schema_valid": not call_errors if calls else None,
            "tool_match": tool_match,
            "exact_proposal": passed,
            "fields": fields,
        },
        "error_tags": sorted(set(errors)),
        "proposed_calls": signatures,
        "call_diagnostics": calls,
        "model_reply": parsed["reply"],
        "finish_reason": parsed["finish_reason"],
        "reply_review": {"status": "pending", "rubric": rubric},
        "policy_status": "not_evaluated",
        "system_action_status": "not_evaluated",
    }


def evaluate_case(case, generate, *, prompt_version="v13", settings=None,
                  tool_routing=ToolRoutingMode.ALL):
    """Make one model call. The callback must be a generator, not a tool executor."""
    validate_case(case)
    payload, routing = build_model_request(
        case["prompt"], prompt_version=prompt_version,
        settings=settings if settings is not None else default_model_settings(temperature=0.0),
        tool_routing=tool_routing,
    )
    record = {
        "id": case["id"], "family": case["family"], "prompt": case["prompt"],
        "rationale": case["rationale"], "expected": deepcopy(case["expected"]),
        "known_fields": deepcopy(case.get("known_fields", {})),
        "model_called": True, "model_request": deepcopy(payload), "routing": routing,
        "raw_response": None, "error": None,
    }
    started = perf_counter()
    try:
        response = generate(payload)
    except Exception as error:
        record["generation_seconds"] = round(perf_counter() - started, 6)
        record.update({
            "automatic_status": "error", "checks": None, "error_tags": [],
            "error": {"stage": "generation", "type": type(error).__name__, "message": str(error)},
            "reply_review": {"status": "not_evaluated"},
            "policy_status": "not_evaluated", "system_action_status": "not_evaluated",
        })
    else:
        record["generation_seconds"] = round(perf_counter() - started, 6)
        record["raw_response"] = deepcopy(response)
        record.update(score_proposal(case, response))
        # Availability is diagnostic only. It cannot turn a correct gold proposal into a model error.
        record["outside_route_calls"] = [
            call for call in record["proposed_calls"] if call["name"] not in routing["tools"]
        ]
    record["elapsed_seconds"] = round(perf_counter() - started, 6)
    return record


def summarize(results):
    evaluated = [r for r in results if r["automatic_status"] != "error"]
    counts = Counter(r["automatic_status"] for r in results)
    fields = {}
    for field in FIELDS:
        values = [r["checks"]["fields"][field] for r in evaluated if r["checks"]["fields"][field] is not None]
        fields[field] = {"correct": sum(values), "total": len(values)}
    by_action = {}
    by_behavior = {}
    for result in results:
        behavior = result["expected"]["behavior"]
        group = by_behavior.setdefault(behavior, {"pass": 0, "fail": 0, "error": 0})
        group[result["automatic_status"]] += 1
        if behavior == "call":
            label = result["expected"]["action"]
            group = by_action.setdefault(label, {"pass": 0, "fail": 0, "error": 0})
            group[result["automatic_status"]] += 1
    callable_results = [r for r in evaluated if r["expected"]["behavior"] == "call"]
    noncall_results = [r for r in evaluated if r["expected"]["behavior"] != "call"]
    return {
        "total": len(results),
        "automatic": {status: counts[status] for status in ("pass", "fail", "error")},
        "accuracy": counts["pass"] / len(evaluated) if evaluated else None,
        "accuracy_denominator": len(evaluated),
        "fields": fields, "by_action": by_action, "by_behavior": by_behavior,
        "action_exact": {"correct": sum(r["checks"]["exact_proposal"] for r in callable_results),
                         "total": len(callable_results)},
        "tool_selection": {"correct": sum(r["checks"]["tool_match"] for r in callable_results),
                           "total": len(callable_results)},
        "unexpected_call": {"count": sum("unexpected_call" in r["error_tags"] for r in noncall_results),
                            "total": len(noncall_results)},
        "error_tags": dict(Counter(tag for r in evaluated for tag in r["error_tags"])),
        "reply_review_pending": sum(r["reply_review"]["status"] == "pending" for r in results),
        "policy_status": "not_evaluated", "system_action_status": "not_evaluated",
    }


def _hash_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*args):
    result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DATASET_PATH)
    parser.add_argument("--responses", type=Path, default=FIXTURES_PATH)
    parser.add_argument("--mode", choices=("scripted", "live"), default="scripted")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", action="append", dest="case_ids")
    parser.add_argument("--prompt-version", choices=SYSTEM_PROMPTS, default="v13")
    parser.add_argument("--tool-routing", choices=[m.value for m in ToolRoutingMode], default="all")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--endpoint", default=ENDPOINT)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--model-file", type=Path, help="GGUF whose hash is recorded for a local live run")
    parser.add_argument("--runtime-version", help="Exact llama.cpp build/version for a live run")
    args = parser.parse_args()
    if args.mode == "live" and (not args.model_file or not args.runtime_version):
        parser.error("live mode requires --model-file and --runtime-version for reproducibility")
    try:
        dataset_bytes, cases = load_cases(args.cases)
        fixtures = strict_json(args.responses.read_bytes()) if args.mode == "scripted" else None
        if args.case_ids:
            unknown = set(args.case_ids) - {case["id"] for case in cases}
            if unknown:
                raise ValueError(f"Unknown case IDs: {sorted(unknown)}")
            cases = [case for case in cases if case["id"] in args.case_ids]
        if fixtures is not None:
            if not isinstance(fixtures, dict) or not isinstance(fixtures.get("responses"), dict):
                raise ValueError("Expected fixture responses keyed by case ID")
            for case in cases:
                if case["id"] not in fixtures["responses"]:
                    raise ValueError(f"Missing fixture: {case['id']}")
    except (OSError, ValueError, TypeError) as error:
        parser.error(str(error))
    settings = default_model_settings(model_id=args.model, temperature=args.temperature)
    if args.seed is not None:
        settings["seed"] = args.seed
    report = {
        "scoring_version": SCORING_VERSION,
        "started_at": datetime.now(timezone.utc).isoformat(), "mode": args.mode,
        "scope": "Proposal only; scripted results are evaluator checks, not Qwen performance.",
        "prompt_version": args.prompt_version, "tool_routing": args.tool_routing,
        "settings": settings, "endpoint": args.endpoint if args.mode == "live" else None,
        "dataset": {"path": str(args.cases), "sha256": hashlib.sha256(dataset_bytes).hexdigest(),
                    "selected_case_ids": [case["id"] for case in cases]},
        "fixtures_sha256": _hash_file(args.responses) if fixtures is not None else None,
        "model_sha256": _hash_file(args.model_file) if args.mode == "live" else None,
        "runtime_version": args.runtime_version if args.mode == "live" else "scripted",
        "provenance": {"commit": _git("rev-parse", "HEAD"), "status": _git("status", "--porcelain"),
                       "tracked_diff": _git("diff", "HEAD", "--", "src", "scripts", "tests", "evals", "README.md"),
                       "code_sha256": {}, "untracked_source": {}},
        "cases": [], "summary": summarize([]),
    }
    for folder in (ROOT / "src/nexus", ROOT / "scripts"):
        for path in sorted(folder.rglob("*.py")):
            report["provenance"]["code_sha256"][str(path.relative_to(ROOT))] = _hash_file(path)
    for relative in _git("ls-files", "--others", "--exclude-standard", "--", "src", "scripts").splitlines():
        path = ROOT / relative
        if path.suffix in (".py", ".sh"):
            report["provenance"]["untracked_source"][relative] = path.read_text(encoding="utf-8")
    # Exclusive output prevents accidental overwrite; flush after every response.
    with args.output.open("x", encoding="utf-8") as output:
        for case in cases:
            generate = (
                (lambda payload, case_id=case["id"]: deepcopy(fixtures["responses"][case_id]))
                if fixtures is not None else (lambda payload: send_chat(args.endpoint, payload))
            )
            result = evaluate_case(case, generate, prompt_version=args.prompt_version,
                                   settings=settings, tool_routing=ToolRoutingMode(args.tool_routing))
            report["cases"].append(result)
            report["summary"] = summarize(report["cases"])
            output.seek(0)
            json.dump(report, output, ensure_ascii=False, indent=2, allow_nan=False)
            output.truncate()
            output.flush()
            print(f"{result['id']}: {result['automatic_status']} {result['error_tags']}")
        print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    return 1 if any(r["automatic_status"] != "pass" for r in report["cases"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
