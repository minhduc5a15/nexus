"""Strictly score free-generated assistant text for the SFT sanity records."""

from __future__ import annotations

from collections import Counter
import json
import re
from typing import Any

from scripts.eval_proposals import exact_equal, schema_errors, strict_json, validate_case


TOOL_CALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
END_TOKEN = "<|im_end|>"


def strip_end_token(text: str) -> str:
    value = text.strip()
    if value.endswith(END_TOKEN):
        value = value[: -len(END_TOKEN)].rstrip()
    return value


def parse_assistant_text(text: object) -> dict[str, Any]:
    result = {"calls": [], "reply": None, "errors": []}
    if not isinstance(text, str):
        result["errors"].append("invalid_text")
        return result
    clean = strip_end_token(text)
    matches = list(TOOL_CALL.finditer(clean))
    has_tag = "<tool_call>" in clean or "</tool_call>" in clean
    if has_tag and not matches:
        result["errors"].append("malformed_tool_call_tag")
        return result
    if not matches:
        result["reply"] = clean
        return result
    if len(matches) != 1:
        result["errors"].append("multiple_calls")
    outside = TOOL_CALL.sub("", clean).strip()
    if outside:
        result["errors"].append("extra_text")
    for match in matches:
        try:
            call = strict_json(match.group(1))
        except (TypeError, ValueError):
            result["errors"].append("invalid_call_json")
            continue
        if not isinstance(call, dict) or set(call) != {"name", "arguments"}:
            result["errors"].append("invalid_call_shape")
            continue
        errors = schema_errors(call["name"], call["arguments"])
        result["errors"].extend(errors)
        result["calls"].append(call)
    return result


def expected_call(case: dict[str, Any]) -> dict[str, Any] | None:
    assistant = case["messages"][-1]
    calls = assistant.get("tool_calls", [])
    if not calls:
        return None
    function = calls[0]["function"]
    return {"name": function["name"], "arguments": function["arguments"]}


def score_case(
    case: dict[str, Any], generated_text: str, *, exact_token_sequence: bool
) -> dict[str, Any]:
    parsed = parse_assistant_text(generated_text)
    gold = expected_call(case)
    errors = list(parsed["errors"])
    if gold is None:
        proposal_correct = not parsed["calls"] and not errors
        if parsed["calls"]:
            errors.append("unexpected_call")
    else:
        proposal_correct = (
            len(parsed["calls"]) == 1
            and exact_equal(parsed["calls"][0], gold)
            and not errors
        )
        if not parsed["calls"]:
            errors.append("missing_call")
        elif not exact_equal(parsed["calls"][0], gold):
            if parsed["calls"][0].get("name") != gold["name"]:
                errors.append("wrong_tool")
            if not exact_equal(parsed["calls"][0].get("arguments"), gold["arguments"]):
                errors.append("arguments_mismatch")
    return {
        "id": case["id"],
        "behavior": case["behavior"],
        "action": case["action"],
        "generated_text": generated_text,
        "parsed": parsed,
        "expected_call": gold,
        "proposal_correct": proposal_correct,
        "exact_token_sequence": exact_token_sequence,
        "error_tags": sorted(set(errors)),
    }


def summarize(cases: list[dict[str, Any]]) -> dict[str, Any]:
    behavior = {}
    for name in ("call", "clarify", "no_action"):
        rows = [case for case in cases if case["behavior"] == name]
        behavior[name] = {
            "cases": len(rows),
            "proposal_correct": sum(case["proposal_correct"] for case in rows),
            "exact_token_sequences": sum(case["exact_token_sequence"] for case in rows),
        }
    return {
        "cases": len(cases),
        "proposal_correct": sum(case["proposal_correct"] for case in cases),
        "exact_token_sequences": sum(case["exact_token_sequence"] for case in cases),
        "behavior": behavior,
        "error_tags": dict(sorted(Counter(
            error for case in cases for error in case["error_tags"]
        ).items())),
    }


def score_proposal_text(case: dict[str, Any], generated_text: str) -> dict[str, Any]:
    """Score HF free-generation text against one proposal-evaluator case."""

    validate_case(case)
    parsed = parse_assistant_text(generated_text)
    expected = case["expected"]
    calls = parsed["calls"]
    errors = list(parsed["errors"])
    accepted = expected["accepted_calls"]
    if expected["behavior"] == "call":
        exact = len(calls) == 1 and any(
            exact_equal(calls[0], candidate) for candidate in accepted
        )
        if not calls:
            errors.append("missing_call")
        elif len(calls) == 1 and not exact:
            expected_names = {candidate["name"] for candidate in accepted}
            if calls[0].get("name") not in expected_names:
                errors.append("wrong_tool")
            expected_arguments = [candidate["arguments"] for candidate in accepted]
            if not any(
                exact_equal(calls[0].get("arguments"), arguments)
                for arguments in expected_arguments
            ):
                errors.append("arguments_mismatch")
    else:
        exact = not calls
        if calls:
            errors.extend(
                [
                    "unexpected_call",
                    "call_during_clarification"
                    if expected["behavior"] == "clarify"
                    else "call_without_authorization",
                ]
            )
    passed = exact and not errors
    return {
        "id": case["id"],
        "family": case["family"],
        "behavior": expected["behavior"],
        "action": expected["action"],
        "prompt": case["prompt"],
        "generated_text": generated_text,
        "parsed": parsed,
        "accepted_calls": accepted,
        "automatic_status": "pass" if passed else "fail",
        "proposal_correct": passed,
        "error_tags": sorted(set(errors)),
        "reply_review": {
            "required": expected["behavior"] != "call",
            "status": "pending"
            if expected["behavior"] != "call"
            else "not_required",
        },
    }


def summarize_proposal_texts(cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize model-only proposal results without policy or side effects."""

    by_behavior: dict[str, dict[str, int]] = {}
    by_action: dict[str, dict[str, int]] = {}
    for case in cases:
        status = case["automatic_status"]
        behavior = by_behavior.setdefault(
            case["behavior"], {"pass": 0, "fail": 0}
        )
        behavior[status] += 1
        if case["behavior"] == "call":
            action = by_action.setdefault(case["action"], {"pass": 0, "fail": 0})
            action[status] += 1
    return {
        "cases": len(cases),
        "pass": sum(case["automatic_status"] == "pass" for case in cases),
        "fail": sum(case["automatic_status"] == "fail" for case in cases),
        "by_behavior": by_behavior,
        "by_action": by_action,
        "unexpected_calls": sum(
            "unexpected_call" in case["error_tags"] for case in cases
        ),
        "reply_review_pending": sum(
            case["reply_review"]["status"] == "pending" for case in cases
        ),
        "error_tags": dict(
            sorted(
                Counter(
                    tag for case in cases for tag in case["error_tags"]
                ).items()
            )
        ),
    }
