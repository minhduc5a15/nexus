"""Validate NEXUS SFT data without loading a model or executing a tool."""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from nexus.agent.prompts import SYSTEM_PROMPTS
from nexus.agent.tools import TOOL_DEFINITIONS
from scripts.audit_proposal_data import shape
from scripts.eval_proposals import ACTION_TO_TOOL, schema_errors, strict_json


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "evals/sft_sanity_v1.json"
DEFAULT_MANIFEST = ROOT / "evals/sft_sanity_manifest_v1.json"
DEVELOPMENT = ROOT / "evals/model_proposal_development_v1.json"
HOLDOUT = ROOT / "evals/model_proposal_holdout_v1.json"
EXPECTED_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
EXPECTED_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
EXPECTED_BALANCE = {"call": 8, "clarify": 4, "no_action": 4}
NEAR_DUPLICATE_THRESHOLD = 0.92


class SftDataError(ValueError):
    """The sanity dataset is unsafe or inconsistent with the runtime contract."""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def provider_tools() -> list[dict[str, Any]]:
    return [
        {"type": "function", "function": deepcopy(definition)}
        for definition in TOOL_DEFINITIONS
    ]


def materialize_case(case: dict[str, Any], *, prompt_version: str) -> dict[str, Any]:
    """Build the exact conversational record consumed by a tool-calling trainer."""

    return {
        "id": case["id"],
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPTS[prompt_version]},
            *deepcopy(case["messages"]),
        ],
        "tools": provider_tools(),
    }


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SftDataError(message)


def _assistant_calls(message: dict[str, Any]) -> list[dict[str, Any]]:
    calls = message.get("tool_calls", [])
    _require(isinstance(calls, list), "assistant.tool_calls must be a list")
    return calls


def _validate_call(case: dict[str, Any], call: dict[str, Any], prompt: str) -> None:
    _require(isinstance(call, dict) and set(call) == {"type", "function"}, f"Invalid call envelope: {case['id']}")
    _require(call["type"] == "function" and isinstance(call["function"], dict), f"Invalid function call: {case['id']}")
    function = call["function"]
    _require(set(function) == {"name", "arguments"}, f"Call must contain only name and arguments: {case['id']}")
    expected_tool = ACTION_TO_TOOL[case["action"]]
    _require(function["name"] == expected_tool, f"Wrong tool for action in {case['id']}")
    _require(not schema_errors(function["name"], function["arguments"]), f"Invalid arguments in {case['id']}")
    arguments = function["arguments"]
    for field in ("content", "when"):
        if field in arguments:
            _require(arguments[field] in prompt, f"{field} is not verbatim in {case['id']}")
    if "id" in arguments:
        ids = {int(value) for value in re.findall(r"(?<!\d)\d+(?!\d)", prompt)}
        _require(arguments["id"] in ids, f"ID is not grounded in {case['id']}")
    if "scope" in arguments:
        evidence = {
            "today": ("hôm nay",),
            "tomorrow": ("ngày mai", "mai"),
            "overdue": ("quá hạn", "trễ hạn"),
        }
        _require(any(term in prompt.casefold() for term in evidence[arguments["scope"]]), f"Scope is not grounded in {case['id']}")


def _validate_case(case: object) -> None:
    _require(isinstance(case, dict), "Each SFT case must be an object")
    required = {"id", "family", "behavior", "action", "rationale", "messages"}
    _require(set(case) == required, f"Unexpected case fields: {case.get('id')}")
    for field in ("id", "family", "rationale"):
        _require(isinstance(case[field], str) and bool(case[field].strip()), f"Missing {field}")
    behavior = case["behavior"]
    _require(behavior in EXPECTED_BALANCE, f"Unknown behavior in {case['id']}")
    action = case["action"]
    if behavior == "no_action":
        _require(action is None, f"no_action must not name an action: {case['id']}")
    else:
        _require(action in ACTION_TO_TOOL, f"Invalid action in {case['id']}")
    messages = case["messages"]
    _require(isinstance(messages, list) and len(messages) == 2, f"Each case needs user + assistant: {case['id']}")
    user, assistant = messages
    _require(isinstance(user, dict) and set(user) == {"role", "content"}, f"Invalid user message: {case['id']}")
    _require(user["role"] == "user" and isinstance(user["content"], str) and user["content"].strip(), f"Invalid user content: {case['id']}")
    _require(isinstance(assistant, dict) and assistant.get("role") == "assistant", f"Invalid assistant message: {case['id']}")
    _require(isinstance(assistant.get("content"), str), f"Assistant content must be a string: {case['id']}")
    _require(set(assistant).issubset({"role", "content", "tool_calls"}), f"Unexpected assistant fields: {case['id']}")
    calls = _assistant_calls(assistant)
    _require(len(calls) <= 1, f"More than one tool call in {case['id']}")
    if behavior == "call":
        _require(len(calls) == 1 and assistant["content"] == "", f"Call target must contain one call only: {case['id']}")
        _validate_call(case, calls[0], user["content"])
    else:
        _require(not calls, f"Non-call target invokes a tool: {case['id']}")
        _require(bool(assistant["content"].strip()), f"Non-call target needs a reply: {case['id']}")


def _prompt_records(path: Path) -> list[dict[str, Any]]:
    document = strict_json(path.read_bytes())
    return [
        {
            "source": str(path.relative_to(ROOT)),
            "id": case["id"],
            "prompt": case["prompt"],
            "shape": shape(case["prompt"], case),
        }
        for case in document["cases"]
    ]


def audit_dataset(path: Path = DEFAULT_DATASET) -> dict[str, Any]:
    document = strict_json(path.read_bytes())
    _require(document.get("schema_version") == 1, "Unsupported SFT dataset schema")
    _require(document.get("dataset_id") == "nexus-sft-sanity-v1", "Unexpected dataset ID")
    model = document.get("base_model", {})
    _require(model == {"id": EXPECTED_MODEL, "revision": EXPECTED_REVISION}, "Base model lock changed")
    contract = document.get("contract", {})
    _require(contract == {"prompt_version": "v13", "tool_routing": "all", "maximum_tool_calls": 1}, "Training contract changed")
    cases = document.get("cases")
    _require(isinstance(cases, list) and len(cases) == 16, "Sanity dataset must contain 16 cases")
    for case in cases:
        _validate_case(case)
    ids = [case["id"] for case in cases]
    families = [case["family"] for case in cases]
    prompts = [case["messages"][0]["content"] for case in cases]
    _require(len(set(ids)) == len(ids), "Duplicate SFT case ID")
    _require(len(set(families)) == len(families), "Duplicate SFT family")
    _require(len(set(prompts)) == len(prompts), "Duplicate SFT prompt")
    balance = Counter(case["behavior"] for case in cases)
    _require(balance == Counter(EXPECTED_BALANCE), f"Unexpected behavior balance: {dict(balance)}")
    actions = Counter(case["action"] for case in cases if case["behavior"] == "call")
    _require(set(actions) == set(ACTION_TO_TOOL), "Sanity calls must cover all seven actions")
    _require(actions["create"] == 2 and all(count == 1 for action, count in actions.items() if action != "create"), "Unexpected action balance")

    comparison = _prompt_records(DEVELOPMENT) + _prompt_records(HOLDOUT)
    exact = []
    masked = []
    near = []
    from difflib import SequenceMatcher

    for case in cases:
        prompt = case["messages"][0]["content"]
        target = shape(prompt, {"expected": {"accepted_calls": [
            call["function"] for call in _assistant_calls(case["messages"][1])
        ]}})
        candidates = [
            (
                SequenceMatcher(None, target, item["shape"], autojunk=False).ratio(),
                item,
            )
            for item in comparison
        ]
        score, nearest = max(candidates, key=lambda pair: pair[0])
        if any(prompt.casefold() == item["prompt"].casefold() for item in comparison):
            exact.append(case["id"])
        if any(target == item["shape"] for item in comparison):
            masked.append(case["id"])
        if score >= NEAR_DUPLICATE_THRESHOLD:
            near.append({
                "id": case["id"],
                "similarity": round(score, 6),
                "nearest": {key: nearest[key] for key in ("source", "id", "prompt")},
            })
    _require(not exact, f"Exact evaluation prompt leakage: {exact}")
    _require(not masked, f"Masked evaluation prompt leakage: {masked}")
    _require(not near, f"Near-duplicate evaluation prompts require rewrite: {[item['id'] for item in near]}")

    prompt_version = contract["prompt_version"]
    records = [materialize_case(case, prompt_version=prompt_version) for case in cases]
    try:
        display_path = str(path.relative_to(ROOT))
    except ValueError:
        display_path = str(path)
    return {
        "dataset_id": document["dataset_id"],
        "path": display_path,
        "sha256": sha256(path),
        "cases": len(cases),
        "behavior_balance": dict(balance),
        "action_balance": dict(actions),
        "base_model": model,
        "prompt_version": prompt_version,
        "tools_per_record": len(provider_tools()),
        "materialized_records": len(records),
        "development_sha256": sha256(DEVELOPMENT),
        "holdout_sha256": sha256(HOLDOUT),
        "leakage": {"exact": exact, "masked": masked, "near": near},
    }


def verify_manifest(path: Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    manifest = strict_json(path.read_bytes())
    _require(manifest.get("schema_version") == 1, "Unsupported SFT manifest schema")
    _require(manifest.get("dataset_id") == "nexus-sft-sanity-v1", "Unexpected manifest dataset ID")
    dataset = manifest.get("dataset", {})
    dataset_path = ROOT / dataset.get("path", "")
    _require(dataset_path == DEFAULT_DATASET, "Manifest points to an unexpected dataset")
    _require(dataset.get("count") == 16, "Manifest case count changed")
    _require(dataset.get("sha256") == sha256(dataset_path), "SFT dataset hash mismatch")
    _require(
        manifest.get("base_model") == {"id": EXPECTED_MODEL, "revision": EXPECTED_REVISION},
        "Manifest base model changed",
    )
    contract = manifest.get("contract", {})
    _require(contract.get("prompt_version") == "v13", "Manifest prompt version changed")
    _require(
        contract.get("prompt_sha256") == hashlib.sha256(SYSTEM_PROMPTS["v13"].encode("utf-8")).hexdigest(),
        "System prompt hash mismatch",
    )
    _require(contract.get("tool_routing") == "all" and contract.get("maximum_tool_calls") == 1, "Manifest tool contract changed")
    _require(contract.get("tools_sha256") == canonical_sha256(TOOL_DEFINITIONS), "Tool schema hash mismatch")
    for name, expected_path in (("development", DEVELOPMENT), ("holdout", HOLDOUT)):
        spec = manifest.get("evaluation_inputs", {}).get(name, {})
        _require(ROOT / spec.get("path", "") == expected_path, f"Unexpected {name} path")
        _require(spec.get("sha256") == sha256(expected_path), f"{name} hash mismatch")
    _require(
        manifest.get("loss_contract") == {
            "method": "tokenize prompt with generation prefix, tokenize full conversation, require exact prefix, mask prompt labels with -100",
            "assistant_only": True,
            "minimum_supervised_tokens": 1,
        },
        "Assistant-only loss contract changed",
    )
    _require(
        manifest.get("tokenizer_validation") == {
            "transformers": "4.57.1",
            "jinja2": "3.1.6",
            "maximum_sequence_tokens": 2048,
        },
        "Tokenizer validation environment changed",
    )
    scope = manifest.get("scope", {})
    _require(scope and not any(scope.values()), "Sanity-data stage must not run inference, training or tools")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    try:
        summary = audit_dataset(args.dataset.resolve())
        if args.dataset.resolve() == DEFAULT_DATASET and args.manifest.resolve() == DEFAULT_MANIFEST:
            verify_manifest(args.manifest.resolve())
            summary["manifest_sha256"] = sha256(args.manifest.resolve())
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("Offline audit only: no tokenizer, model, GPU, tool, policy or SQLite was used.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
