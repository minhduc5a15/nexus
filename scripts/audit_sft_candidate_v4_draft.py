"""Audit the 40-case SFT amendment before human review or training."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
from difflib import SequenceMatcher
import json
from pathlib import Path
from typing import Any

from nexus.core.deadlines import VIETNAM_TIMEZONE, parse_deadline
from nexus.agent.policy import PolicyResult, policy_for_tool
from scripts.audit_proposal_data import shape
from scripts.audit_sft_data import _validate_case, canonical_sha256, sha256
from scripts.audit_sft_pilot import _audit_responses
from scripts.build_sft_candidate_v4_draft import OUTPUT, ROOT, render_document
from scripts.eval_proposals import load_cases, strict_json


SOURCE = ROOT / "evals/sft_candidate_v3.json"
REVIEW = ROOT / "evals/sft_candidate_v4_label_review.json"
COMPARISONS = (
    ROOT / "evals/model_proposal_development_v2.json",
    ROOT / "evals/model_proposal_holdout_v2.json",
    ROOT / "evals/model_proposal_framing_diagnostic_v1.json",
)
REFERENCE = datetime(2026, 10, 21, 12, tzinfo=VIETNAM_TIMEZONE)
NEAR_THRESHOLD = 0.92


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _case_shape(case: dict[str, Any]) -> str:
    prompt = case["messages"][0]["content"]
    calls = [item["function"] for item in case["messages"][1].get("tool_calls", [])]
    return shape(prompt, {"expected": {"accepted_calls": calls}})


def verify_label_review(cases: list[dict[str, Any]]) -> dict[str, Any]:
    review = strict_json(REVIEW.read_bytes())
    _require(review.get("schema_version") == 1 and review.get("dataset_id") == "nexus-sft-candidate-v4-draft", "Review identity changed")
    _require(review.get("dataset_sha256") == sha256(OUTPUT), "Review dataset hash changed")
    _require(review.get("reviewer") == {"name": "Codex", "kind": "assistant"}, "Reviewer provenance changed")
    _require(review.get("scope") == "40 new cases only; source v3 review remains historical", "Review scope changed")
    reviewed = review.get("cases")
    _require(isinstance(reviewed, list) and len(reviewed) == 40, "Review coverage changed")
    for case, item in zip(cases, reviewed):
        _require(item.get("id") == case["id"] and item.get("case_sha256") == canonical_sha256(case), f"Review case hash/order changed: {case['id']}")
        _require(item.get("decision") == "approved_assistant" and isinstance(item.get("note"), str) and item["note"].strip(), f"Review decision missing: {case['id']}")
    _require(review.get("project_owner_review_complete") is False, "Assistant review cannot claim project-owner approval")
    _require(review.get("training_authorized") is False and review.get("final_holdout_v2_inference_authorized") is False, "Review cannot authorize training/holdout")
    return {"sha256": sha256(REVIEW), "assistant_reviewed": len(reviewed), "project_owner_review_complete": False}


def audit_dataset(path: Path = OUTPUT) -> dict[str, Any]:
    document = strict_json(path.read_bytes())
    source = strict_json(SOURCE.read_bytes())
    _require(document.get("schema_version") == 1 and document.get("dataset_id") == "nexus-sft-candidate-v4-draft", "Wrong draft identity")
    _require(document.get("base_model") == source["base_model"] and document.get("contract") == source["contract"], "Model/tool contract changed")
    _require(document.get("source_v3_sha256") == sha256(SOURCE), "v3 source hash changed")
    cases = document.get("cases")
    _require(isinstance(cases, list) and len(cases) == 640, "Expected exactly 640 cases")
    _require(cases[:600] == source["cases"], "Reviewed v3 prefix changed")
    new_cases = cases[600:]
    _require(len(new_cases) == 40, "Expected exactly 40 new cases")
    if path.resolve() == OUTPUT.resolve():
        _require(path.read_text(encoding="utf-8") == render_document(), "Draft differs from builder")

    for case in new_cases:
        validatable = dict(case)
        validatable.pop("target_primary", None)
        _validate_case(validatable)
        if case["action"] == "deadline" and case["behavior"] == "call":
            when = case["messages"][1]["tool_calls"][0]["function"]["arguments"]["when"]
            parse_deadline(when, reference=REFERENCE)
    _require(len({case["id"] for case in cases}) == 640, "Duplicate ID")
    _require(len({case["messages"][0]["content"] for case in cases}) == 640, "Duplicate prompt")
    _require(len({case["family"] for case in new_cases}) == 40, "New families must be distinct")
    targets = Counter(case["target_primary"] for case in new_cases)
    _require(targets == {"deadline_framing_coverage": 10, "list_completed_contract": 10, "boundary_control": 10, "other_action_regression": 10}, "Amendment allocation changed")
    behavior = Counter(case["behavior"] for case in cases)
    _require(behavior == {"call": 450, "clarify": 95, "no_action": 95}, "Behavior balance changed")
    actions = Counter(case["action"] for case in cases if case["behavior"] == "call")
    _require(actions == {"deadline": 70, "list": 70, "create": 62, "complete": 62, "edit": 62, "delete": 62, "due_query": 62}, "Action balance changed")
    responses = _audit_responses(new_cases)

    # Policy is measured separately. It does not define the gold SFT labels.
    policy_compatible = Counter()
    policy_gap = []
    for case in new_cases:
        if case["behavior"] != "call":
            continue
        function = case["messages"][1]["tool_calls"][0]["function"]
        decision = policy_for_tool(
            case["messages"][0]["content"], function["name"], function["arguments"],
            reference_time=REFERENCE,
        )
        if decision.result in (PolicyResult.ALLOW, PolicyResult.REQUIRES_CONFIRMATION):
            policy_compatible[case["action"]] += 1
        else:
            policy_gap.append({"id": case["id"], "action": case["action"], "result": decision.result.value, "reason": decision.reason.value})

    comparison_rows = []
    comparison_hashes = {}
    for other_path in COMPARISONS:
        _, other_cases = load_cases(other_path)
        comparison_rows.extend((other_path.name, case["id"], case["prompt"], shape(case["prompt"], case)) for case in other_cases)
        comparison_hashes[other_path.name] = sha256(other_path)
    exact, masked, near = [], [], []
    for case in new_cases:
        prompt = case["messages"][0]["content"]
        masked_prompt = _case_shape(case)
        if any(prompt.casefold() == old_prompt.casefold() for _, _, old_prompt, _ in comparison_rows):
            exact.append(case["id"])
        if any(masked_prompt == old_shape for _, _, _, old_shape in comparison_rows):
            masked.append(case["id"])
        closest = max(
            (SequenceMatcher(None, masked_prompt, old_shape, autojunk=False).ratio(), source_name, old_id)
            for source_name, old_id, _, old_shape in comparison_rows
        )
        if closest[0] >= NEAR_THRESHOLD:
            near.append({"id": case["id"], "score": round(closest[0], 6), "source": closest[1], "source_id": closest[2]})
    _require(not exact and not masked and not near, f"Evaluation prompt overlap: exact={exact}, masked={masked}, near={near}")
    return {
        "dataset_sha256": sha256(path),
        "source_v3_sha256": sha256(SOURCE),
        "cases": len(cases), "preserved_v3": 600, "added": 40,
        "targets": dict(targets), "behavior": dict(behavior), "call_actions": dict(actions),
        "response_checks": responses,
        "policy_compatibility_diagnostic": {
            "compatible_calls": sum(policy_compatible.values()),
            "incompatible_calls": len(policy_gap),
            "compatible_by_action": dict(policy_compatible),
            "gaps": policy_gap,
            "note": "Policy compatibility is diagnostic, not the gold label or proof of model accuracy.",
        },
        "comparison_hashes": comparison_hashes,
        "evaluation_overlap": {"exact": exact, "masked": masked, "near_at_0_92": near},
        "label_review": verify_label_review(new_cases) if path.resolve() == OUTPUT.resolve() else None,
        "status": "draft_assistant_reviewed_owner_gate_separate_not_authorized_for_training",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=OUTPUT)
    args = parser.parse_args()
    try:
        summary = audit_dataset(args.dataset)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
