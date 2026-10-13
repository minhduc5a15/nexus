"""Audit the reviewed 600-case NEXUS SFT candidate v3 dataset."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any

from nexus.agent.prompts import SYSTEM_PROMPTS
from nexus.agent.tools import TOOL_DEFINITIONS
from scripts.audit_sft_data import (
    EXPECTED_MODEL,
    EXPECTED_REVISION,
    ROOT,
    _validate_case,
    canonical_sha256,
    materialize_case,
    sha256,
)
from scripts.audit_sft_pilot import _audit_leakage, _audit_responses
from scripts.build_sft_candidate_v3 import render_document
from scripts.eval_proposals import ACTION_TO_TOOL, strict_json


DEFAULT_DATASET = ROOT / "evals/sft_candidate_v3.json"
DEFAULT_MANIFEST = ROOT / "evals/sft_candidate_manifest_v3.json"
DEFAULT_REVIEW = ROOT / "evals/sft_candidate_review_v3.json"
DESIGN = ROOT / "evals/sft_dataset_v3_design.json"
SOURCE_V2 = ROOT / "evals/sft_candidate_v2.json"
EXPECTED_BEHAVIORS = Counter({"call": 420, "clarify": 90, "no_action": 90})
EXPECTED_ACTIONS = Counter({action: 60 for action in ACTION_TO_TOOL})


class CandidateV3Error(ValueError):
    """The candidate v3 dataset violates its frozen design."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CandidateV3Error(message)


def _load(path: Path) -> dict[str, Any]:
    value = strict_json(path.read_bytes())
    _require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def audit_dataset(path: Path = DEFAULT_DATASET) -> dict[str, Any]:
    document = _load(path)
    _require(document.get("schema_version") == 1, "Unsupported candidate v3 schema")
    _require(document.get("dataset_id") == "nexus-sft-candidate-v3", "Unexpected candidate v3 ID")
    _require(document.get("base_model") == {"id": EXPECTED_MODEL, "revision": EXPECTED_REVISION}, "Base model lock changed")
    _require(document.get("contract") == {"prompt_version": "v13", "tool_routing": "all", "maximum_tool_calls": 1}, "Training contract changed")
    cases = document.get("cases")
    _require(isinstance(cases, list) and len(cases) == 600, "Candidate v3 must contain 600 cases")
    for case in cases:
        validatable = dict(case)
        validatable.pop("target_primary", None)
        _validate_case(validatable)

    source_cases = _load(SOURCE_V2)["cases"]
    _require(cases[:400] == source_cases, "The 400 reviewed v2 cases are not preserved verbatim")
    new_cases = cases[400:]
    _require(all(isinstance(case.get("target_primary"), str) for case in new_cases), "New case is missing target_primary")
    if path.resolve() == DEFAULT_DATASET.resolve():
        _require(path.read_text(encoding="utf-8") == render_document(), "Tracked candidate v3 is not reproducible")

    ids = [case["id"] for case in cases]
    prompts = [case["messages"][0]["content"] for case in cases]
    _require(len(ids) == len(set(ids)), "Duplicate candidate v3 case ID")
    _require(len(prompts) == len(set(prompts)), "Duplicate candidate v3 prompt")

    behaviors = Counter(case["behavior"] for case in cases)
    actions = Counter(case["action"] for case in cases if case["behavior"] == "call")
    _require(behaviors == EXPECTED_BEHAVIORS, f"Unexpected behavior balance: {dict(behaviors)}")
    _require(actions == EXPECTED_ACTIONS, f"Unexpected action balance: {dict(actions)}")

    design = _load(DESIGN)
    expected_targets = Counter({row["id"]: row["v3_new_examples"] for row in design["taxonomy"]})
    targets = Counter(case["target_primary"] for case in new_cases)
    _require(targets == expected_targets, f"Taxonomy allocation mismatch: {dict(targets)}")

    family_members: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        family_members[case["family"]].append(case)
    for family, members in family_members.items():
        signatures = {(case["behavior"], case["action"]) for case in members}
        _require(len(signatures) == 1, f"Family crosses behavior/action labels: {family}")
        _require(len(members) <= 4, f"Family contains too many lexical variants: {family}")

    responses = _audit_responses(new_cases)
    leakage = _audit_leakage(new_cases)
    _require(not leakage["exact"], f"Exact evaluation leakage: {leakage['exact']}")
    _require(not leakage["masked"], f"Masked evaluation leakage: {leakage['masked']}")
    _require(not leakage["near"], f"Near evaluation leakage: {[row['id'] for row in leakage['near']]}")
    records = [materialize_case(case, prompt_version="v13") for case in cases]
    return {
        "dataset_id": document["dataset_id"],
        "path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
        "sha256": sha256(path),
        "cases": len(cases),
        "preserved_v2_cases": 400,
        "new_cases": len(new_cases),
        "behavior_balance": dict(behaviors),
        "action_balance": dict(actions),
        "taxonomy_allocation": dict(targets),
        "families": len(family_members),
        "largest_family": max(map(len, family_members.values())),
        "materialized_records": len(records),
        "tools_per_record": len(records[0]["tools"]),
        "response_checks": responses,
        "leakage": {key: leakage[key] for key in ("method", "threshold", "exact", "masked", "near", "maximum_similarity")},
    }


def verify_manifest(path: Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    manifest = _load(path)
    _require(manifest.get("schema_version") == 1 and manifest.get("dataset_id") == "nexus-sft-candidate-v3", "Unexpected v3 manifest")
    dataset = manifest.get("dataset", {})
    _require(ROOT / dataset.get("path", "") == DEFAULT_DATASET, "Manifest dataset path changed")
    _require(dataset.get("count") == 600 and dataset.get("sha256") == sha256(DEFAULT_DATASET), "Manifest dataset lock changed")
    source = manifest.get("source_candidate_v2", {})
    _require(ROOT / source.get("path", "") == SOURCE_V2 and source.get("sha256") == sha256(SOURCE_V2), "v2 source lock changed")
    _require(source.get("count") == 400 and source.get("preserved_as_prefix") is True, "v2 preservation contract changed")
    design = manifest.get("design", {})
    _require(ROOT / design.get("path", "") == DESIGN and design.get("sha256") == sha256(DESIGN), "Design lock changed")
    builder = manifest.get("builder", {})
    _require(builder.get("path") == "scripts/build_sft_candidate_v3.py", "Unexpected builder path")
    _require(builder.get("sha256") == sha256(ROOT / builder["path"]), "Builder lock changed")
    contract = manifest.get("contract", {})
    _require(contract.get("prompt_version") == "v13", "Prompt version changed")
    _require(contract.get("prompt_sha256") == hashlib.sha256(SYSTEM_PROMPTS["v13"].encode()).hexdigest(), "Prompt hash changed")
    _require(contract.get("tools_sha256") == canonical_sha256(TOOL_DEFINITIONS), "Tool schema hash changed")
    _require(contract.get("tool_routing") == "all" and contract.get("maximum_tool_calls") == 1, "Tool contract changed")
    final_v2 = manifest.get("evaluation_inputs", {}).get("final_holdout_v2", {})
    _require(final_v2.get("path") == "evals/model_proposal_holdout_v2.json", "Final holdout v2 path changed")
    _require(final_v2.get("sha256") == sha256(ROOT / final_v2["path"]) and final_v2.get("count") == 120, "Final holdout v2 lock changed")
    _require(final_v2.get("human_review_path") == "evals/model_proposal_holdout_human_review_v2.json", "Final holdout review path changed")
    _require(final_v2.get("human_review_sha256") == sha256(ROOT / final_v2["human_review_path"]), "Final holdout review lock changed")
    _require(final_v2.get("human_review_complete") is True, "Final holdout label review is incomplete")
    _require(final_v2.get("model_inference_authorized_by_protocol") is False, "Final holdout inference must remain closed")
    _require(manifest.get("training_authorized") is False, "Manifest must not authorize training")
    return manifest


def verify_review(path: Path = DEFAULT_REVIEW) -> dict[str, Any]:
    review = _load(path)
    _require(review.get("schema_version") == 1 and review.get("dataset_id") == "nexus-sft-candidate-v3", "Unexpected v3 review")
    _require(review.get("reviewer") == {"name": "codex", "kind": "assistant"}, "Unexpected v3 reviewer")
    _require(review.get("dataset_sha256") == sha256(DEFAULT_DATASET), "Review dataset hash mismatch")
    _require(review.get("design_sha256") == sha256(DESIGN), "Review design hash mismatch")
    cases = _load(DEFAULT_DATASET)["cases"]
    _require(review.get("reviewed_case_ids") == [case["id"] for case in cases], "Review coverage or order changed")
    scope = review.get("review_scope", {})
    _require(scope == {"inherited_reviewed_v2_cases": 400, "new_v3_cases_reviewed": 200, "total_reviewed": 600}, "Review scope changed")
    checks = review.get("checks", {})
    _require(len(checks) == 10 and all(checks.values()), "Review checklist is incomplete")
    _require(review.get("training_authorized") is False, "Review must not authorize training")
    _require(review.get("final_holdout_v2_inference_authorized") is False, "Review must keep final holdout v2 closed")
    return review


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--review", type=Path, default=DEFAULT_REVIEW)
    args = parser.parse_args()
    try:
        summary = audit_dataset(args.dataset.resolve())
        if args.dataset.resolve() == DEFAULT_DATASET.resolve():
            verify_manifest(args.manifest.resolve())
            verify_review(args.review.resolve())
            summary["manifest_sha256"] = sha256(args.manifest.resolve())
            summary["review_sha256"] = sha256(args.review.resolve())
    except (OSError, ValueError, KeyError, TypeError, CandidateV3Error) as error:
        parser.error(str(error))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("Offline audit only: no tokenizer, model, GPU, tool, policy, database or final-holdout inference was used.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
