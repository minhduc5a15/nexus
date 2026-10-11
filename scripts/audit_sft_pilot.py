"""Audit the 100-case NEXUS SFT pilot without loading or training a model."""

from __future__ import annotations

import argparse
from collections import Counter
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from nexus.agent.prompts import SYSTEM_PROMPTS
from nexus.agent.tools import TOOL_DEFINITIONS
from scripts.audit_proposal_data import shape
from scripts.audit_sft_data import (
    DEVELOPMENT,
    EXPECTED_MODEL,
    EXPECTED_REVISION,
    HOLDOUT,
    NEAR_DUPLICATE_THRESHOLD,
    ROOT,
    SftDataError,
    _assistant_calls,
    _prompt_records,
    _require,
    _validate_case,
    canonical_sha256,
    materialize_case,
    sha256,
)
from scripts.build_sft_pilot_v1 import SANITY_DATASET, render_document
from scripts.eval_proposals import ACTION_TO_TOOL, strict_json


DEFAULT_DATASET = ROOT / "evals/sft_pilot_v1.json"
DEFAULT_MANIFEST = ROOT / "evals/sft_pilot_manifest_v1.json"
DEFAULT_REVIEW = ROOT / "evals/sft_pilot_review_v1.json"
EXPECTED_BEHAVIORS = {"call": 56, "clarify": 22, "no_action": 22}
EXPECTED_ACTIONS = {action: 8 for action in ACTION_TO_TOOL}
IMPLIED_MUTATION = re.compile(
    r"(?:^|[.!?]\s*)(?:tôi\s+|nexus\s+)?(?:đã|sẽ)\s+(?:lưu|thêm|xóa|xoá|sửa|đổi|hoàn thành|hoàn tất|đặt|cập nhật)\b",
    re.IGNORECASE,
)


def _target_shape(case: dict[str, Any]) -> str:
    accepted_calls = [call["function"] for call in _assistant_calls(case["messages"][1])]
    return shape(
        case["messages"][0]["content"],
        {"expected": {"accepted_calls": accepted_calls}},
    )


def _audit_leakage(cases: list[dict[str, Any]]) -> dict[str, Any]:
    comparison = _prompt_records(DEVELOPMENT) + _prompt_records(HOLDOUT)
    exact: list[str] = []
    masked: list[str] = []
    near: list[dict[str, Any]] = []
    nearest_rows: list[dict[str, Any]] = []
    for case in cases:
        prompt = case["messages"][0]["content"]
        target = _target_shape(case)
        ranked = [
            (SequenceMatcher(None, target, item["shape"], autojunk=False).ratio(), item)
            for item in comparison
        ]
        score, nearest = max(ranked, key=lambda pair: pair[0])
        row = {
            "id": case["id"],
            "similarity": round(score, 6),
            "nearest": {key: nearest[key] for key in ("source", "id", "prompt")},
        }
        nearest_rows.append(row)
        if any(prompt.casefold() == item["prompt"].casefold() for item in comparison):
            exact.append(case["id"])
        if any(target == item["shape"] for item in comparison):
            masked.append(case["id"])
        if score >= NEAR_DUPLICATE_THRESHOLD:
            near.append(row)
    return {
        "method": "NFKC/casefold; mask gold content/when and digits; SequenceMatcher screening only",
        "threshold": NEAR_DUPLICATE_THRESHOLD,
        "exact": exact,
        "masked": masked,
        "near": near,
        "maximum_similarity": max(row["similarity"] for row in nearest_rows),
        "nearest": nearest_rows,
    }


def _audit_responses(cases: list[dict[str, Any]]) -> dict[str, Any]:
    clarify = [case for case in cases if case["behavior"] == "clarify"]
    no_action = [case for case in cases if case["behavior"] == "no_action"]
    missing_questions = [
        case["id"]
        for case in clarify
        if "?" not in case["messages"][1]["content"]
    ]
    implied_mutations = [
        case["id"]
        for case in clarify + no_action
        if IMPLIED_MUTATION.search(case["messages"][1]["content"])
    ]
    _require(not missing_questions, f"Clarification target is not a question: {missing_questions}")
    _require(not implied_mutations, f"Non-call reply implies a mutation: {implied_mutations}")
    return {
        "clarification_questions": len(clarify),
        "no_action_replies": len(no_action),
        "implied_mutation_matches": implied_mutations,
    }


def audit_dataset(path: Path = DEFAULT_DATASET) -> dict[str, Any]:
    document = strict_json(path.read_bytes())
    _require(document.get("schema_version") == 1, "Unsupported pilot schema")
    _require(document.get("dataset_id") == "nexus-sft-pilot-v1", "Unexpected pilot dataset ID")
    _require(document.get("base_model") == {"id": EXPECTED_MODEL, "revision": EXPECTED_REVISION}, "Base model lock changed")
    _require(document.get("contract") == {"prompt_version": "v13", "tool_routing": "all", "maximum_tool_calls": 1}, "Training contract changed")
    cases = document.get("cases")
    _require(isinstance(cases, list) and len(cases) == 100, "Pilot dataset must contain 100 cases")
    for case in cases:
        _validate_case(case)

    ids = [case["id"] for case in cases]
    families = [case["family"] for case in cases]
    prompts = [case["messages"][0]["content"] for case in cases]
    _require(len(ids) == len(set(ids)), "Duplicate pilot case ID")
    _require(len(families) == len(set(families)), "Duplicate pilot family")
    _require(len(prompts) == len(set(prompts)), "Duplicate pilot prompt")

    behaviors = Counter(case["behavior"] for case in cases)
    actions = Counter(case["action"] for case in cases if case["behavior"] == "call")
    _require(behaviors == Counter(EXPECTED_BEHAVIORS), f"Unexpected behavior balance: {dict(behaviors)}")
    _require(actions == Counter(EXPECTED_ACTIONS), f"Unexpected action balance: {dict(actions)}")

    sanity = strict_json(SANITY_DATASET.read_bytes())
    _require(cases[: len(sanity["cases"])] == sanity["cases"], "The 16 sanity cases are not preserved verbatim")
    _require(path.read_text(encoding="utf-8") == render_document(), "Tracked pilot is not reproducible from its builder")

    responses = _audit_responses(cases)
    leakage = _audit_leakage(cases)
    _require(not leakage["exact"], f"Exact evaluation prompt leakage: {leakage['exact']}")
    _require(not leakage["masked"], f"Masked evaluation prompt leakage: {leakage['masked']}")
    _require(not leakage["near"], f"Near-duplicate evaluation prompts require rewrite: {[row['id'] for row in leakage['near']]}")

    records = [materialize_case(case, prompt_version="v13") for case in cases]
    return {
        "dataset_id": document["dataset_id"],
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256(path),
        "cases": len(cases),
        "behavior_balance": dict(behaviors),
        "action_balance": dict(actions),
        "sanity_cases_preserved": len(sanity["cases"]),
        "materialized_records": len(records),
        "tools_per_record": len(records[0]["tools"]),
        "response_checks": responses,
        "leakage": {key: leakage[key] for key in ("method", "threshold", "exact", "masked", "near", "maximum_similarity")},
    }


def verify_manifest(path: Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    manifest = strict_json(path.read_bytes())
    _require(manifest.get("schema_version") == 1, "Unsupported pilot manifest schema")
    _require(manifest.get("dataset_id") == "nexus-sft-pilot-v1", "Unexpected pilot manifest ID")
    dataset = manifest.get("dataset", {})
    _require(ROOT / dataset.get("path", "") == DEFAULT_DATASET, "Manifest points to an unexpected pilot dataset")
    _require(dataset.get("count") == 100 and dataset.get("sha256") == sha256(DEFAULT_DATASET), "Pilot dataset lock changed")
    source = manifest.get("source_sanity", {})
    _require(ROOT / source.get("path", "") == SANITY_DATASET, "Unexpected sanity source")
    _require(source.get("count") == 16 and source.get("sha256") == sha256(SANITY_DATASET), "Sanity source lock changed")
    _require(manifest.get("builder", {}).get("path") == "scripts/build_sft_pilot_v1.py", "Unexpected pilot builder")
    _require(manifest["builder"].get("sha256") == sha256(ROOT / manifest["builder"]["path"]), "Pilot builder lock changed")
    _require(manifest.get("base_model") == {"id": EXPECTED_MODEL, "revision": EXPECTED_REVISION}, "Manifest base model changed")
    contract = manifest.get("contract", {})
    _require(contract.get("prompt_version") == "v13", "Manifest prompt version changed")
    _require(contract.get("prompt_sha256") == hashlib.sha256(SYSTEM_PROMPTS["v13"].encode("utf-8")).hexdigest(), "System prompt lock changed")
    _require(contract.get("tool_routing") == "all" and contract.get("maximum_tool_calls") == 1, "Manifest tool contract changed")
    _require(contract.get("tools_sha256") == canonical_sha256(TOOL_DEFINITIONS), "Tool schema lock changed")
    for name, expected_path in (("development", DEVELOPMENT), ("holdout", HOLDOUT)):
        spec = manifest.get("evaluation_inputs", {}).get(name, {})
        _require(ROOT / spec.get("path", "") == expected_path, f"Unexpected {name} path")
        _require(spec.get("sha256") == sha256(expected_path), f"{name} lock changed")
    _require(
        manifest.get("tokenizer_validation") == {
            "transformers": "4.57.1",
            "jinja2": "3.1.6",
            "maximum_sequence_tokens": 2048,
            "observed_maximum_tokens": 1362,
            "assistant_only_mask": "verified_by_exact_token_prefix",
        },
        "Tokenizer validation lock changed",
    )
    _require(manifest.get("training_authorized") is False, "Pilot review does not authorize training")
    return manifest


def verify_review(path: Path = DEFAULT_REVIEW) -> dict[str, Any]:
    review = strict_json(path.read_bytes())
    _require(review.get("schema_version") == 1 and review.get("dataset_id") == "nexus-sft-pilot-v1", "Unexpected pilot review")
    _require(review.get("dataset_sha256") == sha256(DEFAULT_DATASET), "Review belongs to a different pilot dataset")
    cases = strict_json(DEFAULT_DATASET.read_bytes())["cases"]
    _require(review.get("reviewed_case_ids") == [case["id"] for case in cases], "Review does not cover all cases in order")
    checks = review.get("checks", {})
    required = {
        "request_intent_and_behavior",
        "tool_name_and_argument_schema",
        "verbatim_content_and_time_grounding",
        "single_exact_id",
        "clarification_asks_for_missing_information",
        "no_action_reply_does_not_claim_mutation",
    }
    _require(set(checks) == required and all(value is True for value in checks.values()), "Review checklist is incomplete")
    _require(review.get("label_source") == "repository-authored; no model generated or judged the gold labels", "Unclear label provenance")
    _require(review.get("training_authorized") is False, "Review must stop before training")
    return review


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--review", type=Path, default=DEFAULT_REVIEW)
    args = parser.parse_args()
    try:
        summary = audit_dataset(args.dataset.resolve())
        if args.dataset.resolve() == DEFAULT_DATASET:
            verify_manifest(args.manifest.resolve())
            verify_review(args.review.resolve())
            summary["manifest_sha256"] = sha256(args.manifest.resolve())
            summary["review_sha256"] = sha256(args.review.resolve())
    except (OSError, ValueError, KeyError, TypeError, SftDataError) as error:
        parser.error(str(error))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("Offline audit only: no tokenizer, model, GPU, tool, policy or SQLite was used.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
