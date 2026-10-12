"""Audit the 400-case NEXUS SFT candidate dataset without loading a model."""

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
    DEVELOPMENT,
    EXPECTED_MODEL,
    EXPECTED_REVISION,
    HOLDOUT,
    ROOT,
    SftDataError,
    _validate_case,
    canonical_sha256,
    materialize_case,
    sha256,
)
from scripts.audit_sft_pilot import _audit_leakage, _audit_responses
from scripts.build_sft_candidate_v2 import PILOT_DATASET, render_document
from scripts.eval_proposals import ACTION_TO_TOOL, strict_json


DEFAULT_DATASET = ROOT / "evals/sft_candidate_v2.json"
DEFAULT_MANIFEST = ROOT / "evals/sft_candidate_manifest_v2.json"
DEFAULT_REVIEW = ROOT / "evals/sft_candidate_review_v2.json"
EXPECTED_BEHAVIORS = {"call": 280, "clarify": 60, "no_action": 60}
EXPECTED_ACTIONS = {action: 40 for action in ACTION_TO_TOOL}


class CandidateDataError(SftDataError):
    """The candidate dataset or its review violates the frozen contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CandidateDataError(message)


def _load(path: Path) -> dict[str, Any]:
    value = strict_json(path.read_bytes())
    _require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def audit_dataset(path: Path = DEFAULT_DATASET) -> dict[str, Any]:
    document = _load(path)
    _require(document.get("schema_version") == 1, "Unsupported candidate schema")
    _require(document.get("dataset_id") == "nexus-sft-candidate-v2", "Unexpected candidate dataset ID")
    _require(
        document.get("base_model") == {"id": EXPECTED_MODEL, "revision": EXPECTED_REVISION},
        "Base model lock changed",
    )
    _require(
        document.get("contract") == {"prompt_version": "v13", "tool_routing": "all", "maximum_tool_calls": 1},
        "Training contract changed",
    )
    cases = document.get("cases")
    _require(isinstance(cases, list) and len(cases) == 400, "Candidate dataset must contain 400 cases")
    for case in cases:
        _validate_case(case)

    ids = [case["id"] for case in cases]
    prompts = [case["messages"][0]["content"] for case in cases]
    _require(len(ids) == len(set(ids)), "Duplicate candidate case ID")
    _require(len(prompts) == len(set(prompts)), "Duplicate candidate prompt")

    pilot = _load(PILOT_DATASET)
    _require(cases[:100] == pilot["cases"], "The frozen 100-case pilot is not preserved verbatim")
    if path.resolve() == DEFAULT_DATASET.resolve():
        _require(path.read_text(encoding="utf-8") == render_document(), "Tracked candidate is not reproducible from its builder")

    behaviors = Counter(case["behavior"] for case in cases)
    actions = Counter(case["action"] for case in cases if case["behavior"] == "call")
    _require(behaviors == Counter(EXPECTED_BEHAVIORS), f"Unexpected behavior balance: {dict(behaviors)}")
    _require(actions == Counter(EXPECTED_ACTIONS), f"Unexpected action balance: {dict(actions)}")

    family_members: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        family_members[case["family"]].append(case)
    for family, members in family_members.items():
        signatures = {(case["behavior"], case["action"]) for case in members}
        _require(len(signatures) == 1, f"Family crosses behavior/action labels: {family}")
        _require(len(members) <= 4, f"Family contains too many lexical variants: {family}")

    responses = _audit_responses(cases)
    leakage = _audit_leakage(cases)
    _require(not leakage["exact"], f"Exact evaluation prompt leakage: {leakage['exact']}")
    _require(not leakage["masked"], f"Masked evaluation prompt leakage: {leakage['masked']}")
    _require(not leakage["near"], f"Near-duplicate evaluation prompts require rewrite: {[row['id'] for row in leakage['near']]}")

    records = [materialize_case(case, prompt_version="v13") for case in cases]
    try:
        display_path = str(path.relative_to(ROOT))
    except ValueError:
        display_path = str(path)
    return {
        "dataset_id": document["dataset_id"],
        "path": display_path,
        "sha256": sha256(path),
        "cases": len(cases),
        "source_pilot_cases": 100,
        "new_cases": 300,
        "behavior_balance": dict(behaviors),
        "action_balance": dict(actions),
        "families": len(family_members),
        "largest_family": max(len(members) for members in family_members.values()),
        "materialized_records": len(records),
        "tools_per_record": len(records[0]["tools"]),
        "response_checks": responses,
        "leakage": {key: leakage[key] for key in ("method", "threshold", "exact", "masked", "near", "maximum_similarity")},
    }


def verify_manifest(path: Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    manifest = _load(path)
    _require(manifest.get("schema_version") == 1, "Unsupported candidate manifest schema")
    _require(manifest.get("dataset_id") == "nexus-sft-candidate-v2", "Unexpected manifest dataset ID")
    dataset = manifest.get("dataset", {})
    _require(ROOT / dataset.get("path", "") == DEFAULT_DATASET, "Manifest points to an unexpected dataset")
    _require(dataset.get("count") == 400 and dataset.get("sha256") == sha256(DEFAULT_DATASET), "Candidate dataset lock changed")
    source = manifest.get("source_pilot", {})
    _require(ROOT / source.get("path", "") == PILOT_DATASET, "Unexpected pilot source")
    _require(source.get("count") == 100 and source.get("sha256") == sha256(PILOT_DATASET), "Pilot source lock changed")
    builder = manifest.get("builder", {})
    _require(builder.get("path") == "scripts/build_sft_candidate_v2.py", "Unexpected candidate builder")
    _require(builder.get("sha256") == sha256(ROOT / builder["path"]), "Candidate builder lock changed")
    _require(
        manifest.get("base_model") == {"id": EXPECTED_MODEL, "revision": EXPECTED_REVISION},
        "Manifest base model changed",
    )
    contract = manifest.get("contract", {})
    _require(contract.get("prompt_version") == "v13", "Manifest prompt version changed")
    _require(
        contract.get("prompt_sha256") == hashlib.sha256(SYSTEM_PROMPTS["v13"].encode("utf-8")).hexdigest(),
        "System prompt lock changed",
    )
    _require(contract.get("tool_routing") == "all" and contract.get("maximum_tool_calls") == 1, "Tool contract changed")
    _require(contract.get("tools_sha256") == canonical_sha256(TOOL_DEFINITIONS), "Tool schema lock changed")
    for name, expected_path in (("development", DEVELOPMENT), ("holdout", HOLDOUT)):
        spec = manifest.get("evaluation_inputs", {}).get(name, {})
        _require(ROOT / spec.get("path", "") == expected_path, f"Unexpected {name} path")
        _require(spec.get("sha256") == sha256(expected_path), f"{name} lock changed")
    _require(
        manifest.get("tokenizer_validation")
        == {
            "tokenizer": EXPECTED_MODEL,
            "revision": EXPECTED_REVISION,
            "transformers": "4.57.1",
            "jinja2": "3.1.6",
            "method": "exact token prefix; prompt labels are -100 and only the final assistant suffix is supervised",
            "maximum_sequence_tokens": 2048,
            "records": 400,
            "minimum_total_tokens": 1296,
            "median_total_tokens": 1316.0,
            "maximum_total_tokens": 1362,
            "maximum_total_case_ids": ["pilot_deadline_day_order", "candidate_deadline_07_03"],
            "prompt_tokens": {"minimum": 1286, "median": 1294.0, "maximum": 1315},
            "supervised_tokens": {"minimum": 9, "median": 23.0, "maximum": 47, "sum": 9834},
            "assistant_only_mask": "verified_by_exact_token_prefix",
        },
        "Tokenizer validation lock changed",
    )
    _require(manifest.get("training_authorized") is False, "Data-review checkpoint must not authorize training")
    _require(manifest.get("final_holdout_inference") is False, "Final holdout must remain closed")
    return manifest


def verify_review(path: Path = DEFAULT_REVIEW) -> dict[str, Any]:
    review = _load(path)
    _require(review.get("schema_version") == 1 and review.get("dataset_id") == "nexus-sft-candidate-v2", "Unexpected candidate review")
    _require(review.get("dataset_sha256") == sha256(DEFAULT_DATASET), "Review belongs to a different dataset")
    cases = _load(DEFAULT_DATASET)["cases"]
    _require(review.get("reviewed_case_ids") == [case["id"] for case in cases], "Review does not cover all cases in order")
    checks = review.get("checks", {})
    required = {
        "request_intent_and_behavior",
        "tool_name_and_argument_schema",
        "verbatim_content_and_time_grounding",
        "single_exact_id",
        "clarification_asks_only_for_missing_information",
        "no_action_reply_does_not_claim_mutation",
        "unsupported_inverse_actions_do_not_map_to_forward_tools",
        "evaluation_family_leakage_screened",
    }
    _require(set(checks) == required and all(checks.values()), "Review checklist is incomplete")
    _require(review.get("training_authorized") is False, "Review must stop before a paid training run")
    _require(review.get("open_final_holdout") is False, "Review must keep final holdout closed")
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
    except (OSError, ValueError, KeyError, TypeError, CandidateDataError) as error:
        parser.error(str(error))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("Offline audit only: no tokenizer, model, GPU, tool, policy, database or final-holdout inference was used.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
