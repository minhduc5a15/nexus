"""Audit the fresh proposal development set and the observed SFT v3 deadline regression."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path

from nexus.core.deadlines import VIETNAM_TIMEZONE, parse_deadline
from scripts.audit_proposal_data import shape, validate_split
from scripts.build_model_proposal_development_v2 import OUTPUT, render_document
from scripts.eval_proposals import load_cases, strict_json


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "evals/sft_candidate_v3.json"
OLD_DEV = ROOT / "evals/model_proposal_development_v1.json"
SEALED_HOLDOUT = ROOT / "evals/model_proposal_holdout_v2.json"
RAW_RESULT = ROOT / "evals/results/sft-candidate-v3-development-20261016d/sft-candidate-v3/development.json"
TRACKED_REVIEW = ROOT / "evals/sft_candidate_development_review_v3.json"
MANIFEST = ROOT / "evals/model_proposal_development_v2_manifest.json"
SIMILARITY_THRESHOLD = 0.85
REFERENCE = datetime(2026, 10, 20, 12, tzinfo=VIETNAM_TIMEZONE)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _train_shape(case: dict) -> str:
    prompt = case["messages"][0]["content"]
    calls = [item["function"] for item in case["messages"][1].get("tool_calls", [])]
    return shape(prompt, {"expected": {"accepted_calls": calls}})


def _comparison_records() -> list[tuple[str, str, str, str]]:
    records = []
    for path in (OLD_DEV, SEALED_HOLDOUT):
        _, cases = load_cases(path)
        records.extend((path.name, case["id"], case["prompt"], shape(case["prompt"], case)) for case in cases)
    for case in strict_json(TRAIN.read_bytes())["cases"]:
        prompt = case["messages"][0]["content"]
        records.append((TRAIN.name, case["id"], prompt, _train_shape(case)))
    return records


def audit_dataset(path: Path = OUTPUT) -> dict:
    source, cases = load_cases(path)
    if path.resolve() == OUTPUT.resolve() and source.decode("utf-8") != render_document():
        raise ValueError("Development v2 differs from its builder")
    if len({case["prompt"] for case in cases}) != len(cases):
        raise ValueError("Duplicate development v2 prompt")
    action_balance = validate_split(cases, 80, 8, 24)
    behaviors = Counter(case["expected"]["behavior"] for case in cases)
    if behaviors != Counter({"call": 56, "clarify": 12, "no_action": 12}):
        raise ValueError(f"Unexpected behavior balance: {dict(behaviors)}")
    for case in cases:
        if case.get("origin") != "authored_development_v2":
            raise ValueError(f"Unexpected case origin: {case['id']}")
        calls = case["expected"]["accepted_calls"]
        if len(calls) > 1:
            raise ValueError(f"More than one gold proposal: {case['id']}")
        if calls:
            arguments = calls[0]["arguments"]
            for field in ("content", "when"):
                if field in arguments and arguments[field] not in case["prompt"]:
                    raise ValueError(f"Nonverbatim {field}: {case['id']}")
            if "when" in arguments:
                parse_deadline(arguments["when"], reference=REFERENCE)

    comparisons = _comparison_records()
    exact, near = [], []
    for case in cases:
        target = shape(case["prompt"], case)
        score, closest = max(
            ((SequenceMatcher(None, target, old_shape, autojunk=False).ratio(), (source, old_id, prompt))
             for source, old_id, prompt, old_shape in comparisons),
            key=lambda item: item[0],
        )
        if score == 1 or any(case["prompt"].casefold() == old_prompt.casefold() for _, _, old_prompt, _ in comparisons):
            exact.append(case["id"])
        if score >= SIMILARITY_THRESHOLD:
            near.append({"id": case["id"], "score": round(score, 6), "source": closest[0], "source_id": closest[1]})
    if exact or near:
        raise ValueError(f"Development v2 overlaps old evaluation/training families: exact={exact}, near={near}")
    return {
        "dataset_sha256": sha256(path),
        "count": len(cases),
        "behavior_balance": dict(behaviors),
        "action_balance": action_balance,
        "deadline_call_dimensions": dict(Counter(case["dimensions"][0] for case in cases if case["expected"]["action"] == "deadline" and case["expected"]["behavior"] == "call")),
        "exact_or_masked_overlap": exact,
        "near_overlap_at_0_85": near,
        "comparison_sources": {path.name: sha256(path) for path in (TRAIN, OLD_DEV, SEALED_HOLDOUT)},
    }


def audit_deadline_regression() -> dict:
    train = strict_json(TRAIN.read_bytes())["cases"]
    calls = [case for case in train if case["action"] == "deadline" and case["behavior"] == "call"]
    clarify = [case for case in train if case["action"] == "deadline" and case["behavior"] == "clarify"]
    time_counts = Counter(case["messages"][1]["tool_calls"][0]["function"]["arguments"]["when"] for case in calls)
    review = strict_json(TRACKED_REVIEW.read_bytes())
    source_hash = review["source"]["development_report_sha256"]
    if RAW_RESULT.exists() and sha256(RAW_RESULT) != source_hash:
        raise ValueError("Local raw development report differs from tracked review provenance")
    dev = {case["id"]: case for case in load_cases(OLD_DEV)[1]}
    failures = []
    for case_id in review["step_300_status_changes"]["regressed_automatic_case_ids"]:
        gold = dev[case_id]
        if gold["expected"]["action"] != "deadline":
            continue
        arguments = gold["expected"]["accepted_calls"][0]["arguments"]
        if str(arguments["id"]) not in gold["prompt"] or arguments["when"] not in gold["prompt"]:
            raise ValueError(f"A supposed full deadline request lacks a gold slot: {case_id}")
        failures.append({"id": case_id, "gold_id": arguments["id"], "gold_when": arguments["when"]})
    return {
        "training_sha256": sha256(TRAIN),
        "tracked_review_sha256": sha256(TRACKED_REVIEW),
        "raw_report_sha256_from_review": source_hash,
        "deadline_call_cases": len(calls),
        "deadline_clarification_cases": len(clarify),
        "distinct_when_values": len(time_counts),
        "most_repeated_when": time_counts.most_common(8),
        "step300_full_request_missing_calls": failures,
    }


def verify_manifest() -> None:
    manifest = strict_json(MANIFEST.read_bytes())
    expected = {
        "schema_version": 1,
        "dataset": {"path": str(OUTPUT.relative_to(ROOT)), "sha256": sha256(OUTPUT), "cases": 80},
        "builder": {"path": "scripts/build_model_proposal_development_v2.py", "sha256": sha256(ROOT / "scripts/build_model_proposal_development_v2.py")},
        "sources": {str(path.relative_to(ROOT)): sha256(path) for path in (TRAIN, OLD_DEV, SEALED_HOLDOUT, TRACKED_REVIEW)},
        "status": "offline_labels_authored_human_review_pending_no_model_inference",
    }
    if manifest != expected:
        raise ValueError("Development v2 manifest or source hash changed")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-manifest", action="store_true")
    args = parser.parse_args()
    try:
        result = {"development_v2": audit_dataset(), "deadline_regression": audit_deadline_regression()}
        if args.verify_manifest:
            verify_manifest()
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("Offline only: no model, tool, policy, session or SQLite execution.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
