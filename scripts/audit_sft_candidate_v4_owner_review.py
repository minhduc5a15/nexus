"""Validate an owner review against the current 40-case SFT v4 draft.

This is a data-review gate only. It never authorizes GPU training.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.audit_sft_data import canonical_sha256
from scripts.eval_proposals import strict_json


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "evals/sft_candidate_v4_draft.json"
REVIEW = ROOT / "evals/results/sft_candidate_v4_owner_review.json"
DECISIONS = {"pending", "approved", "needs_correction"}
REVIEWER = {"name": "project_owner", "kind": "human"}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_review(review: Any, dataset_hash: str, cases: list[dict], *, require_complete: bool = False) -> dict:
    """Reject stale hashes, altered case identities and incomplete owner reviews."""
    _require(isinstance(review, dict), "Review must be an object")
    _require(set(review) == {"schema_version", "dataset_id", "dataset_sha256", "reviewer", "revision", "updated_at", "cases"}, "Unexpected review fields")
    _require(review["schema_version"] == 1 and review["dataset_id"] == "nexus-sft-candidate-v4-draft", "Wrong review identity")
    _require(review["dataset_sha256"] == dataset_hash, "Review dataset hash changed")
    _require(review["reviewer"] == REVIEWER, "Owner review provenance changed")
    _require(type(review["revision"]) is int and review["revision"] >= 0, "Invalid review revision")
    timestamp = review["updated_at"]
    _require(isinstance(timestamp, str), "Owner review has no save timestamp")
    try:
        _require(datetime.fromisoformat(timestamp).tzinfo is not None, "Save timestamp requires timezone")
    except ValueError as error:
        raise ValueError("Invalid save timestamp") from error
    rows = review["cases"]
    _require(isinstance(rows, list) and len(rows) == len(cases) == 40, "Owner review must contain all 40 cases")
    counts: Counter[str] = Counter()
    pending_ids = []
    correction_ids = []
    for case, row in zip(cases, rows):
        _require(isinstance(row, dict) and set(row) == {"id", "case_sha256", "decision", "notes"}, f"Invalid review row for {case['id']}")
        _require(row["id"] == case["id"] and row["case_sha256"] == canonical_sha256(case), f"Case hash/order changed at {case['id']}")
        _require(isinstance(row["decision"], str) and row["decision"] in DECISIONS, f"Invalid decision for {case['id']}")
        _require(isinstance(row["notes"], str) and len(row["notes"]) <= 4000, f"Invalid notes for {case['id']}")
        if row["decision"] == "needs_correction":
            _require(bool(row["notes"].strip()), f"Correction reason missing for {case['id']}")
            correction_ids.append(case["id"])
        elif row["decision"] == "pending":
            pending_ids.append(case["id"])
        counts[row["decision"]] += 1
    complete = counts["approved"] == 40
    if require_complete:
        _require(complete, f"Owner review incomplete: pending={pending_ids}, corrections={correction_ids}")
    return {
        "dataset_sha256": dataset_hash,
        "cases": 40,
        "approved": counts["approved"],
        "pending": counts["pending"],
        "needs_correction": counts["needs_correction"],
        "pending_ids": pending_ids,
        "correction_ids": correction_ids,
        "owner_label_review_complete": complete,
        "training_authorized": False,
    }


def audit_review(review_path: Path = REVIEW, dataset_path: Path = DATASET, *, require_complete: bool = False) -> dict:
    source = dataset_path.read_bytes()
    dataset = strict_json(source)
    _require(dataset.get("dataset_id") == "nexus-sft-candidate-v4-draft", "Unexpected dataset")
    cases = dataset.get("cases")
    _require(isinstance(cases, list) and len(cases) == 640, "Expected 640 SFT cases")
    review = strict_json(review_path.read_bytes())
    return validate_review(review, hashlib.sha256(source).hexdigest(), cases[600:], require_complete=require_complete)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path, default=REVIEW)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    try:
        summary = audit_review(args.review, args.dataset, require_complete=args.require_complete)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
