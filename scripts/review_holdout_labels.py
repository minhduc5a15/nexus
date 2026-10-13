"""Create, edit and validate the human review of every final-holdout label."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.eval_proposals import load_cases, strict_json


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "evals/model_proposal_holdout_v1.json"
DEFAULT_REVIEW = ROOT / "evals/model_proposal_holdout_human_review_v1.json"
DECISIONS = {"pending", "approved", "needs_correction"}
REVIEWER = {"name": "project_owner", "kind": "human"}


class HoldoutReviewError(ValueError):
    """The human label review does not match the locked holdout."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise HoldoutReviewError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest_json(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _snapshot(case: dict[str, Any]) -> dict[str, Any]:
    return {
        key: case[key]
        for key in (
            "id",
            "prompt",
            "family",
            "rationale",
            "dimensions",
            "origin",
            "expected",
        )
    }


def review_id_for_dataset(dataset_path: Path) -> str:
    name = dataset_path.name
    if name == "model_proposal_holdout_v1.json":
        return "nexus-model-proposal-holdout-human-review-v1"
    if name == "model_proposal_holdout_v2.json":
        return "nexus-model-proposal-holdout-human-review-v2"
    raise HoldoutReviewError(f"Unsupported holdout dataset: {name}")


def create_review(dataset_path: Path = DEFAULT_DATASET) -> dict[str, Any]:
    source, cases = load_cases(dataset_path)
    return {
        "schema_version": 1,
        "review_id": review_id_for_dataset(dataset_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "path": str(dataset_path.resolve().relative_to(ROOT)),
            "sha256": hashlib.sha256(source).hexdigest(),
            "cases": len(cases),
        },
        "reviewer": dict(REVIEWER),
        "cases": [
            {
                "id": case["id"],
                "case_sha256": digest_json(_snapshot(case)),
                "snapshot": _snapshot(case),
                "decision": "pending",
                "notes": "",
            }
            for case in cases
        ],
    }


def load_review(path: Path) -> dict[str, Any]:
    value = strict_json(path.read_bytes())
    _require(isinstance(value, dict), "Review must be a JSON object")
    return value


def validate_review(
    review: dict[str, Any],
    dataset_path: Path = DEFAULT_DATASET,
    *,
    require_complete: bool = False,
) -> dict[str, Any]:
    expected = create_review(dataset_path)
    _require(review.get("schema_version") == 1, "Unsupported review schema")
    _require(review.get("review_id") == expected["review_id"], "Unexpected review ID")
    _require(review.get("dataset") == expected["dataset"], "Review dataset hash or identity changed")
    _require(review.get("reviewer") == REVIEWER, "Reviewer must remain project_owner/human")
    try:
        created_at = datetime.fromisoformat(review["created_at"])
        _require(created_at.tzinfo is not None, "created_at must include a timezone")
    except (KeyError, TypeError, ValueError) as error:
        raise HoldoutReviewError("created_at must be a timezone-aware ISO timestamp") from error

    rows = review.get("cases")
    _require(isinstance(rows, list), "Review cases must be a list")
    _require(len(rows) == len(expected["cases"]) == 120, "Review must contain all 120 cases")
    counts: Counter[str] = Counter()
    correction_ids = []
    for actual, locked in zip(rows, expected["cases"]):
        _require(isinstance(actual, dict), "Every review row must be an object")
        _require(
            set(actual) == {"id", "case_sha256", "snapshot", "decision", "notes"},
            f"Unexpected review fields in {locked['id']}",
        )
        for field in ("id", "case_sha256", "snapshot"):
            _require(
                digest_json(actual.get(field)) == digest_json(locked[field]),
                f"Immutable review field changed for {locked['id']}: {field}",
            )
        decision = actual.get("decision")
        notes = actual.get("notes")
        _require(decision in DECISIONS, f"Invalid decision for {locked['id']}")
        _require(isinstance(notes, str), f"Notes must be text for {locked['id']}")
        if decision == "needs_correction":
            _require(notes.strip(), f"Correction needs a note for {locked['id']}")
            correction_ids.append(locked["id"])
        counts[decision] += 1

    complete = counts["approved"] == 120
    if require_complete:
        _require(complete, "Human review is incomplete or requests label corrections")
    return {
        "review_id": review["review_id"],
        "dataset_sha256": expected["dataset"]["sha256"],
        "reviewer": REVIEWER,
        "counts": {
            decision: counts[decision]
            for decision in ("approved", "needs_correction", "pending")
        },
        "correction_ids": correction_ids,
        "complete": complete,
        "model_inference_authorized": complete,
    }


def write_review(path: Path, value: dict[str, Any], *, exclusive: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive:
        with path.open("x", encoding="utf-8") as output:
            json.dump(value, output, ensure_ascii=False, indent=2)
            output.write("\n")
        return
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _display(row: dict[str, Any], index: int, total: int) -> None:
    case = row["snapshot"]
    expected = case["expected"]
    print(f"\n[{index + 1}/{total}] {case['id']} — family: {case['family']}")
    print("Prompt:")
    print(case["prompt"])
    print(f"Rationale: {case['rationale']}")
    print(f"Dimensions: {', '.join(case['dimensions'])}")
    print(
        "Expected: "
        + json.dumps(
            {
                "behavior": expected["behavior"],
                "action": expected["action"],
                "accepted_calls": expected["accepted_calls"],
                "missing_slots": expected["missing_slots"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"Current decision: {row['decision']}")
    if row["notes"]:
        print(f"Notes: {row['notes']}")


def interactive(review_path: Path, dataset_path: Path = DEFAULT_DATASET) -> dict[str, Any]:
    review = load_review(review_path)
    validate_review(review, dataset_path)
    rows = review["cases"]
    index = next(
        (position for position, row in enumerate(rows) if row["decision"] != "approved"),
        len(rows),
    )
    while index < len(rows):
        row = rows[index]
        _display(row, index, len(rows))
        command = input("[a] approve, [f] needs correction, [s] skip, [p] previous, [q] save/quit: ").strip().casefold()
        if command == "a":
            row["decision"] = "approved"
            row["notes"] = ""
            index += 1
        elif command == "f":
            note = input("Mô tả chính xác phần nhãn cần sửa: ").strip()
            if not note:
                print("Cần ghi lý do để đánh dấu needs_correction.")
                continue
            row["decision"] = "needs_correction"
            row["notes"] = note
            index += 1
        elif command == "s":
            index += 1
        elif command == "p":
            index = max(0, index - 1)
        elif command == "q":
            break
        else:
            print("Lệnh không hợp lệ.")
            continue
        write_review(review_path, review, exclusive=False)
    write_review(review_path, review, exclusive=False)
    return validate_review(review, dataset_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--review", type=Path, default=DEFAULT_REVIEW)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("init")
    subparsers.add_parser("interactive")
    check = subparsers.add_parser("check")
    check.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    try:
        dataset = args.dataset.resolve()
        review_path = args.review.resolve()
        if args.command == "init":
            write_review(review_path, create_review(dataset), exclusive=True)
            result = validate_review(load_review(review_path), dataset)
        elif args.command == "interactive":
            result = interactive(review_path, dataset)
        else:
            result = validate_review(
                load_review(review_path),
                dataset,
                require_complete=args.require_complete,
            )
    except (OSError, TypeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
