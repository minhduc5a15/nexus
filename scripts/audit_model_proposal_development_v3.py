"""Audit the new model-only development split before any v4 inference."""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path

from nexus.core.deadlines import VIETNAM_TIMEZONE, parse_deadline
from scripts.audit_proposal_data import shape, validate_split
from scripts.build_model_proposal_development_v3 import OUTPUT, render_document
from scripts.eval_proposals import load_cases, strict_json


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "evals/sft_candidate_v4_draft.json"
COMPARISONS = (
    ROOT / "evals/model_proposal_development_v1.json",
    ROOT / "evals/model_proposal_development_v2.json",
    ROOT / "evals/model_proposal_holdout_v2.json",
    ROOT / "evals/model_proposal_framing_diagnostic_v1.json",
)
REFERENCE = datetime(2026, 11, 3, 12, tzinfo=VIETNAM_TIMEZONE)
SIMILARITY_THRESHOLD = 0.85


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def comparison_records() -> list[tuple[str, str, str, str]]:
    records = []
    for path in COMPARISONS:
        _, cases = load_cases(path)
        records.extend((path.name, case["id"], case["prompt"], shape(case["prompt"], case)) for case in cases)
    training = strict_json(TRAIN.read_bytes())
    for case in training["cases"]:
        prompt = case["messages"][0]["content"]
        calls = [item["function"] for item in case["messages"][1].get("tool_calls", [])]
        records.append((TRAIN.name, case["id"], prompt, shape(prompt, {"expected": {"accepted_calls": calls}})))
    return records


def audit_dataset(path: Path = OUTPUT) -> dict:
    source, cases = load_cases(path)
    if path.resolve() == OUTPUT.resolve() and source.decode("utf-8") != render_document():
        raise ValueError("Development v3 differs from its builder")
    action_balance = validate_split(cases, 40, 4, 12)
    behaviors = Counter(case["expected"]["behavior"] for case in cases)
    if behaviors != Counter({"call": 28, "clarify": 6, "no_action": 6}):
        raise ValueError(f"Unexpected behavior balance: {dict(behaviors)}")
    families = Counter(case["family"] for case in cases)
    if Counter(families.values()) != Counter({2: 14, 1: 12}):
        raise ValueError("Unexpected family grouping")
    for case in cases:
        if case.get("origin") != "authored_development_v3":
            raise ValueError(f"Unexpected origin: {case['id']}")
        calls = case["expected"]["accepted_calls"]
        if len(calls) > 1:
            raise ValueError(f"More than one gold call: {case['id']}")
        if calls:
            arguments = calls[0]["arguments"]
            if "id" in arguments and str(arguments["id"]) not in case["prompt"]:
                raise ValueError(f"ID not in prompt: {case['id']}")
            for field in ("content", "when"):
                if field in arguments and arguments[field] not in case["prompt"]:
                    raise ValueError(f"Nonverbatim {field}: {case['id']}")
            if "when" in arguments:
                parse_deadline(arguments["when"], reference=REFERENCE)
    records = comparison_records()
    exact, near = [], []
    for case in cases:
        target = shape(case["prompt"], case)
        score, closest = max(
            ((SequenceMatcher(None, target, old_shape, autojunk=False).ratio(), (name, old_id))
             for name, old_id, _, old_shape in records),
            key=lambda item: item[0],
        )
        if any(target == old_shape or case["prompt"].casefold() == prompt.casefold()
               for _, _, prompt, old_shape in records):
            exact.append(case["id"])
        if score >= SIMILARITY_THRESHOLD:
            near.append({"id": case["id"], "score": round(score, 6), "source": closest[0], "source_id": closest[1]})
    if exact or near:
        raise ValueError(f"Development v3 overlaps comparison sources: exact={exact}, near={near}")
    return {
        "dataset_sha256": hashlib.sha256(source).hexdigest(),
        "cases": len(cases),
        "behavior_balance": dict(behaviors),
        "action_balance": action_balance,
        "family_sizes": dict(Counter(families.values())),
        "exact_or_masked_overlap": exact,
        "near_overlap_at_0_85": near,
        "comparison_sha256": {path.name: sha256(path) for path in (TRAIN, *COMPARISONS)},
        "note": "Lexical screening only; a person must still review labels and semantic family separation.",
    }


def main() -> int:
    result = audit_dataset()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
