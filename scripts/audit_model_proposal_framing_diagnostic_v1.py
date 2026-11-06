"""Validate the exposed paired-framing diagnostic without model inference."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any

from nexus.core.deadlines import VIETNAM_TIMEZONE, parse_deadline
from scripts.build_model_proposal_framing_diagnostic_v1 import OUTPUT, render_document
from scripts.eval_proposals import load_cases


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = datetime(2026, 10, 21, 12, tzinfo=VIETNAM_TIMEZONE)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_dataset(path: Path = OUTPUT) -> dict[str, Any]:
    raw, cases = load_cases(path)
    if path.resolve() == OUTPUT.resolve() and raw.decode("utf-8") != render_document():
        raise ValueError("Diagnostic differs from its builder")
    if len(cases) != 20:
        raise ValueError("Expected exactly 20 cases")
    if len({case["id"] for case in cases}) != 20 or len({case["prompt"] for case in cases}) != 20:
        raise ValueError("Duplicate ID or prompt")
    if Counter(case["expected"]["action"] for case in cases) != {"deadline": 12, "list": 8}:
        raise ValueError("Action balance changed")

    expected_old = {
        "deadline_01": "model_proposal_development_v1.json",
        "deadline_02": "model_proposal_development_v1.json",
        "deadline_03": "model_proposal_development_v1.json",
        "list_01": "model_proposal_development_v2.json",
        "list_02": "model_proposal_development_v2.json",
    }
    old_prompts = {}
    for name in set(expected_old.values()):
        _, old = load_cases(ROOT / "evals" / name)
        old_prompts[name] = {case["prompt"] for case in old}
    final_path = ROOT / "evals/model_proposal_holdout_v2.json"
    _, final_cases = load_cases(final_path)
    final_prompts = {case["prompt"] for case in final_cases}

    for index in range(0, len(cases), 2):
        left, right = cases[index:index + 2]
        pair = left["pair_id"]
        if left["variant"] != "a" or right["variant"] != "b" or right["pair_id"] != pair:
            raise ValueError(f"Broken pair: {pair}")
        if left["expected"] != right["expected"] or left["prompt"] == right["prompt"]:
            raise ValueError(f"Pair changed gold call or duplicated prompt: {pair}")
        if left["prompt"] in final_prompts or right["prompt"] in final_prompts:
            raise ValueError(f"Diagnostic contains a final-holdout prompt: {pair}")
        for case in (left, right):
            if case["expected"]["behavior"] != "call" or len(case["expected"]["accepted_calls"]) != 1:
                raise ValueError(f"Unexpected label: {case['id']}")
            args = case["expected"]["accepted_calls"][0]["arguments"]
            if pair.startswith("deadline_"):
                if str(args["id"]) not in case["prompt"] or args["when"] not in case["prompt"]:
                    raise ValueError(f"ID/time not verbatim: {case['id']}")
                parse_deadline(args["when"], reference=REFERENCE)
        if pair in expected_old:
            if left["prompt"] not in old_prompts[expected_old[pair]]:
                raise ValueError(f"Previously exposed failure changed: {pair}")
        elif left["prompt"] in set().union(*old_prompts.values()):
            raise ValueError(f"New diagnostic prompt already in old development: {pair}")

    return {
        "dataset_sha256": sha256(path),
        "cases": len(cases),
        "pairs": len(cases) // 2,
        "reused_exposed_failures": len(expected_old),
        "exact_final_holdout_overlap": 0,
        "final_holdout_sha256": sha256(final_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    print(json.dumps(audit_dataset(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
