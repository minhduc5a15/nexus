"""Rescore the saved paired-framing report from raw model responses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.audit_model_proposal_framing_diagnostic_v1 import ROOT, sha256
from scripts.eval_proposals import load_cases, strict_json
from scripts.score_sft_generation import score_proposal_text, summarize_proposal_texts


DATASET = ROOT / "evals/model_proposal_framing_diagnostic_v1.json"
PLAN = ROOT / "evals/model_proposal_framing_diagnostic_v1_run_plan.json"
DEFAULT_REPORT = ROOT / "evals/results/framing-base-step300-20261021/sft-framing-diagnostic-v1/diagnostic.json"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def audit_report(path: Path = DEFAULT_REPORT) -> dict[str, Any]:
    report = strict_json(path.read_bytes())
    plan = strict_json(PLAN.read_bytes())
    _, cases = load_cases(DATASET)
    _require(report.get("protocol_id") == plan["protocol_id"] and report.get("run_id") == plan["run_id"], "Run identity changed")
    _require(report.get("stage") == "exposed_diagnostic" and report.get("state", {}).get("status") == "complete", "Run is incomplete")
    for field, expected in (
        ("plan_sha256", sha256(PLAN)),
        ("dataset_sha256", sha256(DATASET)),
        ("source_train_report_sha256", plan["source_train_report_sha256"]),
        ("adapter_state_sha256", plan["source_adapter_state_sha256"]),
        ("generation", plan["generation"]),
        ("no_tool_execution", True),
        ("final_holdout_inference", False),
    ):
        _require(report.get(field) == expected, f"Report provenance changed: {field}")

    summary: dict[str, Any] = {"raw_report_sha256": sha256(path), "pairs": []}
    expected_ids = [case["id"] for case in cases]
    for group_name in ("baseline", "candidate"):
        group = report[group_name]
        rows = group["cases"]
        _require(len(rows) == len(cases) and [row.get("id") for row in rows] == expected_ids, f"{group_name} case coverage/order changed")
        for case, row in zip(cases, rows):
            _require(isinstance(row.get("generated_text"), str), f"Missing raw response: {case['id']}")
            rescored = score_proposal_text(case, row["generated_text"])
            for key in ("prompt", "parsed", "accepted_calls", "automatic_status", "proposal_correct", "error_tags"):
                _require(row.get(key) == rescored[key], f"Scoring drift: {group_name}/{case['id']}/{key}")
        expected_summary = summarize_proposal_texts(rows)
        _require(group["summary"] == expected_summary, f"{group_name} summary drift")
        summary[group_name] = {
            "automatic": expected_summary,
            "failed_cases": [
                {"id": row["id"], "error_tags": row["error_tags"], "generated_text": row["generated_text"]}
                for row in rows if row["automatic_status"] != "pass"
            ],
        }
    for index in range(0, len(cases), 2):
        left, right = cases[index:index + 2]
        summary["pairs"].append({
            "pair_id": left["pair_id"],
            "action": left["expected"]["action"],
            "origin": left["origin"],
            "baseline": [report["baseline"]["cases"][offset]["automatic_status"] for offset in (index, index + 1)],
            "candidate": [report["candidate"]["cases"][offset]["automatic_status"] for offset in (index, index + 1)],
        })
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    try:
        summary = audit_report(args.report)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
