"""Verify the saved base/step-300 proposal report without running a model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.audit_model_proposal_development_v2 import sha256
from scripts.eval_proposals import load_cases, strict_json
from scripts.score_sft_generation import score_proposal_text, summarize_proposal_texts


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "evals/model_proposal_development_v2.json"
PLAN = ROOT / "evals/model_proposal_development_v2_run_plan.json"
DEFAULT_REPORT = ROOT / "evals/results/dev2-base-step300-20261021b/sft-development-v2/development.json"
REVIEW = ROOT / "evals/model_proposal_development_v2_semantic_review.json"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def audit_report(path: Path = DEFAULT_REPORT) -> dict[str, Any]:
    report = strict_json(path.read_bytes())
    plan = strict_json(PLAN.read_bytes())
    _, cases = load_cases(DATASET)
    _require(report.get("protocol_id") == plan["protocol_id"], "Protocol ID changed")
    _require(report.get("stage") == "development" and report.get("state", {}).get("status") == "complete", "Run is incomplete")
    _require(report.get("run_id") == "dev2-base-step300-20261021b", "Unexpected run ID")
    _require(report.get("plan_sha256") == sha256(PLAN), "Run plan hash changed")
    _require(report.get("dataset_sha256") == sha256(DATASET), "Dataset hash changed")
    _require(report.get("source_train_report_sha256") == plan["source_train_report_sha256"], "Source train report changed")
    _require(report.get("adapter_state_sha256") == plan["source_adapter_state_sha256"], "Adapter changed")
    _require(report.get("generation") == plan["generation"], "Generation settings changed")
    _require(report.get("no_tool_execution") is True and report.get("final_holdout_inference") is False, "Side-effect or holdout guard changed")

    summary: dict[str, Any] = {"report_sha256": sha256(path), "run_id": report["run_id"]}
    expected_ids = [case["id"] for case in cases]
    for group_name in ("baseline", "candidate"):
        group = report[group_name]
        rows = group["cases"]
        _require(len(rows) == 80 and [row.get("id") for row in rows] == expected_ids, f"{group_name} case coverage/order changed")
        for case, row in zip(cases, rows):
            _require(isinstance(row.get("generated_text"), str), f"Missing raw response: {case['id']}")
            rescored = score_proposal_text(case, row["generated_text"])
            for key in ("prompt", "parsed", "accepted_calls", "automatic_status", "proposal_correct", "error_tags", "reply_review"):
                _require(row.get(key) == rescored[key], f"{group_name}/{case['id']} scoring drift: {key}")
        expected_summary = summarize_proposal_texts(rows)
        _require(group["summary"] == expected_summary, f"{group_name} summary drift")
        summary[group_name] = {
            "automatic": expected_summary,
            "failed_case_ids": [row["id"] for row in rows if row["automatic_status"] == "fail"],
            "noncall_reply_count": sum(row["behavior"] != "call" and not row["parsed"]["calls"] for row in rows),
        }
    return summary


def audit_semantic_review(report_summary: dict[str, Any], path: Path = REVIEW) -> dict[str, Any]:
    review = strict_json(path.read_bytes())
    report = strict_json(DEFAULT_REPORT.read_bytes())
    _require(sha256(DEFAULT_REPORT) == report_summary["report_sha256"], "Semantic review is not for the audited report")
    _require(review.get("schema_version") == 1, "Unsupported review schema")
    _require(review.get("reviewer") == {"name": "Codex", "kind": "assistant"}, "Reviewer provenance changed")
    _require(review.get("raw_report_sha256") == report_summary["report_sha256"], "Review belongs to another raw report")
    _, cases = load_cases(DATASET)
    control_ids = [case["id"] for case in cases if case["expected"]["behavior"] != "call"]
    _require(review.get("reviewed_control_ids") == control_ids, "Semantic review does not cover all controls in order")
    _require(review.get("rubric") == "No action on controls; reply asks only genuinely missing information, respects negation/quotation, and describes supported capabilities accurately.", "Review rubric changed")
    result = {}
    for group in ("baseline", "candidate"):
        failures = review.get("reply_failures", {}).get(group)
        _require(isinstance(failures, dict) and set(failures) <= set(control_ids), f"Invalid {group} reply failures")
        _require(all(isinstance(reason, str) and reason.strip() for reason in failures.values()), f"{group} reply failure lacks a reason")
        rows = {row["id"]: row for row in report[group]["cases"]}
        _require(all(not rows[case_id]["parsed"]["calls"] for case_id in failures), f"{group} tool text was reviewed as a reply")
        result[group] = {"control_cases": len(control_ids), "semantic_failures": len(failures) + review["tool_call_control_failures"][group], "reply_failure_ids": list(failures)}
    _require(
        review.get("tool_call_control_failures") == {
            group: report_summary[group]["automatic"]["unexpected_calls"]
            for group in ("baseline", "candidate")
        },
        "Control tool-call count changed",
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--semantic-review", action="store_true")
    args = parser.parse_args()
    try:
        summary = audit_report(args.report)
        if args.semantic_review:
            summary["semantic_review"] = audit_semantic_review(summary)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
