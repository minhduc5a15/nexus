"""Report-audit tests use synthetic generation and never load a model."""

from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from scripts import audit_model_proposal_development_v2_report as audit
from scripts.audit_model_proposal_development_v2 import sha256
from scripts.eval_proposals import load_cases
from scripts.score_sft_generation import score_proposal_text, summarize_proposal_texts


class DevelopmentV2ReportTests(unittest.TestCase):
    def make_report(self):
        plan = json.loads(audit.PLAN.read_text(encoding="utf-8"))
        _, cases = load_cases(audit.DATASET)
        rows = []
        for case in cases:
            calls = case["expected"]["accepted_calls"]
            if calls:
                text = "<tool_call>\n" + json.dumps(calls[0], ensure_ascii=False) + "\n</tool_call><|im_end|>"
            else:
                text = "Bạn muốn cung cấp phần còn thiếu?" if case["expected"]["behavior"] == "clarify" else "Tôi chưa thực hiện thao tác."
            rows.append(score_proposal_text(case, text))
        group = {"summary": summarize_proposal_texts(rows), "cases": rows}
        return {
            "protocol_id": plan["protocol_id"],
            "run_id": "dev2-base-step300-20261021b",
            "stage": "development",
            "state": {"status": "complete"},
            "plan_sha256": sha256(audit.PLAN),
            "dataset_sha256": sha256(audit.DATASET),
            "source_train_report_sha256": plan["source_train_report_sha256"],
            "adapter_state_sha256": plan["source_adapter_state_sha256"],
            "generation": plan["generation"],
            "no_tool_execution": True,
            "final_holdout_inference": False,
            "baseline": deepcopy(group),
            "candidate": deepcopy(group),
        }

    def write(self, directory, name, value):
        path = Path(directory) / name
        path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        return path

    def test_complete_synthetic_report_recalculates(self):
        with TemporaryDirectory() as directory:
            report = self.write(directory, "report.json", self.make_report())
            result = audit.audit_report(report)
            self.assertEqual(result["baseline"]["automatic"]["pass"], 80)
            self.assertEqual(result["candidate"]["automatic"]["pass"], 80)

    def test_scoring_drift_and_holdout_flag_are_rejected(self):
        with TemporaryDirectory() as directory:
            report = self.make_report()
            report["baseline"]["cases"][0]["automatic_status"] = "fail"
            path = self.write(directory, "drift.json", report)
            with self.assertRaisesRegex(ValueError, "scoring drift"):
                audit.audit_report(path)
            report = self.make_report()
            report["final_holdout_inference"] = True
            path = self.write(directory, "holdout.json", report)
            with self.assertRaisesRegex(ValueError, "Side-effect or holdout guard"):
                audit.audit_report(path)

    def test_semantic_review_must_match_raw_report_hash(self):
        with TemporaryDirectory() as directory:
            report_path = self.write(directory, "report.json", self.make_report())
            summary = audit.audit_report(report_path)
            _, cases = load_cases(audit.DATASET)
            controls = [case["id"] for case in cases if case["expected"]["behavior"] != "call"]
            review = {
                "schema_version": 1,
                "reviewer": {"name": "Codex", "kind": "assistant"},
                "raw_report_sha256": "0" * 64,
                "reviewed_control_ids": controls,
                "rubric": "No action on controls; reply asks only genuinely missing information, respects negation/quotation, and describes supported capabilities accurately.",
                "tool_call_control_failures": {"baseline": 0, "candidate": 0},
                "reply_failures": {"baseline": {}, "candidate": {}},
            }
            review_path = self.write(directory, "review.json", review)
            with patch.object(audit, "DEFAULT_REPORT", report_path):
                with self.assertRaisesRegex(ValueError, "another raw report"):
                    audit.audit_semantic_review(summary, review_path)


if __name__ == "__main__":
    unittest.main()
