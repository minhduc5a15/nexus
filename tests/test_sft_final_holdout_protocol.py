from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import modal_sft_final_holdout as runner
from scripts import sft_final_holdout_protocol as protocol
from scripts.score_sft_generation import score_proposal_text, summarize_proposal_texts


class SftFinalHoldoutProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = protocol.load_json(protocol.MANIFEST)
        _, cls.cases = protocol.load_cases(
            protocol.ROOT / cls.manifest["holdout"]["path"]
        )

    def generated_text(self, case):
        expected = case["expected"]
        if expected["behavior"] != "call":
            return "Bạn vui lòng bổ sung thông tin còn thiếu."
        payload = json.dumps(expected["accepted_calls"][0], ensure_ascii=False)
        return f"<tool_call>\n{payload}\n</tool_call>"

    def passing_rows(self):
        return [
            score_proposal_text(case, self.generated_text(case))
            for case in self.cases
        ]

    def fail_call(self, rows, index):
        rows[index] = score_proposal_text(
            self.cases[index], "Không có thao tác nào được thực hiện."
        )

    def test_protocol_locks_human_review_candidate_resources_and_budget(self):
        summary = protocol.validate_protocol()
        self.assertEqual(summary["human_review"]["counts"]["approved"], 120)
        self.assertEqual(summary["candidate"]["label"], "step-200")
        self.assertEqual(summary["resources"], {
            "gpu": "T4",
            "cpu": 2,
            "memory_gib": 32,
            "timeout_seconds": 2400,
        })
        self.assertEqual(summary["maximum_compute_usd"], 0.626976)
        self.assertEqual(summary["authorized_total_usd"], 0.7)
        self.assertTrue(summary["holdout_authorized"])
        self.assertFalse(summary["gguf_rollout_authorized"])

    def test_runner_has_resume_refusal_and_side_effect_guards(self):
        text = protocol.RUNNER.read_text(encoding="utf-8")
        self.assertIn("Refusing to overwrite completed final holdout", text)
        self.assertIn('state.get("status") not in {"running", "interrupted"}', text)
        self.assertIn("result_volume.commit()", text)
        self.assertNotIn("sqlite3", text)
        self.assertNotIn("run_turn(", text)

    def test_progress_resumes_only_an_exact_case_prefix(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        path = Path(temporary.name) / "progress.json"
        expected_ids = [case["id"] for case in self.cases]
        fresh = runner._load_progress(
            path,
            run_id="final-run",
            label="baseline",
            expected_ids=expected_ids,
        )
        self.assertEqual(fresh["cases"], [])
        fresh["cases"] = [{"id": expected_ids[0]}]
        path.write_text(json.dumps(fresh), encoding="utf-8")
        resumed = runner._load_progress(
            path,
            run_id="final-run",
            label="baseline",
            expected_ids=expected_ids,
        )
        self.assertEqual([row["id"] for row in resumed["cases"]], expected_ids[:1])
        resumed["cases"][0]["id"] = expected_ids[1]
        path.write_text(json.dumps(resumed), encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "case sequence"):
            runner._load_progress(
                path,
                run_id="final-run",
                label="baseline",
                expected_ids=expected_ids,
            )

    def test_train_evidence_rejects_adapter_or_report_mismatch(self):
        candidate = self.manifest["candidate"]
        report = {
            "run_id": candidate["train_run_id"],
            "state": {"status": "complete"},
            "adapters": {
                "step-200": {
                    "directory_sha256": candidate["adapter_directory_sha256"],
                    "state_sha256_reloaded": candidate["adapter_state_sha256"],
                }
            },
        }
        result = protocol.validate_train_evidence(
            report,
            candidate,
            report_sha256=candidate["train_report_sha256"],
            adapter_directory_sha256=candidate["adapter_directory_sha256"],
        )
        self.assertEqual(result["label"], "step-200")
        for field, value, message in (
            ("report", "0" * 64, "Train report hash"),
            ("directory", "0" * 64, "Adapter directory hash"),
        ):
            with self.subTest(field=field):
                with self.assertRaisesRegex(protocol.FinalHoldoutProtocolError, message):
                    protocol.validate_train_evidence(
                        report,
                        candidate,
                        report_sha256=(
                            value if field == "report" else candidate["train_report_sha256"]
                        ),
                        adapter_directory_sha256=(
                            value if field == "directory" else candidate["adapter_directory_sha256"]
                        ),
                    )

    def test_automatic_acceptance_passes_only_with_all_frozen_gates(self):
        baseline = self.passing_rows()
        candidate = self.passing_rows()
        action_indexes = [
            index for index, case in enumerate(self.cases)
            if case["expected"]["behavior"] == "call"
        ]
        for index in action_indexes[:14]:
            self.fail_call(baseline, index)
        for index in action_indexes[:3]:
            self.fail_call(candidate, index)
        decision = protocol.evaluate_automatic_acceptance(
            baseline, candidate, self.manifest["acceptance"]
        )
        self.assertEqual(decision["candidate_action_exact"], 81)
        self.assertEqual(decision["observed_error_reduction"], 11)
        self.assertTrue(decision["automatic_eligible"])
        self.assertTrue(all(decision["gates"].values()))
        self.assertFalse(decision["gguf_rollout_authorized"])

    def test_each_automatic_acceptance_failure_is_observable(self):
        acceptance = self.manifest["acceptance"]
        action_indexes = [
            index for index, case in enumerate(self.cases)
            if case["expected"]["behavior"] == "call"
        ]

        baseline = self.passing_rows()
        candidate = self.passing_rows()
        for index in action_indexes[:5]:
            self.fail_call(baseline, index)
            self.fail_call(candidate, index)
        decision = protocol.evaluate_automatic_acceptance(baseline, candidate, acceptance)
        self.assertFalse(decision["gates"]["candidate_action_exact_minimum"])
        self.assertFalse(decision["gates"]["minimum_error_reduction"])

        baseline = self.passing_rows()
        candidate = self.passing_rows()
        create_index = next(
            index for index, case in enumerate(self.cases)
            if case["expected"]["action"] == "create"
        )
        for index in action_indexes[12:26]:
            self.fail_call(baseline, index)
        self.fail_call(candidate, create_index)
        decision = protocol.evaluate_automatic_acceptance(baseline, candidate, acceptance)
        self.assertIn("create", decision["action_regressions"])
        self.assertFalse(decision["gates"]["no_action_regression"])

        baseline = self.passing_rows()
        candidate = self.passing_rows()
        control_index = next(
            index for index, case in enumerate(self.cases)
            if case["expected"]["behavior"] != "call"
        )
        candidate[control_index] = score_proposal_text(
            self.cases[control_index],
            '<tool_call>{"name":"delete_task","arguments":{"id":1}}</tool_call>',
        )
        decision = protocol.evaluate_automatic_acceptance(baseline, candidate, acceptance)
        self.assertTrue(decision["candidate_only_safety_regressions"])
        self.assertFalse(decision["gates"]["controls_not_below_baseline"])
        self.assertFalse(decision["gates"]["no_candidate_only_safety_regression"])

        baseline = self.passing_rows()
        candidate = self.passing_rows()
        candidate[0] = deepcopy(candidate[0])
        candidate[0]["error_tags"] = ["generation_error"]
        candidate[0]["automatic_status"] = "fail"
        decision = protocol.evaluate_automatic_acceptance(baseline, candidate, acceptance)
        self.assertEqual(decision["generation_error_ids"], [candidate[0]["id"]])
        self.assertFalse(decision["gates"]["complete_without_generation_error"])

    def test_report_is_rescored_from_raw_generated_text(self):
        rows = self.passing_rows()
        summary = summarize_proposal_texts(rows)
        decision = protocol.evaluate_automatic_acceptance(
            rows, deepcopy(rows), self.manifest["acceptance"]
        )
        report = {
            "schema_version": 1,
            "protocol_id": self.manifest["protocol_id"],
            "run_id": "final-run",
            "stage": "final_holdout",
            "dataset": self.manifest["holdout"],
            "results": {
                "baseline": {"summary": summary, "cases": rows},
                "step-200": {"summary": summary, "cases": deepcopy(rows)},
            },
            "automatic_decision": decision,
            "manual_noncall_review": "pending",
            "provenance": {
                "manifest_sha256": protocol.EXPECTED_MANIFEST_SHA256,
                "holdout_sha256": self.manifest["holdout"]["sha256"],
                "tools_or_database_executed": False,
                "response_repair": False,
            },
        }
        validated = protocol.validate_report(
            report, self.manifest, expected_run_id="final-run"
        )
        self.assertFalse(validated["gguf_rollout_authorized"])
        report["results"]["step-200"]["cases"][0]["generated_text"] = "changed"
        with self.assertRaisesRegex(protocol.FinalHoldoutProtocolError, "score mismatch"):
            protocol.validate_report(report, self.manifest, expected_run_id="final-run")

    def test_manual_review_binds_all_72_responses_and_cannot_authorize_rollout(self):
        rows = self.passing_rows()
        automatic = protocol.evaluate_automatic_acceptance(
            rows, deepcopy(rows), self.manifest["acceptance"]
        )
        report = {
            "run_id": "final-run",
            "dataset": self.manifest["holdout"],
            "candidate": self.manifest["candidate"],
            "automatic_decision": automatic,
            "provenance": {
                "runner_sha256": "runner-hash",
            },
            "results": {
                "baseline": {"cases": rows},
                "step-200": {"cases": deepcopy(rows)},
            },
        }
        reviews = {}
        summary = {}
        for label in ("baseline", "step-200"):
            semantic_rows = []
            for raw in report["results"][label]["cases"]:
                if raw["behavior"] == "call":
                    continue
                semantic_rows.append({
                    "id": raw["id"],
                    "behavior": raw["behavior"],
                    "prompt": raw["prompt"],
                    "generated_text": raw["generated_text"],
                    "response_sha256": protocol.digest_json(raw["generated_text"]),
                    "automatic_status": raw["automatic_status"],
                    "decision": "pass",
                    "reason": "Response is semantically appropriate.",
                })
            reviews[label] = semantic_rows
            summary[label] = {"pass": 36, "fail": 0, "total": 36}
        review = {
            "schema_version": 1,
            "review_id": "nexus-sft-final-holdout-review-v1",
            "reviewer": {"name": "codex", "kind": "assistant"},
            "source": {
                "run_id": "final-run",
                "report_sha256": "report-hash",
                "manifest_sha256": protocol.EXPECTED_MANIFEST_SHA256,
                "runner_sha256": "runner-hash",
                "holdout_sha256": self.manifest["holdout"]["sha256"],
                "human_label_review_sha256": self.manifest["holdout"]["human_review_sha256"],
                "adapter_state_sha256": self.manifest["candidate"]["adapter_state_sha256"],
            },
            "automatic_decision": automatic,
            "reviews": reviews,
            "semantic_noncall_summary": summary,
            "decision": {
                "automatic_eligible": True,
                "manual_noncall_not_below_baseline": True,
                "candidate_status": "eligible_for_gguf",
                "gguf_export_eligible": True,
                "deployment_rollout_authorized": False,
                "final_holdout_status": "passed_candidate_selection",
                "next_step": "prepare_gguf_export_and_deployment_parity_plan",
            },
        }
        result = protocol.validate_manual_review(
            review, report, report_sha256="report-hash"
        )
        self.assertEqual(result["semantic_noncall_summary"]["step-200"]["pass"], 36)
        self.assertFalse(result["decision"]["deployment_rollout_authorized"])

        broken = deepcopy(review)
        broken["reviews"]["step-200"][0]["generated_text"] = "changed"
        with self.assertRaisesRegex(protocol.FinalHoldoutProtocolError, "raw response mismatch"):
            protocol.validate_manual_review(broken, report, report_sha256="report-hash")

        broken = deepcopy(review)
        broken["decision"]["deployment_rollout_authorized"] = True
        with self.assertRaisesRegex(protocol.FinalHoldoutProtocolError, "Final candidate decision"):
            protocol.validate_manual_review(broken, report, report_sha256="report-hash")


if __name__ == "__main__":
    unittest.main()
