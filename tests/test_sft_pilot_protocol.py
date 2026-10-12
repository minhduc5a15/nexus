from copy import deepcopy
import unittest

from scripts import sft_pilot_protocol as protocol
from scripts.score_sft_generation import summarize_proposal_texts


def scored_case(index, status="pass"):
    return {
        "id": f"case-{index}",
        "behavior": "call",
        "action": "create",
        "automatic_status": status,
        "error_tags": [] if status == "pass" else ["missing_call"],
        "reply_review": {"status": "not_required"},
    }


class SftPilotProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = protocol.load_json(protocol.MANIFEST)

    def test_protocol_locks_budget_data_and_excludes_final_holdout(self):
        summary = protocol.validate_protocol()
        self.assertEqual(summary["dataset"]["count"], 100)
        self.assertEqual(summary["development"]["count"], 80)
        self.assertEqual(summary["training_steps"], 200)
        self.assertEqual(summary["maximum_planned_gpu_cost_usd"], 0.7872)
        self.assertFalse(summary["final_holdout_inference"])

    def test_modal_images_separate_training_from_development(self):
        tree = __import__("ast").parse(protocol.RUNNER.read_text(encoding="utf-8"))
        training = protocol._image_local_paths(tree, "training_image")
        development = protocol._image_local_paths(tree, "development_image")
        self.assertFalse(any("development" in path for path in training))
        self.assertIn("evals/model_proposal_development_v1.json", development)
        self.assertFalse(any("holdout" in path for path in training + development))
        runner = protocol.RUNNER.read_text(encoding="utf-8")
        self.assertIn('"final_holdout_not_used": True', runner)
        self.assertNotIn('"final_holdout_inference": False,\n        }', runner)

    def test_completed_train_report_is_accepted_and_failed_gate_is_rejected(self):
        report = {
            "protocol_id": self.manifest["protocol_id"],
            "run_id": "pilot-train",
            "stage": "train",
            "model": self.manifest["model"],
            "dataset": self.manifest["dataset"],
            "training": {"completed_steps": 200, "training_loss": 0.4},
            "adapter": {
                "directory_sha256": "abc",
                "state_sha256_after": "state",
                "state_sha256_reloaded": "state",
            },
            "provenance": {
                "manifest_sha256": protocol.EXPECTED_MANIFEST_SHA256,
                "final_holdout_inference": False,
            },
            "gates": {"completed_all_steps": True, "adapter_reloaded_identically": True},
            "state": {"status": "complete"},
        }
        result = protocol.validate_train_report(
            report, self.manifest, expected_run_id="pilot-train"
        )
        self.assertEqual(result["completed_steps"], 200)
        broken = deepcopy(report)
        broken["gates"]["completed_all_steps"] = False
        with self.assertRaisesRegex(protocol.PilotProtocolError, "gates"):
            protocol.validate_train_report(
                broken, self.manifest, expected_run_id="pilot-train"
            )

    def test_development_summary_is_recomputed_from_raw_cases(self):
        baseline = [scored_case(index) for index in range(80)]
        candidate = [scored_case(index, "fail" if index == 0 else "pass") for index in range(80)]
        report = {
            "protocol_id": self.manifest["protocol_id"],
            "run_id": "pilot-development",
            "stage": "development",
            "train_run_id": "pilot-train",
            "dataset": self.manifest["development"],
            "baseline": {"cases": baseline, "summary": summarize_proposal_texts(baseline)},
            "candidate": {"cases": candidate, "summary": summarize_proposal_texts(candidate)},
            "comparison": {"pass_delta": -1, "unexpected_call_delta": 0},
            "provenance": {"final_holdout_inference": False},
            "gates": {"baseline_completed": True, "candidate_completed": True},
            "state": {"status": "complete"},
        }
        result = protocol.validate_development_report(
            report,
            self.manifest,
            expected_run_id="pilot-development",
            expected_train_run_id="pilot-train",
        )
        self.assertEqual(result["comparison"]["pass_delta"], -1)
        broken = deepcopy(report)
        broken["candidate"]["summary"]["pass"] = 80
        with self.assertRaisesRegex(protocol.PilotProtocolError, "summary mismatch"):
            protocol.validate_development_report(
                broken,
                self.manifest,
                expected_run_id="pilot-development",
                expected_train_run_id="pilot-train",
            )

    def test_manual_review_must_cover_raw_noncall_outputs(self):
        baseline = [
            {
                "id": "clarify",
                "behavior": "clarify",
                "action": "create",
                "automatic_status": "pass",
                "error_tags": [],
                "reply_review": {"status": "pending"},
                "generated_text": "Bạn muốn thêm gì?",
            }
        ]
        candidate = deepcopy(baseline)
        report = {
            "run_id": "dev",
            "baseline": {"cases": baseline, "summary": summarize_proposal_texts(baseline)},
            "candidate": {"cases": candidate, "summary": summarize_proposal_texts(candidate)},
            "comparison": {"pass_delta": 0, "unexpected_call_delta": 0},
        }
        row = {
            "id": "clarify",
            "decision": "pass",
            "reason": "Asks for the missing content.",
            "generated_text": "Bạn muốn thêm gì?",
            "automatic_status": "pass",
        }
        review = {
            "schema_version": 1,
            "review_id": "nexus-sft-pilot-development-review-v1",
            "source": {
                "development_run_id": "dev",
                "development_report_sha256": "dev-sha",
                "train_report_sha256": "train-sha",
            },
            "failed_run": {"state_sha256": "failed-sha"},
            "automatic_summary": {
                "baseline": report["baseline"]["summary"],
                "candidate": report["candidate"]["summary"],
                "comparison": report["comparison"],
            },
            "semantic_noncall_summary": {
                "baseline": {"pass": 1, "fail": 0, "total": 1},
                "candidate": {"pass": 1, "fail": 0, "total": 1},
            },
            "baseline_reviews": [deepcopy(row)],
            "candidate_reviews": [deepcopy(row)],
            "decision": {"rollout_eligible": False, "open_final_holdout": False},
        }
        result = protocol.validate_development_review(
            review,
            report,
            report_sha256="dev-sha",
            train_report_sha256="train-sha",
            failed_state_sha256="failed-sha",
        )
        self.assertFalse(result["rollout_eligible"])
        broken = deepcopy(review)
        broken["candidate_reviews"][0]["generated_text"] = "rewritten"
        with self.assertRaisesRegex(protocol.PilotProtocolError, "text mismatch"):
            protocol.validate_development_review(
                broken,
                report,
                report_sha256="dev-sha",
                train_report_sha256="train-sha",
                failed_state_sha256="failed-sha",
            )


if __name__ == "__main__":
    unittest.main()
