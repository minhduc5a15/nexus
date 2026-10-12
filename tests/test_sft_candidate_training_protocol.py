from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from scripts import sft_candidate_training_protocol as protocol


class SftCandidateTrainingProtocolTests(unittest.TestCase):
    def write_text(self, text, name):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        path = Path(temporary.name) / name
        path.write_text(text, encoding="utf-8")
        return path

    def test_protocol_locks_data_budget_checkpoints_and_holdout_boundary(self):
        summary = protocol.validate_protocol()
        self.assertEqual(summary["dataset"]["count"], 400)
        self.assertEqual(summary["optimizer_steps"], 200)
        self.assertEqual(summary["microbatches"], 800)
        self.assertEqual(summary["checkpoints"], ["step-100", "step-200"])
        self.assertEqual(summary["probe_balance"], {"call": 28, "clarify": 7, "no_action": 5})
        self.assertEqual(summary["maximum_planned_gpu_seconds"], 8400)
        self.assertEqual(summary["maximum_planned_gpu_cost_usd"], 1.3776)
        self.assertEqual(summary["requested_total_cost_cap_usd"], 1.5)
        self.assertTrue(summary["training_authorized"])
        self.assertFalse(summary["final_holdout_inference"])

    def test_runner_has_local_and_remote_authorization_guards(self):
        runner = protocol.RUNNER.read_text(encoding="utf-8")
        self.assertIn('if manifest.get("training_authorized") is not True', runner)
        self.assertIn('if not protocol["training_authorized"]', runner)
        self.assertIn('save_strategy="steps"', runner)
        self.assertIn('adapter-step-100', runner)
        self.assertIn('adapter-step-200', runner)

    def test_training_image_excludes_development_and_holdout(self):
        tree = protocol.ast.parse(protocol.RUNNER.read_text(encoding="utf-8"))
        training = protocol._image_local_paths(tree, "training_image")
        development = protocol._image_local_paths(tree, "development_image")
        self.assertFalse(any("development" in path.casefold() for path in training))
        self.assertFalse(any("holdout" in path.casefold() for path in training + development))
        self.assertIn("evals/model_proposal_development_v1.json", development)

    def test_missing_remote_authorization_guard_is_rejected(self):
        runner = protocol.RUNNER.read_text(encoding="utf-8").replace(
            'if manifest.get("training_authorized") is not True:',
            'if False:',
            1,
        )
        with self.assertRaisesRegex(protocol.CandidateProtocolError, "Remote training authorization"):
            protocol.validate_protocol(runner_path=self.write_text(runner, "runner.py"))

    def test_probe_must_cover_every_action_and_non_call_behavior(self):
        manifest = protocol.load_json(protocol.MANIFEST)
        dataset = protocol.load_json(protocol.ROOT / manifest["dataset"]["path"])
        broken = deepcopy(manifest)
        broken["training"]["teacher_forced_probe_ids"][0] = broken["training"]["teacher_forced_probe_ids"][1]
        with self.assertRaisesRegex(protocol.CandidateProtocolError, "40 unique"):
            protocol._validate_probe(broken, dataset)

    def development_review_fixture(self):
        case_ids = [f"noncall_{index:02d}" for index in range(24)]

        def result_set(label, unexpected_calls):
            return {
                "summary": {"unexpected_calls": unexpected_calls},
                "cases": [
                    {
                        "id": case_id,
                        "behavior": "no_action",
                        "automatic_status": "pass",
                        "generated_text": f"{label} reply {case_id}",
                    }
                    for case_id in case_ids
                ],
            }

        development_report = {
            "run_id": "development-run",
            "baseline": result_set("baseline", 2),
            "candidates": {
                "step-100": result_set("step-100", 3),
                "step-200": result_set("step-200", 1),
            },
            "comparisons": {
                "step-100": {"pass_delta": 0, "unexpected_call_delta": 1},
                "step-200": {"pass_delta": 0, "unexpected_call_delta": -1},
            },
        }
        train_report = {
            "run_id": "train-run",
            "adapters": {
                "step-100": {"state_sha256_reloaded": "adapter-100"},
                "step-200": {"state_sha256_reloaded": "adapter-200"},
            },
        }
        review = {
            "schema_version": 1,
            "review_id": "nexus-sft-candidate-development-review-v2",
            "source": {
                "train_run_id": "train-run",
                "development_run_id": "development-run",
                "train_report_sha256": "train-report",
                "development_report_sha256": "development-report",
                "manifest_sha256": protocol.EXPECTED_MANIFEST_SHA256,
                "adapter_state_sha256": {
                    "step-100": "adapter-100",
                    "step-200": "adapter-200",
                },
            },
            "automatic_summary": {
                "baseline": development_report["baseline"]["summary"],
                "candidates": {
                    label: result["summary"]
                    for label, result in development_report["candidates"].items()
                },
                "comparisons": development_report["comparisons"],
            },
            "reviews": {},
            "semantic_noncall_summary": {},
            "step_200_status_changes": {
                "fixed_automatic_case_ids": [],
                "regressed_automatic_case_ids": [],
            },
            "decision": {
                "selected_checkpoint": "step-200",
                "rollout_eligible": False,
                "open_final_holdout": False,
            },
        }
        for label, result in {
            "baseline": development_report["baseline"],
            **development_report["candidates"],
        }.items():
            review["reviews"][label] = [
                {
                    "id": case["id"],
                    "generated_text": case["generated_text"],
                    "automatic_status": case["automatic_status"],
                    "decision": "pass",
                    "reason": "Câu trả lời phù hợp với no-action.",
                }
                for case in result["cases"]
            ]
            review["semantic_noncall_summary"][label] = {
                "pass": 24,
                "fail": 0,
                "total": 24,
            }
        return review, development_report, train_report

    def test_development_review_is_bound_to_raw_outputs_and_keeps_holdout_closed(self):
        review, development_report, train_report = self.development_review_fixture()
        summary = protocol.validate_development_review(
            review,
            development_report,
            train_report,
            development_report_sha256="development-report",
            train_report_sha256="train-report",
        )
        self.assertEqual(summary["selected_checkpoint"], "step-200")
        self.assertFalse(summary["rollout_eligible"])
        self.assertFalse(summary["final_holdout_opened"])

        review["reviews"]["step-200"][0]["generated_text"] = "edited after review"
        with self.assertRaisesRegex(protocol.CandidateProtocolError, "review text mismatch"):
            protocol.validate_development_review(
                review,
                development_report,
                train_report,
                development_report_sha256="development-report",
                train_report_sha256="train-report",
            )

    def test_development_review_cannot_authorize_holdout_or_rollout(self):
        review, development_report, train_report = self.development_review_fixture()
        for field, message in (
            ("rollout_eligible", "cannot authorize rollout"),
            ("open_final_holdout", "keep final holdout closed"),
        ):
            broken = deepcopy(review)
            broken["decision"][field] = True
            with self.subTest(field=field):
                with self.assertRaisesRegex(protocol.CandidateProtocolError, message):
                    protocol.validate_development_review(
                        broken,
                        development_report,
                        train_report,
                        development_report_sha256="development-report",
                        train_report_sha256="train-report",
                    )


if __name__ == "__main__":
    unittest.main()
