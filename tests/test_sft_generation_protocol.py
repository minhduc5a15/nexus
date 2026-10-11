from copy import deepcopy
import json
from pathlib import Path
import unittest

from scripts.sft_generation_protocol import (
    GenerationProtocolError,
    validate_protocol,
    validate_manual_review,
    validate_report,
)
from scripts.sft_sanity_protocol import file_sha256, load_json
from scripts.score_sft_generation import summarize


ROOT = Path(__file__).resolve().parents[1]


class SftGenerationProtocolTests(unittest.TestCase):
    def test_tracked_protocol_locks_training_only_and_t4(self):
        result = validate_protocol()
        self.assertEqual(result["dataset"]["count"], 16)
        self.assertEqual(result["generation"]["gpu"], "T4")
        self.assertFalse(result["final_holdout_inference"])
        review = validate_manual_review()
        self.assertEqual(review["summary"]["baseline"], {"pass": 8, "fail": 0, "total": 8})
        self.assertEqual(review["summary"]["candidate"], {"pass": 7, "fail": 1, "total": 8})

    def fake_report(self):
        manifest = load_json(ROOT / "evals/sft_generation_manifest_v1.json")
        dataset = load_json(ROOT / manifest["dataset"]["path"])
        baseline_cases = []
        candidate_cases = []
        for index, case in enumerate(dataset["cases"]):
            common = {
                "id": case["id"],
                "behavior": case["behavior"],
                "action": case["action"],
                "error_tags": [],
            }
            baseline_cases.append({
                **common,
                "proposal_correct": index < 8,
                "exact_token_sequence": index < 4,
            })
            candidate_cases.append({
                **common,
                "proposal_correct": True,
                "exact_token_sequence": index < 14,
            })
        baseline = summarize(baseline_cases)
        candidate = summarize(candidate_cases)
        report = {
            "protocol_id": manifest["protocol_id"],
            "run_id": "generation-1",
            "stage": "generation",
            "dataset_sha256": manifest["dataset"]["sha256"],
            "generation_manifest_sha256": file_sha256(
                ROOT / "evals/sft_generation_manifest_v1.json"
            ),
            "generation": manifest["generation"],
            "source_overfit": {"run_id": manifest["source_overfit"]["run_id"]},
            "adapter": {
                "directory_sha256": manifest["source_overfit"]["adapter_directory_sha256"],
                "state_sha256": manifest["source_overfit"]["adapter_state_sha256"],
            },
            "baseline": {"summary": baseline, "cases": baseline_cases},
            "candidate": {"summary": candidate, "cases": candidate_cases},
            "comparison": {
                "proposal_correct_delta": candidate["proposal_correct"] - baseline["proposal_correct"],
                "exact_token_sequence_delta": candidate["exact_token_sequences"] - baseline["exact_token_sequences"],
            },
            "provenance": {"final_holdout_inference": False},
            "gates": {
                "source_overfit_valid": True,
                "adapter_directory_hash_matches": True,
                "adapter_state_hash_matches": True,
                "baseline_completed": True,
                "candidate_completed": True,
            },
            "state": {"status": "complete"},
        }
        return report, manifest

    def test_report_recomputes_summaries_and_comparison(self):
        report, manifest = self.fake_report()
        result = validate_report(
            report, manifest, expected_run_id="generation-1"
        )
        self.assertEqual(result["candidate"]["proposal_correct"], 16)
        self.assertEqual(result["candidate"]["exact_token_sequences"], 14)
        for mutation in ("summary", "adapter", "holdout", "gate"):
            broken = deepcopy(report)
            if mutation == "summary":
                broken["candidate"]["summary"]["proposal_correct"] = 15
            elif mutation == "adapter":
                broken["adapter"]["state_sha256"] = "0" * 64
            elif mutation == "holdout":
                broken["provenance"]["final_holdout_inference"] = True
            else:
                broken["gates"]["candidate_completed"] = False
            with self.subTest(mutation=mutation), self.assertRaises(GenerationProtocolError):
                validate_report(
                    broken, manifest, expected_run_id="generation-1"
                )


if __name__ == "__main__":
    unittest.main()
