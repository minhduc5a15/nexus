from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from scripts import audit_sft_candidate_v3 as audit
from scripts.build_sft_candidate_v3 import build_document, render_document


class SftCandidateV3Tests(unittest.TestCase):
    def setUp(self):
        self.dataset = json.loads(audit.DEFAULT_DATASET.read_text(encoding="utf-8"))

    def write_dataset(self, value) -> Path:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "candidate.json"
        path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        return path

    def test_dataset_is_reproducible_balanced_and_separated_from_evaluation(self):
        self.assertEqual(render_document(), audit.DEFAULT_DATASET.read_text(encoding="utf-8"))
        summary = audit.audit_dataset()
        self.assertEqual(summary["cases"], 600)
        self.assertEqual(summary["preserved_v2_cases"], 400)
        self.assertEqual(summary["new_cases"], 200)
        self.assertEqual(summary["behavior_balance"], {"call": 420, "clarify": 90, "no_action": 90})
        self.assertTrue(all(value == 60 for value in summary["action_balance"].values()))
        self.assertEqual(summary["leakage"]["exact"], [])
        self.assertEqual(summary["leakage"]["masked"], [])
        self.assertEqual(summary["leakage"]["near"], [])

    def test_v2_prefix_and_taxonomy_allocation_cannot_drift(self):
        broken = deepcopy(self.dataset)
        broken["cases"][0]["rationale"] = "changed"
        with self.assertRaisesRegex(audit.CandidateV3Error, "400 reviewed v2"):
            audit.audit_dataset(self.write_dataset(broken))

        broken = deepcopy(self.dataset)
        broken["cases"][400]["target_primary"] = "balanced_regression_coverage"
        with self.assertRaisesRegex(audit.CandidateV3Error, "Taxonomy allocation"):
            audit.audit_dataset(self.write_dataset(broken))

    def test_manifest_and_review_keep_training_and_holdout_closed(self):
        manifest = audit.verify_manifest()
        review = audit.verify_review()
        self.assertFalse(manifest["training_authorized"])
        self.assertFalse(manifest["evaluation_inputs"]["final_holdout_v2"]["model_inference_authorized"])
        self.assertFalse(review["training_authorized"])
        self.assertFalse(review["final_holdout_v2_inference_authorized"])

    def test_builder_returns_fresh_data(self):
        first = build_document()
        second = build_document()
        first["cases"][400]["messages"][0]["content"] = "changed"
        self.assertNotEqual(first, second)


if __name__ == "__main__":
    unittest.main()
