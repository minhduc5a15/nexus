from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from scripts import audit_sft_candidate_v2 as audit
from scripts.build_sft_candidate_v2 import render_document
from scripts.inspect_sft_tokens import audit_selected_dataset


class SftCandidateV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.document = json.loads(audit.DEFAULT_DATASET.read_text(encoding="utf-8"))

    def write_json(self, value, name="value.json"):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        path = Path(temporary.name) / name
        path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        return path

    def test_candidate_is_balanced_reviewed_locked_and_reproducible(self):
        summary = audit.audit_dataset()
        manifest = audit.verify_manifest()
        review = audit.verify_review()
        self.assertEqual(summary["cases"], 400)
        self.assertEqual(summary["source_pilot_cases"], 100)
        self.assertEqual(summary["new_cases"], 300)
        self.assertEqual(summary["behavior_balance"], audit.EXPECTED_BEHAVIORS)
        self.assertEqual(summary["action_balance"], audit.EXPECTED_ACTIONS)
        self.assertEqual(summary["largest_family"], 4)
        self.assertEqual(summary["leakage"]["exact"], [])
        self.assertEqual(summary["leakage"]["masked"], [])
        self.assertEqual(summary["leakage"]["near"], [])
        self.assertEqual(audit.DEFAULT_DATASET.read_text(encoding="utf-8"), render_document())
        self.assertEqual(manifest["dataset"]["sha256"], summary["sha256"])
        self.assertEqual(len(review["reviewed_case_ids"]), 400)
        self.assertFalse(manifest["training_authorized"])
        self.assertFalse(review["training_authorized"])
        self.assertFalse(review["open_final_holdout"])

    def test_frozen_pilot_is_preserved_as_exact_prefix(self):
        pilot = json.loads(audit.PILOT_DATASET.read_text(encoding="utf-8"))
        self.assertEqual(self.document["cases"][:100], pilot["cases"])

    def test_token_inspector_dispatches_to_candidate_audit(self):
        summary = audit_selected_dataset(audit.DEFAULT_DATASET)
        self.assertEqual(summary["dataset_id"], "nexus-sft-candidate-v2")
        self.assertEqual(summary["cases"], 400)

    def test_inverse_actions_and_unsupported_time_do_not_call_tools(self):
        targets = {
            "candidate_no_action_unsupported_uncomplete",
            "candidate_no_action_unsupported_clear_deadline",
            "candidate_no_action_unsupported_week",
            "candidate_no_action_unsupported_weekday",
        }
        selected = {case["id"]: case for case in self.document["cases"] if case["id"] in targets}
        self.assertEqual(set(selected), targets)
        for case in selected.values():
            with self.subTest(case=case["id"]):
                self.assertEqual(case["behavior"], "no_action")
                self.assertNotIn("tool_calls", case["messages"][1])

    def test_family_cannot_cross_behavior_or_action(self):
        value = deepcopy(self.document)
        value["cases"][100]["family"] = value["cases"][132]["family"]
        with self.assertRaisesRegex(audit.CandidateDataError, "Family crosses"):
            audit.audit_dataset(self.write_json(value))

    def test_review_must_cover_every_case_in_order(self):
        review = json.loads(audit.DEFAULT_REVIEW.read_text(encoding="utf-8"))
        review["reviewed_case_ids"][0], review["reviewed_case_ids"][1] = (
            review["reviewed_case_ids"][1],
            review["reviewed_case_ids"][0],
        )
        with self.assertRaisesRegex(audit.CandidateDataError, "does not cover"):
            audit.verify_review(self.write_json(review, "review.json"))

    def test_manifest_locks_tokenizer_measurements(self):
        manifest = json.loads(audit.DEFAULT_MANIFEST.read_text(encoding="utf-8"))
        manifest["tokenizer_validation"]["maximum_total_tokens"] += 1
        with self.assertRaisesRegex(audit.CandidateDataError, "Tokenizer validation"):
            audit.verify_manifest(self.write_json(manifest, "manifest.json"))


if __name__ == "__main__":
    unittest.main()
