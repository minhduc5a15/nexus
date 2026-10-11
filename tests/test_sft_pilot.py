from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from scripts import audit_sft_pilot as audit
from scripts.build_sft_pilot_v1 import render_document


class SftPilotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.document = json.loads(audit.DEFAULT_DATASET.read_text(encoding="utf-8"))

    def write_json(self, value, name="value.json"):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        path = Path(temporary.name) / name
        path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        return path

    def test_pilot_is_balanced_reviewed_locked_and_reproducible(self):
        summary = audit.audit_dataset()
        manifest = audit.verify_manifest()
        review = audit.verify_review()
        self.assertEqual(summary["cases"], 100)
        self.assertEqual(summary["behavior_balance"], audit.EXPECTED_BEHAVIORS)
        self.assertEqual(summary["action_balance"], audit.EXPECTED_ACTIONS)
        self.assertEqual(summary["leakage"]["exact"], [])
        self.assertEqual(summary["leakage"]["masked"], [])
        self.assertEqual(summary["leakage"]["near"], [])
        self.assertEqual(audit.DEFAULT_DATASET.read_text(encoding="utf-8"), render_document())
        self.assertEqual(manifest["dataset"]["sha256"], summary["sha256"])
        self.assertEqual(len(review["reviewed_case_ids"]), 100)
        self.assertFalse(manifest["training_authorized"])
        self.assertFalse(review["training_authorized"])

    def test_every_call_uses_one_schema_valid_grounded_proposal(self):
        for case in self.document["cases"]:
            with self.subTest(case=case["id"]):
                audit._validate_case(case)

    def test_non_call_reply_that_claims_future_mutation_is_rejected(self):
        cases = deepcopy(self.document["cases"])
        target = next(case for case in cases if case["behavior"] == "no_action")
        target["messages"][1]["content"] = "Tôi sẽ lưu câu này thành một task."
        with self.assertRaisesRegex(audit.SftDataError, "implies a mutation"):
            audit._audit_responses(cases)

    def test_evaluation_prompt_leakage_is_detected(self):
        cases = deepcopy(self.document["cases"])
        development = json.loads(audit.DEVELOPMENT.read_text(encoding="utf-8"))
        cases[0]["messages"][0]["content"] = development["cases"][0]["prompt"]
        leakage = audit._audit_leakage(cases)
        self.assertIn(cases[0]["id"], leakage["exact"])

    def test_review_must_cover_every_case_in_dataset_order(self):
        review = json.loads(audit.DEFAULT_REVIEW.read_text(encoding="utf-8"))
        review["reviewed_case_ids"].pop()
        with self.assertRaisesRegex(audit.SftDataError, "does not cover all cases"):
            audit.verify_review(self.write_json(review, "review.json"))


if __name__ == "__main__":
    unittest.main()
