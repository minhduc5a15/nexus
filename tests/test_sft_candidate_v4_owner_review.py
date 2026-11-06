"""The owner-review gate must not confuse feedback with approval."""

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import unittest

from scripts.audit_sft_candidate_v4_owner_review import validate_review
from scripts.audit_sft_data import canonical_sha256


DATASET = Path(__file__).resolve().parents[1] / "evals/sft_candidate_v4_draft.json"


class SftCandidateV4OwnerReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = DATASET.read_bytes()
        cls.dataset_hash = hashlib.sha256(source).hexdigest()
        cls.cases = json.loads(source)["cases"][600:]

    def review(self):
        return {
            "schema_version": 1,
            "dataset_id": "nexus-sft-candidate-v4-draft",
            "dataset_sha256": self.dataset_hash,
            "reviewer": {"name": "project_owner", "kind": "human"},
            "revision": 1,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "cases": [
                {"id": case["id"], "case_sha256": canonical_sha256(case), "decision": "approved", "notes": ""}
                for case in self.cases
            ],
        }

    def test_complete_review_is_only_a_label_gate(self):
        summary = validate_review(self.review(), self.dataset_hash, self.cases, require_complete=True)
        self.assertEqual(summary["approved"], 40)
        self.assertTrue(summary["owner_label_review_complete"])
        self.assertFalse(summary["training_authorized"])

    def test_pending_or_correction_does_not_pass_complete_gate(self):
        review = self.review()
        review["cases"][0]["decision"] = "pending"
        review["cases"][1]["decision"] = "needs_correction"
        review["cases"][1]["notes"] = "Bỏ dấu # trước ID."
        summary = validate_review(review, self.dataset_hash, self.cases)
        self.assertEqual((summary["approved"], summary["pending"], summary["needs_correction"]), (38, 1, 1))
        self.assertFalse(summary["owner_label_review_complete"])
        with self.assertRaisesRegex(ValueError, "Owner review incomplete"):
            validate_review(review, self.dataset_hash, self.cases, require_complete=True)

    def test_stale_dataset_or_changed_case_identity_is_rejected(self):
        review = self.review()
        for altered in ("0" * 64, self.dataset_hash[:-1] + "0"):
            with self.subTest(altered=altered):
                wrong = deepcopy(review)
                wrong["dataset_sha256"] = altered
                with self.assertRaises(ValueError):
                    validate_review(wrong, self.dataset_hash, self.cases)
        wrong = deepcopy(review)
        wrong["cases"][0]["case_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "Case hash/order changed"):
            validate_review(wrong, self.dataset_hash, self.cases)

    def test_correction_requires_reason_and_review_cannot_claim_training(self):
        review = self.review()
        review["cases"][0]["decision"] = "needs_correction"
        with self.assertRaisesRegex(ValueError, "Correction reason missing"):
            validate_review(review, self.dataset_hash, self.cases)
        review = self.review()
        review["training_authorized"] = True
        with self.assertRaisesRegex(ValueError, "Unexpected review fields"):
            validate_review(review, self.dataset_hash, self.cases)


if __name__ == "__main__":
    unittest.main()
