from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from scripts import review_holdout_labels as review


class HoldoutLabelReviewTests(unittest.TestCase):
    def setUp(self):
        self.value = review.create_review()

    def approve_all(self):
        for row in self.value["cases"]:
            row["decision"] = "approved"

    def test_template_locks_all_120_cases_in_dataset_order(self):
        summary = review.validate_review(self.value)
        self.assertEqual(len(self.value["cases"]), 120)
        self.assertEqual(summary["counts"], {
            "approved": 0,
            "needs_correction": 0,
            "pending": 120,
        })
        self.assertFalse(summary["complete"])
        self.assertFalse(summary["model_inference_authorized"])

    def test_complete_review_authorizes_inference(self):
        self.approve_all()
        summary = review.validate_review(self.value, require_complete=True)
        self.assertEqual(summary["counts"]["approved"], 120)
        self.assertTrue(summary["complete"])
        self.assertTrue(summary["model_inference_authorized"])

    def test_pending_or_correction_blocks_complete_gate(self):
        self.approve_all()
        self.value["cases"][0]["decision"] = "pending"
        with self.assertRaisesRegex(review.HoldoutReviewError, "incomplete"):
            review.validate_review(self.value, require_complete=True)

        self.value["cases"][0]["decision"] = "needs_correction"
        self.value["cases"][0]["notes"] = "Expected content loses punctuation."
        summary = review.validate_review(self.value)
        self.assertEqual(summary["correction_ids"], [self.value["cases"][0]["id"]])
        with self.assertRaisesRegex(review.HoldoutReviewError, "incomplete"):
            review.validate_review(self.value, require_complete=True)

    def test_correction_requires_a_note(self):
        self.value["cases"][0]["decision"] = "needs_correction"
        with self.assertRaisesRegex(review.HoldoutReviewError, "needs a note"):
            review.validate_review(self.value)

    def test_immutable_case_snapshot_and_order_cannot_change(self):
        mutations = []
        changed_prompt = deepcopy(self.value)
        changed_prompt["cases"][0]["snapshot"]["prompt"] += " changed"
        mutations.append(changed_prompt)
        changed_digest = deepcopy(self.value)
        changed_digest["cases"][0]["case_sha256"] = "0" * 64
        mutations.append(changed_digest)
        reordered = deepcopy(self.value)
        reordered["cases"][0], reordered["cases"][1] = (
            reordered["cases"][1],
            reordered["cases"][0],
        )
        mutations.append(reordered)
        for mutation in mutations:
            with self.subTest():
                with self.assertRaisesRegex(review.HoldoutReviewError, "Immutable"):
                    review.validate_review(mutation)

    def test_dataset_hash_mismatch_is_rejected(self):
        self.value["dataset"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(review.HoldoutReviewError, "dataset hash"):
            review.validate_review(self.value)

    def test_write_review_refuses_to_overwrite_on_init(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        path = Path(temporary.name) / "review.json"
        review.write_review(path, self.value, exclusive=True)
        with self.assertRaises(FileExistsError):
            review.write_review(path, self.value, exclusive=True)


if __name__ == "__main__":
    unittest.main()
