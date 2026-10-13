from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from scripts import audit_model_proposal_holdout_v2 as audit
from scripts import review_holdout_labels as review
from scripts.build_model_proposal_holdout_v2 import render_document


class ModelProposalHoldoutV2Tests(unittest.TestCase):
    def setUp(self):
        self.document = json.loads(audit.DEFAULT_HOLDOUT.read_text(encoding="utf-8"))

    def write_holdout(self, value) -> Path:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "model_proposal_holdout_v2.json"
        path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        return path

    def test_holdout_is_reproducible_balanced_and_separated(self):
        self.assertEqual(render_document(), audit.DEFAULT_HOLDOUT.read_text(encoding="utf-8"))
        summary = audit.audit_holdout()
        self.assertEqual(summary["cases"], 120)
        self.assertEqual(summary["behavior_balance"], {"call": 84, "clarify": 12, "no_action": 24})
        self.assertTrue(all(value == 12 for value in summary["action_balance"].values()))
        self.assertEqual(summary["leakage"]["exact"], [])
        self.assertEqual(summary["leakage"]["masked"], [])
        self.assertEqual(summary["leakage"]["near"], [])
        self.assertFalse(summary["model_inference_authorized"])

    def test_training_prompt_leakage_is_rejected(self):
        broken = deepcopy(self.document)
        training = json.loads(audit.TRAINING.read_text(encoding="utf-8"))
        broken["cases"][0]["prompt"] = training["cases"][0]["messages"][0]["content"]
        broken["cases"][0]["expected"] = {
            "behavior": "call",
            "action": "create",
            "accepted_calls": [
                training["cases"][0]["messages"][1]["tool_calls"][0]["function"]
            ],
            "missing_slots": [],
        }
        with self.assertRaisesRegex(audit.HoldoutV2Error, "Exact prompt leakage"):
            audit.audit_holdout(self.write_holdout(broken))

    def test_synthetic_markers_are_rejected_and_terminal_punctuation_is_flexible(self):
        first = self.document["cases"][0]
        contents = {
            call["arguments"]["content"]
            for call in first["expected"]["accepted_calls"]
        }
        self.assertEqual(contents, {"đo lại chiều rộng cái kệ", "đo lại chiều rộng cái kệ."})

        broken = deepcopy(self.document)
        broken["cases"][1]["prompt"] += " DATA(example)"
        with self.assertRaisesRegex(audit.HoldoutV2Error, "Synthetic prompt markers"):
            audit.audit_holdout(self.write_holdout(broken))

    def test_human_review_is_bound_to_v2_and_starts_closed(self):
        value = review.create_review(audit.DEFAULT_HOLDOUT)
        self.assertEqual(value["review_id"], "nexus-model-proposal-holdout-human-review-v2")
        result = review.validate_review(value, audit.DEFAULT_HOLDOUT)
        self.assertEqual(result["counts"]["pending"], 120)
        self.assertFalse(result["complete"])
        self.assertFalse(result["model_inference_authorized"])
        with self.assertRaisesRegex(review.HoldoutReviewError, "incomplete"):
            review.validate_review(value, audit.DEFAULT_HOLDOUT, require_complete=True)

    def test_tracked_manifest_locks_complete_review_but_keeps_protocol_closed(self):
        manifest = audit.verify_manifest()
        self.assertTrue(manifest["gates"]["human_label_review_complete"])
        self.assertFalse(manifest["gates"]["training_authorized"])
        self.assertFalse(manifest["gates"]["final_holdout_inference_authorized_by_protocol"])

    def test_v1_review_identity_remains_unchanged(self):
        self.assertEqual(
            review.review_id_for_dataset(review.DEFAULT_DATASET),
            "nexus-model-proposal-holdout-human-review-v1",
        )

    def test_draft_one_feedback_preserves_the_human_corrections(self):
        path = audit.ROOT / "evals/model_proposal_holdout_v2_draft1_feedback.json"
        feedback = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(feedback["counts"], {"approved": 24, "needs_correction": 23, "pending_at_stop": 73})
        self.assertEqual(len(feedback["reviewed_cases"]), 47)


if __name__ == "__main__":
    unittest.main()
