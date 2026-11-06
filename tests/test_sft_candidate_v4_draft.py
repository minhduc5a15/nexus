"""Offline guards for the unreviewed 40-case training amendment."""

import json
from pathlib import Path
import tempfile
import unittest

from scripts.audit_sft_candidate_v4_draft import audit_dataset
from scripts.inspect_sft_tokens import audit_selected_dataset


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "evals/sft_candidate_v4_draft.json"


class CandidateV4DraftTests(unittest.TestCase):
    def test_preserves_v3_and_exposes_policy_gap_without_using_it_as_gold(self):
        summary = audit_dataset()
        self.assertEqual((summary["preserved_v3"], summary["added"]), (600, 40))
        self.assertEqual(summary["evaluation_overlap"], {"exact": [], "masked": [], "near_at_0_92": []})
        self.assertEqual(summary["policy_compatibility_diagnostic"]["compatible_calls"], 30)
        self.assertEqual(summary["policy_compatibility_diagnostic"]["incompatible_calls"], 0)
        self.assertEqual(summary["label_review"]["assistant_reviewed"], 40)
        self.assertEqual(summary["status"], "draft_assistant_reviewed_owner_gate_separate_not_authorized_for_training")

    def test_prefix_mutation_is_rejected(self):
        document = json.loads(DATASET.read_text(encoding="utf-8"))
        document["cases"][0]["messages"][0]["content"] = "tampered"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tampered.json"
            path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "prefix changed"):
                audit_dataset(path)

    def test_noncall_reply_cannot_claim_side_effect(self):
        document = json.loads(DATASET.read_text(encoding="utf-8"))
        control = next(case for case in document["cases"][600:] if case["behavior"] == "no_action")
        control["messages"][1]["content"] = "Đã xóa task."
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tampered.json"
            path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "implies a mutation"):
                audit_dataset(path)

    def test_shared_token_inspector_accepts_reviewed_v4_draft(self):
        summary = audit_selected_dataset(DATASET)
        self.assertEqual(summary["cases"], 640)


if __name__ == "__main__":
    unittest.main()
