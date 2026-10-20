"""Offline checks for the new development labels and the deadline audit."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from scripts import audit_model_proposal_development_v2 as audit
from scripts.build_model_proposal_development_v2 import build_document


class DevelopmentV2AuditTests(unittest.TestCase):
    def test_balanced_dataset_is_reproducible_and_locked(self):
        result = audit.audit_dataset()
        self.assertEqual(result["count"], 80)
        self.assertEqual(result["deadline_call_dimensions"], {"relative_time": 6, "absolute_time": 2})
        self.assertFalse(result["exact_or_masked_overlap"])
        self.assertFalse(result["near_overlap_at_0_85"])
        audit.verify_manifest()

    def test_deadline_audit_uses_tracked_evidence(self):
        result = audit.audit_deadline_regression()
        self.assertEqual(result["deadline_call_cases"], 60)
        self.assertEqual(result["deadline_clarification_cases"], 18)
        self.assertEqual(
            [case["id"] for case in result["step300_full_request_missing_calls"]],
            ["deadline_relative", "dev_057", "dev_058"],
        )

    def test_deadline_audit_does_not_require_ignored_raw_report(self):
        with TemporaryDirectory() as directory:
            with patch.object(audit, "RAW_RESULT", Path(directory) / "absent.json"):
                self.assertEqual(len(audit.audit_deadline_regression()["step300_full_request_missing_calls"]), 3)

    def test_nonverbatim_deadline_label_is_rejected(self):
        document = build_document()
        deadline = next(case for case in document["cases"] if case["id"] == "dev2_deadline_01")
        deadline["expected"]["accepted_calls"][0]["arguments"]["when"] = "8 giờ sáng mai"
        with TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Nonverbatim when"):
                audit.audit_dataset(path)

    def test_new_case_with_old_training_prompt_is_rejected(self):
        document = build_document()
        list_case = next(case for case in document["cases"] if case["id"] == "dev2_list_01")
        list_case["prompt"] = "Xem danh sách việc đã lưu."
        with TemporaryDirectory() as directory:
            path = Path(directory) / "overlap.json"
            path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "overlaps old evaluation/training"):
                audit.audit_dataset(path)


if __name__ == "__main__":
    unittest.main()
