from __future__ import annotations

from copy import deepcopy
import unittest

from scripts import review_sft_final_holdout_v2 as review_protocol
from scripts.sft_final_holdout_protocol_v2 import load_json


class FinalHoldoutV2SemanticReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not review_protocol.REPORT.is_file():
            raise unittest.SkipTest("Ignored raw Modal report is not available in this checkout")
        cls.report = load_json(review_protocol.REPORT)
        cls.manifest = load_json(review_protocol.MANIFEST)
        cls.review = load_json(review_protocol.REVIEW)

    def test_all_108_responses_are_reviewed_and_bound_to_raw_report(self):
        summary = review_protocol.validate_review(
            self.review, self.report, self.manifest
        )
        self.assertEqual(summary["reviewed_responses"], 108)
        self.assertEqual(summary["semantic_noncall_summary"]["v3-step-300"]["pass"], 35)
        self.assertEqual(summary["semantic_noncall_summary"]["v4-step-320"]["pass"], 33)
        self.assertFalse(summary["decision"]["gguf_export_eligible"])

    def test_changing_raw_response_binding_is_rejected(self):
        changed = deepcopy(self.review)
        changed["reviews"]["v4-step-320"][0]["response_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "differs from raw responses"):
            review_protocol.validate_review(changed, self.report, self.manifest)

    def test_unreviewed_or_reclassified_case_is_rejected(self):
        missing = deepcopy(self.review)
        missing["reviews"]["base"].pop()
        with self.assertRaisesRegex(ValueError, "differs from raw responses"):
            review_protocol.validate_review(missing, self.report, self.manifest)

        changed = deepcopy(self.review)
        changed["reviews"]["v4-step-320"][11]["decision"] = "pass"
        with self.assertRaisesRegex(ValueError, "differs from raw responses"):
            review_protocol.validate_review(changed, self.report, self.manifest)


if __name__ == "__main__":
    unittest.main()
