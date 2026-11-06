"""Offline gates for the reviewed v4 draft and independent development set."""

from __future__ import annotations

from copy import deepcopy
import unittest

from scripts.audit_sft_candidate_v4_preflight import MANIFEST, audit_preflight, validate_manifest
from scripts.eval_proposals import strict_json


class SftCandidateV4PreflightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = strict_json(MANIFEST.read_bytes())

    def test_reviewed_data_and_diagnostic_split_are_locked(self) -> None:
        summary = audit_preflight()
        self.assertEqual(summary["owner_approved"], 40)
        self.assertEqual(summary["development_cases"], 40)
        self.assertEqual(summary["target_optimizer_steps"], 320)
        self.assertEqual(summary["maximum_observed_tokens"], 1368)
        self.assertFalse(summary["training_authorized"])
        self.assertFalse(summary["inference_authorized"])

    def test_changed_owner_review_hash_is_rejected(self) -> None:
        changed = deepcopy(self.manifest)
        changed["dataset"]["owner_review_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "Hash changed"):
            validate_manifest(changed)

    def test_changed_development_hash_is_rejected(self) -> None:
        changed = deepcopy(self.manifest)
        changed["development"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "Hash changed"):
            validate_manifest(changed)

    def test_manifest_cannot_grant_gpu_or_holdout_access(self) -> None:
        changed = deepcopy(self.manifest)
        changed["gates"]["training_authorized"] = True
        with self.assertRaisesRegex(ValueError, "Preflight gate changed"):
            validate_manifest(changed)
        changed = deepcopy(self.manifest)
        changed["gates"]["final_holdout_inference_authorized"] = True
        with self.assertRaisesRegex(ValueError, "Preflight gate changed"):
            validate_manifest(changed)

    def test_token_limit_cannot_be_silently_changed(self) -> None:
        changed = deepcopy(self.manifest)
        changed["tokenizer_validation"]["maximum_observed_tokens"] = 2049
        with self.assertRaisesRegex(ValueError, "Tokenizer validation changed"):
            validate_manifest(changed)

    def test_budget_cannot_be_marked_approved_or_miscalculated(self) -> None:
        changed = deepcopy(self.manifest)
        changed["execution_design"]["budget_authorized"] = True
        with self.assertRaisesRegex(ValueError, "Budget must remain unapproved"):
            validate_manifest(changed)
        changed = deepcopy(self.manifest)
        changed["execution_design"]["maximum_planned_compute_usd"] = 1.0
        with self.assertRaisesRegex(ValueError, "Compute estimate changed"):
            validate_manifest(changed)


if __name__ == "__main__":
    unittest.main()
