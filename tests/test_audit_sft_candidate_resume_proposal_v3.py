from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from scripts import audit_sft_candidate_resume_proposal_v3 as audit


class AuditSftCandidateResumeProposalV3Tests(unittest.TestCase):
    def write(self, value) -> Path:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "proposal.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def test_proposal_locks_source_target_budget_and_closed_holdout(self):
        summary = audit.validate()
        self.assertEqual(summary["source_checkpoint_step"], 25)
        self.assertEqual(summary["remaining_optimizer_steps"], 275)
        self.assertAlmostEqual(summary["maximum_compute_usd"], 2.302366666666667)
        self.assertEqual(summary["requested_additional_cost_cap_usd"], 2.45)
        self.assertFalse(summary["training_authorized"])
        self.assertFalse(summary["final_holdout_v2_inference"])

    def test_source_hash_and_budget_cannot_drift(self):
        original = audit.load(audit.PROPOSAL)
        broken = deepcopy(original)
        broken["resume_source"]["directory_sha256"] = "0" * 64
        with self.assertRaisesRegex(audit.ResumeProposalError, "Source checkpoint hash"):
            audit.validate(self.write(broken))
        broken = deepcopy(original)
        broken["budget"]["maximum_compute_usd"] += 0.01
        with self.assertRaisesRegex(audit.ResumeProposalError, "Budget arithmetic"):
            audit.validate(self.write(broken))

    def test_proposal_cannot_claim_approval_or_open_later_scope(self):
        original = audit.load(audit.PROPOSAL)
        broken = deepcopy(original)
        broken["gates_before_authorization"]["project_owner_approved_additional_cost"] = True
        with self.assertRaisesRegex(audit.ResumeProposalError, "unreceived approval"):
            audit.validate(self.write(broken))
        broken = deepcopy(original)
        broken["scope"]["final_holdout_v2_inference"] = True
        with self.assertRaisesRegex(audit.ResumeProposalError, "later stages"):
            audit.validate(self.write(broken))


if __name__ == "__main__":
    unittest.main()
