from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from scripts import audit_modal_sft_v3_canary as audit


class AuditModalSftV3CanaryTests(unittest.TestCase):
    def write(self, value) -> Path:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "decision.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def test_completed_canary_is_reproducible_and_keeps_paid_gates_closed(self):
        summary = audit.validate()
        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["heartbeats"], 3)
        self.assertEqual(summary["gpu_cost_usd"], 0.0)
        self.assertFalse(summary["training_authorized"])
        self.assertFalse(summary["final_holdout_inference"])

    def test_gpu_or_failed_gate_is_rejected(self):
        original = audit.load(audit.DECISION)
        broken = deepcopy(original)
        broken["billing"]["breakdown_usd"]["gpu"] = 0.01
        with self.assertRaisesRegex(audit.CanaryAuditError, "GPU cost"):
            audit.validate(self.write(broken))
        broken = deepcopy(original)
        broken["gates"]["duplicate_run_id_rejected_before_spawn"] = False
        with self.assertRaisesRegex(audit.CanaryAuditError, "gate failed"):
            audit.validate(self.write(broken))

    def test_billing_arithmetic_is_locked(self):
        original = audit.load(audit.DECISION)
        broken = deepcopy(original)
        broken["billing"]["deploy_and_canary_metered_cost_usd"] += 0.01
        with self.assertRaisesRegex(audit.CanaryAuditError, "billing arithmetic"):
            audit.validate(self.write(broken))


if __name__ == "__main__":
    unittest.main()
