from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from scripts import audit_sft_dataset_v3_design as audit


class SftDatasetV3DesignTests(unittest.TestCase):
    def setUp(self):
        self.design = json.loads(audit.DEFAULT_DESIGN.read_text(encoding="utf-8"))

    def write_design(self, value) -> Path:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "design.json"
        path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        return path

    def test_tracked_design_binds_evidence_and_keeps_both_future_gates_closed(self):
        summary = audit.audit_design()
        self.assertEqual(summary["evidence_cases"], 24)
        self.assertEqual(summary["automatic_failure_cases"], 15)
        self.assertEqual(summary["semantic_failure_cases"], 14)
        self.assertEqual(summary["v3_new_cases"], 200)
        self.assertEqual(summary["final_holdout_v2_cases"], 120)
        self.assertFalse(summary["training_authorized"])
        self.assertFalse(summary["final_holdout_v2_inference_authorized"])

    def test_unknown_evidence_case_is_rejected(self):
        broken = deepcopy(self.design)
        broken["evidence"][0]["case_id"] = "invented_case"
        with self.assertRaisesRegex(audit.DatasetV3DesignError, "Unknown evidence case"):
            audit.audit_design(self.write_design(broken))

    def test_quota_drift_is_rejected(self):
        broken = deepcopy(self.design)
        broken["taxonomy"][0]["v3_new_examples"] += 1
        with self.assertRaisesRegex(audit.DatasetV3DesignError, "allocation must total 200"):
            audit.audit_design(self.write_design(broken))

    def test_design_cannot_authorize_training_or_holdout_inference(self):
        broken = deepcopy(self.design)
        broken["dataset_v3"]["training_authorized"] = True
        with self.assertRaisesRegex(audit.DatasetV3DesignError, "must not authorize training"):
            audit.audit_design(self.write_design(broken))

        broken = deepcopy(self.design)
        broken["final_holdout_v2"]["model_inference_authorized"] = True
        with self.assertRaisesRegex(audit.DatasetV3DesignError, "inference must remain closed"):
            audit.audit_design(self.write_design(broken))


if __name__ == "__main__":
    unittest.main()
