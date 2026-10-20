"""Offline guards for the independent development-v2 Modal runner."""

from copy import deepcopy
import json
from pathlib import Path
import unittest

import modal_sft_development_v2 as runner


ROOT = Path(__file__).resolve().parents[1]


class DevelopmentV2RunPlanTests(unittest.TestCase):
    def setUp(self):
        self.plan = json.loads((ROOT / "evals/model_proposal_development_v2_run_plan.json").read_text(encoding="utf-8"))

    def test_locked_plan_validates_without_gpu(self):
        runner.validate_plan(self.plan, ROOT)

    def test_dataset_hash_mismatch_is_rejected(self):
        plan = deepcopy(self.plan)
        plan["dataset"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "dataset changed"):
            runner.validate_plan(plan, ROOT)

    def test_training_and_holdout_flags_cannot_be_enabled(self):
        for key in ("training_authorized", "final_holdout_inference_authorized"):
            plan = deepcopy(self.plan)
            plan[key] = True
            with self.assertRaises(ValueError):
                runner.validate_plan(plan, ROOT)

    def test_base_revision_and_adapter_are_fixed(self):
        for field, value in (("model", {"id": "other", "revision": "0"}), ("source_adapter_state_sha256", "0" * 64)):
            plan = deepcopy(self.plan)
            plan[field] = value
            with self.assertRaises(ValueError):
                runner.validate_plan(plan, ROOT)

    def test_runner_does_not_package_holdout(self):
        source = (ROOT / "modal_sft_development_v2.py").read_text(encoding="utf-8")
        self.assertNotIn('add_local_file("evals/model_proposal_holdout_v2.json"', source)
        self.assertIn("previous._generate_development_cases", source)


if __name__ == "__main__":
    unittest.main()
