from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from scripts import audit_sft_candidate_training_plan_v3 as audit


class SftCandidateTrainingPlanV3Tests(unittest.TestCase):
    def write_plan(self, value) -> Path:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "plan.json"
        path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        return path

    def test_plan_locks_data_steps_budget_and_closed_holdout(self):
        summary = audit.validate_plan()
        self.assertEqual(summary["dataset_cases"], 600)
        self.assertEqual(summary["optimizer_steps"], 300)
        self.assertEqual(summary["microbatches"], 1200)
        self.assertEqual(summary["checkpoints"], ["step-150", "step-300"])
        self.assertEqual(summary["probe"]["behaviors"], {"call": 28, "clarify": 7, "no_action": 5})
        self.assertEqual(summary["maximum_planned_gpu_seconds"], 10200)
        self.assertAlmostEqual(summary["maximum_compute_usd"], 2.428632)
        self.assertEqual(summary["requested_total_cost_cap_usd"], 2.6)
        self.assertTrue(summary["training_authorized"])
        self.assertFalse(summary["final_holdout_v2_inference_authorized"])
        self.assertTrue(summary["runner_authored"])
        self.assertEqual(summary["attempt"]["status"], "interrupted")
        self.assertEqual(summary["attempt"]["last_logged_optimizer_step"], 184)
        self.assertEqual(summary["attempt"]["retained_adapters"], 0)
        self.assertEqual(summary["attempt"]["metered_cost_usd"], 1.0208283)
        self.assertFalse(summary["attempt"]["replacement_run_authorized_at_interruption"])
        self.assertEqual(summary["replacement"]["additional_cost_cap_usd"], 2.6)
        self.assertTrue(summary["replacement"]["consumed"])
        self.assertEqual(
            summary["replacement"]["attempt"]["last_logged_optimizer_step"], 68
        )
        self.assertEqual(
            summary["replacement"]["attempt"]["metered_cost_usd"], 0.45206372
        )
        self.assertEqual(summary["replacement"]["attempt"]["retained_adapters"], 0)
        self.assertFalse(
            summary["replacement"]["attempt"]["third_run_authorized"]
        )
        self.assertEqual(
            summary["authorized_run_ids"]["train"],
            "sft-candidate-v3-train-20261014c",
        )
        self.assertEqual(summary["active_authorization"]["additional_cost_cap_usd"], 2.6)
        self.assertFalse(summary["active_authorization"]["consumed"])

    def test_authorization_and_holdout_state_are_locked(self):
        original = audit.load_json(audit.PLAN)
        for field, message in (
            ("training_authorized", "must be authorized"),
            ("holdout", "must remain closed"),
        ):
            broken = deepcopy(original)
            if field == "training_authorized":
                broken[field] = False
            else:
                broken["final_holdout"]["model_inference_authorized"] = True
            with self.subTest(field=field):
                with self.assertRaisesRegex(audit.TrainingPlanV3Error, message):
                    audit.validate_plan(self.write_plan(broken))

    def test_budget_and_step_arithmetic_cannot_drift(self):
        original = audit.load_json(audit.PLAN)
        broken = deepcopy(original)
        broken["training"]["maximum_steps"] = 299
        with self.assertRaisesRegex(audit.TrainingPlanV3Error, "Optimizer-step arithmetic"):
            audit.validate_plan(self.write_plan(broken))

        broken = deepcopy(original)
        broken["budget"]["maximum_compute_usd"] += 0.01
        with self.assertRaisesRegex(audit.TrainingPlanV3Error, "Total budget arithmetic"):
            audit.validate_plan(self.write_plan(broken))


if __name__ == "__main__":
    unittest.main()
