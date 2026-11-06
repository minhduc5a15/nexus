"""CPU-only checks for the locked v4 Modal runner."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import modal_sft_candidate_v4 as runner
from scripts.audit_sft_candidate_v4_execution import PLAN, RUNNER, validate_protocol


class SftCandidateV4ExecutionTests(unittest.TestCase):
    def write(self, name: str, content: str) -> Path:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / name
        path.write_text(content, encoding="utf-8")
        return path

    def test_offline_plan_locks_separate_images_and_scoped_gpu_authorization(self) -> None:
        summary = validate_protocol()
        self.assertEqual(summary["training_cases"], 640)
        self.assertEqual(summary["development_cases"], 40)
        paths = summary["packaged_paths"]
        self.assertNotIn("evals/model_proposal_development_v3.json", paths["train_only"])
        self.assertNotIn("evals/sft_candidate_v4_draft.json", paths["development_only"])
        self.assertFalse(any("review_sft_v4_web" in path for path in paths["shared"] + paths["train_only"] + paths["development_only"]))
        self.assertTrue(summary["training_authorized"])
        self.assertTrue(summary["development_inference_authorized"])
        self.assertFalse(summary["final_holdout_inference_authorized"])

    def test_local_and_remote_guard_accept_approved_stage_only(self) -> None:
        for stage, run_id in (("train", runner.TRAIN_RUN_ID), ("development", runner.DEVELOPMENT_RUN_ID)):
            with self.subTest(stage=stage):
                self.assertEqual(runner._authorize(stage, run_id)["run_ids"][stage], run_id)
        with self.assertRaisesRegex(PermissionError, "Stage or run ID"):
            runner._authorize("train", runner.DEVELOPMENT_RUN_ID)

    def test_plan_cannot_grant_authorization(self) -> None:
        changed = deepcopy(json.loads(PLAN.read_text(encoding="utf-8")))
        changed["authorization"]["training_authorized"] = False
        with self.assertRaisesRegex(ValueError, "Authorization gate changed"):
            validate_protocol(plan_path=self.write("plan.json", json.dumps(changed)))

    def test_runner_cannot_package_extra_local_file_or_change_resources(self) -> None:
        source = RUNNER.read_text(encoding="utf-8")
        extra = source.replace(
            'training_image = base_image.add_local_file(',
            'base_image = base_image.add_local_file("scripts/review_sft_v4_web.py", "/repo/web.py")\ntraining_image = base_image.add_local_file(',
            1,
        )
        with self.assertRaisesRegex(ValueError, "packages unexpected files"):
            validate_protocol(runner_path=self.write("runner.py", extra))
        changed = source.replace('image=training_image, gpu="T4", timeout=1800', 'image=training_image, gpu="T4", timeout=1900', 1)
        with self.assertRaisesRegex(ValueError, "Train function resources changed"):
            validate_protocol(runner_path=self.write("runner.py", changed))

    def test_missing_authorization_guard_is_rejected(self) -> None:
        source = RUNNER.read_text(encoding="utf-8")
        changed = source.replace('plan["authorization"].get("training_authorized") is not True', "False", 1)
        with self.assertRaisesRegex(ValueError, "Remote authorization guard missing"):
            validate_protocol(runner_path=self.write("runner.py", changed))

    def test_synchronous_submission_is_rejected(self) -> None:
        source = RUNNER.read_text(encoding="utf-8")
        changed = source.replace("run_train.spawn(chosen)", "run_train.remote(chosen)", 1)
        with self.assertRaisesRegex(ValueError, "asynchronous submission missing"):
            validate_protocol(runner_path=self.write("runner.py", changed))


if __name__ == "__main__":
    unittest.main()
