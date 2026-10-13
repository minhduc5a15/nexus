from __future__ import annotations

import ast
from pathlib import Path
import tempfile
import unittest

from scripts import sft_candidate_training_protocol_v3 as protocol


class SftCandidateTrainingProtocolV3Tests(unittest.TestCase):
    def write_runner(self, text: str) -> Path:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "runner.py"
        path.write_text(text, encoding="utf-8")
        return path

    def test_preflight_locks_runner_data_resources_and_checkpoints(self):
        summary = protocol.validate_protocol()
        self.assertEqual(summary["dataset"]["count"], 600)
        self.assertEqual(summary["optimizer_steps"], 300)
        self.assertEqual(summary["microbatches"], 1200)
        self.assertEqual(summary["checkpoints"], ["step-150", "step-300"])
        self.assertEqual(summary["probe_balance"], {"call": 28, "clarify": 7, "no_action": 5})
        self.assertEqual(summary["maximum_planned_gpu_seconds"], 10200)
        self.assertEqual(summary["requested_total_cost_cap_usd"], 2.6)
        self.assertFalse(summary["training_authorized"])
        self.assertFalse(summary["final_holdout_inference"])

    def test_runner_has_two_independent_authorization_guards(self):
        runner = protocol.RUNNER.read_text(encoding="utf-8")
        self.assertIn('if manifest.get("training_authorized") is not True', runner)
        self.assertIn('if not protocol["training_authorized"]', runner)
        self.assertIn('adapter-step-150', runner)
        self.assertIn('adapter-step-300', runner)
        self.assertNotIn('add_local_file("modal_sft_candidate.py"', runner)
        self.assertIn('Path("/repo/modal_sft_candidate_v3.py")', runner)

    def test_images_exclude_holdouts_and_training_excludes_development(self):
        tree = ast.parse(protocol.RUNNER.read_text(encoding="utf-8"))
        training = protocol._image_local_paths(tree, "training_image")
        development = protocol._image_local_paths(tree, "development_image")
        self.assertFalse(any("development" in path.casefold() for path in training))
        self.assertFalse(any("holdout" in path.casefold() for path in training + development))
        self.assertIn("evals/model_proposal_development_v1.json", development)

    def test_missing_local_or_remote_guard_is_rejected(self):
        original = protocol.RUNNER.read_text(encoding="utf-8")
        variants = (
            (
                original.replace('if manifest.get("training_authorized") is not True:', "if False:", 1),
                "Remote training authorization",
            ),
            (
                original.replace('if not protocol["training_authorized"]:', "if False:", 1),
                "Local training authorization",
            ),
        )
        for runner, message in variants:
            with self.subTest(message=message):
                with self.assertRaisesRegex(protocol.CandidateProtocolError, message):
                    protocol.validate_protocol(runner_path=self.write_runner(runner))

    def test_resource_or_manifest_lock_drift_is_rejected(self):
        original = protocol.RUNNER.read_text(encoding="utf-8")
        runner = original.replace("memory=16384", "memory=8192", 1)
        with self.assertRaisesRegex(protocol.CandidateProtocolError, "Train Modal resources"):
            protocol.validate_protocol(runner_path=self.write_runner(runner))

        runner = original.replace(protocol.EXPECTED_MANIFEST_SHA256, "0" * 64, 1)
        with self.assertRaisesRegex(protocol.CandidateProtocolError, "Runner manifest lock"):
            protocol.validate_protocol(runner_path=self.write_runner(runner))


if __name__ == "__main__":
    unittest.main()
