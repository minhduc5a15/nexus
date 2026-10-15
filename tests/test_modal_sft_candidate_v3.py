from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import modal_sft_candidate_v3 as runner


class ModalSftCandidateV3RecoveryTests(unittest.TestCase):
    def output(self) -> Path:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "run"
        path.mkdir()
        return path

    def test_new_input_starts_without_checkpoint(self):
        state_path, state, checkpoint = runner._prepare_train_state(
            self.output(), modal_input_id="in-1", now="2026-10-14T00:00:00+00:00"
        )
        self.assertEqual(state_path.name, "train-state.json")
        self.assertEqual(state["modal_input_id"], "in-1")
        self.assertEqual(state["resume_count"], 0)
        self.assertIsNone(checkpoint)

    def test_same_input_resumes_latest_checkpoint(self):
        output = self.output()
        (output / "trainer-checkpoints/checkpoint-25").mkdir(parents=True)
        (output / "trainer-checkpoints/checkpoint-50").mkdir()
        (output / "trainer-checkpoints/checkpoint-25/.nexus-complete").write_text(
            "in-1\n", encoding="utf-8"
        )
        (output / "trainer-checkpoints/checkpoint-50/.nexus-complete").write_text(
            "in-1\n", encoding="utf-8"
        )
        (output / "trainer-checkpoints/checkpoint-75").mkdir()
        (output / "trainer-checkpoints/not-a-checkpoint").mkdir()
        (output / "train-state.json").write_text(
            json.dumps(
                {
                    "status": "interrupted",
                    "started_at": "start",
                    "modal_input_id": "in-1",
                    "resume_count": 0,
                    "error": {"type": "KeyboardInterrupt", "message": ""},
                    "finished_at": "finish",
                }
            ),
            encoding="utf-8",
        )
        _, state, checkpoint = runner._prepare_train_state(
            output, modal_input_id="in-1", now="resume"
        )
        self.assertEqual(checkpoint.name, "checkpoint-50")
        self.assertEqual(state["resume_count"], 1)
        self.assertEqual(state["last_recovery_checkpoint"], "checkpoint-50")
        self.assertEqual(state["interruptions"][0]["error"]["type"], "KeyboardInterrupt")

    def test_checkpoint_from_another_input_is_rejected(self):
        output = self.output()
        checkpoint = output / "trainer-checkpoints/checkpoint-25"
        checkpoint.mkdir(parents=True)
        (checkpoint / ".nexus-complete").write_text("in-2\n", encoding="utf-8")
        (output / "train-state.json").write_text(
            json.dumps({"status": "interrupted", "modal_input_id": "in-1"}),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(FileExistsError, "checkpoint belongs"):
            runner._prepare_train_state(output, modal_input_id="in-1")

    def test_different_input_cannot_reuse_state(self):
        output = self.output()
        (output / "train-state.json").write_text(
            json.dumps({"status": "interrupted", "modal_input_id": "in-1"}),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(FileExistsError, "different Modal input"):
            runner._prepare_train_state(output, modal_input_id="in-2")

    def test_completed_report_cannot_be_overwritten(self):
        output = self.output()
        (output / "train.json").write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(FileExistsError, "Completed candidate"):
            runner._prepare_train_state(output, modal_input_id="in-1")

    def checkpoint(self, output: Path, *, step: int = 25, input_id: str = "old-input"):
        checkpoint = output / f"checkpoint-{step}"
        checkpoint.mkdir(parents=True)
        for name in (
            "adapter_config.json",
            "adapter_model.safetensors",
            "optimizer.pt",
            "rng_state.pth",
            "scheduler.pt",
        ):
            (checkpoint / name).write_text(name, encoding="utf-8")
        (checkpoint / ".nexus-complete").write_text(input_id + "\n", encoding="utf-8")
        (checkpoint / "trainer_state.json").write_text(
            json.dumps({"global_step": step}), encoding="utf-8"
        )
        return checkpoint

    def test_locked_cross_run_checkpoint_can_bootstrap_new_input(self):
        volume = self.output()
        source_output = volume / "source" / "sft-candidate-v3"
        checkpoint = self.checkpoint(source_output / "trainer-checkpoints")
        original_output = runner._output
        self.addCleanup(setattr, runner, "_output", original_output)
        runner._output = lambda run_id: volume / run_id / "sft-candidate-v3"
        spec = {
            "source_run_id": "source",
            "source_modal_input_id": "old-input",
            "checkpoint_step": 25,
            "directory_sha256": runner._directory_sha256(checkpoint),
        }
        path, provenance = runner._validated_cross_run_resume_checkpoint(spec)
        self.assertEqual(path, checkpoint)
        target = volume / "target"
        target.mkdir()
        _, state, resume = runner._prepare_train_state(
            target,
            modal_input_id="new-input",
            bootstrap_checkpoint=path,
            bootstrap_provenance=provenance,
            now="now",
        )
        self.assertEqual(resume, checkpoint)
        self.assertEqual(state["bootstrap_resume"], provenance)
        self.assertEqual(state["resume_count"], 0)

    def test_cross_run_checkpoint_rejects_hash_step_marker_and_missing_state(self):
        volume = self.output()
        checkpoint = self.checkpoint(volume / "source/sft-candidate-v3/trainer-checkpoints")
        original_output = runner._output
        self.addCleanup(setattr, runner, "_output", original_output)
        runner._output = lambda run_id: volume / run_id / "sft-candidate-v3"
        base = {
            "source_run_id": "source",
            "source_modal_input_id": "old-input",
            "checkpoint_step": 25,
            "directory_sha256": runner._directory_sha256(checkpoint),
        }
        variants = (
            ({**base, "directory_sha256": "0" * 64}, "directory hash mismatch"),
            ({**base, "source_modal_input_id": "wrong"}, "input ID mismatch"),
            ({**base, "checkpoint_step": 24}, "incomplete"),
        )
        for spec, message in variants:
            with self.subTest(message=message):
                with self.assertRaisesRegex(RuntimeError, message):
                    runner._validated_cross_run_resume_checkpoint(spec)
        (checkpoint / "optimizer.pt").unlink()
        with self.assertRaisesRegex(RuntimeError, "incomplete"):
            runner._validated_cross_run_resume_checkpoint(base)

    def test_same_input_recovery_checkpoint_takes_precedence_over_bootstrap(self):
        output = self.output()
        local = self.checkpoint(output / "trainer-checkpoints", step=50, input_id="new")
        bootstrap = self.checkpoint(output / "bootstrap", step=25, input_id="old")
        (output / "train-state.json").write_text(
            json.dumps(
                {
                    "status": "interrupted",
                    "modal_input_id": "new",
                    "bootstrap_resume": {"checkpoint_step": 25},
                }
            ),
            encoding="utf-8",
        )
        _, state, resume = runner._prepare_train_state(
            output,
            modal_input_id="new",
            bootstrap_checkpoint=bootstrap,
            bootstrap_provenance={"checkpoint_step": 25},
        )
        self.assertEqual(resume, local)
        self.assertEqual(state["last_recovery_checkpoint"], "checkpoint-50")

    def test_development_same_input_resumes_and_records_interruption(self):
        output = self.output()
        (output / "development-state.json").write_text(
            json.dumps(
                {
                    "status": "interrupted",
                    "started_at": "start",
                    "modal_input_id": "in-1",
                    "resume_count": 0,
                    "error": {"type": "KeyboardInterrupt", "message": ""},
                    "finished_at": "finish",
                }
            ),
            encoding="utf-8",
        )
        state_path, state = runner._prepare_development_state(
            output, modal_input_id="in-1", now="resume"
        )
        self.assertEqual(state_path.name, "development-state.json")
        self.assertEqual(state["resume_count"], 1)
        self.assertEqual(state["resumed_at"], "resume")
        self.assertEqual(state["interruptions"][0]["error"]["type"], "KeyboardInterrupt")

    def test_development_different_input_cannot_reuse_state(self):
        output = self.output()
        (output / "development-state.json").write_text(
            json.dumps({"status": "interrupted", "modal_input_id": "in-1"}),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(FileExistsError, "different Modal input"):
            runner._prepare_development_state(output, modal_input_id="in-2")

    def test_development_partial_must_be_an_ordered_prefix(self):
        output = self.output()
        partial = output / "partial.json"
        cases = [{"id": "one"}, {"id": "two"}]
        partial.write_text(json.dumps([{"id": "one"}]), encoding="utf-8")
        self.assertEqual(
            runner._load_partial_development_rows(partial, cases), [{"id": "one"}]
        )
        partial.write_text(json.dumps([{"id": "two"}]), encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "case order changed"):
            runner._load_partial_development_rows(partial, cases)


if __name__ == "__main__":
    unittest.main()
