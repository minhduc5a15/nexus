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
