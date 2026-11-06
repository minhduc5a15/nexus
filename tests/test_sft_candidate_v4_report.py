"""Recheck downloaded v4 artifacts from their raw data, without a model."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from scripts.audit_sft_candidate_v4_report import (
    directory_sha256,
    sha256,
    validate_development_report,
    validate_train_report,
)
from scripts.audit_sft_candidate_v4_execution import PLAN
from scripts.eval_proposals import load_cases
from scripts.score_sft_generation import score_proposal_text, summarize_proposal_texts


ROOT = Path(__file__).resolve().parents[1]


class SftCandidateV4ReportTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.plan = json.loads(PLAN.read_text(encoding="utf-8"))

    def write(self, path: Path, data: dict) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        return path

    def train_fixture(self) -> tuple[Path, dict]:
        output = self.root / self.plan["run_ids"]["train"] / "nexus-sft-candidate-v4"
        adapter = output / "adapter-step-320"
        adapter.mkdir(parents=True)
        (adapter / "adapter_model.safetensors").write_bytes(b"adapter fixture")
        report = {
            "protocol_id": "nexus-sft-candidate-v4",
            "run_id": self.plan["run_ids"]["train"],
            "stage": "train",
            "plan_sha256": sha256(PLAN),
            "dataset_sha256": self.plan["dataset"]["sha256"],
            "model": {key: self.plan["model"][key] for key in ("id", "revision")},
            "state": {"stage": "train", "status": "complete"},
            "training": {"completed_steps": 320, "loss": 0.2, "history": [{"step": 320, "loss": 0.2}]},
            "adapter": {
                "state_sha256_before": "a" * 64,
                "state_sha256_after": "b" * 64,
                "state_sha256_reloaded": "b" * 64,
                "directory_sha256": directory_sha256(adapter),
            },
            "resume": {
                "source_run_id": self.plan["resume"]["source_run_id"],
                "checkpoint_step": self.plan["resume"]["checkpoint_step"],
                "trainer_state_sha256": self.plan["resume"]["trainer_state_sha256"],
            },
            "final_holdout_inference": False,
        }
        return self.write(output / "train.json", report), report

    def development_fixture(self, train_path: Path) -> tuple[Path, dict]:
        _, cases = load_cases(ROOT / self.plan["development"]["path"])
        rows = []
        for case in cases:
            accepted = case["expected"]["accepted_calls"]
            generated = (
                '<tool_call>' + json.dumps(accepted[0], ensure_ascii=False) + '</tool_call>'
                if accepted else "Bạn muốn cung cấp thêm thông tin?"
            )
            row = score_proposal_text(case, generated)
            row.update(prompt_tokens=1300, generated_tokens=12, generation_seconds=0.1)
            rows.append(row)
        model_result = {"cases": rows, "summary": summarize_proposal_texts(rows)}
        report = {
            "protocol_id": "nexus-sft-candidate-v4",
            "run_id": self.plan["run_ids"]["development"],
            "stage": "development",
            "plan_sha256": sha256(PLAN),
            "dataset_sha256": self.plan["development"]["sha256"],
            "candidate_train_report_sha256": sha256(train_path),
            "source_v3_train_report_sha256": self.plan["source_v3"]["train_report_sha256"],
            "generation": self.plan["generation"],
            "state": {"stage": "development", "status": "complete"},
            "no_tool_execution": True,
            "final_holdout_inference": False,
            "models": {name: deepcopy(model_result) for name in ("base", "v3-step-300", "v4-step-320")},
        }
        output = self.root / self.plan["run_ids"]["development"] / "nexus-sft-candidate-v4"
        return self.write(output / "development.json", report), report

    def test_train_report_checks_adapter_bytes_and_completed_steps(self) -> None:
        path, report = self.train_fixture()
        self.assertEqual(validate_train_report(path)["completed_steps"], 320)
        changed = deepcopy(report)
        changed["training"]["completed_steps"] = 319
        self.write(path, changed)
        with self.assertRaisesRegex(ValueError, "optimizer step"):
            validate_train_report(path)
        self.write(path, report)
        (path.parent / "adapter-step-320" / "adapter_model.safetensors").write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "Adapter directory hash"):
            validate_train_report(path)

    def test_development_report_replays_each_raw_response_and_summary(self) -> None:
        train_path, _ = self.train_fixture()
        path, report = self.development_fixture(train_path)
        result = validate_development_report(path, train_path)
        self.assertEqual(result["scores"], {name: 40 for name in report["models"]})
        changed = deepcopy(report)
        changed["models"]["base"]["cases"][0]["generated_text"] = "<tool_call>{invalid}</tool_call>"
        self.write(path, changed)
        with self.assertRaisesRegex(ValueError, "scoring changed"):
            validate_development_report(path, train_path)
        changed = deepcopy(report)
        changed["models"]["v4-step-320"]["cases"].pop()
        self.write(path, changed)
        with self.assertRaisesRegex(ValueError, "case order changed"):
            validate_development_report(path, train_path)


if __name__ == "__main__":
    unittest.main()
