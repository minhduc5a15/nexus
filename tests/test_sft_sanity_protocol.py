from copy import deepcopy
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from scripts.sft_sanity_protocol import (
    SftProtocolError,
    file_sha256,
    load_json,
    validate_overfit_report,
    validate_probe_gate,
    validate_protocol,
)


ROOT = Path(__file__).resolve().parents[1]


class SftSanityProtocolTests(unittest.TestCase):
    def copy_protocol(self, destination: Path) -> None:
        for relative in (
            "evals/sft_sanity_v1.json",
            "evals/sft_sanity_manifest_v1.json",
            "evals/sft_sanity_training_manifest_v1.json",
            "modal_sft_sanity.py",
        ):
            source = ROOT / relative
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)

    def test_tracked_protocol_defaults_to_local_preflight_and_excludes_evals(self):
        summary = validate_protocol()
        self.assertEqual(summary["dataset"]["count"], 16)
        self.assertEqual(summary["gpu"], "T4")
        self.assertEqual(summary["runner"]["default_stage"], "preflight")
        self.assertEqual(summary["maximum_planned_gpu_seconds"], 3600)
        self.assertFalse(summary["final_holdout_inference"])

    def test_dataset_change_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.copy_protocol(root)
            dataset = root / "evals/sft_sanity_v1.json"
            dataset.write_bytes(dataset.read_bytes() + b"\n")
            with self.assertRaisesRegex(SftProtocolError, "dataset hash mismatch"):
                validate_protocol(root=root)

    def test_gpu_or_step_change_cannot_drift_from_manifest(self):
        for before, after, message in (
            ('gpu="T4"', 'gpu="A10G"', "Probe decorator differs"),
            ("MAX_STEPS = 64", "MAX_STEPS = 65", "Runner constant differs"),
            ("fp16=False", "fp16=True", "SFTTrainer precision"),
        ):
            with self.subTest(after=after), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                self.copy_protocol(root)
                runner = root / "modal_sft_sanity.py"
                runner.write_text(
                    runner.read_text(encoding="utf-8").replace(before, after, 1),
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(SftProtocolError, message):
                    validate_protocol(root=root)

    def test_runner_cannot_include_final_holdout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.copy_protocol(root)
            runner = root / "modal_sft_sanity.py"
            source = runner.read_text(encoding="utf-8")
            source = source.replace(
                '.add_local_file("modal_sft_sanity.py", "/repo/modal_sft_sanity.py")',
                '.add_local_file("evals/model_proposal_holdout_v1.json", "/repo/holdout.json")',
                1,
            )
            runner.write_text(source, encoding="utf-8")
            with self.assertRaisesRegex(SftProtocolError, "must not include holdout"):
                validate_protocol(root=root)

    def passing_probe(self) -> tuple[dict, dict]:
        manifest = load_json(ROOT / "evals/sft_sanity_training_manifest_v1.json")
        gates = {name: True for name in manifest["gates"]["probe"]}
        report = {
            "protocol_id": manifest["protocol_id"],
            "run_id": "probe-1",
            "stage": "probe",
            "dataset_sha256": manifest["dataset"]["sha256"],
            "model": manifest["model"],
            "provenance": {
                "training_manifest_sha256": file_sha256(
                    ROOT / "evals/sft_sanity_training_manifest_v1.json"
                )
            },
            "configuration": {
                key: deepcopy(manifest[key])
                for key in ("runtime", "quantization", "adapter", "training")
            },
            "parameters": {"trainable": 123},
            "gradient_probe": {"loss": 1.25},
            "adapter": {"directory_sha256": "a" * 64},
            "gates": gates,
            "state": {"status": "complete"},
        }
        return report, manifest

    def test_probe_gate_requires_every_piece_of_gradient_evidence(self):
        report, manifest = self.passing_probe()
        self.assertEqual(
            validate_probe_gate(report, manifest, expected_run_id="probe-1"),
            {
                "run_id": "probe-1",
                "adapter_sha256": "a" * 64,
                "trainable_parameters": 123,
                "loss": 1.25,
            },
        )
        for gate in manifest["gates"]["probe"]:
            broken = deepcopy(report)
            broken["gates"][gate] = False
            with self.subTest(gate=gate), self.assertRaisesRegex(SftProtocolError, "Probe gates failed"):
                validate_probe_gate(broken, manifest, expected_run_id="probe-1")

    def test_probe_gate_rejects_another_dataset_or_incomplete_run(self):
        report, manifest = self.passing_probe()
        report["dataset_sha256"] = "0" * 64
        with self.assertRaisesRegex(SftProtocolError, "dataset mismatch"):
            validate_probe_gate(report, manifest, expected_run_id="probe-1")
        report, manifest = self.passing_probe()
        report["configuration"]["adapter"]["rank"] = 8
        with self.assertRaisesRegex(SftProtocolError, "configuration mismatch"):
            validate_probe_gate(report, manifest, expected_run_id="probe-1")
        report, manifest = self.passing_probe()
        report["state"]["status"] = "interrupted"
        with self.assertRaisesRegex(SftProtocolError, "not complete"):
            validate_probe_gate(report, manifest, expected_run_id="probe-1")

    def passing_overfit(self) -> tuple[dict, dict]:
        manifest = load_json(ROOT / "evals/sft_sanity_training_manifest_v1.json")
        report = {
            "protocol_id": manifest["protocol_id"],
            "run_id": "overfit-1",
            "stage": "overfit",
            "dataset_sha256": manifest["dataset"]["sha256"],
            "model": manifest["model"],
            "provenance": {
                "training_manifest_sha256": file_sha256(
                    ROOT / "evals/sft_sanity_training_manifest_v1.json"
                ),
                "final_holdout_inference": False,
            },
            "approved_probe": {"run_id": "probe-1"},
            "parameters": {
                "only_lora_trainable": True,
                "trainable_dtypes": ["torch.float32"],
            },
            "training": {
                "maximum_steps": 64,
                "completed_steps": 64,
                "training_loss": 0.5,
                "runtime_seconds": 10.0,
            },
            "teacher_forced_before": {
                "cases": 16,
                "supervised_tokens": 100,
                "mean_loss": 2.0,
                "token_accuracy": 0.5,
                "exact_sequences": 2,
            },
            "teacher_forced_after": {
                "cases": 16,
                "supervised_tokens": 100,
                "mean_loss": 0.2,
                "token_accuracy": 0.9,
                "exact_sequences": 12,
            },
            "adapter": {
                "state_sha256_before": "a" * 64,
                "state_sha256_after": "b" * 64,
                "state_sha256_reloaded": "b" * 64,
            },
            "gates": {name: True for name in manifest["gates"]["overfit"]},
            "state": {"status": "complete"},
        }
        return report, manifest

    def test_overfit_report_rechecks_metrics_and_reloaded_adapter(self):
        report, manifest = self.passing_overfit()
        result = validate_overfit_report(
            report,
            manifest,
            expected_run_id="overfit-1",
            expected_probe_run_id="probe-1",
        )
        self.assertEqual(result["steps"], 64)
        self.assertEqual(result["adapter_state_sha256"], "b" * 64)
        for mutation in ("loss", "reload", "holdout"):
            broken = deepcopy(report)
            if mutation == "loss":
                broken["teacher_forced_after"]["mean_loss"] = 3.0
            elif mutation == "reload":
                broken["adapter"]["state_sha256_reloaded"] = "c" * 64
            else:
                broken["provenance"]["final_holdout_inference"] = True
            with self.subTest(mutation=mutation), self.assertRaises(SftProtocolError):
                validate_overfit_report(
                    broken,
                    manifest,
                    expected_run_id="overfit-1",
                    expected_probe_run_id="probe-1",
                )


if __name__ == "__main__":
    unittest.main()
