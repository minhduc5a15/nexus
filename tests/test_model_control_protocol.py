import json
from pathlib import Path
import shutil
import tempfile
import unittest

from scripts.model_control_protocol import (
    ProtocolError,
    file_sha256,
    load_json,
    validate_canary_gate,
    validate_control_protocol,
    validate_development_report,
)


ROOT = Path(__file__).resolve().parents[1]


class ModelControlProtocolTests(unittest.TestCase):
    def copy_protocol(self, destination: Path) -> None:
        (destination / "evals").mkdir()
        for relative in (
            "evals/model_control_manifest_v1.json",
            "evals/model_control_canary_v1.json",
            "evals/model_proposal_development_v1.json",
            "evals/model_proposal_holdout_v1.json",
            "modal_control_benchmark.py",
        ):
            source = ROOT / relative
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)

    def test_tracked_protocol_is_frozen_and_excludes_holdout(self):
        summary = validate_control_protocol(root=ROOT)
        self.assertEqual(summary["datasets"]["canary"]["count"], 10)
        self.assertEqual(summary["datasets"]["development"]["count"], 80)
        self.assertFalse(summary["holdout_inference"])
        self.assertEqual(summary["runner"]["default_stage"], "preflight")
        self.assertEqual(summary["maximum_planned_gpu_seconds"], 4800)

    def test_dataset_byte_change_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.copy_protocol(root)
            canary = root / "evals/model_control_canary_v1.json"
            canary.write_bytes(canary.read_bytes() + b"\n")
            with self.assertRaisesRegex(ProtocolError, "canary dataset hash mismatch"):
                validate_control_protocol(root=root)

    def test_canary_case_cannot_diverge_from_development_even_with_new_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.copy_protocol(root)
            canary_path = root / "evals/model_control_canary_v1.json"
            canary = load_json(canary_path)
            canary["cases"][0]["prompt"] += " đổi"
            canary_path.write_text(json.dumps(canary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            manifest_path = root / "evals/model_control_manifest_v1.json"
            manifest = load_json(manifest_path)
            manifest["datasets"]["canary"]["sha256"] = file_sha256(canary_path)
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ProtocolError, "Canary case changed from development"):
                validate_control_protocol(root=root)

    def test_holdout_cannot_replace_development_input(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.copy_protocol(root)
            holdout = root / "evals/model_proposal_holdout_v1.json"
            manifest_path = root / "evals/model_control_manifest_v1.json"
            manifest = load_json(manifest_path)
            manifest["datasets"]["development"].update(
                path="evals/model_proposal_holdout_v1.json",
                sha256=file_sha256(holdout),
                count=120,
            )
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ProtocolError, "must not reference holdout"):
                validate_control_protocol(root=root)

    def test_runner_and_manifest_cannot_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.copy_protocol(root)
            runner = root / "modal_control_benchmark.py"
            runner.write_text(
                runner.read_text(encoding="utf-8").replace(
                    'MODEL_SHA256 = "66cbfc7624628ca4725a6a0581b95f2155aa6aa073a6ed4ff8d158794ee8fee3"',
                    'MODEL_SHA256 = "' + "0" * 64 + '"',
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ProtocolError, "Runner constant differs from manifest: MODEL_SHA256"):
                validate_control_protocol(root=root)

    def test_canary_gate_accepts_failures_but_rejects_generation_errors(self):
        manifest = load_json(ROOT / "evals/model_control_manifest_v1.json")
        ids = manifest["datasets"]["canary"]["case_ids"]
        report = {
            "protocol_id": manifest["protocol_id"],
            "run_id": "canary-1",
            "block": "control-canary",
            "prompt_version": "v13",
            "tool_routing": "all",
            "settings": manifest["runtime"]["settings"],
            "model_sha256": manifest["model"]["sha256"],
            "dataset": {
                "path": manifest["datasets"]["canary"]["path"],
                "sha256": manifest["datasets"]["canary"]["sha256"],
            },
            "cases": [
                {"id": case_id, "automatic_status": "fail" if index == 0 else "pass"}
                for index, case_id in enumerate(ids)
            ],
            "state": {"status": "complete"},
        }
        self.assertEqual(
            validate_canary_gate(report, manifest, expected_run_id="canary-1"),
            {"run_id": "canary-1", "cases": 10, "passed": 9, "failed": 1},
        )
        report["cases"][3]["automatic_status"] = "error"
        with self.assertRaisesRegex(ProtocolError, "generation error"):
            validate_canary_gate(report, manifest, expected_run_id="canary-1")

    def test_development_report_requires_frozen_cases_and_canary_gate(self):
        manifest = load_json(ROOT / "evals/model_control_manifest_v1.json")
        development = load_json(ROOT / manifest["datasets"]["development"]["path"])
        gate = {"run_id": "canary-1", "cases": 10, "passed": 3, "failed": 7}
        cases = [
            {"id": case["id"], "automatic_status": "pass"}
            for case in development["cases"]
        ]
        report = {
            "protocol_id": manifest["protocol_id"],
            "run_id": "development-1",
            "block": "control-development",
            "prompt_version": manifest["runtime"]["prompt_version"],
            "tool_routing": manifest["runtime"]["tool_routing"],
            "settings": manifest["runtime"]["settings"],
            "model_sha256": manifest["model"]["sha256"],
            "dataset": {
                "path": manifest["datasets"]["development"]["path"],
                "sha256": manifest["datasets"]["development"]["sha256"],
            },
            "approved_canary": gate,
            "cases": cases,
            "summary": {"automatic": {"pass": 80, "fail": 0, "error": 0}},
            "state": {"status": "complete", "scored_completed": 80},
        }
        self.assertEqual(
            validate_development_report(
                report,
                manifest,
                expected_run_id="development-1",
                expected_canary_gate=gate,
            ),
            {
                "run_id": "development-1",
                "cases": 80,
                "passed": 80,
                "failed": 0,
                "canary_run_id": "canary-1",
            },
        )

        report["cases"][0], report["cases"][1] = report["cases"][1], report["cases"][0]
        with self.assertRaisesRegex(ProtocolError, "cases or order changed"):
            validate_development_report(
                report,
                manifest,
                expected_run_id="development-1",
                expected_canary_gate=gate,
            )


if __name__ == "__main__":
    unittest.main()
