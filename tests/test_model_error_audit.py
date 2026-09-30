from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts import audit_model_errors as audit


class ModelErrorAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.value = json.loads(audit.AUDIT_PATH.read_text())

    def write_audit(self, value, directory):
        path = Path(directory) / "audit.json"
        path.write_text(json.dumps(value, ensure_ascii=False))
        return path

    def synthetic_sources(self, value=None):
        value = deepcopy(value or self.value)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name) / "reports"
        root.mkdir()
        filler_ids = [f"passing_{index:02d}" for index in range(61)]
        for name in audit.CONFIGURATIONS:
            spec = value["source_run"]["reports"][name]
            cases = []
            for item in value["cases"]:
                observation = item["observations"][name]
                cases.append({
                    "id": item["id"],
                    "family": item["family"],
                    "prompt": item["prompt"],
                    "expected": item["expected"],
                    **deepcopy(observation),
                })
            for case_id in filler_ids:
                cases.append({
                    "id": case_id,
                    "family": "synthetic_pass",
                    "prompt": case_id,
                    "expected": {"behavior": "no_action", "action": None, "accepted_calls": [], "missing_slots": []},
                    "automatic_status": "pass",
                    "proposed_calls": [],
                    "model_reply": "Không có thao tác.",
                    "error_tags": [],
                    "routing": {"mode": spec["routing"], "request_kind": "other", "tools": [], "fallback": False},
                })
            report = {
                "prompt_version": spec["prompt_version"],
                "tool_routing": spec["routing"],
                "model_sha256": spec["model_sha256"],
                "dataset": {"sha256": value["source_run"]["dataset_sha256"]},
                "summary": {"automatic": spec["automatic"]},
                "cases": cases,
            }
            path = root / f"{name}.json"
            path.write_text(json.dumps(report, ensure_ascii=False))
            spec["relative_path"] = path.name
            spec["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        value["source_run"]["reports_root"] = str(root)
        return value, root, temporary.name

    def test_frozen_audit_is_reproducible_without_ignored_modal_artifacts(self):
        value, reports_root, directory = self.synthetic_sources()
        summary = audit.validate_audit(self.write_audit(value, directory), reports_root)
        self.assertEqual(summary["unique_failed_cases"], 19)
        self.assertEqual(summary["failure_observations"], 61)
        self.assertEqual(
            summary["configuration_failures"],
            {"q4_all": 14, "q4_classified": 17, "q8_all": 13, "q8_classified": 17},
        )
        self.assertEqual(
            summary["training_disposition_counts"],
            {"architecture_only": 2, "sft_regression_guard": 3, "sft_target": 14},
        )
        self.assertFalse(self.value["source_run"]["holdout_inference"])
        self.assertNotIn("holdout", self.value["source_run"]["dataset"].casefold())

    def test_source_hash_and_raw_observation_changes_are_rejected(self):
        for mutation in ("hash", "observation"):
            with self.subTest(mutation=mutation):
                value, reports_root, directory = self.synthetic_sources()
                if mutation == "hash":
                    value["source_run"]["reports"]["q4_all"]["sha256"] = "0" * 64
                else:
                    value["cases"][0]["observations"]["q4_all"]["proposed_calls"] = []
                with self.assertRaisesRegex(ValueError, "hash mismatch|Observation differs"):
                    audit.validate_audit(self.write_audit(value, directory), reports_root)

    def test_missing_case_or_failure_attribution_is_rejected(self):
        for mutation in ("case", "attribution"):
            with self.subTest(mutation=mutation):
                value, reports_root, directory = self.synthetic_sources()
                if mutation == "case":
                    value["cases"].pop()
                else:
                    target = next(case for case in value["cases"] if case["failed_configurations"])
                    target["failure_attribution"].pop(target["failed_configurations"][0])
                with self.assertRaisesRegex(ValueError, "exactly cover|exactly one attribution"):
                    audit.validate_audit(self.write_audit(value, directory), reports_root)

    def test_wrong_exposure_failures_cannot_be_marked_as_sft_targets(self):
        value, reports_root, directory = self.synthetic_sources()
        target = next(case for case in value["cases"] if case["id"] == "dev_061")
        target["training_disposition"] = "sft_target"
        value["summary"] = audit.summarize(value["cases"])
        with self.assertRaisesRegex(ValueError, "Wrong-exposure case cannot drive SFT"):
            audit.validate_audit(self.write_audit(value, directory), reports_root)

    def test_passing_configuration_cannot_receive_failure_attribution(self):
        value, reports_root, directory = self.synthetic_sources()
        target = next(case for case in value["cases"] if case["id"] == "dev_043")
        target["failure_attribution"]["q4_classified"] = {
            "owner": "model",
            "primary_cause": "missed_action",
            "secondary_causes": [],
        }
        with self.assertRaisesRegex(ValueError, "exactly one attribution"):
            audit.validate_audit(self.write_audit(value, directory), reports_root)


if __name__ == "__main__":
    unittest.main()
