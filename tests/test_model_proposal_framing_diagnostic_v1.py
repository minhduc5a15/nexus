"""Offline checks for the paired framing diagnostic and frozen GPU request."""

from copy import deepcopy
import json
from pathlib import Path
import unittest

import modal_sft_framing_diagnostic_v1 as runner
from scripts.audit_model_proposal_framing_diagnostic_v1 import audit_dataset
from scripts.build_model_proposal_framing_diagnostic_v1 import build_document


ROOT = Path(__file__).resolve().parents[1]


class FramingDiagnosticTests(unittest.TestCase):
    def test_pairs_have_same_gold_call_and_valid_provenance(self):
        summary = audit_dataset()
        self.assertEqual((summary["cases"], summary["pairs"], summary["reused_exposed_failures"]), (20, 10, 5))
        cases = build_document()["cases"]
        for left, right in zip(cases[::2], cases[1::2]):
            self.assertEqual(left["expected"], right["expected"])
            self.assertEqual(left["pair_id"], right["pair_id"])

    def test_frozen_plan_rejects_changed_dataset_or_adapter(self):
        plan = json.loads((ROOT / "evals/model_proposal_framing_diagnostic_v1_run_plan.json").read_text(encoding="utf-8"))
        runner.validate_plan(plan, ROOT)
        altered = deepcopy(plan)
        altered["dataset"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "dataset changed"):
            runner.validate_plan(altered, ROOT)
        altered = deepcopy(plan)
        altered["source_adapter_state_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "Adapter source changed"):
            runner.validate_plan(altered, ROOT)

    def test_diagnostic_cannot_open_training_or_final_holdout(self):
        plan = json.loads((ROOT / "evals/model_proposal_framing_diagnostic_v1_run_plan.json").read_text(encoding="utf-8"))
        for key in ("training_authorized", "final_holdout_inference_authorized"):
            altered = deepcopy(plan)
            altered[key] = True
            with self.assertRaises(ValueError):
                runner.validate_plan(altered, ROOT)
        runner_source = (ROOT / "modal_sft_framing_diagnostic_v1.py").read_text(encoding="utf-8")
        self.assertNotIn('add_local_file("evals/model_proposal_holdout_v2.json"', runner_source)


if __name__ == "__main__":
    unittest.main()
