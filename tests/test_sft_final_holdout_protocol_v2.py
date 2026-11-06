from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from scripts import sft_final_holdout_protocol_v2 as protocol
from scripts.eval_proposals import load_cases
from scripts.score_sft_generation import summarize_proposal_texts


class FinalHoldoutProtocolV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = protocol.load_json(protocol.MANIFEST)
        _, cls.cases = load_cases(protocol.ROOT / cls.manifest["holdout"]["path"])

    def passing_rows(self):
        rows = []
        for case in self.cases:
            expected = case["expected"]
            rows.append(
                {
                    "id": case["id"],
                    "family": case["family"],
                    "behavior": expected["behavior"],
                    "action": expected["action"],
                    "prompt": case["prompt"],
                    "generated_text": "ok",
                    "parsed": {"calls": expected["accepted_calls"][:1], "reply": None, "errors": []},
                    "accepted_calls": expected["accepted_calls"],
                    "automatic_status": "pass",
                    "proposal_correct": True,
                    "error_tags": [],
                    "reply_review": {
                        "required": expected["behavior"] != "call",
                        "status": "pending" if expected["behavior"] != "call" else "not_required",
                    },
                    "generation_error": None,
                }
            )
        return rows

    def test_frozen_protocol_passes_without_opening_modal(self):
        summary = protocol.validate_protocol()
        self.assertTrue(summary["holdout_authorized"])
        self.assertEqual(summary["approved_labels"], 120)
        self.assertEqual(summary["configurations"], ["base", "v3-step-300", "v4-step-320"])
        self.assertLess(summary["maximum_compute_usd"], 0.7)
        self.assertFalse(summary["tools_or_database_executed"])

    def test_manifest_hash_mismatch_closes_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.json"
            value = deepcopy(self.manifest)
            value["holdout_authorized"] = False
            manifest.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(protocol.FinalHoldoutV2Error, "manifest hash"):
                protocol.validate_protocol(manifest_path=manifest)

    def test_candidate_must_not_regress_incumbent(self):
        passing = self.passing_rows()
        candidate = deepcopy(passing)
        action_index = next(index for index, row in enumerate(candidate) if row["behavior"] == "call")
        candidate[action_index]["automatic_status"] = "fail"
        candidate[action_index]["proposal_correct"] = False
        candidate[action_index]["error_tags"] = ["wrong_arguments"]
        models = {
            "base": {"cases": deepcopy(passing)},
            "v3-step-300": {"cases": deepcopy(passing)},
            "v4-step-320": {"cases": candidate},
        }
        decision = protocol.evaluate_acceptance(models, self.manifest["acceptance"])
        self.assertFalse(decision["automatic_eligible"])
        self.assertFalse(decision["gates"]["candidate_total_not_below_incumbent"])
        self.assertFalse(decision["gates"]["no_action_regression_vs_base_or_incumbent"])

    def test_generation_error_closes_gate(self):
        rows = self.passing_rows()
        broken = deepcopy(rows)
        broken[0]["automatic_status"] = "fail"
        broken[0]["error_tags"] = ["generation_error"]
        models = {
            "base": {"cases": deepcopy(rows)},
            "v3-step-300": {"cases": deepcopy(rows)},
            "v4-step-320": {"cases": broken},
        }
        decision = protocol.evaluate_acceptance(models, self.manifest["acceptance"])
        self.assertFalse(decision["gates"]["complete_without_generation_error"])
        self.assertFalse(decision["automatic_eligible"])

    def test_all_passing_rows_meet_automatic_gates_but_do_not_authorize_rollout(self):
        rows = self.passing_rows()
        models = {
            label: {"cases": deepcopy(rows), "summary": summarize_proposal_texts(rows)}
            for label in ("base", "v3-step-300", "v4-step-320")
        }
        decision = protocol.evaluate_acceptance(models, self.manifest["acceptance"])
        self.assertTrue(decision["automatic_eligible"])
        self.assertEqual(decision["manual_noncall_review"], "pending")
        self.assertFalse(decision["gguf_rollout_authorized"])


if __name__ == "__main__":
    unittest.main()
