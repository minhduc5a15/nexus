import contextlib
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from nexus.agent.client import build_model_request, run_turn
from nexus.agent.routing import ToolRoutingMode
from nexus.agent.tools import TOOL_DEFINITIONS
from scripts import eval_proposals as evaluator


def response(name=None, arguments=None, reply=None):
    message = {"role": "assistant", "content": reply}
    if name is not None:
        message["tool_calls"] = [{"id": "test", "type": "function", "function": {
            "name": name, "arguments": json.dumps(arguments, ensure_ascii=False),
        }}]
    return {"choices": [{"finish_reason": "tool_calls" if name else "stop", "message": message}]}


class ProposalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _, cases = evaluator.load_cases(evaluator.DATASET_PATH)
        cls.cases = {case["id"]: case for case in cases}
        cls.fixtures = json.loads(evaluator.FIXTURES_PATH.read_text())

    def score(self, id, name=None, arguments=None, reply=None):
        return evaluator.score_proposal(self.cases[id], response(name, arguments, reply))

    def test_pilot_covers_seven_actions_and_independent_reviewable_labels(self):
        self.assertEqual(len(self.cases), 24)
        actions = [c["expected"]["action"] for c in self.cases.values() if c["expected"]["behavior"] == "call"]
        self.assertEqual(set(actions), set(evaluator.ACTION_TO_TOOL))
        self.assertTrue(all(actions.count(action) == 2 for action in set(actions)))
        self.assertTrue(all(c["family"] and c["rationale"] for c in self.cases.values()))

    def test_synthetic_pilot_has_exactly_six_known_failures_without_execution(self):
        original = deepcopy(self.fixtures)
        with (
            patch("sqlite3.connect", side_effect=AssertionError("SQLite must never open")) as connect,
            patch("nexus.agent.client.execute_tool", side_effect=AssertionError("no execution")) as execute,
            patch("nexus.agent.tools.execute_tool", side_effect=AssertionError("no dispatcher")) as dispatch,
            patch("nexus.agent.client.policy_for_tool", side_effect=AssertionError("no authorization")) as policy,
        ):
            results = [evaluator.evaluate_case(case, lambda _, id=id: self.fixtures["responses"][id])
                       for id, case in self.cases.items()]
        for mocked in (connect, execute, dispatch, policy):
            mocked.assert_not_called()
        failed = [r["id"] for r in results if r["automatic_status"] == "fail"]
        self.assertEqual(failed, self.fixtures["expected_automatic_failures"])
        self.assertEqual(self.fixtures, original)
        summary = evaluator.summarize(results)
        self.assertEqual(summary["automatic"], {"pass": 18, "fail": 6, "error": 0})
        self.assertEqual(summary["reply_review_pending"], 24)
        self.assertEqual(summary["policy_status"], "not_evaluated")
        self.assertEqual(summary["system_action_status"], "not_evaluated")

    def test_requests_match_runtime_in_both_modes_without_database(self):
        settings = {"model": "sentinel", "temperature": 0.0, "seed": 17}
        for mode in ToolRoutingMode:
            with self.subTest(mode=mode), patch("sqlite3.connect", side_effect=AssertionError):
                runtime = Mock(return_value=response(reply="No tool"))
                run_turn(Path("must-not-exist.db"), "Xem danh sách", runtime,
                         prompt_version="v13", settings=settings, tool_routing=mode)
                proposal = Mock(return_value=response("list_tasks", {}))
                result = evaluator.evaluate_case(self.cases["list_direct"], proposal,
                                                  settings=settings, tool_routing=mode)
                # Compare identical user input, not two different phrasings.
                expected, _ = build_model_request(self.cases["list_direct"]["prompt"],
                                                   prompt_version="v13", settings=settings, tool_routing=mode)
                self.assertEqual(proposal.call_args.args[0], expected)
                first = runtime.call_args.args[0]
                first["messages"][-1]["content"] = self.cases["list_direct"]["prompt"]
                self.assertEqual(first, expected)
                self.assertEqual(result["model_request"], expected)
                self.assertEqual(result["routing"]["mode"], mode.value)

    def test_payloads_do_not_share_mutable_schema_or_settings(self):
        definitions = deepcopy(TOOL_DEFINITIONS)
        settings = {"chat_template_kwargs": {"enable_thinking": False}}
        payload, _ = build_model_request("Xem danh sách", settings=settings)
        payload["tools"][0]["function"]["name"] = "mutated"
        payload["chat_template_kwargs"]["enable_thinking"] = True
        self.assertEqual(TOOL_DEFINITIONS, definitions)
        self.assertFalse(settings["chat_template_kwargs"]["enable_thinking"])

    def test_original_request_response_and_arguments_are_preserved(self):
        raw = response("create_task", {"content": "kiểm tra hóa đơn."})
        raw["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = '{ "content" : "kiểm tra hóa đơn." }'
        saved = deepcopy(raw)
        def generate(payload):
            payload["messages"].clear()
            return raw
        result = evaluator.evaluate_case(self.cases["create_literal"], generate)
        self.assertEqual(raw, saved)
        self.assertEqual(result["raw_response"], saved)
        self.assertTrue(result["model_request"]["messages"])
        self.assertEqual(result["proposed_calls"][0]["arguments"], {"content": "kiểm tra hóa đơn."})

    def test_classifier_does_not_determine_gold_or_score(self):
        case = self.cases["list_paraphrase"]
        with patch("scripts.eval_proposals.build_model_request", return_value=(
            {"messages": [], "tools": []},
            {"mode": "classified", "request_kind": "create", "tools": ["create_task"], "fallback": False},
        )):
            result = evaluator.evaluate_case(case, lambda _: response("list_tasks", {}))
        self.assertEqual(result["automatic_status"], "pass")
        self.assertEqual(result["outside_route_calls"], [{"name": "list_tasks", "arguments": {}}])

    def test_tool_miss_is_separate_from_wrong_tool(self):
        missing = self.score("edit_literal", reply="Đã sửa.")
        wrong = self.score("edit_literal", "create_task", {"content": "Nộp bản cuối."})
        self.assertIn("missing_call", missing["error_tags"])
        self.assertNotIn("id_mismatch", missing["error_tags"])
        self.assertIn("id_unavailable", missing["error_tags"])
        self.assertIn("wrong_tool", wrong["error_tags"])
        self.assertFalse(missing["checks"]["fields"]["id"])

    def test_boolean_float_and_string_ids_never_equal_gold_integer(self):
        case = deepcopy(self.cases["complete_id"])
        case["expected"]["accepted_calls"][0]["arguments"]["id"] = 1
        for value in (True, 1.0, "1", 0, -1, None):
            with self.subTest(value=value):
                result = evaluator.score_proposal(case, response("complete_task", {"id": value}))
                self.assertFalse(result["checks"]["fields"]["id"])
                self.assertFalse(result["checks"]["exact_proposal"])
                self.assertIn("invalid_argument:id", result["error_tags"])

    def test_schema_flags_extra_missing_and_invalid_enum(self):
        extra = self.score("delete_direct", "delete_task", {"id": 3, "confirmed": True})
        missing = self.score("edit_literal", "update_task", {"id": 4})
        enum = self.score("due_today", "list_tasks_by_deadline", {"scope": "next_week"})
        self.assertIn("extra_argument:confirmed", extra["error_tags"])
        self.assertIn("missing_argument:content", missing["error_tags"])
        self.assertIn("invalid_enum:scope", enum["error_tags"])

    def test_punctuation_truncation_and_case_changes_are_visible(self):
        punctuation = self.score("create_literal", "create_task", {"content": "kiểm tra hóa đơn"})
        casing = self.score("create_literal", "create_task", {"content": "Kiểm tra hóa đơn."})
        self.assertIn("content_truncated", punctuation["error_tags"])
        self.assertIn("content_punctuation_missing", punctuation["error_tags"])
        self.assertIn("content_not_verbatim", casing["error_tags"])

    def test_missing_reordered_and_artificial_newlines_fail_exact_content(self):
        expected = self.cases["create_multiline"]["expected"]["accepted_calls"][0]["arguments"]["content"]
        for text, tag in (
            (expected.splitlines()[0], "content_line_count_changed"),
            ("\n".join(reversed(expected.splitlines())), "content_lines_reordered"),
            (expected.replace("gọi mẹ", "gọi\nmẹ"), "content_line_count_changed"),
        ):
            with self.subTest(text=text):
                result = self.score("create_multiline", "create_task", {"content": text})
                self.assertFalse(result["checks"]["exact_proposal"])
                self.assertIn(tag, result["error_tags"])

    def test_rewritten_time_and_changed_scope_have_distinct_field_metrics(self):
        time = self.score("deadline_absolute", "set_task_deadline", {"id": 5, "when": "08:00 ngày 27/09/2026"})
        scope = self.score("due_today", "list_tasks_by_deadline", {"scope": "tomorrow"})
        self.assertTrue(time["checks"]["fields"]["id"])
        self.assertFalse(time["checks"]["fields"]["when"])
        self.assertIn("when_not_verbatim", time["error_tags"])
        self.assertIn("scope_mismatch", scope["error_tags"])

    def test_whole_alternatives_cannot_be_combined_field_by_field(self):
        case = deepcopy(self.cases["edit_literal"])
        case["expected"]["accepted_calls"].append({"name": "update_task", "arguments": {"id": 8, "content": "khác"}})
        result = evaluator.score_proposal(case, response("update_task", {"id": 4, "content": "khác"}))
        self.assertTrue(result["checks"]["fields"]["id"])
        self.assertTrue(result["checks"]["fields"]["content"])
        self.assertFalse(result["checks"]["exact_proposal"])

    def test_multiple_calls_remain_visible_and_never_pass(self):
        raw = response("list_tasks", {})
        raw["choices"][0]["message"]["tool_calls"] *= 2
        result = evaluator.score_proposal(self.cases["list_direct"], raw)
        self.assertEqual(len(result["proposed_calls"]), 2)
        self.assertIn("multiple_calls", result["error_tags"])
        self.assertEqual(result["automatic_status"], "fail")

    def test_bad_json_is_not_repaired_and_duplicate_keys_are_rejected(self):
        for text in ('{"id":7,}', '{"id":1,"id":7}', '{"id":NaN}', '```json\n{"id":7}\n```'):
            with self.subTest(text=text):
                raw = response("complete_task", {})
                raw["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = text
                result = evaluator.score_proposal(self.cases["complete_id"], raw)
                self.assertIn("invalid_arguments_json", result["error_tags"])
                self.assertEqual(result["proposed_calls"][0]["arguments"], text)

    def test_dict_arguments_are_not_silently_accepted_as_json_text(self):
        raw = response("complete_task", {"id": 7})
        raw["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = {"id": 7}
        result = evaluator.score_proposal(self.cases["complete_id"], raw)
        self.assertIn("invalid_arguments_encoding", result["error_tags"])
        self.assertFalse(result["checks"]["exact_proposal"])

    def test_malformed_envelopes_fail_without_crashing(self):
        variants = [None, [], {}, {"choices": [None]}, {"choices": [{"message": []}]}]
        for raw_calls in ({"function": {}}, "bad", [None], [{"id": "", "type": "function", "function": {"name": "list_tasks", "arguments": "{}"}}]):
            raw = response("list_tasks", {})
            raw["choices"][0]["message"]["tool_calls"] = raw_calls
            variants.append(raw)
        for raw in variants:
            with self.subTest(raw=raw):
                result = evaluator.score_proposal(self.cases["list_direct"], raw)
                self.assertEqual(result["automatic_status"], "fail")

    def test_truncated_or_empty_tool_finish_cannot_pass_no_action(self):
        for finish in ("length", "tool_calls"):
            raw = response(reply="Được")
            raw["choices"][0]["finish_reason"] = finish
            result = evaluator.score_proposal(self.cases["bare_statement"], raw)
            self.assertEqual(result["automatic_status"], "fail")

    def test_unknown_tool_is_diagnostic_not_dispatched(self):
        result = self.score("list_direct", "invented_tool", {})
        self.assertIn("unknown_tool", result["error_tags"])

    def test_calls_during_clarification_negation_and_unsupported_requests_fail(self):
        for case_id, tag in (
            ("missing_complete_id", "call_during_clarification"),
            ("negated_delete", "call_without_authorization"),
            ("unsupported_reminder", "call_without_authorization"),
        ):
            with self.subTest(case_id=case_id):
                result = self.score(case_id, "complete_task", {"id": 1})
                self.assertIn("unexpected_call", result["error_tags"])
                self.assertIn(tag, result["error_tags"])

    def test_no_call_success_claim_never_counts_as_finished_manual_review(self):
        result = self.score("missing_create", reply="Đã lưu xong!")
        self.assertEqual(result["automatic_status"], "pass")
        self.assertEqual(result["reply_review"]["status"], "pending")
        self.assertIn("content", " ".join(result["reply_review"]["rubric"]))
        self.assertEqual(result["system_action_status"], "not_evaluated")

    def test_transport_failure_is_separate_and_never_retried(self):
        generate = Mock(side_effect=OSError("connection refused"))
        failed = evaluator.evaluate_case(self.cases["list_direct"], generate)
        good = evaluator.evaluate_case(self.cases["list_direct"], lambda _: response("list_tasks", {}))
        generate.assert_called_once()
        self.assertEqual(failed["automatic_status"], "error")
        self.assertIsNone(failed["raw_response"])
        self.assertEqual(failed["error_tags"], [])
        summary = evaluator.summarize([failed, good])
        self.assertEqual(summary["accuracy_denominator"], 1)
        self.assertEqual(summary["accuracy"], 1.0)
        self.assertEqual(summary["automatic"]["error"], 1)

    def test_bad_gold_labels_and_duplicate_ids_fail_before_generating(self):
        case = deepcopy(self.cases["complete_id"])
        case["expected"]["accepted_calls"][0]["arguments"]["id"] = True
        generate = Mock()
        with self.assertRaises(ValueError):
            evaluator.evaluate_case(case, generate)
        generate.assert_not_called()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps({"schema_version": 1, "cases": [self.cases["list_direct"]] * 2}))
            with self.assertRaisesRegex(ValueError, "Duplicate case"):
                evaluator.load_cases(path)

    def test_live_cli_requires_provenance_before_any_network_request(self):
        with (
            patch("sys.argv", ["eval_proposals", "--mode", "live", "--output", "unused.json"]),
            patch("scripts.eval_proposals.send_chat") as send,
            contextlib.redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit) as stopped,
        ):
            evaluator.main()
        self.assertEqual(stopped.exception.code, 2)
        send.assert_not_called()

    def test_malformed_dataset_objects_are_rejected_cleanly(self):
        for value in (None, [], {"expected": []}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                evaluator.validate_case(value)

    def test_cli_records_raw_data_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            argv = ["eval_proposals", "--output", str(output)]
            with patch("sys.argv", argv), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(evaluator.main(), 1)  # six intentionally wrong fixtures
            report = json.loads(output.read_text())
            self.assertEqual(report["mode"], "scripted")
            self.assertEqual(len(report["cases"]), 24)
            self.assertTrue(report["dataset"]["sha256"])
            self.assertTrue(report["provenance"]["commit"])
            self.assertTrue(report["provenance"]["code_sha256"])
            self.assertIsNone(report["model_sha256"])
            saved = output.read_bytes()
            with patch("sys.argv", argv), self.assertRaises(FileExistsError):
                evaluator.main()
            self.assertEqual(output.read_bytes(), saved)


if __name__ == "__main__":
    unittest.main()
