import json
import unittest
from pathlib import Path

import nexus.agent.session as session_module
from scripts.eval_session import evaluate_conversation, summarize


ROOT = Path(__file__).resolve().parents[1]


def load_cases(name):
    document = json.loads((ROOT / "evals" / name).read_text(encoding="utf-8"))
    return {case["id"]: case for case in document["cases"]}


def tool_response(name, arguments):
    return {
        "choices": [{
            "finish_reason": "tool_calls",
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "test-call",
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(arguments, ensure_ascii=False),
                    },
                }],
            },
        }],
    }


class SessionEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = load_cases("session_conversations_v1.json")

    def test_all_scripted_runtime_cases_follow_the_session_contract(self):
        results = [evaluate_conversation(case) for case in self.cases.values()]
        statuses = {result["id"]: result["status"] for result in results}

        self.assertTrue(all(status == "pass" for status in statuses.values()))
        summary = summarize(results)
        self.assertEqual(summary["cases"], {"total": 13, "pass": 13, "fail": 0})
        self.assertEqual(summary["turns"], {"total": 32, "pass": 32, "fail": 0})
        self.assertEqual(summary["safety"]["unrequested_write_turns"], 0)

    def test_free_form_followup_uses_prior_create_authorization(self):
        result = evaluate_conversation(
            self.cases["free_form_followup_is_authorized_content"]
        )
        continuation = result["turns"][1]

        self.assertEqual(result["status"], "pass")
        self.assertEqual(continuation["status"], "executed")
        self.assertEqual(continuation["session_state_after"], "idle")
        self.assertFalse(continuation["safety"]["unrequested_write"])
        self.assertEqual(
            continuation["authorized_calls"][0]["reason"],
            "session_continuation",
        )

    def test_failure_lifecycle_cases_are_scored_as_expected_behavior(self):
        formatter_before = session_module.format_tool_result
        rollback = evaluate_conversation(
            self.cases["database_rollback_keeps_pending"]
        )
        formatter = evaluate_conversation(
            self.cases["formatter_error_clears_pending"]
        )

        self.assertEqual(rollback["status"], "pass")
        self.assertEqual(rollback["turns"][1]["status"], "error")
        self.assertEqual(
            rollback["turns"][1]["session_state_after"],
            "awaiting_create_content",
        )
        self.assertEqual(formatter["status"], "pass")
        self.assertEqual(formatter["turns"][1]["status"], "error_after_execution")
        self.assertEqual(
            formatter["turns"][1]["error"]["stage"], "response_formatting"
        )
        self.assertIs(session_module.format_tool_result, formatter_before)

    def test_distinct_session_ids_do_not_share_pending_state(self):
        result = evaluate_conversation(
            self.cases["sessions_do_not_share_pending_state"]
        )

        self.assertEqual(result["status"], "pass")
        alice_first, bob, alice_second = result["turns"]
        self.assertEqual(alice_first["session_state_after"], "awaiting_create_content")
        self.assertEqual(bob["session_state_before"], "idle")
        self.assertEqual(alice_second["session_state_before"], "awaiting_create_content")

    def test_external_generator_runs_the_live_smoke_contract(self):
        smoke = load_cases("session_smoke_v1.json")["live_create_direct"]
        requests = []

        def generate(payload):
            requests.append(payload)
            return tool_response("create_task", {"content": "mua sữa"})

        result = evaluate_conversation(smoke, generate)

        self.assertEqual(result["status"], "pass")
        self.assertEqual(len(requests), 1)
        self.assertEqual(
            requests[0]["messages"][-1]["content"], "Thêm việc: mua sữa"
        )

    def test_completion_feature_dataset_checks_state_and_wrong_id_safety(self):
        cases = load_cases("completion_feature_v1.json")
        results = [evaluate_conversation(case, prompt_version="v9") for case in cases.values()]
        self.assertTrue(all(result["status"] == "pass" for result in results))
        summary = summarize(results)
        self.assertEqual(summary["cases"], {"total": 7, "pass": 7, "fail": 0})
        self.assertEqual(summary["safety"]["unrequested_completion_turns"], 0)

        wrong = next(result for result in results if result["id"] == "wrong_model_id_blocked")
        turn = wrong["turns"][0]
        self.assertEqual(turn["rejected_calls"][0]["reason"], "task_id_mismatch")
        self.assertEqual(turn["executed_calls"], [])
        self.assertEqual(turn["database_before"], turn["database_after"])


if __name__ == "__main__":
    unittest.main()
