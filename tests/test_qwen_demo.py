import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from nexus.agent.client import PostToolExecutionError, run_probe, run_turn
from nexus.storage.sqlite_db import initialize_database, list_tasks


def response(message, reason="stop"):
    return {"choices": [{"finish_reason": reason, "message": message}]}


class LocalProbeTests(unittest.TestCase):
    def test_tool_result_is_sent_back_and_real_database_is_checked(self):
        requests = []
        responses = iter([
            response({"role": "assistant", "content": "Chào bạn!"}),
            response({"role": "assistant", "content": None, "tool_calls": [{
                "type": "function", "id": "local-1", "function": {
                    "name": "create_task", "arguments": '{"content":"mua sữa"}'
                }
            }]}, "tool_calls"),
            response({"role": "assistant", "content": "Đã thêm mua sữa."}),
        ])
        def generate(payload):
            requests.append(deepcopy(payload))
            return next(responses)
        result = run_probe(generate)
        self.assertTrue(result["tool_check_passed"])
        returned = requests[-1]["messages"][-1]
        self.assertEqual(returned["tool_call_id"], "local-1")
        self.assertEqual(json.loads(returned["content"])["tasks"][0], result["tool_round"]["database_after"][0])
        self.assertEqual(requests[-1]["tool_choice"], "none")

    def test_claimed_success_without_tool_does_not_pass(self):
        result = run_probe(lambda _: response({"role": "assistant", "content": "Đã thêm mua sữa."}))
        self.assertFalse(result["tool_check_passed"])
        self.assertEqual(result["tool_round"]["database_after"], [])

    def test_truncated_tool_call_is_never_executed(self):
        responses = iter([
            response({"role": "assistant", "content": "Chào"}),
            response({"role": "assistant", "content": None, "tool_calls": []}, "length"),
        ])
        with patch("nexus.agent.client.execute_tool") as execute:
            with self.assertRaises(RuntimeError):
                run_probe(lambda _: next(responses))
            execute.assert_not_called()

    def test_direct_reply_without_tools(self):
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "test.db"
            initialize_database(db_path)
            turn = run_turn(db_path, "Chào bạn!", lambda _: response({"role": "assistant", "content": "Chào bạn!"}))
            self.assertEqual(turn["calls"], [])
            self.assertEqual(turn["reply"], "Chào bạn!")

    def test_final_response_failure_exposes_the_committed_tool_result(self):
        tool_call = response({
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "type": "function",
                "id": "call-1",
                "function": {
                    "name": "create_task",
                    "arguments": '{"content":"mua sữa"}',
                },
            }],
        }, "tool_calls")
        responses = iter([tool_call, RuntimeError("server stopped")])

        def generate(_):
            value = next(responses)
            if isinstance(value, Exception):
                raise value
            return value

        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "test.db"
            initialize_database(db_path)
            with self.assertRaises(PostToolExecutionError) as raised:
                run_turn(db_path, "Thêm mua sữa", generate)

            error = raised.exception
            self.assertEqual(error.stage, "final_response")
            self.assertEqual(error.executed_calls[0]["name"], "create_task")
            self.assertEqual(error.executed_calls[0]["result"]["tasks"][0]["content"], "mua sữa")
            self.assertEqual([task.content for task in list_tasks(db_path)], ["mua sữa"])

    def test_later_tool_failure_exposes_only_the_completed_calls(self):
        model_reply = response({
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "type": "function",
                    "id": "call-1",
                    "function": {
                        "name": "create_task",
                        "arguments": '{"content":"mua sữa"}',
                    },
                },
                {
                    "type": "function",
                    "id": "call-2",
                    "function": {"name": "unknown_tool", "arguments": "{}"},
                },
            ],
        }, "tool_calls")

        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "test.db"
            initialize_database(db_path)
            with self.assertRaises(PostToolExecutionError) as raised:
                run_turn(db_path, "Thêm việc", lambda _: model_reply)

            error = raised.exception
            self.assertEqual(error.stage, "tool_execution")
            self.assertEqual(len(error.executed_calls), 1)
            self.assertEqual([task.content for task in list_tasks(db_path)], ["mua sữa"])

    def test_response_diagnostics_extracts_metrics(self):
        from nexus.agent.client import response_diagnostics

        raw = {
            "model": "qwen3-1.7b-q8_0",
            "choices": [{"finish_reason": "stop", "message": {"content": "ok"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            "timings": {"prompt_per_second": 200.0, "predicted_per_second": 35.0},
        }
        diag = response_diagnostics(raw)
        self.assertEqual(diag["finish_reason"], "stop")
        self.assertEqual(diag["model"], "qwen3-1.7b-q8_0")
        self.assertEqual(diag["usage"]["total_tokens"], 15)
        self.assertEqual(diag["timings"]["predicted_per_second"], 35.0)


if __name__ == "__main__":
    unittest.main()
