import unittest
from scripts.eval_qwen import evaluate_case


def mock_response(message, reason="stop", usage=None, timings=None):
    return {
        "model": "qwen3-1.7b-q8_0",
        "choices": [{"finish_reason": reason, "message": message}],
        "usage": usage or {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        "timings": timings or {"prompt_per_second": 100.0, "predicted_per_second": 30.0},
    }


class LocalEvaluationTests(unittest.TestCase):
    def test_hallucinated_success_does_not_pass_without_tool_and_saved_task(self):
        case = {
            "id": "create",
            "prompt": "Thêm mua sữa",
            "initial_tasks": [],
            "expected_calls": [{"name": "create_task", "arguments": {"content": "mua sữa"}}],
            "expected_tasks": ["mua sữa"],
            "reply_expectation": "Xác nhận",
        }
        result = evaluate_case(case, lambda _: mock_response({"role": "assistant", "content": "Đã thêm mua sữa."}))
        self.assertEqual(result["status"], "fail")
        self.assertFalse(result["checks"]["calls_match"])
        self.assertFalse(result["checks"]["tasks_match"])

    def test_unwanted_valid_write_is_detected(self):
        case = {
            "id": "greeting",
            "prompt": "Chào",
            "initial_tasks": ["giữ nguyên"],
            "expected_calls": [],
            "expected_tasks": ["giữ nguyên"],
            "reply_expectation": "Chào",
        }
        responses = iter([
            mock_response({
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "type": "function",
                    "id": "call-1",
                    "function": {"name": "create_task", "arguments": '{"content": "Chào"}'},
                }],
            }, reason="tool_calls"),
            mock_response({"role": "assistant", "content": "Chào bạn"}),
        ])
        result = evaluate_case(case, lambda _: next(responses))
        self.assertEqual(result["status"], "fail")
        self.assertEqual(len(result["database_after"]), 2)

    def test_server_error_is_distinct_from_model_behavior_failure(self):
        case = {
            "id": "greeting",
            "prompt": "Chào",
            "initial_tasks": [],
            "expected_calls": [],
            "expected_tasks": [],
            "reply_expectation": "Chào",
        }
        def unavailable(_):
            raise RuntimeError("Connection refused")

        result = evaluate_case(case, unavailable)
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["api_requests"], 1)
        self.assertIsNone(result["reply"])

    def test_successful_turn_records_diagnostics_and_passes(self):
        case = {
            "id": "create_one",
            "prompt": "Thêm mua sữa",
            "initial_tasks": [],
            "expected_calls": [{"name": "create_task", "arguments": {"content": "mua sữa"}}],
            "expected_tasks": ["mua sữa"],
            "reply_expectation": "Xác nhận",
        }
        responses = iter([
            mock_response({
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "type": "function",
                    "id": "call-1",
                    "function": {"name": "create_task", "arguments": '{"content": "mua sữa"}'},
                }],
            }, reason="tool_calls", timings={"predicted_per_second": 42.5}),
            mock_response({"role": "assistant", "content": "Đã thêm mua sữa."}, timings={"predicted_per_second": 38.0}),
        ])
        result = evaluate_case(case, lambda _: next(responses))
        self.assertEqual(result["status"], "pass")
        self.assertTrue(result["checks"]["calls_match"])
        self.assertTrue(result["checks"]["tasks_match"])
        self.assertEqual(len(result["response_diagnostics"]), 2)
        self.assertEqual(result["response_diagnostics"][0]["timings"]["predicted_per_second"], 42.5)


if __name__ == "__main__":
    unittest.main()

