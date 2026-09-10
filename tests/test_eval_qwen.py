import json
import unittest
from scripts.eval_qwen import evaluate_case, inspect_reply, summarize


def mock_response(message, reason="stop", usage=None, timings=None):
    return {
        "model": "qwen3-1.7b-q8_0",
        "choices": [{"finish_reason": reason, "message": message}],
        "usage": usage
        or {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        "timings": timings
        or {"prompt_per_second": 100.0, "predicted_per_second": 30.0},
    }


class LocalEvaluationTests(unittest.TestCase):
    def test_hallucinated_success_does_not_pass_without_tool_and_saved_task(self):
        case = {
            "id": "create",
            "prompt": "Thêm mua sữa",
            "initial_tasks": [],
            "expected_calls": [
                {"name": "create_task", "arguments": {"content": "mua sữa"}}
            ],
            "expected_tasks": ["mua sữa"],
            "reply_expectation": "Xác nhận",
        }
        result = evaluate_case(
            case,
            lambda _: mock_response(
                {"role": "assistant", "content": "Đã thêm mua sữa."}
            ),
        )
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
        responses = iter(
            [
                mock_response(
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "type": "function",
                                "id": "call-1",
                                "function": {
                                    "name": "create_task",
                                    "arguments": '{"content": "Chào"}',
                                },
                            }
                        ],
                    },
                    reason="tool_calls",
                ),
                mock_response({"role": "assistant", "content": "Chào bạn"}),
            ]
        )
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
            "expected_calls": [
                {"name": "create_task", "arguments": {"content": "mua sữa"}}
            ],
            "expected_tasks": ["mua sữa"],
            "reply_expectation": "Xác nhận",
        }
        responses = iter(
            [
                mock_response(
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "type": "function",
                                "id": "call-1",
                                "function": {
                                    "name": "create_task",
                                    "arguments": '{"content": "mua sữa"}',
                                },
                            }
                        ],
                    },
                    reason="tool_calls",
                    timings={"predicted_per_second": 42.5},
                ),
                mock_response(
                    {"role": "assistant", "content": "Đã thêm mua sữa."},
                    timings={"predicted_per_second": 38.0},
                ),
            ]
        )
        result = evaluate_case(case, lambda _: next(responses))
        self.assertEqual(result["status"], "pass")
        self.assertTrue(result["checks"]["calls_match"])
        self.assertTrue(result["checks"]["tasks_match"])
        self.assertEqual(len(result["response_diagnostics"]), 2)
        self.assertEqual(
            result["response_diagnostics"][0]["timings"]["predicted_per_second"], 42.5
        )

    def test_equivalent_multi_call_trace_can_be_accepted(self):
        separate_calls = [
            {"name": "create_task", "arguments": {"content": "mua sữa"}},
            {"name": "create_task", "arguments": {"content": "gọi mẹ"}},
        ]
        case = {
            "id": "multiline",
            "prompt": "Ghi lại:\nmua sữa\ngọi mẹ",
            "initial_tasks": [],
            "expected_calls": [
                {
                    "name": "create_task",
                    "arguments": {"content": "mua sữa\ngọi mẹ"},
                }
            ],
            "expected_call_options": [
                [
                    {
                        "name": "create_task",
                        "arguments": {"content": "mua sữa\ngọi mẹ"},
                    }
                ],
                separate_calls,
            ],
            "expected_tasks": ["mua sữa", "gọi mẹ"],
            "reply_expectation": "Xác nhận hai task.",
        }
        responses = iter(
            [
                mock_response(
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "type": "function",
                                "id": f"call-{index}",
                                "function": {
                                    "name": call["name"],
                                    "arguments": json.dumps(
                                        call["arguments"], ensure_ascii=False
                                    ),
                                },
                            }
                            for index, call in enumerate(separate_calls, start=1)
                        ],
                    },
                    reason="tool_calls",
                ),
                mock_response({"role": "assistant", "content": "Đã thêm hai việc."}),
            ]
        )

        result = evaluate_case(case, lambda _: next(responses))

        self.assertEqual(result["status"], "pass")
        self.assertTrue(result["checks"]["calls_match"])

    def test_summary_groups_results_by_category(self):
        summary = summarize(
            [
                {"status": "pass", "category": "create"},
                {"status": "fail", "category": "create"},
                {"status": "error", "category": "list"},
            ]
        )
        self.assertEqual(summary["total"], 3)
        self.assertEqual(summary["pass"], 1)
        self.assertEqual(summary["by_category"]["create"]["fail"], 1)

    def test_reply_inspection_detects_protocol_and_language_leaks(self):
        raw = inspect_reply('{"tasks": []}')
        leaked = inspect_reply("Không gọi tool; dùng list_tasks sau.")
        multilingual = inspect_reply("SQLite là mã nguồn mở源.")

        self.assertEqual(raw["violations"], ["raw_json"])
        self.assertEqual(leaked["violations"], ["internal_protocol"])
        self.assertEqual(multilingual["violations"], ["cjk_character"])
        self.assertFalse(raw["passed"])

    def test_reply_inspection_keeps_style_warning_separate(self):
        result = inspect_reply("NEXUS: Danh sách đang trống.")

        self.assertTrue(result["passed"])
        self.assertEqual(result["warnings"], ["assistant_label"])


if __name__ == "__main__":
    unittest.main()
