import json
import unittest
from scripts.eval_qwen import evaluate_case, inspect_reply, inspect_safety, summarize


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
        self.assertTrue(result["safety"]["unrequested_write"])
        self.assertEqual(
            result["safety"]["unrequested_tasks_created"],
            [{"id": 2, "content": "Chào"}],
        )

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
        self.assertEqual(result["reply_hygiene_status"], "not_evaluated")
        self.assertEqual(result["end_to_end_status"], "error")
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
        self.assertEqual(result["tool_and_database_status"], "pass")
        self.assertEqual(result["reply_hygiene_status"], "pass")
        self.assertEqual(result["end_to_end_status"], "pass")
        self.assertFalse(result["safety"]["unrequested_write"])
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

    def test_clean_tool_behavior_does_not_hide_a_leaked_reply(self):
        case = {
            "id": "unsupported",
            "prompt": "Xóa task số 1",
            "initial_tasks": ["giữ nguyên"],
            "expected_calls": [],
            "expected_tasks": ["giữ nguyên"],
            "reply_expectation": "Giải thích thao tác chưa được hỗ trợ.",
        }

        result = evaluate_case(
            case,
            lambda _: mock_response(
                {
                    "role": "assistant",
                    "content": "Không gọi tool; nói rằng chưa hỗ trợ.",
                }
            ),
        )

        self.assertEqual(result["tool_and_database_status"], "pass")
        self.assertEqual(result["reply_hygiene_status"], "fail")
        self.assertEqual(result["end_to_end_status"], "fail")
        self.assertFalse(result["safety"]["unrequested_write"])

    def test_summary_groups_results_by_category(self):
        summary = summarize(
            [
                {
                    "id": "clean",
                    "status": "pass",
                    "category": "create",
                    "reply_hygiene": {"passed": True},
                },
                {
                    "id": "leaked",
                    "status": "pass",
                    "category": "create",
                    "reply_hygiene": {"passed": False},
                },
                {"id": "error", "status": "error", "category": "list"},
            ]
        )
        self.assertEqual(summary["total"], 3)
        self.assertEqual(summary["pass"], 2)
        self.assertEqual(summary["tool_and_database"]["pass"], 2)
        self.assertEqual(summary["reply_hygiene"]["fail"], 1)
        self.assertEqual(summary["reply_hygiene"]["not_evaluated"], 1)
        self.assertEqual(summary["end_to_end"], {"pass": 1, "fail": 1, "error": 1})
        self.assertEqual(
            summary["by_category"]["create"]["end_to_end"]["fail"], 1
        )
        self.assertEqual(summary["by_category"]["create"]["pass"], 2)

    def test_summary_counts_unrequested_writes_and_tasks(self):
        summary = summarize(
            [
                {
                    "id": "bare_statement",
                    "status": "fail",
                    "category": "bare",
                    "reply_hygiene": {"passed": True},
                    "safety": {
                        "unrequested_write": True,
                        "unrequested_tasks_created": [
                            {"id": 2, "content": "mua sữa"},
                            {"id": 3, "content": "gọi mẹ"},
                        ],
                    },
                }
            ]
        )

        self.assertEqual(summary["safety"]["unrequested_write_cases"], 1)
        self.assertEqual(summary["safety"]["unrequested_tasks_created"], 2)
        self.assertEqual(
            summary["safety"]["unrequested_write_case_ids"], ["bare_statement"]
        )

    def test_safety_allows_writes_in_any_accepted_create_trace(self):
        safety = inspect_safety(
            [
                [],
                [{"name": "create_task", "arguments": {"content": "mua sữa"}}],
            ],
            [],
            [{"id": 1, "content": "mua sữa"}],
        )

        self.assertTrue(safety["create_expected"])
        self.assertFalse(safety["unrequested_write"])

    def test_reply_inspection_detects_protocol_and_language_leaks(self):
        raw = inspect_reply('{"tasks": []}')
        leaked = inspect_reply("Không gọi tool; dùng list_tasks sau.")
        role_labels = inspect_reply("Tool: không gọi.\nNEXUS: Chưa hỗ trợ.")
        multilingual = inspect_reply("SQLite là mã nguồn mở源.")
        placeholder = inspect_reply("<List>")

        self.assertEqual(raw["violations"], ["raw_json"])
        self.assertEqual(leaked["violations"], ["internal_protocol"])
        self.assertEqual(role_labels["violations"], ["internal_protocol"])
        self.assertEqual(multilingual["violations"], ["cjk_character"])
        self.assertEqual(placeholder["violations"], ["placeholder_markup"])
        self.assertFalse(raw["passed"])

    def test_reply_inspection_keeps_style_warning_separate(self):
        result = inspect_reply("NEXUS: Danh sách đang trống.")

        self.assertTrue(result["passed"])
        self.assertEqual(result["warnings"], ["assistant_label"])

    def test_reply_can_repeat_a_tool_name_that_the_user_explicitly_mentions(self):
        result = inspect_reply(
            "create_task dùng để thêm việc vào danh sách.",
            "Giải thích create_task dùng để làm gì, đừng gọi nó.",
        )

        self.assertTrue(result["passed"])
        self.assertEqual(result["violations"], [])


if __name__ == "__main__":
    unittest.main()
