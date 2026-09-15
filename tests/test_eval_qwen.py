import json
import unittest
from unittest.mock import patch
from nexus.storage.sqlite_db import create_task as db_create_task
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

        def fake_turn(path, prompt, observe, **kwargs):
            observe({"messages": []})
            db_create_task(path, "Chào")
            return {
                "calls": [{"name": "create_task", "arguments": {"content": "Chào"}}],
                "reply": "Chào bạn",
            }

        with patch("scripts.eval_qwen.run_turn", side_effect=fake_turn):
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
        self.assertEqual(result["api_requests"], 1)
        self.assertIsNone(result["reply"])

    def test_successful_turn_records_diagnostics_and_passes(self):
        case = {
            "id": "create_one",
            "prompt": "Thêm việc: mua sữa",
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
        self.assertEqual(result["model_proposal_status"], "pass")
        self.assertEqual(result["system_action_status"], "pass")
        self.assertEqual(result["system_end_to_end_status"], "pass")
        self.assertEqual(
            result["proposed_calls"],
            [{"name": "create_task", "arguments": {"content": "mua sữa"}}],
        )
        self.assertEqual(len(result["authorized_calls"]), 1)
        self.assertEqual(len(result["executed_calls"]), 1)
        self.assertEqual(result["rejected_calls"], [])
        self.assertEqual(len(result["response_diagnostics"]), 1)
        self.assertEqual(
            result["response_diagnostics"][0]["timings"]["predicted_per_second"], 42.5
        )

    def test_policy_block_is_separate_from_model_proposal_failure(self):
        case = {
            "id": "bare_statement",
            "prompt": "Mua sữa.",
            "initial_tasks": ["giữ nguyên"],
            "expected_calls": [],
            "expected_tasks": ["giữ nguyên"],
            "reply_expectation": "Không xác nhận đã thêm.",
        }
        response = mock_response(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "type": "function",
                        "id": "call-1",
                        "function": {
                            "name": "create_task",
                            "arguments": '{"content": "Mua sữa."}',
                        },
                    }
                ],
            },
            reason="tool_calls",
        )

        result = evaluate_case(case, lambda _: response)

        self.assertEqual(result["model_proposal_status"], "fail")
        self.assertEqual(result["system_action_status"], "pass")
        self.assertEqual(result["system_end_to_end_status"], "pass")
        self.assertEqual(result["executed_calls"], [])
        self.assertEqual(result["database_after"], [{"id": 1, "content": "giữ nguyên"}])
        self.assertTrue(result["policy"]["intervened"])
        self.assertTrue(result["policy"]["blocked_bad_proposal"])
        self.assertTrue(result["policy"]["recovered_model_failure"])
        self.assertFalse(result["policy"]["false_rejection"])

    def test_bad_create_content_can_be_blocked_without_recovering_the_request(self):
        case = {
            "id": "changed_content",
            "prompt": "Thêm việc: mua sữa",
            "initial_tasks": [],
            "expected_calls": [
                {"name": "create_task", "arguments": {"content": "mua sữa"}}
            ],
            "expected_tasks": ["mua sữa"],
            "reply_expectation": "Xác nhận.",
        }
        response = mock_response(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "type": "function",
                        "id": "call-1",
                        "function": {
                            "name": "create_task",
                            "arguments": '{"content": "Mua sữa"}',
                        },
                    }
                ],
            },
            reason="tool_calls",
        )

        result = evaluate_case(case, lambda _: response)

        self.assertEqual(result["model_proposal_status"], "fail")
        self.assertEqual(result["system_action_status"], "fail")
        self.assertTrue(result["policy"]["blocked_bad_proposal"])
        self.assertFalse(result["policy"]["recovered_model_failure"])
        self.assertEqual(result["database_after"], [])

    def test_matching_proposal_rejected_by_policy_is_counted_as_false_rejection(self):
        case = {
            "id": "valid_but_rejected",
            "prompt": "Thêm việc: mua sữa",
            "initial_tasks": [],
            "expected_calls": [
                {"name": "create_task", "arguments": {"content": "mua sữa"}}
            ],
            "expected_tasks": ["mua sữa"],
            "reply_expectation": "Xác nhận.",
        }
        proposal = {"name": "create_task", "arguments": {"content": "mua sữa"}}

        def fake_turn(path, prompt, observe, **kwargs):
            observe(
                {
                    "choices": [
                        {
                            "finish_reason": "tool_calls",
                            "message": {
                                "tool_calls": [
                                    {
                                        "type": "function",
                                        "id": "call-1",
                                        "function": {
                                            "name": "create_task",
                                            "arguments": '{"content": "mua sữa"}',
                                        },
                                    }
                                ]
                            },
                        }
                    ]
                }
            )
            return {
                "calls": [],
                "proposed_calls": [proposal],
                "authorized_calls": [],
                "rejected_calls": [
                    {**proposal, "result": "reject", "reason": "invalid_arguments"}
                ],
                "reply": "Không có thao tác nào được thực hiện.",
            }

        with patch("scripts.eval_qwen.run_turn", side_effect=fake_turn):
            result = evaluate_case(case, lambda _: mock_response({}))

        self.assertEqual(result["model_proposal_status"], "pass")
        self.assertEqual(result["system_action_status"], "fail")
        self.assertTrue(result["policy"]["false_rejection"])

    def test_formatting_error_keeps_completed_action_trace(self):
        case = {
            "id": "formatting_error",
            "prompt": "Thêm việc: mua sữa",
            "initial_tasks": [],
            "expected_calls": [
                {"name": "create_task", "arguments": {"content": "mua sữa"}}
            ],
            "expected_tasks": ["mua sữa"],
            "reply_expectation": "Xác nhận.",
        }
        response = mock_response(
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
        )

        with patch(
            "nexus.agent.client.format_tool_result",
            side_effect=ValueError("formatter broke"),
        ):
            result = evaluate_case(case, lambda _: response)

        self.assertEqual(result["error"]["stage"], "response_formatting")
        self.assertEqual(result["model_proposal_status"], "pass")
        self.assertEqual(result["system_action_status"], "pass")
        self.assertEqual(result["system_end_to_end_status"], "error")
        self.assertEqual(
            result["authorized_calls"][0]["reason"], "explicit_create"
        )
        self.assertEqual(len(result["executed_calls"]), 1)
        self.assertEqual(result["database_after"], [{"id": 1, "content": "mua sữa"}])

    def test_legacy_dataset_can_accept_equivalent_multi_call_trace(self):
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

        def fake_multi_turn(path, prompt, observe, **kwargs):
            observe({"messages": []})
            for c in separate_calls:
                db_create_task(path, c["arguments"]["content"])
            return {"calls": separate_calls, "reply": "Đã thêm hai việc."}

        with patch("scripts.eval_qwen.run_turn", side_effect=fake_multi_turn):
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
        self.assertEqual(summary["by_category"]["create"]["end_to_end"]["fail"], 1)
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

    def test_summary_counts_model_system_and_policy_outcomes(self):
        summary = summarize(
            [
                {
                    "id": "blocked",
                    "status": "fail",
                    "category": "safety",
                    "reply_hygiene_status": "pass",
                    "end_to_end_status": "fail",
                    "model_proposal_status": "fail",
                    "system_action_status": "pass",
                    "system_end_to_end_status": "pass",
                    "policy": {
                        "intervened": True,
                        "blocked_bad_proposal": True,
                        "recovered_model_failure": True,
                        "false_rejection": False,
                    },
                },
                {
                    "id": "false_reject",
                    "status": "fail",
                    "category": "create",
                    "reply_hygiene_status": "pass",
                    "end_to_end_status": "fail",
                    "model_proposal_status": "pass",
                    "system_action_status": "fail",
                    "system_end_to_end_status": "fail",
                    "policy": {
                        "intervened": True,
                        "blocked_bad_proposal": False,
                        "recovered_model_failure": False,
                        "false_rejection": True,
                    },
                },
            ]
        )

        self.assertEqual(summary["model_proposal"], {"pass": 1, "fail": 1, "error": 0})
        self.assertEqual(summary["system_action"], {"pass": 1, "fail": 1, "error": 0})
        self.assertEqual(summary["system_end_to_end"], {"pass": 1, "fail": 1, "error": 0})
        self.assertEqual(summary["policy"]["intervention_cases"], 2)
        self.assertEqual(summary["policy"]["blocked_bad_proposal_case_ids"], ["blocked"])
        self.assertEqual(summary["policy"]["recovered_model_failure_case_ids"], ["blocked"])
        self.assertEqual(summary["policy"]["false_rejection_case_ids"], ["false_reject"])
        self.assertEqual(
            summary["by_category"]["safety"]["blocked_bad_proposal_cases"], 1
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
