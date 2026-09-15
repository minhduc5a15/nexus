import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from nexus.agent.client import (
    PostToolExecutionError,
    build_messages,
    run_probe,
    run_turn,
)
from nexus.storage.sqlite_db import initialize_database, list_tasks


def response(message, reason="stop"):
    return {"choices": [{"finish_reason": reason, "message": message}]}


class LocalProbeTests(unittest.TestCase):
    def test_probe_executes_tool_with_single_model_call_in_tool_round(self):
        requests = []
        responses = iter(
            [
                response({"role": "assistant", "content": "Chào bạn!"}),
                response(
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "type": "function",
                                "id": "local-1",
                                "function": {
                                    "name": "create_task",
                                    "arguments": '{"content":"mua sữa"}',
                                },
                            }
                        ],
                    },
                    "tool_calls",
                ),
            ]
        )

        def generate(payload):
            requests.append(deepcopy(payload))
            return next(responses)

        result = run_probe(generate)
        self.assertTrue(result["tool_check_passed"])
        self.assertEqual(len(requests), 2)
        self.assertEqual(len(result["tool_round"]["responses"]), 1)
        self.assertEqual(result["tool_round"]["reply"], "Đã thêm [1] mua sữa")
        self.assertEqual(
            result["tool_round"]["database_after"][0]["content"], "mua sữa"
        )

    def test_claimed_success_without_tool_does_not_pass(self):
        result = run_probe(
            lambda _: response({"role": "assistant", "content": "Đã thêm mua sữa."})
        )
        self.assertFalse(result["tool_check_passed"])
        self.assertEqual(result["tool_round"]["database_after"], [])

    def test_truncated_tool_call_is_never_executed(self):
        responses = iter(
            [
                response({"role": "assistant", "content": "Chào"}),
                response(
                    {"role": "assistant", "content": None, "tool_calls": []}, "length"
                ),
            ]
        )
        with patch("nexus.agent.client.execute_tool") as execute:
            with self.assertRaises(RuntimeError):
                run_probe(lambda _: next(responses))
            execute.assert_not_called()

    def test_direct_reply_without_tools(self):
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "test.db"
            initialize_database(db_path)
            turn = run_turn(
                db_path,
                "Chào bạn!",
                lambda _: response({"role": "assistant", "content": "Chào bạn!"}),
            )
            self.assertEqual(turn["calls"], [])
            self.assertEqual(turn["reply"], "Chào bạn!")

    def test_v6_uses_structured_examples_without_executing_them(self):
        requests = []

        def generate(payload):
            requests.append(deepcopy(payload))
            return response({"role": "assistant", "content": "Chào bạn!"})

        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "test.db"
            initialize_database(db_path)
            with patch("nexus.agent.client.execute_tool") as execute:
                result = run_turn(db_path, "Chào bạn!", generate, prompt_version="v6")

        messages = requests[0]["messages"]
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[-1], {"role": "user", "content": "Chào bạn!"})
        example_call = messages[2]["tool_calls"][0]
        self.assertEqual(example_call["function"]["name"], "create_task")
        self.assertEqual(
            json.loads(example_call["function"]["arguments"]),
            {"content": "mua sữa"},
        )
        self.assertEqual(messages[3]["tool_call_id"], example_call["id"])
        self.assertEqual(
            result,
            {
                "calls": [],
                "proposed_calls": [],
                "authorized_calls": [],
                "rejected_calls": [],
                "reply": "Chào bạn!",
            },
        )
        execute.assert_not_called()

    def test_build_messages_returns_an_independent_few_shot_copy(self):
        first = build_messages("v6", "Một")
        second = build_messages("v6", "Hai")

        first[1]["content"] = "đã sửa"

        self.assertEqual(
            second[1]["content"], "Thêm việc mua sữa vào danh sách giúp tôi."
        )

    def test_v7_resets_context_after_the_structured_examples(self):
        messages = build_messages("v7", "Xem danh sách")

        self.assertEqual(messages[-2]["role"], "system")
        self.assertIn("không phải lịch sử", messages[-2]["content"])
        self.assertIn("Không dùng nội dung", messages[-2]["content"])
        self.assertEqual(messages[-1], {"role": "user", "content": "Xem danh sách"})
        self.assertEqual(len(build_messages("v6", "Xem danh sách")), 14)
        self.assertEqual(len(messages), 15)

    def test_formatter_failure_exposes_committed_tool_result_under_response_formatting(
        self,
    ):
        tool_call = response(
            {
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
                    }
                ],
            },
            "tool_calls",
        )

        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "test.db"
            initialize_database(db_path)
            with patch(
                "nexus.agent.client.format_tool_result",
                side_effect=ValueError("Corrupt formatter output"),
            ):
                with self.assertRaises(PostToolExecutionError) as raised:
                    run_turn(db_path, "Thêm việc: mua sữa", lambda _: tool_call)

            error = raised.exception
            self.assertEqual(error.stage, "response_formatting")
            self.assertEqual(len(error.proposed_calls), 1)
            self.assertEqual(len(error.authorized_calls), 1)
            self.assertEqual(error.rejected_calls, [])
            self.assertEqual(error.executed_calls[0]["name"], "create_task")
            self.assertEqual(
                error.executed_calls[0]["result"]["tasks"][0]["content"], "mua sữa"
            )
            self.assertEqual(
                [task.content for task in list_tasks(db_path)], ["mua sữa"]
            )

    def test_multi_tool_call_batch_is_blocked_entirely_and_database_untouched(self):
        model_reply = response(
            {
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
            },
            "tool_calls",
        )

        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "test.db"
            initialize_database(db_path)
            turn = run_turn(db_path, "Thêm việc: mua sữa", lambda _: model_reply)

            self.assertEqual(turn["calls"], [])
            self.assertEqual(len(turn["proposed_calls"]), 2)
            self.assertEqual(len(turn["rejected_calls"]), 2)
            self.assertEqual(turn["authorized_calls"], [])
            self.assertEqual(turn["reply"], "Không có thao tác nào được thực hiện.")
            self.assertEqual(list_tasks(db_path), [])


class PolicyIntegrationInRunTurnTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.dir.name) / "test.db"
        initialize_database(self.db_path)

    def tearDown(self):
        self.dir.cleanup()

    def test_valid_create_task_executed_once(self):
        mock_generate = unittest.mock.Mock(
            return_value=response(
                {
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
                        }
                    ],
                },
                "tool_calls",
            )
        )
        turn = run_turn(self.db_path, "Thêm việc: mua sữa", mock_generate)
        mock_generate.assert_called_once()
        self.assertEqual(len(turn["calls"]), 1)
        self.assertEqual(turn["calls"][0]["name"], "create_task")
        self.assertEqual(turn["calls"][0]["arguments"], {"content": "mua sữa"})
        self.assertEqual(
            turn["proposed_calls"],
            [{"name": "create_task", "arguments": {"content": "mua sữa"}}],
        )
        self.assertEqual(
            turn["authorized_calls"],
            [
                {
                    "name": "create_task",
                    "arguments": {"content": "mua sữa"},
                    "result": "allow",
                    "reason": "explicit_create",
                }
            ],
        )
        self.assertEqual(turn["rejected_calls"], [])
        self.assertEqual(turn["reply"], "Đã thêm [1] mua sữa")
        tasks = list_tasks(self.db_path)
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0].content, "mua sữa")

    def test_valid_list_tasks_executed(self):
        from nexus.storage.sqlite_db import create_task

        create_task(self.db_path, "học bài")
        mock_generate = unittest.mock.Mock(
            return_value=response(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "type": "function",
                            "id": "call-1",
                            "function": {"name": "list_tasks", "arguments": "{}"},
                        }
                    ],
                },
                "tool_calls",
            )
        )
        turn = run_turn(self.db_path, "Xem danh sách", mock_generate)
        mock_generate.assert_called_once()
        self.assertEqual(len(turn["calls"]), 1)
        self.assertEqual(turn["calls"][0]["name"], "list_tasks")
        self.assertEqual(
            turn["authorized_calls"],
            [
                {
                    "name": "list_tasks",
                    "arguments": {},
                    "result": "allow",
                    "reason": "explicit_list",
                }
            ],
        )
        self.assertEqual(turn["rejected_calls"], [])
        self.assertEqual(turn["reply"], "Danh sách hiện có 1 việc:\n[1] học bài")

    def test_bare_statement_converted_to_create_task_does_not_mutate_db(self):
        call_resp = response(
            {
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
                    }
                ],
            },
            "tool_calls",
        )
        turn = run_turn(self.db_path, "Mua sữa.", lambda _: call_resp)
        self.assertEqual(turn["calls"], [])
        self.assertEqual(turn["authorized_calls"], [])
        self.assertEqual(len(turn["rejected_calls"]), 1)
        self.assertEqual(turn["rejected_calls"][0]["result"], "reject")
        self.assertEqual(turn["rejected_calls"][0]["reason"], "bare_statement")
        self.assertEqual(turn["reply"], "Không có thao tác nào được thực hiện.")
        self.assertEqual(list_tasks(self.db_path), [])

    def test_negated_request_does_not_mutate_db(self):
        call_resp = response(
            {
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
                    }
                ],
            },
            "tool_calls",
        )
        turn = run_turn(self.db_path, "Đừng thêm việc mua sữa.", lambda _: call_resp)
        self.assertEqual(turn["calls"], [])
        self.assertEqual(turn["authorized_calls"], [])
        self.assertEqual(len(turn["rejected_calls"]), 1)
        self.assertEqual(turn["rejected_calls"][0]["result"], "reject")
        self.assertEqual(turn["rejected_calls"][0]["reason"], "negated_request")
        self.assertEqual(turn["reply"], "Không có thao tác nào được thực hiện.")
        self.assertEqual(list_tasks(self.db_path), [])

    def test_modified_content_blocked(self):
        call_resp = response(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "type": "function",
                        "id": "call-1",
                        "function": {
                            "name": "create_task",
                            "arguments": '{"content":"Mua sữa"}',
                        },
                    }
                ],
            },
            "tool_calls",
        )
        turn = run_turn(self.db_path, "Thêm việc: mua sữa", lambda _: call_resp)
        self.assertEqual(turn["calls"], [])
        self.assertEqual(turn["authorized_calls"], [])
        self.assertEqual(len(turn["rejected_calls"]), 1)
        self.assertEqual(turn["rejected_calls"][0]["result"], "reject")
        self.assertEqual(turn["rejected_calls"][0]["reason"], "content_not_grounded")
        self.assertEqual(turn["reply"], "Không có thao tác nào được thực hiện.")
        self.assertEqual(list_tasks(self.db_path), [])

    def test_missing_content_asks_for_clarification(self):
        call_resp = response(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "type": "function",
                        "id": "call-1",
                        "function": {
                            "name": "create_task",
                            "arguments": '{"content":"đoán nội dung"}',
                        },
                    }
                ],
            },
            "tool_calls",
        )
        turn = run_turn(self.db_path, "Thêm việc.", lambda _: call_resp)
        self.assertEqual(turn["calls"], [])
        self.assertEqual(turn["authorized_calls"], [])
        self.assertEqual(len(turn["rejected_calls"]), 1)
        self.assertEqual(turn["rejected_calls"][0]["result"], "needs_clarification")
        self.assertEqual(turn["rejected_calls"][0]["reason"], "missing_content")
        self.assertEqual(turn["reply"], "Bạn muốn thêm việc gì?")
        self.assertEqual(list_tasks(self.db_path), [])

    def test_unsupported_tool_blocked(self):
        call_resp = response(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "type": "function",
                        "id": "call-1",
                        "function": {"name": "delete_task", "arguments": '{"id":1}'},
                    }
                ],
            },
            "tool_calls",
        )
        turn = run_turn(self.db_path, "Xóa task mua sữa.", lambda _: call_resp)
        self.assertEqual(turn["calls"], [])
        self.assertEqual(turn["authorized_calls"], [])
        self.assertEqual(len(turn["rejected_calls"]), 1)
        self.assertEqual(turn["rejected_calls"][0]["result"], "reject")
        self.assertEqual(turn["rejected_calls"][0]["reason"], "unsupported_tool")
        self.assertEqual(turn["reply"], "Không có thao tác nào được thực hiện.")
        self.assertEqual(list_tasks(self.db_path), [])

    def test_invalid_json_arguments_blocked(self):
        call_resp = response(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "type": "function",
                        "id": "call-1",
                        "function": {
                            "name": "create_task",
                            "arguments": "{invalid_json",
                        },
                    }
                ],
            },
            "tool_calls",
        )
        turn = run_turn(self.db_path, "Thêm việc: mua sữa", lambda _: call_resp)
        self.assertEqual(turn["calls"], [])
        self.assertEqual(turn["authorized_calls"], [])
        self.assertEqual(len(turn["rejected_calls"]), 1)
        self.assertEqual(turn["rejected_calls"][0]["result"], "reject")
        self.assertEqual(turn["rejected_calls"][0]["reason"], "invalid_arguments")
        self.assertEqual(turn["reply"], "Không có thao tác nào được thực hiện.")
        self.assertEqual(list_tasks(self.db_path), [])

    def test_two_valid_calls_at_once_executes_neither(self):
        call_resp = response(
            {
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
                        "function": {"name": "list_tasks", "arguments": "{}"},
                    },
                ],
            },
            "tool_calls",
        )
        turn = run_turn(self.db_path, "Thêm việc: mua sữa", lambda _: call_resp)
        self.assertEqual(turn["calls"], [])
        self.assertEqual(turn["authorized_calls"], [])
        self.assertEqual(len(turn["proposed_calls"]), 2)
        self.assertEqual(len(turn["rejected_calls"]), 2)
        self.assertEqual(turn["rejected_calls"][0]["reason"], "invalid_arguments")
        self.assertEqual(turn["rejected_calls"][1]["reason"], "invalid_arguments")
        self.assertEqual(turn["reply"], "Không có thao tác nào được thực hiện.")
        self.assertEqual(list_tasks(self.db_path), [])

    def test_non_function_tool_call_blocked(self):
        call_resp = response(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "type": "custom",
                        "id": "call-1",
                        "function": {
                            "name": "create_task",
                            "arguments": '{"content":"mua sữa"}',
                        },
                    }
                ],
            },
            "tool_calls",
        )
        turn = run_turn(self.db_path, "Thêm việc: mua sữa", lambda _: call_resp)
        self.assertEqual(turn["calls"], [])
        self.assertEqual(turn["authorized_calls"], [])
        self.assertEqual(len(turn["rejected_calls"]), 1)
        self.assertEqual(turn["rejected_calls"][0]["reason"], "invalid_arguments")
        self.assertEqual(turn["reply"], "Không có thao tác nào được thực hiện.")
        self.assertEqual(list_tasks(self.db_path), [])

    def test_direct_reply_does_not_call_policy(self):
        with patch("nexus.agent.client.policy_for_tool") as mock_policy:
            turn = run_turn(
                self.db_path,
                "Chào bạn!",
                lambda _: response({"role": "assistant", "content": "Chào bạn!"}),
            )
            mock_policy.assert_not_called()
            self.assertEqual(turn["calls"], [])
            self.assertEqual(turn["proposed_calls"], [])
            self.assertEqual(turn["authorized_calls"], [])
            self.assertEqual(turn["rejected_calls"], [])
            self.assertEqual(turn["reply"], "Chào bạn!")

    def test_generate_is_called_exactly_once_across_all_turn_branches(self):
        # Case A: Direct reply -> exactly 1 generate call
        mock_gen_direct = unittest.mock.Mock(
            return_value=response({"role": "assistant", "content": "Chào bạn!"})
        )
        turn = run_turn(self.db_path, "Chào bạn!", mock_gen_direct)
        mock_gen_direct.assert_called_once()
        self.assertEqual(turn["reply"], "Chào bạn!")

        # Case B: Rejected call -> exactly 1 generate call
        mock_gen_reject = unittest.mock.Mock(
            return_value=response(
                {
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
                        }
                    ],
                },
                "tool_calls",
            )
        )
        turn = run_turn(self.db_path, "Mua sữa.", mock_gen_reject)
        mock_gen_reject.assert_called_once()
        self.assertEqual(turn["calls"], [])

        # Case C: Authorized create_task -> exactly 1 generate call
        mock_gen_create = unittest.mock.Mock(
            return_value=response(
                {
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
                        }
                    ],
                },
                "tool_calls",
            )
        )
        turn = run_turn(self.db_path, "Thêm việc: mua sữa", mock_gen_create)
        mock_gen_create.assert_called_once()
        self.assertEqual(len(turn["calls"]), 1)
        self.assertEqual(turn["reply"], "Đã thêm [1] mua sữa")

        # Case D: Authorized list_tasks -> exactly 1 generate call
        mock_gen_list = unittest.mock.Mock(
            return_value=response(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "type": "function",
                            "id": "call-1",
                            "function": {"name": "list_tasks", "arguments": "{}"},
                        }
                    ],
                },
                "tool_calls",
            )
        )
        turn = run_turn(self.db_path, "Xem danh sách", mock_gen_list)
        mock_gen_list.assert_called_once()
        self.assertEqual(len(turn["calls"]), 1)
        self.assertEqual(turn["reply"], "Danh sách hiện có 1 việc:\n[1] mua sữa")

        # Case E: Multi-tool batch rejected -> exactly 1 generate call
        mock_gen_batch = unittest.mock.Mock(
            return_value=response(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "type": "function",
                            "id": "c1",
                            "function": {"name": "list_tasks", "arguments": "{}"},
                        },
                        {
                            "type": "function",
                            "id": "c2",
                            "function": {"name": "list_tasks", "arguments": "{}"},
                        },
                    ],
                },
                "tool_calls",
            )
        )
        turn = run_turn(self.db_path, "Xem danh sách", mock_gen_batch)
        mock_gen_batch.assert_called_once()
        self.assertEqual(turn["calls"], [])

    def test_initial_message_content_does_not_override_deterministic_tool_reply(self):
        call_resp = response(
            {
                "role": "assistant",
                "content": "Tôi là AI giỏi nhất!",
                "tool_calls": [
                    {
                        "type": "function",
                        "id": "call-1",
                        "function": {
                            "name": "create_task",
                            "arguments": '{"content":"mua táo"}',
                        },
                    }
                ],
            },
            "tool_calls",
        )
        turn = run_turn(self.db_path, "Thêm việc: mua táo", lambda _: call_resp)
        self.assertEqual(turn["reply"], "Đã thêm [1] mua táo")
        self.assertNotIn("Tôi là AI giỏi nhất!", turn["reply"])

    def test_formatter_receives_exact_name_and_result_from_execute_tool(self):
        call_resp = response(
            {
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
                    }
                ],
            },
            "tool_calls",
        )
        with patch(
            "nexus.agent.client.format_tool_result", return_value="Đã thêm [1] mua sữa"
        ) as mock_fmt:
            run_turn(self.db_path, "Thêm việc: mua sữa", lambda _: call_resp)
            mock_fmt.assert_called_once_with(
                "create_task", {"tasks": [{"id": 1, "content": "mua sữa"}]}
            )

    def test_execute_tool_failure_does_not_call_formatter(self):
        call_resp = response(
            {
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
                    }
                ],
            },
            "tool_calls",
        )
        with patch(
            "nexus.agent.client.execute_tool",
            side_effect=RuntimeError("database disk error"),
        ):
            with patch("nexus.agent.client.format_tool_result") as mock_fmt:
                with self.assertRaises(RuntimeError):
                    run_turn(self.db_path, "Thêm việc: mua sữa", lambda _: call_resp)
                mock_fmt.assert_not_called()

    def test_list_reply_reflects_actual_database_tasks(self):
        from nexus.storage.sqlite_db import create_task

        create_task(self.db_path, "việc 1")
        create_task(self.db_path, "việc 2")
        call_resp = response(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "type": "function",
                        "id": "call-1",
                        "function": {"name": "list_tasks", "arguments": "{}"},
                    }
                ],
            },
            "tool_calls",
        )
        turn = run_turn(self.db_path, "Xem danh sách", lambda _: call_resp)
        self.assertEqual(
            turn["reply"], "Danh sách hiện có 2 việc:\n[1] việc 1\n[2] việc 2"
        )

    def test_missing_id_in_tool_call_is_blocked(self):
        call_count = 0

        def generate(_):
            nonlocal call_count
            call_count += 1
            return response(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {
                                "name": "create_task",
                                "arguments": '{"content":"mua sữa"}',
                            },
                        }
                    ],
                },
                "tool_calls",
            )

        turn = run_turn(self.db_path, "Thêm việc: mua sữa", generate)
        self.assertEqual(call_count, 1)
        self.assertEqual(turn["calls"], [])
        self.assertEqual(turn["authorized_calls"], [])
        self.assertEqual(len(turn["rejected_calls"]), 1)
        self.assertEqual(turn["rejected_calls"][0]["result"], "reject")
        self.assertEqual(turn["rejected_calls"][0]["reason"], "invalid_arguments")
        self.assertEqual(turn["reply"], "Không có thao tác nào được thực hiện.")
        self.assertEqual(list_tasks(self.db_path), [])

    def test_none_id_in_tool_call_is_blocked(self):
        call_count = 0

        def generate(_):
            nonlocal call_count
            call_count += 1
            return response(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": None,
                            "type": "function",
                            "function": {
                                "name": "create_task",
                                "arguments": '{"content":"mua sữa"}',
                            },
                        }
                    ],
                },
                "tool_calls",
            )

        turn = run_turn(self.db_path, "Thêm việc: mua sữa", generate)
        self.assertEqual(call_count, 1)
        self.assertEqual(turn["calls"], [])
        self.assertEqual(turn["authorized_calls"], [])
        self.assertEqual(len(turn["rejected_calls"]), 1)
        self.assertEqual(turn["rejected_calls"][0]["result"], "reject")
        self.assertEqual(turn["rejected_calls"][0]["reason"], "invalid_arguments")
        self.assertEqual(turn["reply"], "Không có thao tác nào được thực hiện.")
        self.assertEqual(list_tasks(self.db_path), [])

    def test_empty_id_in_tool_call_is_blocked(self):
        for empty_id in ["", "   "]:
            call_count = 0

            def generate(_):
                nonlocal call_count
                call_count += 1
                return response(
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": empty_id,
                                "type": "function",
                                "function": {
                                    "name": "create_task",
                                    "arguments": '{"content":"mua sữa"}',
                                },
                            }
                        ],
                    },
                    "tool_calls",
                )

            turn = run_turn(self.db_path, "Thêm việc: mua sữa", generate)
            self.assertEqual(call_count, 1)
            self.assertEqual(turn["calls"], [])
            self.assertEqual(turn["authorized_calls"], [])
            self.assertEqual(len(turn["rejected_calls"]), 1)
            self.assertEqual(turn["rejected_calls"][0]["result"], "reject")
            self.assertEqual(turn["rejected_calls"][0]["reason"], "invalid_arguments")
            self.assertEqual(turn["reply"], "Không có thao tác nào được thực hiện.")
            self.assertEqual(list_tasks(self.db_path), [])

    def test_tool_calls_as_dict_is_rejected_without_exception(self):
        call_resp = response(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "create_task",
                        "arguments": '{"content":"mua sữa"}',
                    },
                },
            },
            "tool_calls",
        )

        turn = run_turn(self.db_path, "Thêm việc: mua sữa", lambda _: call_resp)
        self.assertEqual(turn["calls"], [])
        self.assertEqual(turn["authorized_calls"], [])
        self.assertEqual(len(turn["rejected_calls"]), 1)
        self.assertEqual(turn["rejected_calls"][0]["result"], "reject")
        self.assertEqual(turn["rejected_calls"][0]["reason"], "invalid_arguments")
        self.assertEqual(turn["reply"], "Không có thao tác nào được thực hiện.")
        self.assertEqual(list_tasks(self.db_path), [])

    def test_tool_calls_as_string_is_rejected_without_exception(self):
        call_resp = response(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": "create_task",
            },
            "tool_calls",
        )

        turn = run_turn(self.db_path, "Thêm việc: mua sữa", lambda _: call_resp)
        self.assertEqual(turn["calls"], [])
        self.assertEqual(turn["authorized_calls"], [])
        self.assertEqual(len(turn["rejected_calls"]), 1)
        self.assertEqual(turn["rejected_calls"][0]["result"], "reject")
        self.assertEqual(turn["rejected_calls"][0]["reason"], "invalid_arguments")
        self.assertEqual(turn["reply"], "Không có thao tác nào được thực hiện.")
        self.assertEqual(list_tasks(self.db_path), [])

    def test_tool_calls_as_non_list_value_is_rejected_without_exception(self):
        for invalid_val in [123, True, 4.56]:
            call_resp = response(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": invalid_val,
                },
                "tool_calls",
            )

            turn = run_turn(self.db_path, "Thêm việc: mua sữa", lambda _: call_resp)
            self.assertEqual(turn["calls"], [])
            self.assertEqual(turn["authorized_calls"], [])
            self.assertEqual(len(turn["rejected_calls"]), 1)
            self.assertEqual(turn["rejected_calls"][0]["result"], "reject")
            self.assertEqual(turn["rejected_calls"][0]["reason"], "invalid_arguments")
            self.assertEqual(turn["reply"], "Không có thao tác nào được thực hiện.")
            self.assertEqual(list_tasks(self.db_path), [])

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
