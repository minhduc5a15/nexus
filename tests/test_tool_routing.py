import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from nexus.agent.client import run_turn
from nexus.agent.routing import (
    ToolRoutingMode,
    all_tool_names,
    select_tool_route,
)
from nexus.agent.session import AgentSession
from nexus.storage.sqlite_db import create_task, initialize_database, list_tasks


def model_response(name=None, arguments=None):
    message = {"role": "assistant", "content": None}
    reason = "stop"
    if name is not None:
        reason = "tool_calls"
        message["tool_calls"] = [{
            "id": "route-test",
            "type": "function",
            "function": {
                "name": name,
                "arguments": json.dumps(arguments, ensure_ascii=False),
            },
        }]
    return {"choices": [{"finish_reason": reason, "message": message}]}


class ToolRouteSelectionTests(unittest.TestCase):
    def test_each_recognized_kind_routes_to_one_related_tool(self):
        examples = {
            "Thêm việc: mua sữa": "create_task",
            "Xem danh sách": "list_tasks",
            "Hoàn thành việc 3.": "complete_task",
            "Sửa việc 3 thành mua trà": "update_task",
            "Xóa việc 3.": "delete_task",
            "Đặt hạn việc 3 lúc 8 giờ sáng mai.": "set_task_deadline",
            "Xem việc đến hạn hôm nay.": "list_tasks_by_deadline",
            "Hoàn thành việc": "complete_task",
            "Xóa việc 1 và 2": "delete_task",
            "Đặt hạn việc 3": "set_task_deadline",
            "Xem việc theo hạn": "list_tasks_by_deadline",
        }
        for prompt, expected in examples.items():
            with self.subTest(prompt=prompt):
                route = select_tool_route(prompt, ToolRoutingMode.CLASSIFIED)
                self.assertEqual(route.tools, (expected,))
                self.assertFalse(route.fallback)

    def test_create_content_with_delete_word_stays_on_create(self):
        route = select_tool_route(
            "Thêm việc: xóa file nháp", ToolRoutingMode.CLASSIFIED
        )
        self.assertEqual(route.request_kind.value, "create")
        self.assertEqual(route.tools, ("create_task",))

    def test_deadline_query_takes_precedence_over_general_list(self):
        route = select_tool_route(
            "Xem việc đến hạn hôm nay.", ToolRoutingMode.CLASSIFIED
        )
        self.assertEqual(route.request_kind.value, "deadline_query")
        self.assertEqual(route.tools, ("list_tasks_by_deadline",))

    def test_other_falls_back_to_all_tools(self):
        route = select_tool_route("Cuối tuần có gì vui?", ToolRoutingMode.CLASSIFIED)
        self.assertEqual(route.request_kind.value, "other")
        self.assertEqual(route.tools, all_tool_names())
        self.assertTrue(route.fallback)

    def test_all_mode_reproduces_full_tool_exposure(self):
        route = select_tool_route("Thêm việc: mua sữa", ToolRoutingMode.ALL)
        self.assertEqual(route.tools, all_tool_names())
        self.assertFalse(route.fallback)

    def test_mode_requires_enum(self):
        with self.assertRaises(TypeError):
            select_tool_route("Xem danh sách", "classified")


class RoutedRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temp.name) / "routing.db"
        initialize_database(self.database_path)

    def tearDown(self):
        self.temp.cleanup()

    def test_payload_exposes_only_routed_tool_and_trace_records_route(self):
        requests = []
        result = run_turn(
            self.database_path,
            "Thêm việc: mua sữa",
            lambda payload: requests.append(deepcopy(payload))
            or model_response("create_task", {"content": "mua sữa"}),
            tool_routing=ToolRoutingMode.CLASSIFIED,
        )
        names = [tool["function"]["name"] for tool in requests[0]["tools"]]
        self.assertEqual(names, ["create_task"])
        self.assertEqual(result["status"], "executed")
        self.assertEqual(result["routing"], {
            "mode": "classified",
            "request_kind": "create",
            "tools": ["create_task"],
            "fallback": False,
        })

    def test_known_tool_outside_route_is_rejected_without_side_effect(self):
        create_task(self.database_path, "giữ nguyên")
        result = run_turn(
            self.database_path,
            "Sửa việc 1 thành bản mới",
            lambda _: model_response("delete_task", {"id": 1}),
            tool_routing=ToolRoutingMode.CLASSIFIED,
        )
        self.assertEqual(result["status"], "rejected")
        self.assertEqual(result["calls"], [])
        self.assertEqual(result["authorized_calls"], [])
        self.assertEqual(result["rejected_calls"][0]["reason"], "tool_not_available")
        self.assertEqual([task.content for task in list_tasks(self.database_path)], ["giữ nguyên"])

    def test_unknown_tool_keeps_unsupported_tool_reason(self):
        result = run_turn(
            self.database_path,
            "Cuối tuần có gì vui?",
            lambda _: model_response("invented_tool", {}),
            tool_routing=ToolRoutingMode.CLASSIFIED,
        )
        self.assertEqual(result["rejected_calls"][0]["reason"], "unsupported_tool")

    def test_batch_is_rejected_before_any_routed_call_executes(self):
        reply = model_response("create_task", {"content": "a"})
        call = reply["choices"][0]["message"]["tool_calls"][0]
        reply["choices"][0]["message"]["tool_calls"] = [call, deepcopy(call)]
        result = run_turn(
            self.database_path,
            "Thêm việc: a",
            lambda _: reply,
            tool_routing=ToolRoutingMode.CLASSIFIED,
        )
        self.assertEqual(result["status"], "rejected")
        self.assertEqual(result["calls"], [])
        self.assertEqual(list_tasks(self.database_path), [])
        self.assertEqual(
            [item["reason"] for item in result["rejected_calls"]],
            ["invalid_arguments", "invalid_arguments"],
        )

    def test_session_forwards_route_while_continuation_skips_model(self):
        session = AgentSession(
            self.database_path, tool_routing=ToolRoutingMode.CLASSIFIED
        )
        first = session.run_turn("Thêm việc", lambda _: self.fail("model called"))
        self.assertEqual(first["status"], "needs_clarification")
        second = session.run_turn("mua sữa", lambda _: self.fail("model called"))
        self.assertEqual(second["status"], "executed")
        requests = []
        third = session.run_turn(
            "Xem danh sách",
            lambda payload: requests.append(payload)
            or model_response("list_tasks", {}),
        )
        self.assertEqual(third["routing"]["tools"], ["list_tasks"])
        self.assertEqual(
            [tool["function"]["name"] for tool in requests[0]["tools"]],
            ["list_tasks"],
        )


if __name__ == "__main__":
    unittest.main()
