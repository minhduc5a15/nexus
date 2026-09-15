import unittest
from dataclasses import FrozenInstanceError

from nexus.agent.policy import PolicyResult, PolicyReason, ToolDecision


class TestToolDecision(unittest.TestCase):

    def test_valid_combinations(self):
        """Ensure valid combinations can be successfully created."""
        # Allow
        ToolDecision(PolicyResult.ALLOW, PolicyReason.EXPLICIT_CREATE)
        ToolDecision(PolicyResult.ALLOW, PolicyReason.EXPLICIT_LIST)

        # Needs Clarification
        ToolDecision(PolicyResult.NEEDS_CLARIFICATION, PolicyReason.MISSING_CONTENT)

        # Reject (All 6 reasons)
        ToolDecision(PolicyResult.REJECT, PolicyReason.NEGATED_REQUEST)
        ToolDecision(PolicyResult.REJECT, PolicyReason.UNSUPPORTED_ACTION)
        ToolDecision(PolicyResult.REJECT, PolicyReason.UNSUPPORTED_TOOL)
        ToolDecision(PolicyResult.REJECT, PolicyReason.BARE_STATEMENT)
        ToolDecision(PolicyResult.REJECT, PolicyReason.INVALID_ARGUMENTS)
        ToolDecision(PolicyResult.REJECT, PolicyReason.CONTENT_NOT_GROUNDED)

    def test_invalid_combinations_rejected(self):
        """Ensure invalid combinations raise a ValueError."""
        with self.assertRaises(ValueError):
            ToolDecision(PolicyResult.ALLOW, PolicyReason.NEGATED_REQUEST)

        with self.assertRaises(ValueError):
            ToolDecision(PolicyResult.NEEDS_CLARIFICATION, PolicyReason.EXPLICIT_CREATE)

        with self.assertRaises(ValueError):
            ToolDecision(PolicyResult.REJECT, PolicyReason.MISSING_CONTENT)

    def test_invalid_types_rejected(self):
        """Ensure passing raw strings or None raises a TypeError."""
        # Reject raw strings
        with self.assertRaises(TypeError):
            ToolDecision("allow", PolicyReason.EXPLICIT_CREATE)

        with self.assertRaises(TypeError):
            ToolDecision(PolicyResult.ALLOW, "explicit_create")

        # Reject None values
        with self.assertRaises(TypeError):
            ToolDecision(None, PolicyReason.EXPLICIT_CREATE)

        with self.assertRaises(TypeError):
            ToolDecision(PolicyResult.ALLOW, None)

    def test_decision_is_immutable(self):
        """Ensure the decision object cannot be modified after creation."""
        decision = ToolDecision(PolicyResult.ALLOW, PolicyReason.EXPLICIT_CREATE)

        with self.assertRaises(FrozenInstanceError):
            decision.result = PolicyResult.REJECT

        with self.assertRaises(FrozenInstanceError):
            decision.reason = PolicyReason.EXPLICIT_LIST

    def test_enum_string_values(self):
        """Verify all string values of the results and reasons."""
        # All 3 Results
        self.assertEqual(PolicyResult.ALLOW.value, "allow")
        self.assertEqual(PolicyResult.REJECT.value, "reject")
        self.assertEqual(PolicyResult.NEEDS_CLARIFICATION.value, "needs_clarification")

        # All 9 Reasons
        self.assertEqual(PolicyReason.EXPLICIT_CREATE.value, "explicit_create")
        self.assertEqual(PolicyReason.EXPLICIT_LIST.value, "explicit_list")
        self.assertEqual(PolicyReason.MISSING_CONTENT.value, "missing_content")
        self.assertEqual(PolicyReason.NEGATED_REQUEST.value, "negated_request")
        self.assertEqual(PolicyReason.UNSUPPORTED_ACTION.value, "unsupported_action")
        self.assertEqual(PolicyReason.UNSUPPORTED_TOOL.value, "unsupported_tool")
        self.assertEqual(PolicyReason.BARE_STATEMENT.value, "bare_statement")
        self.assertEqual(PolicyReason.INVALID_ARGUMENTS.value, "invalid_arguments")
        self.assertEqual(
            PolicyReason.CONTENT_NOT_GROUNDED.value, "content_not_grounded"
        )


from nexus.agent.policy import policy_for_list_tasks


class TestPolicyForListTasks(unittest.TestCase):
    def test_invalid_arguments_and_edge_cases(self):
        self.assertEqual(
            policy_for_create_task(None, {"content": "mua sữa"}).reason,
            PolicyReason.INVALID_ARGUMENTS,
        )
        self.assertEqual(
            policy_for_create_task("  ", {"content": "mua sữa"}).reason,
            PolicyReason.INVALID_ARGUMENTS,
        )

        args = {"content": "mua sữa"}
        policy_for_create_task("Thêm việc: mua sữa", args)
        self.assertEqual(args, {"content": "mua sữa"})

        self.assertEqual(
            policy_for_create_task("tHêM ViỆc: Mua SỮA", {"content": "Mua SỮA"}).result,
            PolicyResult.ALLOW,
        )
        self.assertEqual(
            policy_for_create_task("Thêm việc: Mua SỮA", {"content": "mua sữa"}).reason,
            PolicyReason.CONTENT_NOT_GROUNDED,
        )

    def test_invalid_arguments(self):
        self.assertEqual(
            policy_for_list_tasks(123, {}).reason, PolicyReason.INVALID_ARGUMENTS
        )
        self.assertEqual(
            policy_for_list_tasks("   ", {}).reason, PolicyReason.INVALID_ARGUMENTS
        )
        self.assertEqual(
            policy_for_list_tasks("xem danh sách", []).reason,
            PolicyReason.INVALID_ARGUMENTS,
        )
        self.assertEqual(
            policy_for_list_tasks("xem danh sách", {"a": 1}).reason,
            PolicyReason.INVALID_ARGUMENTS,
        )

        # Test None
        self.assertEqual(
            policy_for_list_tasks(None, {}).reason, PolicyReason.INVALID_ARGUMENTS
        )
        self.assertEqual(
            policy_for_list_tasks("xem danh sách", None).reason,
            PolicyReason.INVALID_ARGUMENTS,
        )

    def test_negated_request(self):
        phrases = [
            "Đừng xem danh sách.",
            "Không cần hiển thị todo.",
            "Cho tôi biết `list_tasks` hoạt động ra sao, không cần chạy.",
            " ĐỪNG    xem danh Sách  \n",
            "Không xem danh sách.",
            "Tôi không muốn xem todo.",
            "Chưa cần mở danh sách.",
            "Khỏi hiển thị todo.",
            "Không mở danh sách.",
            "Không liệt kê task.",
            "Khỏi xem todo.",
            "Tôi không muốn mở danh sách.",
        ]
        for ph in phrases:
            dec = policy_for_list_tasks(ph, {})
            self.assertEqual(dec.result, PolicyResult.REJECT)
            self.assertEqual(dec.reason, PolicyReason.NEGATED_REQUEST)

    def test_unsupported_action(self):
        phrases = [
            "Xóa toàn bộ danh sách.",
            "Hoàn thành task số 2.",
            "Note giúp: xem danh sách mua sắm.",
            " Sửa    \n danh sách",
        ]
        for ph in phrases:
            dec = policy_for_list_tasks(ph, {})
            self.assertEqual(dec.result, PolicyResult.REJECT)
            self.assertEqual(dec.reason, PolicyReason.UNSUPPORTED_ACTION)

    def test_explicit_list(self):
        phrases = [
            "Xem danh sách",
            "Mở danh sách việc đã ghi cho tôi.",
            "Todo hiện giờ gồm gì?",
            "Tôi đã note những gì vậy?",
            "Danh sách của mình đang có gì không?",
            "Liệt kê tất cả task hiện có.",
            "Có những việc nào trong danh sách của tôi?",
            "Show todo của tui với.",
            "Kiểm tra xem tôi đã lưu gì chưa.",
            "   xEm   DaNh sÁcH   ",
            "Xem danh sách để khỏi quên.",
        ]
        for ph in phrases:
            dec = policy_for_list_tasks(ph, {})
            self.assertEqual(dec.result, PolicyResult.ALLOW)
            self.assertEqual(dec.reason, PolicyReason.EXPLICIT_LIST)

    def test_bare_statement(self):
        phrases = [
            "Hôm nay trời đẹp.",
            "Mua sữa.",
            "Đây là danh sách mua sắm của tôi.",
            "Todo là tên một ứng dụng.",
            "Danh sách này khá dài.",
            "Liệt kê các số nguyên tố.",
            "Kiểm tra xem trời có mưa không.",
            "Bữa sáng gồm gì?",
            "Hôm qua bạn đã làm những gì vậy?",
            "Có những việc nào giúp tôi ngủ ngon?",
        ]
        for ph in phrases:
            dec = policy_for_list_tasks(ph, {})
            self.assertEqual(dec.result, PolicyResult.REJECT)
            self.assertEqual(dec.reason, PolicyReason.BARE_STATEMENT)

    def test_arguments_unmutated(self):
        # We need to test when arguments are empty (valid case), but let's test invalid cases too
        # to ensure it's not mutated
        args = {"a": 1}
        policy_for_list_tasks("xem danh sách", args)
        self.assertEqual(args, {"a": 1})

        valid_args = {}
        policy_for_list_tasks("xem danh sách", valid_args)
        self.assertEqual(valid_args, {})


from nexus.agent.policy import policy_for_create_task


class TestPolicyForCreateTasks(unittest.TestCase):
    def test_allow(self):
        cases = [
            ("Thêm việc: mua sữa", {"content": "mua sữa"}),
            ("Thêm task mua sữa", {"content": "mua sữa"}),
            ("Ghi lại việc: gọi cho mẹ", {"content": "gọi cho mẹ"}),
            ("Ghi việc nộp báo cáo", {"content": "nộp báo cáo"}),
            ("Lưu việc: đặt lịch khám", {"content": "đặt lịch khám"}),
            ("Note giúp: thay pin chuột", {"content": "thay pin chuột"}),
            ("Bỏ vào todo: tưới cây", {"content": "tưới cây"}),
            ("Thêm các việc:\nmua sữa\nmua bánh", {"content": "mua sữa\nmua bánh"}),
            (
                "Thêm việc: đi ăn lúc 8:00 và ngủ lúc 10:00",
                {"content": "đi ăn lúc 8:00 và ngủ lúc 10:00"},
            ),
            ("Thêm việc: sửa xe.", {"content": "sửa xe."}),
            ("Thêm việc: hoàn thành báo cáo.", {"content": "hoàn thành báo cáo."}),
            ("Note giúp: xóa file tạm.", {"content": "xóa file tạm."}),
            (
                "Thêm việc: mua thuốc không cần kê đơn.",
                {"content": "mua thuốc không cần kê đơn."},
            ),
            ("Ghi việc: đừng quên gọi cho mẹ.", {"content": "đừng quên gọi cho mẹ."}),
            ("Thêm việc: sửa task mua sữa.", {"content": "sửa task mua sữa."}),
            ("Thêm việc: đừng quên mua sữa.", {"content": "đừng quên mua sữa."}),
        ]
        for prompt, args in cases:
            dec = policy_for_create_task(prompt, args)
            self.assertEqual(dec.result, PolicyResult.ALLOW, f"Failed on: {prompt}")

    def test_bare(self):
        cases = [
            "Mua sữa.",
            "Tôi muốn mua sữa.",
            "Thêm taskbar vào màn hình",
            "Note giúpđỡ tôi",
            "Tôi không muốn uống cà phê.",
            "Tôi sẽ sửa xe.",
        ]
        for prompt in cases:
            dec = policy_for_create_task(prompt, {"content": "mua sữa"})
            self.assertEqual(
                dec.reason, PolicyReason.BARE_STATEMENT, f"Failed on: {prompt}"
            )

    def test_negate(self):
        cases = ["Đừng thêm việc mua sữa.", "Không cần ghi việc gọi mẹ."]
        for prompt in cases:
            dec = policy_for_create_task(prompt, {"content": "mua sữa"})
            self.assertEqual(
                dec.reason, PolicyReason.NEGATED_REQUEST, f"Failed on: {prompt}"
            )

    def test_unsupported(self):
        cases = ["Xóa task mua sữa.", "Sửa task thành mua bánh.", "Sửa task mua sữa."]
        for prompt in cases:
            dec = policy_for_create_task(prompt, {"content": "mua sữa"})
            self.assertEqual(
                dec.reason, PolicyReason.UNSUPPORTED_ACTION, f"Failed on: {prompt}"
            )

    def test_grounding(self):
        cases = [
            ("Note giúp: mua sữa", {"content": "Mua sữa"}),
            ("Thêm việc: mua sữa", {"content": "mua sữa lúc 8h"}),
            ("Thêm các việc:\nmua sữa\nmua bánh", {"content": "mua sữa"}),
            ("Thêm việc: mua sữa", {"content": "mua sữ"}),
        ]
        for prompt, args in cases:
            dec = policy_for_create_task(prompt, args)
            self.assertEqual(
                dec.reason, PolicyReason.CONTENT_NOT_GROUNDED, f"Failed on: {prompt}"
            )

    def test_missing_content(self):
        cases = ["Thêm việc.", "Ghi lại task:", "Note giúp tôi.", "Thêm việc:   "]
        for prompt in cases:
            dec = policy_for_create_task(prompt, {"content": ""})
            self.assertEqual(
                dec.reason, PolicyReason.MISSING_CONTENT, f"Failed on: {prompt}"
            )

    def test_invalid_arguments_and_edge_cases(self):
        self.assertEqual(
            policy_for_create_task(None, {"content": "mua sữa"}).reason,
            PolicyReason.INVALID_ARGUMENTS,
        )
        self.assertEqual(
            policy_for_create_task("  ", {"content": "mua sữa"}).reason,
            PolicyReason.INVALID_ARGUMENTS,
        )

        args = {"content": "mua sữa"}
        policy_for_create_task("Thêm việc: mua sữa", args)
        self.assertEqual(args, {"content": "mua sữa"})

        self.assertEqual(
            policy_for_create_task("tHêM ViỆc: Mua SỮA", {"content": "Mua SỮA"}).result,
            PolicyResult.ALLOW,
        )
        self.assertEqual(
            policy_for_create_task("Thêm việc: Mua SỮA", {"content": "mua sữa"}).reason,
            PolicyReason.CONTENT_NOT_GROUNDED,
        )

    def test_invalid_arguments(self):
        prompt = "Thêm việc: mua sữa"
        self.assertEqual(
            policy_for_create_task(prompt, None).reason, PolicyReason.INVALID_ARGUMENTS
        )
        self.assertEqual(
            policy_for_create_task(prompt, "chuỗi").reason,
            PolicyReason.INVALID_ARGUMENTS,
        )
        self.assertEqual(
            policy_for_create_task(prompt, []).reason, PolicyReason.INVALID_ARGUMENTS
        )
        self.assertEqual(
            policy_for_create_task(prompt, {}).reason, PolicyReason.INVALID_ARGUMENTS
        )
        self.assertEqual(
            policy_for_create_task(prompt, {"content": 123}).reason,
            PolicyReason.INVALID_ARGUMENTS,
        )
        self.assertEqual(
            policy_for_create_task(prompt, {"content": "   "}).reason,
            PolicyReason.INVALID_ARGUMENTS,
        )
        self.assertEqual(
            policy_for_create_task(
                prompt, {"content": "mua sữa", "other": "thing"}
            ).reason,
            PolicyReason.INVALID_ARGUMENTS,
        )


from unittest.mock import patch
from nexus.agent.policy import policy_for_tool


class TestPolicyForTool(unittest.TestCase):
    def test_valid_create_task_allowed(self):
        prompt = "Thêm việc: mua sữa"
        args = {"content": "mua sữa"}
        decision = policy_for_tool(prompt, "create_task", args)
        self.assertEqual(decision.result, PolicyResult.ALLOW)
        self.assertEqual(decision.reason, PolicyReason.EXPLICIT_CREATE)

    def test_valid_list_tasks_allowed(self):
        prompt = "Xem danh sách"
        args = {}
        decision = policy_for_tool(prompt, "list_tasks", args)
        self.assertEqual(decision.result, PolicyResult.ALLOW)
        self.assertEqual(decision.reason, PolicyReason.EXPLICIT_LIST)

    def test_sub_policy_rejection_preserved(self):
        # create_task rejections preserved
        self.assertEqual(
            policy_for_tool(
                "Đừng thêm việc mua sữa.", "create_task", {"content": "mua sữa"}
            ),
            ToolDecision(PolicyResult.REJECT, PolicyReason.NEGATED_REQUEST),
        )
        self.assertEqual(
            policy_for_tool("Sửa task mua sữa.", "create_task", {"content": "mua sữa"}),
            ToolDecision(PolicyResult.REJECT, PolicyReason.UNSUPPORTED_ACTION),
        )
        self.assertEqual(
            policy_for_tool("Mua sữa.", "create_task", {"content": "mua sữa"}),
            ToolDecision(PolicyResult.REJECT, PolicyReason.BARE_STATEMENT),
        )
        self.assertEqual(
            policy_for_tool("Thêm việc.", "create_task", {"content": ""}),
            ToolDecision(
                PolicyResult.NEEDS_CLARIFICATION, PolicyReason.MISSING_CONTENT
            ),
        )
        self.assertEqual(
            policy_for_tool(
                "Thêm việc: mua sữa", "create_task", {"content": "mua bánh"}
            ),
            ToolDecision(PolicyResult.REJECT, PolicyReason.CONTENT_NOT_GROUNDED),
        )

        # list_tasks rejections preserved
        self.assertEqual(
            policy_for_tool("Đừng xem danh sách.", "list_tasks", {}),
            ToolDecision(PolicyResult.REJECT, PolicyReason.NEGATED_REQUEST),
        )
        self.assertEqual(
            policy_for_tool("Xóa toàn bộ danh sách.", "list_tasks", {}),
            ToolDecision(PolicyResult.REJECT, PolicyReason.UNSUPPORTED_ACTION),
        )
        self.assertEqual(
            policy_for_tool("Hôm nay trời đẹp.", "list_tasks", {}),
            ToolDecision(PolicyResult.REJECT, PolicyReason.BARE_STATEMENT),
        )
        self.assertEqual(
            policy_for_tool("Xem danh sách", "list_tasks", {"invalid": 1}),
            ToolDecision(PolicyResult.REJECT, PolicyReason.INVALID_ARGUMENTS),
        )

    def test_unsupported_tool(self):
        for tool_name in [
            "delete_task",
            "update_task",
            "complete_task",
            "query_database",
            "random_tool",
        ]:
            dec = policy_for_tool("Xem danh sách", tool_name, {})
            self.assertEqual(dec.result, PolicyResult.REJECT, f"Failed for {tool_name}")
            self.assertEqual(
                dec.reason, PolicyReason.UNSUPPORTED_TOOL, f"Failed for {tool_name}"
            )

    def test_invalid_tool_name(self):
        for invalid_name in [
            "",
            "   ",
            None,
            123,
            45.6,
            ["create_task"],
            {"name": "create_task"},
        ]:
            dec = policy_for_tool(
                "Thêm việc: mua sữa", invalid_name, {"content": "mua sữa"}
            )
            self.assertEqual(
                dec.result, PolicyResult.REJECT, f"Failed for {invalid_name!r}"
            )
            self.assertEqual(
                dec.reason,
                PolicyReason.INVALID_ARGUMENTS,
                f"Failed for {invalid_name!r}",
            )

    def test_pass_through_exact_inputs_and_unmutated_arguments(self):
        args = {"content": "mua sữa"}
        policy_for_tool("Thêm việc: mua sữa", "create_task", args)
        self.assertEqual(args, {"content": "mua sữa"})

        empty_args = {}
        policy_for_tool("Xem danh sách", "list_tasks", empty_args)
        self.assertEqual(empty_args, {})

        # Verify exact arguments passed to sub-policies
        with patch("nexus.agent.policy.policy_for_create_task") as mock_create:
            mock_create.return_value = ToolDecision(
                PolicyResult.ALLOW, PolicyReason.EXPLICIT_CREATE
            )
            prompt = "Thêm việc: mua sữa"
            policy_for_tool(prompt, "create_task", args)
            mock_create.assert_called_once_with(prompt, args)

        with patch("nexus.agent.policy.policy_for_list_tasks") as mock_list:
            mock_list.return_value = ToolDecision(
                PolicyResult.ALLOW, PolicyReason.EXPLICIT_LIST
            )
            prompt = "Xem danh sách"
            policy_for_tool(prompt, "list_tasks", empty_args)
            mock_list.assert_called_once_with(prompt, empty_args)

    def test_no_tool_execution_or_database_access(self):
        with patch("nexus.agent.tools.execute_tool") as mock_exec:
            policy_for_tool("Thêm việc: mua sữa", "create_task", {"content": "mua sữa"})
            policy_for_tool("Xem danh sách", "list_tasks", {})
            policy_for_tool("Xóa task", "delete_task", {})
            mock_exec.assert_not_called()


if __name__ == "__main__":
    unittest.main()
