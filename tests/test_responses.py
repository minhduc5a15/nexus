import copy
import unittest

from nexus.agent.policy import PolicyReason
from nexus.agent.responses import format_rejection, format_tool_result


class ToolResponseFormatterTests(unittest.TestCase):
    def test_rejection_reply_is_fixed_by_reason(self):
        self.assertEqual(format_rejection(PolicyReason.UNSUPPORTED_ACTION),
                         "NEXUS hiện chỉ hỗ trợ thêm, xem, hoàn thành, sửa và xóa việc theo ID; ngoài ra có thể đặt hạn theo ID.")
        self.assertEqual(format_rejection(PolicyReason.CONTENT_BOUNDARY_MISMATCH),
                         "Tôi chưa lưu việc vì nội dung không khớp lời bạn. Hãy viết lại yêu cầu.")
        self.assertEqual(format_rejection(PolicyReason.INVALID_ARGUMENTS),
                         "Không có thao tác nào được thực hiện.")

    def test_create_single_task_format(self):
        result = {"tasks": [{"id": 1, "content": "mua sữa", "completed": False, "due_at": None}]}
        formatted = format_tool_result("create_task", result)
        self.assertEqual(formatted, "Đã thêm [1] mua sữa")

    def test_create_multiple_tasks_format(self):
        result = {
            "tasks": [
                {"id": 1, "content": "mua sữa", "completed": False, "due_at": None},
                {"id": 2, "content": "gọi cho mẹ", "completed": False, "due_at": None},
            ]
        }
        expected = "Đã thêm 2 việc:\n[1] mua sữa\n[2] gọi cho mẹ"
        self.assertEqual(format_tool_result("create_task", result), expected)

        result_3 = {
            "tasks": [
                {"id": 1, "content": "việc 1", "completed": False, "due_at": None},
                {"id": 2, "content": "việc 2", "completed": False, "due_at": None},
                {"id": 3, "content": "việc 3", "completed": False, "due_at": None},
            ]
        }
        expected_3 = "Đã thêm 3 việc:\n[1] việc 1\n[2] việc 2\n[3] việc 3"
        self.assertEqual(format_tool_result("create_task", result_3), expected_3)

    def test_list_tasks_empty_format(self):
        result = {"tasks": []}
        self.assertEqual(format_tool_result("list_tasks", result), "Danh sách trống.")

    def test_list_tasks_single_and_multiple_format(self):
        single_result = {"tasks": [{"id": 1, "content": "mua sữa", "completed": False, "due_at": None}]}
        expected_single = "Danh sách hiện có 1 việc:\n[1] [ ] mua sữa"
        self.assertEqual(
            format_tool_result("list_tasks", single_result), expected_single
        )

        multiple_result = {
            "tasks": [
                {"id": 1, "content": "mua sữa", "completed": False, "due_at": None},
                {"id": 2, "content": "gọi cho mẹ", "completed": True, "due_at": None},
            ]
        }
        expected_multiple = "Danh sách hiện có 2 việc:\n[1] [ ] mua sữa\n[2] [x] gọi cho mẹ"
        self.assertEqual(
            format_tool_result("list_tasks", multiple_result), expected_multiple
        )

    def test_preserves_casing_punctuation_and_whitespace_verbatim(self):
        content = "   Mua SỮA & Táo! (loại 1.5% béo)   \t "
        create_result = {"tasks": [{"id": 10, "content": content, "completed": False, "due_at": None}]}
        self.assertEqual(
            format_tool_result("create_task", create_result),
            f"Đã thêm [10] {content}",
        )

        list_result = {"tasks": [{"id": 10, "content": content, "completed": False, "due_at": None}]}
        self.assertEqual(
            format_tool_result("list_tasks", list_result),
            f"Danh sách hiện có 1 việc:\n[10] [ ] {content}",
        )

    def test_rejects_unknown_or_invalid_tools(self):
        invalid_tools = [
            "delete_task",
            "update_task",
            "CREATE_TASK",
            "list",
            "create",
            "",
            "   ",
            None,
            123,
            True,
            ["create_task"],
            {"name": "create_task"},
        ]
        sample_result = {"tasks": [{"id": 1, "content": "mua sữa", "completed": False, "due_at": None}]}
        for tool in invalid_tools:
            with self.subTest(tool=tool):
                with self.assertRaises(ValueError):
                    format_tool_result(tool, sample_result)

    def test_rejects_invalid_result_type(self):
        invalid_results = [
            None,
            "tasks",
            123,
            True,
            False,
            [{"id": 1, "content": "mua sữa"}],
            ("tasks", []),
        ]
        for result in invalid_results:
            with self.subTest(result=result):
                with self.assertRaises(ValueError):
                    format_tool_result("create_task", result)

    def test_rejects_missing_or_extra_fields_in_result(self):
        # Missing 'tasks' key
        with self.assertRaises(ValueError):
            format_tool_result("list_tasks", {})

        # Extra key in result
        with self.assertRaises(ValueError):
            format_tool_result(
                "list_tasks",
                {"tasks": [{"id": 1, "content": "mua sữa"}], "status": "ok"},
            )
        with self.assertRaises(ValueError):
            format_tool_result("create_task", {"tasks": [], "extra": 123})

    def test_rejects_invalid_tasks_type(self):
        invalid_tasks_values = [
            None,
            "mua sữa",
            123,
            True,
            {"id": 1, "content": "mua sữa"},
            ({"id": 1, "content": "mua sữa"},),
        ]
        for val in invalid_tasks_values:
            with self.subTest(tasks=val):
                with self.assertRaises(ValueError):
                    format_tool_result("create_task", {"tasks": val})

    def test_rejects_invalid_task_item_type(self):
        invalid_items = [
            None,
            "not a dict",
            123,
            True,
            False,
            [1, "mua sữa"],
            ("id", 1),
        ]
        for item in invalid_items:
            with self.subTest(item=item):
                with self.assertRaises(ValueError):
                    format_tool_result("create_task", {"tasks": [item]})

    def test_rejects_missing_or_extra_fields_in_task(self):
        # Missing 'id'
        with self.assertRaises(ValueError):
            format_tool_result("create_task", {"tasks": [{"content": "mua sữa"}]})

        # Missing 'content'
        with self.assertRaises(ValueError):
            format_tool_result("create_task", {"tasks": [{"id": 1}]})

        # Extra field in task
        with self.assertRaises(ValueError):
            format_tool_result(
                "create_task",
                {"tasks": [{"id": 1, "content": "mua sữa", "done": False}]},
            )

    def test_rejects_zero_negative_boolean_or_string_id(self):
        invalid_ids = [
            0,
            -1,
            -999,
            True,
            False,
            "1",
            "id-1",
            1.0,
            1.5,
            None,
            [1],
        ]
        for task_id in invalid_ids:
            with self.subTest(task_id=task_id):
                with self.assertRaises(ValueError):
                    format_tool_result(
                        "create_task",
                        {"tasks": [{"id": task_id, "content": "mua sữa"}]},
                    )

    def test_rejects_empty_whitespace_only_or_non_string_content(self):
        invalid_contents = [
            "",
            "   ",
            "\t",
            "\n",
            "\r\n  \t",
            None,
            123,
            True,
            False,
            ["mua sữa"],
            {"text": "mua sữa"},
        ]
        for content in invalid_contents:
            with self.subTest(content=content):
                with self.assertRaises(ValueError):
                    format_tool_result(
                        "create_task",
                        {"tasks": [{"id": 1, "content": content}]},
                    )

    def test_create_task_rejects_empty_tasks_list(self):
        with self.assertRaises(ValueError):
            format_tool_result("create_task", {"tasks": []})

    def test_formatter_does_not_mutate_input_result(self):
        original = {
            "tasks": [
                {"id": 1, "content": "mua sữa", "completed": False, "due_at": None},
                {"id": 2, "content": "gọi cho mẹ", "completed": False, "due_at": None},
            ]
        }
        deep_copied = copy.deepcopy(original)

        # Call formatter for create_task
        _ = format_tool_result("create_task", original)
        self.assertEqual(original, deep_copied)

        # Call formatter for list_tasks
        _ = format_tool_result("list_tasks", original)
        self.assertEqual(original, deep_copied)

    def test_supports_keyword_name(self):
        result = {"tasks": [{"id": 1, "content": "mua sữa", "completed": False, "due_at": None}]}
        self.assertEqual(
            format_tool_result(name="create_task", result=result),
            "Đã thêm [1] mua sữa",
        )
        self.assertEqual(
            format_tool_result(name="list_tasks", result=result),
            "Danh sách hiện có 1 việc:\n[1] [ ] mua sữa",
        )

    def test_complete_result_formats_all_statuses(self):
        task = {"id": 3, "content": "gọi mẹ", "completed": True, "due_at": None}
        self.assertEqual(
            format_tool_result("complete_task", {"status": "completed", "task": task}),
            "Đã hoàn thành [3] gọi mẹ",
        )
        self.assertEqual(
            format_tool_result(
                "complete_task", {"status": "already_completed", "task": task}
            ),
            "Việc [3] đã hoàn thành trước đó: gọi mẹ",
        )
        self.assertEqual(
            format_tool_result(
                "complete_task", {"status": "not_found", "task": None}
            ),
            "Không tìm thấy việc có ID đã yêu cầu.",
        )

    def test_complete_result_schema_is_strict(self):
        valid_task = {"id": 1, "content": "mua sữa", "completed": True, "due_at": None}
        invalid = [
            {"status": "completed", "task": None},
            {"status": "not_found", "task": valid_task},
            {"status": "unknown", "task": valid_task},
            {"status": "completed", "task": {**valid_task, "completed": False, "due_at": None}},
            {"status": "completed", "task": {**valid_task, "completed": 1}},
            {"status": "completed", "task": valid_task, "extra": True},
        ]
        for result in invalid:
            with self.subTest(result=result), self.assertRaises(ValueError):
                format_tool_result("complete_task", result)

    def test_update_result_formats_all_statuses(self):
        task = {"id": 3, "content": "mua sữa không đường", "completed": True, "due_at": None}
        self.assertEqual(
            format_tool_result("update_task", {"status": "updated", "task": task}),
            "Đã sửa [3] thành: mua sữa không đường",
        )
        self.assertEqual(
            format_tool_result("update_task", {"status": "unchanged", "task": task}),
            "Việc [3] đã có nội dung này: mua sữa không đường",
        )
        self.assertEqual(
            format_tool_result("update_task", {"status": "not_found", "task": None}),
            "Không tìm thấy việc có ID đã yêu cầu.",
        )

    def test_update_result_schema_is_strict(self):
        valid_task = {"id": 1, "content": "mới", "completed": False, "due_at": None}
        invalid = [
            {"status": "updated", "task": None},
            {"status": "not_found", "task": valid_task},
            {"status": "unknown", "task": valid_task},
            {"status": "updated", "task": {**valid_task, "completed": 0}},
            {"status": "updated", "task": valid_task, "extra": True},
        ]
        for result in invalid:
            with self.subTest(result=result), self.assertRaises(ValueError):
                format_tool_result("update_task", result)

    def test_delete_result_formats_all_statuses(self):
        task = {"id": 3, "content": "gọi mẹ", "completed": False, "due_at": None}
        self.assertEqual(
            format_tool_result("delete_task", {"status": "deleted", "task": task}),
            "Đã xóa [3] gọi mẹ",
        )
        self.assertEqual(
            format_tool_result("delete_task", {"status": "not_found", "task": None}),
            "Không tìm thấy việc có ID đã yêu cầu.",
        )
        stale = format_tool_result(
            "delete_task",
            {"status": "stale", "task": {**task, "completed": True, "due_at": None}},
        )
        self.assertIn("đã thay đổi", stale)
        self.assertIn("[x]", stale)

    def test_delete_result_schema_is_strict(self):
        task = {"id": 1, "content": "mới", "completed": False, "due_at": None}
        for result in (
            {"status": "deleted", "task": None},
            {"status": "not_found", "task": task},
            {"status": "unknown", "task": task},
            {"status": "deleted", "task": task, "extra": True},
        ):
            with self.subTest(result=result), self.assertRaises(ValueError):
                format_tool_result("delete_task", result)


if __name__ == "__main__":
    unittest.main()
