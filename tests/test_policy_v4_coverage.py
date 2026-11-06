"""Regression for reviewed DEADLINE/LIST intents and nearby non-commands."""

from datetime import datetime
import json
from pathlib import Path
import tempfile
import unittest

from nexus.agent.client import run_turn
from nexus.agent.policy import PolicyResult, RequestKind, classify_request, policy_for_tool
from nexus.core.deadlines import VIETNAM_TIMEZONE, format_deadline
from nexus.storage.sqlite_db import complete_task, create_task, initialize_database, list_tasks


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = datetime(2026, 10, 21, 12, tzinfo=VIETNAM_TIMEZONE)


class PolicyV4CoverageTests(unittest.TestCase):
    def test_target_deadline_and_list_calls_are_authorized(self):
        cases = json.loads((ROOT / "evals/sft_candidate_v4_draft.json").read_text(encoding="utf-8"))["cases"][600:620]
        for case in cases:
            with self.subTest(id=case["id"]):
                prompt = case["messages"][0]["content"]
                call = case["messages"][1]["tool_calls"][0]["function"]
                decision = policy_for_tool(prompt, call["name"], call["arguments"], reference_time=REFERENCE)
                self.assertEqual(decision.result, PolicyResult.ALLOW)
                self.assertEqual(classify_request(prompt), RequestKind(case["action"]))

    def test_same_frames_with_other_payloads_are_supported(self):
        cases = (
            ("Cho tôi đặt hạn việc 145: 7 giờ 15 phút sáng ngày mai.", "set_task_deadline", {"id": 145, "when": "7 giờ 15 phút sáng ngày mai"}),
            ("Chốt deadline của task #973 ở 18:25 hôm nay.", "set_task_deadline", {"id": 973, "when": "18:25 hôm nay"}),
            ("Việc 264: đặt hạn 9 giờ tối mai.", "set_task_deadline", {"id": 264, "when": "9 giờ tối mai"}),
            ("Hạn của task #741, hãy chuyển sang 08:40 ngày mai.", "set_task_deadline", {"id": 741, "when": "08:40 ngày mai"}),
            ("Với việc ID 359, hạn mới là 10:05 hôm nay; ghi lại.", "set_task_deadline", {"id": 359, "when": "10:05 hôm nay"}),
            ("Đọc danh sách NEXUS, kể cả task đã xong.", "list_tasks", {}),
            ("Đưa ra mọi đầu việc hiện có, bao gồm các mục đã hoàn thành.", "list_tasks", {}),
            ("Tôi cần xem cả việc chưa làm lẫn việc đã làm rồi.", "list_tasks", {}),
            ("Xem toàn bộ danh sách task kèm thời hạn.", "list_tasks", {}),
        )
        for prompt, tool, arguments in cases:
            with self.subTest(prompt=prompt):
                self.assertEqual(policy_for_tool(prompt, tool, arguments, reference_time=REFERENCE).result, PolicyResult.ALLOW)

    def test_neighboring_cases_remain_blocked(self):
        cases = (
            ("Câu ví dụ là ‘đặt hạn việc 621 lúc 8 giờ sáng mai’; hãy giải thích ý nghĩa thôi.", "set_task_deadline", {"id": 621, "when": "8 giờ sáng mai"}),
            ("Đừng đặt hạn việc 622 vào 19:20 hôm nay.", "set_task_deadline", {"id": 622, "when": "19:20 hôm nay"}),
            ("Trong tài liệu có câu ‘xem danh sách kể cả mục hoàn thành’; tôi chỉ đang trích dẫn.", "list_tasks", {}),
            ("Không cần hiển thị danh sách, dù có việc đã hoàn thành.", "list_tasks", {}),
            ("Chỉ hiện riêng những task đã hoàn thành, đừng hiện task chưa xong.", "list_tasks", {}),
            ("Xem riêng task đã hoàn thành.", "list_tasks", {}),
            ("Xem danh sách, nhưng đừng xem nữa.", "list_tasks", {}),
            ("Xem danh sách, đừng hiện task chưa xong.", "list_tasks", {}),
            ("Xem toàn bộ danh sách, rồi xóa task 3.", "list_tasks", {}),
            ("Xem danh sách, xóa task 3 luôn.", "list_tasks", {}),
        )
        for prompt, tool, arguments in cases:
            with self.subTest(prompt=prompt):
                self.assertNotEqual(policy_for_tool(prompt, tool, arguments, reference_time=REFERENCE).result, PolicyResult.ALLOW)

    def test_wrong_id_time_and_scope_are_not_authorized(self):
        prompt = "Đặt hạn việc 358 ở 8 giờ sáng mai, đúng ID đó."
        for arguments in (
            {"id": 8, "when": "8 giờ sáng mai"},
            {"id": 358, "when": "8 giờ sáng"},
            {"id": 358, "when": "08:00 ngày mai"},
        ):
            with self.subTest(arguments=arguments):
                self.assertNotEqual(policy_for_tool(prompt, "set_task_deadline", arguments, reference_time=REFERENCE).result, PolicyResult.ALLOW)
        self.assertNotEqual(policy_for_tool("Xem việc đến hạn hôm nay.", "list_tasks", {}).result, PolicyResult.ALLOW)
        for due_prompt in (
            "Mở nhóm đầu việc có mốc chót thuộc hôm nay.",
            "Cho tôi xem task nào đã quá hạn.",
            "Xem task có deadline ngày mai.",
            "Show tasks due on 2032-04-18.",
            "Đưa ra các task chưa làm xong mà thời hạn đã trôi qua.",
            "Show các task overdue giúp mình.",
            "Mở nhóm task đã trễ deadline.",
            "Liệt kê task đến kỳ xử lý ngày mai.",
            "Cho tôi xem việc phải làm xong vào ngày mai.",
        ):
            with self.subTest(due_prompt=due_prompt):
                self.assertNotEqual(policy_for_tool(due_prompt, "list_tasks", {}).result, PolicyResult.ALLOW)

    def test_runtime_uses_real_sqlite_state_and_rejects_wrong_id(self):
        def response(name, arguments):
            return {"choices": [{"finish_reason": "tool_calls", "message": {
                "content": None, "tool_calls": [{"id": "call-1", "type": "function", "function": {
                    "name": name, "arguments": json.dumps(arguments, ensure_ascii=False),
                }}],
            }}]}

        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "tasks.db"
            initialize_database(database)
            first = create_task(database, "đọc sách")
            second = create_task(database, "gọi mẹ")
            complete_task(database, second.id)

            listed = run_turn(
                database, "Cho tôi xem danh sách, kể cả các việc đã hoàn thành.",
                lambda _: response("list_tasks", {}), prompt_version="v13",
                reference_time=REFERENCE,
            )
            self.assertEqual(listed["status"], "executed")
            self.assertIn(f"[{second.id}] [x] gọi mẹ", listed["reply"])
            self.assertEqual(len(list_tasks(database)), 2)

            prompt = f"Cho tôi đặt hạn việc {first.id}: 6 giờ 10 phút sáng mai."
            rejected = run_turn(
                database, prompt,
                lambda _: response("set_task_deadline", {"id": second.id, "when": "6 giờ 10 phút sáng mai"}),
                prompt_version="v13", reference_time=REFERENCE,
            )
            self.assertEqual(rejected["status"], "rejected")
            self.assertIsNone(list_tasks(database)[0].due_at)

            updated = run_turn(
                database, prompt,
                lambda _: response("set_task_deadline", {"id": first.id, "when": "6 giờ 10 phút sáng mai"}),
                prompt_version="v13", reference_time=REFERENCE,
            )
            self.assertEqual(updated["status"], "executed")
            self.assertEqual(format_deadline(list_tasks(database)[0].due_at), "22/10/2026 06:10")


if __name__ == "__main__":
    unittest.main()
