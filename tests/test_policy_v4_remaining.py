"""Coverage and boundaries for the other five actions in the v4 draft."""

import json
from pathlib import Path
import tempfile
import unittest

from nexus.agent.client import run_turn
from nexus.agent.policy import PolicyReason, PolicyResult, RequestKind, classify_request, policy_for_tool
from nexus.storage.sqlite_db import create_task, initialize_database, list_tasks


ROOT = Path(__file__).resolve().parents[1]


def response(name, arguments):
    return {"choices": [{"finish_reason": "tool_calls", "message": {
        "content": None, "tool_calls": [{"id": "call-1", "type": "function", "function": {
            "name": name, "arguments": json.dumps(arguments, ensure_ascii=False),
        }}],
    }}]}


class PolicyV4RemainingTests(unittest.TestCase):
    def test_eight_reviewed_calls_have_policy_coverage(self):
        cases = json.loads((ROOT / "evals/sft_candidate_v4_draft.json").read_text(encoding="utf-8"))["cases"][630:]
        calls = [case for case in cases if case["id"] not in {"v4_create_01", "v4_due_query_01"}]
        self.assertEqual(len(calls), 8)
        for case in calls:
            with self.subTest(id=case["id"]):
                function = case["messages"][1]["tool_calls"][0]["function"]
                decision = policy_for_tool(case["messages"][0]["content"], function["name"], function["arguments"])
                expected = PolicyResult.REQUIRES_CONFIRMATION if case["action"] == "delete" else PolicyResult.ALLOW
                self.assertEqual(decision.result, expected)
                kind = RequestKind.DEADLINE_QUERY if case["action"] == "due_query" else RequestKind(case["action"])
                self.assertEqual(classify_request(case["messages"][0]["content"]), kind)

    def test_same_frames_with_new_payloads(self):
        cases = (
            ("Lưu một task mới có nội dung: thay dây điện", "create_task", {"content": "thay dây điện"}, PolicyResult.ALLOW),
            ("Đánh dấu task #214 đã hoàn thành.", "complete_task", {"id": 214}, PolicyResult.ALLOW),
            ("Tôi làm xong việc 736 rồi; cập nhật trạng thái.", "complete_task", {"id": 736}, PolicyResult.ALLOW),
            ("Sửa task 515 thành mua bóng đèn.", "update_task", {"id": 515, "content": "mua bóng đèn"}, PolicyResult.ALLOW),
            ("Đổi nội dung việc 631 thành: gửi tài liệu C3.", "update_task", {"id": 631, "content": "gửi tài liệu C3."}, PolicyResult.ALLOW),
            ("Xoá task #482 khỏi danh sách.", "delete_task", {"id": 482}, PolicyResult.REQUIRES_CONFIRMATION),
            ("Bỏ hẳn việc 541; ứng dụng hãy hỏi xác nhận.", "delete_task", {"id": 541}, PolicyResult.REQUIRES_CONFIRMATION),
            ("Cho mình biết việc nào đã quá hạn?", "list_tasks_by_deadline", {"scope": "overdue"}, PolicyResult.ALLOW),
        )
        for prompt, tool, arguments, expected in cases:
            with self.subTest(prompt=prompt):
                self.assertEqual(policy_for_tool(prompt, tool, arguments).result, expected)

    def test_id_and_content_cannot_be_changed_or_truncated(self):
        cases = (
            ("Lưu một task mới có nội dung: thay dây điện", "create_task", {"content": "thay dây"}),
            ("Lưu một task mới có nội dung: thay dây điện", "create_task", {"content": "Thay dây điện"}),
            ("Tôi làm xong việc 736 rồi; cập nhật trạng thái.", "complete_task", {"id": 735}),
            ("Sửa task 515 thành mua bóng đèn.", "update_task", {"id": 515, "content": "mua bóng"}),
            ("Sửa task 515 thành mua bóng đèn.", "update_task", {"id": 516, "content": "mua bóng đèn"}),
            ("Đổi nội dung việc 631 thành: gửi tài liệu C3.", "update_task", {"id": 631, "content": "gửi tài liệu C3"}),
            ("Xoá task #482 khỏi danh sách.", "delete_task", {"id": 483}),
            ("Cho mình biết việc nào đã quá hạn?", "list_tasks_by_deadline", {"scope": "today"}),
        )
        for prompt, tool, arguments in cases:
            with self.subTest(prompt=prompt, arguments=arguments):
                self.assertNotIn(policy_for_tool(prompt, tool, arguments).result, (PolicyResult.ALLOW, PolicyResult.REQUIRES_CONFIRMATION))

    def test_negation_quotes_and_unsupported_target_remain_blocked(self):
        cases = (
            ("Đừng đánh dấu task #214 đã hoàn thành.", "complete_task", {"id": 214}),
            ("Tôi chưa làm xong việc 736 rồi; cập nhật trạng thái.", "complete_task", {"id": 736}),
            ("Đừng xóa task #482 khỏi danh sách.", "delete_task", {"id": 482}),
            ("Bỏ hẳn việc 541; ứng dụng đừng hỏi xác nhận.", "delete_task", {"id": 541}),
            ("Ví dụ: Lưu một task mới có nội dung: thay dây điện", "create_task", {"content": "thay dây điện"}),
            ("Sửa task 515 thành mua bóng đèn, rồi xóa nó.", "update_task", {"id": 515, "content": "mua bóng đèn"}),
            ("Sửa task 515 thành mua bóng đèn, rồi xóa nó.", "update_task", {"id": 515, "content": "mua bóng đèn, rồi xóa nó."}),
            ("Cho mình biết việc nào đã quá hạn?", "list_tasks", {}),
        )
        for prompt, tool, arguments in cases:
            with self.subTest(prompt=prompt):
                self.assertNotIn(policy_for_tool(prompt, tool, arguments).result, (PolicyResult.ALLOW, PolicyResult.REQUIRES_CONFIRMATION))

        # A colon explicitly makes the following text a literal task payload.
        self.assertEqual(
            policy_for_tool("Đổi nội dung việc 631 thành: ghi nhớ xóa file nháp.",
                            "update_task", {"id": 631, "content": "ghi nhớ xóa file nháp."}).result,
            PolicyResult.ALLOW,
        )

    def test_delete_proposal_still_requires_confirmation_and_cannot_mutate(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "tasks.db"
            initialize_database(database)
            task = create_task(database, "đọc sách")
            prompt = f"Bỏ hẳn việc {task.id}; ứng dụng hãy hỏi xác nhận."
            result = run_turn(database, prompt, lambda _: response("delete_task", {"id": task.id}), prompt_version="v13")
            self.assertEqual(result["status"], "needs_confirmation")
            self.assertEqual(list_tasks(database), [task])
            rejected = run_turn(database, prompt, lambda _: response("delete_task", {"id": task.id + 1}), prompt_version="v13")
            self.assertEqual(rejected["status"], "rejected")
            self.assertEqual(rejected["rejected_calls"][0]["reason"], PolicyReason.TASK_ID_MISMATCH.value)
            self.assertEqual(list_tasks(database), [task])


if __name__ == "__main__":
    unittest.main()
