import unittest

from nexus.agent.policy import (
    PolicyReason,
    PolicyResult,
    RequestKind,
    classify_request,
    deadline_scope_reply,
    policy_for_list_tasks,
    policy_for_list_tasks_by_deadline,
    policy_for_tool,
)


class DeadlineQueryPolicyTests(unittest.TestCase):
    def test_supported_framings_map_to_exact_scope(self):
        cases = (
            ("Xem việc đến hạn hôm nay.", "today"),
            ("Ngày mai có những task nào đến hạn?", "tomorrow"),
            ("Cho tôi xem các việc quá hạn.", "overdue"),
            ("Liệt kê những task đã trễ hạn!", "overdue"),
        )
        for prompt, scope in cases:
            with self.subTest(prompt=prompt):
                decision = policy_for_list_tasks_by_deadline(
                    prompt, {"scope": scope}
                )
                self.assertEqual(decision.result, PolicyResult.ALLOW)
                self.assertEqual(
                    decision.reason, PolicyReason.EXPLICIT_DEADLINE_QUERY
                )
                self.assertEqual(classify_request(prompt), RequestKind.DEADLINE_QUERY)

    def test_scope_mismatch_and_argument_schema_are_rejected(self):
        prompt = "Xem việc đến hạn hôm nay."
        mismatch = policy_for_list_tasks_by_deadline(
            prompt, {"scope": "tomorrow"}
        )
        self.assertEqual(mismatch.reason, PolicyReason.DEADLINE_SCOPE_MISMATCH)
        for arguments in ({}, {"scope": "today", "extra": 1}, {"scope": True}, {"scope": "week"}):
            with self.subTest(arguments=arguments):
                decision = policy_for_list_tasks_by_deadline(prompt, arguments)
                self.assertEqual(decision.reason, PolicyReason.INVALID_ARGUMENTS)

    def test_missing_scope_is_a_clarification_and_followup_is_narrow(self):
        for prompt in ("Xem việc theo hạn", "Cho mình xem các task đến hạn."):
            with self.subTest(prompt=prompt):
                decision = policy_for_list_tasks_by_deadline(
                    prompt, {"scope": "today"}
                )
                self.assertEqual(decision.result, PolicyResult.NEEDS_CLARIFICATION)
                self.assertEqual(
                    decision.reason, PolicyReason.MISSING_DEADLINE_SCOPE
                )
                self.assertEqual(
                    classify_request(prompt), RequestKind.MISSING_DEADLINE_SCOPE
                )
        self.assertEqual(deadline_scope_reply("hôm nay."), "today")
        self.assertEqual(deadline_scope_reply("mai"), "tomorrow")
        self.assertEqual(deadline_scope_reply("quá hạn!"), "overdue")
        self.assertIsNone(deadline_scope_reply("tuần này"))

    def test_ambiguous_negated_and_wrong_tools_do_not_get_authorized(self):
        ambiguous = policy_for_list_tasks_by_deadline(
            "Hôm nay tôi có gì?", {"scope": "today"}
        )
        self.assertEqual(ambiguous.reason, PolicyReason.BARE_STATEMENT)
        negated = policy_for_list_tasks_by_deadline(
            "Đừng xem việc đến hạn hôm nay.", {"scope": "today"}
        )
        self.assertEqual(negated.reason, PolicyReason.NEGATED_REQUEST)
        generic = policy_for_list_tasks(
            "Xem việc đến hạn hôm nay.", {}
        )
        self.assertEqual(generic.reason, PolicyReason.DEADLINE_SCOPE_MISMATCH)
        wrong_query_tool = policy_for_tool(
            "Xem danh sách", "list_tasks_by_deadline", {"scope": "today"}
        )
        self.assertEqual(wrong_query_tool.reason, PolicyReason.BARE_STATEMENT)


if __name__ == "__main__":
    unittest.main()
