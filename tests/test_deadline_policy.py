import unittest
from datetime import datetime

from nexus.agent.policy import (
    PolicyReason,
    PolicyResult,
    RequestKind,
    classify_request,
    deadline_request_fields,
    policy_for_set_task_deadline,
    policy_for_tool,
)
from nexus.core.deadlines import VIETNAM_TIMEZONE


class DeadlinePolicyTests(unittest.TestCase):
    def setUp(self):
        self.reference = datetime(2026, 9, 26, 12, 0, tzinfo=VIETNAM_TIMEZONE)

    def decide(self, prompt, arguments):
        return policy_for_set_task_deadline(
            prompt, arguments, reference_time=self.reference
        )

    def test_allows_supported_framings_with_verbatim_time(self):
        cases = (
            ("Đặt hạn việc 3 lúc 8 giờ sáng mai.", {"id": 3, "when": "8 giờ sáng mai"}),
            ("Đặt deadline task #3 là 27/09/2026 08:00.", {"id": 3, "when": "27/09/2026 08:00"}),
            ("Đặt thời hạn cho việc 3 vào 08:00 ngày 27/09/2026.", {"id": 3, "when": "08:00 ngày 27/09/2026"}),
            ("Đặt hạn việc 3 lúc 8 giờ sáng mai.", {"id": 3, "when": "8 giờ sáng mai."}),
        )
        for prompt, arguments in cases:
            with self.subTest(prompt=prompt, arguments=arguments):
                decision = self.decide(prompt, arguments)
                self.assertEqual(decision.result, PolicyResult.ALLOW)
                self.assertEqual(decision.reason, PolicyReason.EXPLICIT_DEADLINE)

    def test_requires_exact_id_and_exact_complete_time_span(self):
        cases = (
            ({"id": 4, "when": "8 giờ sáng mai"}, PolicyReason.TASK_ID_MISMATCH),
            ({"id": 3, "when": "08:00 ngày mai"}, PolicyReason.CONTENT_NOT_GROUNDED),
            ({"id": 3, "when": "8 giờ sáng"}, PolicyReason.CONTENT_BOUNDARY_MISMATCH),
            ({"id": 3, "when": "mai"}, PolicyReason.CONTENT_BOUNDARY_MISMATCH),
        )
        prompt = "Đặt hạn việc 3 lúc 8 giờ sáng mai."
        for arguments, reason in cases:
            with self.subTest(arguments=arguments):
                decision = self.decide(prompt, arguments)
                self.assertEqual(decision.result, PolicyResult.REJECT)
                self.assertEqual(decision.reason, reason)

    def test_missing_multiple_invalid_negated_and_content_lookup(self):
        cases = (
            ("Đặt hạn lúc 8 giờ sáng mai.", {"id": 1, "when": "8 giờ sáng mai"}, PolicyResult.NEEDS_CLARIFICATION, PolicyReason.MISSING_DEADLINE_ID),
            ("Đặt hạn việc 1 và 2 lúc 8 giờ sáng mai.", {"id": 1, "when": "8 giờ sáng mai"}, PolicyResult.NEEDS_CLARIFICATION, PolicyReason.MULTIPLE_TASK_IDS),
            ("Đặt hạn việc 3.", {"id": 3, "when": "8 giờ sáng mai"}, PolicyResult.NEEDS_CLARIFICATION, PolicyReason.MISSING_DEADLINE_TIME),
            ("Đặt hạn việc 3 lúc tuần sau.", {"id": 3, "when": "tuần sau"}, PolicyResult.NEEDS_CLARIFICATION, PolicyReason.INVALID_DEADLINE_TIME),
            ("Đừng đặt hạn việc 3 lúc 8 giờ sáng mai.", {"id": 3, "when": "8 giờ sáng mai"}, PolicyResult.REJECT, PolicyReason.NEGATED_REQUEST),
            ("Đặt hạn việc mua sữa lúc 8 giờ sáng mai.", {"id": 3, "when": "8 giờ sáng mai"}, PolicyResult.REJECT, PolicyReason.BARE_STATEMENT),
        )
        for prompt, arguments, result, reason in cases:
            with self.subTest(prompt=prompt):
                decision = self.decide(prompt, arguments)
                self.assertEqual(decision.result, result)
                self.assertEqual(decision.reason, reason)

    def test_argument_schema_is_strict_and_unmutated(self):
        prompt = "Đặt hạn việc 3 lúc 8 giờ sáng mai."
        invalid = (
            None, [], {"id": True, "when": "8 giờ sáng mai"},
            {"id": 3.0, "when": "8 giờ sáng mai"},
            {"id": 3, "when": ""}, {"id": 3, "when": "8 giờ\nmai"},
            {"id": 3, "when": "8 giờ sáng mai", "due_at": 1},
        )
        for arguments in invalid:
            with self.subTest(arguments=arguments):
                self.assertEqual(
                    self.decide(prompt, arguments).reason,
                    PolicyReason.INVALID_ARGUMENTS,
                )
        arguments = {"id": 3, "when": "8 giờ sáng mai"}
        self.decide(prompt, arguments)
        self.assertEqual(arguments, {"id": 3, "when": "8 giờ sáng mai"})

    def test_request_classification_and_field_extraction(self):
        self.assertEqual(
            classify_request("Đặt hạn việc 3 lúc 8 giờ sáng mai."),
            RequestKind.DEADLINE,
        )
        self.assertEqual(classify_request("Đặt hạn lúc 8 giờ sáng mai."), RequestKind.MISSING_DEADLINE_ID)
        self.assertEqual(classify_request("Đặt hạn việc 3."), RequestKind.MISSING_DEADLINE_TIME)
        self.assertEqual(classify_request("Đặt hạn việc 1 và 2 lúc 8 giờ sáng mai."), RequestKind.MULTIPLE_DEADLINE)
        self.assertEqual(
            deadline_request_fields("Đặt hạn việc 3 lúc 8 giờ sáng mai."),
            (3, "8 giờ sáng mai"),
        )

    def test_deadline_word_inside_create_content_remains_create(self):
        prompt = "Thêm việc: đặt deadline task 3 lúc 8 giờ sáng mai."
        self.assertEqual(classify_request(prompt), RequestKind.CREATE)
        decision = policy_for_tool(
            prompt,
            "create_task",
            {"content": "đặt deadline task 3 lúc 8 giờ sáng mai."},
            reference_time=self.reference,
        )
        self.assertEqual(decision.result, PolicyResult.ALLOW)


if __name__ == "__main__":
    unittest.main()
