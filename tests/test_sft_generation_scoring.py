import unittest

from scripts.score_sft_generation import parse_assistant_text, score_case, summarize


def call_case():
    return {
        "id": "call",
        "behavior": "call",
        "action": "complete",
        "messages": [
            {"role": "user", "content": "Hoàn thành việc 3"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "type": "function",
                    "function": {"name": "complete_task", "arguments": {"id": 3}},
                }],
            },
        ],
    }


def no_call_case():
    return {
        "id": "no-call",
        "behavior": "no_action",
        "action": None,
        "messages": [
            {"role": "user", "content": "Mua sữa."},
            {"role": "assistant", "content": "Bạn có muốn lưu việc đó không?"},
        ],
    }


class SftGenerationScoringTests(unittest.TestCase):
    def test_exact_call_and_end_token_are_accepted(self):
        text = '<tool_call>\n{"name":"complete_task","arguments":{"id":3}}\n</tool_call><|im_end|>\n'
        parsed = parse_assistant_text(text)
        self.assertEqual(parsed["errors"], [])
        result = score_case(call_case(), text, exact_token_sequence=True)
        self.assertTrue(result["proposal_correct"])
        self.assertTrue(result["exact_token_sequence"])

    def test_wrong_arguments_extra_text_and_multiple_calls_are_visible(self):
        wrong = '<tool_call>{"name":"complete_task","arguments":{"id":4}}</tool_call>'
        result = score_case(call_case(), wrong, exact_token_sequence=False)
        self.assertFalse(result["proposal_correct"])
        self.assertIn("arguments_mismatch", result["error_tags"])
        extra = parse_assistant_text("Tôi sẽ làm. " + wrong)
        self.assertIn("extra_text", extra["errors"])
        multiple = parse_assistant_text(wrong + wrong)
        self.assertIn("multiple_calls", multiple["errors"])

    def test_malformed_or_unexpected_call_is_not_silently_dropped(self):
        malformed = parse_assistant_text("<tool_call>{bad}</tool_call>")
        self.assertIn("invalid_call_json", malformed["errors"])
        text = '<tool_call>{"name":"complete_task","arguments":{"id":3}}</tool_call>'
        result = score_case(no_call_case(), text, exact_token_sequence=False)
        self.assertFalse(result["proposal_correct"])
        self.assertIn("unexpected_call", result["error_tags"])

    def test_free_text_only_proves_no_call_for_non_action(self):
        result = score_case(no_call_case(), "Một câu trả lời khác.", exact_token_sequence=False)
        self.assertTrue(result["proposal_correct"])
        self.assertFalse(result["exact_token_sequence"])
        summary = summarize([result])
        self.assertEqual(summary["proposal_correct"], 1)
        self.assertEqual(summary["exact_token_sequences"], 0)


if __name__ == "__main__":
    unittest.main()
