from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from nexus.agent.prompts import SYSTEM_PROMPTS
from nexus.agent.tools import TOOL_DEFINITIONS
from scripts import audit_sft_data as audit
from scripts.inspect_sft_tokens import build_assistant_only_labels, inspect_record


class FakeTokenizer:
    def apply_chat_template(self, messages, *, tools, add_generation_prompt, tokenize):
        if tools != audit.provider_tools():
            raise AssertionError("Unexpected tools")
        if tokenize is False:
            return "rendered conversation"
        if messages[-1]["role"] == "assistant":
            return [10, 11, 12, 20, 21]
        if not add_generation_prompt:
            raise AssertionError("Prompt must end with an assistant generation prefix")
        return [10, 11, 12]

    def convert_ids_to_tokens(self, token_ids):
        return [f"token-{token_id}" for token_id in token_ids]


class BadPrefixTokenizer(FakeTokenizer):
    def apply_chat_template(self, messages, *, tools, add_generation_prompt, tokenize):
        value = super().apply_chat_template(
            messages, tools=tools, add_generation_prompt=add_generation_prompt, tokenize=tokenize
        )
        if tokenize and messages[-1]["role"] == "assistant":
            return [99, *value[1:]]
        return value


class SftDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.document = json.loads(audit.DEFAULT_DATASET.read_text(encoding="utf-8"))

    def write_dataset(self, document):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        path = Path(temporary.name) / "sft.json"
        path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
        return path

    def test_sanity_dataset_is_balanced_locked_and_has_no_eval_leakage(self):
        summary = audit.audit_dataset()
        manifest = audit.verify_manifest()
        self.assertEqual(summary["cases"], 16)
        self.assertEqual(summary["behavior_balance"], {"call": 8, "clarify": 4, "no_action": 4})
        self.assertEqual(summary["leakage"], {"exact": [], "masked": [], "near": []})
        self.assertEqual(manifest["dataset"]["sha256"], summary["sha256"])

    def test_materialized_record_uses_exact_runtime_prompt_and_all_tools(self):
        record = audit.materialize_case(self.document["cases"][0], prompt_version="v13")
        self.assertEqual(record["messages"][0], {"role": "system", "content": SYSTEM_PROMPTS["v13"]})
        self.assertEqual(record["tools"], [
            {"type": "function", "function": definition} for definition in TOOL_DEFINITIONS
        ])
        self.assertEqual(record["messages"][-1]["role"], "assistant")

    def test_mutated_call_contract_is_rejected(self):
        for mutation in ("multiple", "wrong_tool", "ungrounded_content", "call_on_no_action"):
            with self.subTest(mutation=mutation):
                value = deepcopy(self.document)
                if mutation == "multiple":
                    value["cases"][0]["messages"][1]["tool_calls"] *= 2
                elif mutation == "wrong_tool":
                    value["cases"][0]["messages"][1]["tool_calls"][0]["function"]["name"] = "list_tasks"
                elif mutation == "ungrounded_content":
                    value["cases"][0]["messages"][1]["tool_calls"][0]["function"]["arguments"]["content"] = "viết lại"
                else:
                    value["cases"][-1]["messages"][1]["tool_calls"] = [
                        {"type": "function", "function": {"name": "complete_task", "arguments": {"id": 608}}}
                    ]
                with self.assertRaises(audit.SftDataError):
                    audit.audit_dataset(self.write_dataset(value))

    def test_exact_evaluation_prompt_leakage_is_rejected(self):
        value = deepcopy(self.document)
        development = json.loads(audit.DEVELOPMENT.read_text(encoding="utf-8"))
        value["cases"][0]["messages"][0]["content"] = development["cases"][0]["prompt"]
        value["cases"][0]["messages"][1]["tool_calls"][0]["function"]["arguments"]["content"] = "kiểm tra hóa đơn."
        with self.assertRaisesRegex(audit.SftDataError, "leakage"):
            audit.audit_dataset(self.write_dataset(value))

    def test_assistant_only_mask_covers_only_target_suffix(self):
        record = audit.materialize_case(self.document["cases"][0], prompt_version="v13")
        encoded = build_assistant_only_labels(record, FakeTokenizer())
        self.assertEqual(encoded["labels"], [-100, -100, -100, 20, 21])
        self.assertEqual(encoded["prompt_tokens"], 3)
        self.assertEqual(encoded["supervised_tokens"], 2)
        result = inspect_record(record, FakeTokenizer())
        self.assertEqual([row["loss"] for row in result["tokens"]], [False, False, False, True, True])
        self.assertEqual(result["rendered"], "rendered conversation")

    def test_tokenizer_prefix_mismatch_is_rejected(self):
        record = audit.materialize_case(self.document["cases"][0], prompt_version="v13")
        with self.assertRaisesRegex(ValueError, "not a suffix"):
            build_assistant_only_labels(record, BadPrefixTokenizer())


if __name__ == "__main__":
    unittest.main()
