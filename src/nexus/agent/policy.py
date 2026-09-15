import re
from typing import Any
from enum import Enum
from dataclasses import dataclass


class PolicyResult(str, Enum):
    ALLOW = "allow"
    REJECT = "reject"
    NEEDS_CLARIFICATION = "needs_clarification"


class PolicyReason(str, Enum):
    EXPLICIT_CREATE = "explicit_create"
    EXPLICIT_LIST = "explicit_list"
    MISSING_CONTENT = "missing_content"
    NEGATED_REQUEST = "negated_request"
    UNSUPPORTED_ACTION = "unsupported_action"
    UNSUPPORTED_TOOL = "unsupported_tool"
    BARE_STATEMENT = "bare_statement"
    INVALID_ARGUMENTS = "invalid_arguments"
    CONTENT_NOT_GROUNDED = "content_not_grounded"


@dataclass(frozen=True)
class ToolDecision:
    result: PolicyResult
    reason: PolicyReason

    def __post_init__(self):
        # 1. Strictly check data types to prevent raw strings or None
        if not isinstance(self.result, PolicyResult):
            raise TypeError(
                f"result must be of type PolicyResult, got {type(self.result).__name__}"
            )

        if not isinstance(self.reason, PolicyReason):
            raise TypeError(
                f"reason must be of type PolicyReason, got {type(self.reason).__name__}"
            )

        # 2. Define the exact set of valid pairs
        valid_pairs = {
            (PolicyResult.ALLOW, PolicyReason.EXPLICIT_CREATE),
            (PolicyResult.ALLOW, PolicyReason.EXPLICIT_LIST),
            (PolicyResult.NEEDS_CLARIFICATION, PolicyReason.MISSING_CONTENT),
            (PolicyResult.REJECT, PolicyReason.NEGATED_REQUEST),
            (PolicyResult.REJECT, PolicyReason.UNSUPPORTED_ACTION),
            (PolicyResult.REJECT, PolicyReason.UNSUPPORTED_TOOL),
            (PolicyResult.REJECT, PolicyReason.BARE_STATEMENT),
            (PolicyResult.REJECT, PolicyReason.INVALID_ARGUMENTS),
            (PolicyResult.REJECT, PolicyReason.CONTENT_NOT_GROUNDED),
        }

        # 3. Reject any combination outside the valid set
        if (self.result, self.reason) not in valid_pairs:
            raise ValueError(
                f"Invalid combination: {self.result.value} + {self.reason.value}"
            )


def policy_for_list_tasks(prompt: Any, arguments: Any) -> ToolDecision:
    if not isinstance(prompt, str) or not prompt.strip():
        return ToolDecision(PolicyResult.REJECT, PolicyReason.INVALID_ARGUMENTS)
    if not isinstance(arguments, dict) or len(arguments) > 0:
        return ToolDecision(PolicyResult.REJECT, PolicyReason.INVALID_ARGUMENTS)

    p = re.sub(r"\s+", " ", prompt.lower().strip())

    strong_negation = ["đừng", "không cần", "chưa cần", "không muốn", "không chạy"]
    action_words = ["xem", "mở", "hiển thị", "liệt kê", "show", "kiểm tra"]

    has_strong_negation = any(kw in p for kw in strong_negation)
    has_khong_action = any(
        f"không {act}" in p or f"khỏi {act}" in p for act in action_words
    )

    if has_strong_negation or has_khong_action:
        return ToolDecision(PolicyResult.REJECT, PolicyReason.NEGATED_REQUEST)

    unsupported_keywords = ["xóa", "sửa", "hoàn thành", "note giúp", "thêm", "cập nhật"]
    if any(kw in p for kw in unsupported_keywords):
        return ToolDecision(PolicyResult.REJECT, PolicyReason.UNSUPPORTED_ACTION)

    intent_words = [
        "xem",
        "mở",
        "liệt kê",
        "show",
        "kiểm tra",
        "gồm gì",
        "những gì",
        "việc nào",
        "gì chưa",
        "có gì không",
    ]
    object_words = ["danh sách", "todo", "task", "đã ghi", "đã lưu", "đã note"]

    has_intent = any(iw in p for iw in intent_words)
    has_object = any(ow in p for ow in object_words)

    if has_intent and has_object:
        return ToolDecision(PolicyResult.ALLOW, PolicyReason.EXPLICIT_LIST)

    return ToolDecision(PolicyResult.REJECT, PolicyReason.BARE_STATEMENT)


def policy_for_create_task(prompt: Any, arguments: Any) -> ToolDecision:
    if not isinstance(prompt, str) or not prompt.strip():
        return ToolDecision(PolicyResult.REJECT, PolicyReason.INVALID_ARGUMENTS)

    command_patterns = [
        r"thêm các việc",
        r"thêm việc",
        r"thêm task",
        r"ghi lại việc",
        r"ghi lại task",
        r"ghi việc",
        r"lưu việc",
        r"note giúp tôi",
        r"note giúp",
        r"bỏ vào todo",
    ]

    pattern = r"^\s*(?:" + "|".join(command_patterns) + r")(?:\s*[:.]\s*|\s+|\s*$)"
    match = re.match(pattern, prompt, re.IGNORECASE)

    if match:
        extracted_content = prompt[match.end() :].strip()

        if not extracted_content:
            return ToolDecision(
                PolicyResult.NEEDS_CLARIFICATION, PolicyReason.MISSING_CONTENT
            )

        if (
            not isinstance(arguments, dict)
            or set(arguments.keys()) != {"content"}
            or not isinstance(arguments["content"], str)
            or not arguments["content"].strip()
        ):
            return ToolDecision(PolicyResult.REJECT, PolicyReason.INVALID_ARGUMENTS)

        if arguments["content"] != extracted_content:
            return ToolDecision(PolicyResult.REJECT, PolicyReason.CONTENT_NOT_GROUNDED)

        return ToolDecision(PolicyResult.ALLOW, PolicyReason.EXPLICIT_CREATE)

    p = re.sub(r"\s+", " ", prompt.lower().strip())

    negation_pattern = r"(?:đừng|không cần|chưa cần|không muốn|khỏi|không)\s+(?:(?:\w+)\s+){0,2}(?:thêm|ghi|lưu|note|tạo|bỏ)"
    unsupported_pattern = r"(?:xóa|sửa|hoàn thành|đánh dấu|cập nhật)\s+(?:(?:\w+)\s+){0,2}(?:task|việc|danh sách|todo)"

    if re.search(negation_pattern, p):
        return ToolDecision(PolicyResult.REJECT, PolicyReason.NEGATED_REQUEST)

    if re.search(unsupported_pattern, p):
        return ToolDecision(PolicyResult.REJECT, PolicyReason.UNSUPPORTED_ACTION)

    return ToolDecision(PolicyResult.REJECT, PolicyReason.BARE_STATEMENT)


def policy_for_tool(prompt: Any, tool_name: Any, arguments: Any) -> ToolDecision:
    if not isinstance(tool_name, str) or not tool_name.strip():
        return ToolDecision(PolicyResult.REJECT, PolicyReason.INVALID_ARGUMENTS)

    if tool_name == "create_task":
        return policy_for_create_task(prompt, arguments)
    if tool_name == "list_tasks":
        return policy_for_list_tasks(prompt, arguments)

    return ToolDecision(PolicyResult.REJECT, PolicyReason.UNSUPPORTED_TOOL)
