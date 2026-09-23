"""Authorize supported requests and verify verbatim, complete content spans."""

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
    CONTENT_BOUNDARY_MISMATCH = "content_boundary_mismatch"


class RequestKind(str, Enum):
    """Request framing used by the session; not a tool authorization."""

    CREATE = "create"
    MISSING_CREATE = "missing_create"
    LIST = "list"
    UNSUPPORTED = "unsupported"
    NEGATED = "negated"
    OTHER = "other"


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
            (PolicyResult.REJECT, PolicyReason.CONTENT_BOUNDARY_MISMATCH),
        }

        # 3. Reject any combination outside the valid set
        if (self.result, self.reason) not in valid_pairs:
            raise ValueError(
                f"Invalid combination: {self.result.value} + {self.reason.value}"
            )


# These patterns describe request framing only. Task text is never normalized.
_PRONOUN = r"(?:tôi|mình|tui)"
_H = r"[^\S\r\n]+"
_NUMBER_WORD = r"(?:một|hai|ba|bốn|tư|năm|sáu|bảy|tám|chín|mười|mươi|trăm|nghìn|ngàn|lẻ|linh)"
_COUNT = rf"(?:[0-9]+|{_NUMBER_WORD}(?:{_H}{_NUMBER_WORD})*)"
_OBJECT = rf"(?:(?:các|những|{_COUNT}){_H})?(?:việc|task)\b"
_COURTESY = rf"(?:giúp|giùm|hộ)(?:{_H}{_PRONOUN})?\b"
# "sau" is framing only immediately before a delimiter (or a complete line
# instruction). In "Thêm việc sau giờ làm", it remains part of the task.
_LINE_INSTRUCTION = rf"[^\S\r\n]*,{_H}mỗi{_H}dòng{_H}một{_H}việc\b"
_FRAME_END = r"(?=[^\S\r\n]*(?::|\r|\n|$))"
_INTRO = (
    rf"{_OBJECT}(?:(?:{_H}sau\b)?{_LINE_INSTRUCTION}{_FRAME_END}"
    rf"|{_H}sau\b{_FRAME_END})?"
)
_REQUEST_PREFACE = rf"(?:(?:hãy|xin|vui{_H}lòng|làm{_H}ơn|nhờ{_H}bạn|bạn|{_PRONOUN}{_H}muốn|nhớ){_H})?"
_CREATE_HEAD = re.compile(
    rf"^\s*{_REQUEST_PREFACE}(?:"
    rf"(?:thêm|ghi(?:{_H}lại)?|lưu|tạo){_H}(?:{_COURTESY}{_H})?{_INTRO}"
    rf"|note{_H}{_COURTESY}(?:{_H}{_INTRO})?"
    rf"|bỏ{_H}vào{_H}todo\b"
    rf"|(?P<give>cho{_H}{_OBJECT})"
    rf")(?=\s|:|[.!?]?$)",
    re.IGNORECASE,
)
_NEGATED_CREATE = re.compile(
    r"\b(?:đừng|không cần|chưa cần|không muốn|khỏi|không)\s+"
    r"(?:\w+\s+){0,2}(?:thêm|ghi|lưu|note|tạo|bỏ|thực hiện)\b", re.IGNORECASE,
)
_UNSUPPORTED = re.compile(
    r"\b(?:xóa|sửa|hoàn thành|đánh dấu|cập nhật)\s+"
    r"(?:\w+\s+){0,2}(?:task|việc|danh sách|todo)\b", re.IGNORECASE,
)
_META = re.compile(r"\b(?:ví dụ|giải thích|chỉ là câu|có nghĩa là)\b", re.IGNORECASE)
_SUFFIX_WORD = re.compile(
    rf"(?:vào{_H}danh{_H}sách|giúp{_H}{_PRONOUN}|được{_H}không|nhé|với)$",
    re.IGNORECASE,
)
_GIVE_END = re.compile(
    rf"{_H}vào{_H}danh{_H}sách(?:{_H}(?:giúp{_H}{_PRONOUN}|nhé|với))*[.?!]*\s*$",
    re.IGNORECASE,
)
_UNSUPPORTED_HEAD = re.compile(
    rf"^\s*(?:(?:hãy|xin|vui{_H}lòng|làm{_H}ơn|bạn){_H})?"
    rf"(?:(?:xóa|sửa|hoàn{_H}thành|đánh{_H}dấu|cập{_H}nhật|"
    rf"đổi{_H}tên|sắp{_H}xếp){_H}(?:\w+{_H}){{0,2}}"
    rf"(?:task|việc|danh{_H}sách|todo)\b"
    rf"|đặt{_H}nhắc{_H}nhở\b|tắt{_H}thông{_H}báo\b)",
    re.IGNORECASE,
)
_NEGATED_COMMAND_HEAD = re.compile(
    rf"^\s*(?:{_PRONOUN}{_H})?(?:đừng|không{_H}cần|chưa{_H}cần|"
    rf"không{_H}muốn|khỏi|không){_H}(?:"
    rf"(?:thêm|ghi|lưu|note|tạo){_H}(?:{_COURTESY}{_H})?{_OBJECT}"
    rf"|(?:xem|mở|hiển{_H}thị|liệt{_H}kê){_H}(?:danh{_H}sách|todo|task)"
    rf")\b",
    re.IGNORECASE,
)
_LIST_COMMAND_HEAD = re.compile(
    rf"^\s*(?:(?:hãy|xin|vui{_H}lòng|làm{_H}ơn){_H})?"
    rf"(?:(?:xem|mở|hiển{_H}thị|liệt{_H}kê|show|kiểm{_H}tra){_H}"
    rf"|cho{_H}{_PRONOUN}{_H}(?:xem|biết){_H})"
    rf"(?:danh{_H}sách|todo|task|các{_H}việc|những{_H}việc|"
    rf"việc{_H}đã{_H}(?:ghi|lưu|note))\b",
    re.IGNORECASE,
)
_LIST_STATE_QUESTION = re.compile(
    rf"^(?:(?:hiện{_H}tại|hiện{_H}giờ|bây{_H}giờ|lúc{_H}này){_H})?"
    rf"{_PRONOUN}{_H}(?:đang{_H})?có{_H}"
    rf"(?:(?:những|các){_H})?(?:việc|task|todo){_H}(?:nào|gì)"
    rf"(?:{_H}(?:rồi|nhỉ|vậy|thế))?[.?!]*$",
    re.IGNORECASE,
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

    if _META.search(p):
        return ToolDecision(PolicyResult.REJECT, PolicyReason.BARE_STATEMENT)

    # Questions about the caller's current tasks are explicit LIST requests.
    # Keep this anchored so a task-related phrase inside a longer story does
    # not grant read authorization.
    if _LIST_STATE_QUESTION.fullmatch(p):
        return ToolDecision(PolicyResult.ALLOW, PolicyReason.EXPLICIT_LIST)

    # Demand a request/question opening, rather than a verb anywhere in a story.
    opening = re.match(
        rf"^(?:(?:hãy|xin|vui lòng)\s+)?(?:"
        rf"xem|mở|hiển thị|liệt kê|show|kiểm tra|cho\s+{_PRONOUN}\s+(?:xem|biết)"
        rf"|bạn\s+đọc\s+lại|{_PRONOUN}\s+(?:muốn\s+xem\s+lại|còn)"
        rf"|{_PRONOUN}\s+đã\s+(?:ghi|lưu|note)"
        rf"|tính\s+đến\s+giờ\s+{_PRONOUN}\s+đã\s+(?:ghi|lưu|note)"
        rf"|danh sách|todo|task|có những việc (?:nào|gì))\b", p
    )
    if opening is None:
        return ToolDecision(PolicyResult.REJECT, PolicyReason.BARE_STATEMENT)

    intent_words = [
        "hiển thị",
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
        "có gì",
        "việc gì",
        "task nào",
        "đọc lại",
    ]
    object_words = ["danh sách", "todo", "task", "các việc", "đã ghi", "đã lưu", "đã note"]

    has_intent = any(iw in p for iw in intent_words)
    has_object = any(ow in p for ow in object_words)

    if has_intent and has_object:
        return ToolDecision(PolicyResult.ALLOW, PolicyReason.EXPLICIT_LIST)

    return ToolDecision(PolicyResult.REJECT, PolicyReason.BARE_STATEMENT)


@dataclass(frozen=True)
class _CreateRequest:
    """Authorized request framing, with offsets into the untouched prompt."""

    content_start: int
    content_end: int
    literal: bool
    optional_punctuation_end: int


def _authorize_create(prompt: str) -> _CreateRequest | ToolDecision:
    head = _CREATE_HEAD.match(prompt)
    if head is None:
        if _NEGATED_CREATE.search(prompt):
            return ToolDecision(PolicyResult.REJECT, PolicyReason.NEGATED_REQUEST)
        if _UNSUPPORTED.search(prompt):
            return ToolDecision(PolicyResult.REJECT, PolicyReason.UNSUPPORTED_ACTION)
        return ToolDecision(PolicyResult.REJECT, PolicyReason.BARE_STATEMENT)

    # A delimiter counts only immediately after the request header. Colons in
    # task text (C++: vector, 8:00) cannot switch the parsing mode.
    tail = prompt[head.end():]
    if head.group('give') and not _GIVE_END.search(tail):
        return ToolDecision(PolicyResult.REJECT, PolicyReason.BARE_STATEMENT)
    delimiter = re.match(r"[^\S\r\n]*(?::|\r\n|\r|\n)", tail)
    literal = delimiter is not None
    start = head.end() + (delimiter.end() if delimiter else 0)
    while start < len(prompt) and prompt[start].isspace():
        start += 1
    end = len(prompt.rstrip())

    if not literal:
        # Natural-language cancellation/meta clauses are not task payload.
        # Literal content, however, can legitimately contain these same words.
        if _NEGATED_CREATE.search(prompt[start:end]):
            return ToolDecision(PolicyResult.REJECT, PolicyReason.NEGATED_REQUEST)
        if _META.search(prompt[start:end]):
            return ToolDecision(PolicyResult.REJECT, PolicyReason.BARE_STATEMENT)

    optional_punctuation_end = end
    if not literal:
        punctuation = re.search(r"[.?!]+$", prompt[start:end])
        if punctuation:
            end = start + punctuation.start()
        suffix_removed = False
        while suffix := _SUFFIX_WORD.search(prompt, start, end):
            if suffix.start() > start and not prompt[suffix.start() - 1].isspace():
                break
            end = suffix.start()
            while end > start and prompt[end - 1].isspace():
                end -= 1
            suffix_removed = True
        if suffix_removed:
            optional_punctuation_end = end

    if start >= end:
        return ToolDecision(PolicyResult.NEEDS_CLARIFICATION, PolicyReason.MISSING_CONTENT)
    return _CreateRequest(start, end, literal, optional_punctuation_end)


def classify_request(prompt: Any) -> RequestKind:
    """Recognize a new command before consuming a pending CREATE follow-up."""
    if not isinstance(prompt, str) or not prompt.strip():
        return RequestKind.OTHER
    create = _authorize_create(prompt)
    if isinstance(create, _CreateRequest):
        return RequestKind.CREATE
    if create.result == PolicyResult.NEEDS_CLARIFICATION:
        return RequestKind.MISSING_CREATE
    listed = policy_for_list_tasks(prompt, {})
    if listed.result == PolicyResult.ALLOW:
        return RequestKind.LIST
    if _NEGATED_COMMAND_HEAD.match(prompt):
        return RequestKind.NEGATED
    # A clearly framed new command supersedes pending content even if the
    # stricter tool policy will reject it after model proposal.
    if _CREATE_HEAD.match(prompt):
        return RequestKind.CREATE
    if _LIST_COMMAND_HEAD.match(prompt):
        return RequestKind.LIST
    if _UNSUPPORTED_HEAD.match(prompt):
        return RequestKind.UNSUPPORTED
    return RequestKind.OTHER


def _ground_content(prompt: str, content: str, request: _CreateRequest) -> ToolDecision:
    """Require a verbatim span AND complete coverage of the requested payload."""
    start = prompt.find(content)
    if start < 0:
        return ToolDecision(PolicyResult.REJECT, PolicyReason.CONTENT_NOT_GROUNDED)

    while start >= 0:
        end = start + len(content)
        if start == request.content_start and end in (
            request.content_end, request.optional_punctuation_end
        ):
            return ToolDecision(PolicyResult.ALLOW, PolicyReason.EXPLICIT_CREATE)
        start = prompt.find(content, start + 1)
    return ToolDecision(PolicyResult.REJECT, PolicyReason.CONTENT_BOUNDARY_MISMATCH)


def policy_for_create_task(prompt: Any, arguments: Any) -> ToolDecision:
    if not isinstance(prompt, str) or not prompt.strip():
        return ToolDecision(PolicyResult.REJECT, PolicyReason.INVALID_ARGUMENTS)

    request = _authorize_create(prompt)
    if isinstance(request, ToolDecision):
        return request
    if (
        not isinstance(arguments, dict)
        or set(arguments) != {"content"}
        or not isinstance(arguments["content"], str)
        or not arguments["content"].strip()
    ):
        return ToolDecision(PolicyResult.REJECT, PolicyReason.INVALID_ARGUMENTS)
    return _ground_content(prompt, arguments["content"], request)


def policy_for_tool(prompt: Any, tool_name: Any, arguments: Any) -> ToolDecision:
    if not isinstance(tool_name, str) or not tool_name.strip():
        return ToolDecision(PolicyResult.REJECT, PolicyReason.INVALID_ARGUMENTS)

    if tool_name == "create_task":
        return policy_for_create_task(prompt, arguments)
    if tool_name == "list_tasks":
        return policy_for_list_tasks(prompt, arguments)

    return ToolDecision(PolicyResult.REJECT, PolicyReason.UNSUPPORTED_TOOL)
