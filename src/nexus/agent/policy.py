"""Authorize supported requests and verify verbatim, complete content spans."""

import re
from datetime import datetime
from typing import Any
from enum import Enum
from dataclasses import dataclass

from nexus.core.deadlines import DeadlineParseError, parse_deadline


class PolicyResult(str, Enum):
    ALLOW = "allow"
    REJECT = "reject"
    NEEDS_CLARIFICATION = "needs_clarification"
    REQUIRES_CONFIRMATION = "requires_confirmation"


class PolicyReason(str, Enum):
    EXPLICIT_CREATE = "explicit_create"
    EXPLICIT_LIST = "explicit_list"
    EXPLICIT_COMPLETE = "explicit_complete"
    EXPLICIT_UPDATE = "explicit_update"
    EXPLICIT_DELETE = "explicit_delete"
    EXPLICIT_DEADLINE = "explicit_deadline"
    EXPLICIT_DEADLINE_QUERY = "explicit_deadline_query"
    MISSING_CONTENT = "missing_content"
    MISSING_TASK_ID = "missing_task_id"
    MISSING_UPDATE_ID = "missing_update_id"
    MISSING_UPDATE_CONTENT = "missing_update_content"
    MISSING_DELETE_ID = "missing_delete_id"
    MISSING_DEADLINE_ID = "missing_deadline_id"
    MISSING_DEADLINE_TIME = "missing_deadline_time"
    INVALID_DEADLINE_TIME = "invalid_deadline_time"
    MISSING_DEADLINE_SCOPE = "missing_deadline_scope"
    DEADLINE_SCOPE_MISMATCH = "deadline_scope_mismatch"
    MULTIPLE_TASK_IDS = "multiple_task_ids"
    TASK_ID_MISMATCH = "task_id_mismatch"
    NEGATED_REQUEST = "negated_request"
    UNSUPPORTED_ACTION = "unsupported_action"
    UNSUPPORTED_TOOL = "unsupported_tool"
    TOOL_NOT_AVAILABLE = "tool_not_available"
    BARE_STATEMENT = "bare_statement"
    INVALID_ARGUMENTS = "invalid_arguments"
    CONTENT_NOT_GROUNDED = "content_not_grounded"
    CONTENT_BOUNDARY_MISMATCH = "content_boundary_mismatch"


class RequestKind(str, Enum):
    """Request framing used by the session; not a tool authorization."""

    CREATE = "create"
    MISSING_CREATE = "missing_create"
    LIST = "list"
    COMPLETE = "complete"
    MISSING_COMPLETE = "missing_complete"
    MULTIPLE_COMPLETE = "multiple_complete"
    EDIT = "edit"
    MISSING_EDIT_ID = "missing_edit_id"
    MISSING_EDIT_CONTENT = "missing_edit_content"
    MULTIPLE_EDIT = "multiple_edit"
    DELETE = "delete"
    MISSING_DELETE_ID = "missing_delete_id"
    MULTIPLE_DELETE = "multiple_delete"
    DEADLINE = "deadline"
    MISSING_DEADLINE_ID = "missing_deadline_id"
    MISSING_DEADLINE_TIME = "missing_deadline_time"
    MULTIPLE_DEADLINE = "multiple_deadline"
    DEADLINE_QUERY = "deadline_query"
    MISSING_DEADLINE_SCOPE = "missing_deadline_scope"
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
            (PolicyResult.ALLOW, PolicyReason.EXPLICIT_COMPLETE),
            (PolicyResult.ALLOW, PolicyReason.EXPLICIT_UPDATE),
            (PolicyResult.ALLOW, PolicyReason.EXPLICIT_DEADLINE),
            (PolicyResult.ALLOW, PolicyReason.EXPLICIT_DEADLINE_QUERY),
            (PolicyResult.REQUIRES_CONFIRMATION, PolicyReason.EXPLICIT_DELETE),
            (PolicyResult.NEEDS_CLARIFICATION, PolicyReason.MISSING_CONTENT),
            (PolicyResult.NEEDS_CLARIFICATION, PolicyReason.MISSING_TASK_ID),
            (PolicyResult.NEEDS_CLARIFICATION, PolicyReason.MULTIPLE_TASK_IDS),
            (PolicyResult.NEEDS_CLARIFICATION, PolicyReason.MISSING_UPDATE_ID),
            (PolicyResult.NEEDS_CLARIFICATION, PolicyReason.MISSING_UPDATE_CONTENT),
            (PolicyResult.NEEDS_CLARIFICATION, PolicyReason.MISSING_DELETE_ID),
            (PolicyResult.NEEDS_CLARIFICATION, PolicyReason.MISSING_DEADLINE_ID),
            (PolicyResult.NEEDS_CLARIFICATION, PolicyReason.MISSING_DEADLINE_TIME),
            (PolicyResult.NEEDS_CLARIFICATION, PolicyReason.INVALID_DEADLINE_TIME),
            (PolicyResult.NEEDS_CLARIFICATION, PolicyReason.MISSING_DEADLINE_SCOPE),
            (PolicyResult.REJECT, PolicyReason.NEGATED_REQUEST),
            (PolicyResult.REJECT, PolicyReason.UNSUPPORTED_ACTION),
            (PolicyResult.REJECT, PolicyReason.UNSUPPORTED_TOOL),
            (PolicyResult.REJECT, PolicyReason.TOOL_NOT_AVAILABLE),
            (PolicyResult.REJECT, PolicyReason.BARE_STATEMENT),
            (PolicyResult.REJECT, PolicyReason.INVALID_ARGUMENTS),
            (PolicyResult.REJECT, PolicyReason.CONTENT_NOT_GROUNDED),
            (PolicyResult.REJECT, PolicyReason.CONTENT_BOUNDARY_MISMATCH),
            (PolicyResult.REJECT, PolicyReason.TASK_ID_MISMATCH),
            (PolicyResult.REJECT, PolicyReason.DEADLINE_SCOPE_MISMATCH),
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
    rf"lưu{_H}(?:một|1){_H}task{_H}mới{_H}có{_H}nội{_H}dung"
    rf"|(?:thêm|ghi(?:{_H}lại)?|lưu|tạo){_H}(?:{_COURTESY}{_H})?{_INTRO}"
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
_COMPLETE_COMMAND_HEAD = re.compile(
    rf"^\s*(?:(?:hãy|xin|vui{_H}lòng|làm{_H}ơn){_H})?"
    rf"(?:hoàn{_H}thành|đánh{_H}dấu){_H}(?:việc|task)\b",
    re.IGNORECASE,
)
_NEGATED_COMPLETE_HEAD = re.compile(
    rf"^\s*(?:{_PRONOUN}{_H})?(?:đừng|không{_H}cần|chưa{_H}cần|"
    rf"không{_H}muốn|khỏi|không){_H}(?:hoàn{_H}thành|đánh{_H}dấu)\b",
    re.IGNORECASE,
)
_COMPLETE_EXACT = re.compile(
    rf"^\s*(?:(?:hãy|xin|vui{_H}lòng|làm{_H}ơn){_H})?(?:"
    rf"hoàn{_H}thành{_H}(?:việc|task){_H}\#?(?P<finish_id>[0-9]+)"
    rf"|đánh{_H}dấu{_H}(?:việc|task){_H}\#?(?P<mark_id>[0-9]+)"
    rf"{_H}(?:là{_H})?(?:đã{_H}xong|đã{_H}hoàn{_H}thành|hoàn{_H}thành)"
    rf")(?:{_H}(?:giúp{_H}{_PRONOUN}|nhé|với))*[.?!]*\s*$",
    re.IGNORECASE,
)
_COMPLETE_REPORTED_DONE = re.compile(
    rf"^\s*{_PRONOUN}{_H}làm{_H}xong{_H}(?:việc|task){_H}#?[0-9]+"
    rf"{_H}rồi\s*;\s*cập{_H}nhật{_H}trạng{_H}thái[.?!]*\s*$",
    re.IGNORECASE,
)
_ID_TOKEN = re.compile(r"(?<!\w)#?([0-9]+)(?!\w)")
_EDIT_COMMAND_HEAD = re.compile(
    rf"^\s*(?:(?:hãy|xin|vui{_H}lòng|làm{_H}ơn){_H})?(?:"
    rf"(?:sửa|cập{_H}nhật){_H}(?:nội{_H}dung{_H})?(?:việc|task)"
    rf"|đổi{_H}nội{_H}dung{_H}(?:việc|task))\b",
    re.IGNORECASE,
)
_NEGATED_EDIT_HEAD = re.compile(
    rf"^\s*(?:{_PRONOUN}{_H})?(?:đừng|không{_H}cần|chưa{_H}cần|"
    rf"không{_H}muốn|khỏi|không){_H}(?:sửa|cập{_H}nhật|đổi{_H}nội{_H}dung)\b",
    re.IGNORECASE,
)
_EDIT_SEPARATOR = re.compile(rf"{_H}thành(?:{_H}|(?=:)|\s*$)", re.IGNORECASE)
_DELETE_COMMAND_HEAD = re.compile(
    rf"^\s*(?:(?:hãy|xin|vui{_H}lòng|làm{_H}ơn){_H})?"
    rf"(?:xóa|xoá|bỏ(?:{_H}hẳn)?){_H}(?:việc|task)\b",
    re.IGNORECASE,
)
_NEGATED_DELETE_HEAD = re.compile(
    rf"^\s*(?:{_PRONOUN}{_H})?(?:đừng|không{_H}cần|chưa{_H}cần|"
    rf"không{_H}muốn|khỏi|không){_H}(?:xóa|xoá|bỏ)\b",
    re.IGNORECASE,
)
_DELETE_EXACT = re.compile(
    rf"^\s*(?:(?:hãy|xin|vui{_H}lòng|làm{_H}ơn){_H})?(?:"
    rf"(?:xóa|xoá){_H}(?:việc|task){_H}\#?(?P<delete_id>[0-9]+)"
    rf"(?:{_H}khỏi{_H}danh{_H}sách)?"
    rf"|bỏ{_H}(?:việc|task){_H}\#?(?P<drop_id>[0-9]+)"
    rf"(?:{_H}khỏi{_H}danh{_H}sách)?"
    rf")(?:{_H}(?:giúp{_H}{_PRONOUN}|nhé|với))*[.?!]*\s*$",
    re.IGNORECASE,
)
_DELETE_WITH_CONFIRMATION_REQUEST = re.compile(
    rf"^\s*(?:bỏ{_H}hẳn){_H}(?:việc|task){_H}#?[0-9]+"
    rf"\s*;\s*ứng{_H}dụng{_H}hãy{_H}hỏi{_H}xác{_H}nhận[.?!]*\s*$",
    re.IGNORECASE,
)


_DEADLINE_COMMAND_HEAD = re.compile(
    rf"^\s*(?:(?:hãy|xin|vui{_H}lòng|làm{_H}ơn){_H})?"
    rf"đặt{_H}(?:hạn|deadline|thời{_H}hạn)\b",
    re.IGNORECASE,
)
_NEGATED_DEADLINE_HEAD = re.compile(
    rf"^\s*(?:{_PRONOUN}{_H})?(?:đừng|không{_H}cần|chưa{_H}cần|"
    rf"không{_H}muốn|khỏi|không){_H}đặt{_H}(?:hạn|deadline|thời{_H}hạn)\b",
    re.IGNORECASE,
)
_DEADLINE_SEPARATOR = re.compile(rf"{_H}(?:lúc|là|vào)(?:{_H}|\s*$)", re.IGNORECASE)
_DEADLINE_BEFORE = re.compile(
    rf"^(?:cho{_H})?(?:(?:việc|task)(?:{_H}#?[0-9]+)?)?$",
    re.IGNORECASE,
)

# Additional *complete* request frames. Their named ID is taken only from the
# task reference, so numbers in the time phrase cannot become extra task IDs.
# Each frame still passes the captured when through the deterministic parser.
_DEADLINE_ADDITIONAL_FRAMES = tuple(re.compile(pattern, re.IGNORECASE) for pattern in (
    rf"^\s*(?:(?:cho{_H}{_PRONOUN}|xin|hãy){_H})?"
    rf"(?:đặt|chốt|set|cập{_H}nhật){_H}(?:hạn|deadline|thời{_H}hạn)"
    rf"(?:{_H}(?:của|cho))?{_H}(?:việc|task){_H}#?(?P<id>[0-9]+)"
    rf"(?:\s*:\s*|{_H}(?:lúc|vào|ở|là|thành){_H})"
    rf"(?P<when>.+?)(?:,\s*đúng{_H}ID{_H}đó)?[.?!]*\s*$",
    rf"^\s*(?:việc|task){_H}#?(?P<id>[0-9]+)\s*:\s*"
    rf"đặt{_H}hạn{_H}(?P<when>.+?)[.?!]*\s*$",
    rf"^\s*hạn{_H}của{_H}(?:việc|task){_H}#?(?P<id>[0-9]+)\s*,\s*"
    rf"hãy{_H}chuyển{_H}sang{_H}(?P<when>.+?)[.?!]*\s*$",
    rf"^\s*với{_H}(?:việc|task){_H}ID{_H}#?(?P<id>[0-9]+)\s*,\s*"
    rf"hạn{_H}mới{_H}là{_H}(?P<when>.+?)(?:;\s*ghi{_H}lại)?[.?!]*\s*$",
))


_DUE_QUERY_EXACT = re.compile(
    rf"^\s*(?:(?:hãy|xin|vui{_H}lòng|làm{_H}ơn){_H})?(?:"
    rf"(?:(?:xem|hiển{_H}thị|liệt{_H}kê){_H}|"
    rf"cho{_H}{_PRONOUN}{_H}xem{_H})"
    rf"(?:(?:các|những){_H})?(?:việc|task){_H}(?:"
    rf"(?:đến|tới){_H}hạn{_H}(?P<dated>hôm{_H}nay|ngày{_H}mai|mai)"
    rf"|(?P<overdue>(?:đã{_H})?(?:quá|trễ){_H}hạn))"
    rf"|(?P<prefix>hôm{_H}nay|ngày{_H}mai|mai){_H}có{_H}"
    rf"(?:(?:những|các){_H})?(?:việc|task){_H}nào{_H}"
    rf"(?:đến|tới){_H}hạn"
    rf")[.?!]*\s*$",
    re.IGNORECASE,
)
_DUE_QUERY_MISSING = re.compile(
    rf"^\s*(?:(?:hãy|xin|vui{_H}lòng|làm{_H}ơn){_H})?(?:"
    rf"(?:(?:xem|hiển{_H}thị|liệt{_H}kê){_H}|"
    rf"cho{_H}{_PRONOUN}{_H}xem{_H})"
    rf"(?:(?:các|những){_H})?(?:việc|task){_H}(?:theo{_H}hạn|(?:đến|tới){_H}hạn)"
    rf")[.?!]*\s*$",
    re.IGNORECASE,
)
_DUE_QUERY_OVERDUE_QUESTION = re.compile(
    rf"^\s*cho{_H}{_PRONOUN}{_H}biết{_H}(?:việc|task){_H}nào"
    rf"{_H}đã{_H}(?:quá|trễ){_H}hạn[.?!]*\s*$",
    re.IGNORECASE,
)
_NEGATED_DUE_QUERY = re.compile(
    rf"^\s*(?:{_PRONOUN}{_H})?(?:đừng|không{_H}cần|chưa{_H}cần|"
    rf"không{_H}muốn|khỏi|không){_H}(?:xem|hiển{_H}thị|liệt{_H}kê)\b",
    re.IGNORECASE,
)
_DUE_SCOPE_REPLY = re.compile(
    rf"^\s*(?P<scope>hôm{_H}nay|ngày{_H}mai|mai|quá{_H}hạn|trễ{_H}hạn)[.?!]?\s*$",
    re.IGNORECASE,
)


def _scope_from_text(value: str) -> str:
    normalized = re.sub(r"\s+", " ", value.casefold().strip())
    if normalized == "hôm nay":
        return "today"
    if normalized in ("mai", "ngày mai"):
        return "tomorrow"
    return "overdue"


def deadline_scope_reply(prompt: str) -> str | None:
    """Return a canonical scope for one narrow session continuation."""
    if not isinstance(prompt, str):
        return None
    match = _DUE_SCOPE_REPLY.fullmatch(prompt)
    return None if match is None else _scope_from_text(match.group("scope"))


def _authorize_deadline_query(prompt: str) -> str | ToolDecision:
    if _NEGATED_DUE_QUERY.match(prompt):
        return ToolDecision(PolicyResult.REJECT, PolicyReason.NEGATED_REQUEST)
    match = _DUE_QUERY_EXACT.fullmatch(prompt)
    if match is not None:
        value = match.group("overdue") or match.group("dated") or match.group("prefix")
        return _scope_from_text(value)
    if _DUE_QUERY_OVERDUE_QUESTION.fullmatch(prompt):
        return "overdue"
    if _DUE_QUERY_MISSING.fullmatch(prompt):
        return ToolDecision(
            PolicyResult.NEEDS_CLARIFICATION,
            PolicyReason.MISSING_DEADLINE_SCOPE,
        )
    return ToolDecision(PolicyResult.REJECT, PolicyReason.BARE_STATEMENT)


def policy_for_list_tasks_by_deadline(prompt: Any, arguments: Any) -> ToolDecision:
    if not isinstance(prompt, str) or not prompt.strip():
        return ToolDecision(PolicyResult.REJECT, PolicyReason.INVALID_ARGUMENTS)
    request = _authorize_deadline_query(prompt)
    if isinstance(request, ToolDecision):
        return request
    if (
        not isinstance(arguments, dict)
        or set(arguments) != {"scope"}
        or not isinstance(arguments["scope"], str)
        or arguments["scope"] not in ("today", "tomorrow", "overdue")
    ):
        return ToolDecision(PolicyResult.REJECT, PolicyReason.INVALID_ARGUMENTS)
    if arguments["scope"] != request:
        return ToolDecision(
            PolicyResult.REJECT, PolicyReason.DEADLINE_SCOPE_MISMATCH
        )
    return ToolDecision(
        PolicyResult.ALLOW, PolicyReason.EXPLICIT_DEADLINE_QUERY
    )


@dataclass(frozen=True)
class _CompleteRequest:
    task_id: int


@dataclass(frozen=True)
class _DeleteRequest:
    task_id: int


@dataclass(frozen=True)
class _DeadlineRequest:
    task_id: int
    when_start: int
    when_end: int
    punctuation_end: int


@dataclass(frozen=True)
class _DeadlineParts:
    ids: tuple[int, ...]
    before: str
    when: str | None
    when_start: int | None
    when_end: int | None
    punctuation_end: int | None


@dataclass(frozen=True)
class _EditRequest:
    task_id: int
    content_start: int
    content_end: int
    optional_punctuation_end: int


@dataclass(frozen=True)
class _EditParts:
    ids: tuple[int, ...]
    before: str
    content: str | None
    content_start: int | None
    content_end: int | None
    optional_punctuation_end: int | None
    literal: bool


def _parse_edit_parts(prompt: str) -> _EditParts | None:
    head = _EDIT_COMMAND_HEAD.match(prompt)
    if head is None:
        return None
    tail = prompt[head.end():]
    separator = _EDIT_SEPARATOR.search(tail)
    if separator is None:
        before = tail.strip().rstrip(".?!").rstrip()
        content = None
        content_start = None
        content_end = None
        optional_punctuation_end = None
        literal = False
    else:
        before = tail[:separator.start()].strip()
        content_start = head.end() + separator.end()
        while content_start < len(prompt) and prompt[content_start].isspace():
            content_start += 1
        literal = content_start < len(prompt) and prompt[content_start] == ":"
        if literal:
            content_start += 1
            while content_start < len(prompt) and prompt[content_start].isspace():
                content_start += 1
        content_end = len(prompt.rstrip())
        optional_punctuation_end = content_end
        if not literal:
            while content_end > content_start and prompt[content_end - 1] in ".?!":
                content_end -= 1
            while content_end > content_start and prompt[content_end - 1].isspace():
                content_end -= 1
        content = (
            prompt[content_start:content_end]
            if content_start < content_end
            else None
        )
    ids = tuple(int(value) for value in _ID_TOKEN.findall(before))
    return _EditParts(ids, before, content, content_start, content_end, optional_punctuation_end, literal)


def edit_request_fields(prompt: str) -> tuple[int | None, str | None]:
    """Return reusable fields from a recognized EDIT framing."""
    parts = _parse_edit_parts(prompt)
    if parts is None:
        return None, None
    task_id = parts.ids[0] if len(parts.ids) == 1 else None
    return task_id, parts.content


def _authorize_edit(prompt: str) -> _EditRequest | ToolDecision:
    if _NEGATED_EDIT_HEAD.match(prompt):
        return ToolDecision(PolicyResult.REJECT, PolicyReason.NEGATED_REQUEST)
    parts = _parse_edit_parts(prompt)
    if parts is None:
        return ToolDecision(PolicyResult.REJECT, PolicyReason.BARE_STATEMENT)
    if len(parts.ids) > 1:
        return ToolDecision(
            PolicyResult.NEEDS_CLARIFICATION, PolicyReason.MULTIPLE_TASK_IDS
        )
    if not parts.ids:
        if parts.before:
            return ToolDecision(PolicyResult.REJECT, PolicyReason.BARE_STATEMENT)
        return ToolDecision(
            PolicyResult.NEEDS_CLARIFICATION, PolicyReason.MISSING_UPDATE_ID
        )
    task_id = parts.ids[0]
    if task_id <= 0 or re.fullmatch(r"#?[0-9]+", parts.before) is None:
        return ToolDecision(PolicyResult.REJECT, PolicyReason.BARE_STATEMENT)
    if parts.content is None:
        return ToolDecision(
            PolicyResult.NEEDS_CLARIFICATION, PolicyReason.MISSING_UPDATE_CONTENT
        )
    if not parts.literal and re.search(
        r"(?:[,;]\s*|\b(?:và|rồi|sau đó|đồng thời)\s+)"
        r"(?:xóa|xoá|thêm|hoàn thành|đánh dấu|đặt hạn)\b",
        parts.content,
        re.IGNORECASE,
    ):
        return ToolDecision(PolicyResult.REJECT, PolicyReason.UNSUPPORTED_ACTION)
    return _EditRequest(task_id, parts.content_start, parts.content_end, parts.optional_punctuation_end)


def policy_for_update_task(prompt: Any, arguments: Any) -> ToolDecision:
    if not isinstance(prompt, str) or not prompt.strip():
        return ToolDecision(PolicyResult.REJECT, PolicyReason.INVALID_ARGUMENTS)
    request = _authorize_edit(prompt)
    if isinstance(request, ToolDecision):
        return request
    if (
        not isinstance(arguments, dict)
        or set(arguments) != {"id", "content"}
        or isinstance(arguments["id"], bool)
        or not isinstance(arguments["id"], int)
        or arguments["id"] <= 0
        or not isinstance(arguments["content"], str)
        or not arguments["content"].strip()
        or "\n" in arguments["content"]
        or "\r" in arguments["content"]
    ):
        return ToolDecision(PolicyResult.REJECT, PolicyReason.INVALID_ARGUMENTS)
    if arguments["id"] != request.task_id:
        return ToolDecision(PolicyResult.REJECT, PolicyReason.TASK_ID_MISMATCH)
    content = arguments["content"]
    expected = prompt[request.content_start:request.content_end]
    with_punctuation = prompt[request.content_start:request.optional_punctuation_end]
    if content in (expected, with_punctuation):
        return ToolDecision(PolicyResult.ALLOW, PolicyReason.EXPLICIT_UPDATE)
    if content not in prompt:
        return ToolDecision(PolicyResult.REJECT, PolicyReason.CONTENT_NOT_GROUNDED)
    return ToolDecision(PolicyResult.REJECT, PolicyReason.CONTENT_BOUNDARY_MISMATCH)


def _parse_deadline_parts(prompt: str) -> _DeadlineParts | None:
    head = _DEADLINE_COMMAND_HEAD.match(prompt)
    legacy = None
    if head is not None:
        tail = prompt[head.end():]
        separator = _DEADLINE_SEPARATOR.search(tail)
        if separator is None:
            before = tail.strip().rstrip(".?!").rstrip()
            when = None
            when_start = None
            when_end = None
            punctuation_end = None
        else:
            before = tail[:separator.start()].strip()
            when_start = head.end() + separator.end()
            while when_start < len(prompt) and prompt[when_start].isspace():
                when_start += 1
            punctuation_end = len(prompt.rstrip())
            when_end = punctuation_end
            while when_end > when_start and prompt[when_end - 1] in ".?!":
                when_end -= 1
            while when_end > when_start and prompt[when_end - 1].isspace():
                when_end -= 1
            when = prompt[when_start:when_end] if when_start < when_end else None
        ids = tuple(int(value) for value in _ID_TOKEN.findall(before))
        legacy = _DeadlineParts(
            ids, before, when, when_start, when_end, punctuation_end
        )
        if _DEADLINE_BEFORE.fullmatch(before):
            return legacy

    for frame in _DEADLINE_ADDITIONAL_FRAMES:
        match = frame.fullmatch(prompt)
        if match is None:
            continue
        task_id = int(match.group("id"))
        when_start, when_end = match.span("when")
        suffix = prompt[when_end:].strip()
        punctuation_end = len(prompt.rstrip()) if re.fullmatch(r"[.?!]+", suffix) else when_end
        return _DeadlineParts(
            (task_id,), f"việc {task_id}", match.group("when"),
            when_start, when_end, punctuation_end,
        )
    return legacy


def deadline_request_fields(prompt: str) -> tuple[int | None, str | None]:
    """Return the reusable ID and raw time phrase from a deadline request."""
    parts = _parse_deadline_parts(prompt)
    if parts is None:
        return None, None
    task_id = parts.ids[0] if len(parts.ids) == 1 else None
    return task_id, parts.when


def _authorize_deadline(
    prompt: str, *, reference_time: datetime | None = None
) -> _DeadlineRequest | ToolDecision:
    if _NEGATED_DEADLINE_HEAD.match(prompt):
        return ToolDecision(PolicyResult.REJECT, PolicyReason.NEGATED_REQUEST)
    parts = _parse_deadline_parts(prompt)
    if parts is None:
        return ToolDecision(PolicyResult.REJECT, PolicyReason.BARE_STATEMENT)
    if len(parts.ids) > 1:
        return ToolDecision(
            PolicyResult.NEEDS_CLARIFICATION, PolicyReason.MULTIPLE_TASK_IDS
        )
    if not _DEADLINE_BEFORE.fullmatch(parts.before):
        return ToolDecision(PolicyResult.REJECT, PolicyReason.BARE_STATEMENT)
    if not parts.ids:
        return ToolDecision(
            PolicyResult.NEEDS_CLARIFICATION, PolicyReason.MISSING_DEADLINE_ID
        )
    task_id = parts.ids[0]
    if task_id <= 0:
        return ToolDecision(PolicyResult.REJECT, PolicyReason.BARE_STATEMENT)
    if parts.when is None:
        return ToolDecision(
            PolicyResult.NEEDS_CLARIFICATION, PolicyReason.MISSING_DEADLINE_TIME
        )
    try:
        parse_deadline(parts.when, reference=reference_time)
    except DeadlineParseError:
        return ToolDecision(
            PolicyResult.NEEDS_CLARIFICATION, PolicyReason.INVALID_DEADLINE_TIME
        )
    return _DeadlineRequest(
        task_id,
        parts.when_start,
        parts.when_end,
        parts.punctuation_end,
    )


def policy_for_set_task_deadline(
    prompt: Any,
    arguments: Any,
    *,
    reference_time: datetime | None = None,
) -> ToolDecision:
    if not isinstance(prompt, str) or not prompt.strip():
        return ToolDecision(PolicyResult.REJECT, PolicyReason.INVALID_ARGUMENTS)
    request = _authorize_deadline(prompt, reference_time=reference_time)
    if isinstance(request, ToolDecision):
        return request
    if (
        not isinstance(arguments, dict)
        or set(arguments) != {"id", "when"}
        or isinstance(arguments["id"], bool)
        or not isinstance(arguments["id"], int)
        or arguments["id"] <= 0
        or not isinstance(arguments["when"], str)
        or not arguments["when"].strip()
        or "\n" in arguments["when"]
        or "\r" in arguments["when"]
    ):
        return ToolDecision(PolicyResult.REJECT, PolicyReason.INVALID_ARGUMENTS)
    if arguments["id"] != request.task_id:
        return ToolDecision(PolicyResult.REJECT, PolicyReason.TASK_ID_MISMATCH)
    when = arguments["when"]
    expected = prompt[request.when_start:request.when_end]
    with_punctuation = prompt[request.when_start:request.punctuation_end]
    if when in (expected, with_punctuation):
        try:
            parse_deadline(when, reference=reference_time)
        except DeadlineParseError:
            return ToolDecision(
                PolicyResult.NEEDS_CLARIFICATION,
                PolicyReason.INVALID_DEADLINE_TIME,
            )
        return ToolDecision(PolicyResult.ALLOW, PolicyReason.EXPLICIT_DEADLINE)
    if when not in prompt:
        return ToolDecision(PolicyResult.REJECT, PolicyReason.CONTENT_NOT_GROUNDED)
    return ToolDecision(
        PolicyResult.REJECT, PolicyReason.CONTENT_BOUNDARY_MISMATCH
    )


def _authorize_delete(prompt: str) -> _DeleteRequest | ToolDecision:
    if _NEGATED_DELETE_HEAD.match(prompt):
        return ToolDecision(PolicyResult.REJECT, PolicyReason.NEGATED_REQUEST)
    head = _DELETE_COMMAND_HEAD.match(prompt)
    if head is None:
        return ToolDecision(PolicyResult.REJECT, PolicyReason.BARE_STATEMENT)

    ids = [int(value) for value in _ID_TOKEN.findall(prompt)]
    unique_ids = list(dict.fromkeys(ids))
    if not unique_ids:
        remainder = prompt[head.end():].strip().rstrip(".?!").strip()
        if remainder:
            return ToolDecision(PolicyResult.REJECT, PolicyReason.BARE_STATEMENT)
        return ToolDecision(
            PolicyResult.NEEDS_CLARIFICATION, PolicyReason.MISSING_DELETE_ID
        )
    if len(unique_ids) != 1 or len(ids) != 1:
        return ToolDecision(
            PolicyResult.NEEDS_CLARIFICATION, PolicyReason.MULTIPLE_TASK_IDS
        )
    exact = _DELETE_EXACT.fullmatch(prompt) or _DELETE_WITH_CONFIRMATION_REQUEST.fullmatch(prompt)
    if exact is None or unique_ids[0] <= 0:
        return ToolDecision(PolicyResult.REJECT, PolicyReason.BARE_STATEMENT)
    return _DeleteRequest(unique_ids[0])


def policy_for_delete_task(prompt: Any, arguments: Any) -> ToolDecision:
    if not isinstance(prompt, str) or not prompt.strip():
        return ToolDecision(PolicyResult.REJECT, PolicyReason.INVALID_ARGUMENTS)
    request = _authorize_delete(prompt)
    if isinstance(request, ToolDecision):
        return request
    if (
        not isinstance(arguments, dict)
        or set(arguments) != {"id"}
        or isinstance(arguments["id"], bool)
        or not isinstance(arguments["id"], int)
        or arguments["id"] <= 0
    ):
        return ToolDecision(PolicyResult.REJECT, PolicyReason.INVALID_ARGUMENTS)
    if arguments["id"] != request.task_id:
        return ToolDecision(PolicyResult.REJECT, PolicyReason.TASK_ID_MISMATCH)
    return ToolDecision(
        PolicyResult.REQUIRES_CONFIRMATION, PolicyReason.EXPLICIT_DELETE
    )


def _authorize_complete(prompt: str) -> _CompleteRequest | ToolDecision:
    if _NEGATED_COMPLETE_HEAD.match(prompt):
        return ToolDecision(PolicyResult.REJECT, PolicyReason.NEGATED_REQUEST)
    if not (_COMPLETE_COMMAND_HEAD.match(prompt) or _COMPLETE_REPORTED_DONE.fullmatch(prompt)):
        return ToolDecision(PolicyResult.REJECT, PolicyReason.BARE_STATEMENT)

    ids = [int(value) for value in _ID_TOKEN.findall(prompt)]
    unique_ids = list(dict.fromkeys(ids))
    if not unique_ids:
        return ToolDecision(
            PolicyResult.NEEDS_CLARIFICATION, PolicyReason.MISSING_TASK_ID
        )
    if len(unique_ids) != 1 or len(ids) != 1:
        return ToolDecision(
            PolicyResult.NEEDS_CLARIFICATION, PolicyReason.MULTIPLE_TASK_IDS
        )
    exact = _COMPLETE_EXACT.fullmatch(prompt) or _COMPLETE_REPORTED_DONE.fullmatch(prompt)
    if exact is None or unique_ids[0] <= 0:
        return ToolDecision(PolicyResult.REJECT, PolicyReason.BARE_STATEMENT)
    return _CompleteRequest(unique_ids[0])


def policy_for_complete_task(prompt: Any, arguments: Any) -> ToolDecision:
    if not isinstance(prompt, str) or not prompt.strip():
        return ToolDecision(PolicyResult.REJECT, PolicyReason.INVALID_ARGUMENTS)
    request = _authorize_complete(prompt)
    if isinstance(request, ToolDecision):
        return request
    if (
        not isinstance(arguments, dict)
        or set(arguments) != {"id"}
        or isinstance(arguments["id"], bool)
        or not isinstance(arguments["id"], int)
        or arguments["id"] <= 0
    ):
        return ToolDecision(PolicyResult.REJECT, PolicyReason.INVALID_ARGUMENTS)
    if arguments["id"] != request.task_id:
        return ToolDecision(PolicyResult.REJECT, PolicyReason.TASK_ID_MISMATCH)
    return ToolDecision(PolicyResult.ALLOW, PolicyReason.EXPLICIT_COMPLETE)


def policy_for_list_tasks(prompt: Any, arguments: Any) -> ToolDecision:
    if not isinstance(prompt, str) or not prompt.strip():
        return ToolDecision(PolicyResult.REJECT, PolicyReason.INVALID_ARGUMENTS)
    if not isinstance(arguments, dict) or len(arguments) > 0:
        return ToolDecision(PolicyResult.REJECT, PolicyReason.INVALID_ARGUMENTS)

    deadline_query = _authorize_deadline_query(prompt)
    if isinstance(deadline_query, str) or (
        isinstance(deadline_query, ToolDecision)
        and deadline_query.reason == PolicyReason.MISSING_DEADLINE_SCOPE
    ):
        return ToolDecision(
            PolicyResult.REJECT, PolicyReason.DEADLINE_SCOPE_MISMATCH
        )

    p = re.sub(r"\s+", " ", prompt.lower().strip())

    # LIST returns every task. A request to select tasks by due date or a
    # deadline scope must not be silently broadened into an all-task LIST.
    all_tasks_with_due_details = (
        re.search(r"\b(?:toàn bộ|tất cả|mọi)\s+(?:danh sách|task|việc|đầu việc)\b", p)
        and re.search(r"\b(?:kèm|cùng với)\s+(?:thời hạn|deadline)\b", p)
    )
    if not all_tasks_with_due_details and re.search(
        r"\b(?:đến|tới|quá|trễ)\s+hạn\b|\b(?:theo|có|thuộc)\s+"
        r"(?:thời\s+)?hạn\b|\bthời\s+hạn\b|\bmốc\s+chót\b|"
        r"\bdeadline\s+(?:thuộc|vào|ngày|hôm)\b|\btrễ\s+deadline\b|"
        r"\b(?:due|overdue)\b|\bđến\s+kỳ\s+xử\s+lý\b|"
        r"\bphải\s+(?:làm\s+)?xong\s+(?:vào\s+)?(?:hôm\s+nay|ngày\s+mai)\b",
        p,
    ):
        return ToolDecision(PolicyResult.REJECT, PolicyReason.DEADLINE_SCOPE_MISMATCH)

    # Negation must target the LIST action. "Chưa hoàn thành" and "không lọc"
    # describe which stored tasks to show, not a request to avoid LIST.
    if _NEGATED_COMMAND_HEAD.match(prompt) or re.match(
        rf"^\s*(?:{_PRONOUN}{_H})?(?:đừng|không{_H}cần|chưa{_H}cần|"
        rf"không{_H}muốn|khỏi|không){_H}(?:xem|mở|hiển{_H}thị|"
        rf"liệt{_H}kê|show|đọc)\b", p
    ):
        return ToolDecision(PolicyResult.REJECT, PolicyReason.NEGATED_REQUEST)

    # An explicit request not to execute also cancels a quoted/meta LIST.
    if re.search(r"\b(?:không cần|đừng)\s+(?:chạy|thực hiện)\b", p):
        return ToolDecision(PolicyResult.REJECT, PolicyReason.NEGATED_REQUEST)

    if re.search(
        r"\b(?:nhưng|mà)\s+(?:đừng|không cần|chưa cần|không muốn|khỏi)\s+"
        r"(?:xem|mở|hiển thị|liệt kê|show|đọc)\b", p
    ):
        return ToolDecision(PolicyResult.REJECT, PolicyReason.NEGATED_REQUEST)

    # A request to view only completed tasks needs a filter that LIST does not
    # provide. Mentioning completed tasks as part of *all* tasks is supported.
    if re.search(r"\b(?:chỉ|riêng|lọc riêng)\b", p) and re.search(
        r"\b(?:đã hoàn thành|đã xong|completed)\b", p
    ):
        return ToolDecision(PolicyResult.REJECT, PolicyReason.UNSUPPORTED_ACTION)
    if re.search(
        r"\b(?:đừng|không)\s+(?:hiện|hiển thị|đưa ra)\s+"
        r"(?:các\s+)?(?:task|việc)\s+chưa\s+(?:xong|hoàn thành)\b", p
    ):
        return ToolDecision(PolicyResult.REJECT, PolicyReason.UNSUPPORTED_ACTION)

    if re.search(
        r"(?:[,;]\s*|\b(?:và|rồi|sau đó|đồng thời)\s+)"
        r"(?:xóa|xoá|sửa|thêm|cập nhật|hoàn thành|đánh dấu)\b", p
    ):
        return ToolDecision(PolicyResult.REJECT, PolicyReason.UNSUPPORTED_ACTION)

    # A command for another action cannot authorize LIST merely because its
    # object mentions a list. Normalize framing whitespace, never task data.
    if _UNSUPPORTED_HEAD.match(p) or _CREATE_HEAD.match(prompt):
        return ToolDecision(PolicyResult.REJECT, PolicyReason.UNSUPPORTED_ACTION)

    # "Việc đã thêm" is an ambiguous description of stored entries. Keep the
    # existing session contract until a separately reviewed LIST frame covers it.
    if re.search(r"\bdanh sách việc đã thêm\b", p):
        return ToolDecision(PolicyResult.REJECT, PolicyReason.BARE_STATEMENT)

    # "Có những việc nào giúp tôi ngủ ngon?" asks for advice, not saved tasks.
    if re.match(r"^có những việc (?:nào|gì)\b", p) and not re.search(
        r"\b(?:danh sách|đã lưu|đã ghi|đã note|hiện có)\b", p
    ):
        return ToolDecision(PolicyResult.REJECT, PolicyReason.BARE_STATEMENT)

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
        rf"xem|mở|đọc|đưa\s+ra|hiển thị|liệt kê|show|kiểm tra|cho\s+{_PRONOUN}\s+(?:xem|biết)"
        rf"|bạn\s+đọc\s+lại|{_PRONOUN}\s+(?:muốn\s+xem\s+lại|còn)"
        rf"|{_PRONOUN}\s+cần\s+xem"
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
        "đọc",
        "đưa ra",
    ]
    object_words = ["danh sách", "todo", "task", "các việc", "những việc", "mọi việc", "đầu việc", "việc chưa làm", "đã ghi", "đã lưu", "đã note"]

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
    completed = _authorize_complete(prompt)
    if isinstance(completed, _CompleteRequest):
        return RequestKind.COMPLETE
    if completed.reason == PolicyReason.MISSING_TASK_ID:
        return RequestKind.MISSING_COMPLETE
    if completed.reason == PolicyReason.MULTIPLE_TASK_IDS:
        return RequestKind.MULTIPLE_COMPLETE
    edited = _authorize_edit(prompt)
    if isinstance(edited, _EditRequest):
        return RequestKind.EDIT
    if edited.reason == PolicyReason.MISSING_UPDATE_ID:
        return RequestKind.MISSING_EDIT_ID
    if edited.reason == PolicyReason.MISSING_UPDATE_CONTENT:
        return RequestKind.MISSING_EDIT_CONTENT
    if edited.reason == PolicyReason.MULTIPLE_TASK_IDS:
        return RequestKind.MULTIPLE_EDIT
    if edited.reason == PolicyReason.NEGATED_REQUEST:
        return RequestKind.NEGATED
    deleted = _authorize_delete(prompt)
    if isinstance(deleted, _DeleteRequest):
        return RequestKind.DELETE
    if deleted.reason == PolicyReason.MISSING_DELETE_ID:
        return RequestKind.MISSING_DELETE_ID
    if deleted.reason == PolicyReason.MULTIPLE_TASK_IDS:
        return RequestKind.MULTIPLE_DELETE
    if deleted.reason == PolicyReason.NEGATED_REQUEST:
        return RequestKind.NEGATED
    deadline = _authorize_deadline(prompt)
    if isinstance(deadline, _DeadlineRequest):
        return RequestKind.DEADLINE
    if deadline.reason == PolicyReason.MISSING_DEADLINE_ID:
        return RequestKind.MISSING_DEADLINE_ID
    if deadline.reason in (
        PolicyReason.MISSING_DEADLINE_TIME,
        PolicyReason.INVALID_DEADLINE_TIME,
    ):
        return RequestKind.MISSING_DEADLINE_TIME
    if deadline.reason == PolicyReason.MULTIPLE_TASK_IDS:
        return RequestKind.MULTIPLE_DEADLINE
    if deadline.reason == PolicyReason.NEGATED_REQUEST:
        return RequestKind.NEGATED
    deadline_query = _authorize_deadline_query(prompt)
    if isinstance(deadline_query, str):
        return RequestKind.DEADLINE_QUERY
    if deadline_query.reason == PolicyReason.MISSING_DEADLINE_SCOPE:
        return RequestKind.MISSING_DEADLINE_SCOPE
    if deadline_query.reason == PolicyReason.NEGATED_REQUEST:
        return RequestKind.NEGATED
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
    if _COMPLETE_COMMAND_HEAD.match(prompt):
        return RequestKind.COMPLETE
    if _EDIT_COMMAND_HEAD.match(prompt):
        return RequestKind.EDIT
    if _DELETE_COMMAND_HEAD.match(prompt):
        return RequestKind.DELETE
    if _DEADLINE_COMMAND_HEAD.match(prompt):
        return RequestKind.DEADLINE
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


def policy_for_tool(
    prompt: Any,
    tool_name: Any,
    arguments: Any,
    *,
    reference_time: datetime | None = None,
) -> ToolDecision:
    if not isinstance(tool_name, str) or not tool_name.strip():
        return ToolDecision(PolicyResult.REJECT, PolicyReason.INVALID_ARGUMENTS)

    if tool_name == "create_task":
        return policy_for_create_task(prompt, arguments)
    if tool_name == "list_tasks":
        return policy_for_list_tasks(prompt, arguments)
    if tool_name == "list_tasks_by_deadline":
        return policy_for_list_tasks_by_deadline(prompt, arguments)
    if tool_name == "complete_task":
        return policy_for_complete_task(prompt, arguments)
    if tool_name == "update_task":
        return policy_for_update_task(prompt, arguments)
    if tool_name == "delete_task":
        return policy_for_delete_task(prompt, arguments)
    if tool_name == "set_task_deadline":
        return policy_for_set_task_deadline(
            prompt, arguments, reference_time=reference_time
        )

    return ToolDecision(PolicyResult.REJECT, PolicyReason.UNSUPPORTED_TOOL)
