"""In-memory continuation of incomplete task commands and DELETE confirmation."""

import re
from datetime import datetime
from copy import deepcopy
from enum import Enum
from pathlib import Path

from nexus.agent.client import PostToolExecutionError, TurnStatus, chat, run_turn
from nexus.agent.prompts import DEFAULT_PROMPT_VERSION
from nexus.agent.routing import DEFAULT_TOOL_ROUTING, ToolRoutingMode
from nexus.agent.policy import (
    PolicyReason,
    RequestKind,
    classify_request,
    deadline_request_fields,
    deadline_scope_reply,
    edit_request_fields,
)
from nexus.agent.responses import format_rejection, format_tool_result
from nexus.agent.tools import execute_tool
from nexus.core.deadlines import DeadlineParseError, format_deadline, parse_deadline, vietnam_now
from nexus.core.models import Task
from nexus.storage.sqlite_db import get_task


_CANCEL = re.compile(r"^\s*(?:thôi|hủy|huỷ)[.!]?\s*$", re.IGNORECASE)
_NO_ACTION = "Không có thao tác nào được thực hiện."
_ASK_CONTENT = "Bạn muốn thêm việc gì?"
_ASK_COMPLETE_ID = "Bạn muốn hoàn thành việc có ID nào?"
_ASK_EDIT_ID = "Bạn muốn sửa việc có ID nào?"
_ASK_DELETE_ID = "Bạn muốn xóa việc có ID nào?"
_ASK_DEADLINE_ID = "Bạn muốn đặt hạn cho việc có ID nào?"
_ASK_DEADLINE_TEXT = "Bạn muốn đặt thời hạn khi nào? Ví dụ: 8 giờ sáng mai."
_ASK_DEADLINE_SCOPE = "Bạn muốn xem việc đến hạn hôm nay, ngày mai hay đã quá hạn?"
_CONFIRM_DELETE_YES = re.compile(
    r"^\s*(?:có|đồng ý|xác nhận|xóa|xoá)[.!]?\s*$", re.IGNORECASE
)
_CONFIRM_DELETE_NO = re.compile(
    r"^\s*(?:không|thôi|hủy|huỷ)[.!]?\s*$", re.IGNORECASE
)
_COMPLETE_ID_REPLY = re.compile(
    r"^\s*(?:#?(?P<bare>[0-9]+)|(?:việc|task)\s+#?(?P<labelled>[0-9]+))[.!]?\s*$",
    re.IGNORECASE,
)


class SessionState(str, Enum):
    """Conversation state kept in memory by one AgentSession instance."""

    IDLE = "idle"
    AWAITING_CREATE_CONTENT = "awaiting_create_content"
    AWAITING_COMPLETE_ID = "awaiting_complete_id"
    AWAITING_EDIT_ID = "awaiting_edit_id"
    AWAITING_EDIT_CONTENT = "awaiting_edit_content"
    AWAITING_DELETE_ID = "awaiting_delete_id"
    AWAITING_DELETE_CONFIRMATION = "awaiting_delete_confirmation"
    AWAITING_DEADLINE_ID = "awaiting_deadline_id"
    AWAITING_DEADLINE_TEXT = "awaiting_deadline_text"
    AWAITING_DEADLINE_SCOPE = "awaiting_deadline_scope"


def _session_result(
    status: TurnStatus,
    reply: str,
    *,
    source: str = "session",
    calls: list[dict] | None = None,
    authorized_calls: list[dict] | None = None,
    confirmation: dict | None = None,
) -> dict:
    result = {
        "calls": calls if calls is not None else [],
        "proposed_calls": [],
        "authorized_calls": authorized_calls if authorized_calls is not None else [],
        "rejected_calls": [],
        "status": status.value,
        "model_reply": None,
        "source": source,
        "reply": reply,
    }
    if confirmation is not None:
        result["confirmation"] = confirmation
    return result


class AgentSession:
    """One in-memory conversation bound to one database.

    Create one instance per conversation. Instances must not be shared between
    users, and their state is deliberately not persisted across restarts.
    """

    def __init__(
        self,
        database_path: str | Path,
        *,
        prompt_version: str = DEFAULT_PROMPT_VERSION,
        settings: dict | None = None,
        clock=None,
        tool_routing: ToolRoutingMode = DEFAULT_TOOL_ROUTING,
    ) -> None:
        self.database_path = Path(database_path)
        self.prompt_version = prompt_version
        self.settings = deepcopy(settings)
        self.clock = clock or vietnam_now
        if not isinstance(tool_routing, ToolRoutingMode):
            raise TypeError("tool_routing must be a ToolRoutingMode")
        self.tool_routing = tool_routing
        self.pending_edit_id: int | None = None
        self.pending_edit_content: str | None = None
        self.pending_delete_task: Task | None = None
        self.pending_deadline_id: int | None = None
        self.pending_deadline_text: str | None = None
        self.pending_deadline_reference: datetime | None = None
        self._set_state(SessionState.IDLE)

    def _set_state(
        self,
        state: SessionState,
        *,
        edit_id: int | None = None,
        edit_content: str | None = None,
        delete_task: Task | None = None,
        deadline_id: int | None = None,
        deadline_text: str | None = None,
        deadline_reference: datetime | None = None,
    ) -> None:
        self.state = state
        self.pending_edit_id = edit_id if state == SessionState.AWAITING_EDIT_CONTENT else None
        self.pending_edit_content = (
            edit_content if state == SessionState.AWAITING_EDIT_ID else None
        )
        self.pending_delete_task = (
            delete_task
            if state == SessionState.AWAITING_DELETE_CONFIRMATION
            else None
        )
        self.pending_deadline_id = (
            deadline_id if state == SessionState.AWAITING_DEADLINE_TEXT else None
        )
        self.pending_deadline_text = (
            deadline_text if state == SessionState.AWAITING_DEADLINE_ID else None
        )
        self.pending_deadline_reference = (
            deadline_reference
            if state == SessionState.AWAITING_DEADLINE_ID
            and deadline_text is not None
            else None
        )

    @property
    def pending_create(self) -> bool:
        """Backward-compatible view of the explicit session state."""
        return self.state == SessionState.AWAITING_CREATE_CONTENT

    @pending_create.setter
    def pending_create(self, value: bool) -> None:
        self._set_state(
            SessionState.AWAITING_CREATE_CONTENT if value else SessionState.IDLE
        )

    @property
    def pending_complete(self) -> bool:
        return self.state == SessionState.AWAITING_COMPLETE_ID

    @property
    def pending_edit(self) -> bool:
        return self.state in (
            SessionState.AWAITING_EDIT_ID,
            SessionState.AWAITING_EDIT_CONTENT,
        )

    def _edit_content_question(self) -> str:
        return f"Bạn muốn đổi nội dung việc {self.pending_edit_id} thành gì?"

    def _delete_confirmation_question(self) -> str:
        task = self.pending_delete_task
        if task is None:
            raise RuntimeError("delete confirmation requires a task snapshot")
        marker = "x" if task.completed else " "
        description = f"[{task.id}] [{marker}] {task.content}"
        if task.due_at is not None:
            description += f" — hạn {format_deadline(task.due_at)}"
        return f"Bạn có chắc muốn xóa {description}?"

    def _finish(self, result: dict, state_before: SessionState) -> dict:
        """Attach the state transition to every completed turn."""
        return {
            **result,
            "session_state_before": state_before.value,
            "session_state_after": self.state.value,
        }

    def run_turn(self, prompt: str, generate=chat) -> dict:
        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string")
        state_before = self.state
        turn_reference = vietnam_now(self.clock())
        if not prompt.strip():
            if self.pending_create:
                result = _session_result(TurnStatus.NEEDS_CLARIFICATION, _ASK_CONTENT)
            elif self.pending_complete:
                result = _session_result(
                    TurnStatus.NEEDS_CLARIFICATION, _ASK_COMPLETE_ID
                )
            elif self.state == SessionState.AWAITING_EDIT_ID:
                result = _session_result(TurnStatus.NEEDS_CLARIFICATION, _ASK_EDIT_ID)
            elif self.state == SessionState.AWAITING_EDIT_CONTENT:
                result = _session_result(
                    TurnStatus.NEEDS_CLARIFICATION, self._edit_content_question()
                )
            elif self.state == SessionState.AWAITING_DELETE_ID:
                result = _session_result(
                    TurnStatus.NEEDS_CLARIFICATION, _ASK_DELETE_ID
                )
            elif self.state == SessionState.AWAITING_DELETE_CONFIRMATION:
                result = _session_result(
                    TurnStatus.NEEDS_CONFIRMATION,
                    self._delete_confirmation_question(),
                    confirmation=self._delete_confirmation_payload(),
                )
            elif self.state == SessionState.AWAITING_DEADLINE_ID:
                result = _session_result(
                    TurnStatus.NEEDS_CLARIFICATION, _ASK_DEADLINE_ID
                )
            elif self.state == SessionState.AWAITING_DEADLINE_TEXT:
                result = _session_result(
                    TurnStatus.NEEDS_CLARIFICATION, _ASK_DEADLINE_TEXT
                )
            elif self.state == SessionState.AWAITING_DEADLINE_SCOPE:
                result = _session_result(
                    TurnStatus.NEEDS_CLARIFICATION, _ASK_DEADLINE_SCOPE
                )
            else:
                result = _session_result(TurnStatus.NO_TOOL, _NO_ACTION)
            return self._finish(result, state_before)

        if self.state != SessionState.IDLE and _CANCEL.fullmatch(prompt):
            cancelled_state = self.state
            self._set_state(SessionState.IDLE)
            if cancelled_state == SessionState.AWAITING_CREATE_CONTENT:
                reply = "Đã hủy yêu cầu thêm việc. Không có việc nào được lưu."
            elif cancelled_state == SessionState.AWAITING_COMPLETE_ID:
                reply = "Đã hủy yêu cầu hoàn thành việc. Không có thay đổi nào được lưu."
            elif cancelled_state in (
                SessionState.AWAITING_EDIT_ID, SessionState.AWAITING_EDIT_CONTENT
            ):
                reply = "Đã hủy yêu cầu sửa việc. Không có thay đổi nào được lưu."
            elif cancelled_state in (
                SessionState.AWAITING_DEADLINE_ID,
                SessionState.AWAITING_DEADLINE_TEXT,
            ):
                reply = "Đã hủy yêu cầu đặt thời hạn. Không có thay đổi nào được lưu."
            elif cancelled_state == SessionState.AWAITING_DEADLINE_SCOPE:
                reply = "Đã hủy yêu cầu xem việc theo hạn."
            else:
                reply = "Đã hủy yêu cầu xóa việc. Không có việc nào bị xóa."
            result = _session_result(TurnStatus.CANCELLED, reply)
            return self._finish(result, state_before)

        if self.state == SessionState.AWAITING_DELETE_CONFIRMATION:
            if _CONFIRM_DELETE_YES.fullmatch(prompt):
                result = self._delete_followup()
                return self._finish(result, state_before)
            if _CONFIRM_DELETE_NO.fullmatch(prompt):
                self._set_state(SessionState.IDLE)
                result = _session_result(
                    TurnStatus.CANCELLED,
                    "Đã hủy yêu cầu xóa việc. Không có việc nào bị xóa.",
                )
                return self._finish(result, state_before)

        kind = classify_request(prompt)
        if kind == RequestKind.MISSING_CREATE:
            self._set_state(SessionState.AWAITING_CREATE_CONTENT)
            result = _session_result(TurnStatus.NEEDS_CLARIFICATION, _ASK_CONTENT)
            return self._finish(result, state_before)

        if kind in (RequestKind.MISSING_COMPLETE, RequestKind.MULTIPLE_COMPLETE):
            self._set_state(SessionState.AWAITING_COMPLETE_ID)
            result = _session_result(
                TurnStatus.NEEDS_CLARIFICATION, _ASK_COMPLETE_ID
            )
            return self._finish(result, state_before)

        if kind in (RequestKind.MISSING_EDIT_ID, RequestKind.MULTIPLE_EDIT):
            _task_id, content = edit_request_fields(prompt)
            self._set_state(
                SessionState.AWAITING_EDIT_ID, edit_content=content
            )
            result = _session_result(TurnStatus.NEEDS_CLARIFICATION, _ASK_EDIT_ID)
            return self._finish(result, state_before)

        if kind == RequestKind.MISSING_EDIT_CONTENT:
            task_id, _content = edit_request_fields(prompt)
            self._set_state(SessionState.AWAITING_EDIT_CONTENT, edit_id=task_id)
            result = _session_result(
                TurnStatus.NEEDS_CLARIFICATION, self._edit_content_question()
            )
            return self._finish(result, state_before)

        if kind in (RequestKind.MISSING_DELETE_ID, RequestKind.MULTIPLE_DELETE):
            self._set_state(SessionState.AWAITING_DELETE_ID)
            result = _session_result(
                TurnStatus.NEEDS_CLARIFICATION, _ASK_DELETE_ID
            )
            return self._finish(result, state_before)

        if kind in (RequestKind.MISSING_DEADLINE_ID, RequestKind.MULTIPLE_DEADLINE):
            _task_id, when = deadline_request_fields(prompt)
            self._set_state(
                SessionState.AWAITING_DEADLINE_ID,
                deadline_text=when,
                deadline_reference=turn_reference if when is not None else None,
            )
            result = _session_result(
                TurnStatus.NEEDS_CLARIFICATION, _ASK_DEADLINE_ID
            )
            return self._finish(result, state_before)

        if kind == RequestKind.MISSING_DEADLINE_TIME:
            task_id, _when = deadline_request_fields(prompt)
            self._set_state(
                SessionState.AWAITING_DEADLINE_TEXT, deadline_id=task_id
            )
            result = _session_result(
                TurnStatus.NEEDS_CLARIFICATION, _ASK_DEADLINE_TEXT
            )
            return self._finish(result, state_before)

        if kind == RequestKind.MISSING_DEADLINE_SCOPE:
            self._set_state(SessionState.AWAITING_DEADLINE_SCOPE)
            result = _session_result(
                TurnStatus.NEEDS_CLARIFICATION, _ASK_DEADLINE_SCOPE
            )
            return self._finish(result, state_before)

        if kind in (RequestKind.UNSUPPORTED, RequestKind.NEGATED):
            pending_state = self.state
            self._set_state(SessionState.IDLE)
            if kind == RequestKind.NEGATED and pending_state != SessionState.IDLE:
                if pending_state == SessionState.AWAITING_CREATE_CONTENT:
                    reply = "Đã hủy yêu cầu thêm việc. Không có việc nào được lưu."
                elif pending_state == SessionState.AWAITING_COMPLETE_ID:
                    reply = "Đã hủy yêu cầu hoàn thành việc. Không có thay đổi nào được lưu."
                elif pending_state in (
                    SessionState.AWAITING_EDIT_ID, SessionState.AWAITING_EDIT_CONTENT
                ):
                    reply = "Đã hủy yêu cầu sửa việc. Không có thay đổi nào được lưu."
                elif pending_state in (
                    SessionState.AWAITING_DEADLINE_ID,
                    SessionState.AWAITING_DEADLINE_TEXT,
                ):
                    reply = "Đã hủy yêu cầu đặt thời hạn. Không có thay đổi nào được lưu."
                elif pending_state == SessionState.AWAITING_DEADLINE_SCOPE:
                    reply = "Đã hủy yêu cầu xem việc theo hạn."
                else:
                    reply = "Đã hủy yêu cầu xóa việc. Không có việc nào bị xóa."
                result = _session_result(TurnStatus.CANCELLED, reply)
            else:
                reply = (
                    format_rejection(PolicyReason.UNSUPPORTED_ACTION)
                    if kind == RequestKind.UNSUPPORTED
                    else _NO_ACTION
                )
                result = _session_result(TurnStatus.REJECTED, reply)
            return self._finish(result, state_before)

        if (
            self.state == SessionState.AWAITING_DEADLINE_SCOPE
            and kind == RequestKind.OTHER
        ):
            scope = deadline_scope_reply(prompt)
            if scope is None:
                result = _session_result(
                    TurnStatus.NEEDS_CLARIFICATION, _ASK_DEADLINE_SCOPE
                )
            else:
                result = self._deadline_query_followup(scope, turn_reference)
            return self._finish(result, state_before)

        if self.pending_create and kind == RequestKind.OTHER:
            result = self._save_followup(prompt)
            return self._finish(result, state_before)

        if self.pending_complete and kind == RequestKind.OTHER:
            match = _COMPLETE_ID_REPLY.fullmatch(prompt)
            if match is None:
                result = _session_result(
                    TurnStatus.NEEDS_CLARIFICATION, _ASK_COMPLETE_ID
                )
            else:
                task_id = int(match.group("bare") or match.group("labelled"))
                if task_id <= 0:
                    result = _session_result(
                        TurnStatus.NEEDS_CLARIFICATION, _ASK_COMPLETE_ID
                    )
                else:
                    result = self._complete_followup(task_id)
            return self._finish(result, state_before)

        if self.state == SessionState.AWAITING_DELETE_ID and kind == RequestKind.OTHER:
            match = _COMPLETE_ID_REPLY.fullmatch(prompt)
            if match is None:
                result = _session_result(
                    TurnStatus.NEEDS_CLARIFICATION, _ASK_DELETE_ID
                )
            else:
                task_id = int(match.group("bare") or match.group("labelled"))
                if task_id <= 0:
                    result = _session_result(
                        TurnStatus.NEEDS_CLARIFICATION, _ASK_DELETE_ID
                    )
                else:
                    result = self._prepare_delete(task_id)
            return self._finish(result, state_before)

        if self.state == SessionState.AWAITING_DEADLINE_ID and kind == RequestKind.OTHER:
            match = _COMPLETE_ID_REPLY.fullmatch(prompt)
            if match is None:
                result = _session_result(
                    TurnStatus.NEEDS_CLARIFICATION, _ASK_DEADLINE_ID
                )
            else:
                task_id = int(match.group("bare") or match.group("labelled"))
                if task_id <= 0:
                    result = _session_result(
                        TurnStatus.NEEDS_CLARIFICATION, _ASK_DEADLINE_ID
                    )
                elif self.pending_deadline_text is None:
                    self._set_state(
                        SessionState.AWAITING_DEADLINE_TEXT, deadline_id=task_id
                    )
                    result = _session_result(
                        TurnStatus.NEEDS_CLARIFICATION, _ASK_DEADLINE_TEXT
                    )
                else:
                    reference = self.pending_deadline_reference or turn_reference
                    try:
                        parse_deadline(self.pending_deadline_text, reference=reference)
                    except DeadlineParseError:
                        self._set_state(
                            SessionState.AWAITING_DEADLINE_TEXT,
                            deadline_id=task_id,
                        )
                        result = _session_result(
                            TurnStatus.NEEDS_CLARIFICATION, _ASK_DEADLINE_TEXT
                        )
                    else:
                        result = self._deadline_followup(
                            task_id, self.pending_deadline_text, reference
                        )
            return self._finish(result, state_before)

        if self.state == SessionState.AWAITING_DEADLINE_TEXT and kind == RequestKind.OTHER:
            try:
                parse_deadline(prompt, reference=turn_reference)
            except DeadlineParseError:
                result = _session_result(
                    TurnStatus.NEEDS_CLARIFICATION, _ASK_DEADLINE_TEXT
                )
            else:
                result = self._deadline_followup(
                    self.pending_deadline_id, prompt, turn_reference
                )
            return self._finish(result, state_before)

        if self.state == SessionState.AWAITING_EDIT_ID and kind == RequestKind.OTHER:
            match = _COMPLETE_ID_REPLY.fullmatch(prompt)
            if match is None:
                result = _session_result(TurnStatus.NEEDS_CLARIFICATION, _ASK_EDIT_ID)
            else:
                task_id = int(match.group("bare") or match.group("labelled"))
                if task_id <= 0:
                    result = _session_result(TurnStatus.NEEDS_CLARIFICATION, _ASK_EDIT_ID)
                elif self.pending_edit_content is None:
                    self._set_state(
                        SessionState.AWAITING_EDIT_CONTENT, edit_id=task_id
                    )
                    result = _session_result(
                        TurnStatus.NEEDS_CLARIFICATION,
                        self._edit_content_question(),
                    )
                else:
                    result = self._update_followup(task_id, self.pending_edit_content)
            return self._finish(result, state_before)

        if self.state == SessionState.AWAITING_EDIT_CONTENT and kind == RequestKind.OTHER:
            if "\n" in prompt or "\r" in prompt or not prompt.strip():
                result = _session_result(
                    TurnStatus.NEEDS_CLARIFICATION, self._edit_content_question()
                )
            else:
                result = self._update_followup(self.pending_edit_id, prompt)
            return self._finish(result, state_before)

        if (
            self.state == SessionState.AWAITING_DELETE_CONFIRMATION
            and kind == RequestKind.OTHER
        ):
            result = _session_result(
                TurnStatus.NEEDS_CONFIRMATION,
                self._delete_confirmation_question(),
                confirmation=self._delete_confirmation_payload(),
            )
            return self._finish(result, state_before)

        # A new explicit command replaces any pending request.
        self._set_state(SessionState.IDLE)
        result = run_turn(
            self.database_path,
            prompt,
            generate,
            prompt_version=self.prompt_version,
            settings=self.settings,
            reference_time=turn_reference,
            tool_routing=self.tool_routing,
        )
        if result["status"] == TurnStatus.NEEDS_CONFIRMATION.value:
            confirmation = result.get("confirmation")
            task_data = confirmation.get("task") if isinstance(confirmation, dict) else None
            if not isinstance(task_data, dict):
                raise RuntimeError("delete confirmation is missing its task snapshot")
            self._set_state(
                SessionState.AWAITING_DELETE_CONFIRMATION,
                delete_task=Task(**task_data),
            )
        if result["status"] == TurnStatus.NEEDS_CLARIFICATION.value:
            rejected = result.get("rejected_calls", [])
            reasons_by_tool = {
                call.get("name"): call.get("reason") for call in rejected
            }
            deadline_reason = reasons_by_tool.get("set_task_deadline")
            if reasons_by_tool.get("list_tasks_by_deadline") == "missing_deadline_scope":
                self._set_state(SessionState.AWAITING_DEADLINE_SCOPE)
            elif deadline_reason in {"missing_deadline_id", "multiple_task_ids"}:
                _task_id, when = deadline_request_fields(prompt)
                self._set_state(
                    SessionState.AWAITING_DEADLINE_ID,
                    deadline_text=when,
                    deadline_reference=turn_reference if when is not None else None,
                )
            elif deadline_reason in {
                "missing_deadline_time", "invalid_deadline_time"
            }:
                task_id, _when = deadline_request_fields(prompt)
                self._set_state(
                    SessionState.AWAITING_DEADLINE_TEXT, deadline_id=task_id
                )
            elif reasons_by_tool.get("complete_task") in {
                "missing_task_id", "multiple_task_ids"
            }:
                self._set_state(SessionState.AWAITING_COMPLETE_ID)
            elif reasons_by_tool.get("update_task") == "missing_update_id":
                _task_id, content = edit_request_fields(prompt)
                self._set_state(
                    SessionState.AWAITING_EDIT_ID, edit_content=content
                )
            elif reasons_by_tool.get("update_task") == "missing_update_content":
                task_id, _content = edit_request_fields(prompt)
                self._set_state(
                    SessionState.AWAITING_EDIT_CONTENT, edit_id=task_id
                )
            elif reasons_by_tool.get("delete_task") == "missing_delete_id":
                self._set_state(SessionState.AWAITING_DELETE_ID)
            elif reasons_by_tool.get("create_task") == "missing_content":
                self._set_state(SessionState.AWAITING_CREATE_CONTENT)
        return self._finish({**result, "source": "model"}, state_before)

    def _save_followup(self, content: str) -> dict:
        arguments = {"content": content}
        authorized_calls = [{
            "name": "create_task",
            "arguments": arguments,
            "result": "allow",
            "reason": "session_continuation",
        }]
        # execute_tool is atomic for multiline CREATE. Keep pending on failure
        # so the user may deliberately retry with a new message.
        result = execute_tool(self.database_path, "create_task", arguments)
        calls = [{"name": "create_task", "arguments": arguments, "result": result}]
        self._set_state(SessionState.IDLE)
        try:
            reply = format_tool_result("create_task", result)
        except Exception as error:
            raise PostToolExecutionError(
                "response_formatting",
                calls,
                error,
                authorized_calls=authorized_calls,
            ) from error
        return _session_result(
            TurnStatus.EXECUTED,
            reply,
            source="session_continuation",
            calls=calls,
            authorized_calls=authorized_calls,
        )

    def _complete_followup(self, task_id: int) -> dict:
        arguments = {"id": task_id}
        authorized_calls = [{
            "name": "complete_task",
            "arguments": arguments,
            "result": "allow",
            "reason": "session_continuation",
        }]
        # Keep pending if SQLite fails before returning a committed result.
        result = execute_tool(self.database_path, "complete_task", arguments)
        calls = [{"name": "complete_task", "arguments": arguments, "result": result}]
        self._set_state(SessionState.IDLE)
        try:
            reply = format_tool_result("complete_task", result)
        except Exception as error:
            raise PostToolExecutionError(
                "response_formatting",
                calls,
                error,
                authorized_calls=authorized_calls,
            ) from error
        return _session_result(
            TurnStatus.EXECUTED,
            reply,
            source="session_continuation",
            calls=calls,
            authorized_calls=authorized_calls,
        )

    def _update_followup(self, task_id: int, content: str) -> dict:
        arguments = {"id": task_id, "content": content}
        authorized_calls = [{
            "name": "update_task",
            "arguments": arguments,
            "result": "allow",
            "reason": "session_continuation",
        }]
        # Keep the current pending state if SQLite fails before returning.
        result = execute_tool(self.database_path, "update_task", arguments)
        calls = [{"name": "update_task", "arguments": arguments, "result": result}]
        self._set_state(SessionState.IDLE)
        try:
            reply = format_tool_result("update_task", result)
        except Exception as error:
            raise PostToolExecutionError(
                "response_formatting",
                calls,
                error,
                authorized_calls=authorized_calls,
            ) from error
        return _session_result(
            TurnStatus.EXECUTED,
            reply,
            source="session_continuation",
            calls=calls,
            authorized_calls=authorized_calls,
        )

    def _deadline_query_followup(
        self, scope: str, reference_time: datetime
    ) -> dict:
        arguments = {"scope": scope}
        authorized_calls = [{
            "name": "list_tasks_by_deadline",
            "arguments": arguments,
            "result": "allow",
            "reason": "session_continuation",
        }]
        result = execute_tool(
            self.database_path,
            "list_tasks_by_deadline",
            arguments,
            reference_time=reference_time,
        )
        calls = [{
            "name": "list_tasks_by_deadline",
            "arguments": arguments,
            "result": result,
        }]
        self._set_state(SessionState.IDLE)
        try:
            reply = format_tool_result("list_tasks_by_deadline", result)
        except Exception as error:
            raise PostToolExecutionError(
                "response_formatting",
                calls,
                error,
                authorized_calls=authorized_calls,
            ) from error
        return _session_result(
            TurnStatus.EXECUTED,
            reply,
            source="session_continuation",
            calls=calls,
            authorized_calls=authorized_calls,
        )

    def _deadline_followup(
        self, task_id: int, when: str, reference_time: datetime
    ) -> dict:
        arguments = {"id": task_id, "when": when}
        authorized_calls = [{
            "name": "set_task_deadline",
            "arguments": arguments,
            "result": "allow",
            "reason": "session_continuation",
        }]
        result = execute_tool(
            self.database_path,
            "set_task_deadline",
            arguments,
            reference_time=reference_time,
        )
        calls = [{
            "name": "set_task_deadline",
            "arguments": arguments,
            "result": result,
        }]
        self._set_state(SessionState.IDLE)
        try:
            reply = format_tool_result("set_task_deadline", result)
        except Exception as error:
            raise PostToolExecutionError(
                "response_formatting",
                calls,
                error,
                authorized_calls=authorized_calls,
            ) from error
        status = (
            TurnStatus.NOT_FOUND
            if result.get("status") == "not_found"
            else TurnStatus.EXECUTED
        )
        return _session_result(
            status,
            reply,
            source="session_continuation",
            calls=calls,
            authorized_calls=authorized_calls,
        )

    def _delete_confirmation_payload(self) -> dict:
        task = self.pending_delete_task
        if task is None:
            raise RuntimeError("delete confirmation requires a task snapshot")
        return {
            "name": "delete_task",
            "arguments": {"id": task.id},
            "task": {
                "id": task.id,
                "content": task.content,
                "completed": task.completed,
                "due_at": task.due_at,
            },
        }

    def _prepare_delete(self, task_id: int) -> dict:
        task = get_task(self.database_path, task_id)
        if task is None:
            self._set_state(SessionState.IDLE)
            return _session_result(
                TurnStatus.NOT_FOUND, "Không tìm thấy việc có ID đã yêu cầu."
            )
        self._set_state(
            SessionState.AWAITING_DELETE_CONFIRMATION, delete_task=task
        )
        return _session_result(
            TurnStatus.NEEDS_CONFIRMATION,
            self._delete_confirmation_question(),
            source="session_continuation",
            authorized_calls=[{
                "name": "delete_task",
                "arguments": {"id": task_id},
                "result": "requires_confirmation",
                "reason": "session_continuation",
            }],
            confirmation=self._delete_confirmation_payload(),
        )

    def _delete_followup(self) -> dict:
        snapshot = self.pending_delete_task
        if snapshot is None:
            raise RuntimeError("delete confirmation requires a task snapshot")
        arguments = {"id": snapshot.id}
        authorized_calls = [{
            "name": "delete_task",
            "arguments": arguments,
            "result": "allow",
            "reason": "session_confirmation",
        }]
        result = execute_tool(
            self.database_path,
            "delete_task",
            arguments,
            confirmed_task=snapshot,
        )
        calls = [{"name": "delete_task", "arguments": arguments, "result": result}]
        status = result.get("status")
        if status == "stale":
            task_data = result.get("task")
            if not isinstance(task_data, dict):
                raise RuntimeError("stale delete result requires the current task")
            self._set_state(
                SessionState.AWAITING_DELETE_CONFIRMATION,
                delete_task=Task(**task_data),
            )
            reply = format_tool_result("delete_task", result)
            return _session_result(
                TurnStatus.NEEDS_CONFIRMATION,
                reply,
                source="session_confirmation",
                calls=calls,
                authorized_calls=authorized_calls,
                confirmation=self._delete_confirmation_payload(),
            )

        self._set_state(SessionState.IDLE)
        try:
            reply = format_tool_result("delete_task", result)
        except Exception as error:
            if status == "deleted":
                raise PostToolExecutionError(
                    "response_formatting",
                    calls,
                    error,
                    authorized_calls=authorized_calls,
                ) from error
            raise
        return _session_result(
            TurnStatus.EXECUTED if status == "deleted" else TurnStatus.NOT_FOUND,
            reply,
            source="session_confirmation",
            calls=calls,
            authorized_calls=authorized_calls,
        )
