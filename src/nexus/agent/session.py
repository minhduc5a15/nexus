"""In-memory continuation of incomplete CREATE and COMPLETE requests."""

import re
from copy import deepcopy
from enum import Enum
from pathlib import Path

from nexus.agent.client import PostToolExecutionError, TurnStatus, chat, run_turn
from nexus.agent.prompts import DEFAULT_PROMPT_VERSION
from nexus.agent.policy import PolicyReason, RequestKind, classify_request
from nexus.agent.responses import format_rejection, format_tool_result
from nexus.agent.tools import execute_tool


_CANCEL = re.compile(r"^\s*(?:thôi|hủy|huỷ)[.!]?\s*$", re.IGNORECASE)
_NO_ACTION = "Không có thao tác nào được thực hiện."
_ASK_CONTENT = "Bạn muốn thêm việc gì?"
_ASK_COMPLETE_ID = "Bạn muốn hoàn thành việc có ID nào?"
_COMPLETE_ID_REPLY = re.compile(
    r"^\s*(?:#?(?P<bare>[0-9]+)|(?:việc|task)\s+#?(?P<labelled>[0-9]+))[.!]?\s*$",
    re.IGNORECASE,
)


class SessionState(str, Enum):
    """Conversation state kept in memory by one AgentSession instance."""

    IDLE = "idle"
    AWAITING_CREATE_CONTENT = "awaiting_create_content"
    AWAITING_COMPLETE_ID = "awaiting_complete_id"


def _session_result(
    status: TurnStatus,
    reply: str,
    *,
    source: str = "session",
    calls: list[dict] | None = None,
    authorized_calls: list[dict] | None = None,
) -> dict:
    return {
        "calls": calls if calls is not None else [],
        "proposed_calls": [],
        "authorized_calls": authorized_calls if authorized_calls is not None else [],
        "rejected_calls": [],
        "status": status.value,
        "model_reply": None,
        "source": source,
        "reply": reply,
    }


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
    ) -> None:
        self.database_path = Path(database_path)
        self.prompt_version = prompt_version
        self.settings = deepcopy(settings)
        self.state = SessionState.IDLE

    @property
    def pending_create(self) -> bool:
        """Backward-compatible view of the explicit session state."""
        return self.state == SessionState.AWAITING_CREATE_CONTENT

    @pending_create.setter
    def pending_create(self, value: bool) -> None:
        self.state = (
            SessionState.AWAITING_CREATE_CONTENT if value else SessionState.IDLE
        )

    @property
    def pending_complete(self) -> bool:
        return self.state == SessionState.AWAITING_COMPLETE_ID

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
        if not prompt.strip():
            if self.pending_create:
                result = _session_result(TurnStatus.NEEDS_CLARIFICATION, _ASK_CONTENT)
            elif self.pending_complete:
                result = _session_result(
                    TurnStatus.NEEDS_CLARIFICATION, _ASK_COMPLETE_ID
                )
            else:
                result = _session_result(TurnStatus.NO_TOOL, _NO_ACTION)
            return self._finish(result, state_before)

        if self.state != SessionState.IDLE and _CANCEL.fullmatch(prompt):
            cancelled_state = self.state
            self.state = SessionState.IDLE
            if cancelled_state == SessionState.AWAITING_CREATE_CONTENT:
                reply = "Đã hủy yêu cầu thêm việc. Không có việc nào được lưu."
            else:
                reply = "Đã hủy yêu cầu hoàn thành việc. Không có thay đổi nào được lưu."
            result = _session_result(TurnStatus.CANCELLED, reply)
            return self._finish(result, state_before)

        kind = classify_request(prompt)
        if kind == RequestKind.MISSING_CREATE:
            self.state = SessionState.AWAITING_CREATE_CONTENT
            result = _session_result(TurnStatus.NEEDS_CLARIFICATION, _ASK_CONTENT)
            return self._finish(result, state_before)

        if kind in (RequestKind.MISSING_COMPLETE, RequestKind.MULTIPLE_COMPLETE):
            self.state = SessionState.AWAITING_COMPLETE_ID
            result = _session_result(
                TurnStatus.NEEDS_CLARIFICATION, _ASK_COMPLETE_ID
            )
            return self._finish(result, state_before)

        if kind in (RequestKind.UNSUPPORTED, RequestKind.NEGATED):
            pending_state = self.state
            self.state = SessionState.IDLE
            if kind == RequestKind.NEGATED and pending_state != SessionState.IDLE:
                if pending_state == SessionState.AWAITING_CREATE_CONTENT:
                    reply = "Đã hủy yêu cầu thêm việc. Không có việc nào được lưu."
                else:
                    reply = "Đã hủy yêu cầu hoàn thành việc. Không có thay đổi nào được lưu."
                result = _session_result(TurnStatus.CANCELLED, reply)
            else:
                reply = (
                    format_rejection(PolicyReason.UNSUPPORTED_ACTION)
                    if kind == RequestKind.UNSUPPORTED
                    else _NO_ACTION
                )
                result = _session_result(TurnStatus.REJECTED, reply)
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

        # A new explicit CREATE/LIST/COMPLETE replaces any pending request.
        self.state = SessionState.IDLE
        result = run_turn(
            self.database_path,
            prompt,
            generate,
            prompt_version=self.prompt_version,
            settings=self.settings,
        )
        if result["status"] == TurnStatus.NEEDS_CLARIFICATION.value:
            reasons = {
                call.get("reason") for call in result.get("rejected_calls", [])
            }
            if reasons & {"missing_task_id", "multiple_task_ids"}:
                self.state = SessionState.AWAITING_COMPLETE_ID
            else:
                self.state = SessionState.AWAITING_CREATE_CONTENT
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
        self.state = SessionState.IDLE
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
        self.state = SessionState.IDLE
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
