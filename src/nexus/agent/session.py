"""In-memory continuation of an explicitly requested but incomplete CREATE."""

import re
from copy import deepcopy
from pathlib import Path

from nexus.agent.client import PostToolExecutionError, TurnStatus, chat, run_turn
from nexus.agent.policy import PolicyReason, RequestKind, classify_request
from nexus.agent.responses import format_rejection, format_tool_result
from nexus.agent.tools import execute_tool


_CANCEL = re.compile(r"^\s*(?:thôi|hủy|huỷ)[.!]?\s*$", re.IGNORECASE)
_NO_ACTION = "Không có thao tác nào được thực hiện."
_ASK_CONTENT = "Bạn muốn thêm việc gì?"


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
    """One conversation bound to one database; not shared across users."""

    def __init__(
        self,
        database_path: str | Path,
        *,
        prompt_version: str = "v1",
        settings: dict | None = None,
    ) -> None:
        self.database_path = Path(database_path)
        self.prompt_version = prompt_version
        self.settings = deepcopy(settings)
        self.pending_create = False

    def run_turn(self, prompt: str, generate=chat) -> dict:
        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string")
        if not prompt.strip():
            if self.pending_create:
                return _session_result(TurnStatus.NEEDS_CLARIFICATION, _ASK_CONTENT)
            return _session_result(TurnStatus.NO_TOOL, _NO_ACTION)

        if self.pending_create and _CANCEL.fullmatch(prompt):
            self.pending_create = False
            return _session_result(
                TurnStatus.CANCELLED,
                "Đã hủy yêu cầu thêm việc. Không có việc nào được lưu.",
            )

        kind = classify_request(prompt)
        if kind == RequestKind.MISSING_CREATE:
            self.pending_create = True
            return _session_result(TurnStatus.NEEDS_CLARIFICATION, _ASK_CONTENT)

        if kind in (RequestKind.UNSUPPORTED, RequestKind.NEGATED):
            had_pending = self.pending_create
            self.pending_create = False
            if kind == RequestKind.NEGATED and had_pending:
                return _session_result(
                    TurnStatus.CANCELLED,
                    "Đã hủy yêu cầu thêm việc. Không có việc nào được lưu.",
                )
            reply = (
                format_rejection(PolicyReason.UNSUPPORTED_ACTION)
                if kind == RequestKind.UNSUPPORTED
                else _NO_ACTION
            )
            return _session_result(TurnStatus.REJECTED, reply)

        if self.pending_create and kind == RequestKind.OTHER:
            return self._save_followup(prompt)

        # A new explicit CREATE/LIST replaces any pending request.
        self.pending_create = False
        result = run_turn(
            self.database_path,
            prompt,
            generate,
            prompt_version=self.prompt_version,
            settings=self.settings,
        )
        if result["status"] == TurnStatus.NEEDS_CLARIFICATION.value:
            self.pending_create = True
        return {**result, "source": "model"}

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
        self.pending_create = False
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
