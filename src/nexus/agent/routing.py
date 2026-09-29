"""Select the tool schemas exposed to the model for one request."""

from dataclasses import dataclass
from enum import Enum

from nexus.agent.policy import RequestKind, classify_request
from nexus.agent.tools import TOOL_DEFINITIONS


class ToolRoutingMode(str, Enum):
    ALL = "all"
    CLASSIFIED = "classified"


DEFAULT_TOOL_ROUTING = ToolRoutingMode.ALL

_ALL_TOOL_NAMES = tuple(tool["name"] for tool in TOOL_DEFINITIONS)
_TOOL_BY_NAME = {tool["name"]: tool for tool in TOOL_DEFINITIONS}
_KIND_TO_TOOL = {
    RequestKind.CREATE: "create_task",
    RequestKind.MISSING_CREATE: "create_task",
    RequestKind.LIST: "list_tasks",
    RequestKind.COMPLETE: "complete_task",
    RequestKind.MISSING_COMPLETE: "complete_task",
    RequestKind.MULTIPLE_COMPLETE: "complete_task",
    RequestKind.EDIT: "update_task",
    RequestKind.MISSING_EDIT_ID: "update_task",
    RequestKind.MISSING_EDIT_CONTENT: "update_task",
    RequestKind.MULTIPLE_EDIT: "update_task",
    RequestKind.DELETE: "delete_task",
    RequestKind.MISSING_DELETE_ID: "delete_task",
    RequestKind.MULTIPLE_DELETE: "delete_task",
    RequestKind.DEADLINE: "set_task_deadline",
    RequestKind.MISSING_DEADLINE_ID: "set_task_deadline",
    RequestKind.MISSING_DEADLINE_TIME: "set_task_deadline",
    RequestKind.MULTIPLE_DEADLINE: "set_task_deadline",
    RequestKind.DEADLINE_QUERY: "list_tasks_by_deadline",
    RequestKind.MISSING_DEADLINE_SCOPE: "list_tasks_by_deadline",
}


@dataclass(frozen=True)
class ToolRoute:
    mode: ToolRoutingMode
    request_kind: RequestKind
    tools: tuple[str, ...]
    fallback: bool

    def trace(self) -> dict:
        return {
            "mode": self.mode.value,
            "request_kind": self.request_kind.value,
            "tools": list(self.tools),
            "fallback": self.fallback,
        }

    def definitions(self) -> list[dict]:
        return [_TOOL_BY_NAME[name] for name in self.tools]


def select_tool_route(
    prompt: str,
    mode: ToolRoutingMode = DEFAULT_TOOL_ROUTING,
) -> ToolRoute:
    """Classify one request and return only the schemas the model may see."""
    if not isinstance(mode, ToolRoutingMode):
        raise TypeError("mode must be a ToolRoutingMode")
    request_kind = classify_request(prompt)
    if mode == ToolRoutingMode.ALL:
        return ToolRoute(mode, request_kind, _ALL_TOOL_NAMES, False)
    tool_name = _KIND_TO_TOOL.get(request_kind)
    if tool_name is None:
        return ToolRoute(mode, request_kind, _ALL_TOOL_NAMES, True)
    return ToolRoute(mode, request_kind, (tool_name,), False)


def all_tool_names() -> tuple[str, ...]:
    return _ALL_TOOL_NAMES
