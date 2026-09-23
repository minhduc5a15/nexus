from dataclasses import dataclass
from enum import Enum

@dataclass(frozen=True)
class Task:
    id: int
    content: str
    completed: bool = False


class CompletionStatus(str, Enum):
    COMPLETED = "completed"
    ALREADY_COMPLETED = "already_completed"
    NOT_FOUND = "not_found"


@dataclass(frozen=True)
class CompletionResult:
    status: CompletionStatus
    task: Task | None


class UpdateStatus(str, Enum):
    UPDATED = "updated"
    UNCHANGED = "unchanged"
    NOT_FOUND = "not_found"


@dataclass(frozen=True)
class UpdateResult:
    status: UpdateStatus
    task: Task | None
