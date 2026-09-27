from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True)
class Task:
    id: int
    content: str
    completed: bool = False
    due_at: int | None = None


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


class DeleteStatus(str, Enum):
    DELETED = "deleted"
    NOT_FOUND = "not_found"
    STALE = "stale"


@dataclass(frozen=True)
class DeleteResult:
    status: DeleteStatus
    task: Task | None


class DeadlineStatus(str, Enum):
    SET = "set"
    UPDATED = "updated"
    UNCHANGED = "unchanged"
    NOT_FOUND = "not_found"


@dataclass(frozen=True)
class DeadlineResult:
    status: DeadlineStatus
    task: Task | None
