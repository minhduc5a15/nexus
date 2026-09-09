from dataclasses import dataclass

@dataclass(frozen=True)
class Task:
    id: int
    content: str
