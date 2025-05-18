from dataclasses import dataclass
from typing import Callable, Any

@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    function: Callable[..., Any]

@dataclass(frozen=True)
class Answer:
    task_id: str
    content: str
    
    def to_submission_format(self):
        return {
            "task_id": self.task_id,
            "submitted_answer": self.content
        } 