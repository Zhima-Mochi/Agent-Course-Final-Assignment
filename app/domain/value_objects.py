from dataclasses import dataclass
from typing import Callable, Any, List, Optional, Dict
from typing_extensions import TypedDict, Annotated
from langgraph.graph.message import add_messages


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


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    file_path: Optional[str]
    question: str
    answer: Optional[str]
    tool_hint: Optional[str]
    turn: int
    next: Optional[str]
