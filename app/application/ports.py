from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Callable, Protocol, ContextManager

from app.domain.task import QuestionTask  # Assuming QuestionTask is defined here
from app.domain.value_objects import Answer  # Assuming Answer is defined here
from app.domain.conversation import Message  # For LLM interaction
from app.domain.tool import Tool  # For tool interactions

class TaskGatewayPort(ABC):
    @abstractmethod
    def fetch_tasks(self) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def submit_answers(
        self, user_id: str, answers: List[Dict[str, Any]], space_id: Optional[str]
    ) -> Dict[str, Any]:
        pass
