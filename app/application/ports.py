from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class TaskGatewayPort(ABC):
    @abstractmethod
    def fetch_tasks(self) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def submit_answers(
        self, user_id: str, answers: List[Dict[str, Any]], space_id: Optional[str]
    ) -> Dict[str, Any]:
        pass
