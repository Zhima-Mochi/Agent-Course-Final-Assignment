from typing import List, Any, Dict
from enum import Enum

class Message:
    # Basic message structure, can be expanded
    def __init__(self, content: str, sender: str, metadata: Dict[str, Any] = None):
        self.content = content
        self.sender = sender
        self.metadata = metadata or {}
