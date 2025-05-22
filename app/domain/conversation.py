from typing import List, Any, Dict
from enum import Enum

class Message:
    # Basic message structure, can be expanded
    def __init__(self, content: str, sender: str, metadata: Dict[str, Any] = None):
        self.content = content
        self.sender = sender
        self.metadata = metadata or {}

class ConversationState(Enum):
    STARTED = "started"
    PROCESSING = "processing"
    WAITING_FOR_TOOL = "waiting_for_tool"
    TOOL_EXECUTED = "tool_executed"
    COMPLETED = "completed"
    ERROR = "error"

class Conversation:
    """
    Represents a conversation, acting as an Aggregate Root.
    It encapsulates messages and the state machine of the conversation.
    """
    def __init__(self, conversation_id: str, initial_message: Message):
        self.conversation_id = conversation_id
        self.messages: List[Message] = [initial_message]
        self.state: ConversationState = ConversationState.STARTED
        self.context: Dict[str, Any] = {} # For storing intermediate data, tool outputs, etc.

    def add_message(self, message: Message):
        self.messages.append(message)

    def set_state(self, state: ConversationState):
        self.state = state

    def update_context(self, key: str, value: Any):
        self.context[key] = value

    # More methods to manage conversation flow, e.g.:
    # - select_next_step()
    # - apply_tool_result(tool_name: str, result: Any)
    # - get_full_history() -> List[Dict] 