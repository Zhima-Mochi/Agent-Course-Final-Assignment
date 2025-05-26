from abc import ABC, abstractmethod
from typing import List, Dict, Any
from app.domain.message import Message # Assuming Conversation model is defined

class PromptStrategy(ABC):
    """
    Interface for different prompt engineering strategies.
    This allows for A/B testing or selecting strategies based on context.
    """
    @abstractmethod
    def generate_prompt(self, conversation_history: List[Message], user_query: str, available_tools: List[str] = None, context: Dict[str, Any] = None) -> Any:
        """Generates a prompt based on the conversation and strategy."""
        pass

    @abstractmethod
    def get_system_prompt(self) -> str:
        """Returns the main system prompt/template for the agent."""
        pass

class BasicPromptStrategy(PromptStrategy):
    def generate_prompt(self, conversation_history: List[Message], user_query: str, available_tools: List[str] = None, context: Dict[str, Any] = None) -> Any:
        # This will likely return a list of Langchain-compatible messages
        # For now, let's represent it as a simple string or a list of dicts
        # In a real scenario, this would format history, user_query, and tool descriptions.
        
        prompt_messages = []
        for msg in conversation_history:
            if msg.sender == "user":
                prompt_messages.append({"role": "user", "content": msg.content})
            elif msg.sender == "assistant":
                prompt_messages.append({"role": "assistant", "content": msg.content})
            # Could add tool calls/results here too

        prompt_messages.append({"role": "user", "content": user_query})
        
        if available_tools:
            # Simplified representation of tool usage instructions
            tool_description = "You can use the following tools: " + ", ".join(available_tools)
            # This would be much more sophisticated, perhaps using a dedicated system message
            prompt_messages.append({"role": "system", "content": tool_description})

        return prompt_messages 

    def get_system_prompt(self) -> str:
        return (
            "You are a precise AI assistant.\n"
            "- First, break the question into smaller tasks (if needed).\n"
            "- If the input includes structured data (e.g. tables), always try to parse and reason directly.\n"
            "- Never guess or assume filenames or formats.\n"
            "- Search the web for information if needed.\n"
            "- If internal reasoning is required, output exactly `#CONTINUE`.\n"
            "- Otherwise, output the final answer in the requested format."
        )