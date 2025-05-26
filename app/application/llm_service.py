from typing import Any, List, Optional
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI

from app.domain.conversation import Message  # Domain model for messages
from app.domain.tool import Tool  # Domain model for tools
from app.config import settings, AppSettings

# Helper to convert domain Message to LangChain BaseMessage
def _to_langchain_message(domain_message: Message) -> BaseMessage:
    """Convert domain Message to LangChain BaseMessage."""
    if domain_message.sender.lower() == "user":
        from langchain_core.messages import HumanMessage
        return HumanMessage(content=domain_message.content)
    elif domain_message.sender.lower() == "assistant":
        from langchain_core.messages import AIMessage
        return AIMessage(content=domain_message.content)
    elif domain_message.sender.lower() == "system":
        from langchain_core.messages import SystemMessage
        return SystemMessage(content=domain_message.content)
    else:
        # Default fallback
        from langchain_core.messages import HumanMessage
        return HumanMessage(content=f"{domain_message.sender}: {domain_message.content}")

class OpenAILLMService:
    """Direct implementation of LLM service using OpenAI/LangChain in the application layer."""
    
    def __init__(self, app_settings: Optional[AppSettings] = None):
        self.settings = app_settings or settings
        
        if not self.settings.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not found in settings.")
        
        self._llm = ChatOpenAI(
            model=self.settings.OPENAI_MODEL_NAME,
            max_tokens=self.settings.OPENAI_MAX_TOKENS,
            temperature=self.settings.OPENAI_TEMPERATURE,
            api_key=self.settings.OPENAI_API_KEY,
            request_timeout=self.settings.OPENAI_TIMEOUT
        )

    def invoke_llm(self, messages: List[Message], tool_choice: Optional[str] = None, tools: Optional[List[Tool]] = None) -> Any:
        """Invokes the OpenAI LLM with the given messages and optional tools."""
        langchain_messages = [_to_langchain_message(msg) for msg in messages]
        
        # For now, we're not handling tools binding at this level
        # Tool binding is typically done in the agent implementation
        response = self._llm.invoke(langchain_messages)
        return response

    def get_llm(self) -> ChatOpenAI:
        """Get the underlying LangChain LLM instance."""
        return self._llm 