from typing import Any, List, Optional
import os

from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI

from app.application.ports import LLMServicePort
from app.domain.conversation import Message # Domain model for messages
from app.domain.tool import Tool # Domain model for tools
from app.config import settings, AppSettings # Import the centralized settings and AppSettings for type hinting

# Helper to convert domain Message to LangChain BaseMessage
def _to_langchain_message(domain_message: Message) -> BaseMessage:
    # This is a simplified conversion. May need more complex logic
    # based on sender type (user, assistant, system, tool)
    # and how langchain expects roles.
    if domain_message.sender.lower() == "user":
        from langchain_core.messages import HumanMessage
        return HumanMessage(content=domain_message.content)
    elif domain_message.sender.lower() == "assistant":
        from langchain_core.messages import AIMessage
        # Potentially handle tool_calls if part of domain_message.metadata
        return AIMessage(content=domain_message.content)
    elif domain_message.sender.lower() == "system":
        from langchain_core.messages import SystemMessage
        return SystemMessage(content=domain_message.content)
    # Add other types like ToolMessage if necessary
    else:
        # Default or raise error for unknown sender type
        from langchain_core.messages import HumanMessage # Fallback
        return HumanMessage(content=f"{domain_message.sender}: {domain_message.content}")

def _to_langchain_tool(domain_tool: Tool) -> dict:
    # Converts domain Tool to LangChain tool format (simplified)
    # LangChain's bind_tools expects a list of Pydantic models or callables
    # This might need to refer to the actual tool functions or more structured definitions
    # For now, this is a placeholder for how tools might be described to the LLM.
    return {
        "name": domain_tool.name,
        "description": domain_tool.description,
        # "parameters": domain_tool.input_schema # If defined
    }

class OpenAILLMAdapter(LLMServicePort):
    def __init__(self, app_settings: AppSettings): # Accept AppSettings instance
        self.settings = app_settings # Store settings instance
        
        if not self.settings.OPENAI_API_KEY:
            # Pydantic should have already raised an error if OPENAI_API_KEY is missing and not Optional
            raise ValueError("OPENAI_API_KEY not found in settings.")
        
        self._llm = ChatOpenAI(
            model=self.settings.OPENAI_MODEL_NAME,
            max_tokens=self.settings.OPENAI_MAX_TOKENS,
            temperature=self.settings.OPENAI_TEMPERATURE,
            api_key=self.settings.OPENAI_API_KEY,
            request_timeout=self.settings.OPENAI_TIMEOUT
        )

    def invoke_llm(self, messages: List[Message], tool_choice: Optional[str] = None, tools: Optional[List[Tool]] = None) -> Any:
        """Invokes the OpenAI LLM with the given messages and optional tools.

        Args:
            messages: A list of domain Message objects.
            tool_choice: Optional name of a tool to force the LLM to use.
            tools: Optional list of domain Tool objects available for the LLM.

        Returns:
            The raw response from the LangChain LLM invocation.
        """
        langchain_messages = [_to_langchain_message(msg) for msg in messages]
        
        invocation_kwargs = {}
        if tools:
            # Langchain's .bind_tools expects the tools themselves (callables or Pydantic models)
            # not just descriptions. This part is tricky and depends on how tools are registered
            # with the agent/LangGraph. For direct LLM call, this would typically be tool schemas.
            # The `AIAgent` uses `llm.bind_tools(self.tools, ...)` where self.tools are actual tool functions.
            # For a generic LLMServicePort, we might only pass descriptions or schemas.
            # This adapter might need access to the actual tool callables if it were to bind them here.
            # For now, assuming `tools` are descriptions for the LLM to know about.
            # This part may need significant refinement based on usage context (direct call vs. agent).
            # For a simple non-agentic call with tool descriptions:
            # langchain_tools_description = [{"type": "function", "function": _to_langchain_tool(t)} for t in tools]
            # invocation_kwargs["tools"] = langchain_tools_description
            # if tool_choice:
            # invocation_kwargs["tool_choice"] = {"type": "function", "function": {"name": tool_choice}}
            pass # Deferring complex tool binding to agent framework for now
        
        # If this adapter is used by an agent that already bound tools, we just pass messages.
        # If it's a direct call and tools are to be considered, logic above needs to be robust.
        
        response = self._llm.invoke(langchain_messages, **invocation_kwargs)
        return response # This is a LangChain AIMessage object 

    def get_llm(self) -> ChatOpenAI:
        return self._llm 