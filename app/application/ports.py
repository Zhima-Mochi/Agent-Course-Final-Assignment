from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Callable, Protocol
from contextlib import ContextDecorator

from app.domain.task import QuestionTask # Assuming QuestionTask is defined here
from app.domain.value_objects import Answer # Assuming Answer is defined here
from app.domain.conversation import Message # For LLM interaction
from app.domain.tool import Tool # For tool interactions

class ToolProviderPort(ABC):
    """Interface for tool provider services"""
    
    @abstractmethod
    def get_tools(self) -> List[Callable]:
        """Get all available tools"""
        pass
    
    @abstractmethod
    def get_tool_by_name(self, name: str) -> Optional[Callable]:
        """Get a specific tool by name"""
        pass
    
    @abstractmethod
    def register_tool(self, tool: Callable) -> None:
        """Register a new tool"""
        pass

class LLMServicePort(Protocol):
    @abstractmethod
    def invoke_llm(self, messages: List[Message], tool_choice: Optional[str] = None, tools: Optional[List[Tool]] = None) -> Any:
        pass

    @abstractmethod
    def get_llm(self) -> Any:
        """Returns the underlying concrete LLM instance (e.g., Langchain ChatModel)."""
        pass

class ToolSelectorPort(ABC):
    @abstractmethod
    def select_tool(self, task: QuestionTask, available_tools: List[Tool]) -> Optional[Tool]:
        pass

class ToolRunnerPort(ABC):
    @abstractmethod
    def run_tool(self, tool_name: str, task: QuestionTask, file_path: Optional[str], current_state: Optional[Dict[str, Any]]) -> Any:
        """Runs a specific tool with given parameters and state."""
        pass 
class FileServicePort(ABC):
    @abstractmethod
    def download_task_file(self, task_id: str, file_name: Optional[str]) -> Optional[str]: # Returns path to file
        pass

class TaskGatewayPort(ABC):
    @abstractmethod
    def fetch_tasks(self) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def submit_answers(self, user_id: str, answers: List[Dict[str, Any]], space_id: Optional[str]) -> Dict[str, Any]:
        pass

class AgentGraphPort(ABC):
    @abstractmethod
    def invoke(self, state: Dict[str, Any]) -> Dict[str, Any]:
        pass

    @abstractmethod
    def get_graph(self) -> Any: # Returns the compiled LangGraph agent/graph
        pass

class AgentInitializationPort(ABC):
    @abstractmethod
    def initialize_agent_graph(self) -> AgentGraphPort:
        pass 

class TracerPort(ABC):
    """Simplified interface for tracing and observability."""

    @abstractmethod
    def trace(self, name: str, metadata: Optional[Dict[str, Any]] = None) -> ContextDecorator:
        """Start a trace/span. Use as a context manager:  
        with tracer.trace('op', {...}): ..."""
        pass

    @abstractmethod
    def span(self, name: str, metadata: Optional[Dict[str, Any]] = None) -> ContextDecorator:
        """Attach metadata or error info to the current span/trace."""
        pass