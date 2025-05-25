from typing import List, Dict, Any, Optional, Callable
from app.application.ports import ToolProviderPort
from app.infrastructure.tools_module import init_tools
import logging

logger = logging.getLogger(__name__)

class LangchainToolProvider(ToolProviderPort):
    """
    Concrete implementation of ToolProviderPort using Langchain tools
    """
    def __init__(self):
        self._tools = {}
        self._initialize_tools()
    
    def _initialize_tools(self):
        """Initialize tools from the Langchain tools module"""
        langchain_tools = init_tools()
        for tool in langchain_tools:
            self.register_tool(tool)
        logger.info(f"Initialized {len(self._tools)} tools from Langchain")
    
    def get_tools(self) -> List[Callable]:
        """Get all available tools"""
        return list(self._tools.values())
    
    def get_tool_by_name(self, name: str) -> Optional[Callable]:
        """Get a specific tool by name"""
        return self._tools.get(name)
    
    def register_tool(self, tool: Callable) -> None:
        """Register a new tool"""
        self._tools[tool.name] = tool 