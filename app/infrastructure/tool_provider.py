from typing import List, Dict, Any, Optional, Callable
from app.infrastructure.tools_module import init_tools
from langchain_core.tools import BaseTool
import logging
logger = logging.getLogger(__name__)

class ToolProvider:
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
    
    def get_tools(self) -> List[BaseTool]:
        """Get all available tools"""
        return list(self._tools.values())
    
    def get_tool_by_name(self, name: str) -> Optional[BaseTool]:
        """Get a specific tool by name"""
        return self._tools.get(name)
    
    def register_tool(self, tool: BaseTool) -> None:
        """Register a new tool"""
        self._tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}, Description: {tool.description}")
        