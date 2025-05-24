from typing import List, Dict, Any, Optional, Callable
from app.application.ports import ToolPort, ToolProviderPort
from app.infrastructure.tools_module import init_tools
import logging

logger = logging.getLogger(__name__)

class LangchainToolAdapter(ToolPort):
    """
    Adapter to wrap Langchain tools in the ToolPort interface
    """
    def __init__(self, langchain_tool):
        self.langchain_tool = langchain_tool
        
    @property
    def name(self) -> str:
        return self.langchain_tool.name
    
    @property
    def description(self) -> str:
        return self.langchain_tool.description
    
    def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute the Langchain tool with the given parameters"""
        return self.langchain_tool(**kwargs)

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
            self.register_tool(LangchainToolAdapter(tool))
        logger.info(f"Initialized {len(self._tools)} tools from Langchain")
    
    def get_tools(self) -> List[ToolPort]:
        """Get all available tools"""
        return list(self._tools.values())
    
    def get_tool_by_name(self, name: str) -> Optional[ToolPort]:
        """Get a specific tool by name"""
        return self._tools.get(name)
    
    def register_tool(self, tool: ToolPort) -> None:
        """Register a new tool"""
        self._tools[tool.name] = tool 