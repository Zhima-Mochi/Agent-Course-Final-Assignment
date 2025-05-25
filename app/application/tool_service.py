from typing import List, Dict, Any, Optional, Callable
from app.application.ports import ToolProviderPort
from app.domain.tool import Tool

class ToolService:
    """
    Application service for working with tools through the domain interface.
    This follows the dependency inversion principle by depending on the domain interfaces,
    not on concrete infrastructure implementations.
    """
    
    def __init__(self, tool_provider: ToolProviderPort):
        self.tool_provider = tool_provider
        
    def get_all_tools(self) -> List[Tool]:
        """
        Get all available tools as domain Tool objects
        """
        tool_ports = self.tool_provider.get_tools()
        return [
            Tool(name=tool.name, description=tool.description)
            for tool in tool_ports
        ]
    
    def get_tool_by_name(self, name: str) -> Optional[Tool]:
        """
        Get a specific tool by name as a domain Tool object
        """
        tool_port = self.tool_provider.get_tool_by_name(name)
        if tool_port:
            return Tool(name=tool_port.name, description=tool_port.description)
        return None
    
    def execute_tool(self, name: str, **kwargs) -> Dict[str, Any]:
        """
        Execute a tool by name with the given parameters
        """
        tool = self.tool_provider.get_tool_by_name(name)
        if not tool:
            raise ValueError(f"Tool not found: {name}")
        return tool.execute(**kwargs) 