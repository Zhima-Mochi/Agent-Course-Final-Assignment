from typing import List, Dict, Any, Callable
from pydantic import BaseModel, Field

class Tool(BaseModel):
    """
    Value Object describing a tool's capabilities, cost, and usage restrictions.
    """
    name: str
    description: str
    # callable_action: Callable # This would be the actual function/method to call
    # cost: float = 0.0 # Example: cost per invocation
    # usage_limits: Dict[str, Any] = Field(default_factory=dict) # Example: rate limits
    # input_schema: Dict[str, Any] = Field(default_factory=dict) # To validate inputs
    # output_schema: Dict[str, Any] = Field(default_factory=dict) # To validate outputs

    # In a more complex system, this might include methods to:
    # - validate_input(input_data: Dict) -> bool
    # - estimate_cost(input_data: Dict) -> float

class ToolSet:
    """
    Represents a collection of available tools.
    """
    def __init__(self, tools: List[Tool]):
        self.tools_by_name: Dict[str, Tool] = {tool.name: tool for tool in tools}

    def get_tool(self, name: str) -> Tool | None:
        return self.tools_by_name.get(name)

    def list_tools(self) -> List[Tool]:
        return list(self.tools_by_name.values())

    def get_tool_names(self) -> List[str]:
        return list(self.tools_by_name.keys()) 