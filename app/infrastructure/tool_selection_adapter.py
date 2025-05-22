from typing import List, Optional

from app.application.ports import ToolSelectorPort
from app.domain.task import QuestionTask
from app.domain.tool import Tool # Assuming Tool domain object exists
from app.infrastructure.tool_router import route_tool as select_tool_name

class BasicToolSelectorAdapter(ToolSelectorPort):
    """Adapter for selecting a tool based on predefined routing logic."""

    def select_tool(self, task: QuestionTask, available_tools: List[Tool]) -> Optional[Tool]:
        """Selects a tool based on the task and available tools.
        
        Args:
            task: The QuestionTask to process.
            available_tools: A list of available Tool domain objects.
            
        Returns:
            The selected Tool object or None if no specific tool is chosen or found.
        """
        # The original route_tool function returns a tool name (string).
        tool_name = select_tool_name(task)

        # Find the Tool object from the available_tools list that matches the selected name.
        for tool in available_tools:
            if tool.name == tool_name:
                return tool
        
        # Fallback or if tool_name doesn't match any Tool object
        # This could mean 'llm_tool' implies no specific domain tool, but direct LLM handling.
        # Or, if 'llm_tool' is also a Tool object, it should be found above.
        # Depending on how "llm_tool" is conceptualized (as a specific tool or general LLM fallback)
        # this part might need adjustment.
        # For now, if a specific tool name was returned by route_tool and not found in available_tools list,
        # it implies an issue or that the tool is not formally registered in the ToolSet.
        # If tool_name is 'llm_tool' (the default), we might return None to indicate no specific tool.
        if tool_name == "llm_tool": # Default fallback from route_tool
            return None 
            
        # Log a warning if a specific tool was routed but not found in the ToolSet
        import logging
        logging.warning(f"Tool '{tool_name}' selected by router but not found in available tools.")
        return None 