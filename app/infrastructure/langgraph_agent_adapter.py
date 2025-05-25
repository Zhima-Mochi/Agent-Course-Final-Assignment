import logging
from typing import Any, Dict, List, Optional, Callable

from langchain_core.tools import BaseTool  # For type hinting

from app.application.ports import AgentInitializationPort, AgentGraphPort, LLMServicePort
from app.application.ai_agent import AIAgent  # Updated import path
# Ensure this is the correct AgentState
from app.domain.value_objects import AgentState
# Import PromptStrategy
from app.domain.prompt_strategy import PromptStrategy, BasicPromptStrategy
# from app.infrastructure.tools_module import init_tools # No longer called here
# from app.infrastructure.llm_service import LLMService # Original direct usage
# from app.infrastructure.openai_llm_adapter import OpenAILLMAdapter # No longer needed for isinstance checks

logger = logging.getLogger(__name__)


class LangGraphAgentGraphAdapter(AgentGraphPort):
    """Adapter that wraps a compiled LangGraph graph."""

    def __init__(self, compiled_graph: Any):
        if compiled_graph is None:
            raise ValueError("Compiled graph cannot be None.")
        self.compiled_graph = compiled_graph

    def invoke(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Invokes the compiled LangGraph with the given state."""
        # Ensure the input state is compatible with what the AIAgent graph expects (AgentState)
        # This might involve transformation if the input `state` dict isn't directly AgentState
        # For now, assuming it is or can be directly used.
        # If state is a dict, it needs to be converted to AgentState or ensure AgentState can be a TypedDict

        # The AIAgent graph in domain.agent.py uses a specific AgentState dataclass.
        # We need to ensure the input `state` dict is compatible or converted.
        # For example, if AgentState is a Pydantic model or dataclass:
        # agent_state_instance = AgentState(**state)
        # result = self.compiled_graph.invoke(agent_state_instance.dict(exclude_none=True))

        # However, the LangGraph itself might operate on dicts that conform to AgentState structure.
        # Let's assume for now that the state dict is directly usable.
        # The AgentState from domain.value_objects seems to be a TypedDict, which is fine.
        return self.compiled_graph.invoke(state)

    def get_graph(self) -> Any:
        return self.compiled_graph


class LangGraphAgentInitializerAdapter(AgentInitializationPort):
    """Adapter for initializing the LangGraph-based AIAgent."""

    def __init__(self,
                 llm_service_port: LLMServicePort,
                 # Accept our Callable objects directly
                 lc_tools: List[Callable],
                 prompt_strategy: Optional[PromptStrategy] = None):
        self.llm_service_port = llm_service_port
        self.tools = lc_tools  # Store as tools, not specifically lc_tools
        self.prompt_strategy = prompt_strategy if prompt_strategy is not None else BasicPromptStrategy()

        if not self.tools:
            logger.warning(
                "LangGraphAgentInitializerAdapter initialized with an empty list of tools.")

    def initialize_agent_graph(self) -> AgentGraphPort:
        try:
            # Grab the concrete LLM via the port
            llm = self.llm_service_port.get_llm()

            # Ensure llm is not None
            if llm is None:
                raise ValueError(
                    "LLM instance from llm_service_port.get_llm() is None.")

            # Extract actual tools for tools that are adapters
            actual_tools = []
            for tool in self.tools:
                # Use the tool directly if it's not an adapter
                actual_tools.append(tool)

            agent = AIAgent(
                name="StructuredMultimodalAgent",
                llm=llm,
                tools=actual_tools,  # Use the extracted tools
                prompt_strategy=self.prompt_strategy,
                max_turns=10
            )
            graph = agent.to_langgraph_agent()
            return LangGraphAgentGraphAdapter(graph)

        except Exception as e:
            logger.exception("Failed to build LangGraph agent")
            raise
