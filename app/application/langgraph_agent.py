import logging
import re
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode

from app.domain.value_objects import AgentState
from app.domain.prompt_strategy import PromptStrategy, BasicPromptStrategy
from app.application.llm_service import OpenAILLMService
from langfuse.callback import CallbackHandler
from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class LangGraphAgent:
    """Direct implementation of an agent using LangGraph in the application layer."""

    name: str
    llm_service: OpenAILLMService
    tools: List[Callable]
    max_turns: int = 10
    enable_tracing: bool = False

    def __post_init__(self):
        self.prompt_strategy = BasicPromptStrategy()
        
        if not self.tools:
            logger.warning(f"[{self.name}] Initialized with no tools.")

    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process the current state and return an updated state"""
        try:
            llm = self.llm_service.get_llm()
            if llm is None:
                raise ValueError("LLM from llm_service.get_llm() is None")

            graph = self._build_langgraph_agent(llm)
            
            callbacks = []
            if self.enable_tracing:
                handler = CallbackHandler(
                    public_key=settings.LANGFUSE_PUBLIC_KEY,
                    secret_key=settings.LANGFUSE_SECRET_KEY,
                    host=settings.LANGFUSE_HOST,
                )
                callbacks.append(handler)
                
            if callbacks:
                graph = graph.with_config({"callbacks": callbacks})
                
            return graph.invoke(state)
        except Exception as e:
            logger.exception(f"[{self.name}] Failed to build LangGraph agent")
            raise

    def _build_langgraph_agent(self, llm: ChatOpenAI) -> CompiledStateGraph:
        """Build a LangGraph agent using the provided LLM."""
        llm_w_tools = llm.bind_tools(self.tools, parallel_tool_calls=True)
        
        # Get system message from prompt strategy
        if hasattr(self.prompt_strategy, "get_system_prompt"):
            system_prompt_content = self.prompt_strategy.get_system_prompt()
        elif isinstance(self.prompt_strategy, BasicPromptStrategy):
            system_prompt_content = (
                "You are a helpful AI assistant. Please respond to the user's request."
            )
            logger.warning(f"[{self.name}] PromptStrategy does not have get_system_prompt, using default")
        else:
            system_prompt_content = "You are an AI assistant."
            logger.warning(
                f"[{self.name}] PromptStrategy lacks get_system_prompt and is not BasicPromptStrategy, using generic"
            )

        sys_msg = SystemMessage(content=system_prompt_content)

        # ---------- retriever ---------- #
        def retriever(state: AgentState) -> AgentState:
            """Initialize the state with the user's question."""
            state.setdefault("messages", [])
            if not state["messages"] or state["messages"][0].content != sys_msg.content:
                state["messages"].insert(0, sys_msg)
            state["turn"] = 0
            state["next"] = None
            return state

        # ---------- planner ---------- #
        def planner(state: AgentState) -> AgentState:
            """Break the question into smaller tasks (if needed)."""
            state["next"] = "assistant_think"

            planner_prompt_messages = [
                SystemMessage(content="You are a task planner."),
                HumanMessage(
                    content=(
                        "Break the user question into the *fewest* concrete steps. "
                        "Return them as a bullet list (each line starting with '-'). "
                        "If no breakdown is required, return a single bullet. "
                        "Do NOT solve the problem now.\n\n"
                        f"User question: {state['question']}"
                    )
                ),
            ]
            try:
                plan_response = llm.invoke(planner_prompt_messages)
                raw_plan = plan_response.content
                lines = [
                    line.strip()
                    for line in re.split(r"[\n]+", raw_plan)
                    if line.strip()
                ]
                tasks = ["The tasks are: "]
                for line in lines:
                    task = re.sub(r"^[-•\d\.)\s]+", "", line).strip()
                    if task:
                        tasks.append(task)
                tasks.append("Answer the question.")

                current_messages = state.get("messages", [])
                current_messages.append(HumanMessage(content="\n".join(tasks)))
                state["messages"] = current_messages
                logger.info(f"[{self.name}] Planner tasks: {tasks}")
            except Exception as e:
                logger.exception(f"[{self.name}] Error during planning: {e}")
            return state

        # ---------- assistant_think ---------- #
        def assistant_think(state: AgentState) -> AgentState:
            """Generate an answer using the tasks or internal reasoning."""
            logger.info(f"[{self.name}] Assistant thinking...")

            if state.get("turn", 0) >= self.max_turns:
                logger.warning(
                    f"[{self.name}] Max turns reached ({state.get('turn', 0)}); finalizing"
                )
                state["next"] = "assistant_finalize"
                return state

            messages = state.get("messages", [])

            if state.get("turn", 0) == 0:
                messages.append(
                    HumanMessage(
                        content="To solve this correctly, reason through the steps carefully before answering. Let's break it down step by step."
                    )
                )

            try:
                result = llm_w_tools.invoke(messages)
                logger.info(f"[{self.name}] Assistant response: {result.content or 'use tool calls'}")
            except Exception as e:
                logger.exception(f"[{self.name}] LLM invocation failed: {e}")
                messages.append(
                    AIMessage(
                        content=f"Error: failed to invoke LLM with tools, {e}")
                )
                state["messages"] = messages
                state["next"] = "assistant_finalize"
                return state

            messages.append(
                AIMessage(
                    content=result.content,
                    additional_kwargs=result.additional_kwargs,
                    tool_calls=result.tool_calls or [],
                )
            )
            state["messages"] = messages
            state["turn"] = state.get("turn", 0) + 1

            if result.tool_calls:
                state["next"] = "tools_node"
            elif "#CONTINUE" in result.content.upper():
                state["next"] = "assistant_think"
            else:
                state["next"] = "assistant_finalize"
            return state

        # ---------- assistant_finalize ---------- #
        def assistant_finalize(state: AgentState) -> AgentState:
            """Extract the final answer from the generated text."""
            raw = (
                state["messages"][-1].content.strip()
                if state["messages"]
                else "❌ No answer generated."
            )
            clean = re.sub(r"```.*?```", "", raw, flags=re.DOTALL)
            final = None

            refine_prompt_messages = [
                SystemMessage(content="You are an answer extraction assistant."),
                HumanMessage(
                    content=(
                        f"Question: {state.get('question', '')}\n"
                        "Extract ONLY the final answer from the text below. No explanations or conversational pleasantries.\n---\n"
                        "If the answer is for numeric value, extract it as a number. If it's a list, extract it as a list. If it's a string, extract it as a string.\n"
                        f"{clean}\n---"
                    )
                ),
            ]
            try:
                refined_response = llm.invoke(refine_prompt_messages)
                final = refined_response.content.splitlines()[0].strip(" '\"")
                final = final.rstrip(".")
            except Exception:
                logger.exception(f"[{self.name}] Error refining answer")
                final = clean[:500]  # Ensure final is not None

            current_messages = state.get("messages", [])
            current_messages.append(
                AIMessage(
                    content=(
                        final
                        if final is not None
                        else "Could not extract final answer."
                    )
                )
            )
            state["messages"] = current_messages
            state["next"] = END
            state["answer"] = final
            logger.info(f"[{self.name}] Final answer: {final}")
            return state

        # Build graph
        g = StateGraph(AgentState)
        g.add_node("retriever", retriever)
        g.add_node("planner", planner)
        g.add_node("assistant_think", assistant_think)
        g.add_node("tools_node", ToolNode(self.tools))
        g.add_node("assistant_finalize", assistant_finalize)

        g.add_edge(START, "retriever")
        g.add_edge("retriever", "planner")
        g.add_conditional_edges(
            "planner",
            lambda s: s.get("next", "assistant_think"),
            {
                "assistant_think": "assistant_think",
                "assistant_finalize": "assistant_finalize",
            },
        )
        g.add_conditional_edges(
            "assistant_think",
            lambda s: s["next"],
            {
                "tools_node": "tools_node",
                "assistant_think": "assistant_think",
                "assistant_finalize": "assistant_finalize",
            },
        )
        g.add_edge("tools_node", "assistant_think")
        g.add_edge("assistant_finalize", END)

        logger.info(f"[{self.name}] Graph compiled; MAX_TURNS={self.max_turns}")
        graph = g.compile()
        return graph 