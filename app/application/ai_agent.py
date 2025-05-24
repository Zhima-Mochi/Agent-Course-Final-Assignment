from __future__ import annotations
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

# from langgraph.graph.message import add_messages # This import seems unused
from app.domain.value_objects import (
    AgentState,
)  # Still depends on this domain value object
from app.domain.prompt_strategy import (
    PromptStrategy,
    BasicPromptStrategy,
)  # Import PromptStrategy

logger = logging.getLogger(__name__)

# SYSTEM_TEMPLATE will now come from PromptStrategy
# MAX_TURNS = 5


@dataclass
class AIAgent:
    """LangGraph agent with a *planner* stage and explicit continue logic."""

    name: str
    # Still directly ChatOpenAI, not LLMServicePort. Handled by adapter for now.
    llm: ChatOpenAI
    tools: List[Any]  # List of tools that can be used by the agent
    prompt_strategy: PromptStrategy  # Added prompt_strategy
    max_turns: int = 5  # Made max_turns a configurable parameter

    def to_langgraph_agent(self) -> Any:
        """
        Compile the agent's state graph.

        The graph has four stages:
        - `retriever`: initialize the state with the user's question
        - `planner`: break the question into smaller tasks (if needed)
        - `assistant_think`: generate an answer using the tasks or internal reasoning
        - `assistant_finalize`: extract the final answer from the generated text
        """

        llm_w_tools = self.llm.bind_tools(self.tools, parallel_tool_calls=True)

        # Get system message from prompt strategy
        # Assuming generate_prompt can return a simple string for system message if no history/query needed for it
        # or we add a specific method to PromptStrategy for system prompts.
        # For now, let's assume generate_prompt is flexible or we use a dedicated method if defined.
        # BasicPromptStrategy.generate_prompt expects history and query. This is not ideal for system message.
        # Let's add a dedicated method to PromptStrategy for the main system prompt.
        if hasattr(self.prompt_strategy, "get_system_prompt"):
            system_prompt_content = self.prompt_strategy.get_system_prompt()
        elif isinstance(self.prompt_strategy, BasicPromptStrategy):
            # Fallback for BasicPromptStrategy if get_system_prompt is not yet added
            # This is a placeholder, ideally BasicPromptStrategy would also have get_system_prompt
            system_prompt_content = (
                "You are a helpful AI assistant. Please respond to the user's request."
            )
            logger.warning(
                "PromptStrategy does not have get_system_prompt, using default for AIAgent system message."
            )
        else:
            # Generic fallback or raise error
            system_prompt_content = "You are an AI assistant."
            logger.warning(
                "PromptStrategy lacks get_system_prompt and is not BasicPromptStrategy, using generic system message."
            )

        sys_msg = SystemMessage(content=system_prompt_content)

        # ---------- retriever ---------- #
        def retriever(state: AgentState) -> AgentState:
            """Initialize the state with the user's question."""
            state.setdefault("messages", [])
            # Ensure sys_msg is always first, even if messages already exist (e.g. from a previous turn in a longer conversation)
            # However, AgentState messages are typically re-initialized per invoke for this agent.
            if not state["messages"] or state["messages"][0].content != sys_msg.content:
                state["messages"].insert(0, sys_msg)
            state["turn"] = 0
            state["next"] = None  # Ensure 'next' is initialized

            return state

        # ---------- planner ---------- #
        def planner(state: AgentState) -> AgentState:
            """Break the question into smaller tasks (if needed)."""

            state["next"] = "assistant_think"

            # Example of using PromptStrategy for planner - IF PromptStrategy supports it
            # For now, keeping original planner prompt. This would be a further refactoring.
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
                plan_response = self.llm.invoke(planner_prompt_messages)
                raw_plan = plan_response.content
                # Accept '-', '•', or numbered lists
                lines = [
                    line.strip()
                    for line in re.split(r"[\n]+", raw_plan)
                    if line.strip()
                ]
                tasks = ["The tasks are: "]
                for line in lines:
                    # Remove common prefixes
                    task = re.sub(r"^[-•\d\.)\s]+", "", line).strip()
                    if task:
                        tasks.append(task)
                tasks.append("Answer the question.")

                # Append planning result as a new HumanMessage to guide the assistant_think stage
                # This assumes messages are additive.
                current_messages = state.get("messages", [])
                current_messages.append(HumanMessage(content="\n".join(tasks)))
                state["messages"] = current_messages

                logger.info(f"[{self.name}] Planner tasks: {tasks}")
            except Exception as e:
                logger.exception(f"[{self.name}] Error during planning, {e}")
            return state

        # ---------- assistant_think ---------- #
        def assistant_think(state: AgentState) -> AgentState:
            """Generate an answer using the tasks or internal reasoning (CoT enabled if set)."""
            # logger.info(f"Previous last  message: {state.get('messages', ["No messages"])[-1]}")
            logger.info(f"[{self.name}] Assistant thinking...")

            if state.get("turn", 0) >= self.max_turns:  # Use self.max_turns
                logger.warning(
                    f"[{self.name}] Max turns reached ({state.get('turn', 0)}); finalizing"
                )
                state["next"] = "assistant_finalize"
                return state

            # Get current messages from state
            messages = state.get("messages", [])

            if state.get("turn", 0) == 0:
                messages.append(
                    HumanMessage(
                        content="To solve this correctly, reason through the steps carefully before answering. Let's break it down step by step."
                    )
                )

            try:
                # Ensure messages are passed correctly. messages should be List[BaseMessage]
                result = llm_w_tools.invoke(messages)
                logger.info(
                    f"[{self.name}] Assistant response: {result.content or 'use tool calls'}")
            except Exception as e:
                logger.exception(f"[{self.name}] LLM invocation failed: {e}")
                messages.append(
                    AIMessage(
                        content=f"Error: failed to invoke LLM with tools, {e}")
                )
                state["messages"] = messages
                state["next"] = "assistant_finalize"
                return state

            # Update message history with response
            messages.append(
                AIMessage(
                    content=result.content,
                    additional_kwargs=result.additional_kwargs,
                    tool_calls=result.tool_calls or [],
                )
            )
            state["messages"] = messages
            state["turn"] = (
                state.get("turn", 0) + 1
            )  # use .get for safety and increment

            # Routing logic with additional safeguards
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

            refine_prompt_messages = [  # Changed to list of messages
                SystemMessage(
                    content="You are an answer extraction assistant."),
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
                refined_response = self.llm.invoke(
                    refine_prompt_messages
                )  # invoke with list of messages
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
            logger.info(f"[{self.name}] Final answer: {final}")
            return state

        # ------------------------------------------------------------------
        # build graph
        # ------------------------------------------------------------------
        g = StateGraph(
            AgentState
        )  # AgentState should be a TypedDict or Pydantic model for LangGraph
        g.add_node("retriever", retriever)
        g.add_node("planner", planner)
        g.add_node("assistant_think", assistant_think)
        g.add_node("tools_node", ToolNode(self.tools))
        g.add_node("assistant_finalize", assistant_finalize)

        g.add_edge(START, "retriever")
        g.add_edge("retriever", "planner")
        g.add_conditional_edges(
            "planner",
            lambda s: s.get("next", "assistant_think"),  # Use .get for safety
            {
                "assistant_think": "assistant_think",
                # Should planner directly go to finalize? Unlikely.
                "assistant_finalize": "assistant_finalize",
            },
        )
        g.add_conditional_edges(
            "assistant_think",
            # 'next' should always be set by assistant_think
            lambda s: s["next"],
            {
                "tools_node": "tools_node",
                "assistant_think": "assistant_think",
                "assistant_finalize": "assistant_finalize",
            },
        )
        g.add_edge("tools_node", "assistant_think")
        g.add_edge("assistant_finalize", END)

        logger.info(
            f"[{self.name}] Graph compiled; MAX_TURNS={self.max_turns}")

        graph = g.compile()
        # For debugging graph structure:
        # try:
        #     graph.get_graph().draw_mermaid_png(output_file_path="graph.png")
        #     logger.info("Agent graph diagram saved to graph.png")
        # except Exception as e:
        #     logger.warning(f"Could not draw agent graph: {e}. Ensure graphviz and dependencies are installed.")

        return graph
