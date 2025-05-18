from __future__ import annotations
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from app.domain.value_objects import AgentState

logger = logging.getLogger(__name__)

SYSTEM_TEMPLATE = (
    "You are a precise AI assistant.\n"
    "- First, break the question into smaller tasks (if needed).\n"
    "- Use available tools when helpful.\n"
    "- If you still need to reason internally, output exactly `#CONTINUE`.\n"
    "- Otherwise, output the final answer in the format requested."
)
MAX_TURNS = 5  # safeguard against infinite loops


@dataclass
class AIAgent:
    """LangGraph agent with a *planner* stage and explicit continue logic."""

    name: str
    llm: ChatOpenAI
    tools: List[Any]  # List of tools that can be used by the agent

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
        sys_msg = SystemMessage(content=SYSTEM_TEMPLATE)

        # ---------- retriever ---------- #
        def retriever(state: AgentState) -> AgentState:
            """Initialize the state with the user's question."""
            state.setdefault("messages", [])
            state["messages"].insert(0, sys_msg)
            state["turn"] = 0
            state["next"] = None

            return state

        # ---------- planner ---------- #
        def planner(state: AgentState) -> AgentState:
            """Break the question into smaller tasks (if needed)."""

            state["next"] = "assistant_think"
            # Enhanced planning prompt with bullet requirement
            prompt = [
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
                plan_response = self.llm.invoke(prompt)
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

                state["messages"].append(HumanMessage(content="\n".join(tasks)))
                logger.info(f"[{self.name}] Planner tasks: {tasks}")
            except Exception as e:
                logger.exception(f"[{self.name}] Error during planning, {e}")
            return state

        # ---------- assistant_think ---------- #
        def assistant_think(state: AgentState) -> AgentState:
            """Generate an answer using the tasks or internal reasoning."""
            logger.info(f"[{self.name}] Assistant think: {state['messages'][-1]}")
            if state["turn"] >= MAX_TURNS:
                logger.warning(
                    f"[{self.name}] Max turns reached ({state['turn']}); finalizing"
                )
                state["next"] = "assistant_finalize"
                return state

            try:
                result = llm_w_tools.invoke(state["messages"])
                logger.info(f"[{self.name}] Assistant think result: content = {result.content}, tool_calls = {result.tool_calls}")
            except Exception as e:
                logger.exception(f"[{self.name}] Error invoking LLM with tools, {e}")
                state["messages"].append(
                    AIMessage(content=f"Error: failed to invoke LLM with tools, {e}")
                )
                state["next"] = "assistant_finalize"
                return state
            state["messages"].append(
                AIMessage(
                    content=result.content, additional_kwargs=result.additional_kwargs
                )
            )
            state["turn"] += 1
            calls = result.additional_kwargs.get("tool_calls")
            if calls:
                next_hop = "tools_node"
            elif "#CONTINUE" in result.content.upper():
                next_hop = "assistant_think"
            else:
                next_hop = "assistant_finalize"
            state["next"] = next_hop
            return state

        # ---------- assistant_finalize ---------- #
        def assistant_finalize(state: AgentState) -> AgentState:
            """Extract the final answer from the generated text."""
            raw = state["messages"][-1].content.strip() or "❌ No answer generated."
            clean = re.sub(r"```.*?```", "", raw, flags=re.DOTALL)
            clean = re.sub(r"#CONTINUE", "", clean, flags=re.IGNORECASE).strip()
            final = None
            # Numeric match
            if m := re.fullmatch(r".*?(-?\d+(?:\.\d+)?)\D*$", clean):
                final = m.group(1)
            # Multiple-choice match
            elif m := re.fullmatch(r".*?\b([A-D](?:,[A-D])*)\b.*", clean):
                final = m.group(1)
            else:
                # Fallback extraction
                refine = HumanMessage(
                    content=(
                        f"Question: {state.get('question', '')}\n"
                        "Extract ONLY the final answer from the text below. No explanations.\n---\n"
                        f"{clean}\n---"
                    )
                )
                try:
                    refined = self.llm.invoke([refine]).content
                    final = refined.splitlines()[0].strip(" '\"")
                except Exception:
                    logger.exception(f"[{self.name}] Error refining answer")
                    final = clean[:500]
            state["messages"].append(AIMessage(content=final))
            state["next"] = END
            logger.info(f"[{self.name}] Final answer: {final}")
            return state

        # ------------------------------------------------------------------
        # build graph
        # ------------------------------------------------------------------
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

        logger.info(f"[{self.name}] Graph compiled; MAX_TURNS={MAX_TURNS}")

        graph = g.compile()
        # graph.get_graph().draw_png("graph.png")

        return graph
