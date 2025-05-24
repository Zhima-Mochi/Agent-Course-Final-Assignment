import os
import pandas as pd
import logging
from typing import Tuple, Dict, Any, List, Optional
from langchain_core.messages import HumanMessage

import gradio as gr
from gradio.oauth import OAuthProfile
from app.domain.task import QuestionTask
from app.domain.value_objects import Answer, AgentState
from app.application.ports import (
    FileServicePort,
    ToolSelectorPort,
    AgentGraphPort,
    AgentInitializationPort,
    TaskGatewayPort
)
from app.infrastructure.env_config import initialize_environment
from app.domain.tool import Tool as DomainTool
from app.infrastructure.tools_module import init_tools as get_langchain_tools

logger = logging.getLogger(__name__)
initialize_environment()

class TaskProcessor:
    """Handles the processing of individual tasks using injected services (ports)."""
    
    def __init__(self, 
                 file_service: FileServicePort,
                 tool_selector: ToolSelectorPort,
                 agent_graph: AgentGraphPort,
                 available_tools_descriptions: Optional[List[DomainTool]] = None):
        self.file_service = file_service
        self.tool_selector = tool_selector
        self.agent_graph = agent_graph
        self.available_tools_descriptions = available_tools_descriptions if available_tools_descriptions else []

    def _process_media(self, task: QuestionTask) -> Optional[str]:
        """Download media file using FileServicePort."""
        if not task.file_name:
            return None
        try:
            file_path = self.file_service.download_task_file(task.task_id, task.file_name)
            if file_path:
                logger.debug(f"[Task {task.task_id}] Media file downloaded to {file_path}")
            else:
                logger.warning(f"[Task {task.task_id}] Media file download failed or no path returned.")
            return file_path
        except Exception as e:
            logger.exception(f"[Task {task.task_id}] ❌ media download error via FileService: {e}")
            return None

    def _invoke_agent(self, task: QuestionTask, selected_tool_name: Optional[str], file_path: Optional[str]) -> str:
        """Invoke the agent graph to get an answer."""
        try:
            initial_messages = [HumanMessage(content=task.question)]
            if task.file_name and file_path:
                initial_messages = [HumanMessage(content=f"{task.question}\n\n(Context: File '{task.file_name}' is available at path: {file_path})\n(Hint: Tool '{selected_tool_name if selected_tool_name else 'N/A'}' was suggested for this task.)")]

            state = AgentState(
                question=task.question,
                file_name=task.file_name,
                file_path=file_path,
                tool_used=selected_tool_name,
                llm_output=None,
                messages=initial_messages,
            )
            
            state_dict = dict(state)

            logger.debug(f"[Task {task.task_id}] Invoking agent graph with initial state: {state_dict}")
            state_out = self.agent_graph.invoke(state_dict)
            
            if not state_out.get("messages"):
                logger.error(f"[Task {task.task_id}] Agent output does not contain 'messages'. State: {state_out}")
                return "ERROR: Agent did not produce messages."

            answer_text = state_out["messages"][-1].content.strip()
            logger.info(f"[Task {task.task_id}] ✅ Agent answer generated: {answer_text[:100]}...")
            return answer_text
        except Exception as e:
            error_msg = f"ERROR_AGENT_INVOKE: {e}"
            logger.exception(f"[Task {task.task_id}] ❌ agent-invoke error: {e}")
            return error_msg

    def process_task(self, task: QuestionTask) -> Dict[str, Any]:
        """Process a single task through the new pipeline using ports."""
        logger.info(f"[Task {task.task_id}] ▶ {task.question}")
        
        # Media download (if applicable) via FileServicePort
        file_path = self._process_media(task)
        
        # Tool routing/selection via ToolSelectorPort
        selected_tool_domain_object: Optional[Any] = None
        selected_tool_name_hint: Optional[str] = "llm_tool"
        try:
            _selected_domain_tool = self.tool_selector.select_tool(task, self.available_tools_descriptions)
            if _selected_domain_tool:
                selected_tool_name_hint = _selected_domain_tool.name
            else:
                selected_tool_name_hint = "llm_tool"
            
            logger.info(f"[Task {task.task_id}] 🔧 Tool suggested by selector: {selected_tool_name_hint}")

        except Exception as e:
            logger.exception(f"[Task {task.task_id}] ❌ tool-routing error via ToolSelector: {e}")
            selected_tool_name_hint = "llm_tool"
        
        # Agent invocation via AgentGraphPort
        answer_text = self._invoke_agent(task, selected_tool_name_hint, file_path)
        
        return {
            "answer": Answer(task_id=task.task_id, content=answer_text),
            "result": {
                "Task ID": task.task_id,
                "Question": task.question,
                "Submitted Answer": answer_text,
                "Source": "Agent"
            }
        }

class Orchestrator:
    """Orchestrates the process using injected services (ports and adapters)."""
    
    def __init__(self,
                 task_gateway: TaskGatewayPort,
                 agent_initializer_port: AgentInitializationPort,
                 file_service: FileServicePort,
                 tool_selector: ToolSelectorPort):
        self.task_gateway = task_gateway
        self.file_service = file_service
        self.tool_selector = tool_selector

        # Initialize Langchain tools and corresponding domain tool descriptions
        self.langchain_tools: List[Any] = get_langchain_tools()
        self.domain_tool_descriptions: List[DomainTool] = []
        for lc_tool in self.langchain_tools:
            if hasattr(lc_tool, 'name') and hasattr(lc_tool, 'description'):
                self.domain_tool_descriptions.append(
                    DomainTool(name=lc_tool.name, description=lc_tool.description)
                )
            else:
                # Handle cases where tools might not have name/description as expected
                logger.warning(f"Tool object {str(lc_tool)} lacks name or description attribute.")

        self.agent_initializer_port = agent_initializer_port

        self.agent_graph_port = self._initialize_agent_once()

    def _initialize_agent_once(self) -> AgentGraphPort:
        """Initialize the AI agent graph using the configured AgentInitializationPort."""
        logger.debug("Initializing agent graph via port...")
        try:
            # The agent_initializer_port should already have been configured with necessary tools
            # (e.g. Langchain tools) and LLM service port during its own instantiation.
            agent_graph_adapter = self.agent_initializer_port.initialize_agent_graph()
            logger.info("Agent graph created successfully via port.")
            return agent_graph_adapter
        except Exception as e:
            logger.exception("Failed to initialize agent graph via port.")
            raise RuntimeError("Agent initialization failed") from e
    
    def _submit_results(self, profile: OAuthProfile, answers: List[Answer], space_id: Optional[str]) -> Tuple[str, Dict]:
        """Submit answers using TaskGatewayPort."""
        if not answers:
            logger.warning("No answers to submit")
            return "No answers were generated to submit.", {
                "username": profile.username,
                "score": "N/A",
                "correct_count": 0,
                "total_attempted": 0,
                "message": "No answers generated."
            }

        answers_payload = [ans.to_submission_format() for ans in answers]
        logger.info(f"Submitting {len(answers)} answers for user {profile.username}")
        
        try:
            submission_response = self.task_gateway.submit_answers(profile.username, answers_payload, space_id)
            return "Submission successful", submission_response
        except Exception as e:
            logger.exception(f"Error submitting answers via TaskGatewayPort for user {profile.username}: {e}")
            return f"Submission failed: {str(e)}", {
                "error": str(e)
            }

    def run_all_tasks(self, profile: OAuthProfile, space_id: Optional[str]) -> Tuple[str, Optional[pd.DataFrame]]:
        """Run all tasks and submit answers using injected services."""
        if profile and profile.username:
            logger.info(f"Starting task execution for user: {profile.username}")
        else:
            profile = OAuthProfile(username="empty_profile")
            logger.warning("User not logged in or username missing in profile, using test_user")

        try:
            if not self.agent_graph_port:
                logger.error("Agent graph not available.")
                return "Error: Agent graph not initialized.", None

            questions_data = self.task_gateway.fetch_tasks()
            logger.info(f"Fetched {len(questions_data)} tasks from gateway")
            
            processor = TaskProcessor(
                file_service=self.file_service,
                tool_selector=self.tool_selector,
                agent_graph=self.agent_graph_port,
                available_tools_descriptions=self.domain_tool_descriptions
            )
            
            answers: List[Answer] = []
            results_log: List[Dict[str, Any]] = []
            
            for item in questions_data:
                task = QuestionTask(
                    task_id=str(item.get("task_id")),
                    question=item.get("question"),
                    file_name=item.get("file_name")
                )
                
                try:
                    result_info = processor.process_task(task)
                    answers.append(result_info["answer"])
                    results_log.append(result_info["result"])
                except Exception as e:
                    logger.error(f"Error processing task {task.task_id}: {str(e)}", exc_info=True)
                    results_log.append({
                        "Task ID": task.task_id,
                        "Question": task.question,
                        "Submitted Answer": f"ERROR_PROCESSING_TASK: {str(e)}",
                        "Source": "OrchestratorError"
                    })
            
            submission_status_msg, result_data = self._submit_results(profile, answers, space_id)
            
            if "error" in result_data or "failed" in submission_status_msg.lower():
                status_display = f"Submission Problem: {submission_status_msg}\nDetails: {result_data.get('message', str(result_data))}"
            elif result_data.get('message') == "No answers generated.":
                status_display = submission_status_msg
            else:
                status_display = (
                    f"Submission Successful!\nUser: {result_data.get('username', profile.username)}\n"
                    f"Score: {result_data.get('score', 'N/A')}% ({result_data.get('correct_count', '?')}/"
                    f"{result_data.get('total_attempted', '?')} correct)\n"
                    f"Message: {result_data.get('message', 'No message received.')}"
                )
            
            logger.info(f"Submission complete for user {profile.username}. Status: {status_display}")
            return status_display, pd.DataFrame(results_log)
            
        except Exception as e:
            logger.exception(f"Unexpected error in run_all_tasks for user {profile.username}: {str(e)}")
            return f"Unexpected error: {e}", None
