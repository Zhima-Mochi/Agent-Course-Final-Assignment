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
    TaskGatewayPort,
    TracerPort
)
from app.domain.tool import Tool as DomainTool
from app.application.tool_service import ToolService

logger = logging.getLogger(__name__)

class TaskProcessor:
    """Handles the processing of individual tasks using injected services (ports)."""
    
    def __init__(self, 
                 file_service: FileServicePort,
                 tool_selector: ToolSelectorPort,
                 agent_graph: AgentGraphPort,
                 tracer: TracerPort,
                 available_tools_descriptions: Optional[List[DomainTool]] = None):
        self.file_service = file_service
        self.tool_selector = tool_selector
        self.agent_graph = agent_graph
        self.tracer = tracer
        self.available_tools_descriptions = available_tools_descriptions if available_tools_descriptions else []

    def _process_media(self, task: QuestionTask) -> Optional[str]:
        """Download media file using FileServicePort."""
        trace_metadata = {
            "task.id": task.task_id,
            "task.has_file": bool(task.file_name),
            "file.download.success": False # Default to false
        }
        with self.tracer.span(name=f"process_media", metadata=trace_metadata):
            if not task.file_name:
                return None
            try:
                file_path = self.file_service.download_task_file(task.task_id, task.file_name)
                if file_path:
                    logger.debug(f"[Task {task.task_id}] Media file downloaded to {file_path}")
                    trace_metadata["file.download.success"] = True
                    trace_metadata["file.path"] = file_path
                else:
                    logger.warning(f"[Task {task.task_id}] Media file download failed or no path returned.")
                    # trace_metadata["file.download.success"] remains False
                return file_path
            except Exception as e:
                logger.exception(f"[Task {task.task_id}] ❌ media download error via FileService: {e}")
                trace_metadata["error"] = str(e)
                # trace_metadata["file.download.success"] remains False
                return None

    def _invoke_agent(self, task: QuestionTask, selected_tool_name: Optional[str], file_path: Optional[str]) -> str:
        """Invoke the agent graph to get an answer."""
        invoke_metadata = {
            "task.id": task.task_id,
            "task.has_file": bool(task.file_name),
            "task.selected_tool": selected_tool_name or "none",
            "agent.success": False # Default to False
        }
        with self.tracer.span(name=f"invoke_agent_for_task_{task.task_id}", metadata=invoke_metadata):
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
                
                graph_invoke_metadata = {"agent.input_length": len(str(state_dict))}
                with self.tracer.span(name="agent_graph_invoke", metadata=graph_invoke_metadata):
                    state_out = self.agent_graph.invoke(state_dict)
                    graph_invoke_metadata["agent.output_length"] = len(str(state_out))
                
                if not state_out.get("messages"):
                    logger.error(f"[Task {task.task_id}] Agent output does not contain 'messages'. State: {state_out}")
                    invoke_metadata["agent.error"] = "No messages in output"
                    return "ERROR: Agent did not produce messages."

                answer_text = state_out["messages"][-1].content.strip()
                logger.info(f"[Task {task.task_id}] ✅ Agent answer generated: {answer_text[:100]}...")
                invoke_metadata["agent.success"] = True
                invoke_metadata["agent.answer_length"] = len(answer_text)
                return answer_text
            except Exception as e:
                error_msg = f"ERROR_AGENT_INVOKE: {e}"
                logger.exception(f"[Task {task.task_id}] ❌ agent-invoke error: {e}")
                invoke_metadata["agent.error"] = str(e)
                return error_msg

    def process_task(self, task: QuestionTask) -> Dict[str, Any]:
        """Process a single task through the new pipeline using ports."""
        process_task_metadata = {
            "task.id": task.task_id,
            "task.question": task.question,
            "task.has_file": bool(task.file_name)
        }
        # This is a top-level operation for a task, so use `trace`
        with self.tracer.trace(name=f"process_task_{task.task_id}", metadata=process_task_metadata):
            logger.info(f"[Task {task.task_id}] ▶ {task.question}")
            
            file_path = self._process_media(task) # This now uses tracer.span
            process_task_metadata['file.path.result'] = file_path # Log result if needed

            selected_tool_name_hint: Optional[str] = "llm_tool"
            tool_selection_metadata = {"task.id": task.task_id}
            try:
                with self.tracer.span(name="tool_selection", metadata=tool_selection_metadata):
                    _selected_domain_tool = self.tool_selector.select_tool(task, self.available_tools_descriptions)
                    if _selected_domain_tool:
                        selected_tool_name_hint = _selected_domain_tool.name
                        tool_selection_metadata["tool.selected"] = selected_tool_name_hint
                    else:
                        # selected_tool_name_hint remains "llm_tool"
                        tool_selection_metadata["tool.selected"] = "llm_tool"
                
                logger.info(f"[Task {task.task_id}] 🔧 Tool suggested by selector: {selected_tool_name_hint}")
                process_task_metadata["tool.selected"] = selected_tool_name_hint

            except Exception as e:
                logger.exception(f"[Task {task.task_id}] ❌ tool-routing error via ToolSelector: {e}")
                selected_tool_name_hint = "llm_tool" # Fallback
                tool_selection_metadata["error"] = str(e)
                process_task_metadata["tool.selection.error"] = str(e)
                process_task_metadata["tool.selected"] = "llm_tool"
            
            answer_text = self._invoke_agent(task, selected_tool_name_hint, file_path) # This now uses tracer.span
            process_task_metadata["answer.length"] = len(answer_text)
            
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
                 tool_selector: ToolSelectorPort,
                 tool_service: ToolService,
                 tracer: TracerPort):
        self.task_gateway = task_gateway
        self.file_service = file_service
        self.tool_selector = tool_selector
        self.tool_service = tool_service
        self.tracer = tracer

        # Use the tool service to get domain tools
        self.domain_tool_descriptions = self.tool_service.get_all_tools()
        
        # Get Langchain tools via the agent initializer (which should be configured properly)
        self.agent_initializer_port = agent_initializer_port
        self.agent_graph_port = self._initialize_agent_once()

    def _initialize_agent_once(self) -> AgentGraphPort:
        """Initialize the AI agent graph using the configured AgentInitializationPort."""
        metadata = {"agent.initialization.success": False}  # Default to False
        # This is a top-level operation for initialization, so use `trace`
        with self.tracer.trace(name="initialize_agent_graph", metadata=metadata):
            logger.debug("Initializing agent graph via port...")
            try:
                # The agent_initializer_port should already have been configured with necessary tools
                # (e.g. Langchain tools) and LLM service port during its own instantiation.
                agent_graph_adapter = self.agent_initializer_port.initialize_agent_graph()
                logger.info("Agent graph created successfully via port.")
                metadata["agent.initialization.success"] = True
                return agent_graph_adapter
            except Exception as e:
                logger.exception("Failed to initialize agent graph via port.")
                metadata["agent.initialization.error"] = str(e)
                # metadata["agent.initialization.success"] remains False
                raise RuntimeError("Agent initialization failed") from e
    
    def _submit_results(self, profile: OAuthProfile, answers: List[Answer], space_id: Optional[str]) -> Tuple[str, Dict]:
        """Submit answers using TaskGatewayPort."""
        submit_metadata = {
            "user.username": profile.username if profile else "unknown_user",
            "answers.count": len(answers),
            "space_id": space_id or "none",
            "submit.success": False  # Default to False
        }
        # This is typically a span within run_all_tasks.
        with self.tracer.span(name="submit_results", metadata=submit_metadata):
            if not answers:
                logger.warning("No answers to submit")
                submit_metadata["submit.reason"] = "no_answers"
                return "No answers were generated to submit.", {
                    "username": profile.username if profile else "unknown_user",
                    "score": "N/A",
                    "correct_count": 0,
                    "total_attempted": 0,
                    "message": "No answers generated."
                }

            answers_payload = [ans.to_submission_format() for ans in answers]
            logger.info(f"Submitting {len(answers)} answers for user {profile.username if profile else 'unknown_user'}")
            
            try:
                submission_response = self.task_gateway.submit_answers(profile.username if profile else "unknown_user", answers_payload, space_id)
                submit_metadata["submit.success"] = True
                if "score" in submission_response:
                    submit_metadata["score"] = submission_response["score"]
                if "correct_count" in submission_response:
                    submit_metadata["correct_count"] = submission_response["correct_count"]
                if "total_attempted" in submission_response:
                    submit_metadata["total_attempted"] = submission_response["total_attempted"]
                return "Submission successful", submission_response
            except Exception as e:
                logger.exception(f"Error submitting answers via TaskGatewayPort for user {profile.username if profile else 'unknown_user'}: {e}")
                # submit_metadata["submit.success"] remains False
                submit_metadata["submit.error"] = str(e)
                return f"Submission failed: {str(e)}", {
                    "error": str(e)
                }

    def run_all_tasks(self, profile: OAuthProfile, space_id: Optional[str]) -> Tuple[str, Optional[pd.DataFrame]]:
        """Run all tasks and submit answers using injected services."""
        current_username = profile.username if profile and profile.username else "empty_profile"
        
        run_all_metadata = {
            "user.username": current_username,
            "space_id": space_id or "none",
            "agent.available": False,  # Default
            "submission.success": False  # Default
        }
        # This is a top-level operation, so use `trace`
        with self.tracer.trace(name="run_all_tasks", metadata=run_all_metadata):
            effective_profile = profile
            if current_username == "empty_profile":
                effective_profile = OAuthProfile(username="empty_profile", email=None, name=None)
                logger.warning("User not logged in or username missing in profile, using empty_profile")
            else:
                logger.info(f"Starting task execution for user: {current_username}")

            try:
                if not self.agent_graph_port:
                    logger.error("Agent graph not available.")
                    # run_all_metadata["agent.available"] remains False
                    return "Error: Agent graph not initialized.", None
                
                run_all_metadata["agent.available"] = True

                fetch_tasks_metadata = {}
                with self.tracer.span(name="fetch_tasks", metadata=fetch_tasks_metadata):
                    questions_data = self.task_gateway.fetch_tasks()
                    fetch_tasks_metadata["tasks.count"] = len(questions_data)
                    logger.info(f"Fetched {len(questions_data)} tasks from gateway")
                run_all_metadata["tasks.fetched_count"] = fetch_tasks_metadata.get("tasks.count", 0)
                
                processor = TaskProcessor(
                    file_service=self.file_service,
                    tool_selector=self.tool_selector,
                    agent_graph=self.agent_graph_port,
                    tracer=self.tracer,
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
                        error_span_metadata = {
                            "task.id": task.task_id,
                            "error": str(e)
                        }
                        with self.tracer.span(name=f"task_processing_error_{task.task_id}", metadata=error_span_metadata):
                            pass # Error is already logged and metadata captured
                        
                        results_log.append({
                            "Task ID": task.task_id,
                            "Question": task.question,
                            "Submitted Answer": f"ERROR_PROCESSING_TASK: {str(e)}",
                            "Source": "OrchestratorError"
                        })
                
                run_all_metadata["answers.generated_count"] = len(answers)
                
                submission_status_msg, result_data = self._submit_results(effective_profile, answers, space_id)
                
                if "error" in result_data or "failed" in submission_status_msg.lower():
                    status_display = f"Submission Problem: {submission_status_msg}\nDetails: {result_data.get('message', str(result_data))}"
                    run_all_metadata["submission.details"] = result_data.get('message', str(result_data))
                    # run_all_metadata["submission.success"] remains False
                elif result_data.get('message') == "No answers generated.":
                    status_display = submission_status_msg
                    run_all_metadata["submission.reason"] = "no_answers"
                    # run_all_metadata["submission.success"] remains False
                else:
                    status_display = (
                        f"Submission Successful!\nUser: {result_data.get('username', current_username)}\n"
                        f"Score: {result_data.get('score', 'N/A')}% ({result_data.get('correct_count', '?')}/"
                        f"{result_data.get('total_attempted', '?')} correct)\n"
                        f"Message: {result_data.get('message', 'No message received.')}"
                    )
                    run_all_metadata["submission.success"] = True
                    if "score" in result_data:
                        run_all_metadata["score"] = result_data["score"]
                    if "correct_count" in result_data:
                         run_all_metadata["correct_count"] = result_data["correct_count"]
                    if "total_attempted" in result_data:
                         run_all_metadata["total_attempted"] = result_data["total_attempted"]

                logger.info(f"Submission complete for user {current_username}. Status: {status_display}")
                return status_display, pd.DataFrame(results_log)
                
            except Exception as e:
                logger.exception(f"Unexpected error in run_all_tasks for user {current_username}: {str(e)}")
                run_all_metadata["error"] = str(e)
                return f"Unexpected error: {e}", None
