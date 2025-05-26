import logging
import os
import tempfile
from typing import Dict, Any, List, Optional
from langchain_core.messages import HumanMessage
import requests

from app.domain.value_objects import Answer, AgentState, QuestionTask
from app.application.langgraph_agent import LangGraphAgent

logger = logging.getLogger(__name__)

class TaskProcessor:
    """Handles the processing of individual tasks using injected services (ports)."""
    
    def __init__(self, 
                 langgraph_agent: LangGraphAgent,
                 ):
        self.langgraph_agent = langgraph_agent

    def _process_media(self, task: QuestionTask) -> Optional[str]:
        """Download media file using FileServicePort."""
        if not task.file_name:
            return None
        try:
            file_path = self.download_task_file(task.task_id, task.file_name)
            
            if file_path:
                logger.debug(f"[Task {task.task_id}] Media file downloaded to {file_path}")
            else:
                logger.warning(f"[Task {task.task_id}] Media file download failed or no path returned.")
            return file_path
        except Exception as e:
            logger.exception(f"[Task {task.task_id}] ❌ media download error via FileService: {e}")
            return None

    def _invoke_agent(self, task: QuestionTask,  file_path: Optional[str],selected_tool_name: Optional[str] = None) -> str:
        """Invoke the agent graph to get an answer."""
        try:
            initial_messages = [HumanMessage(content=task.question)]
            if file_path:
                initial_messages = [HumanMessage(content=f"{task.question}\n\n(Context: File is available at path: {file_path})\n(Hint: Tool '{selected_tool_name if selected_tool_name else 'N/A'}' was suggested for this task.)")]

            state = AgentState(
                question=task.question,
                file_path=file_path,
                tool_hint=selected_tool_name,
                answer=None,
                messages=initial_messages,
            )
            state_dict = dict(state)
            logger.debug(f"[Task {task.task_id}] Invoking agent graph with initial state: {state_dict}")
            
            state_out = self.langgraph_agent.process(state_dict)  # Use process instead of invoke
            
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
        
        file_path = self._process_media(task)

        
        answer_text = self._invoke_agent(task, file_path)
        
        return {
            "answer": Answer(task_id=task.task_id, content=answer_text),
            "result": {
                "Task ID": task.task_id,
                "Question": task.question,
                "Submitted Answer": answer_text,
                "Source": "Agent"
            }
        }
        
    def download_task_file(self, task_id: str, file_name: Optional[str] = None) -> Optional[str]:
        """Download a file for a specific task to a temporary location.
        The file_name argument is kept for interface consistency but not used in this implementation.
        """
        if not task_id: # Or if file_name is essential and not provided, handle accordingly
            return None
        try:
            response = requests.get(f"{self.BASE_URL}/files/{task_id}", timeout=30)
            response.raise_for_status()
            
            # Determine a safe suffix for the temp file based on content or a generic one
            # For simosicity, not using file_name to derive suffix here, but could be an improvement
            suffix = os.path.splitext(file_name)[1] if file_name else ".tmp"
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                temp_file.write(response.content)
                return temp_file.name
        except requests.exceptions.RequestException as e:
            # Log the error, e.g., logger.error(f"Failed to download file for task {task_id}: {e}")
            print(f"Failed to download file for task {task_id}: {e}") # Placeholder for logging
            return None