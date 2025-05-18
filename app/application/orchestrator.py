import os
import pandas as pd
import logging
from contextlib import suppress
from typing import Tuple, Dict, Any, List, Optional
from langchain_core.messages import HumanMessage
from dataclasses import dataclass

import gradio as gr
from app.domain.agent import AIAgent
from app.domain.task import QuestionTask
from app.domain.value_objects import Answer
from app.infrastructure.http_gateway import APIGateway
from app.infrastructure.tools_module import init_tools
from app.infrastructure.llm_service import LLMService
from app.services.media_service import parse_uploaded_media
from app.infrastructure.tool_router import route_tool
from app.infrastructure.vector_tools import retrieve_answer, store_to_vectordb
from app.infrastructure.env_config import initialize_environment

logger = logging.getLogger(__name__)
initialize_environment()

class TaskProcessor:
    """Handles the processing of individual tasks including media, memory, and agent invocation."""
    
    def __init__(self, agent_graph):
        self.agent_graph = agent_graph
    
    def process_media(self, task: QuestionTask) -> Optional[Any]:
        """Handle media parsing for the task."""
        if not task.file_name:
            return None
            
        try:
            file_path = f"task_files/{task.file_name}"
            APIGateway.fetch_task_file(task.task_id, file_path)
            parsed_input = parse_uploaded_media(task.file_name, file_path)
            logger.debug(f"[Task {task.task_id}] Media parsed")
            return parsed_input
        except Exception as e:
            logger.exception(f"[Task {task.task_id}] ❌ media-parse error: {e}")
            return None

    def check_memory(self, question: str, task_id: str) -> Optional[str]:
        """Check if answer exists in memory."""
        try:
            mem_answer = retrieve_answer(question or "")
            if mem_answer and not mem_answer.startswith("❌"):
                logger.info(f"[Task {task_id}] 🧠 Memory hit")
                return mem_answer
        except Exception as e:
            logger.warning(f"[Task {task_id}] ❌ memory-lookup error: {e}")
        return None

    def invoke_agent(self, task: QuestionTask, tool_name: str, parsed_input: Optional[Any]) -> str:
        """Invoke the agent to get an answer."""
        try:
            state = {
                "question": task.question,
                "file": task.file_name,
                "parsed_input": parsed_input,
                "tool_used": tool_name,
                "llm_output": None,
                "messages": [HumanMessage(content=task.question)]  # Add question as initial message
            }
            state_out = self.agent_graph.invoke(state)
            answer_text = state_out["messages"][-1].content.strip()
            logger.info(f"[Task {task.task_id}] ✅ Agent answer generated")
            return answer_text
        except Exception as e:
            error_msg = f"ERROR: {e}"
            logger.exception(f"[Task {task.task_id}] ❌ agent-invoke error: {e}")
            return error_msg

    def process_task(self, task: QuestionTask) -> Dict[str, Any]:
        """Process a single task through the entire pipeline."""
        logger.info(f"[Task {task.task_id}] ▶ {task.question}")
        
        # 1. Media parsing
        parsed_input = self.process_media(task)
        
        # 2. Memory lookup
        if mem_answer := self.check_memory(task.question, task.task_id):
            return {
                "answer": Answer(task_id=task.task_id, content=mem_answer),
                "result": {
                    "Task ID": task.task_id, 
                    "Question": task.question, 
                    "Submitted Answer": mem_answer
                }
            }
        
        # 3. Tool routing
        try:
            tool_name = route_tool(task, parsed_input)
            logger.info(f"[Task {task.task_id}] 🔧 Tool chosen: {tool_name}")
        except Exception as e:
            logger.exception(f"[Task {task.task_id}] ❌ tool-routing error: {e}")
            tool_name = "llm_tool"  # safe fallback
        
        # 4. Agent invocation
        answer_text = self.invoke_agent(task, tool_name, parsed_input)
        
        # # 5. Store to vector DB
        # with suppress(Exception):
        #     store_to_vectordb(
        #         texts=[task.question],
        #         metadatas=[{"answer": answer_text}]
        #     )
        
        return {
            "answer": Answer(task_id=task.task_id, content=answer_text),
            "result": {
                "Task ID": task.task_id,
                "Question": task.question,
                "Submitted Answer": answer_text
            }
        }

class Orchestrator:
    """Orchestrates the entire process of fetching questions, answering them, and submitting responses"""
    
    @staticmethod
    def _initialize_agent() -> Any:
        """Initialize the AI agent and its components."""
        llm = LLMService.create_llm_openai()
        tools = init_tools()
        logger.debug("LLM and tools initialized")
        
        agent = AIAgent(
            name="Multimodal AI Agent",
            llm=llm,
            tools=tools
        )
        
        return agent.to_langgraph_agent()
    
    @staticmethod
    def _submit_results(profile: gr.OAuthProfile, answers: List[Answer]) -> Tuple[str, Dict]:
        """Submit answers and return the result status."""
        answers_payload = [ans.to_submission_format() for ans in answers]
        space_id = os.getenv("SPACE_ID")
        logger.info(f"Submitting {len(answers)} answers for user {profile.username}")
        return APIGateway.submit_answers(profile.username, answers_payload, space_id)
    
    @classmethod
    def run_all_tasks(cls, profile: gr.OAuthProfile) -> Tuple[str, Optional[pd.DataFrame]]:
        """Run all tasks and submit answers"""
        if not profile:
            logger.warning("User not logged in to Hugging Face")
            return "Please Login to Hugging Face.", None
        
        try:
            logger.info(f"Starting task execution for user: {profile.username}")
            
            # Initialize agent
            agent_graph = cls._initialize_agent()
            logger.info("Agent graph created successfully")
            
            # Fetch questions
            questions_data = APIGateway.fetch_questions()
            logger.info(f"Fetched {len(questions_data)} questions from API")
            
            # Process tasks
            processor = TaskProcessor(agent_graph)
            answers: List[Answer] = []
            results_log: List[Dict[str, Any]] = []
            
            # Limit to first question for testing
            questions_data = questions_data[1:2]
            
            for item in questions_data:
                task = QuestionTask(
                    task_id=item.get("task_id"),
                    question=item.get("question"),
                    file_name=item.get("file_name")
                )
                
                try:
                    result = processor.process_task(task)
                    answers.append(result["answer"])
                    results_log.append(result["result"])
                except Exception as e:
                    logger.error(f"Error processing task {task.task_id}: {str(e)}")
                    results_log.append({
                        "Task ID": task.task_id,
                        "Question": task.question,
                        "Submitted Answer": str(e)
                    })
            
            # Handle submission results
            if not answers:
                logger.warning("No answers to submit")
                return "No answers submitted.", pd.DataFrame(results_log)
            
            # Submit and format results
            result_data = cls._submit_results(profile, answers)
            status = (
                f"Submission Successful!\nUser: {result_data.get('username')}\n"
                f"Score: {result_data.get('score', 'N/A')}% ({result_data.get('correct_count', '?')}/"
                f"{result_data.get('total_attempted', '?')} correct)\n"
                f"Message: {result_data.get('message', 'No message received.')}"
            )
            logger.info(f"Submission complete. Score: {result_data.get('score', 'N/A')}%")
            return status, pd.DataFrame(results_log)
            
        except Exception as e:
            logger.exception(f"Unexpected error in run_all_tasks: {str(e)}")
            return f"Unexpected error: {e}", None
