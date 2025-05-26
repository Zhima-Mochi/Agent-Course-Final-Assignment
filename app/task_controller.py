import pandas as pd
import logging
from typing import Tuple, Dict, Any, List, Optional

import gradio as gr
from gradio.oauth import OAuthProfile
from app.domain.value_objects import QuestionTask, Answer
from app.application.ports import TaskGatewayPort
from app.application.task_processor import TaskProcessor

logger = logging.getLogger(__name__)


class TaskController:
    """Input adapter that handles workflow coordination and external interactions."""

    def __init__(self, task_gateway: TaskGatewayPort, task_processor: TaskProcessor):
        """Initialize with necessary dependencies."""
        self.task_gateway = task_gateway
        self.task_processor = task_processor

    def submit_answers(self, profile: OAuthProfile, answers: List[Answer], space_id: Optional[str]) -> Tuple[str, Dict]:
        """Submit answers using TaskGatewayPort."""
        if not answers:
            logger.warning("No answers to submit")
            return "No answers were generated to submit.", {
                "username": profile.username if profile else "unknown_user",
                "score": "N/A",
                "correct_count": 0,
                "total_attempted": 0,
                "message": "No answers generated."
            }

        answers_payload = [ans.to_submission_format() for ans in answers]
        username = profile.username if profile else "unknown_user"
        logger.info(f"Submitting {len(answers)} answers for user {username}")

        try:
            submission_response = self.task_gateway.submit_answers(
                username, answers_payload, space_id)
            return "Submission successful", submission_response
        except Exception as e:
            logger.exception(
                f"Error submitting answers via TaskGatewayPort for user {username}: {e}")
            return f"Submission failed: {str(e)}", {
                "error": str(e)
            }

    def process_all_tasks(self, profile: OAuthProfile, space_id: Optional[str]) -> Tuple[str, Optional[pd.DataFrame]]:
        """Process all tasks and submit answers."""
        current_username = profile.username if profile and profile.username else "empty_profile"

        effective_profile = profile
        if current_username == "empty_profile":
            effective_profile = OAuthProfile(
                username="empty_profile", email=None, name=None)
            logger.warning(
                "User not logged in or username missing in profile, using empty_profile")
        else:
            logger.info(
                f"Starting task execution for user: {current_username}")

        try:
            questions_data = self.task_gateway.fetch_tasks()
            logger.info(f"Fetched {len(questions_data)} tasks from gateway")

            answers: List[Answer] = []
            results_log: List[Dict[str, Any]] = []

            for item in questions_data:
                task = QuestionTask(
                    task_id=str(item.get("task_id")),
                    question=item.get("question"),
                    file_name=item.get("file_name")
                )

                try:
                    result_info = self.task_processor.process_task(task)
                    answers.append(result_info["answer"])
                    results_log.append(result_info["result"])
                except Exception as e:
                    logger.error(
                        f"Error processing task {task.task_id}: {str(e)}", exc_info=True)

                    results_log.append({
                        "Task ID": task.task_id,
                        "Question": task.question,
                        "Submitted Answer": f"ERROR_PROCESSING_TASK: {str(e)}",
                        "Source": "ControllerError"
                    })

            submission_status_msg, result_data = self.submit_answers(
                effective_profile, answers, space_id)

            if "error" in result_data or "failed" in submission_status_msg.lower():
                status_display = f"Submission Problem: {submission_status_msg}\nDetails: {result_data.get('message', str(result_data))}"
            elif result_data.get('message') == "No answers generated.":
                status_display = submission_status_msg
            else:
                status_display = (
                    f"Submission Successful!\nUser: {result_data.get('username', current_username)}\n"
                    f"Score: {result_data.get('score', 'N/A')}% ({result_data.get('correct_count', '?')}/"
                    f"{result_data.get('total_attempted', '?')} correct)\n"
                    f"Message: {result_data.get('message', 'No message received.')}"
                )

            logger.info(
                f"Submission complete for user {current_username}. Status: {status_display}")
            return status_display, pd.DataFrame(results_log)

        except Exception as e:
            logger.exception(
                f"Unexpected error in process_all_tasks for user {current_username}: {str(e)}")
            return f"Unexpected error: {e}", None
