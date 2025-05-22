import os
import requests
import tempfile
from typing import List, Dict, Any, Optional

from app.application.ports import TaskGatewayPort, FileServicePort
from app.config import settings, AppSettings # Import centralized settings and AppSettings for type hinting

class APITaskGatewayAdapter(TaskGatewayPort, FileServicePort):
    """Adapter for the Task API, implementing both TaskGatewayPort and FileServicePort"""
    
    BASE_URL = "https://agents-course-unit4-scoring.hf.space"
    
    def __init__(self, app_settings: AppSettings): # Accept AppSettings instance
        self.settings = app_settings
    
    def fetch_tasks(self) -> List[Dict[str, Any]]:
        """Fetch all tasks (questions) from the API"""
        response = requests.get(f"{self.BASE_URL}/questions", timeout=30)
        response.raise_for_status()
        return response.json()
    
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
            # For simplicity, not using file_name to derive suffix here, but could be an improvement
            suffix = os.path.splitext(file_name)[1] if file_name else ".tmp"
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                temp_file.write(response.content)
                return temp_file.name
        except requests.exceptions.RequestException as e:
            # Log the error, e.g., logger.error(f"Failed to download file for task {task_id}: {e}")
            print(f"Failed to download file for task {task_id}: {e}") # Placeholder for logging
            return None
            
    def submit_answers(self, user_id: str, answers_payload: List[Dict[str, str]], space_id: Optional[str]) -> Dict[str, Any]:
        """Submit answers to the API"""
        # Use SPACE_ID from settings if not provided directly
        effective_space_id = space_id or self.settings.SPACE_ID

        if not effective_space_id:
            effective_space_id = "none"

        response = requests.post(
            f"{self.BASE_URL}/submit", 
            json={
                "username": user_id.strip(),
                "agent_code": f"https://huggingface.co/spaces/{effective_space_id}/tree/main",
                "answers": answers_payload
            },
            timeout=120 # This timeout could also be from settings
        )
        response.raise_for_status()
        return response.json() 