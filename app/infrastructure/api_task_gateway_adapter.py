import os
import requests
import tempfile
from typing import List, Dict, Any, Optional

from app.application.ports import TaskGatewayPort
from app.config import settings, AppSettings # Import centralized settings and AppSettings for type hinting

class APITaskGatewayAdapter(TaskGatewayPort):
    """Adapter for the Task API"""
    
    BASE_URL = "https://agents-course-unit4-scoring.hf.space"
    
    def __init__(self, app_settings: AppSettings): # Accept AppSettings instance
        self.settings = app_settings
    
    def fetch_tasks(self) -> List[Dict[str, Any]]:
        """Fetch all tasks (questions) from the API"""
        response = requests.get(f"{self.BASE_URL}/questions", timeout=30)
        response.raise_for_status()
        return response.json()
    
    
            
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