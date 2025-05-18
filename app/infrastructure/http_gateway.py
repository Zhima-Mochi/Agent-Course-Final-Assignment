import os
import requests
from typing import List, Dict, Any

class APIGateway:
    """Gateway for interacting with external HTTP APIs"""
    
    BASE_URL = "https://agents-course-unit4-scoring.hf.space"
    
    @classmethod
    def fetch_questions(cls) -> List[Dict[str, Any]]:
        """Fetch all questions from the API"""
        response = requests.get(f"{cls.BASE_URL}/questions", timeout=30)
        response.raise_for_status()
        return response.json()
    
    @classmethod
    def fetch_task_file(cls, task_id: str, file_path: str) -> None:
        """Download a file for a specific task"""
        response = requests.get(f"{cls.BASE_URL}/files/{task_id}", timeout=30)
        response.raise_for_status()
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Write the file content
        with open(file_path, "wb") as f:
            f.write(response.content)
    
    @classmethod
    def submit_answers(cls, username: str, answers_payload: List[Dict[str, str]], space_id: str) -> Dict[str, Any]:
        """Submit answers to the API"""
        response = requests.post(
            f"{cls.BASE_URL}/submit", 
            json={
                "username": username.strip(),
                "agent_code": f"https://huggingface.co/spaces/{space_id}/tree/main",
                "answers": answers_payload
            },
            timeout=120
        )
        response.raise_for_status()
        return response.json() 