from dataclasses import dataclass
from typing import Optional
import os

@dataclass
class QuestionTask:
    task_id: str
    question: str
    file_name: Optional[str] = None
    
    @property
    def file_path(self) -> Optional[str]:
        """Return the file path if a file name is provided"""
        return f"task_files/{self.file_name}" if self.file_name else None
        
    @property
    def file_type(self) -> Optional[str]:
        """Return the file type (extension) if a file name is provided"""
        if not self.file_name:
            return None
        _, ext = os.path.splitext(self.file_name)
        return ext.lower()[1:] if ext else None