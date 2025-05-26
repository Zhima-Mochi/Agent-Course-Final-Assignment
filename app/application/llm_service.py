from typing import Any, List, Optional
from langchain_openai import ChatOpenAI

from app.config import settings, AppSettings

class OpenAILLMService:
    """Direct implementation of LLM service using OpenAI/LangChain in the application layer."""
    
    def __init__(self, app_settings: Optional[AppSettings] = None):
        self.settings = app_settings or settings
        
        if not self.settings.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not found in settings.")
        
        self._llm = ChatOpenAI(
            model=self.settings.OPENAI_MODEL_NAME,
            max_tokens=self.settings.OPENAI_MAX_TOKENS,
            temperature=self.settings.OPENAI_TEMPERATURE,
            api_key=self.settings.OPENAI_API_KEY,
            request_timeout=self.settings.OPENAI_TIMEOUT
        )

    def get_llm(self) -> ChatOpenAI:
        """Get the underlying LangChain LLM instance."""
        return self._llm 