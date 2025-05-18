import os
from typing import Dict, List, Any
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI


class LLMService:
    """Service for interacting with LLMs"""

    @staticmethod
    def create_llm_openai() -> ChatOpenAI:
        """Create a ChatOpenAI instance with optimal settings for agent usage"""
        return ChatOpenAI(
            model="gpt-4o-mini",  # Could upgrade to gpt-4o for better performance if budget allows
            max_tokens=2000,      # Increased from 1000 to allow for more detailed reasoning
            temperature=0.2,      # Lower temperature for more deterministic outputs (better for structured answers)
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
    @staticmethod
    def create_reasoning_llm() -> ChatOpenAI:
        """Create a specialized LLM instance for reasoning and structured output"""
        return ChatOpenAI(
            model="gpt-4o-mini",
            max_tokens=4000,      # Higher token limit for complex reasoning
            temperature=0.0,      # Zero temperature for maximum determinism
            api_key=os.getenv("OPENAI_API_KEY")
        )
