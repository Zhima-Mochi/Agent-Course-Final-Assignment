from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional, Literal
import os

class AppSettings(BaseSettings):
    # env_prefix can be used if your env variables have a common prefix e.g. MYAPP_
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')

    OPENAI_API_KEY: str = "your_api_key"
    SPACE_ID: Optional[str] = None # Or some other default if appropriate
    
    GOOGLE_SEARCH_API_KEY: Optional[str] = None
    GOOGLE_SEARCH_ENGINE_ID: Optional[str] = None
    LOG_LEVEL: str = "INFO"

    # Centralized OpenAI client configurations
    OPENAI_MODEL_NAME: str = "gpt-4o-mini"
    OPENAI_TEMPERATURE: float = 0.2
    OPENAI_TIMEOUT: int = 30 # Corresponds to env var OPENAI_TIMEOUT
    OPENAI_MAX_TOKENS: int = 2000
    
    # Cache settings
    CACHE_DIRECTORY: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'cache')
    
    # Telemetry settings
    TELEMETRY_PROVIDER: Literal["langfuse", "none"] = "none"
    
    # Langfuse settings
    ENABLE_TRACING: bool = False
    LANGFUSE_PUBLIC_KEY: Optional[str] = None
    LANGFUSE_SECRET_KEY: Optional[str] = None 
    LANGFUSE_HOST: str = "https://cloud.langfuse.com"
    LANGFUSE_PROJECT_NAME: Optional[str] = "agent-evaluation"
    LANGFUSE_DEBUG: bool = False

# Create a single instance to be used throughout the application
settings = AppSettings() 