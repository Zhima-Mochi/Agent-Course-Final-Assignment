from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field # Import Field for env aliases if needed
from typing import Optional

class AppSettings(BaseSettings):
    # env_prefix can be used if your env variables have a common prefix e.g. MYAPP_
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')

    OPENAI_API_KEY: str
    SPACE_ID: Optional[str] = None # Or some other default if appropriate
  
    LOG_LEVEL: str = "INFO"

    # Centralized OpenAI client configurations
    OPENAI_MODEL_NAME: str = "gpt-4o-mini"
    OPENAI_TEMPERATURE: float = 0.2
    # For OPENAI_TIMEOUT, if the .env variable is OPENAI_TIMEOUT, 
    # pydantic-settings will map it to a field named openai_timeout (case-insensitive matching for env vars).
    # If the field name in the model is OPENAI_TIMEOUT, it will also map.
    # Using Field(..., alias='OPENAI_TIMEOUT_ENV_VAR') would be if env var name differs significantly.
    OPENAI_TIMEOUT: int = 30 # Corresponds to env var OPENAI_TIMEOUT
    OPENAI_MAX_TOKENS: int = 2000

# Create a single instance to be used throughout the application
settings = AppSettings() 