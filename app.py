import logging
from typing import Optional
import os
import gradio as gr
import gradio.oauth
import requests
import inspect
import pandas as pd
from app.application.orchestrator import Orchestrator
from app.config import settings

# Infrastructure - Adapters
from app.infrastructure.api_task_gateway_adapter import APITaskGatewayAdapter
from app.infrastructure.tool_selection_adapter import BasicToolSelectorAdapter
from app.infrastructure.openai_llm_adapter import OpenAILLMAdapter
from app.infrastructure.langgraph_agent_adapter import LangGraphAgentInitializerAdapter

# Domain - for PromptStrategy if needed directly, or through adapter construction
from app.domain.prompt_strategy import BasicPromptStrategy

# Tooling - for Langchain tools if needed directly for adapter construction
from app.infrastructure.tools_module import init_tools as get_langchain_tools

# Create logs directory
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# Configure root logger with both console and file handlers
# Get log level from settings, default to INFO if not valid
log_level_str = settings.LOG_LEVEL.upper()
numeric_log_level = getattr(logging, log_level_str, logging.INFO)

logging.basicConfig(
    level=numeric_log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Console handler
        logging.FileHandler(os.path.join(log_dir, "app.log"))  # File handler
    ]
)
logger = logging.getLogger(__name__)
logger.info(f"Application starting with log level: {log_level_str}")

# --- Initialize Application Components (Dependency Injection) ---
try:
    # 1. Settings object is `app.config.settings`
    logger.info(f"SPACE_ID from settings: {settings.SPACE_ID}")
    logger.info(f"OpenAI Model from settings: {settings.OPENAI_MODEL_NAME}")

    # 2. Instantiate Adapters, passing the settings object
    task_gateway_adapter = APITaskGatewayAdapter(app_settings=settings)
    file_service_adapter = task_gateway_adapter 
    tool_selector_adapter = BasicToolSelectorAdapter()
    
    llm_service_adapter = OpenAILLMAdapter(app_settings=settings)
    # model_name, temperature etc. are now read from settings within OpenAILLMAdapter itself
    
    langchain_tools_list = get_langchain_tools()
    prompt_strategy_instance = BasicPromptStrategy()
    
    agent_initializer_adapter = LangGraphAgentInitializerAdapter(
        llm_service_port=llm_service_adapter,
        lc_tools=langchain_tools_list, # Note: was langchain_tools, changed to lc_tools in adapter
        prompt_strategy=prompt_strategy_instance
    )

    # 3. Instantiate Orchestrator
    orchestrator_instance = Orchestrator(
        task_gateway=task_gateway_adapter,
        agent_initializer_port=agent_initializer_adapter,
        file_service=file_service_adapter,
        tool_selector=tool_selector_adapter
    )
    logger.info("Orchestrator and all dependencies initialized successfully.")

except Exception as e:
    logger.exception("Failed to initialize application components. Application will not run correctly.")
    orchestrator_instance = None

# --- Gradio Interface Function ---
def run_evaluation_wrapper(profile: gradio.oauth.OAuthProfile) -> tuple[str, Optional[pd.DataFrame]]:
    # --- Determine HF Space Runtime URL and Repo URL ---
    if profile:
        username= f"{profile.username}"
        logger.info(f"User logged in: {username}")
    else:
        logger.info("User not logged in.")
        return "Please Login to Hugging Face with the button.", None

    if orchestrator_instance is None:
        error_msg = "Application components failed to initialize. Cannot run evaluation."
        logger.error(error_msg)
        return error_msg, None

    # The space_id is now read from settings by the adapter if not provided.
    # Orchestrator's run_all_tasks expects space_id. We can pass settings.SPACE_ID
    # or None and let the adapter pick it up from settings.
    # For clarity, passing it from settings if available, or None.
    space_id_to_pass = settings.SPACE_ID or os.getenv("SPACE_ID")

    logger.info(f"run_evaluation_wrapper called. Profile: {profile.username if profile else 'No Profile'}, Space ID to pass: {space_id_to_pass}")
    return orchestrator_instance.run_all_tasks(profile, space_id_to_pass)


# --- Build Gradio Interface using Blocks ---
with gr.Blocks() as demo:
    gr.Markdown("# Basic Agent Evaluation Runner")
    gr.Markdown(
        """
        **Instructions:**

        1.  Please clone this space, then modify the code to define your agent's logic, the tools, the necessary packages, etc ...
        2.  Log in to your Hugging Face account using the button below. This uses your HF username for submission.
        3.  Click 'Run Evaluation & Submit All Answers' to fetch questions, run your agent, submit answers, and see the score.

        ---
        **Disclaimers:**
        Once clicking on the "submit button, it can take quite some time ( this is the time for the agent to go through all the questions).
        This space provides a basic setup and is intentionally sub-optimal to encourage you to develop your own, more robust solution. For instance for the delay process of the submit button, a solution could be to cache the answers and submit in a seperate action or even to answer the questions in async.
        """
    )

    gr.LoginButton()

    run_button = gr.Button("Run Evaluation & Submit All Answers")

    status_output = gr.Textbox(label="Run Status / Submission Result", lines=5, interactive=False)
    # Removed max_rows=10 from DataFrame constructor
    results_table = gr.DataFrame(label="Questions and Agent Answers", wrap=True)

    run_button.click(
        fn=run_evaluation_wrapper,
        outputs=[status_output, results_table]
    )

if __name__ == "__main__":
    print("\n" + "-"*30 + " App Starting " + "-"*30)
    space_host_startup = os.getenv("SPACE_HOST")
    
    if space_host_startup:
        print(f"✅ SPACE_HOST found: {space_host_startup}")
        print(f"   Runtime URL should be: https://{space_host_startup}.hf.space")
    else:
        print("ℹ️  SPACE_HOST environment variable not found (running locally?).")

    if settings.SPACE_ID: # Use settings for SPACE_ID check
        print(f"✅ SPACE_ID found via settings: {settings.SPACE_ID}")
        print(f"   Repo URL: https://huggingface.co/spaces/{settings.SPACE_ID}")
        print(f"   Repo Tree URL: https://huggingface.co/spaces/{settings.SPACE_ID}/tree/main")
    else:
        print("ℹ️  SPACE_ID environment variable not found in settings (running locally?). Repo URL cannot be determined.")

    print("-"*(60 + len(" App Starting ")) + "\n")

    print("Launching Gradio Interface for Basic Agent Evaluation...")
    if orchestrator_instance is None:
        print("ERROR: Orchestrator failed to initialize. Gradio interface might not function correctly or at all.")
        # Optionally, do not launch, or launch with a clear error message in the UI itself.
        # For now, it will launch, but run_evaluation_wrapper will show an error.

    demo.launch(debug=True, share=False) # Consider debug=False for production