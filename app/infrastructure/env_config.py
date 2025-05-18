import os
import base64
from dotenv import load_dotenv

def initialize_environment():
    """Set up environment variables and configuration for the application"""
    # Load environment variables from .env file
    load_dotenv()
    
    # Set up telemetry if Langfuse credentials are provided
    if os.getenv('LANGFUSE_PUBLIC_KEY') and os.getenv('LANGFUSE_SECRET_KEY'):
        auth = base64.b64encode(
            f"{os.getenv('LANGFUSE_PUBLIC_KEY')}:{os.getenv('LANGFUSE_SECRET_KEY')}".encode()
        ).decode()
        os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = "https://cloud.langfuse.com/api/public/otel"
        os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"Authorization=Basic {auth}"
    
    # Log startup information
    print("\n" + "-"*30 + " Environment Initialized " + "-"*30)
    space_host = os.getenv("SPACE_HOST")
    space_id = os.getenv("SPACE_ID")

    if space_host:
        print(f"✅ SPACE_HOST found: {space_host}")
        print(f"   Runtime URL: https://{space_host}.hf.space")
    
    if space_id:
        print(f"✅ SPACE_ID found: {space_id}")
        print(f"   Repo URL: https://huggingface.co/spaces/{space_id}")
    
    print("-"*76 + "\n") 