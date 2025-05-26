---
title: Agent-Course-Final-Assignment
emoji: 🧑‍🔬🤖
colorFrom: indigo
colorTo: indigo
sdk: gradio
sdk_version: 5.25.2
app_file: app.py
pinned: false
hf_oauth: true
hf_oauth_expiration_minutes: 480
---

# Agent-Course-Final-Assignment 🧑‍🔬🤖

A **Gradio-powered AI-Agent demo** that showcases how to orchestrate multiple LLM tools (search, audio-to-text, code-execution, data-viz, etc.) with **LangGraph / LangChain** into a single, login-protected workflow.  
The codebase is the capstone project for the "Build Your Own Agent" course.

---

## ✨ Key Features

| Feature | What it does |
|---------|--------------|
| **LangGraph Agent** | `app/application/langgraph_agent.py` implements a full agent with planning, thinking, and tool execution stages. |
| **Task Processing Pipeline** | `app/application/task_processor.py` handles task execution with file processing capabilities. |
| **Extensive Tool Library** | `app/infrastructure/tools_module.py` provides ready-to-use tools for search, transcription, code execution, and more. |
| **Gradio UI with HF OAuth** | `app.py` delivers a web interface with Hugging Face authentication for user tracking. |
| **Structured Answer Format** | Results are returned as both chat text and a pandas DataFrame for analysis. |
| **Langfuse Tracing** | Built-in LLM observability with conditional Langfuse integration. |

---

## 📂 Project Layout

```
app/
├─ application/          # Core agent implementation
│  ├─ langgraph_agent.py # LangGraph-based agent with planning, thinking, tools
│  ├─ task_processor.py  # Task execution pipeline
│  ├─ llm_service.py     # LLM service interface
│  └─ ports.py           # Abstract interfaces
├─ domain/               # Business domain objects
│  ├─ value_objects.py   # Data models for state management
│  └─ prompt_strategy.py # Prompt generation strategies
├─ infrastructure/       # External service implementations
│  ├─ tools_module.py    # Tool implementations
│  └─ tool_provider.py   # Tool registration and management
├─ task_controller.py    # Orchestrates evaluation workflow
└─ config.py             # Environment and settings
app.py                   # Gradio entry-point
requirements.txt         # pip dependencies
pyproject.toml           # poetry configuration
```

---

## 🛠️ Quick Start

### 1. Clone & Install

```bash
git clone https://huggingface.co/spaces/Zhima-Mochi/Agent-Course-Final-Assignment
cd Agent-Course-Final-Assignment

# Poetry (recommended)
poetry install
poetry shell

# or plain pip
pip install -r requirements.txt
```

### 2. Configure Environment Variables

| Variable                        | Description                         |
| ------------------------------- | ----------------------------------- |
| `OPENAI_API_KEY`                | Key for OpenAI API                  |
| `OPENAI_MODEL_NAME`             | Model name for OpenAI API           |
| `GOOGLE_SEARCH_API_KEY`         | Key for Google Search API           |
| `GOOGLE_SEARCH_ENGINE_ID`       | Google Search Engine ID             |
| `SPACE_ID` (optional)           | HF Space slug if deploying there    |
| `ENABLE_TRACING`                | Enable Langfuse tracing (true/false)|
| `LANGFUSE_PUBLIC_KEY`           | Langfuse public API key             |
| `LANGFUSE_SECRET_KEY`           | Langfuse secret API key             |
| `LANGFUSE_HOST`                 | Langfuse host URL                   |
| `LOG_LEVEL`                     | Logging level (INFO, DEBUG, etc.)   |

Create a `.env` file or export them in your shell.

### 3. Run Locally

```bash
poetry run python app.py
```

Login with your HF account and hit **"Run all tasks"**.

---

## 🏗️ Architecture Overview

The application follows a clean architecture pattern with these key components:

1. **LangGraph Agent** - A state machine that manages:
   - Task planning: Breaking complex problems into steps
   - Assistant thinking: Reasoning through solutions
   - Tool execution: Using relevant tools when needed
   - Answer finalization: Refining responses for clarity

2. **Task Processing Pipeline** - Handles:
   - Media file processing
   - Agent invocation with proper context
   - Result formatting

3. **Tools System** - Extensible tool registry with:
   - Web search capabilities
   - Media processing
   - Code execution
   - Data visualization

Adding new tools is straightforward:

```python
from app.infrastructure.tool_provider import ToolProvider

@tool
def my_custom_tool(input: str) -> str:
    """Tool description goes here."""
    # Implementation
    return result

# Register with the provider
tool_provider = ToolProvider()
tool_provider.register_tool(my_custom_tool)
```

---

## 🚀 Deployment to Hugging Face

1. Create a new Space on Hugging Face with **SDK = gradio**
2. Push your code to the Space repository
3. Add all required environment variables to the Space's **Secrets** panel
4. The Gradio interface will deploy automatically

---

## 📊 Observability with Langfuse

This project includes integration with [Langfuse](https://langfuse.com) for LLM observability and tracing.

### What's being traced

The application uses Langfuse to trace:

- LLM calls through the LangGraph agent
- Tool selections and executions
- Agent state transitions
- Performance metrics

View traces in the Langfuse UI to analyze agent behavior, identify bottlenecks, and improve performance.
