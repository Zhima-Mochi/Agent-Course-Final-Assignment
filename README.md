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
| **Orchestrator + Agent graph** | `app/application/orchestrator.py`, `ai_agent.py` wire every tool into a LangGraph-style state machine. |
| **Tool routing layer** | `infrastructure/tools_module.py` registers search, yt-dlp transcription, code-runner, plot renderer, etc. |
| **Gradio UI with HF OAuth** | `app.py` exposes a one-click web UI; users must sign in with Hugging Face to run tasks. |
| **Data-frame output** | Answers are returned both as chat text and as a Pandas `DataFrame` for easy CSV export. |
| **Safe code execution** | Built-in sandbox. |

---

## 📂 Project Layout
```
app/
├─ application/        # orchestrator, agent graph, ports
│  ├─ ai_agent.py
│  ├─ orchestrator.py
│  └─ ports.py
├─ domain/             # tasks, value objects, tools (trimmed)
├─ infrastructure/
│  └─ tools_module.py  # concrete tool adapters
├─ config.py           # env & settings helper
└─ ...
app.py                  # Gradio entry-point
requirements.txt        # pip users
pyproject.toml          # poetry users
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

### 2. Set Environment Variables

| Variable                        | Description                        |
| ------------------------------- | ---------------------------------- |
| `OPENAI_API_KEY`                | Key for OpenAI API                 |
| `OPENAI_MODEL_NAME`             | Model name for OpenAI API          |
| `GOOGLE_SEARCH_API_KEY`         | Key for Google Search API          |
| `GOOGLE_SEARCH_ENGINE_ID`       | Google Search Engine ID            |
| `SPACE_ID` (optional)           | HF Space slug if deploying there   |

Create a `.env` file or export them in your shell.

### 3. Run Locally

```bash
poetry run python app.py
```

Login with your HF account and hit **"Run all tasks"**.

---

## 🏗️ Architecture in 60 sec

1. **Ports & Adapters** — `ports.py` defines abstract service contracts.
2. **Domain layer** — pure Python models (`Task`, `Tool`, `Answer`, `AgentState`).
3. **Application layer** — Orchestrator builds the LangGraph, selects tools, maintains conversation state.
4. **Infrastructure layer** — concrete adapters (web search, yt-dlp, matplotlib plotter, etc.).
   New tools can be added in two ways:

   ```python
   # Option 1: Quick approach with LangChain decorator
   @tool
   def my_tool(input: str) -> str:
       """Tool description goes here."""
       # Implementation
       return result
   
   # Register it with the provider
   from app.infrastructure.tool_provider import LangchainToolAdapter
   tool_provider.register_tool(LangchainToolAdapter(my_tool))
   
   # Option 2: Clean architecture approach
   from app.application.ports import ToolPort
   
   class MyCustomTool(ToolPort):
       @property
       def name(self) -> str:
           return "my_custom_tool"
       
       @property
       def description(self) -> str:
           return "Does something useful with the input"
       
       def execute(self, input: str, **kwargs) -> Dict[str, Any]:
           # Implementation
           return {"result": processed_result}
   
   # Register with any ToolProviderPort implementation
   tool_provider.register_tool(MyCustomTool())
   ```

---

## 🚀 Deployment to Hugging Face

1. Push the repo to a Space with **SDK = gradio**.
2. Set the same env vars in the Space's **Secrets** panel.
3. The Gradio UI boots automatically on every commit.
