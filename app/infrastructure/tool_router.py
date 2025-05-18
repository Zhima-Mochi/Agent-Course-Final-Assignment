from app.domain.task import QuestionTask

def route_tool(task: QuestionTask) -> str:
    """
    Determines the appropriate tool to handle the task based on file type or question pattern.
    Returns the tool name to be invoked in the LangGraph or execution context.
    """
    if is_youtube(task):
        return "youtube_tool"
    if is_audio(task):
        return "asr_tool"
    if is_image(task):
        return "vision_tool"
    if is_excel(task):
        return "excel_tool"
    if is_vector_memory_hit(task):
        return "vector_memory_tool"
    return "llm_tool"  # default fallback


def is_youtube(task: QuestionTask) -> bool:
    q = (task.question or "").lower()
    return "youtube.com/watch" in q or "youtu.be/" in q


def is_audio(task: QuestionTask) -> bool:
    f = (task.file_name or "").lower()
    return f.endswith(".mp3") or f.endswith(".wav") or f.endswith(".m4a")


def is_image(task: QuestionTask) -> bool:
    f = (task.file_name or "").lower()
    return f.endswith((".png", ".jpg", ".jpeg", ".webp"))


def is_excel(task: QuestionTask) -> bool:
    f = (task.file_name or "").lower()
    return f.endswith((".xlsx", ".xls", ".csv"))


def is_vector_memory_hit(task: QuestionTask) -> bool:
    """
    Optional: If you integrate vector memory first-check, route it here before LLM.
    """
    return False  # Placeholder until you call retrieve_answer()
