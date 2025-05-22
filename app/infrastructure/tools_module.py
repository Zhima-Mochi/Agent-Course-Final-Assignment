from duckduckgo_search import DDGS
import fitz  # PyMuPDF
import os
import pandas as pd
import json
import csv
import subprocess
from typing import List, Callable, Dict, Any, Optional, Tuple
from langchain_core.tools import tool
from langchain_community.document_loaders import WikipediaLoader, ArxivLoader
import speech_recognition as sr
from PIL import Image
import pytesseract
import logging
import time
import re
import sys
import io
import threading
import builtins
import math
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from contextlib import redirect_stdout, redirect_stderr

logger = logging.getLogger(__name__)


@tool
def wiki_search(query: str) -> Dict[str, Any]:
    """Search Wikipedia for a query and return maximum 2 results."""
    try:
        search_docs = WikipediaLoader(query=query, load_max_docs=2).load()
    except Exception as e:
        logger.error(f"Wiki search error: {str(e)}")
        return {"error": f"Failed to search Wikipedia: {str(e)}"}
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}\n</Document>'
            for doc in search_docs
        ]
    )
    return {"wiki_results": formatted_search_docs}


@tool
def web_search(query: str, num_results: int = 5) -> Dict[str, Any]:
    """Search the web for a query using DuckDuckGo."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=num_results))
            time.sleep(1.5)  # throttle
        return {"web_results": results}
    except Exception as e:
        logger.error(f"Web search error: {str(e)}")
        return {"error": f"Failed to search web: {str(e)}", "note": "If hit rate limit, do not use this tool."}


@tool
def arxiv_search(query: str, max_results: int = 3) -> Dict[str, Any]:
    """Search arXiv for academic papers."""
    try:
        docs = ArxivLoader(
            query=query, load_max_docs=max_results, load_all_available_meta=True
        ).load()

        formatted_docs = []
        for doc in docs:
            meta = doc.metadata
            formatted_doc = (
                f"Title: {meta.get('Title', 'N/A')}\n"
                f"Authors: {meta.get('Authors', 'N/A')}\n"
                f"Published: {meta.get('Published', 'N/A')}\n"
                f"URL: {meta.get('entry_id', 'N/A')}\n\n"
                f"Abstract: {doc.page_content[:500]}...\n"
            )
            formatted_docs.append(formatted_doc)

        return {"arxiv_results": "\n\n---\n\n".join(formatted_docs)}
    except Exception as e:
        logger.error(f"arXiv search error: {str(e)}")
        return {"error": f"Failed to search arXiv: {str(e)}"}


@tool
def read_pdf(file_path: str) -> Dict[str, Any]:
    """Extract text from a PDF file."""
    full_path = (
        os.path.join("task_files", file_path)
        if not file_path.startswith("task_files")
        else file_path
    )

    if not os.path.exists(full_path):
        return {"error": f"File not found: {full_path}"}

    try:
        doc = fitz.open(full_path)
        text = ""
        for page_num, page in enumerate(doc):
            text += f"--- Page {page_num+1} ---\n"
            text += page.get_text()
            text += "\n\n"
        return {"pdf_content": text}
    except Exception as e:
        logger.error(f"Error reading PDF: {str(e)}")
        return {"error": f"Failed to read PDF: {str(e)}"}


@tool
def analyze_csv(file_path: str, query: str = "") -> Dict[str, Any]:
    """Read and analyze a CSV file, optionally answering a specific query about the data."""
    full_path = (
        os.path.join("task_files", file_path)
        if not file_path.startswith("task_files")
        else file_path
    )

    if not os.path.exists(full_path):
        return {"error": f"File not found: {full_path}"}

    try:
        df = pd.read_csv(full_path)

        # Basic statistics and info
        summary = {
            "columns": list(df.columns),
            "shape": df.shape,
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "head": df.head(5).to_dict(orient="records"),
            "describe": df.describe().to_dict(),
        }

        if query:
            # Include query in the summary for the LLM to use its reasoning
            summary["query"] = query

        return {"csv_analysis": json.dumps(summary, indent=2)}
    except Exception as e:
        logger.error(f"Error analyzing CSV: {str(e)}")
        return {"error": f"Failed to analyze CSV: {str(e)}"}


@tool
def analyze_excel(
    file_path: str, sheet_name: str = None, query: str = ""
) -> Dict[str, Any]:
    """Read and analyze an Excel file, optionally from a specific sheet."""
    full_path = (
        os.path.join("task_files", file_path)
        if not file_path.startswith("task_files")
        else file_path
    )

    if not os.path.exists(full_path):
        return {"error": f"File not found: {full_path}"}

    try:
        if sheet_name:
            df = pd.read_excel(full_path, sheet_name=sheet_name)
        else:
            # Read all sheets
            xls = pd.ExcelFile(full_path)
            sheet_names = xls.sheet_names
            all_data = {}
            for sheet in sheet_names:
                all_data[sheet] = (
                    pd.read_excel(full_path, sheet_name=sheet)
                    .head(5)
                    .to_dict(orient="records")
                )
            return {"excel_sheets": sheet_names, "sample_data": all_data}

        # Basic statistics and info for a single sheet
        summary = {
            "columns": list(df.columns),
            "shape": df.shape,
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "head": df.head(5).to_dict(orient="records"),
            "describe": df.describe().to_dict(),
        }

        if query:
            summary["query"] = query

        return {"excel_analysis": json.dumps(summary, indent=2)}
    except Exception as e:
        logger.error(f"Error analyzing Excel: {str(e)}")
        return {"error": f"Failed to analyze Excel: {str(e)}"}


@tool
def transcribe_audio(file_path: str) -> Dict[str, Any]:
    """Transcribe speech from an audio file."""
    full_path = (
        os.path.join("task_files", file_path)
        if not file_path.startswith("task_files")
        else file_path
    )

    if not os.path.exists(full_path):
        return {"error": f"File not found: {full_path}"}

    try:
        # Ensure wav format for speech_recognition
        base, ext = os.path.splitext(full_path)
        wav_path = base + ".wav"
        # Convert mp3/other to wav via ffmpeg if needed
        if ext.lower() != ".wav":
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    full_path,
                    "-acodec",
                    "pcm_s16le",
                    "-ar",
                    "16000",
                    wav_path,
                ],
                check=True,
            )
        else:
            wav_path = full_path
        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_path) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
        return {"transcription": text}
    except Exception as e:
        logger.error(f"Error transcribing audio: {str(e)}")
        return {"error": f"Failed to transcribe audio: {str(e)}"}


@tool
def ocr_image(file_path: str) -> Dict[str, Any]:
    """Extract text from an image using OCR."""
    full_path = (
        os.path.join("task_files", file_path)
        if not file_path.startswith("task_files")
        else file_path
    )

    if not os.path.exists(full_path):
        return {"error": f"File not found: {full_path}"}

    try:
        image = Image.open(full_path)
        text = pytesseract.image_to_string(image)
        return {"ocr_text": text}
    except Exception as e:
        logger.error(f"Error performing OCR: {str(e)}")
        return {"error": f"Failed to perform OCR: {str(e)}"}


@tool
def list_files(directory: str = "task_files") -> Dict[str, Any]:
    """List all files in the specified directory."""
    directory = (
        os.path.join("task_files", directory)
        if not directory.startswith("task_files")
        else directory
    )

    if not os.path.exists(directory):
        return {"error": f"Directory not found: {directory}"}

    try:
        files = os.listdir(directory)
        return {"files": files}
    except Exception as e:
        logger.error(f"Error listing files: {str(e)}")
        return {"error": f"Failed to list files: {str(e)}"}


# New tools for YouTube transcription


@tool
def get_youtube_transcript(url: str) -> Dict[str, Any]:
    """Download YouTube transcript (if available)"""
    from youtube_transcript_api import YouTubeTranscriptApi

    try:
        video_id = re.search(r"(?:v=|be/)([\w-]+)", url).group(1)
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        text = " ".join([entry["text"] for entry in transcript])
        return {"transcript": text}
    except Exception as e:
        logger.error(f"Transcript error: {str(e)}")
        return {"error": f"Transcript not available: {str(e)}"}


@tool
def get_youtube_video_frames(url: str) -> Dict[str, Any]:
    """Download video from YouTube and extract frames (1 frame every second)"""
    import os
    import cv2
    import tempfile
    import yt_dlp

    try:
        with tempfile.TemporaryDirectory(delete = False) as tmpdir:
            video_path = os.path.join(tmpdir, "video.mp4")
            frame_dir = os.path.join(tmpdir, "frames")
            os.makedirs(frame_dir, exist_ok=True)

            ydl_opts = {
                "format": "bestvideo",
                "outtmpl": video_path,
                "quiet": True,
            }

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            count = 0
            saved = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % int(fps) == 0:
                    frame_path = os.path.join(frame_dir, f"frame_{saved:04d}.jpg")
                    cv2.imwrite(frame_path, frame)
                    saved += 1
                count += 1
            cap.release()
            return {"frame_dir": frame_dir, "frame_count": saved}

    except Exception as e:
        logger.error(f"Video frames error: {str(e)}")
        return {"error": f"Failed to extract frames: {str(e)}"}


@tool
def transcribe_youtube_audio(url: str) -> Dict[str, Any]:
    """Download YouTube audio and transcribe using Whisper"""
    import tempfile
    import yt_dlp

    try:
        with tempfile.TemporaryDirectory(delete = False) as tmpdir:
            audio_path = os.path.join(tmpdir, "audio.m4a")
            ydl_opts = {
                "format": "bestaudio",
                "outtmpl": audio_path,
                "quiet": True,
                "postprocessors": [
                    {
                        "key": "FFmpegExtractAudio",
                        "preferredcodec": "m4a",
                        "preferredquality": "192",
                    }
                ],
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            import whisper

            model = whisper.load_model("base")
            result = model.transcribe(audio_path)
            return {"transcription": result.get("text", "")}
    except Exception as e:
        logger.error(f"YouTube audio transcription failed: {str(e)}")
        return {"error": f"Transcription failed: {str(e)}"}


@tool
def execute_code(code: str) -> Dict[str, Any]:
    """
    Execute Python code provided by the LLM and return the result.
    
    Args:
        code (str): Python code to execute. The code should either:
            1. Print results using print() statements, which will be captured in stdout, or
            2. Assign the final result to a variable named 'result', which will be returned
        
    Returns:
        Dict containing stdout, stderr, and optionally a result value if defined in the code
    """
    # Create a safe globals dictionary with limited built-ins
    def get_safe_globals():
        # Start with an empty globals dictionary
        safe_globals = {}
        
        # Add safe builtins
        safe_builtins = {
            'abs': abs,
            'all': all,
            'any': any,
            'bool': bool,
            'chr': chr,
            'dict': dict,
            'dir': dir,
            'divmod': divmod,
            'enumerate': enumerate,
            'filter': filter,
            'float': float,
            'format': format,
            'frozenset': frozenset,
            'hash': hash,
            'hex': hex,
            'int': int,
            'isinstance': isinstance,
            'issubclass': issubclass,
            'len': len,
            'list': list,
            'map': map,
            'max': max,
            'min': min,
            'oct': oct,
            'ord': ord,
            'pow': pow,
            'print': print,  # Allow print for stdout capture
            'range': range,
            'repr': repr,
            'reversed': reversed,
            'round': round,
            'set': set,
            'slice': slice,
            'sorted': sorted,
            'str': str,
            'sum': sum,
            'tuple': tuple,
            'type': type,
            'zip': zip,
        }
        safe_globals['__builtins__'] = safe_builtins
        
        # Add safe modules
        safe_globals['math'] = math
        
        return safe_globals
    
    # Function to execute code with timeout
    def execute_with_timeout(code_str, globals_dict):
        # Create string buffers for stdout and stderr
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        
        # Create a local namespace for execution
        local_namespace = {}
        
        # Execute the code with redirected stdout and stderr
        with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
            # Execute the code
            result = exec(code_str, globals_dict, local_namespace)
            
        # Return the results
        return {
            "local_namespace": local_namespace,
            "stdout": stdout_buffer.getvalue(),
            "stderr": stderr_buffer.getvalue(),
            "result": result
        }
    
    try:
        # Set a timeout for code execution (5 seconds)
        MAX_EXECUTION_TIME = 5  # seconds
        
        # Check for potentially dangerous operations
        dangerous_patterns = [
            r'\bimport\s+os\b', 
            r'\bimport\s+sys\b',
            r'\bimport\s+subprocess\b',
            r'\bimport\s+shutil\b',
            r'\bopen\s*\(',
            r'\b__import__\s*\(',
            r'\beval\s*\(',
            r'\bexec\s*\(',
            r'\bcompile\s*\(',
            r'\bgetattr\s*\(',
            r'\bsetattr\s*\(',
            r'\bdelattr\s*\(',
            r'\b__\w+__\b',  # Dunder methods/attributes
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, code):
                return {"error": f"Security violation: Potentially dangerous operation detected"}
        
        # Get safe globals
        safe_globals = get_safe_globals()
        
        # Execute code with timeout using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(execute_with_timeout, code, safe_globals)
            try:
                # Wait for the result with timeout
                execution_result = future.result(timeout=MAX_EXECUTION_TIME)
                
                # Extract results
                local_namespace = execution_result["local_namespace"]
                stdout_output = execution_result["stdout"]
                stderr_output = execution_result["stderr"]
                
                # Check if there's a 'result' variable defined in the executed code
                result_value = None
                if 'result' in local_namespace:
                    result_value = local_namespace['result']
                
                # Prepare the response
                response = {
                    "stdout": stdout_output,
                    "stderr": stderr_output,
                }
                
                # Add the result value if it exists
                if result_value is not None:
                    try:
                        # Try to convert the result to a JSON-serializable format
                        json.dumps({"result": result_value})
                        response["result"] = result_value
                    except (TypeError, OverflowError):
                        # If the result can't be JSON serialized, convert it to string
                        response["result"] = str(result_value)
                
                return response
                
            except FutureTimeoutError:
                # Handle timeout
                logger.error("Code execution timed out")
                return {"error": f"Code execution timed out after {MAX_EXECUTION_TIME} seconds"}
    
    except Exception as e:
        logger.error(f"Code execution failed: {str(e)}")
        return {"error": f"Code execution failed: {str(e)}"}


tools = [
    wiki_search,
    web_search,
    arxiv_search,
    read_pdf,
    analyze_csv,
    analyze_excel,
    get_youtube_transcript,
    get_youtube_video_frames,
    transcribe_youtube_audio,
    transcribe_audio,
    ocr_image,
    execute_code,
]


def init_tools() -> List[Callable]:
    """Return a list of all tool functions."""
    return tools
