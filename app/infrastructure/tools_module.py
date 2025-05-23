from duckduckgo_search import DDGS
import fitz  # PyMuPDF
import os
import pandas as pd
import math
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
import tempfile
import requests
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
    """Search the web using DuckDuckGo, fallback to SearXNG if it fails."""
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=num_results))
            time.sleep(1.5)  # throttle
        return {"web_results": results}
    except Exception as e_ddg:
        logger.warning(f"DuckDuckGo failed: {str(e_ddg)} — falling back to SearXNG.")
        try:
            params = {
                "q": query,
                "format": "json",
                "language": "en",
                "count": num_results
            }
            response = requests.get("https://searx.be/search", params=params)
            response.raise_for_status()
            results = response.json().get("results", [])
            return {"web_results": results}
        except Exception as e_searx:
            logger.error(f"SearXNG fallback failed: {str(e_searx)}")
            return {
                "error": f"Both DuckDuckGo and SearXNG failed.\nDDG: {e_ddg}\nSearXNG: {e_searx}",
                "note": "If hit rate limit or connection error, try again later or switch tools."
            }


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


def download_file(url: str) -> Dict[str, Any]:
    """Download a file from a URL and save it to a temporary location.
    
    Args:
        url: The URL of the file to download
        
    Returns:
        A dictionary with the local file path, content type, and other metadata or an error message
    """
    try:
        # Download the file with progress handling for large files
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        # Get content type and determine appropriate file extension
        content_type = response.headers.get('content-type', '').lower()
        
        # Determine file extension based on content type
        extension_map = {
            'text/csv': '.csv',
            'application/vnd.ms-excel': '.xls',
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': '.xlsx',
            'application/pdf': '.pdf',
            'text/html': '.html',
            'text/plain': '.txt',
            'application/json': '.json',
            'image/jpeg': '.jpg',
            'image/png': '.png'
        }
        
        # Try to get extension from URL first, fall back to content-type
        url_extension = os.path.splitext(url)[1]
        if url_extension and len(url_extension) < 10:  # Sanity check on extension length
            file_extension = url_extension
        else:
            file_extension = extension_map.get(content_type, '.tmp')
            
        # Create a temporary file with the appropriate extension
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
            temp_path = tmp_file.name
        
        total_size = int(response.headers.get('content-length', 0))
        logger.info(f"Downloading file from {url} ({total_size} bytes), content-type: {content_type}")
        
        with open(temp_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        logger.info(f"File downloaded successfully to {temp_path}")
        return {
            "file_path": temp_path,
            "content_type": content_type,
            "url": url,
            "file_extension": file_extension
        }
        
    except Exception as e:
        logger.error(f"Error downloading file: {str(e)}")
        return {"error": f"Failed to download file: {str(e)}"}




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
    """Download YouTube audio and transcribe using Whisper
    
    This function attempts to transcribe YouTube audio using multiple whisper implementations:
    1. First tries openai-whisper (original implementation)
    2. Then tries faster-whisper (alternative implementation with better performance)
    3. Finally tries transformers pipeline with whisper model
    
    Args:
        url: YouTube URL to download and transcribe
        
    Returns:
        Dictionary with transcription text or error message
    """
    import tempfile
    import yt_dlp

    try:
        # Download the YouTube audio
        with tempfile.TemporaryDirectory(delete=False) as tmpdir:
            ydl_opts = {
                "format": "bestaudio/best",
                "outtmpl": tmpdir + "/audio",
                "quiet": True,
                "postprocessors": [
                    {
                        "key": "FFmpegExtractAudio",
                        "preferredcodec": "mp3",
                        "preferredquality": "192",
                    }
                ],
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            import whisper

            model = whisper.load_model("base")
            result = model.transcribe(os.path.join(tmpdir, "audio.mp3"))
            return {"transcription": result.get("text", "")}
    except Exception as e:
        logger.error(f"YouTube audio transcription failed: {str(e)}")
        return {"error": f"Transcription failed: {str(e)}"}



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
def execute_code(code: str) -> Dict[str, Any]:
    """
    Execute Python code provided by the LLM and return the result.
    
    Args:
        code (str): Python code to execute. The code should either:
            1. Print results using print() statements, which will be captured in stdout, or
            2. Assign the final result to a variable named 'result', which will be returned
            3. Return a value directly from the function
        
    Returns:
        Dict containing stdout, stderr, and return_value (if any)
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
                    "return_value": execution_result,
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


def resolve_path(file_path: str) -> Tuple[str, str, str]:
    """Resolve a file path to a full path, extracting extension and content type.
    
    Args:
        file_path: Local path or URL to resolve
        
    Returns:
        Tuple of (full_path, file_extension, content_type)
        
    Raises:
        ValueError: If the file cannot be downloaded or found
    """
    if file_path.startswith("http"):
        result = download_file(file_path)
        if "error" in result:
            raise ValueError(result["error"])
        return result["file_path"], result.get("file_extension", "").lower(), result.get("content_type", "").lower()
    
    full_path = os.path.join("task_files", file_path) if not file_path.startswith("task_files") else file_path
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"File not found: {full_path}")
    
    return full_path, os.path.splitext(full_path)[1].lower(), ""


def detect_file_type(extension: str, content_type: str) -> str:
    """Detect file type based on extension and content type.
    
    Args:
        extension: File extension including the dot (e.g., '.pdf')
        content_type: MIME type from HTTP headers
        
    Returns:
        Detected file type as string
    """
    if extension in [".pdf"] or "application/pdf" in content_type:
        return "pdf"
    elif extension in [".xlsx", ".xls", ".xlsm"] or "excel" in content_type or "spreadsheet" in content_type:
        return "excel"
    elif extension in [".csv", ".tsv"] or "csv" in content_type or "text/plain" in content_type:
        return "csv"
    elif extension in [".txt", ".md", ".json", ".xml", ".html", ".htm"] or "text/" in content_type:
        return "text"
    return "unknown"


def process_pdf(path: str) -> Dict[str, Any]:
    """Process a PDF file and extract its text content.
    
    Args:
        path: Path to the PDF file
        
    Returns:
        Dictionary with file content and metadata
    """
    try:
        doc = fitz.open(path)
        text = "\n\n".join(f"--- Page {i+1} ---\n{page.get_text()}" for i, page in enumerate(doc))
        return {"file_content": text, "file_type": "pdf", "file_path": path}
    except Exception as e:
        logger.error(f"Error reading PDF: {str(e)}")
        return {"error": f"Failed to read PDF: {str(e)}"}


def summarize_tabular(df: pd.DataFrame, file_type: str, path: str, query: str = "") -> Dict[str, Any]:
    """Create a summary of a tabular dataframe.
    
    Args:
        df: Pandas DataFrame to summarize
        file_type: Type of file (csv, excel_sheet)
        path: Path to the source file
        query: Optional query about the data
        
    Returns:
        Dictionary with summary information
    """
    summary = {
        "columns": list(df.columns),
        "shape": df.shape,
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "head": df.head(5).to_dict(orient="records"),
        "describe": df.describe().to_dict(),
    }
    if query:
        summary["query"] = query
    return {"file_content": json.dumps(summary, indent=2), "file_type": file_type, "file_path": path}


def process_excel(path: str, sheet_name: str = None, query: str = "") -> Dict[str, Any]:
    """Process an Excel file, optionally focusing on a specific sheet.
    
    Args:
        path: Path to the Excel file
        sheet_name: Optional name of sheet to analyze
        query: Optional query about the data
        
    Returns:
        Dictionary with file content and metadata
    """
    try:
        if sheet_name:
            df = pd.read_excel(path, sheet_name=sheet_name)
            return summarize_tabular(df, "excel_sheet", path, query)
        
        # Read all sheets
        xls = pd.ExcelFile(path)
        sheet_names = xls.sheet_names
        all_data = {}
        for sheet in sheet_names:
            all_data[sheet] = pd.read_excel(path, sheet_name=sheet).head(5).to_dict(orient="records")
        
        return {"file_content": all_data, "sheet_names": sheet_names, "file_type": "excel_workbook", "file_path": path}
    except Exception as e:
        logger.error(f"Error reading Excel: {str(e)}")
        return {"error": f"Failed to read Excel: {str(e)}"}


def process_csv(path: str, query: str = "") -> Dict[str, Any]:
    """Process a CSV file with smart delimiter detection.
    
    Args:
        path: Path to the CSV file
        query: Optional query about the data
        
    Returns:
        Dictionary with file content and metadata
    """
    try:
        # Try standard CSV parsing first
        try:
            df = pd.read_csv(path)
        except Exception:
            # Try with different delimiter detection
            with open(path, 'r', encoding='utf-8', errors='replace') as f:
                sample = f.read(4096)
            
            # Count potential delimiters
            delimiters = {',': 0, '\t': 0, ';': 0, '|': 0}
            for d in delimiters:
                delimiters[d] = sample.count(d)
            
            # Use the most common delimiter
            best_delimiter = max(delimiters.items(), key=lambda x: x[1])[0]
            if best_delimiter != ',' and delimiters[best_delimiter] > 0:
                logger.info(f"Using delimiter: '{best_delimiter}'")
                df = pd.read_csv(path, sep=best_delimiter)
            else:
                # If still failing, try to read as Excel
                logger.info("Trying to read as Excel file")
                df = pd.read_excel(path)
        
        return summarize_tabular(df, "csv", path, query)
    except Exception as e:
        logger.error(f"Error reading CSV: {str(e)}")
        return {"error": f"Failed to read CSV: {str(e)}"}


def process_text_file(path: str) -> Dict[str, Any]:
    """Process a text file.
    
    Args:
        path: Path to the text file
        
    Returns:
        Dictionary with file content and metadata
    """
    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        return {"file_content": content, "file_type": "text", "file_path": path}
    except Exception as e:
        logger.error(f"Error reading text file: {str(e)}")
        return process_binary_file(path)


def process_binary_file(path: str) -> Dict[str, Any]:
    """Process a binary file by returning metadata.
    
    Args:
        path: Path to the binary file
        
    Returns:
        Dictionary with file info and metadata
    """
    try:
        file_size = os.path.getsize(path)
        return {
            "file_info": f"Binary file: {os.path.basename(path)}, Size: {file_size} bytes", 
            "file_type": "binary",
            "file_path": path
        }
    except Exception as e:
        logger.error(f"Error processing binary file: {str(e)}")
        return {"error": f"Failed to read file: {str(e)}"}


@tool
def read_file(file_path: str, query: str = "", sheet_name: str = None) -> Dict[str, Any]:
    """Read and analyze a file of various formats (PDF, CSV, Excel, etc.).
    Automatically detects the file type and processes it accordingly.
    
    Args:
        file_path: Path to the file (local path or URL)
        query: Optional query about the data (for tabular data analysis)
        sheet_name: Optional sheet name (for Excel files)
        
    Returns:
        Dictionary with file content or analysis results and file path
    """
    try:
        # Resolve path and detect file type
        full_path, file_extension, content_type = resolve_path(file_path)
        file_type = detect_file_type(file_extension, content_type)
        
        logger.info(f"Detected file type: {file_type} for {file_path}")
        
        # Map file types to their processors
        processors = {
            "pdf": process_pdf,
            "excel": lambda p: process_excel(p, sheet_name, query),
            "csv": lambda p: process_csv(p, query),
            "text": process_text_file,
            "unknown": process_text_file  # Try text first for unknown types
        }
        
        # Process the file with the appropriate processor
        processor = processors.get(file_type, process_text_file)
        return processor(full_path)
        
    except FileNotFoundError as e:
        return {"error": str(e)}
    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        return {"error": f"Failed to process file: {str(e)}"}


tools = [
    wiki_search,
    web_search,
    arxiv_search,
    read_file,  # Unified file reading tool
    get_youtube_transcript,
    get_youtube_video_frames,
    transcribe_audio,
    ocr_image,
    execute_code,
]


def init_tools() -> List[Callable]:
    """Return a list of all tool functions."""
    return tools
