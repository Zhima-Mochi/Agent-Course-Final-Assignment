import os
import pandas as pd
from typing import Any

def parse_uploaded_media(filename: str, file_path: str) -> Any:
    ext = os.path.splitext(filename)[1].lower()
    try:
        if ext == '.mp3':
            import whisper
            model = whisper.load_model('base')
            result = model.transcribe(file_path)
            return result['text']
        elif ext in ['.png', '.jpg', '.jpeg']:
            try:
                import pytesseract
                from PIL import Image
            except ImportError:
                raise RuntimeError('OCR dependencies not installed')
            img = Image.open(file_path)
            return pytesseract.image_to_string(img)
        elif ext == '.xlsx':
            return pd.read_excel(file_path)
        else:
            return None
    except Exception as e:
        return f"MEDIA_PARSE_ERROR: {e}" 