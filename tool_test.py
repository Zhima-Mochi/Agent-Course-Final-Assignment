
from app.infrastructure.tools_module import get_youtube_video_frames
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print(get_youtube_video_frames.invoke("https://www.youtube.com/watch?v=L1vXCYZAYYM"))