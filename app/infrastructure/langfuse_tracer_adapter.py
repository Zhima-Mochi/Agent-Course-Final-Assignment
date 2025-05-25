import logging
import contextlib
from typing import Any, Dict, Optional, Generator, TypeVar, Callable
from langfuse import Langfuse
from langfuse.decorators import observe
from app.application.ports import TracerPort
from app.config import settings

logger = logging.getLogger(__name__)
T = TypeVar("T")


class LangfuseTracerAdapter(TracerPort):
    """Langfuse-based implementation of TracerPort"""

    def __init__(self):
        self._client = None
        if not (settings.LANGFUSE_PUBLIC_KEY and settings.LANGFUSE_SECRET_KEY):
            logger.warning("Langfuse credentials not provided. Tracing disabled.")
            return
        try:            
            self._client = Langfuse(
                public_key=settings.LANGFUSE_PUBLIC_KEY,
                secret_key=settings.LANGFUSE_SECRET_KEY,
                host=settings.LANGFUSE_HOST,
                debug=settings.LANGFUSE_DEBUG,
            )
            logger.info("Langfuse initialized")
        except Exception as e:
            logger.warning(f"Langfuse init failed: {e}")

    @property
    def enabled(self) -> bool:
        return self._client is not None

    @contextlib.contextmanager
    def trace(self, name: str, metadata: Optional[Dict[str, Any]] = None) -> Generator[None, None, None]:
        if not self.enabled:
            yield
            return

        trace = self._client.trace(name=name, metadata=metadata or {})
        try:
            yield trace
        except Exception as e:
            trace.update(metadata={"error": str(e)})
            raise
        finally:
            trace.update(metadata={"status": "finished"})

    @contextlib.contextmanager
    def span(self, name: str, metadata: Optional[Dict[str, Any]] = None) -> Generator[None, None, None]:
        if not self.enabled:
            yield
            return

        span = self._client.span(name=name, metadata=metadata or {})
        try:
            yield span
        except Exception as e:
            span.update(metadata={"error": str(e)})
            raise
        finally:
            span.end()
