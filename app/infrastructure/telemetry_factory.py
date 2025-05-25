import logging
from typing import Optional, Any, Dict
from app.application.ports import TracerPort
from app.config import settings
from app.infrastructure.langfuse_tracer_adapter import LangfuseTracerAdapter
from contextlib import contextmanager
from typing import Dict, Any, Optional, Callable

logger = logging.getLogger(__name__)

# Singleton pattern for tracer instance
_tracer_instance: Optional[TracerPort] = None

def get_tracer_adapter() -> TracerPort:
    """
    Factory function to get the appropriate tracer implementation.
    Uses a singleton pattern to ensure only one tracer instance exists.
    
    Returns:
        TracerPort: A tracer implementation that conforms to the TracerPort interface
    """
    global _tracer_instance
    
    if _tracer_instance is not None:
        return _tracer_instance
    
    # Determine which tracer to use based on configuration
    tracer_type = settings.TELEMETRY_PROVIDER.lower() if hasattr(settings, 'TELEMETRY_PROVIDER') else "none"
    
    if tracer_type == "langfuse":
        logger.info("Initializing Langfuse tracer")
        _tracer_instance = LangfuseTracerAdapter()
    else:
        logger.warning(f"Unknown telemetry provider: {tracer_type}, using NullTracerAdapter")
        _tracer_instance = NullTracerAdapter()
    
    return _tracer_instance 


class NullTracerAdapter(TracerPort):
    """
    Null implementation of TracerPort that does nothing.
    Used as a no-op tracer.
    """

    @contextmanager
    def trace(self, name: str, metadata: Optional[Dict[str, Any]] = None):
        """Do-nothing implementation of trace."""
        yield

    @contextmanager
    def span(self, name: str, metadata: Optional[Dict[str, Any]] = None):
        """Do-nothing implementation of span."""
        yield

    def observe_decorator(self, name: Optional[str] = None, 
                          metadata: Optional[Dict[str, Any]] = None) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        logger.debug(f"NullTracerAdapter: observe_decorator called for '{name}', returning no-op.")
        def no_op_decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            return func
        return no_op_decorator


