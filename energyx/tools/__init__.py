"""EnergyX tools: 48 runtime tools plus 4 dataset-onboarding tools."""
from .base import Status, ToolResult  # noqa: F401
from .artifacts import ArtifactStore  # noqa: F401
from .registry import (  # noqa: F401
    REGISTRY, ToolContext, ToolSpec, call, counts, for_agent, get,
    runtime_tools, tool, visible_to,
)


_LOADED = False


def load_all() -> None:
    """Import every tool module so the registry is populated. Idempotent."""
    global _LOADED
    if _LOADED:
        return
    from . import (  # noqa: F401
        ai_analytics, consumption, discovery, knowledge, onboarding,
        performance, quality, reporting, structured, weather_spatial,
    )
    _LOADED = True


def ensure_loaded() -> None:
    """Load tools if nothing has yet. Safe to call from anywhere."""
    if not REGISTRY:
        load_all()
