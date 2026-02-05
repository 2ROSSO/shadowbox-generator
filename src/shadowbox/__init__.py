"""
Shadowbox Generator - Transform TCG cards into 3D shadowbox displays.

This package provides tools to:
1. Load card images from URLs or local directories
2. Detect or manually select illustration areas
3. Estimate depth using AI models (Depth Anything v2, etc.)
4. Cluster depth values into discrete layers
5. Generate 3D mesh data
6. Render interactive 3D shadowbox visualization
"""

__version__ = "0.1.0"

# Lazy imports to avoid loading heavy dependencies at import time
__all__ = [
    "__version__",
    "create_pipeline",
    "ShadowboxSettings",
    "CardTemplate",
    "BoundingBox",
    "ShadowboxPipeline",
    "PipelineResult",
    "BasePipelineResult",
]


def __getattr__(name: str):
    """Lazy import for heavy modules."""
    if name == "create_pipeline":
        from shadowbox.core.pipeline import create_pipeline
        return create_pipeline
    if name == "ShadowboxSettings":
        from shadowbox.config.settings import ShadowboxSettings
        return ShadowboxSettings
    if name == "CardTemplate":
        from shadowbox.config.template import CardTemplate
        return CardTemplate
    if name == "BoundingBox":
        from shadowbox.config.template import BoundingBox
        return BoundingBox
    if name == "ShadowboxPipeline":
        from shadowbox.core.pipeline import ShadowboxPipeline
        return ShadowboxPipeline
    if name == "PipelineResult":
        from shadowbox.core.pipeline import PipelineResult
        return PipelineResult
    if name == "BasePipelineResult":
        from shadowbox.core.pipeline import BasePipelineResult
        return BasePipelineResult
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
