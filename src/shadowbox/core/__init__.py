"""Core business logic modules."""

from shadowbox.core.back_panel_factory import create_back_panel
from shadowbox.core.frame_factory import (
    FrameConfig,
    calculate_bounds,
    create_frame,
    create_plane_frame,
    create_walled_frame,
)
from shadowbox.core.mesh import MeshGeneratorProtocol

__all__ = [
    "FrameConfig",
    "MeshGeneratorProtocol",
    "calculate_bounds",
    "create_back_panel",
    "create_frame",
    "create_plane_frame",
    "create_walled_frame",
]
