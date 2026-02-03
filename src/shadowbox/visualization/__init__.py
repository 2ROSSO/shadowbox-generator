"""可視化モジュール。

2D可視化（ヒートマップ、レイヤープレビュー）と
3Dレンダリング機能を提供します。
"""

from shadowbox.visualization.heatmap import (
    create_depth_contour,
    create_depth_heatmap,
    create_depth_histogram,
    create_depth_overlay,
)
from shadowbox.visualization.layers import (
    create_depth_layer_comparison,
    create_labeled_image,
    create_layer_mask_preview,
    create_layer_preview,
    create_stacked_layer_view,
    show_clustering_summary,
)
from shadowbox.visualization.render import (
    RenderOptions,
    ShadowboxRenderer,
    render_layers_exploded,
    render_shadowbox,
)
from shadowbox.visualization.export import (
    export_to_stl,
    export_to_obj,
    export_to_ply,
)

__all__ = [
    # heatmap
    "create_depth_heatmap",
    "create_depth_overlay",
    "create_depth_histogram",
    "create_depth_contour",
    # layers
    "create_layer_preview",
    "create_layer_mask_preview",
    "create_labeled_image",
    "create_depth_layer_comparison",
    "create_stacked_layer_view",
    "show_clustering_summary",
    # render
    "RenderOptions",
    "ShadowboxRenderer",
    "render_shadowbox",
    "render_layers_exploded",
    # export
    "export_to_stl",
    "export_to_obj",
    "export_to_ply",
]
