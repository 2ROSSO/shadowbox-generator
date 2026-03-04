"""背面パネル生成ファクトリモジュール。

シャドーボックスの背面に配置する画像パネルを生成します。
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from shadowbox.core.mesh import LayerMesh


def create_back_panel(
    image: NDArray[np.uint8],
    z: float,
    layer_index: int = 0,
    aspect_scale: tuple[float, float] = (1.0, 1.0),
) -> LayerMesh:
    """背面パネルを生成。

    画像全体のピクセルを含む平面レイヤーを生成します。
    シャドーボックスの背景として使用されます。

    Args:
        image: RGB画像。shape (H, W, 3)。
        z: パネルのZ座標。
        layer_index: レイヤーインデックス。
        aspect_scale: アスペクト比スケール (scale_x, scale_y)。

    Returns:
        LayerMeshオブジェクト（全ピクセル）。
    """
    h, w = image.shape[:2]
    sx, sy = aspect_scale

    # 全ピクセルの座標を取得
    y_coords, x_coords = np.mgrid[0:h, 0:w]
    y_coords = y_coords.flatten()
    x_coords = x_coords.flatten()

    # 座標を[-scale, scale]の範囲に正規化（アスペクト比保持）
    n_pixels = len(x_coords)
    if w > 1:
        vertices_x = ((x_coords / (w - 1)) * 2 - 1) * sx
    else:
        vertices_x = np.zeros(n_pixels, dtype=np.float64)
    if h > 1:
        vertices_y = -((y_coords / (h - 1)) * 2 - 1) * sy
    else:
        vertices_y = np.zeros(n_pixels, dtype=np.float64)
    vertices_z = np.full(n_pixels, z, dtype=np.float64)

    vertices = np.stack([vertices_x, vertices_y, vertices_z], axis=1).astype(np.float32)
    colors = image.reshape(-1, 3).astype(np.uint8)
    pixel_indices = np.stack([y_coords, x_coords], axis=1).astype(np.int32)

    return LayerMesh(
        vertices=vertices,
        colors=colors,
        z_position=z,
        layer_index=layer_index,
        pixel_indices=pixel_indices,
    )
