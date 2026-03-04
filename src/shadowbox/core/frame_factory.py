"""フレーム生成ファクトリモジュール。

フレーム（枠）メッシュを生成するための共通ユーティリティを提供します。
MeshGenerator（depthモード）とTripoSRPipeline（triposrモード）の両方で使用されます。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from shadowbox.core.mesh import FrameMesh, LayerMesh


@dataclass(frozen=True)
class FrameConfig:
    """フレーム生成の設定。

    Attributes:
        z_front: フレーム前面のZ座標。
        z_back: フレーム背面のZ座標。Noneの場合は壁なしフレーム。
        margin: フレーム外側のマージン（イラスト領域からのはみ出し量）。
        frame_color: フレームの色（RGB、0-255）。
        aspect_scale: アスペクト比スケール (scale_x, scale_y)。
            長辺が1.0、短辺が(短辺/長辺)。
    """

    z_front: float = 0.0
    z_back: float | None = None
    margin: float = 0.05
    frame_color: tuple[int, int, int] = (30, 30, 30)
    aspect_scale: tuple[float, float] = (1.0, 1.0)


def create_frame(config: FrameConfig) -> FrameMesh:
    """フレームを生成（壁なし/壁付きを自動判定）。

    Args:
        config: フレーム生成設定。

    Returns:
        FrameMeshオブジェクト。z_backが指定されていれば壁付き、
        Noneなら壁なしフレームを生成。
    """
    if config.z_back is not None:
        return create_walled_frame(config)
    else:
        return create_plane_frame(config)


def create_plane_frame(config: FrameConfig) -> FrameMesh:
    """8頂点・8面の平面フレームを生成。

    イラストを囲む矩形フレーム（前面のみ、壁なし）を生成します。

    Args:
        config: フレーム生成設定。

    Returns:
        FrameMeshオブジェクト（has_walls=False）。
    """
    sx, sy = config.aspect_scale
    outer_x = (1.0 + config.margin) * sx
    outer_y = (1.0 + config.margin) * sy
    inner_x = 1.0 * sx
    inner_y = 1.0 * sy
    z = config.z_front

    # 8頂点: 外側4 + 内側4
    vertices = np.array([
        [-outer_x, -outer_y, z],  # 0: 外側左下
        [+outer_x, -outer_y, z],  # 1: 外側右下
        [+outer_x, +outer_y, z],  # 2: 外側右上
        [-outer_x, +outer_y, z],  # 3: 外側左上
        [-inner_x, -inner_y, z],  # 4: 内側左下
        [+inner_x, -inner_y, z],  # 5: 内側右下
        [+inner_x, +inner_y, z],  # 6: 内側右上
        [-inner_x, +inner_y, z],  # 7: 内側左上
    ], dtype=np.float32)

    # 8三角形: 下辺・右辺・上辺・左辺 各2三角形
    faces = np.array([
        [0, 1, 5], [0, 5, 4],  # 下辺
        [1, 2, 6], [1, 6, 5],  # 右辺
        [2, 3, 7], [2, 7, 6],  # 上辺
        [3, 0, 4], [3, 4, 7],  # 左辺
    ], dtype=np.int32)

    color = np.array(config.frame_color, dtype=np.uint8)

    return FrameMesh(
        vertices=vertices,
        faces=faces,
        color=color,
        z_position=config.z_front,
        z_back=None,
        has_walls=False,
    )


def create_walled_frame(config: FrameConfig) -> FrameMesh:
    """12頂点・16面の壁付きフレームを生成。

    本物のシャドーボックスのように、前面から背面まで繋がる
    3D壁を持つフレームを生成します。

    Args:
        config: フレーム生成設定。z_backは必須です。

    Returns:
        FrameMeshオブジェクト（has_walls=True）。

    Raises:
        ValueError: z_backがNoneの場合。
    """
    if config.z_back is None:
        raise ValueError("z_back must be specified for walled frame")

    sx, sy = config.aspect_scale
    outer_x = (1.0 + config.margin) * sx
    outer_y = (1.0 + config.margin) * sy
    inner_x = 1.0 * sx
    inner_y = 1.0 * sy
    z_front = config.z_front
    z_back = config.z_back

    # 12頂点: 前面外側4 + 前面内側4 + 背面外側4
    vertices = np.array([
        # 前面外側 (0-3)
        [-outer_x, -outer_y, z_front],
        [+outer_x, -outer_y, z_front],
        [+outer_x, +outer_y, z_front],
        [-outer_x, +outer_y, z_front],
        # 前面内側 (4-7)
        [-inner_x, -inner_y, z_front],
        [+inner_x, -inner_y, z_front],
        [+inner_x, +inner_y, z_front],
        [-inner_x, +inner_y, z_front],
        # 背面外側 (8-11)
        [-outer_x, -outer_y, z_back],
        [+outer_x, -outer_y, z_back],
        [+outer_x, +outer_y, z_back],
        [-outer_x, +outer_y, z_back],
    ], dtype=np.float32)

    # 16三角形: 前面枠8 + 外壁8
    faces = np.array([
        # 前面枠
        [0, 1, 5], [0, 5, 4],  # 下辺
        [1, 2, 6], [1, 6, 5],  # 右辺
        [2, 3, 7], [2, 7, 6],  # 上辺
        [3, 0, 4], [3, 4, 7],  # 左辺
        # 外壁（下/右/上/左）
        [0, 8, 9], [0, 9, 1],   # 下壁
        [1, 9, 10], [1, 10, 2],  # 右壁
        [2, 10, 11], [2, 11, 3],  # 上壁
        [3, 11, 8], [3, 8, 0],   # 左壁
    ], dtype=np.int32)

    color = np.array(config.frame_color, dtype=np.uint8)

    return FrameMesh(
        vertices=vertices,
        faces=faces,
        color=color,
        z_position=z_front,
        z_back=z_back,
        has_walls=True,
    )


def calculate_bounds(
    layers: list[LayerMesh],
    frame: FrameMesh | None = None,
) -> tuple[float, float, float, float, float, float]:
    """レイヤーとフレームからバウンディングボックスを計算。

    Args:
        layers: レイヤーメッシュのリスト。
        frame: フレームメッシュ（None可）。

    Returns:
        (min_x, max_x, min_y, max_y, min_z, max_z)のタプル。
        頂点がない場合はデフォルトのバウンズを返す。
    """
    all_vertices: list[NDArray[np.float32]] = []

    for layer in layers:
        if len(layer.vertices) > 0:
            all_vertices.append(layer.vertices)

    if frame is not None:
        all_vertices.append(frame.vertices)

    if not all_vertices:
        # 頂点がない場合はデフォルトのバウンズ
        return (-1.0, 1.0, -1.0, 1.0, -1.0, 0.0)

    combined = np.vstack(all_vertices)

    return (
        float(combined[:, 0].min()),
        float(combined[:, 0].max()),
        float(combined[:, 1].min()),
        float(combined[:, 1].max()),
        float(combined[:, 2].min()),
        float(combined[:, 2].max()),
    )
