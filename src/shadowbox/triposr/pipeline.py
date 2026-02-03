"""TripoSRパイプラインモジュール。

TripoSRを使用した3Dメッシュ生成パイプラインを提供します。
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from shadowbox.config.settings import RenderSettings
from shadowbox.config.template import BoundingBox
from shadowbox.core.mesh import FrameMesh, ShadowboxMesh
from shadowbox.triposr.generator import TripoSRGenerator
from shadowbox.triposr.settings import TripoSRSettings
from shadowbox.utils.image import crop_image, image_to_array, load_image


@dataclass
class TripoSRPipelineResult:
    """TripoSRパイプラインの実行結果。

    Attributes:
        original_image: 元の入力画像（NumPy配列）。
        mesh: 生成された3Dメッシュ。
        bbox: 使用されたバウンディングボックス（クロップした場合）。
    """

    original_image: NDArray[np.uint8]
    mesh: ShadowboxMesh
    bbox: Optional[BoundingBox]


class TripoSRPipeline:
    """TripoSRによる3Dメッシュ生成パイプライン。

    ShadowboxPipelineと同様のインターフェースを持ちますが、
    深度推定+クラスタリングの代わりにTripoSRで直接3Dメッシュを生成します。

    Note:
        このクラスを使用するには、triposr依存関係が必要です:
        pip install shadowbox[triposr]

    Example:
        >>> from shadowbox.triposr import TripoSRPipeline, TripoSRSettings
        >>> settings = TripoSRSettings()
        >>> pipeline = TripoSRPipeline(settings)
        >>> result = pipeline.process(image)
    """

    def __init__(
        self,
        settings: TripoSRSettings,
        render_settings: Optional[RenderSettings] = None,
    ) -> None:
        """パイプラインを初期化。

        Args:
            settings: TripoSR設定。
            render_settings: レンダリング設定（フレーム生成に使用）。
        """
        self._settings = settings
        self._render_settings = render_settings or RenderSettings()
        self._generator = TripoSRGenerator(settings)

    def process(
        self,
        image: Union[str, Path, Image.Image, NDArray],
        bbox: Optional[BoundingBox] = None,
        include_frame: bool = True,
    ) -> TripoSRPipelineResult:
        """画像を処理して3Dメッシュを生成。

        Args:
            image: 入力画像（パス、PIL Image、またはNumPy配列）。
            bbox: イラスト領域のバウンディングボックス（Noneの場合は画像全体）。
            include_frame: フレームを含めるかどうか。

        Returns:
            TripoSRPipelineResult: 生成結果。
        """
        # 画像をロード
        if isinstance(image, (str, Path)):
            pil_image = load_image(image)
        elif isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        else:
            pil_image = image

        original_array = image_to_array(pil_image)

        # バウンディングボックスでクロップ
        if bbox is not None:
            cropped_image = crop_image(pil_image, bbox.x, bbox.y, bbox.width, bbox.height)
        else:
            cropped_image = pil_image

        # TripoSRで3Dメッシュを生成
        print("TripoSRで3Dメッシュを生成中...")
        mesh = self._generator.generate(cropped_image)
        print("3Dメッシュ生成完了")

        # フレームを追加
        if include_frame:
            mesh = self._add_frame_to_mesh(mesh)

        return TripoSRPipelineResult(
            original_image=original_array,
            mesh=mesh,
            bbox=bbox,
        )

    def _add_frame_to_mesh(self, mesh: ShadowboxMesh) -> ShadowboxMesh:
        """メッシュにフレームを追加。

        Args:
            mesh: TripoSRで生成されたメッシュ。

        Returns:
            フレームを追加したShadowboxMesh。
        """
        # メッシュのバウンズからZ範囲を取得
        z_min = mesh.bounds[4]  # min_z
        z_max = mesh.bounds[5]  # max_z

        # フレームを生成
        frame = self._create_frame(z_min, z_max)

        # バウンズを再計算（フレームを含む）
        bounds = self._calculate_bounds_with_frame(mesh, frame)

        return ShadowboxMesh(
            layers=mesh.layers,
            frame=frame,
            bounds=bounds,
        )

    def _create_frame(self, z_back: float, z_front: float) -> FrameMesh:
        """壁付きフレームを生成。

        MeshGeneratorの_create_frame_mesh_with_wallsと同様の構造を持つ
        12頂点・16面のフレームを生成します。

        Args:
            z_back: 背面のZ座標。
            z_front: 前面のZ座標。

        Returns:
            FrameMeshオブジェクト。
        """
        margin = 0.05
        outer = 1.0 + margin
        inner = 1.0

        # 12頂点: 前面外側4 + 前面内側4 + 背面外側4
        vertices = np.array([
            # 前面外側 (0-3)
            [-outer, -outer, z_front],
            [+outer, -outer, z_front],
            [+outer, +outer, z_front],
            [-outer, +outer, z_front],
            # 前面内側 (4-7)
            [-inner, -inner, z_front],
            [+inner, -inner, z_front],
            [+inner, +inner, z_front],
            [-inner, +inner, z_front],
            # 背面外側 (8-11)
            [-outer, -outer, z_back],
            [+outer, -outer, z_back],
            [+outer, +outer, z_back],
            [-outer, +outer, z_back],
        ], dtype=np.float32)

        # 面の構成: 前面枠8三角形 + 外壁8三角形 = 16三角形
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

        # フレームの色（暗い色）
        color = np.array([30, 30, 30], dtype=np.uint8)

        return FrameMesh(
            vertices=vertices,
            faces=faces,
            color=color,
            z_position=z_front,
            z_back=z_back,
            has_walls=True,
        )

    def _calculate_bounds_with_frame(
        self,
        mesh: ShadowboxMesh,
        frame: FrameMesh,
    ) -> tuple:
        """フレームを含むバウンズを計算。

        Args:
            mesh: メッシュ。
            frame: フレームメッシュ。

        Returns:
            (min_x, max_x, min_y, max_y, min_z, max_z)のタプル。
        """
        all_vertices = []

        for layer in mesh.layers:
            if len(layer.vertices) > 0:
                all_vertices.append(layer.vertices)

        all_vertices.append(frame.vertices)

        combined = np.vstack(all_vertices)

        return (
            float(combined[:, 0].min()),
            float(combined[:, 0].max()),
            float(combined[:, 1].min()),
            float(combined[:, 1].max()),
            float(combined[:, 2].min()),
            float(combined[:, 2].max()),
        )

    @property
    def settings(self) -> TripoSRSettings:
        """現在の設定を取得。"""
        return self._settings

    @property
    def render_settings(self) -> RenderSettings:
        """レンダリング設定を取得。"""
        return self._render_settings
