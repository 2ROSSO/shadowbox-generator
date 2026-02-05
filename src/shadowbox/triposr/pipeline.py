"""TripoSRパイプラインモジュール。

TripoSRを使用した3Dメッシュ生成パイプラインを提供します。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from shadowbox.config.settings import ClusteringSettings, RenderSettings
from shadowbox.config.template import BoundingBox
from shadowbox.core.back_panel_factory import create_back_panel
from shadowbox.core.clustering import KMeansLayerClusterer
from shadowbox.core.frame_factory import FrameConfig, calculate_bounds, create_frame
from shadowbox.core.mesh import ShadowboxMesh
from shadowbox.core.pipeline import BasePipelineResult
from shadowbox.triposr.depth_recovery import DepthRecoverySettings, create_depth_extractor
from shadowbox.triposr.generator import TripoSRGenerator
from shadowbox.triposr.mesh_splitter import DepthBasedMeshSplitter, create_split_shadowbox_mesh
from shadowbox.triposr.settings import TripoSRSettings
from shadowbox.utils.image import crop_image, image_to_array, load_image

if TYPE_CHECKING:
    pass


@dataclass
class TripoSRPipelineResult(BasePipelineResult):
    """TripoSRパイプラインの実行結果。

    すべてのフィールドは BasePipelineResult から継承。
    """

    pass  # 共通フィールドのみなので追加フィールドなし


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
        render_settings: RenderSettings | None = None,
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
        image: str | Path | Image.Image | NDArray,
        bbox: BoundingBox | None = None,
        include_frame: bool = True,
        split_by_depth: bool = False,
        num_layers: int | None = None,
    ) -> TripoSRPipelineResult:
        """画像を処理して3Dメッシュを生成。

        Args:
            image: 入力画像（パス、PIL Image、またはNumPy配列）。
            bbox: イラスト領域のバウンディングボックス（Noneの場合は画像全体）。
            include_frame: フレームを含めるかどうか。
            split_by_depth: 深度ベースでメッシュをレイヤーに分割するかどうか。
                Trueの場合、生成されたメッシュを画像平面に投影して深度マップを復元し、
                既存のクラスタリング処理を適用してレイヤー分割を行います。
            num_layers: レイヤー数（split_by_depth=Trueの場合に使用）。
                Noneの場合は自動決定。

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

        # 深度ベースの分割（オプション）
        if split_by_depth:
            print("深度ベースでメッシュを分割中...")
            mesh = self._split_mesh_by_depth(mesh, num_layers)
            print(f"メッシュ分割完了: {mesh.num_layers}レイヤー")

        # 背面パネルを追加（RenderSettings.back_panelを参照）
        if self._render_settings.back_panel:
            mesh = self._add_back_panel_to_mesh(mesh, cropped_image)

        # フレームを追加
        if include_frame:
            mesh = self._add_frame_to_mesh(mesh)

        return TripoSRPipelineResult(
            original_image=original_array,
            mesh=mesh,
            bbox=bbox,
        )

    def _split_mesh_by_depth(
        self,
        mesh: ShadowboxMesh,
        num_layers: int | None = None,
    ) -> ShadowboxMesh:
        """メッシュを深度ベースでレイヤーに分割。

        Args:
            mesh: TripoSRで生成されたメッシュ（単一レイヤー）。
            num_layers: レイヤー数。Noneの場合は自動決定。

        Returns:
            分割されたShadowboxMesh。
        """
        # 元のメッシュから頂点、面、色を取得
        # TripoSRメッシュは単一レイヤーのはず
        if len(mesh.layers) != 1:
            print(f"警告: メッシュが既に{len(mesh.layers)}レイヤーに分割されています")
            return mesh

        layer = mesh.layers[0]

        if layer.faces is None:
            print("警告: メッシュに面情報がありません。分割をスキップします。")
            return mesh

        # 深度復元設定を作成
        depth_settings = DepthRecoverySettings(
            resolution=self._settings.depth_resolution,
            fill_holes=self._settings.depth_fill_holes,
            hole_fill_method=self._settings.depth_fill_method,
        )

        # 深度抽出器とクラスタラーを作成
        depth_extractor = create_depth_extractor(use_pyrender=True)
        clusterer = KMeansLayerClusterer(ClusteringSettings())

        # メッシュ分割器を作成
        splitter = DepthBasedMeshSplitter(
            depth_extractor=depth_extractor,
            clusterer=clusterer,
            settings=depth_settings,
        )

        # 分割を実行
        split_result = splitter.split(
            vertices=layer.vertices,
            faces=layer.faces,
            colors=layer.colors,
            k=num_layers,
            face_assignment_method=self._settings.face_assignment_method,
        )

        # 分割結果からShadowboxMeshを作成
        return create_split_shadowbox_mesh(split_result, mesh.bounds)

    def _add_back_panel_to_mesh(
        self,
        mesh: ShadowboxMesh,
        image: Image.Image,
    ) -> ShadowboxMesh:
        """メッシュに背面パネルを追加。

        Args:
            mesh: TripoSRで生成されたメッシュ。
            image: クロップ済みの入力画像。

        Returns:
            背面パネルを追加したShadowboxMesh。
        """
        image_array = image_to_array(image)
        z_min = mesh.bounds[4]  # min_z

        # メッシュの最背面より少し奥に配置
        back_panel = create_back_panel(
            image_array,
            z=z_min - 0.01,  # 少し奥に
            layer_index=len(mesh.layers),
        )

        new_layers = list(mesh.layers) + [back_panel]

        # バウンズを再計算
        bounds = calculate_bounds(new_layers, mesh.frame)

        return ShadowboxMesh(
            layers=new_layers,
            frame=mesh.frame,
            bounds=bounds,
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

        # RenderSettingsのframe_wall_modeを活用
        if self._render_settings.frame_wall_mode == "none":
            config = FrameConfig(z_front=z_max)
        else:
            config = FrameConfig(z_front=z_max, z_back=z_min)

        frame = create_frame(config)
        bounds = calculate_bounds(mesh.layers, frame)

        return ShadowboxMesh(
            layers=mesh.layers,
            frame=frame,
            bounds=bounds,
        )

    @property
    def settings(self) -> TripoSRSettings:
        """現在の設定を取得。"""
        return self._settings

    @property
    def render_settings(self) -> RenderSettings:
        """レンダリング設定を取得。"""
        return self._render_settings
