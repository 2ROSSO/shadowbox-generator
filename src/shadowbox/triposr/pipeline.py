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
from shadowbox.core.depth_to_mesh import DepthToMeshInput, DepthToMeshProcessor
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

    Attributes:
        cropped_image: クロップされたイラスト領域（NumPy配列）。
        depth_map: 復元された深度マップ（共通パス使用時）。
        labels: 各ピクセルのレイヤーインデックス（共通パス使用時）。
        centroids: 各レイヤーの深度セントロイド（共通パス使用時）。
        optimal_k: 使用されたレイヤー数（共通パス使用時）。
    """

    cropped_image: NDArray[np.uint8] | None = None
    depth_map: NDArray[np.float32] | None = None
    labels: NDArray[np.int32] | None = None
    centroids: NDArray[np.float32] | None = None
    optimal_k: int | None = None


class TripoSRPipeline:
    """TripoSRによる3Dメッシュ生成パイプライン。

    ShadowboxPipelineと同様のインターフェースを持ちますが、
    深度推定+クラスタリングの代わりにTripoSRで直接3Dメッシュを生成します。

    共通パス（デフォルト）では、TripoSRメッシュから深度マップを復元し、
    DepthToMeshProcessor経由でDepthモードと同じ後処理を適用します。

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
        depth_to_mesh: DepthToMeshProcessor | None = None,
    ) -> None:
        """パイプラインを初期化。

        Args:
            settings: TripoSR設定。
            render_settings: レンダリング設定（フレーム生成に使用）。
            depth_to_mesh: 深度→メッシュ共通処理器（共通パス使用時）。
        """
        self._settings = settings
        self._render_settings = render_settings or RenderSettings()
        self._generator = TripoSRGenerator(settings)
        self._depth_to_mesh = depth_to_mesh

    def process(
        self,
        image: str | Path | Image.Image | NDArray,
        bbox: BoundingBox | None = None,
        include_frame: bool = True,
        split_by_depth: bool = False,
        num_layers: int | None = None,
        k: int | None = None,
        use_raw_depth: bool = False,
        depth_scale: float = 1.0,
        max_resolution: int | None = None,
        include_card_frame: bool = False,
    ) -> TripoSRPipelineResult:
        """画像を処理して3Dメッシュを生成。

        Args:
            image: 入力画像（パス、PIL Image、またはNumPy配列）。
            bbox: イラスト領域のバウンディングボックス（Noneの場合は画像全体）。
            include_frame: フレームを含めるかどうか。
            split_by_depth: ネイティブメッシュ分割パス（レガシー）。
                Trueの場合、TripoSRメッシュの三角形面を深度で分割。
            num_layers: レイヤー数（split_by_depth=Trueの場合に使用）。
            k: レイヤー数（共通パス。Noneの場合は自動探索）。
            use_raw_depth: 生の深度値を使用するかどうか（共通パス）。
            depth_scale: 生深度モード時の深度スケール（共通パス）。
            max_resolution: 最大解像度。指定すると画像をダウンサンプリング。
            include_card_frame: カードのフレーム部分を含めるかどうか（共通パス）。

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

        # ダウンサンプリング（max_resolution指定時）
        if max_resolution is not None:
            cropped_image = self._downsample_if_needed(cropped_image, max_resolution)

        # TripoSRで3Dメッシュを生成
        print("TripoSRで3Dメッシュを生成中...")
        triposr_mesh = self._generator.generate(cropped_image)
        print("3Dメッシュ生成完了")

        # split_by_depth=True: レガシーのネイティブメッシュ分割パス
        if split_by_depth:
            return self._process_legacy_split(
                triposr_mesh, cropped_image, original_array, bbox,
                include_frame, num_layers,
            )

        # 共通パス: 深度復元 → DepthToMeshProcessor
        if self._depth_to_mesh is not None:
            return self._process_shared_path(
                triposr_mesh, cropped_image, original_array, bbox,
                include_frame, k, use_raw_depth, depth_scale, include_card_frame,
            )

        # depth_to_meshが未設定の場合はレガシーパスにフォールバック
        return self._process_legacy_direct(
            triposr_mesh, cropped_image, original_array, bbox, include_frame,
        )

    def _process_shared_path(
        self,
        triposr_mesh: ShadowboxMesh,
        cropped_image: Image.Image,
        original_array: NDArray[np.uint8],
        bbox: BoundingBox | None,
        include_frame: bool,
        k: int | None,
        use_raw_depth: bool,
        depth_scale: float,
        include_card_frame: bool,
    ) -> TripoSRPipelineResult:
        """共通パス: 深度復元 → DepthToMeshProcessor でメッシュ生成。"""
        # メッシュから深度マップを復元
        print("メッシュから深度マップを復元中...")
        depth_map = self._recover_depth_from_mesh(triposr_mesh)
        print(f"深度マップ復元完了: {depth_map.shape}")

        cropped_array = image_to_array(cropped_image)

        # クロップ画像を深度マップに合わせてリサイズ
        depth_h, depth_w = depth_map.shape
        img_h, img_w = cropped_array.shape[:2]
        if (img_h, img_w) != (depth_h, depth_w):
            resized = Image.fromarray(cropped_array).resize(
                (depth_w, depth_h), Image.Resampling.LANCZOS
            )
            cropped_array = image_to_array(resized)

        # カードフレーム統合: 深度マップを元画像サイズに埋め込む
        if include_card_frame and bbox is not None:
            # 深度マップを bbox サイズにリサイズ
            depth_uint8 = (depth_map * 255).astype(np.uint8)
            depth_resized = np.array(
                Image.fromarray(depth_uint8).resize(
                    (bbox.width, bbox.height), Image.Resampling.LANCZOS
                )
            ).astype(np.float32) / 255.0

            # 元画像サイズのゼロ深度マップを作成（フレーム部分は0.0=最前面）
            full_h, full_w = original_array.shape[:2]
            full_depth = np.zeros((full_h, full_w), dtype=np.float32)
            full_depth[
                bbox.y : bbox.y + bbox.height,
                bbox.x : bbox.x + bbox.width,
            ] = depth_resized

            depth_map = full_depth
            cropped_array = original_array

        # DepthToMeshProcessor に合流
        input_data = DepthToMeshInput(
            cropped_image=cropped_array,
            depth_map=depth_map,
            original_image=original_array if include_card_frame else None,
            bbox=bbox if include_card_frame else None,
        )
        mesh_result = self._depth_to_mesh.process(
            input_data,
            k=k,
            include_frame=include_frame,
            include_card_frame=include_card_frame,
            use_raw_depth=use_raw_depth,
            depth_scale=depth_scale,
        )

        return TripoSRPipelineResult(
            original_image=original_array,
            mesh=mesh_result.mesh,
            bbox=bbox,
            cropped_image=cropped_array,
            depth_map=depth_map,
            labels=mesh_result.labels,
            centroids=mesh_result.centroids,
            optimal_k=mesh_result.optimal_k,
        )

    def _process_legacy_split(
        self,
        triposr_mesh: ShadowboxMesh,
        cropped_image: Image.Image,
        original_array: NDArray[np.uint8],
        bbox: BoundingBox | None,
        include_frame: bool,
        num_layers: int | None,
    ) -> TripoSRPipelineResult:
        """レガシーパス: ネイティブメッシュ三角形面分割。"""
        print("深度ベースでメッシュを分割中...")
        mesh = self._split_mesh_by_depth(triposr_mesh, num_layers)
        print(f"メッシュ分割完了: {mesh.num_layers}レイヤー")

        if self._render_settings.back_panel:
            mesh = self._add_back_panel_to_mesh(mesh, cropped_image)

        if include_frame:
            mesh = self._add_frame_to_mesh(mesh)

        return TripoSRPipelineResult(
            original_image=original_array,
            mesh=mesh,
            bbox=bbox,
        )

    def _process_legacy_direct(
        self,
        triposr_mesh: ShadowboxMesh,
        cropped_image: Image.Image,
        original_array: NDArray[np.uint8],
        bbox: BoundingBox | None,
        include_frame: bool,
    ) -> TripoSRPipelineResult:
        """レガシーパス: 直接メッシュ（分割なし）。"""
        mesh = triposr_mesh

        if self._render_settings.back_panel:
            mesh = self._add_back_panel_to_mesh(mesh, cropped_image)

        if include_frame:
            mesh = self._add_frame_to_mesh(mesh)

        return TripoSRPipelineResult(
            original_image=original_array,
            mesh=mesh,
            bbox=bbox,
        )

    def _recover_depth_from_mesh(
        self,
        mesh: ShadowboxMesh,
    ) -> NDArray[np.float32]:
        """TripoSRメッシュから深度マップを復元。

        Args:
            mesh: TripoSRで生成されたメッシュ（単一レイヤー）。

        Returns:
            正規化された深度マップ (H, W), [0, 1]。
        """
        layer = mesh.layers[0]

        if layer.faces is None:
            raise ValueError("メッシュに面情報がありません。深度復元できません。")

        depth_settings = DepthRecoverySettings(
            resolution=self._settings.depth_resolution,
            fill_holes=self._settings.depth_fill_holes,
            hole_fill_method=self._settings.depth_fill_method,
        )

        depth_extractor = create_depth_extractor(use_pyrender=True)

        raw_depth = depth_extractor.extract_depth(
            layer.vertices,
            layer.faces,
            depth_settings.resolution,
        )

        # 穴埋め処理
        if depth_settings.fill_holes:
            from shadowbox.triposr.depth_recovery import fill_depth_holes

            raw_depth = fill_depth_holes(
                raw_depth, method=depth_settings.hole_fill_method
            )

        # 正規化 [0, 1]
        valid_mask = raw_depth > 0
        if valid_mask.any():
            d_min = raw_depth[valid_mask].min()
            d_max = raw_depth[valid_mask].max()
            if d_max > d_min:
                depth_map = np.clip((raw_depth - d_min) / (d_max - d_min), 0, 1).astype(
                    np.float32
                )
            else:
                depth_map = np.zeros_like(raw_depth)
        else:
            depth_map = np.zeros_like(raw_depth)

        return depth_map

    def _downsample_if_needed(
        self,
        image: Image.Image,
        max_resolution: int,
    ) -> Image.Image:
        """必要に応じて画像をダウンサンプリング。

        Args:
            image: 入力画像。
            max_resolution: 最大解像度。

        Returns:
            ダウンサンプリング済み画像。
        """
        w, h = image.size
        if max(w, h) <= max_resolution:
            return image

        scale = max_resolution / max(w, h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        return image.resize((new_w, new_h), Image.Resampling.LANCZOS)

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
        if len(mesh.layers) != 1:
            print(f"警告: メッシュが既に{len(mesh.layers)}レイヤーに分割されています")
            return mesh

        layer = mesh.layers[0]

        if layer.faces is None:
            print("警告: メッシュに面情報がありません。分割をスキップします。")
            return mesh

        depth_settings = DepthRecoverySettings(
            resolution=self._settings.depth_resolution,
            fill_holes=self._settings.depth_fill_holes,
            hole_fill_method=self._settings.depth_fill_method,
        )

        depth_extractor = create_depth_extractor(use_pyrender=True)
        clusterer = KMeansLayerClusterer(ClusteringSettings())

        splitter = DepthBasedMeshSplitter(
            depth_extractor=depth_extractor,
            clusterer=clusterer,
            settings=depth_settings,
        )

        split_result = splitter.split(
            vertices=layer.vertices,
            faces=layer.faces,
            colors=layer.colors,
            k=num_layers,
            face_assignment_method=self._settings.face_assignment_method,
        )

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

        back_panel = create_back_panel(
            image_array,
            z=z_min - 0.01,
            layer_index=len(mesh.layers),
        )

        new_layers = list(mesh.layers) + [back_panel]
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
        z_min = mesh.bounds[4]  # min_z
        z_max = mesh.bounds[5]  # max_z

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
