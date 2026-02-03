"""シャドーボックス生成パイプラインモジュール。

このモジュールは、画像からシャドーボックスメッシュを生成する
一連の処理を統合したパイプラインを提供します。
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from shadowbox.config.loader import ConfigLoaderProtocol, YAMLConfigLoader
from shadowbox.config.settings import ShadowboxSettings
from shadowbox.config.template import BoundingBox
from shadowbox.core.clustering import KMeansLayerClusterer, LayerClustererProtocol
from shadowbox.core.depth import (
    DepthEstimatorProtocol,
    MockDepthEstimator,
    create_depth_estimator,
)
from shadowbox.core.mesh import MeshGenerator, ShadowboxMesh
from shadowbox.utils.image import crop_image, image_to_array, load_image


@dataclass
class PipelineResult:
    """パイプラインの実行結果を格納するデータクラス。

    Attributes:
        original_image: 元の入力画像（NumPy配列）。
        cropped_image: クロップされたイラスト領域（NumPy配列）。
        depth_map: 深度マップ（0=近い、1=遠い）。
        labels: 各ピクセルのレイヤーインデックス。
        centroids: 各レイヤーの深度セントロイド。
        mesh: 生成された3Dメッシュ。
        optimal_k: 使用されたレイヤー数。
        bbox: 使用されたバウンディングボックス（クロップした場合）。
    """

    original_image: NDArray[np.uint8]
    cropped_image: NDArray[np.uint8]
    depth_map: NDArray[np.float32]
    labels: NDArray[np.int32]
    centroids: NDArray[np.float32]
    mesh: ShadowboxMesh
    optimal_k: int
    bbox: BoundingBox | None


class ShadowboxPipeline:
    """シャドーボックス生成の統合パイプライン。

    画像の読み込みから3Dメッシュ生成までの全工程を管理します。
    依存性注入により、各コンポーネントを差し替え可能です。

    Attributes:
        depth_estimator: 深度推定器。
        clusterer: クラスタラー。
        mesh_generator: メッシュジェネレーター。
        config_loader: 設定ローダー。

    Example:
        >>> from shadowbox import create_pipeline, ShadowboxSettings
        >>> settings = ShadowboxSettings()
        >>> pipeline = create_pipeline(settings)
        >>> result = pipeline.process(image, auto_detect=True)
    """

    def __init__(
        self,
        depth_estimator: DepthEstimatorProtocol,
        clusterer: LayerClustererProtocol,
        mesh_generator: MeshGenerator,
        config_loader: ConfigLoaderProtocol,
    ) -> None:
        """パイプラインを初期化。

        Args:
            depth_estimator: 深度推定器インスタンス。
            clusterer: クラスタラーインスタンス。
            mesh_generator: メッシュジェネレーターインスタンス。
            config_loader: 設定ローダーインスタンス。
        """
        self._depth_estimator = depth_estimator
        self._clusterer = clusterer
        self._mesh_generator = mesh_generator
        self._config_loader = config_loader

    def process(
        self,
        image: Image.Image | NDArray | str,
        template_name: str | None = None,
        custom_bbox: BoundingBox | None = None,
        auto_detect: bool = False,
        k: int | None = None,
        include_frame: bool = True,
        include_card_frame: bool = False,
        use_raw_depth: bool = False,
        depth_scale: float = 1.0,
        max_resolution: int | None = None,
    ) -> PipelineResult:
        """画像からシャドーボックスメッシュを生成。

        Args:
            image: 入力画像。PIL Image、NumPy配列、またはファイルパス/URL。
            template_name: 使用するカードテンプレート名。
            custom_bbox: カスタムバウンディングボックス（テンプレートより優先）。
            auto_detect: イラスト領域を自動検出するかどうか。
            k: レイヤー数（Noneの場合は自動探索）。use_raw_depth=Trueの場合は無視。
            include_frame: フレームを含めるかどうか。
            include_card_frame: カードのフレーム部分を含めるかどうか。
                Trueの場合、イラスト領域外のピクセルを最前面の深度で統合。
            use_raw_depth: 生の深度値を使用するかどうか。
                Trueの場合、クラスタリングをスキップし、各ピクセルの深度を
                直接Z座標として使用（より滑らかな深度表現）。
            depth_scale: 生深度モード時の深度スケール（大きいほど立体感が増す）。
            max_resolution: 最大解像度（ピクセル）。指定すると画像をダウンサンプリング
                してポイント数を削減し、レンダリングを高速化。
                例: max_resolution=200 → 最大200x200ピクセル（40,000ポイント）

        Returns:
            PipelineResultオブジェクト。

        Raises:
            ValueError: テンプレートが見つからない場合。

        Example:
            >>> # テンプレートを使用
            >>> result = pipeline.process(image, template_name="pokemon_standard")
            >>>
            >>> # カスタムバウンディングボックスを使用
            >>> bbox = BoundingBox(x=50, y=100, width=400, height=300)
            >>> result = pipeline.process(image, custom_bbox=bbox)
            >>>
            >>> # 自動検出を使用（将来実装予定）
            >>> result = pipeline.process(image, auto_detect=True)
            >>>
            >>> # カードフレームも含める
            >>> result = pipeline.process(image, auto_detect=True, include_card_frame=True)
            >>>
            >>> # 生の深度を使用（クラスタリングなし）
            >>> result = pipeline.process(image, use_raw_depth=True, depth_scale=1.5)
            >>>
            >>> # 軽量プレビュー（最大200x200ピクセル）
            >>> result = pipeline.process(image, auto_detect=True, max_resolution=200)
        """
        # 1. 画像を読み込み
        if isinstance(image, (str, Image.Image)):
            pil_image = load_image(image)
        elif isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        else:
            pil_image = image

        original_array = image_to_array(pil_image)

        # 2. イラスト領域を決定
        bbox = self._resolve_bbox(
            pil_image,
            template_name,
            custom_bbox,
            auto_detect,
        )

        # 3. 必要に応じてクロップ
        if bbox is not None:
            cropped_pil = crop_image(
                pil_image,
                x=bbox.x,
                y=bbox.y,
                width=bbox.width,
                height=bbox.height,
            )
        else:
            cropped_pil = pil_image

        # 3.5. ダウンサンプリング（max_resolution指定時）
        if max_resolution is not None:
            cropped_pil, pil_image, bbox = self._downsample_if_needed(
                cropped_pil, pil_image, bbox, max_resolution
            )
            # original_arrayも更新
            original_array = image_to_array(pil_image)

        # 4. 深度推定とカードフレーム統合
        if include_card_frame and bbox is not None:
            # イラスト領域のみで深度推定
            depth_map_cropped = self._depth_estimator.estimate(cropped_pil)

            # 元画像サイズの深度マップを作成（フレーム部分は0.0=最前面）
            full_height, full_width = original_array.shape[:2]
            depth_map = np.zeros((full_height, full_width), dtype=np.float32)

            # イラスト領域に推定深度を貼り付け
            depth_map[bbox.y : bbox.y + bbox.height, bbox.x : bbox.x + bbox.width] = (
                depth_map_cropped
            )

            # 以降の処理は元画像全体で実施
            cropped_array = original_array
        else:
            # 既存の処理
            cropped_array = image_to_array(cropped_pil)
            depth_map = self._depth_estimator.estimate(cropped_pil)

        # 5. 生深度モード or クラスタリングモード
        if use_raw_depth:
            # 生深度モード: クラスタリングをスキップ
            labels = np.zeros_like(depth_map, dtype=np.int32)  # ダミー
            centroids = np.array([0.5], dtype=np.float32)  # ダミー
            optimal_k = 1

            # 生深度メッシュを生成
            mesh = self._mesh_generator.generate_raw_depth(
                cropped_array,
                depth_map,
                include_frame=include_frame,
                depth_scale=depth_scale,
            )
        else:
            # クラスタリングモード（従来）
            if k is None:
                optimal_k = self._clusterer.find_optimal_k(depth_map)
            else:
                optimal_k = k

            # 6. クラスタリング
            labels, centroids = self._clusterer.cluster(depth_map, optimal_k)

            # 7. カードフレーム統合時: フレーム領域を特別ラベルでマーク
            if include_card_frame and bbox is not None:
                # フレームマスクを作成（イラスト領域外）
                frame_mask = np.ones_like(labels, dtype=bool)
                frame_mask[bbox.y : bbox.y + bbox.height, bbox.x : bbox.x + bbox.width] = (
                    False
                )
                # フレームピクセルを -1 でマーク（累積計算から除外するため）
                labels[frame_mask] = -1

            # 8. メッシュ生成
            mesh = self._mesh_generator.generate(
                cropped_array,
                labels,
                centroids,
                include_frame=include_frame,
            )

        return PipelineResult(
            original_image=original_array,
            cropped_image=cropped_array,
            depth_map=depth_map,
            labels=labels,
            centroids=centroids,
            mesh=mesh,
            optimal_k=optimal_k,
            bbox=bbox,
        )

    def _downsample_if_needed(
        self,
        cropped_pil: Image.Image,
        original_pil: Image.Image,
        bbox: BoundingBox | None,
        max_resolution: int,
    ) -> tuple:
        """必要に応じて画像をダウンサンプリング。

        Args:
            cropped_pil: クロップ済み画像。
            original_pil: 元画像。
            bbox: バウンディングボックス。
            max_resolution: 最大解像度。

        Returns:
            (ダウンサンプリング済みcropped_pil, original_pil, 調整済みbbox)のタプル。
        """
        w, h = cropped_pil.size

        # 最大解像度を超えていない場合はそのまま返す
        if max(w, h) <= max_resolution:
            return cropped_pil, original_pil, bbox

        # スケール係数を計算
        scale = max_resolution / max(w, h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        # クロップ画像をリサイズ
        cropped_pil = cropped_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # include_card_frame用にoriginal_pilもリサイズ
        orig_w, orig_h = original_pil.size
        new_orig_w = int(orig_w * scale)
        new_orig_h = int(orig_h * scale)
        original_pil = original_pil.resize(
            (new_orig_w, new_orig_h), Image.Resampling.LANCZOS
        )

        # bboxもスケール
        if bbox is not None:
            bbox = BoundingBox(
                x=int(bbox.x * scale),
                y=int(bbox.y * scale),
                width=new_w,
                height=new_h,
            )

        return cropped_pil, original_pil, bbox

    def _resolve_bbox(
        self,
        image: Image.Image,
        template_name: str | None,
        custom_bbox: BoundingBox | None,
        auto_detect: bool,
    ) -> BoundingBox | None:
        """使用するバウンディングボックスを決定。

        優先順位: custom_bbox > template > auto_detect > None（画像全体）

        Args:
            image: 入力画像。
            template_name: テンプレート名。
            custom_bbox: カスタムバウンディングボックス。
            auto_detect: 自動検出フラグ。

        Returns:
            決定されたBoundingBox、またはNone（画像全体を使用）。
        """
        # カスタムバウンディングボックスが最優先
        if custom_bbox is not None:
            return custom_bbox

        # テンプレートを使用
        if template_name is not None:
            try:
                template = self._config_loader.load_template(template_name)
                return template.illustration_area
            except FileNotFoundError as e:
                raise ValueError(f"テンプレートが見つかりません: {template_name}") from e

        # 自動検出（将来実装予定）
        if auto_detect:
            # TODO: detection/region.pyの実装後に置き換え
            # 現時点では画像全体を使用
            return None

        # デフォルト: 画像全体を使用
        return None


def create_pipeline(
    settings: ShadowboxSettings | None = None,
    use_mock_depth: bool = False,
):
    """設定に基づいてパイプラインを作成するファクトリ関数。

    依存性注入を内部で処理し、設定に応じた
    パイプラインインスタンスを返します。

    Args:
        settings: シャドーボックス設定。Noneの場合はデフォルト。
        use_mock_depth: テスト用にモック深度推定器を使用するかどうか。

    Returns:
        model_mode="depth"の場合: ShadowboxPipelineインスタンス。
        model_mode="triposr"の場合: TripoSRPipelineインスタンス。

    Example:
        >>> # デフォルト設定（深度推定モード）
        >>> pipeline = create_pipeline()
        >>>
        >>> # TripoSRモード
        >>> settings = ShadowboxSettings(model_mode="triposr")
        >>> pipeline = create_pipeline(settings)
        >>>
        >>> # テスト用（モック深度推定）
        >>> pipeline = create_pipeline(use_mock_depth=True)
    """
    if settings is None:
        settings = ShadowboxSettings()

    # TripoSRモードの場合
    if settings.model_mode == "triposr":
        from shadowbox.triposr import create_triposr_pipeline
        return create_triposr_pipeline(settings.triposr, settings.render)

    # 深度推定モード（デフォルト）
    # 深度推定器を作成
    if use_mock_depth:
        depth_estimator: DepthEstimatorProtocol = MockDepthEstimator()
    else:
        depth_estimator = create_depth_estimator(settings.depth)

    # クラスタラーを作成
    clusterer = KMeansLayerClusterer(settings.clustering)

    # メッシュジェネレーターを作成
    mesh_generator = MeshGenerator(settings.render)

    # 設定ローダーを作成
    config_loader = YAMLConfigLoader(settings.templates_dir)

    return ShadowboxPipeline(
        depth_estimator=depth_estimator,
        clusterer=clusterer,
        mesh_generator=mesh_generator,
        config_loader=config_loader,
    )
