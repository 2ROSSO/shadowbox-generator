"""シャドーボックス生成パイプラインモジュール。

このモジュールは、画像からシャドーボックスメッシュを生成する
一連の処理を統合したパイプラインを提供します。
"""

from dataclasses import dataclass
from typing import Optional

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
    bbox: Optional[BoundingBox]


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
        template_name: Optional[str] = None,
        custom_bbox: Optional[BoundingBox] = None,
        auto_detect: bool = False,
        k: Optional[int] = None,
        include_frame: bool = True,
    ) -> PipelineResult:
        """画像からシャドーボックスメッシュを生成。

        Args:
            image: 入力画像。PIL Image、NumPy配列、またはファイルパス/URL。
            template_name: 使用するカードテンプレート名。
            custom_bbox: カスタムバウンディングボックス（テンプレートより優先）。
            auto_detect: イラスト領域を自動検出するかどうか。
            k: レイヤー数（Noneの場合は自動探索）。
            include_frame: フレームを含めるかどうか。

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

        cropped_array = image_to_array(cropped_pil)

        # 4. 深度推定
        depth_map = self._depth_estimator.estimate(cropped_pil)

        # 5. 最適なkを探索または使用
        if k is None:
            optimal_k = self._clusterer.find_optimal_k(depth_map)
        else:
            optimal_k = k

        # 6. クラスタリング
        labels, centroids = self._clusterer.cluster(depth_map, optimal_k)

        # 7. メッシュ生成
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

    def _resolve_bbox(
        self,
        image: Image.Image,
        template_name: Optional[str],
        custom_bbox: Optional[BoundingBox],
        auto_detect: bool,
    ) -> Optional[BoundingBox]:
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
            except FileNotFoundError:
                raise ValueError(f"テンプレートが見つかりません: {template_name}")

        # 自動検出（将来実装予定）
        if auto_detect:
            # TODO: detection/region.pyの実装後に置き換え
            # 現時点では画像全体を使用
            return None

        # デフォルト: 画像全体を使用
        return None


def create_pipeline(
    settings: Optional[ShadowboxSettings] = None,
    use_mock_depth: bool = False,
) -> ShadowboxPipeline:
    """設定に基づいてパイプラインを作成するファクトリ関数。

    依存性注入を内部で処理し、設定に応じた
    パイプラインインスタンスを返します。

    Args:
        settings: シャドーボックス設定。Noneの場合はデフォルト。
        use_mock_depth: テスト用にモック深度推定器を使用するかどうか。

    Returns:
        設定済みのShadowboxPipelineインスタンス。

    Example:
        >>> # デフォルト設定
        >>> pipeline = create_pipeline()
        >>>
        >>> # カスタム設定
        >>> settings = ShadowboxSettings()
        >>> settings.depth.model_type = "midas"
        >>> pipeline = create_pipeline(settings)
        >>>
        >>> # テスト用（モック深度推定）
        >>> pipeline = create_pipeline(use_mock_depth=True)
    """
    if settings is None:
        settings = ShadowboxSettings()

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
