"""TripoSRパイプラインモジュール。

TripoSRを使用した3Dメッシュ生成パイプラインを提供します。
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from shadowbox.config.template import BoundingBox
from shadowbox.core.mesh import ShadowboxMesh
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

    def __init__(self, settings: TripoSRSettings) -> None:
        """パイプラインを初期化。

        Args:
            settings: TripoSR設定。
        """
        self._settings = settings
        self._generator = TripoSRGenerator(settings)

    def process(
        self,
        image: Union[str, Path, Image.Image, NDArray],
        bbox: Optional[BoundingBox] = None,
    ) -> TripoSRPipelineResult:
        """画像を処理して3Dメッシュを生成。

        Args:
            image: 入力画像（パス、PIL Image、またはNumPy配列）。
            bbox: イラスト領域のバウンディングボックス（Noneの場合は画像全体）。

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

        return TripoSRPipelineResult(
            original_image=original_array,
            mesh=mesh,
            bbox=bbox,
        )

    @property
    def settings(self) -> TripoSRSettings:
        """現在の設定を取得。"""
        return self._settings
