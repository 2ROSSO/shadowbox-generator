"""深度推定モジュール。

このモジュールは、画像から深度マップを推定するための
クラスを提供します。複数のモデルをサポートし、
Protocolベースの設計により差し替えが容易です。
"""

from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from shadowbox.config.settings import DepthEstimationSettings


class DepthEstimatorProtocol(Protocol):
    """深度推定器のプロトコル（DIインターフェース）。

    このプロトコルを実装することで、新しい深度推定モデルを
    追加できます。依存性注入パターンにより、テスト時の
    モック化や異なるモデルへの差し替えが容易になります。

    Example:
        >>> class CustomEstimator:
        ...     def estimate(self, image: Image.Image) -> NDArray[np.float32]:
        ...         # カスタム実装
        ...         pass
    """

    def estimate(self, image: Image.Image) -> NDArray[np.float32]:
        """画像から深度マップを推定。

        Args:
            image: 入力画像（RGB PIL Image）。

        Returns:
            深度マップ。shape (H, W) のfloat32配列。
            値は0.0（最も近い）から1.0（最も遠い）に正規化。
        """
        ...


class DepthAnythingEstimator:
    """Depth Anything v2による深度推定器。

    Hugging Face Transformersのパイプラインを使用して
    深度推定を行います。モデルは初回使用時に自動的に
    ダウンロードされ、キャッシュされます。

    Attributes:
        settings: 深度推定の設定。

    Example:
        >>> from shadowbox.config import DepthEstimationSettings
        >>> settings = DepthEstimationSettings()
        >>> estimator = DepthAnythingEstimator(settings)
        >>> depth_map = estimator.estimate(image)
    """

    def __init__(self, settings: DepthEstimationSettings) -> None:
        """推定器を初期化。

        Args:
            settings: 深度推定の設定。モデル名やデバイスを指定。
        """
        self._settings = settings
        self._pipeline = None  # 遅延初期化

    def _ensure_pipeline(self) -> None:
        """パイプラインが初期化されていることを保証。

        初回呼び出し時にHugging Faceからモデルをダウンロードし、
        パイプラインを初期化します。
        """
        if self._pipeline is not None:
            return

        # transformersは重いので遅延インポート
        from transformers import pipeline

        device = self._resolve_device()
        self._pipeline = pipeline(
            "depth-estimation",
            model=self._settings.model_name,
            device=device,
        )

    def _resolve_device(self) -> str:
        """'auto'デバイス設定を実際のデバイスに解決。

        Returns:
            使用するデバイス文字列（"cpu", "cuda", "mps"のいずれか）。
        """
        if self._settings.device != "auto":
            return self._settings.device

        # 利用可能なデバイスを検出
        import torch

        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"

        return "cpu"

    def estimate(self, image: Image.Image) -> NDArray[np.float32]:
        """画像から深度マップを推定。

        Args:
            image: 入力画像（RGB PIL Image）。

        Returns:
            深度マップ。shape (H, W) のfloat32配列。
            値は0.0（最も近い）から1.0（最も遠い）に正規化。
        """
        self._ensure_pipeline()

        # パイプラインで推定
        result = self._pipeline(image)

        # 結果から深度マップを取得
        depth_image = result["depth"]
        depth = np.array(depth_image, dtype=np.float32)

        # [0, 1]の範囲に正規化
        depth = self._normalize_depth(depth)

        return depth

    def _normalize_depth(self, depth: NDArray[np.float32]) -> NDArray[np.float32]:
        """深度マップを[0, 1]の範囲に正規化。

        Args:
            depth: 正規化前の深度マップ。

        Returns:
            0（近い）から1（遠い）に正規化された深度マップ。
        """
        depth_min = depth.min()
        depth_max = depth.max()

        # ゼロ除算を防止
        if depth_max - depth_min < 1e-8:
            return np.zeros_like(depth)

        normalized = (depth - depth_min) / (depth_max - depth_min)
        return normalized.astype(np.float32)


class MockDepthEstimator:
    """テスト用のモック深度推定器。

    実際のモデルを使用せずに、ランダムまたは
    固定の深度マップを返します。
    """

    def __init__(self, fixed_depth: float | None = None) -> None:
        """モック推定器を初期化。

        Args:
            fixed_depth: 固定の深度値。Noneの場合はグラデーション。
        """
        self._fixed_depth = fixed_depth

    def estimate(self, image: Image.Image) -> NDArray[np.float32]:
        """モック深度マップを生成。

        Args:
            image: 入力画像（サイズ取得に使用）。

        Returns:
            モック深度マップ。
        """
        width, height = image.size

        if self._fixed_depth is not None:
            return np.full((height, width), self._fixed_depth, dtype=np.float32)

        # 上から下へのグラデーション
        gradient = np.linspace(0, 1, height, dtype=np.float32)
        return np.tile(gradient[:, np.newaxis], (1, width))


def create_depth_estimator(settings: DepthEstimationSettings) -> DepthEstimatorProtocol:
    """設定に基づいて深度推定器を作成するファクトリ関数。

    Args:
        settings: 深度推定の設定。

    Returns:
        設定に応じた深度推定器インスタンス。

    Raises:
        ValueError: サポートされていないモデルタイプの場合。

    Example:
        >>> settings = DepthEstimationSettings(model_type="depth_anything")
        >>> estimator = create_depth_estimator(settings)
    """
    if settings.model_type == "depth_anything":
        return DepthAnythingEstimator(settings)
    elif settings.model_type == "midas":
        # MiDaSは将来の拡張として実装予定
        # 現時点ではDepth Anythingにフォールバック
        return DepthAnythingEstimator(settings)
    elif settings.model_type == "zoedepth":
        # ZoeDepthは将来の拡張として実装予定
        return DepthAnythingEstimator(settings)
    else:
        raise ValueError(f"サポートされていないモデルタイプ: {settings.model_type}")
