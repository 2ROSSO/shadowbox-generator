"""深度推定モジュールのテスト。

このモジュールは、深度推定器とユーティリティ関数の
ユニットテストを提供します。
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from shadowbox.config import DepthEstimationSettings
from shadowbox.core.depth import (
    DepthAnythingEstimator,
    MockDepthEstimator,
    create_depth_estimator,
)
from shadowbox.utils.image import (
    array_to_image,
    crop_image,
    image_to_array,
    load_image,
    load_image_from_file,
)


class TestMockDepthEstimator:
    """MockDepthEstimatorのテスト。"""

    def test_estimate_gradient(self, sample_image: Image.Image) -> None:
        """グラデーション深度マップの生成をテスト。"""
        estimator = MockDepthEstimator()
        depth_map = estimator.estimate(sample_image)

        # 形状の確認
        width, height = sample_image.size
        assert depth_map.shape == (height, width)

        # 型の確認
        assert depth_map.dtype == np.float32

        # グラデーションの確認（上が0、下が1）
        assert depth_map[0, 0] == pytest.approx(0.0, abs=0.01)
        assert depth_map[-1, 0] == pytest.approx(1.0, abs=0.01)

    def test_estimate_fixed_depth(self, sample_image: Image.Image) -> None:
        """固定深度マップの生成をテスト。"""
        fixed_value = 0.5
        estimator = MockDepthEstimator(fixed_depth=fixed_value)
        depth_map = estimator.estimate(sample_image)

        # 全ピクセルが同じ値
        assert np.allclose(depth_map, fixed_value)


class TestDepthAnythingEstimator:
    """DepthAnythingEstimatorのテスト。"""

    def test_init(self) -> None:
        """初期化をテスト。"""
        settings = DepthEstimationSettings()
        estimator = DepthAnythingEstimator(settings)

        # パイプラインはまだ初期化されていない
        assert estimator._pipeline is None

    def test_normalize_depth(self) -> None:
        """深度の正規化をテスト。"""
        settings = DepthEstimationSettings()
        estimator = DepthAnythingEstimator(settings)

        # テスト用の深度データ
        depth = np.array([[10, 20], [30, 40]], dtype=np.float32)
        normalized = estimator._normalize_depth(depth)

        # 最小値が0、最大値が1になる
        assert normalized.min() == pytest.approx(0.0)
        assert normalized.max() == pytest.approx(1.0)

    def test_normalize_depth_constant(self) -> None:
        """全て同じ値の深度マップの正規化をテスト。"""
        settings = DepthEstimationSettings()
        estimator = DepthAnythingEstimator(settings)

        # 全て同じ値
        depth = np.full((10, 10), 5.0, dtype=np.float32)
        normalized = estimator._normalize_depth(depth)

        # ゼロ除算を防いで全て0になる
        assert np.allclose(normalized, 0.0)


class TestCreateDepthEstimator:
    """create_depth_estimatorファクトリ関数のテスト。"""

    def test_create_depth_anything(self) -> None:
        """Depth Anything推定器の作成をテスト。"""
        settings = DepthEstimationSettings(model_type="depth_anything")
        estimator = create_depth_estimator(settings)

        assert isinstance(estimator, DepthAnythingEstimator)

    def test_create_midas_fallback(self) -> None:
        """MiDaS指定時のフォールバックをテスト。"""
        settings = DepthEstimationSettings(model_type="midas")
        estimator = create_depth_estimator(settings)

        # 現時点ではDepth Anythingにフォールバック
        assert isinstance(estimator, DepthAnythingEstimator)

    def test_create_invalid_type(self) -> None:
        """無効なモデルタイプでエラーが発生することをテスト。"""
        # 型チェックをバイパス
        settings = DepthEstimationSettings()
        settings.model_type = "invalid"  # type: ignore

        with pytest.raises(ValueError, match="サポートされていない"):
            create_depth_estimator(settings)


class TestImageUtils:
    """画像ユーティリティ関数のテスト。"""

    def test_load_image_from_pil(self, sample_image: Image.Image) -> None:
        """PIL Imageからの読み込みをテスト。"""
        loaded = load_image(sample_image)

        assert isinstance(loaded, Image.Image)
        assert loaded.mode == "RGB"

    def test_load_image_from_file(self, sample_image: Image.Image) -> None:
        """ファイルからの読み込みをテスト。"""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            file_path = Path(tmp.name)

        try:
            # 画像を一時ファイルに保存
            sample_image.save(file_path)

            # ファイルから読み込み
            loaded = load_image(str(file_path))

            assert isinstance(loaded, Image.Image)
            assert loaded.size == sample_image.size

            # 明示的に閉じる
            loaded.close()
        finally:
            # ファイルを削除
            file_path.unlink(missing_ok=True)

    def test_load_image_from_path(self, sample_image: Image.Image) -> None:
        """Pathオブジェクトからの読み込みをテスト。"""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            file_path = Path(tmp.name)

        try:
            sample_image.save(file_path)
            loaded = load_image(file_path)

            assert isinstance(loaded, Image.Image)
            loaded.close()
        finally:
            file_path.unlink(missing_ok=True)

    def test_load_image_file_not_found(self) -> None:
        """存在しないファイルでエラーが発生することをテスト。"""
        with pytest.raises(FileNotFoundError):
            load_image_from_file("nonexistent.png")

    def test_load_image_invalid_type(self) -> None:
        """無効なタイプでエラーが発生することをテスト。"""
        with pytest.raises(TypeError, match="サポートされていない"):
            load_image(123)  # type: ignore

    def test_image_to_array(self, sample_image: Image.Image) -> None:
        """PIL ImageからNumPy配列への変換をテスト。"""
        array = image_to_array(sample_image)

        width, height = sample_image.size
        assert array.shape == (height, width, 3)
        assert array.dtype == np.uint8

    def test_array_to_image(self) -> None:
        """NumPy配列からPIL Imageへの変換をテスト。"""
        array = np.zeros((100, 150, 3), dtype=np.uint8)
        image = array_to_image(array)

        assert isinstance(image, Image.Image)
        assert image.size == (150, 100)  # (width, height)

    def test_crop_image(self, sample_image: Image.Image) -> None:
        """画像の切り抜きをテスト。"""
        cropped = crop_image(sample_image, x=10, y=20, width=50, height=30)

        assert cropped.size == (50, 30)

    def test_rgba_to_rgb_conversion(self) -> None:
        """RGBA画像のRGB変換をテスト。"""
        # 半透明の赤い画像
        rgba_image = Image.new("RGBA", (100, 100), (255, 0, 0, 128))
        rgb_image = load_image(rgba_image)

        assert rgb_image.mode == "RGB"


class TestDepthEstimatorIntegration:
    """深度推定器の統合テスト。

    注意: このテストはモック推定器を使用するため、
    実際のモデルのダウンロードは発生しません。
    実際のモデルを使用した統合テストは別途実行する必要があります。
    """

    def test_mock_estimator_with_loaded_image(self) -> None:
        """読み込んだ画像でモック推定器を使用するテスト。"""
        # テスト画像を作成
        image = Image.new("RGB", (100, 150), color="blue")

        # モック推定器で深度推定
        estimator = MockDepthEstimator()
        depth_map = estimator.estimate(image)

        # 結果の検証
        assert depth_map.shape == (150, 100)
        assert 0.0 <= depth_map.min() <= depth_map.max() <= 1.0
