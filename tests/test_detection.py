"""イラスト領域検出モジュールのテスト。

このモジュールは、RegionDetectorと関連機能の
ユニットテストを提供します。
"""

import numpy as np
import pytest
from PIL import Image

from shadowbox.config.template import BoundingBox
from shadowbox.detection.region import (
    DetectionResult,
    RegionDetector,
    detect_illustration_region,
)


@pytest.fixture
def sample_card_image() -> Image.Image:
    """テスト用のカード画像を作成。

    枠（暗い色）とイラスト領域（明るい色）を持つ模擬画像。
    """
    # 250x350の画像（一般的なカードサイズ比）
    width, height = 250, 350
    image = Image.new("RGB", (width, height), color=(40, 40, 40))  # 暗い枠

    # 中央にイラスト領域を描画
    pixels = image.load()
    illustration_x = 20
    illustration_y = 40
    illustration_w = 210
    illustration_h = 200

    for y in range(illustration_y, illustration_y + illustration_h):
        for x in range(illustration_x, illustration_x + illustration_w):
            # カラフルなイラスト領域をシミュレート
            r = (x * 3 + y) % 256
            g = (x + y * 2) % 256
            b = (x * 2 + y * 3) % 256
            pixels[x, y] = (r, g, b)

    return image


@pytest.fixture
def simple_image() -> Image.Image:
    """シンプルなテスト画像を作成。"""
    return Image.new("RGB", (100, 100), color=(128, 128, 128))


class TestRegionDetector:
    """RegionDetectorのテスト。"""

    def test_init_default(self) -> None:
        """デフォルト初期化をテスト。"""
        detector = RegionDetector()

        assert detector._min_area_ratio == 0.1
        assert detector._max_area_ratio == 0.9
        assert detector._canny_low == 50
        assert detector._canny_high == 150

    def test_init_custom(self) -> None:
        """カスタム初期化をテスト。"""
        detector = RegionDetector(
            min_area_ratio=0.2,
            max_area_ratio=0.8,
            canny_low=30,
            canny_high=100,
        )

        assert detector._min_area_ratio == 0.2
        assert detector._max_area_ratio == 0.8
        assert detector._canny_low == 30
        assert detector._canny_high == 100

    def test_detect_returns_result(self, sample_card_image: Image.Image) -> None:
        """検出が結果を返すことをテスト。"""
        detector = RegionDetector()
        result = detector.detect(sample_card_image)

        assert isinstance(result, DetectionResult)
        assert isinstance(result.bbox, BoundingBox)
        assert 0.0 <= result.confidence <= 1.0
        assert result.method in ["edge_detection", "color_difference", "contour_detection", "default"]

    def test_detect_simple_image(self, simple_image: Image.Image) -> None:
        """シンプルな画像での検出をテスト。"""
        detector = RegionDetector()
        result = detector.detect(simple_image)

        assert isinstance(result, DetectionResult)
        # 単純な画像でも何かしらの結果を返す
        assert result.bbox is not None

    def test_detect_bbox_within_image(self, sample_card_image: Image.Image) -> None:
        """検出されたbboxが画像内に収まることをテスト。"""
        detector = RegionDetector()
        result = detector.detect(sample_card_image)

        width, height = sample_card_image.size
        bbox = result.bbox

        assert bbox.x >= 0
        assert bbox.y >= 0
        assert bbox.x + bbox.width <= width
        assert bbox.y + bbox.height <= height

    def test_detect_with_candidates(self, sample_card_image: Image.Image) -> None:
        """複数候補の検出をテスト。"""
        detector = RegionDetector()
        results = detector.detect_with_candidates(sample_card_image, max_candidates=3)

        assert isinstance(results, list)
        assert len(results) <= 3

        # 信頼度の降順でソートされていることを確認
        if len(results) > 1:
            for i in range(len(results) - 1):
                assert results[i].confidence >= results[i + 1].confidence

    def test_pil_to_cv(self, sample_card_image: Image.Image) -> None:
        """PIL→OpenCV変換をテスト。"""
        detector = RegionDetector()
        cv_image = detector._pil_to_cv(sample_card_image)

        assert cv_image.dtype == np.uint8
        assert len(cv_image.shape) == 3
        assert cv_image.shape[2] == 3  # BGR

    def test_create_default_bbox(self) -> None:
        """デフォルトbbox作成をテスト。"""
        detector = RegionDetector()
        bbox = detector._create_default_bbox(100, 100)

        # 中央60%の領域
        assert bbox.x == 20
        assert bbox.y == 20
        assert bbox.width == 60
        assert bbox.height == 60


class TestDetectionResult:
    """DetectionResultデータクラスのテスト。"""

    def test_create(self) -> None:
        """DetectionResultの作成をテスト。"""
        bbox = BoundingBox(x=10, y=20, width=100, height=150)
        result = DetectionResult(
            bbox=bbox,
            confidence=0.85,
            method="edge_detection",
        )

        assert result.bbox == bbox
        assert result.confidence == 0.85
        assert result.method == "edge_detection"


class TestDetectIllustrationRegion:
    """ユーティリティ関数のテスト。"""

    def test_detect_illustration_region(self, sample_card_image: Image.Image) -> None:
        """detect_illustration_region関数をテスト。"""
        result = detect_illustration_region(sample_card_image)

        assert isinstance(result, DetectionResult)
        assert result.bbox is not None

    def test_detect_with_custom_ratios(self, sample_card_image: Image.Image) -> None:
        """カスタム面積比での検出をテスト。"""
        result = detect_illustration_region(
            sample_card_image,
            min_area_ratio=0.2,
            max_area_ratio=0.7,
        )

        assert isinstance(result, DetectionResult)


class TestEdgeCases:
    """エッジケースのテスト。"""

    def test_very_small_image(self) -> None:
        """非常に小さい画像での検出をテスト。"""
        small_image = Image.new("RGB", (10, 10), color=(128, 128, 128))
        detector = RegionDetector()

        result = detector.detect(small_image)

        assert result is not None
        assert result.bbox is not None

    def test_monochrome_image(self) -> None:
        """単色画像での検出をテスト。"""
        mono_image = Image.new("RGB", (100, 100), color=(50, 50, 50))
        detector = RegionDetector()

        result = detector.detect(mono_image)

        # 単色画像でもデフォルト領域を返す
        assert result is not None
        assert result.bbox is not None

    def test_high_contrast_image(self) -> None:
        """高コントラスト画像での検出をテスト。"""
        # 白黒の縦縞
        image = Image.new("RGB", (100, 100))
        pixels = image.load()
        for y in range(100):
            for x in range(100):
                if x < 50:
                    pixels[x, y] = (255, 255, 255)
                else:
                    pixels[x, y] = (0, 0, 0)

        detector = RegionDetector()
        result = detector.detect(image)

        assert result is not None
