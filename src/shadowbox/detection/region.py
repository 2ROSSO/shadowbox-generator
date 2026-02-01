"""イラスト領域自動検出モジュール。

このモジュールは、TCGカード画像からイラスト領域を
自動的に検出する機能を提供します。
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image

from shadowbox.config.template import BoundingBox


@dataclass
class DetectionResult:
    """検出結果を格納するデータクラス。

    Attributes:
        bbox: 検出されたバウンディングボックス。
        confidence: 検出の信頼度（0.0〜1.0）。
        method: 使用した検出方法。
    """

    bbox: BoundingBox
    confidence: float
    method: str


class RegionDetector:
    """イラスト領域の自動検出器。

    エッジ検出と輪郭検出を使用して、カード画像内の
    イラスト領域を自動的に特定します。

    Attributes:
        min_area_ratio: 最小面積比（カード全体に対する）。
        max_area_ratio: 最大面積比。
        canny_low: Cannyエッジ検出の低閾値。
        canny_high: Cannyエッジ検出の高閾値。

    Example:
        >>> detector = RegionDetector()
        >>> result = detector.detect(image)
        >>> print(result.bbox)
    """

    def __init__(
        self,
        min_area_ratio: float = 0.1,
        max_area_ratio: float = 0.9,
        canny_low: int = 50,
        canny_high: int = 150,
    ) -> None:
        """検出器を初期化。

        Args:
            min_area_ratio: 検出する矩形の最小面積比。
            max_area_ratio: 検出する矩形の最大面積比。
            canny_low: Cannyエッジ検出の低閾値。
            canny_high: Cannyエッジ検出の高閾値。
        """
        self._min_area_ratio = min_area_ratio
        self._max_area_ratio = max_area_ratio
        self._canny_low = canny_low
        self._canny_high = canny_high

    def detect(self, image: Image.Image) -> DetectionResult:
        """イラスト領域を検出。

        Args:
            image: 入力画像（PIL Image）。

        Returns:
            検出結果（バウンディングボックスと信頼度）。

        Example:
            >>> result = detector.detect(card_image)
            >>> if result.confidence > 0.7:
            ...     cropped = image.crop(result.bbox.to_tuple())
        """
        # PILからOpenCV形式に変換
        cv_image = self._pil_to_cv(image)
        height, width = cv_image.shape[:2]

        # 複数の検出手法を試行し、最も良い結果を返す
        results: List[DetectionResult] = []

        # 方法1: エッジベースの検出
        edge_result = self._detect_by_edges(cv_image)
        if edge_result is not None:
            results.append(edge_result)

        # 方法2: 色差ベースの検出
        color_result = self._detect_by_color_difference(cv_image)
        if color_result is not None:
            results.append(color_result)

        # 方法3: 輪郭ベースの検出
        contour_result = self._detect_by_contours(cv_image)
        if contour_result is not None:
            results.append(contour_result)

        # 結果がない場合はデフォルトの矩形を返す
        if not results:
            default_bbox = self._create_default_bbox(width, height)
            return DetectionResult(
                bbox=default_bbox,
                confidence=0.0,
                method="default",
            )

        # 最も信頼度の高い結果を返す
        best_result = max(results, key=lambda r: r.confidence)
        return best_result

    def detect_with_candidates(
        self,
        image: Image.Image,
        max_candidates: int = 5,
    ) -> List[DetectionResult]:
        """複数の候補領域を検出。

        Args:
            image: 入力画像（PIL Image）。
            max_candidates: 返す候補の最大数。

        Returns:
            検出結果のリスト（信頼度順）。

        Example:
            >>> candidates = detector.detect_with_candidates(image, max_candidates=3)
            >>> for result in candidates:
            ...     print(f"{result.method}: {result.confidence:.2f}")
        """
        cv_image = self._pil_to_cv(image)
        height, width = cv_image.shape[:2]

        results: List[DetectionResult] = []

        # エッジベースの候補
        edge_result = self._detect_by_edges(cv_image)
        if edge_result is not None:
            results.append(edge_result)

        # 色差ベースの候補
        color_result = self._detect_by_color_difference(cv_image)
        if color_result is not None:
            results.append(color_result)

        # 輪郭ベースの候補（複数）
        contour_results = self._detect_multiple_by_contours(cv_image, max_candidates)
        results.extend(contour_results)

        # 信頼度でソートして上位を返す
        results.sort(key=lambda r: r.confidence, reverse=True)
        return results[:max_candidates]

    def _pil_to_cv(self, image: Image.Image) -> NDArray[np.uint8]:
        """PIL ImageをOpenCV形式に変換。

        Args:
            image: PIL Image。

        Returns:
            OpenCV形式のBGR画像。
        """
        # RGBからBGRに変換
        rgb = np.array(image.convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def _detect_by_edges(
        self,
        cv_image: NDArray[np.uint8],
    ) -> Optional[DetectionResult]:
        """エッジ検出ベースの領域検出。

        Args:
            cv_image: OpenCV形式の画像。

        Returns:
            検出結果。検出できない場合はNone。
        """
        height, width = cv_image.shape[:2]
        total_area = width * height

        # グレースケール変換
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        # ガウシアンブラーでノイズ除去
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Cannyエッジ検出
        edges = cv2.Canny(blurred, self._canny_low, self._canny_high)

        # 膨張処理でエッジを太くする
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=2)

        # 輪郭を検出
        contours, _ = cv2.findContours(
            dilated,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        # 面積条件を満たす矩形を探す
        valid_rects: List[Tuple[int, int, int, int, float]] = []

        for contour in contours:
            # 矩形近似
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            area_ratio = area / total_area

            if self._min_area_ratio <= area_ratio <= self._max_area_ratio:
                # アスペクト比をチェック（極端に細長い矩形を除外）
                aspect_ratio = w / h if h > 0 else 0
                if 0.3 <= aspect_ratio <= 3.0:
                    # 信頼度を計算（面積比と矩形らしさ）
                    rect_area = w * h
                    contour_area = cv2.contourArea(contour)
                    rectangularity = contour_area / rect_area if rect_area > 0 else 0
                    confidence = rectangularity * 0.7 + (1 - abs(aspect_ratio - 1) / 2) * 0.3
                    valid_rects.append((x, y, w, h, confidence))

        if not valid_rects:
            return None

        # 最も信頼度の高い矩形を選択
        best_rect = max(valid_rects, key=lambda r: r[4])
        x, y, w, h, confidence = best_rect

        return DetectionResult(
            bbox=BoundingBox(x=x, y=y, width=w, height=h),
            confidence=min(confidence, 1.0),
            method="edge_detection",
        )

    def _detect_by_color_difference(
        self,
        cv_image: NDArray[np.uint8],
    ) -> Optional[DetectionResult]:
        """色差ベースの領域検出。

        カードの枠（通常は暗い色）とイラスト部分の色差を利用。

        Args:
            cv_image: OpenCV形式の画像。

        Returns:
            検出結果。検出できない場合はNone。
        """
        height, width = cv_image.shape[:2]
        total_area = width * height

        # HSVに変換
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        _, s, v = cv2.split(hsv)

        # 彩度と明度の組み合わせで閾値処理
        # イラスト部分は通常、枠より彩度・明度が高い
        combined = cv2.addWeighted(s, 0.5, v, 0.5, 0)

        # 大津の閾値処理
        _, thresh = cv2.threshold(combined, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # モルフォロジー処理でノイズ除去
        kernel = np.ones((5, 5), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

        # 輪郭を検出
        contours, _ = cv2.findContours(
            cleaned,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        valid_rects: List[Tuple[int, int, int, int, float]] = []

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            area_ratio = area / total_area

            if self._min_area_ratio <= area_ratio <= self._max_area_ratio:
                aspect_ratio = w / h if h > 0 else 0
                if 0.3 <= aspect_ratio <= 3.0:
                    # 信頼度を計算
                    rect_area = w * h
                    contour_area = cv2.contourArea(contour)
                    rectangularity = contour_area / rect_area if rect_area > 0 else 0
                    # 中央に近いほど高い信頼度
                    center_x = x + w / 2
                    center_y = y + h / 2
                    center_dist = abs(center_x - width / 2) / width + abs(center_y - height / 2) / height
                    center_bonus = max(0, 1 - center_dist)
                    confidence = rectangularity * 0.5 + center_bonus * 0.5
                    valid_rects.append((x, y, w, h, confidence))

        if not valid_rects:
            return None

        best_rect = max(valid_rects, key=lambda r: r[4])
        x, y, w, h, confidence = best_rect

        return DetectionResult(
            bbox=BoundingBox(x=x, y=y, width=w, height=h),
            confidence=min(confidence, 1.0),
            method="color_difference",
        )

    def _detect_by_contours(
        self,
        cv_image: NDArray[np.uint8],
    ) -> Optional[DetectionResult]:
        """輪郭ベースの領域検出。

        Args:
            cv_image: OpenCV形式の画像。

        Returns:
            検出結果。検出できない場合はNone。
        """
        results = self._detect_multiple_by_contours(cv_image, 1)
        return results[0] if results else None

    def _detect_multiple_by_contours(
        self,
        cv_image: NDArray[np.uint8],
        max_count: int = 5,
    ) -> List[DetectionResult]:
        """輪郭ベースで複数の候補を検出。

        Args:
            cv_image: OpenCV形式の画像。
            max_count: 返す最大候補数。

        Returns:
            検出結果のリスト。
        """
        height, width = cv_image.shape[:2]
        total_area = width * height

        # グレースケール変換
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        # 適応的閾値処理
        thresh = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2,
        )

        # 輪郭を検出
        contours, _ = cv2.findContours(
            thresh,
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        results: List[DetectionResult] = []

        for contour in contours:
            # 輪郭を多角形近似
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # 4角形に近い輪郭を探す
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                area = w * h
                area_ratio = area / total_area

                if self._min_area_ratio <= area_ratio <= self._max_area_ratio:
                    # 信頼度を計算
                    contour_area = cv2.contourArea(contour)
                    rect_area = w * h
                    rectangularity = contour_area / rect_area if rect_area > 0 else 0
                    confidence = rectangularity * 0.8 + 0.2  # 4角形ボーナス

                    results.append(
                        DetectionResult(
                            bbox=BoundingBox(x=x, y=y, width=w, height=h),
                            confidence=min(confidence, 1.0),
                            method="contour_detection",
                        )
                    )

        # 信頼度でソートして上位を返す
        results.sort(key=lambda r: r.confidence, reverse=True)
        return results[:max_count]

    def _create_default_bbox(self, width: int, height: int) -> BoundingBox:
        """デフォルトのバウンディングボックスを作成。

        画像中央の60%の領域を返す。

        Args:
            width: 画像幅。
            height: 画像高さ。

        Returns:
            デフォルトのバウンディングボックス。
        """
        margin_x = int(width * 0.2)
        margin_y = int(height * 0.2)

        return BoundingBox(
            x=margin_x,
            y=margin_y,
            width=width - 2 * margin_x,
            height=height - 2 * margin_y,
        )


def detect_illustration_region(
    image: Image.Image,
    min_area_ratio: float = 0.1,
    max_area_ratio: float = 0.9,
) -> DetectionResult:
    """イラスト領域を検出するユーティリティ関数。

    Args:
        image: 入力画像。
        min_area_ratio: 最小面積比。
        max_area_ratio: 最大面積比。

    Returns:
        検出結果。

    Example:
        >>> result = detect_illustration_region(card_image)
        >>> print(f"Detected: {result.bbox}, confidence: {result.confidence:.2f}")
    """
    detector = RegionDetector(
        min_area_ratio=min_area_ratio,
        max_area_ratio=max_area_ratio,
    )
    return detector.detect(image)
