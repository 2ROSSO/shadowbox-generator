"""イラスト領域自動検出モジュール。

このモジュールは、TCGカード画像からイラスト領域を
自動的に検出する機能を提供します。
"""

from dataclasses import dataclass

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
        results: list[DetectionResult] = []

        # 方法1: エッジベースの検出
        edge_result = self._detect_by_edges(cv_image)
        if edge_result is not None:
            results.append(edge_result)

        # 方法2: 色差ベースの検出
        color_result = self._detect_by_hsv_threshold(cv_image)
        if color_result is not None:
            results.append(color_result)

        # 方法3: 輪郭ベースの検出
        contour_result = self._detect_by_contours(cv_image)
        if contour_result is not None:
            results.append(contour_result)

        # 方法4: 彩度・複雑度ベースの検出
        complexity_result = self._detect_by_grid_scoring(cv_image)
        if complexity_result is not None:
            results.append(complexity_result)

        # 方法5: 境界コントラストベースの検出
        boundary_result = self._detect_by_boundary_contrast(cv_image)
        if boundary_result is not None:
            results.append(boundary_result)

        # 方法6: フレーム解析ベースの検出（外→内）
        frame_result = self._detect_by_frame_analysis(cv_image)
        if frame_result is not None:
            results.append(frame_result)

        # 方法7: 水平帯の複雑度分析
        band_result = self._detect_by_band_complexity(cv_image)
        if band_result is not None:
            results.append(band_result)

        # 方法8: 水平線検出（Hough Transform）
        hlines_result = self._detect_by_horizontal_lines(cv_image)
        if hlines_result is not None:
            results.append(hlines_result)

        # 方法9: 中央拡張検出（複雑度ベース）
        center_result = self._detect_by_center_expansion(cv_image)
        if center_result is not None:
            results.append(center_result)

        # 方法10: 勾配の豊かさによる検出
        gradient_result = self._detect_by_gradient_richness(cv_image)
        if gradient_result is not None:
            results.append(gradient_result)

        # 結果がない場合はデフォルトの矩形を返す
        if not results:
            default_bbox = self._create_default_bbox(width, height)
            return DetectionResult(
                bbox=default_bbox,
                confidence=0.0,
                method="default",
            )

        # スコア = 信頼度 × 面積比（大きい領域を優先）
        image_area = width * height

        def score(r: DetectionResult) -> float:
            area_ratio = (r.bbox.width * r.bbox.height) / image_area
            return r.confidence * area_ratio

        best_result = max(results, key=score)
        return best_result

    def detect_with_candidates(
        self,
        image: Image.Image,
        max_candidates: int = 5,
    ) -> list[DetectionResult]:
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

        results: list[DetectionResult] = []

        # エッジベースの候補
        edge_result = self._detect_by_edges(cv_image)
        if edge_result is not None:
            results.append(edge_result)

        # 色差ベースの候補
        color_result = self._detect_by_hsv_threshold(cv_image)
        if color_result is not None:
            results.append(color_result)

        # 輪郭ベースの候補（複数）
        contour_results = self._detect_multiple_by_contours(cv_image, max_candidates)
        results.extend(contour_results)

        # 彩度・複雑度ベースの候補
        complexity_result = self._detect_by_grid_scoring(cv_image)
        if complexity_result is not None:
            results.append(complexity_result)

        # 境界コントラストベースの候補
        boundary_result = self._detect_by_boundary_contrast(cv_image)
        if boundary_result is not None:
            results.append(boundary_result)

        # フレーム解析ベースの候補
        frame_result = self._detect_by_frame_analysis(cv_image)
        if frame_result is not None:
            results.append(frame_result)

        # 水平帯の複雑度分析
        band_result = self._detect_by_band_complexity(cv_image)
        if band_result is not None:
            results.append(band_result)

        # 水平線検出（Hough Transform）
        hlines_result = self._detect_by_horizontal_lines(cv_image)
        if hlines_result is not None:
            results.append(hlines_result)

        # 中央拡張検出（複雑度ベース）
        center_result = self._detect_by_center_expansion(cv_image)
        if center_result is not None:
            results.append(center_result)

        # 勾配の豊かさによる検出
        gradient_result = self._detect_by_gradient_richness(cv_image)
        if gradient_result is not None:
            results.append(gradient_result)

        # スコア = 信頼度 × 面積比（大きい領域を優先）でソート
        image_area = width * height

        def score(r: DetectionResult) -> float:
            area_ratio = (r.bbox.width * r.bbox.height) / image_area
            return r.confidence * area_ratio

        results.sort(key=score, reverse=True)
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
    ) -> DetectionResult | None:
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
        valid_rects: list[tuple[int, int, int, int, float]] = []

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

    def _detect_by_hsv_threshold(
        self,
        cv_image: NDArray[np.uint8],
    ) -> DetectionResult | None:
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

        valid_rects: list[tuple[int, int, int, int, float]] = []

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
                    dx = abs(center_x - width / 2) / width
                    dy = abs(center_y - height / 2) / height
                    center_dist = dx + dy
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
            method="hsv_threshold",
        )

    def _detect_by_grid_scoring(
        self,
        cv_image: NDArray[np.uint8],
    ) -> DetectionResult | None:
        """彩度・複雑度ベースの領域検出。

        イラスト領域は通常:
        - 色の多様性が高い（彩度の分散が大きい）
        - エッジ密度が高い（詳細なテクスチャ）
        - カードの中央付近に位置する

        Args:
            cv_image: OpenCV形式の画像。

        Returns:
            検出結果。検出できない場合はNone。
        """
        height, width = cv_image.shape[:2]

        # グリッドサイズ（縦12分割、横8分割）
        grid_rows, grid_cols = 12, 8
        cell_h = height // grid_rows
        cell_w = width // grid_cols

        # 画像が小さすぎる場合は処理をスキップ
        if cell_h < 2 or cell_w < 2:
            return None

        # HSVに変換して彩度を取得
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1].astype(np.float32)

        # エッジ検出
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        # 各セルのスコアを計算
        scores = np.zeros((grid_rows, grid_cols))

        for row in range(grid_rows):
            for col in range(grid_cols):
                y1 = row * cell_h
                y2 = min((row + 1) * cell_h, height)
                x1 = col * cell_w
                x2 = min((col + 1) * cell_w, width)

                # 彩度の分散（色の多様性）
                cell_sat = saturation[y1:y2, x1:x2]
                sat_variance = np.var(cell_sat) / 255.0  # 正規化

                # エッジ密度（複雑さ）
                cell_edges = edges[y1:y2, x1:x2]
                edge_density = np.sum(cell_edges > 0) / cell_edges.size

                # 彩度の平均（カラフルさ）
                sat_mean = np.mean(cell_sat) / 255.0

                # 位置ボーナス（中央付近を優先）
                # TCGカードは上部10-25%にタイトル、下部30-40%にテキスト
                row_center = (row + 0.5) / grid_rows
                col_center = (col + 0.5) / grid_cols

                # イラストは縦方向で20-70%付近
                if 0.15 <= row_center <= 0.65:
                    vertical_bonus = 1.0
                elif 0.10 <= row_center <= 0.70:
                    vertical_bonus = 0.7
                else:
                    vertical_bonus = 0.3

                # 横方向は中央を優先
                horizontal_bonus = 1.0 - abs(col_center - 0.5) * 0.5

                # 総合スコア
                base_score = (
                    sat_variance * 0.3 +      # 色の多様性
                    edge_density * 0.3 +       # エッジ密度
                    sat_mean * 0.2 +           # 彩度
                    0.2                        # ベーススコア
                )
                scores[row, col] = base_score * vertical_bonus * horizontal_bonus

        # 最適な矩形領域を探索（スライディングウィンドウ）
        best_score = 0
        best_region = None

        # さまざまなウィンドウサイズで探索
        for win_h in range(3, min(8, grid_rows - 1)):  # 高さ3-7セル
            for win_w in range(4, min(7, grid_cols)):   # 幅4-6セル
                for start_row in range(grid_rows - win_h + 1):
                    for start_col in range(grid_cols - win_w + 1):
                        window_scores = scores[
                            start_row:start_row + win_h,
                            start_col:start_col + win_w
                        ]
                        avg_score = np.mean(window_scores)

                        # アスペクト比ボーナス（イラストは縦長〜正方形が多い）
                        aspect = (win_w * cell_w) / (win_h * cell_h)
                        if 0.6 <= aspect <= 1.2:
                            aspect_bonus = 1.0
                        elif 0.4 <= aspect <= 1.5:
                            aspect_bonus = 0.8
                        else:
                            aspect_bonus = 0.5

                        final_score = avg_score * aspect_bonus

                        if final_score > best_score:
                            best_score = final_score
                            best_region = (start_row, start_col, win_h, win_w)

        if best_region is None:
            return None

        start_row, start_col, win_h, win_w = best_region

        # ピクセル座標に変換
        x = start_col * cell_w
        y = start_row * cell_h
        w = win_w * cell_w
        h = win_h * cell_h

        # 境界チェック
        w = min(w, width - x)
        h = min(h, height - y)

        # 面積チェック
        area_ratio = (w * h) / (width * height)
        if not (self._min_area_ratio <= area_ratio <= self._max_area_ratio):
            return None

        # 信頼度（スコアを0-1に正規化）
        confidence = min(best_score * 2.0, 1.0)

        return DetectionResult(
            bbox=BoundingBox(x=x, y=y, width=w, height=h),
            confidence=confidence,
            method="grid_scoring",
        )

    def _detect_by_boundary_contrast(
        self,
        cv_image: NDArray[np.uint8],
    ) -> DetectionResult | None:
        """境界コントラストベースの領域検出。

        イラスト内部の複雑度が高く、境界（枠）が単色であることを利用。
        - 内部: 高い彩度分散・エッジ密度
        - 境界: 低い色分散（単色のカードフレーム）

        Args:
            cv_image: OpenCV形式の画像。

        Returns:
            検出結果。検出できない場合はNone。
        """
        height, width = cv_image.shape[:2]

        # 画像が小さすぎる場合はスキップ
        if height < 100 or width < 80:
            return None

        # HSVに変換
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1].astype(np.float32)

        # グレースケールとエッジ
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 30, 100)

        # 候補領域を探索（複数のサイズ・位置を試行）
        best_score = 0.0
        best_bbox = None

        # 探索範囲の設定（フレームに触れる位置まで探索）
        frame_margin = 5  # カードフレームの推定幅
        boundary_check_width = 15  # 境界チェック用の幅
        step = max(width // 25, 4)  # 探索ステップ（細かく）

        # TCGカードのイラスト領域は通常フレームぎりぎりまで
        for y1 in range(frame_margin, int(height * 0.25), step):
            for y2 in range(int(height * 0.50), height - frame_margin, step):
                for x1 in range(frame_margin, int(width * 0.15), step):
                    for x2 in range(int(width * 0.85), width - frame_margin, step):
                        # 面積チェック
                        w, h = x2 - x1, y2 - y1
                        area_ratio = (w * h) / (width * height)
                        if not (self._min_area_ratio <= area_ratio <= self._max_area_ratio):
                            continue

                        # アスペクト比チェック
                        aspect = w / h if h > 0 else 0
                        if not (0.5 <= aspect <= 1.5):
                            continue

                        # 内部複雑度を計算
                        inner_sat = saturation[y1:y2, x1:x2]
                        inner_edges = edges[y1:y2, x1:x2]

                        sat_variance = np.var(inner_sat) / 255.0
                        edge_density = np.sum(inner_edges > 0) / inner_edges.size
                        inner_complexity = sat_variance * 0.5 + edge_density * 0.5

                        # 境界の単色度を計算（4辺の外側をチェック）
                        boundary_uniformity = self._calculate_boundary_uniformity(
                            cv_image, x1, y1, x2, y2, boundary_check_width
                        )

                        # フレーム近接ボーナス（端に近いほど高スコア）
                        edge_proximity = (
                            (1.0 - x1 / width) * 0.25 +      # 左端に近い
                            (x2 / width) * 0.25 +             # 右端に近い
                            (1.0 - y1 / height) * 0.25 +      # 上端に近い
                            (y2 / height) * 0.25              # 下端に近い
                        )

                        # 総合スコア
                        # 内部複雑度 × 境界単色度 × フレーム近接度
                        score = (
                            inner_complexity * 0.3 +
                            boundary_uniformity * 0.4 +
                            edge_proximity * 0.3
                        )

                        if score > best_score:
                            best_score = score
                            best_bbox = (x1, y1, w, h)

        if best_bbox is None:
            return None

        x, y, w, h = best_bbox
        confidence = min(best_score * 3.0, 1.0)

        return DetectionResult(
            bbox=BoundingBox(x=x, y=y, width=w, height=h),
            confidence=confidence,
            method="boundary_contrast",
        )

    def _calculate_boundary_uniformity(
        self,
        cv_image: NDArray[np.uint8],
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        margin: int,
    ) -> float:
        """境界領域の単色度を計算。

        bboxの外側margin幅の領域の色の均一性を評価。

        Args:
            cv_image: 画像
            x1, y1, x2, y2: bbox座標
            margin: 境界の幅

        Returns:
            単色度スコア（0-1、高いほど単色）
        """
        height, width = cv_image.shape[:2]
        samples = []

        # 上辺の外側
        if y1 >= margin:
            top_strip = cv_image[y1 - margin:y1, x1:x2]
            if top_strip.size > 0:
                samples.append(top_strip.reshape(-1, 3))

        # 下辺の外側
        if y2 + margin <= height:
            bottom_strip = cv_image[y2:y2 + margin, x1:x2]
            if bottom_strip.size > 0:
                samples.append(bottom_strip.reshape(-1, 3))

        # 左辺の外側
        if x1 >= margin:
            left_strip = cv_image[y1:y2, x1 - margin:x1]
            if left_strip.size > 0:
                samples.append(left_strip.reshape(-1, 3))

        # 右辺の外側
        if x2 + margin <= width:
            right_strip = cv_image[y1:y2, x2:x2 + margin]
            if right_strip.size > 0:
                samples.append(right_strip.reshape(-1, 3))

        if not samples:
            return 0.0

        # 全サンプルを結合
        all_samples = np.vstack(samples).astype(np.float32)

        # 色の分散を計算（低いほど単色）
        color_variance = np.mean(np.var(all_samples, axis=0))

        # 分散を単色度スコアに変換（分散が低いほど高スコア）
        # 分散0→スコア1.0、分散2000以上→スコア0.0
        uniformity = max(0.0, 1.0 - color_variance / 2000.0)

        return uniformity

    def _detect_by_frame_analysis(
        self,
        cv_image: NDArray[np.uint8],
    ) -> DetectionResult | None:
        """フレーム解析ベースの領域検出（外→内アプローチ）。

        画像の端からフレーム色をサンプリングし、
        色が変化する境界を見つけてイラスト領域を特定。

        Args:
            cv_image: OpenCV形式の画像。

        Returns:
            検出結果。検出できない場合はNone。
        """
        height, width = cv_image.shape[:2]

        if height < 50 or width < 50:
            return None

        # 1. フレーム色をサンプリング（画像の端から）
        sample_width = max(5, min(20, width // 20))
        sample_height = max(5, min(20, height // 20))

        # 各辺からサンプル取得
        top_strip = cv_image[0:sample_height, :]
        bottom_strip = cv_image[height - sample_height:height, :]
        left_strip = cv_image[:, 0:sample_width]
        right_strip = cv_image[:, width - sample_width:width]

        # フレーム色の代表値（各辺の中央値）
        frame_colors = []
        for strip in [top_strip, bottom_strip, left_strip, right_strip]:
            median_color = np.median(strip.reshape(-1, 3), axis=0)
            frame_colors.append(median_color)
        frame_color = np.mean(frame_colors, axis=0)

        # 2. 各方向から色が変化する位置を探す
        color_threshold = 40  # 色差の閾値

        # 上から下へスキャン（イラスト上端を探す）
        top_boundary = self._find_color_boundary(
            cv_image, frame_color, color_threshold, "top"
        )

        # 下から上へスキャン（イラスト下端を探す）
        bottom_boundary = self._find_color_boundary(
            cv_image, frame_color, color_threshold, "bottom"
        )

        # 左から右へスキャン（イラスト左端を探す）
        left_boundary = self._find_color_boundary(
            cv_image, frame_color, color_threshold, "left"
        )

        # 右から左へスキャン（イラスト右端を探す）
        right_boundary = self._find_color_boundary(
            cv_image, frame_color, color_threshold, "right"
        )

        # 3. 境界が有効かチェック
        if (top_boundary >= bottom_boundary or
            left_boundary >= right_boundary):
            return None

        x = left_boundary
        y = top_boundary
        w = right_boundary - left_boundary
        h = bottom_boundary - top_boundary

        # 面積チェック
        area_ratio = (w * h) / (width * height)
        if not (self._min_area_ratio <= area_ratio <= self._max_area_ratio):
            return None

        # 4. 信頼度を計算（境界の明確さに基づく）
        # 境界付近の色差が大きいほど信頼度が高い
        confidence = self._calculate_boundary_sharpness(
            cv_image, x, y, x + w, y + h, frame_color
        )

        return DetectionResult(
            bbox=BoundingBox(x=x, y=y, width=w, height=h),
            confidence=confidence,
            method="frame_analysis",
        )

    def _find_color_boundary(
        self,
        cv_image: NDArray[np.uint8],
        frame_color: NDArray,
        threshold: float,
        direction: str,
    ) -> int:
        """フレーム色から変化する境界位置を探す。

        Args:
            cv_image: 画像
            frame_color: フレームの代表色
            threshold: 色差の閾値
            direction: 探索方向 ("top", "bottom", "left", "right")

        Returns:
            境界位置（ピクセル座標）
        """
        height, width = cv_image.shape[:2]
        scan_step = max(1, min(height, width) // 100)  # スキャン間隔

        if direction == "top":
            for y in range(0, height // 2, scan_step):
                row = cv_image[y, width // 4:width * 3 // 4]
                avg_color = np.mean(row.reshape(-1, 3), axis=0)
                color_diff = np.linalg.norm(avg_color - frame_color)
                if color_diff > threshold:
                    return max(0, y - scan_step)
            return height // 4

        elif direction == "bottom":
            for y in range(height - 1, height // 2, -scan_step):
                row = cv_image[y, width // 4:width * 3 // 4]
                avg_color = np.mean(row.reshape(-1, 3), axis=0)
                color_diff = np.linalg.norm(avg_color - frame_color)
                if color_diff > threshold:
                    return min(height, y + scan_step)
            return height * 3 // 4

        elif direction == "left":
            for x in range(0, width // 2, scan_step):
                col = cv_image[height // 4:height * 3 // 4, x]
                avg_color = np.mean(col.reshape(-1, 3), axis=0)
                color_diff = np.linalg.norm(avg_color - frame_color)
                if color_diff > threshold:
                    return max(0, x - scan_step)
            return width // 4

        else:  # right
            for x in range(width - 1, width // 2, -scan_step):
                col = cv_image[height // 4:height * 3 // 4, x]
                avg_color = np.mean(col.reshape(-1, 3), axis=0)
                color_diff = np.linalg.norm(avg_color - frame_color)
                if color_diff > threshold:
                    return min(width, x + scan_step)
            return width * 3 // 4

    def _calculate_boundary_sharpness(
        self,
        cv_image: NDArray[np.uint8],
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        frame_color: NDArray,
    ) -> float:
        """境界の明確さ（シャープネス）を計算。

        Args:
            cv_image: 画像
            x1, y1, x2, y2: bbox座標
            frame_color: フレームの代表色

        Returns:
            シャープネススコア（0-1）
        """
        height, width = cv_image.shape[:2]
        margin = 5
        diffs = []

        # 各辺の境界での色差を計算
        # 上辺
        if y1 >= margin:
            outside = cv_image[y1 - margin:y1, x1:x2]
            inside = cv_image[y1:y1 + margin, x1:x2]
            if outside.size > 0 and inside.size > 0:
                diff = np.linalg.norm(
                    np.mean(outside.reshape(-1, 3), axis=0) -
                    np.mean(inside.reshape(-1, 3), axis=0)
                )
                diffs.append(diff)

        # 下辺
        if y2 + margin <= height:
            outside = cv_image[y2:y2 + margin, x1:x2]
            inside = cv_image[y2 - margin:y2, x1:x2]
            if outside.size > 0 and inside.size > 0:
                diff = np.linalg.norm(
                    np.mean(outside.reshape(-1, 3), axis=0) -
                    np.mean(inside.reshape(-1, 3), axis=0)
                )
                diffs.append(diff)

        # 左辺
        if x1 >= margin:
            outside = cv_image[y1:y2, x1 - margin:x1]
            inside = cv_image[y1:y2, x1:x1 + margin]
            if outside.size > 0 and inside.size > 0:
                diff = np.linalg.norm(
                    np.mean(outside.reshape(-1, 3), axis=0) -
                    np.mean(inside.reshape(-1, 3), axis=0)
                )
                diffs.append(diff)

        # 右辺
        if x2 + margin <= width:
            outside = cv_image[y1:y2, x2:x2 + margin]
            inside = cv_image[y1:y2, x2 - margin:x2]
            if outside.size > 0 and inside.size > 0:
                diff = np.linalg.norm(
                    np.mean(outside.reshape(-1, 3), axis=0) -
                    np.mean(inside.reshape(-1, 3), axis=0)
                )
                diffs.append(diff)

        if not diffs:
            return 0.5

        # 平均色差をスコアに変換（差が大きいほど高スコア）
        avg_diff = np.mean(diffs)
        # 色差0→0.0、色差100以上→1.0
        sharpness = min(1.0, avg_diff / 100.0)

        return sharpness

    def _detect_by_band_complexity(
        self,
        cv_image: NDArray[np.uint8],
    ) -> DetectionResult | None:
        """水平帯の複雑度分析ベースの領域検出。

        画像を水平方向に分割し、各帯の「イラストらしさ」を計算。
        高複雑度が連続する領域をイラスト領域として特定。

        複雑度 = 彩度分散 × エッジ密度 × 色相多様性

        Args:
            cv_image: OpenCV形式の画像。

        Returns:
            検出結果。検出できない場合はNone。
        """
        height, width = cv_image.shape[:2]

        if height < 100 or width < 50:
            return None

        # 1. 水平帯に分割して複雑度を計算
        num_bands = 20
        band_height = height // num_bands
        band_scores = []

        # HSVとグレースケールを事前に計算
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        for i in range(num_bands):
            y1 = i * band_height
            y2 = min((i + 1) * band_height, height)

            # 帯のデータを取得
            band_hsv = hsv[y1:y2, :]
            band_edges = edges[y1:y2, :]

            # 彩度分散（色の多様性）
            sat_variance = np.var(band_hsv[:, :, 1].astype(np.float32)) / 255.0

            # エッジ密度（テクスチャの複雑さ）
            edge_density = np.sum(band_edges > 0) / band_edges.size

            # 色相の多様性（ユニークな色相の数）
            hue_binned = band_hsv[:, :, 0] // 18  # 18度単位（20ビン）
            hue_unique = len(np.unique(hue_binned)) / 20.0  # 正規化

            # 複雑度スコア
            score = sat_variance * 0.4 + edge_density * 0.4 + hue_unique * 0.2
            band_scores.append(score)

        # 2. 高複雑度の連続領域を特定
        # 閾値: 上位50%のスコア（より多くの帯を含める）
        threshold = np.percentile(band_scores, 50)

        # デバッグ: スコア分布を確認するためにソート
        sorted_scores = sorted(enumerate(band_scores), key=lambda x: x[1], reverse=True)

        # 連続した高スコア帯を見つける
        high_bands = [i for i, s in enumerate(band_scores) if s >= threshold]

        if not high_bands:
            return None

        # 連続領域を抽出（最大のギャップは3帯まで許容）
        continuous_regions = []
        current_region = [high_bands[0]]

        for i in range(1, len(high_bands)):
            if high_bands[i] - high_bands[i - 1] <= 3:  # 3帯までのギャップを許容
                current_region.append(high_bands[i])
            else:
                if len(current_region) >= 3:  # 最低3帯以上
                    continuous_regions.append(current_region)
                current_region = [high_bands[i]]

        if len(current_region) >= 3:
            continuous_regions.append(current_region)

        if not continuous_regions:
            # フォールバック: 最もスコアが高い帯を中心に拡張
            best_band_idx = sorted_scores[0][0]
            # 上下に3帯ずつ拡張
            start = max(0, best_band_idx - 3)
            end = min(num_bands - 1, best_band_idx + 3)
            best_region = list(range(start, end + 1))
        else:
            # 最大の連続領域を選択
            best_region = max(continuous_regions, key=len)

        # 3. 境界を決定
        top_band = best_region[0]
        bottom_band = best_region[-1]

        # 左右の境界も複雑度ベースで決定
        y1 = top_band * band_height
        y2 = min((bottom_band + 1) * band_height, height)

        # 左右の境界を見つける（列ごとの複雑度）
        left_x, right_x = self._find_horizontal_boundaries(
            cv_image[y1:y2, :], width
        )

        x = left_x
        y = y1
        w = right_x - left_x
        h = y2 - y1

        # 面積チェック
        area_ratio = (w * h) / (width * height)
        if not (self._min_area_ratio <= area_ratio <= self._max_area_ratio):
            return None

        # 4. 信頼度を計算（高複雑度帯の平均スコア）
        region_scores = [band_scores[i] for i in best_region]
        confidence = min(np.mean(region_scores) * 3.0, 1.0)

        return DetectionResult(
            bbox=BoundingBox(x=x, y=y, width=w, height=h),
            confidence=confidence,
            method="band_complexity",
        )

    def _detect_by_horizontal_lines(
        self,
        cv_image: NDArray[np.uint8],
    ) -> DetectionResult | None:
        """水平線検出ベースの領域検出（Hough Transform）。

        カードの区切り線（タイトル下、テキスト領域上）を検出して
        イラスト領域の境界として使用。

        Args:
            cv_image: OpenCV形式の画像。

        Returns:
            検出結果。検出できない場合はNone。
        """
        height, width = cv_image.shape[:2]

        if height < 100 or width < 50:
            return None

        # 1. グレースケール変換とエッジ検出
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        # 2. Hough Transform で直線を検出
        # minLineLength: 画像幅の40%以上（カード幅を横断する線）
        min_line_length = int(width * 0.4)
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=80,
            minLineLength=min_line_length,
            maxLineGap=20,
        )

        if lines is None:
            return None

        # 3. 水平に近い線を抽出（±10度以内）
        horizontal_lines = []
        angle_threshold = np.radians(10)

        for line in lines:
            x1, y1, x2, y2 = line[0]
            # 線の角度を計算
            dx = x2 - x1
            dy = y2 - y1
            angle = abs(np.arctan2(dy, dx))

            # 水平（0度）またはほぼ水平
            if angle < angle_threshold or angle > np.pi - angle_threshold:
                # 線のy座標（中点）
                y_mid = (y1 + y2) // 2
                # 線の長さ
                length = np.sqrt(dx**2 + dy**2)
                horizontal_lines.append((y_mid, length, x1, x2))

        if len(horizontal_lines) < 2:
            return None

        # 4. y座標でソートし、境界線候補を特定
        horizontal_lines.sort(key=lambda line: line[0])

        # 画像を3つの領域に分ける（上部、中央、下部）
        upper_region = int(height * 0.35)  # 上部35%（タイトル領域）
        lower_region = int(height * 0.60)  # 下部40%から（テキスト領域）

        # 上部境界線候補（タイトルとイラストの境界）
        upper_candidates = [
            line for line in horizontal_lines
            if height * 0.08 < line[0] < upper_region
        ]

        # 下部境界線候補（イラストとテキストの境界）
        lower_candidates = [
            line for line in horizontal_lines
            if lower_region < line[0] < height * 0.85
        ]

        # 5. 最も長い線を境界として選択
        top_y = None
        bottom_y = None

        if upper_candidates:
            # 最も長い上部境界線
            best_upper = max(upper_candidates, key=lambda line: line[1])
            top_y = best_upper[0]

        if lower_candidates:
            # 最も長い下部境界線
            best_lower = max(lower_candidates, key=lambda line: line[1])
            bottom_y = best_lower[0]

        # フォールバック: 境界線が見つからない場合はデフォルト値を使用
        if top_y is None:
            top_y = int(height * 0.15)
        if bottom_y is None:
            bottom_y = int(height * 0.70)

        # 境界が逆転している場合は無効
        if top_y >= bottom_y:
            return None

        # 6. 左右の境界を決定
        # 水平線の端点から左右の境界を推定
        left_x = int(width * 0.05)  # デフォルト: 5%のマージン
        right_x = int(width * 0.95)  # デフォルト: 95%

        # 検出された水平線の端点を考慮
        for y, _length, x1, x2 in horizontal_lines:
            if top_y <= y <= bottom_y:
                left_x = max(left_x, min(x1, x2))
                right_x = min(right_x, max(x1, x2))

        # 左右が逆転または範囲が狭すぎる場合
        if right_x - left_x < width * 0.3:
            left_x = int(width * 0.05)
            right_x = int(width * 0.95)

        x = left_x
        y = top_y
        w = right_x - left_x
        h = bottom_y - top_y

        # 面積チェック
        area_ratio = (w * h) / (width * height)
        if not (self._min_area_ratio <= area_ratio <= self._max_area_ratio):
            return None

        # 7. 信頼度を計算
        # 両方の境界線が検出された場合は高信頼度
        confidence = 0.5
        if upper_candidates and lower_candidates:
            # 線の長さに基づくボーナス
            avg_length = (best_upper[1] + best_lower[1]) / 2
            length_ratio = avg_length / width
            confidence = min(0.5 + length_ratio * 0.5, 1.0)

        return DetectionResult(
            bbox=BoundingBox(x=x, y=y, width=w, height=h),
            confidence=confidence,
            method="horizontal_lines",
        )

    def _find_horizontal_boundaries(
        self,
        cv_image: NDArray[np.uint8],
        full_width: int,
    ) -> tuple[int, int]:
        """水平方向の境界を複雑度ベースで決定。

        Args:
            cv_image: 対象領域の画像
            full_width: 元画像の幅

        Returns:
            (左端x, 右端x) のタプル
        """
        height, width = cv_image.shape[:2]

        # 列ごとの複雑度を計算
        num_cols = 10
        col_width = width // num_cols
        col_scores = []

        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        for i in range(num_cols):
            x1 = i * col_width
            x2 = min((i + 1) * col_width, width)

            col_hsv = hsv[:, x1:x2]
            col_edges = edges[:, x1:x2]

            sat_var = np.var(col_hsv[:, :, 1].astype(np.float32)) / 255.0
            edge_density = np.sum(col_edges > 0) / col_edges.size

            score = sat_var * 0.5 + edge_density * 0.5
            col_scores.append(score)

        # 閾値
        threshold = np.percentile(col_scores, 30)

        # 左端を見つける
        left_col = 0
        for i, s in enumerate(col_scores):
            if s >= threshold:
                left_col = i
                break

        # 右端を見つける
        right_col = num_cols - 1
        for i in range(num_cols - 1, -1, -1):
            if col_scores[i] >= threshold:
                right_col = i
                break

        left_x = max(0, left_col * col_width - col_width // 2)
        right_x = min(width, (right_col + 1) * col_width + col_width // 2)

        return left_x, right_x

    # ========================================
    # 統合検出用ヘルパーメソッド
    # ========================================

    def _get_horizontal_lines(
        self,
        cv_image: NDArray[np.uint8],
    ) -> list[tuple[int, float, int, int]]:
        """水平線のリストを取得。

        Args:
            cv_image: OpenCV形式の画像。

        Returns:
            水平線のリスト [(y, length, x1, x2), ...]
            y: 線のy座標（中点）
            length: 線の長さ
            x1, x2: 線の端点x座標
        """
        height, width = cv_image.shape[:2]

        if height < 100 or width < 50:
            return []

        # グレースケール変換とエッジ検出
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        # Hough Transform で直線を検出
        min_line_length = int(width * 0.4)
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=80,
            minLineLength=min_line_length,
            maxLineGap=20,
        )

        if lines is None:
            return []

        # 水平に近い線を抽出（±10度以内）
        horizontal_lines = []
        angle_threshold = np.radians(10)

        for line in lines:
            x1, y1, x2, y2 = line[0]
            dx = x2 - x1
            dy = y2 - y1
            angle = abs(np.arctan2(dy, dx))

            if angle < angle_threshold or angle > np.pi - angle_threshold:
                y_mid = (y1 + y2) // 2
                length = np.sqrt(dx**2 + dy**2)
                horizontal_lines.append((y_mid, length, min(x1, x2), max(x1, x2)))

        # y座標でソート
        horizontal_lines.sort(key=lambda line: line[0])
        return horizontal_lines

    def _get_band_scores(
        self,
        cv_image: NDArray[np.uint8],
        num_bands: int = 20,
    ) -> list[float]:
        """各水平帯の複雑度スコアを取得。

        Args:
            cv_image: OpenCV形式の画像。
            num_bands: 分割する帯の数。

        Returns:
            各帯の複雑度スコアのリスト。
        """
        height, width = cv_image.shape[:2]
        band_height = height // num_bands

        if band_height < 2:
            return [0.0] * num_bands

        # HSVとエッジを事前計算
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        band_scores = []
        for i in range(num_bands):
            y1 = i * band_height
            y2 = min((i + 1) * band_height, height)

            band_hsv = hsv[y1:y2, :]
            band_edges = edges[y1:y2, :]

            # 彩度分散
            sat_variance = np.var(band_hsv[:, :, 1].astype(np.float32)) / 255.0

            # エッジ密度
            edge_density = np.sum(band_edges > 0) / band_edges.size

            # 色相の多様性
            hue_binned = band_hsv[:, :, 0] // 18
            hue_unique = len(np.unique(hue_binned)) / 20.0

            # 複雑度スコア
            score = sat_variance * 0.4 + edge_density * 0.4 + hue_unique * 0.2
            band_scores.append(score)

        return band_scores

    def _get_frame_boundaries(
        self,
        cv_image: NDArray[np.uint8],
        y1: int,
        y2: int,
    ) -> tuple[int, int]:
        """指定y範囲内でフレーム色から左右境界を検出。

        Args:
            cv_image: OpenCV形式の画像。
            y1: 上端y座標。
            y2: 下端y座標。

        Returns:
            (left_x, right_x) のタプル。
        """
        height, width = cv_image.shape[:2]

        # フレーム色をサンプリング（画像の端から）
        sample_width = max(5, min(15, width // 25))

        left_strip = cv_image[y1:y2, 0:sample_width]
        right_strip = cv_image[y1:y2, width - sample_width:width]

        left_frame_color = np.median(left_strip.reshape(-1, 3), axis=0)
        right_frame_color = np.median(right_strip.reshape(-1, 3), axis=0)

        # 色差の閾値
        color_threshold = 35

        # 左から右へスキャン（左境界を探す）
        scan_step = max(1, width // 100)
        left_x = 0
        for x in range(0, width // 2, scan_step):
            col = cv_image[y1:y2, x]
            avg_color = np.mean(col.reshape(-1, 3), axis=0)
            color_diff = np.linalg.norm(avg_color - left_frame_color)
            if color_diff > color_threshold:
                left_x = max(0, x - scan_step)
                break

        # 右から左へスキャン（右境界を探す）
        right_x = width
        for x in range(width - 1, width // 2, -scan_step):
            col = cv_image[y1:y2, x]
            avg_color = np.mean(col.reshape(-1, 3), axis=0)
            color_diff = np.linalg.norm(avg_color - right_frame_color)
            if color_diff > color_threshold:
                right_x = min(width, x + scan_step)
                break

        # 境界が見つからなかった場合のフォールバック
        if left_x >= right_x - width // 4:
            left_x = int(width * 0.05)
            right_x = int(width * 0.95)

        return left_x, right_x

    def _detect_by_center_expansion(
        self,
        cv_image: NDArray[np.uint8],
    ) -> DetectionResult | None:
        """中央拡張による領域検出。

        画像の中央から外側へ広げていき、
        複雑度が急激に下がるポイントを境界とする。

        Args:
            cv_image: OpenCV形式の画像。

        Returns:
            検出結果。検出できない場合はNone。
        """
        height, width = cv_image.shape[:2]

        if height < 100 or width < 50:
            return None

        # 1. 水平帯（上下方向）の複雑度を計算
        num_h_bands = 20
        h_band_height = height // num_h_bands
        h_band_scores = self._get_band_scores(cv_image, num_h_bands)

        # 2. 垂直帯（左右方向）の複雑度を計算
        num_v_bands = 16
        v_band_width = width // num_v_bands
        v_band_scores = self._get_vertical_band_scores(cv_image, num_v_bands)

        # 3. 中央から上下左右へ拡張
        center_h_band = num_h_bands // 2
        center_v_band = num_v_bands // 2

        # 上方向の境界を探す（中央から上へ）
        top_band = self._find_complexity_drop(
            h_band_scores, center_h_band, direction="backward"
        )

        # 下方向の境界を探す（中央から下へ）
        bottom_band = self._find_complexity_drop(
            h_band_scores, center_h_band, direction="forward"
        )

        # 左方向の境界を探す（中央から左へ）
        left_band = self._find_complexity_drop(
            v_band_scores, center_v_band, direction="backward"
        )

        # 右方向の境界を探す（中央から右へ）
        right_band = self._find_complexity_drop(
            v_band_scores, center_v_band, direction="forward"
        )

        # 4. ピクセル座標に変換
        top_y = top_band * h_band_height
        bottom_y = min((bottom_band + 1) * h_band_height, height)
        left_x = left_band * v_band_width
        right_x = min((right_band + 1) * v_band_width, width)

        # 境界チェック
        if top_y >= bottom_y or left_x >= right_x:
            return None

        x = left_x
        y = top_y
        w = right_x - left_x
        h = bottom_y - top_y

        # 面積チェック
        area_ratio = (w * h) / (width * height)
        if not (self._min_area_ratio <= area_ratio <= self._max_area_ratio):
            return None

        # 5. 信頼度計算（検出領域の平均複雑度に基づく）
        region_h_scores = h_band_scores[top_band:bottom_band + 1]
        region_v_scores = v_band_scores[left_band:right_band + 1]

        avg_complexity = (np.mean(region_h_scores) + np.mean(region_v_scores)) / 2
        overall_avg = (np.mean(h_band_scores) + np.mean(v_band_scores)) / 2

        if overall_avg > 0:
            confidence = min(avg_complexity / overall_avg, 1.0)
        else:
            confidence = 0.5

        return DetectionResult(
            bbox=BoundingBox(x=x, y=y, width=w, height=h),
            confidence=confidence,
            method="center_expansion",
        )

    def _get_vertical_band_scores(
        self,
        cv_image: NDArray[np.uint8],
        num_bands: int = 16,
    ) -> list[float]:
        """各垂直帯の複雑度スコアを取得。

        Args:
            cv_image: OpenCV形式の画像。
            num_bands: 分割する帯の数。

        Returns:
            各帯の複雑度スコアのリスト。
        """
        height, width = cv_image.shape[:2]
        band_width = width // num_bands

        if band_width < 2:
            return [0.0] * num_bands

        # HSVとエッジを事前計算
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        band_scores = []
        for i in range(num_bands):
            x1 = i * band_width
            x2 = min((i + 1) * band_width, width)

            band_hsv = hsv[:, x1:x2]
            band_edges = edges[:, x1:x2]

            # 彩度分散
            sat_variance = np.var(band_hsv[:, :, 1].astype(np.float32)) / 255.0

            # エッジ密度
            edge_density = np.sum(band_edges > 0) / band_edges.size

            # 色相の多様性
            hue_binned = band_hsv[:, :, 0] // 18
            hue_unique = len(np.unique(hue_binned)) / 20.0

            # 複雑度スコア
            score = sat_variance * 0.4 + edge_density * 0.4 + hue_unique * 0.2
            band_scores.append(score)

        return band_scores

    def _find_complexity_drop(
        self,
        scores: list[float],
        start_idx: int,
        direction: str,
    ) -> int:
        """複雑度が急激に下がるポイントを探す。

        Args:
            scores: 複雑度スコアのリスト。
            start_idx: 開始インデックス（中央）。
            direction: "forward"（インデックス増加）または "backward"（減少）。

        Returns:
            境界のインデックス。
        """
        n = len(scores)
        if n == 0:
            return start_idx

        # 中央付近の複雑度を基準にする
        window = 3
        center_start = max(0, start_idx - window)
        center_end = min(n, start_idx + window + 1)
        center_avg = np.mean(scores[center_start:center_end])

        # 急激な低下の閾値（中央の50%以下になったら境界）
        drop_threshold = center_avg * 0.5

        if direction == "forward":
            # 前方（インデックス増加）へ探索
            boundary = n - 1
            for i in range(start_idx, n):
                if scores[i] < drop_threshold:
                    # 少し手前を境界とする
                    boundary = max(start_idx, i - 1)
                    break
            return boundary
        else:
            # 後方（インデックス減少）へ探索
            boundary = 0
            for i in range(start_idx, -1, -1):
                if scores[i] < drop_threshold:
                    # 少し先を境界とする
                    boundary = min(start_idx, i + 1)
                    break
            return boundary

    def _detect_by_gradient_richness(
        self,
        cv_image: NDArray[np.uint8],
    ) -> DetectionResult | None:
        """勾配の豊かさによる領域検出。

        モノクロ画像の隣接画素間の微分を計算し、
        その変化の豊かさ（分散）が高い領域をイラストとして検出。
        中央から1pxずつ拡張し、中央平均に対して急激に下がった地点を境界とする。

        Args:
            cv_image: OpenCV形式の画像。

        Returns:
            検出結果。検出できない場合はNone。
        """
        height, width = cv_image.shape[:2]

        if height < 50 or width < 50:
            return None

        # 1. グレースケール変換
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # 2. 勾配（微分）を計算（Sobelフィルタ）
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        # 勾配の大きさ
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)

        # 3. 各垂直ライン（x座標ごと）の勾配の豊かさ（分散）を計算
        vertical_richness = []
        for x in range(width):
            line = grad_magnitude[:, x]
            richness = np.var(line)
            vertical_richness.append(richness)

        # 4. 各水平ライン（y座標ごと）の勾配の豊かさを計算
        horizontal_richness = []
        for y in range(height):
            line = grad_magnitude[y, :]
            richness = np.var(line)
            horizontal_richness.append(richness)

        # 5. 中央領域の平均を基準に閾値を計算
        center_x = width // 2
        center_y = height // 2

        # 中央付近（±20%）の平均豊かさを計算
        v_center_start = int(width * 0.3)
        v_center_end = int(width * 0.7)
        h_center_start = int(height * 0.3)
        h_center_end = int(height * 0.7)

        v_center_avg = np.mean(vertical_richness[v_center_start:v_center_end])
        h_center_avg = np.mean(horizontal_richness[h_center_start:h_center_end])

        # 閾値: 中央平均の25%以下になったら境界
        v_threshold = v_center_avg * 0.25
        h_threshold = h_center_avg * 0.25

        # 6. 中央から拡張して境界を探す（スムージング付き）
        # ノイズ対策: 連続して閾値以下になったら境界とする
        consecutive_required = 2

        # 左境界（中央から左へ）
        left_x = 0
        consecutive = 0
        for x in range(center_x, -1, -1):
            if vertical_richness[x] < v_threshold:
                consecutive += 1
                if consecutive >= consecutive_required:
                    left_x = min(x + consecutive_required, center_x)
                    break
            else:
                consecutive = 0

        # 右境界（中央から右へ）
        right_x = width - 1
        consecutive = 0
        for x in range(center_x, width):
            if vertical_richness[x] < v_threshold:
                consecutive += 1
                if consecutive >= consecutive_required:
                    right_x = max(x - consecutive_required, center_x)
                    break
            else:
                consecutive = 0

        # 上境界（中央から上へ）
        top_y = 0
        consecutive = 0
        for y in range(center_y, -1, -1):
            if horizontal_richness[y] < h_threshold:
                consecutive += 1
                if consecutive >= consecutive_required:
                    top_y = min(y + consecutive_required, center_y)
                    break
            else:
                consecutive = 0

        # 下境界（中央から下へ）
        bottom_y = height - 1
        consecutive = 0
        for y in range(center_y, height):
            if horizontal_richness[y] < h_threshold:
                consecutive += 1
                if consecutive >= consecutive_required:
                    bottom_y = max(y - consecutive_required, center_y)
                    break
            else:
                consecutive = 0

        # 境界チェック
        if left_x >= right_x or top_y >= bottom_y:
            return None

        x = left_x
        y = top_y
        w = right_x - left_x
        h = bottom_y - top_y

        # 面積チェック
        area_ratio = (w * h) / (width * height)
        if not (self._min_area_ratio <= area_ratio <= self._max_area_ratio):
            return None

        # 7. 信頼度計算（検出領域の平均豊かさ / 中央平均）
        region_v_richness = vertical_richness[left_x:right_x + 1]
        region_h_richness = horizontal_richness[top_y:bottom_y + 1]

        avg_region = (np.mean(region_v_richness) + np.mean(region_h_richness)) / 2
        center_avg = (v_center_avg + h_center_avg) / 2

        if center_avg > 0:
            confidence = min(avg_region / center_avg, 1.0)
        else:
            confidence = 0.5

        return DetectionResult(
            bbox=BoundingBox(x=x, y=y, width=w, height=h),
            confidence=confidence,
            method="gradient_richness",
        )

    def _detect_by_contours(
        self,
        cv_image: NDArray[np.uint8],
    ) -> DetectionResult | None:
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
    ) -> list[DetectionResult]:
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

        results: list[DetectionResult] = []

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
