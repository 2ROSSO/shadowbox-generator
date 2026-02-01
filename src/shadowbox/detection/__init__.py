"""イラスト領域検出モジュール。

自動検出と手動選択の両方の機能を提供します。
"""

from shadowbox.detection.region import (
    DetectionResult,
    RegionDetector,
    detect_illustration_region,
)

__all__ = [
    "DetectionResult",
    "RegionDetector",
    "detect_illustration_region",
]
