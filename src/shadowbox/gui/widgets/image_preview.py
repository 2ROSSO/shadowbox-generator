"""画像プレビューウィジェット。

元画像・深度マップ・レイヤーラベルをタブで切り替え表示し、
リージョン選択オーバーレイを重ねます。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from PyQt6.QtCore import QRect, Qt, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QResizeEvent
from PyQt6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from shadowbox.gui.i18n import tr
from shadowbox.gui.region_selector import RegionSelector

if TYPE_CHECKING:
    from PIL import Image


class ImagePreview(QWidget):
    """画像プレビュー + タブ切替 + リージョン選択オーバーレイ。

    Signals:
        region_selected: (x, y, w, h) 選択領域。
        region_cleared: 選択クリア。
    """

    region_selected = pyqtSignal(int, int, int, int)
    region_cleared = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._pixmaps: dict[str, QPixmap] = {}
        self._current_image_size: tuple[int, int] = (0, 0)
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Tab buttons (simple toggle row)
        tab_row = QHBoxLayout()
        self._tab_buttons: dict[str, QLabel] = {}
        self._tab_keys = ["original", "depth", "layers"]
        self._tab_tr_keys = ["tab.original", "tab.depth", "tab.layers"]
        for key, tr_key in zip(self._tab_keys, self._tab_tr_keys, strict=True):
            btn = QLabel(tr(tr_key))
            btn.setAlignment(Qt.AlignmentFlag.AlignCenter)
            btn.setStyleSheet(
                "padding: 4px 12px; border: 1px solid #444; border-radius: 3px;"
            )
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.mousePressEvent = lambda e, k=key: self._switch_tab(k)
            self._tab_buttons[key] = btn
            tab_row.addWidget(btn)

        # Region toggle
        self._region_check = QCheckBox(tr("tab.region_select"))
        self._region_check.toggled.connect(self._on_region_toggle)
        tab_row.addWidget(self._region_check)

        layout.addLayout(tab_row)

        # Stacked image labels
        self._stack = QStackedWidget()
        self._labels: dict[str, QLabel] = {}
        for key in self._tab_keys:
            lbl = QLabel(tr("tab.no_image"))
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setMinimumSize(300, 300)
            lbl.setStyleSheet(
                "background-color: #2a2a2a; color: #888; border: 1px solid #444;"
            )
            self._labels[key] = lbl
            self._stack.addWidget(lbl)
        layout.addWidget(self._stack, stretch=1)

        # Region selector overlay
        self._region_selector = RegionSelector(self._stack)
        self._region_selector.region_selected.connect(self.region_selected)
        self._region_selector.region_cleared.connect(self.region_cleared)
        self._region_selector.setGeometry(self._stack.rect())

        self._current_tab = "original"
        self._update_tab_styles()

    def _switch_tab(self, key: str) -> None:
        idx = list(self._labels.keys()).index(key)
        self._stack.setCurrentIndex(idx)
        self._current_tab = key
        self._update_tab_styles()
        # Re-sync region selector when switching to original tab
        if key == "original" and "original" in self._pixmaps:
            self._display_pixmap("original", self._pixmaps["original"])

    def _update_tab_styles(self) -> None:
        for k, btn in self._tab_buttons.items():
            if k == self._current_tab:
                btn.setStyleSheet(
                    "padding: 4px 12px; border: 1px solid #666; "
                    "border-radius: 3px; background-color: #3a3a3a; color: #fff;"
                )
            else:
                btn.setStyleSheet(
                    "padding: 4px 12px; border: 1px solid #444; "
                    "border-radius: 3px; color: #999;"
                )

    def _on_region_toggle(self, checked: bool) -> None:
        self._region_selector.set_active(checked)
        if checked:
            self._switch_tab("original")
        else:
            self._region_selector.clear_selection()

    def set_image(self, image: Image.Image) -> None:
        """元画像をセット。"""
        self._current_image_size = (image.width, image.height)
        pixmap = self._pil_to_pixmap(image)
        self._pixmaps["original"] = pixmap
        self._display_pixmap("original", pixmap)
        self._switch_tab("original")

    def set_depth_map(self, depth_map: np.ndarray) -> None:
        """深度マップをセット（正規化→カラーマップ表示）。"""
        normalized = depth_map.copy()
        dmin, dmax = normalized.min(), normalized.max()
        if dmax > dmin:
            normalized = (normalized - dmin) / (dmax - dmin)
        vis = (normalized * 255).astype(np.uint8)
        # Viridis-like coloring (simple grayscale for now)
        rgb = np.stack([vis, vis, vis], axis=-1)
        h, w = rgb.shape[:2]
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        self._pixmaps["depth"] = pixmap
        self._display_pixmap("depth", pixmap)

    def set_labels(self, labels: np.ndarray, image: np.ndarray) -> None:
        """レイヤーラベルをセット（色分け表示）。"""
        k = labels.max() + 1
        # Simple color palette
        palette = np.array([
            [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0],
            [255, 0, 255], [0, 255, 255], [128, 128, 0], [128, 0, 128],
            [0, 128, 128], [255, 128, 0],
        ], dtype=np.uint8)
        h, w = labels.shape
        vis = np.zeros((h, w, 3), dtype=np.uint8)
        for i in range(k):
            mask = labels == i
            color = palette[i % len(palette)]
            vis[mask] = (image[mask].astype(float) * 0.5 + color * 0.5).astype(np.uint8)
        qimg = QImage(vis.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        self._pixmaps["layers"] = pixmap
        self._display_pixmap("layers", pixmap)

    def clear(self) -> None:
        """全画像をクリア。"""
        self._pixmaps.clear()
        for lbl in self._labels.values():
            lbl.setPixmap(QPixmap())
            lbl.setText(tr("tab.no_image"))
        self._region_selector.clear_selection()

    def _display_pixmap(self, key: str, pixmap: QPixmap) -> None:
        lbl = self._labels[key]
        target_size = lbl.size()
        scaled = pixmap.scaled(
            target_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        lbl.setPixmap(scaled)

        # Update region selector geometry
        if key == "original" and self._current_image_size[0] > 0:
            self._update_region_rect(lbl, scaled)

    def _update_region_rect(self, lbl: QLabel, scaled_pixmap: QPixmap) -> None:
        lbl_rect = lbl.rect()
        pw, ph = scaled_pixmap.width(), scaled_pixmap.height()
        x = (lbl_rect.width() - pw) // 2
        y = (lbl_rect.height() - ph) // 2
        image_rect = QRect(x, y, pw, ph)
        self._region_selector.set_image_rect(image_rect, self._current_image_size)

    def retranslate(self) -> None:
        """言語変更時にUI文字列を更新。"""
        for key, tr_key in zip(self._tab_keys, self._tab_tr_keys, strict=True):
            self._tab_buttons[key].setText(tr(tr_key))
        self._region_check.setText(tr("tab.region_select"))
        # Update placeholder text for tabs without pixmaps
        for key, lbl in self._labels.items():
            if key not in self._pixmaps:
                lbl.setText(tr("tab.no_image"))

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self._region_selector.setGeometry(self._stack.rect())
        # Re-display current pixmaps
        for key, pixmap in self._pixmaps.items():
            self._display_pixmap(key, pixmap)

    @staticmethod
    def _pil_to_pixmap(image: Image.Image) -> QPixmap:
        arr = np.array(image.convert("RGB"))
        h, w, _ = arr.shape
        qimg = QImage(arr.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(qimg)
