"""PyQt6ネイティブの領域選択オーバーレイ。

ImagePreview上に透過オーバーレイを描画し、
マウスドラッグで矩形領域を選択できます。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt6.QtCore import QPoint, QRect, Qt, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QPainter, QPen
from PyQt6.QtWidgets import QWidget

if TYPE_CHECKING:
    from PyQt6.QtGui import QMouseEvent, QPaintEvent


class RegionSelector(QWidget):
    """画像上に重ねて表示する矩形選択オーバーレイ。

    ドラッグで矩形領域を選択し、選択完了時に region_selected を送出。
    Right-click でリセット。

    Signals:
        region_selected: (x, y, w, h) 画像座標系の選択領域。
        region_cleared: 選択がクリアされたとき。
    """

    region_selected = pyqtSignal(int, int, int, int)
    region_cleared = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        self.setMouseTracking(True)

        self._start: QPoint | None = None
        self._end: QPoint | None = None
        self._selection: QRect | None = None
        self._image_rect: QRect = QRect()
        self._image_size: tuple[int, int] = (0, 0)
        self._active = False

    def set_image_rect(self, rect: QRect, image_size: tuple[int, int]) -> None:
        """表示中の画像領域とオリジナルサイズを設定。

        Args:
            rect: ウィジェット座標系での画像表示領域。
            image_size: (width, height) オリジナル画像のピクセルサイズ。
        """
        self._image_rect = rect
        self._image_size = image_size

    def set_active(self, active: bool) -> None:
        """選択モードの有効/無効を切り替え。"""
        self._active = active
        if not active:
            self.setCursor(Qt.CursorShape.ArrowCursor)
        else:
            self.setCursor(Qt.CursorShape.CrossCursor)

    def clear_selection(self) -> None:
        """選択をクリア。"""
        self._selection = None
        self._start = None
        self._end = None
        self.update()
        self.region_cleared.emit()

    def set_selection(self, x: int, y: int, w: int, h: int) -> None:
        """画像座標系で選択領域を設定（復元用）。シグナルは発行しない。"""
        ir = self._image_rect
        if ir.isEmpty() or self._image_size == (0, 0):
            return
        iw, ih = self._image_size
        scale_x = ir.width() / iw
        scale_y = ir.height() / ih
        wx = int(ir.x() + x * scale_x)
        wy = int(ir.y() + y * scale_y)
        ww = int(w * scale_x)
        wh = int(h * scale_y)
        self._selection = QRect(wx, wy, ww, wh)
        self._start = None
        self._end = None
        self.update()

    def get_selection(self) -> tuple[int, int, int, int] | None:
        """画像座標系での選択領域を取得。

        Returns:
            (x, y, width, height) またはNone。
        """
        if self._selection is None or self._image_rect.isEmpty():
            return None
        return self._widget_to_image(self._selection)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if not self._active:
            return
        if event.button() == Qt.MouseButton.LeftButton:
            self._start = self._clamp_to_image(event.pos())
            self._end = self._start
            self._selection = None
            self.update()
        elif event.button() == Qt.MouseButton.RightButton:
            self.clear_selection()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if not self._active or self._start is None:
            return
        self._end = self._clamp_to_image(event.pos())
        self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if not self._active or self._start is None:
            return
        if event.button() == Qt.MouseButton.LeftButton:
            self._end = self._clamp_to_image(event.pos())
            rect = QRect(self._start, self._end).normalized()
            if rect.width() > 4 and rect.height() > 4:
                self._selection = rect
                img_coords = self._widget_to_image(rect)
                if img_coords:
                    self.region_selected.emit(*img_coords)
            self._start = None
            self._end = None
            self.update()

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Draw current drag
        if self._start is not None and self._end is not None:
            rect = QRect(self._start, self._end).normalized()
            painter.setPen(QPen(QColor(100, 180, 255), 2, Qt.PenStyle.DashLine))
            painter.setBrush(QBrush(QColor(100, 180, 255, 40)))
            painter.drawRect(rect)

        # Draw confirmed selection
        if self._selection is not None:
            painter.setPen(QPen(QColor(80, 200, 120), 2))
            painter.setBrush(QBrush(QColor(80, 200, 120, 30)))
            painter.drawRect(self._selection)

            # Dim outside region
            overlay = QColor(0, 0, 0, 100)
            ir = self._image_rect
            sel = self._selection
            # Top
            painter.fillRect(
                QRect(ir.left(), ir.top(), ir.width(), sel.top() - ir.top()),
                overlay,
            )
            # Bottom
            painter.fillRect(
                QRect(
                    ir.left(), sel.bottom(), ir.width(), ir.bottom() - sel.bottom()
                ),
                overlay,
            )
            # Left
            painter.fillRect(
                QRect(
                    ir.left(), sel.top(), sel.left() - ir.left(), sel.height()
                ),
                overlay,
            )
            # Right
            painter.fillRect(
                QRect(
                    sel.right(), sel.top(), ir.right() - sel.right(), sel.height()
                ),
                overlay,
            )

        painter.end()

    def _clamp_to_image(self, pos: QPoint) -> QPoint:
        x = max(self._image_rect.left(), min(pos.x(), self._image_rect.right()))
        y = max(self._image_rect.top(), min(pos.y(), self._image_rect.bottom()))
        return QPoint(x, y)

    def _widget_to_image(self, rect: QRect) -> tuple[int, int, int, int] | None:
        ir = self._image_rect
        if ir.width() == 0 or ir.height() == 0:
            return None
        iw, ih = self._image_size
        scale_x = iw / ir.width()
        scale_y = ih / ir.height()
        x = int((rect.x() - ir.x()) * scale_x)
        y = int((rect.y() - ir.y()) * scale_y)
        w = int(rect.width() * scale_x)
        h = int(rect.height() * scale_y)
        x = max(0, min(x, iw))
        y = max(0, min(y, ih))
        w = max(1, min(w, iw - x))
        h = max(1, min(h, ih - y))
        return (x, y, w, h)
