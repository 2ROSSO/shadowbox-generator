"""テンプレートエディタ（手動領域選択）モジュール。

このモジュールは、matplotlibを使用してユーザーが
インタラクティブにイラスト領域を選択するGUIを提供します。
"""

from pathlib import Path
from typing import Callable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.widgets import RectangleSelector
from numpy.typing import NDArray
from PIL import Image

from shadowbox.config.template import BoundingBox, CardTemplate


class TemplateEditor:
    """テンプレートエディタ。

    matplotlibのRectangleSelectorを使用して、ユーザーが
    画像上で矩形領域を選択できるGUIを提供します。

    Attributes:
        image: 編集対象の画像。
        selected_bbox: 選択された領域。

    Example:
        >>> editor = TemplateEditor()
        >>> bbox = editor.select_region(image)
        >>> if bbox:
        ...     template = editor.create_template("pokemon", "standard", bbox, image)
    """

    def __init__(self) -> None:
        """エディタを初期化。"""
        self._image: Optional[Image.Image] = None
        self._selected_bbox: Optional[BoundingBox] = None
        self._fig: Optional[plt.Figure] = None
        self._ax: Optional[plt.Axes] = None
        self._selector: Optional[RectangleSelector] = None
        self._current_rect: Optional[Rectangle] = None

    @property
    def selected_bbox(self) -> Optional[BoundingBox]:
        """選択されたバウンディングボックス。"""
        return self._selected_bbox

    def select_region(
        self,
        image: Image.Image,
        title: str = "イラスト領域を選択（ドラッグで矩形を描画）",
        initial_bbox: Optional[BoundingBox] = None,
    ) -> Optional[BoundingBox]:
        """画像上で領域を選択。

        matplotlibのウィンドウが開き、ユーザーがドラッグで
        矩形を選択できます。ウィンドウを閉じると選択が確定します。

        Args:
            image: 領域選択対象の画像。
            title: ウィンドウタイトル。
            initial_bbox: 初期表示する矩形（オプション）。

        Returns:
            選択された領域。キャンセルされた場合はNone。

        Example:
            >>> editor = TemplateEditor()
            >>> bbox = editor.select_region(card_image)
        """
        self._image = image
        self._selected_bbox = initial_bbox

        # 画像をNumPy配列に変換
        img_array = np.array(image)

        # Figure作成
        self._fig, self._ax = plt.subplots(1, 1, figsize=(10, 10))
        self._ax.imshow(img_array)
        self._ax.set_title(title)

        # 初期矩形を描画
        if initial_bbox is not None:
            self._draw_rect(initial_bbox)

        # RectangleSelectorを設定
        self._selector = RectangleSelector(
            self._ax,
            self._on_select,
            useblit=True,
            button=[1],  # 左クリックのみ
            minspanx=5,
            minspany=5,
            spancoords="pixels",
            interactive=True,
            props=dict(facecolor="red", edgecolor="red", alpha=0.3, linewidth=2),
        )

        # 操作説明を追加
        self._fig.text(
            0.5,
            0.02,
            "ドラッグで選択 | Enterで確定 | Escでキャンセル",
            ha="center",
            fontsize=10,
        )

        # キーイベントを設定
        self._fig.canvas.mpl_connect("key_press_event", self._on_key_press)

        # 表示（ブロッキング）
        plt.show()

        return self._selected_bbox

    def _on_select(self, eclick: any, erelease: any) -> None:
        """矩形選択時のコールバック。

        Args:
            eclick: クリック開始イベント。
            erelease: クリック終了イベント。
        """
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)

        # 座標を正規化（左上と右下）
        x = min(x1, x2)
        y = min(y1, y2)
        width = abs(x2 - x1)
        height = abs(y2 - y1)

        if width > 0 and height > 0:
            self._selected_bbox = BoundingBox(x=x, y=y, width=width, height=height)
            self._draw_rect(self._selected_bbox)
            print(f"選択: {self._selected_bbox}")

    def _draw_rect(self, bbox: BoundingBox) -> None:
        """矩形を描画。

        Args:
            bbox: 描画するバウンディングボックス。
        """
        # 既存の矩形を削除
        if self._current_rect is not None:
            self._current_rect.remove()

        # 新しい矩形を描画
        self._current_rect = Rectangle(
            (bbox.x, bbox.y),
            bbox.width,
            bbox.height,
            linewidth=2,
            edgecolor="lime",
            facecolor="none",
            linestyle="--",
        )
        self._ax.add_patch(self._current_rect)
        self._fig.canvas.draw()

    def _on_key_press(self, event: any) -> None:
        """キー押下イベントのハンドラ。

        Args:
            event: キーイベント。
        """
        if event.key == "enter":
            # 確定
            if self._selected_bbox is not None:
                print("領域を確定しました")
                plt.close(self._fig)
        elif event.key == "escape":
            # キャンセル
            self._selected_bbox = None
            print("キャンセルしました")
            plt.close(self._fig)

    def create_template(
        self,
        name: str,
        game: str,
        bbox: BoundingBox,
        image: Image.Image,
        description: str = "",
    ) -> CardTemplate:
        """選択した領域からテンプレートを作成。

        Args:
            name: テンプレート名。
            game: ゲーム名。
            bbox: イラスト領域。
            image: 元画像（サイズ取得用）。
            description: 説明（オプション）。

        Returns:
            作成されたCardTemplate。

        Example:
            >>> template = editor.create_template(
            ...     "standard",
            ...     "pokemon",
            ...     selected_bbox,
            ...     card_image,
            ... )
        """
        width, height = image.size

        return CardTemplate(
            name=name,
            game=game,
            illustration_area=bbox,
            card_width=width,
            card_height=height,
            description=description,
        )


class QuickRegionSelector:
    """簡易領域選択ツール。

    ノートブック環境でも使いやすい、シンプルな領域選択機能。
    """

    @staticmethod
    def select(
        image: Image.Image,
        title: str = "領域を選択",
    ) -> Optional[BoundingBox]:
        """画像上で領域を選択。

        Args:
            image: 選択対象の画像。
            title: ウィンドウタイトル。

        Returns:
            選択された領域。キャンセルされた場合はNone。

        Example:
            >>> bbox = QuickRegionSelector.select(image)
        """
        editor = TemplateEditor()
        return editor.select_region(image, title)

    @staticmethod
    def select_with_preview(
        image: Image.Image,
        title: str = "領域を選択",
    ) -> Tuple[Optional[BoundingBox], Optional[Image.Image]]:
        """領域を選択し、切り抜きプレビューも返す。

        Args:
            image: 選択対象の画像。
            title: ウィンドウタイトル。

        Returns:
            (選択された領域, 切り抜いた画像) のタプル。

        Example:
            >>> bbox, cropped = QuickRegionSelector.select_with_preview(image)
            >>> if cropped:
            ...     cropped.show()
        """
        bbox = QuickRegionSelector.select(image, title)

        if bbox is None:
            return None, None

        # 切り抜き
        cropped = image.crop((bbox.x, bbox.y, bbox.x + bbox.width, bbox.y + bbox.height))
        return bbox, cropped


def select_illustration_region(image: Image.Image) -> Optional[BoundingBox]:
    """イラスト領域を手動選択するユーティリティ関数。

    Args:
        image: 選択対象の画像。

    Returns:
        選択された領域。キャンセルされた場合はNone。

    Example:
        >>> bbox = select_illustration_region(card_image)
        >>> if bbox:
        ...     print(f"Selected: {bbox}")
    """
    return QuickRegionSelector.select(image)
