"""テンプレートエディタ（手動領域選択）モジュール。

このモジュールは、matplotlibを使用してユーザーが
インタラクティブにイラスト領域を選択するGUIを提供します。
"""


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image

from shadowbox.config.template import BoundingBox, CardTemplate


class TemplateEditor:
    """テンプレートエディタ。

    2点クリックで矩形領域を選択できるGUIを提供します。

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
        self._image: Image.Image | None = None
        self._selected_bbox: BoundingBox | None = None
        self._fig: plt.Figure | None = None
        self._ax: plt.Axes | None = None
        self._current_rect: Rectangle | None = None
        self._click_points: list = []
        self._temp_marker: any | None = None

    @property
    def selected_bbox(self) -> BoundingBox | None:
        """選択されたバウンディングボックス。"""
        return self._selected_bbox

    def select_region(
        self,
        image: Image.Image,
        title: str = "イラスト領域を選択",
        initial_bbox: BoundingBox | None = None,
    ) -> BoundingBox | None:
        """画像上で領域を選択。

        matplotlibのウィンドウが開き、ユーザーが2点クリックで
        矩形を選択できます。

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
        self._click_points = []

        # 画像をNumPy配列に変換
        img_array = np.array(image)

        # Figure作成
        self._fig, self._ax = plt.subplots(1, 1, figsize=(10, 10))
        self._ax.imshow(img_array)
        self._ax.set_title(title)

        # 初期矩形を描画
        if initial_bbox is not None:
            self._draw_rect(initial_bbox)

        # 操作説明を追加
        help_text = "左クリック2回で矩形を指定（左上→右下） | 右クリック:リセット | 閉じて確定"
        self._fig.text(0.5, 0.02, help_text, ha="center", fontsize=10)

        # クリックイベントを設定
        self._fig.canvas.mpl_connect("button_press_event", self._on_click)

        # 表示（ブロッキング）
        plt.show()

        return self._selected_bbox

    def _on_click(self, event: any) -> None:
        """クリックイベントのハンドラ。

        Args:
            event: マウスイベント。
        """
        # 画像領域外のクリックは無視
        if event.inaxes != self._ax:
            return

        if event.button == 1:  # 左クリック
            x, y = int(event.xdata), int(event.ydata)
            self._click_points.append((x, y))
            print(f"点{len(self._click_points)}: ({x}, {y})")

            # 1点目のマーカーを表示
            if len(self._click_points) == 1:
                if self._temp_marker is not None:
                    self._temp_marker.remove()
                self._temp_marker = self._ax.plot(x, y, 'r+', markersize=20, markeredgewidth=3)[0]
                self._fig.canvas.draw()

            # 2点揃ったら矩形を描画
            if len(self._click_points) == 2:
                x1, y1 = self._click_points[0]
                x2, y2 = self._click_points[1]

                # 座標を正規化（左上と右下）
                left = min(x1, x2)
                top = min(y1, y2)
                width = abs(x2 - x1)
                height = abs(y2 - y1)

                if width > 0 and height > 0:
                    self._selected_bbox = BoundingBox(x=left, y=top, width=width, height=height)
                    self._draw_rect(self._selected_bbox)
                    print(f"選択完了: {self._selected_bbox}")
                    print("ウィンドウを閉じて確定、または右クリックでリセット")

                # マーカーを削除
                if self._temp_marker is not None:
                    self._temp_marker.remove()
                    self._temp_marker = None

                # クリックポイントをリセット（次の選択に備える）
                self._click_points = []
                self._fig.canvas.draw()

        elif event.button == 3:  # 右クリック：リセット
            self._click_points = []
            if self._temp_marker is not None:
                self._temp_marker.remove()
                self._temp_marker = None
            if self._current_rect is not None:
                self._current_rect.remove()
                self._current_rect = None
            self._selected_bbox = None
            self._fig.canvas.draw()
            print("リセットしました")

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
    ) -> BoundingBox | None:
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
    ) -> tuple[BoundingBox | None, Image.Image | None]:
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


def select_illustration_region(image: Image.Image) -> BoundingBox | None:
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


class JupyterRegionSelector:
    """Jupyter Notebook用の領域選択ツール。

    ipywidgets と %matplotlib widget を使用して、
    Jupyter環境でインタラクティブに領域を選択できます。

    Example:
        >>> selector = JupyterRegionSelector(image)
        >>> selector.show()
        >>> # クリックで選択後...
        >>> bbox = selector.get_bbox()
    """

    def __init__(self, image: Image.Image) -> None:
        """セレクターを初期化。

        Args:
            image: 選択対象の画像。
        """
        self._image = image
        self._selected_bbox: BoundingBox | None = None
        self._click_points: list = []
        self._fig: plt.Figure | None = None
        self._ax: plt.Axes | None = None
        self._temp_marker: any | None = None
        self._current_rect: Rectangle | None = None
        self._status_text: any | None = None

    def show(self) -> None:
        """選択UIを表示。

        Jupyter Notebook のセル内に画像とインタラクティブUIを表示します。
        左クリック2回で矩形を選択し、get_bbox() で結果を取得できます。
        """
        import matplotlib
        if 'widget' not in matplotlib.get_backend().lower():
            print("警告: %matplotlib widget を実行してから使用してください")

        img_array = np.array(self._image)

        self._fig, self._ax = plt.subplots(1, 1, figsize=(10, 10))
        self._ax.imshow(img_array)
        self._ax.set_title("左クリック2回で矩形を選択 | 右クリックでリセット")

        # ステータステキスト
        self._status_text = self._ax.text(
            0.5, -0.05, "1点目をクリックしてください",
            transform=self._ax.transAxes,
            ha="center", fontsize=12, color="blue"
        )

        # クリックイベントを接続
        self._fig.canvas.mpl_connect("button_press_event", self._on_click)

        plt.tight_layout()
        plt.show()

    def _on_click(self, event: any) -> None:
        """クリックイベントのハンドラ。"""
        if event.inaxes != self._ax:
            return

        if event.button == 1:  # 左クリック
            x, y = int(event.xdata), int(event.ydata)
            self._click_points.append((x, y))

            if len(self._click_points) == 1:
                # 1点目のマーカーを表示
                if self._temp_marker is not None:
                    self._temp_marker.remove()
                self._temp_marker = self._ax.plot(
                    x, y, 'r+', markersize=20, markeredgewidth=3
                )[0]
                self._status_text.set_text(f"1点目: ({x}, {y}) - 2点目をクリック")
                self._fig.canvas.draw_idle()

            elif len(self._click_points) == 2:
                # 2点目で矩形を確定
                x1, y1 = self._click_points[0]
                x2, y2 = self._click_points[1]

                left = min(x1, x2)
                top = min(y1, y2)
                width = abs(x2 - x1)
                height = abs(y2 - y1)

                if width > 0 and height > 0:
                    self._selected_bbox = BoundingBox(
                        x=left, y=top, width=width, height=height
                    )
                    self._draw_rect(self._selected_bbox)
                    self._status_text.set_text(
                        f"選択完了: ({left}, {top}, {width}x{height}) | "
                        "右クリックでリセット | get_bbox()で取得"
                    )
                    self._status_text.set_color("green")

                # マーカーを削除
                if self._temp_marker is not None:
                    self._temp_marker.remove()
                    self._temp_marker = None

                self._click_points = []
                self._fig.canvas.draw_idle()

        elif event.button == 3:  # 右クリック：リセット
            self._click_points = []
            if self._temp_marker is not None:
                self._temp_marker.remove()
                self._temp_marker = None
            if self._current_rect is not None:
                self._current_rect.remove()
                self._current_rect = None
            self._selected_bbox = None
            self._status_text.set_text("リセット - 1点目をクリックしてください")
            self._status_text.set_color("blue")
            self._fig.canvas.draw_idle()

    def _draw_rect(self, bbox: BoundingBox) -> None:
        """矩形を描画。"""
        if self._current_rect is not None:
            self._current_rect.remove()

        self._current_rect = Rectangle(
            (bbox.x, bbox.y),
            bbox.width,
            bbox.height,
            linewidth=3,
            edgecolor="lime",
            facecolor="none",
            linestyle="-",
        )
        self._ax.add_patch(self._current_rect)
        self._fig.canvas.draw_idle()

    def get_bbox(self) -> BoundingBox | None:
        """選択された領域を取得。

        Returns:
            選択されたBoundingBox。未選択の場合はNone。
        """
        return self._selected_bbox

    def get_cropped(self) -> Image.Image | None:
        """選択領域で切り抜いた画像を取得。

        Returns:
            切り抜いた画像。未選択の場合はNone。
        """
        if self._selected_bbox is None:
            return None
        bbox = self._selected_bbox
        return self._image.crop((
            bbox.x, bbox.y,
            bbox.x + bbox.width,
            bbox.y + bbox.height
        ))
