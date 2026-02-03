"""画像選択モジュール。

このモジュールは、ディレクトリ内の画像をギャラリー形式で
表示し、インデックス番号で選択する機能を提供します。
"""

from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

from shadowbox.utils.image import load_image, load_image_from_url


class ImageSelector:
    """画像選択ツール。

    指定ディレクトリ内の画像をギャラリー形式で表示し、
    インデックス番号で選択できる機能を提供します。

    Attributes:
        directory: 画像が格納されているディレクトリ。
        images: 読み込まれた画像のリスト。

    Example:
        >>> selector = ImageSelector("./data/cards/")
        >>> selector.show_gallery()  # ギャラリー表示
        >>> image = selector.select(index=3)  # 番号で選択
    """

    # サポートする画像拡張子
    SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}

    def __init__(
        self,
        directory: str | Path,
        recursive: bool = False,
    ) -> None:
        """セレクタを初期化。

        Args:
            directory: 画像ディレクトリのパス。
            recursive: サブディレクトリも検索するかどうか。
        """
        self._directory = Path(directory)
        self._recursive = recursive
        self._image_paths: list[Path] = []
        self._images: list[Image.Image] = []
        self._thumbnails: list[Image.Image] = []

        self._scan_directory()

    @property
    def directory(self) -> Path:
        """画像ディレクトリ。"""
        return self._directory

    @property
    def count(self) -> int:
        """画像の数。"""
        return len(self._image_paths)

    @property
    def image_paths(self) -> list[Path]:
        """画像ファイルパスのリスト。"""
        return self._image_paths.copy()

    def _scan_directory(self) -> None:
        """ディレクトリをスキャンして画像ファイルを収集。"""
        if not self._directory.exists():
            raise FileNotFoundError(f"ディレクトリが見つかりません: {self._directory}")

        if not self._directory.is_dir():
            raise ValueError(f"ディレクトリではありません: {self._directory}")

        # 画像ファイルを検索
        if self._recursive:
            pattern = "**/*"
        else:
            pattern = "*"

        for path in sorted(self._directory.glob(pattern)):
            if path.is_file() and path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                self._image_paths.append(path)

    def show_gallery(
        self,
        columns: int = 4,
        thumbnail_size: tuple[int, int] = (150, 150),
        figsize: tuple[int, int] | None = None,
    ) -> None:
        """画像をギャラリー形式で表示。

        Args:
            columns: 列数。
            thumbnail_size: サムネイルサイズ。
            figsize: Figure全体のサイズ。

        Example:
            >>> selector.show_gallery(columns=5)
        """
        if self.count == 0:
            print("画像が見つかりません")
            return

        # サムネイルを生成
        self._generate_thumbnails(thumbnail_size)

        rows = (self.count + columns - 1) // columns

        if figsize is None:
            figsize = (columns * 2, rows * 2.5)

        fig, axes = plt.subplots(rows, columns, figsize=figsize, squeeze=False)

        for idx, (thumb, path) in enumerate(zip(self._thumbnails, self._image_paths, strict=True)):
            row = idx // columns
            col = idx % columns
            ax = axes[row, col]

            ax.imshow(thumb)
            ax.set_title(f"[{idx}] {path.stem[:15]}", fontsize=8)
            ax.axis("off")

        # 余分なサブプロットを非表示
        for idx in range(self.count, rows * columns):
            row = idx // columns
            col = idx % columns
            axes[row, col].axis("off")

        fig.suptitle(
            f"画像ギャラリー ({self.count}枚) - select(index)で選択",
            fontsize=12,
        )
        fig.tight_layout()
        plt.show()

    def display_grid(
        self,
        columns: int = 4,
        thumbnail_size: tuple[int, int] = (150, 150),
    ) -> None:
        """show_galleryのエイリアス。"""
        self.show_gallery(columns, thumbnail_size)

    def _generate_thumbnails(self, size: tuple[int, int]) -> None:
        """サムネイルを生成。

        Args:
            size: サムネイルサイズ。
        """
        self._thumbnails = []

        for path in self._image_paths:
            try:
                img = Image.open(path)
                img = img.convert("RGB")
                img.thumbnail(size, Image.Resampling.LANCZOS)
                self._thumbnails.append(img)
            except Exception as e:
                # エラー時はプレースホルダーを使用
                placeholder = Image.new("RGB", size, color=(128, 128, 128))
                self._thumbnails.append(placeholder)
                print(f"警告: {path.name} の読み込みに失敗: {e}")

    def select(self, index: int) -> Image.Image:
        """インデックスで画像を選択。

        Args:
            index: 選択する画像のインデックス。

        Returns:
            選択された画像（PIL Image）。

        Raises:
            IndexError: インデックスが範囲外の場合。

        Example:
            >>> image = selector.select(3)
        """
        if index < 0 or index >= self.count:
            raise IndexError(
                f"インデックスが範囲外です: {index} (有効範囲: 0-{self.count - 1})"
            )

        return load_image(self._image_paths[index])

    def get_path(self, index: int) -> Path:
        """インデックスで画像パスを取得。

        Args:
            index: 画像のインデックス。

        Returns:
            画像ファイルのパス。

        Raises:
            IndexError: インデックスが範囲外の場合。
        """
        if index < 0 or index >= self.count:
            raise IndexError(
                f"インデックスが範囲外です: {index} (有効範囲: 0-{self.count - 1})"
            )

        return self._image_paths[index]

    def list_images(self) -> list[str]:
        """画像ファイル名のリストを取得。

        Returns:
            ファイル名（インデックス付き）のリスト。

        Example:
            >>> for name in selector.list_images():
            ...     print(name)
        """
        return [f"[{i}] {p.name}" for i, p in enumerate(self._image_paths)]

    def print_list(self) -> None:
        """画像リストをコンソールに表示。

        Example:
            >>> selector.print_list()
            [0] card_001.png
            [1] card_002.png
            ...
        """
        if self.count == 0:
            print("画像が見つかりません")
            return

        print(f"\n{self._directory} 内の画像 ({self.count}枚):")
        print("-" * 40)
        for name in self.list_images():
            print(name)
        print("-" * 40)
        print("selector.select(index) で画像を選択できます")


class URLImageLoader:
    """URLから画像を読み込むユーティリティクラス。

    Example:
        >>> image = URLImageLoader.load("https://example.com/card.png")
    """

    @staticmethod
    def load(url: str, timeout: int = 30) -> Image.Image:
        """URLから画像を読み込む。

        Args:
            url: 画像のURL。
            timeout: タイムアウト秒数。

        Returns:
            読み込んだ画像。

        Raises:
            ValueError: 読み込みに失敗した場合。

        Example:
            >>> image = URLImageLoader.load("https://example.com/card.png")
        """
        return load_image_from_url(url, timeout)

    @staticmethod
    def load_multiple(urls: list[str], timeout: int = 30) -> list[Image.Image]:
        """複数のURLから画像を読み込む。

        Args:
            urls: 画像URLのリスト。
            timeout: 各URLのタイムアウト秒数。

        Returns:
            読み込んだ画像のリスト。失敗したURLはスキップ。

        Example:
            >>> images = URLImageLoader.load_multiple([url1, url2, url3])
        """
        images = []
        for url in urls:
            try:
                img = load_image_from_url(url, timeout)
                images.append(img)
            except Exception as e:
                print(f"警告: {url} の読み込みに失敗: {e}")

        return images


def create_image_selector(directory: str | Path) -> ImageSelector:
    """画像セレクタを作成するユーティリティ関数。

    Args:
        directory: 画像ディレクトリのパス。

    Returns:
        ImageSelectorインスタンス。

    Example:
        >>> selector = create_image_selector("./data/cards/")
        >>> selector.show_gallery()
    """
    return ImageSelector(directory)


def load_from_url(url: str) -> Image.Image:
    """URLから画像を読み込むユーティリティ関数。

    Args:
        url: 画像のURL。

    Returns:
        読み込んだ画像。

    Example:
        >>> image = load_from_url("https://example.com/card.png")
    """
    return URLImageLoader.load(url)
