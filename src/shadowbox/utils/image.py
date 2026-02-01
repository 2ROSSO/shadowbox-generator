"""画像ユーティリティモジュール。

このモジュールは、画像の読み込み、変換、保存などの
ユーティリティ関数を提供します。
"""

from io import BytesIO
from pathlib import Path
from typing import Union

import numpy as np
import requests
from numpy.typing import NDArray
from PIL import Image


def load_image(source: Union[str, Path, Image.Image]) -> Image.Image:
    """様々なソースから画像を読み込む。

    ファイルパス、URL、またはPIL Imageから画像を読み込みます。
    読み込んだ画像はRGBモードに変換されます。

    Args:
        source: 画像ソース。以下のいずれか:
            - ファイルパス (str または Path)
            - URL (httpまたはhttpsで始まる文字列)
            - PIL Image オブジェクト

    Returns:
        RGBモードのPIL Image。

    Raises:
        FileNotFoundError: ファイルが存在しない場合。
        ValueError: URLからの読み込みに失敗した場合。
        TypeError: サポートされていないソースタイプの場合。

    Example:
        >>> # ファイルから読み込み
        >>> image = load_image("card.png")
        >>>
        >>> # URLから読み込み
        >>> image = load_image("https://example.com/card.png")
        >>>
        >>> # PIL Imageをそのまま渡す
        >>> from PIL import Image
        >>> img = Image.new("RGB", (100, 100))
        >>> image = load_image(img)
    """
    # 既にPIL Imageの場合はそのまま返す（RGB変換のみ）
    if isinstance(source, Image.Image):
        return _ensure_rgb(source)

    # 文字列の場合はパスかURLかを判定
    if isinstance(source, str):
        if source.startswith(("http://", "https://")):
            return load_image_from_url(source)
        source = Path(source)

    # Pathの場合はファイルから読み込み
    if isinstance(source, Path):
        return load_image_from_file(source)

    raise TypeError(
        f"サポートされていないソースタイプです: {type(source).__name__}。"
        "str, Path, または PIL.Image.Image を使用してください。"
    )


def load_image_from_file(file_path: Union[str, Path]) -> Image.Image:
    """ファイルから画像を読み込む。

    Args:
        file_path: 画像ファイルのパス。

    Returns:
        RGBモードのPIL Image。

    Raises:
        FileNotFoundError: ファイルが存在しない場合。
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"画像ファイルが見つかりません: {path}")

    image = Image.open(path)
    return _ensure_rgb(image)


def load_image_from_url(url: str, timeout: int = 30) -> Image.Image:
    """URLから画像を読み込む。

    Args:
        url: 画像のURL。
        timeout: リクエストのタイムアウト秒数。

    Returns:
        RGBモードのPIL Image。

    Raises:
        ValueError: URLからの読み込みに失敗した場合。

    Example:
        >>> image = load_image_from_url("https://example.com/card.png")
    """
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()

        image = Image.open(BytesIO(response.content))
        return _ensure_rgb(image)

    except requests.RequestException as e:
        raise ValueError(f"URLからの画像読み込みに失敗しました: {url}") from e


def _ensure_rgb(image: Image.Image) -> Image.Image:
    """画像をRGBモードに変換。

    RGBA画像の場合は白い背景に合成します。

    Args:
        image: 変換する画像。

    Returns:
        RGBモードの画像。
    """
    if image.mode == "RGB":
        return image

    if image.mode == "RGBA":
        # 透明部分を白い背景で埋める
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3])  # アルファチャンネルをマスクに使用
        return background

    # その他のモード（L, P, CMYKなど）はRGBに変換
    return image.convert("RGB")


def image_to_array(image: Image.Image) -> NDArray[np.uint8]:
    """PIL ImageをNumPy配列に変換。

    Args:
        image: 変換するPIL Image。

    Returns:
        shape (H, W, 3) のuint8 NumPy配列。
    """
    return np.array(image, dtype=np.uint8)


def array_to_image(array: NDArray[np.uint8]) -> Image.Image:
    """NumPy配列をPIL Imageに変換。

    Args:
        array: shape (H, W, 3) のuint8 NumPy配列。

    Returns:
        PIL Image。
    """
    return Image.fromarray(array)


def crop_image(
    image: Image.Image,
    x: int,
    y: int,
    width: int,
    height: int,
) -> Image.Image:
    """画像を指定した領域で切り抜く。

    Args:
        image: 切り抜く元画像。
        x: 左端のX座標。
        y: 上端のY座標。
        width: 切り抜く幅。
        height: 切り抜く高さ。

    Returns:
        切り抜いた画像。
    """
    return image.crop((x, y, x + width, y + height))
