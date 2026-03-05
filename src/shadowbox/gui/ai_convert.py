"""Gemini APIによるAI画像変換モジュール。

実写画像をイラスト風に変換してシャドーボックス化の品質を向上させます。
"""

from __future__ import annotations

import base64
import io
import logging
import time

import requests
from PIL import Image
from PyQt6.QtCore import QThread, pyqtSignal

logger = logging.getLogger(__name__)


class GeminiConvertError(Exception):
    """Gemini API呼び出しに関するエラー。"""


class GeminiConverter:
    """Gemini APIで画像をイラスト風に変換。"""

    ENDPOINT = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        "gemini-2.0-flash-exp-image-generation:generateContent"
    )
    MAX_LONG_EDGE = 2048
    TIMEOUT = 120

    PROMPT = (
        "Transform this image into a vivid illustration style suitable for a 3D shadowbox. "
        "Keep the exact same composition, subject, and layout. "
        "Emphasize depth and dimensionality with clear foreground/midground/background separation. "
        "Use bold outlines and rich colors. "
        "IMPORTANT: Do NOT modify any text, card borders, card frames, logos, or UI elements. "
        "Only transform the illustrated/photographic artwork areas. "
        "Do not add text or watermarks."
    )

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    def convert(self, image: Image.Image) -> Image.Image:
        """画像をイラスト風に変換して返す。

        Args:
            image: 入力PIL画像。

        Returns:
            変換後のPIL画像（元画像と同じサイズ）。

        Raises:
            GeminiConvertError: API呼び出しに失敗した場合。
        """
        original_size = image.size
        resized = self._resize_image(image)
        image_b64 = self._encode_image(resized)

        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": self.PROMPT},
                        {
                            "inlineData": {
                                "mimeType": "image/jpeg",
                                "data": image_b64,
                            }
                        },
                    ]
                }
            ],
            "generationConfig": {
                "responseModalities": ["IMAGE", "TEXT"],
            },
        }

        result_b64 = self._call_api(payload)
        result_image = self._decode_image(result_b64)

        if result_image.size != original_size:
            result_image = result_image.resize(original_size, Image.Resampling.LANCZOS)

        return result_image

    def _resize_image(self, image: Image.Image) -> Image.Image:
        """長辺がMAX_LONG_EDGE以下になるようリサイズ。"""
        w, h = image.size
        long_edge = max(w, h)
        if long_edge <= self.MAX_LONG_EDGE:
            return image
        scale = self.MAX_LONG_EDGE / long_edge
        new_w = int(w * scale)
        new_h = int(h * scale)
        return image.resize((new_w, new_h), Image.Resampling.LANCZOS)

    @staticmethod
    def _encode_image(image: Image.Image) -> str:
        """PIL画像をbase64 JPEGエンコード。"""
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="JPEG", quality=90)
        return base64.b64encode(buf.getvalue()).decode("ascii")

    @staticmethod
    def _decode_image(b64_data: str) -> Image.Image:
        """base64データからPIL画像をデコード。"""
        raw = base64.b64decode(b64_data)
        return Image.open(io.BytesIO(raw)).convert("RGB")

    def _call_api(self, payload: dict) -> str:
        """Gemini APIを呼び出し、レスポンスからbase64画像データを取得。

        429レスポンス時に1回だけ5秒waitでリトライ。

        Returns:
            base64エンコードされた画像データ。

        Raises:
            GeminiConvertError: API呼び出しに失敗した場合。
        """
        url = f"{self.ENDPOINT}?key={self._api_key}"
        headers = {"Content-Type": "application/json"}

        for attempt in range(2):
            try:
                resp = requests.post(
                    url, json=payload, headers=headers, timeout=self.TIMEOUT
                )
            except requests.RequestException as e:
                logger.error("Gemini API network error: %s", e)
                raise GeminiConvertError(f"Network error: {e}") from e

            if resp.status_code == 429 and attempt == 0:
                logger.warning("Gemini API rate limited (429), retrying in 5s...")
                time.sleep(5)
                continue

            if resp.status_code != 200:
                logger.error(
                    "Gemini API error %d: %s", resp.status_code, resp.text[:500]
                )
                raise GeminiConvertError(
                    f"API error {resp.status_code}: {resp.text[:500]}"
                )

            return self._extract_image_data(resp.json())

        raise GeminiConvertError("Rate limited after retry")  # pragma: no cover

    @staticmethod
    def _extract_image_data(response: dict) -> str:
        """APIレスポンスJSONからbase64画像データを抽出。

        Raises:
            GeminiConvertError: 画像データが見つからない場合。
        """
        try:
            candidates = response["candidates"]
            for candidate in candidates:
                for part in candidate["content"]["parts"]:
                    if "inlineData" in part:
                        return part["inlineData"]["data"]
        except (KeyError, IndexError, TypeError) as e:
            raise GeminiConvertError(
                f"Unexpected API response structure: {e}"
            ) from e

        raise GeminiConvertError("No image data in API response")


class AIConvertThread(QThread):
    """バックグラウンドでGemini API変換を実行するスレッド。"""

    finished = pyqtSignal(object)  # PIL.Image
    error = pyqtSignal(str)

    def __init__(self, image: Image.Image, api_key: str) -> None:
        super().__init__()
        self._image = image
        self._api_key = api_key

    def run(self) -> None:
        try:
            converter = GeminiConverter(self._api_key)
            result = converter.convert(self._image)
            self.finished.emit(result)
        except GeminiConvertError as e:
            logger.error("AI convert failed: %s", e)
            self.error.emit(str(e))
        except Exception as e:
            logger.exception("AI convert unexpected error")
            self.error.emit(f"Unexpected error: {e}")
