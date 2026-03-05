"""AI画像変換モジュールのテスト。"""

from __future__ import annotations

import base64
import io
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image


@pytest.fixture()
def sample_image() -> Image.Image:
    """テスト用のRGB画像。"""
    return Image.new("RGB", (100, 80), color=(128, 64, 32))


@pytest.fixture()
def large_image() -> Image.Image:
    """長辺が2048を超える画像。"""
    return Image.new("RGB", (4000, 3000), color=(128, 64, 32))


def _make_b64_jpeg(width: int = 100, height: int = 80) -> str:
    """テスト用base64 JPEGデータを生成。"""
    img = Image.new("RGB", (width, height), color=(200, 100, 50))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _make_api_response(b64_data: str) -> dict:
    """Gemini APIのレスポンス形式を返す。"""
    return {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "inlineData": {
                                "mimeType": "image/jpeg",
                                "data": b64_data,
                            }
                        },
                        {"text": "Here is the transformed image."},
                    ]
                }
            }
        ]
    }


class TestGeminiConverter:
    """GeminiConverterのテスト。"""

    def test_convert_success(self, sample_image: Image.Image) -> None:
        from shadowbox.gui.ai_convert import GeminiConverter

        b64 = _make_b64_jpeg(100, 80)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = _make_api_response(b64)

        converter = GeminiConverter("test-key")
        with patch("shadowbox.gui.ai_convert.requests.post", return_value=mock_resp):
            result = converter.convert(sample_image)

        assert isinstance(result, Image.Image)
        assert result.size == sample_image.size
        assert result.mode == "RGB"

    def test_resize_large_image(self, large_image: Image.Image) -> None:
        from shadowbox.gui.ai_convert import GeminiConverter

        converter = GeminiConverter("test-key")
        resized = converter._resize_image(large_image)
        assert max(resized.size) <= 2048

    def test_resize_small_image_unchanged(self, sample_image: Image.Image) -> None:
        from shadowbox.gui.ai_convert import GeminiConverter

        converter = GeminiConverter("test-key")
        resized = converter._resize_image(sample_image)
        assert resized.size == sample_image.size

    def test_encode_decode_roundtrip(self, sample_image: Image.Image) -> None:
        from shadowbox.gui.ai_convert import GeminiConverter

        b64 = GeminiConverter._encode_image(sample_image)
        decoded = GeminiConverter._decode_image(b64)
        assert decoded.mode == "RGB"
        assert decoded.size == sample_image.size

    def test_api_error_status(self, sample_image: Image.Image) -> None:
        from shadowbox.gui.ai_convert import GeminiConvertError, GeminiConverter

        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "Internal Server Error"

        converter = GeminiConverter("test-key")
        with (
            patch("shadowbox.gui.ai_convert.requests.post", return_value=mock_resp),
            pytest.raises(GeminiConvertError, match="API error 500"),
        ):
            converter.convert(sample_image)

    def test_network_error(self, sample_image: Image.Image) -> None:
        import requests

        from shadowbox.gui.ai_convert import GeminiConvertError, GeminiConverter

        converter = GeminiConverter("test-key")
        with (
            patch(
                "shadowbox.gui.ai_convert.requests.post",
                side_effect=requests.ConnectionError("refused"),
            ),
            pytest.raises(GeminiConvertError, match="Network error"),
        ):
            converter.convert(sample_image)

    def test_no_image_in_response(self, sample_image: Image.Image) -> None:
        from shadowbox.gui.ai_convert import GeminiConvertError, GeminiConverter

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "candidates": [{"content": {"parts": [{"text": "No image"}]}}]
        }

        converter = GeminiConverter("test-key")
        with (
            patch("shadowbox.gui.ai_convert.requests.post", return_value=mock_resp),
            pytest.raises(GeminiConvertError, match="No image data"),
        ):
            converter.convert(sample_image)

    def test_429_retry(self, sample_image: Image.Image) -> None:
        from shadowbox.gui.ai_convert import GeminiConverter

        b64 = _make_b64_jpeg(100, 80)
        rate_resp = MagicMock()
        rate_resp.status_code = 429
        ok_resp = MagicMock()
        ok_resp.status_code = 200
        ok_resp.json.return_value = _make_api_response(b64)

        converter = GeminiConverter("test-key")
        with (
            patch(
                "shadowbox.gui.ai_convert.requests.post",
                side_effect=[rate_resp, ok_resp],
            ),
            patch("shadowbox.gui.ai_convert.time.sleep") as mock_sleep,
        ):
            result = converter.convert(sample_image)

        mock_sleep.assert_called_once_with(5)
        assert isinstance(result, Image.Image)

    def test_malformed_response(self, sample_image: Image.Image) -> None:
        from shadowbox.gui.ai_convert import GeminiConvertError, GeminiConverter

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"bad": "structure"}

        converter = GeminiConverter("test-key")
        with (
            patch("shadowbox.gui.ai_convert.requests.post", return_value=mock_resp),
            pytest.raises(GeminiConvertError, match="Unexpected API response"),
        ):
            converter.convert(sample_image)

    def test_result_resized_to_original(self, sample_image: Image.Image) -> None:
        """変換結果が元画像と異なるサイズでも元サイズにリサイズされる。"""
        from shadowbox.gui.ai_convert import GeminiConverter

        b64 = _make_b64_jpeg(200, 160)  # 元とは異なるサイズ
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = _make_api_response(b64)

        converter = GeminiConverter("test-key")
        with patch("shadowbox.gui.ai_convert.requests.post", return_value=mock_resp):
            result = converter.convert(sample_image)

        assert result.size == sample_image.size


class TestAIConvertThread:
    """AIConvertThreadのシグナルテスト。"""

    def test_finished_signal(self, qtbot, sample_image: Image.Image) -> None:
        from shadowbox.gui.ai_convert import AIConvertThread

        b64 = _make_b64_jpeg(100, 80)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = _make_api_response(b64)

        thread = AIConvertThread(sample_image, "test-key")
        with patch("shadowbox.gui.ai_convert.requests.post", return_value=mock_resp):
            with qtbot.waitSignal(thread.finished, timeout=5000) as blocker:
                thread.start()

        result = blocker.args[0]
        assert isinstance(result, Image.Image)

    def test_error_signal(self, qtbot, sample_image: Image.Image) -> None:
        from shadowbox.gui.ai_convert import AIConvertThread

        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "Server Error"

        thread = AIConvertThread(sample_image, "test-key")
        with patch("shadowbox.gui.ai_convert.requests.post", return_value=mock_resp):
            with qtbot.waitSignal(thread.error, timeout=5000) as blocker:
                thread.start()

        assert "API error 500" in blocker.args[0]
