"""TripoSRモジュールのテスト。

このモジュールは、TripoSRによる3Dメッシュ生成機能の
ユニットテストと統合テストを提供します。
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from shadowbox.config import BoundingBox, ShadowboxSettings
from shadowbox.core.mesh import ShadowboxMesh
from shadowbox.triposr import TripoSRSettings, create_triposr_generator, create_triposr_pipeline
from shadowbox.triposr.generator import TripoSRGenerator, _find_triposr_directory
from shadowbox.triposr.pipeline import TripoSRPipeline, TripoSRPipelineResult


class TestTripoSRSettings:
    """TripoSRSettings設定クラスのテスト。"""

    def test_default_settings(self) -> None:
        """デフォルト設定を確認するテスト。"""
        settings = TripoSRSettings()

        assert settings.model_id == "stabilityai/TripoSR"
        assert settings.device == "auto"
        assert settings.chunk_size == 8192
        assert settings.mc_resolution == 256
        assert settings.foreground_ratio == 0.85
        assert settings.remove_background is True

    def test_custom_settings(self) -> None:
        """カスタム設定を確認するテスト。"""
        settings = TripoSRSettings(
            device="cuda",
            mc_resolution=512,
            remove_background=False,
        )

        assert settings.device == "cuda"
        assert settings.mc_resolution == 512
        assert settings.remove_background is False


class TestFindTripoSRDirectory:
    """_find_triposr_directory関数のテスト。"""

    def test_returns_none_when_tsr_importable(self) -> None:
        """tsrがインポート可能な場合はNoneを返すテスト。"""
        with patch.dict("sys.modules", {"tsr": MagicMock()}):
            result = _find_triposr_directory()
            # 既にインポート可能な場合はNoneを返す
            assert result is None


class TestTripoSRGenerator:
    """TripoSRGeneratorクラスのテスト。"""

    def test_init(self) -> None:
        """初期化のテスト。"""
        settings = TripoSRSettings()
        generator = TripoSRGenerator(settings)

        assert generator._settings == settings
        assert generator._model is None
        assert generator._device is None

    def test_settings_property(self) -> None:
        """settingsプロパティのテスト。"""
        settings = TripoSRSettings(device="cpu")
        generator = TripoSRGenerator(settings)

        assert generator.settings == settings
        assert generator.settings.device == "cpu"


class TestTripoSRPipeline:
    """TripoSRPipelineクラスのテスト。"""

    def test_init(self) -> None:
        """初期化のテスト。"""
        settings = TripoSRSettings()
        pipeline = TripoSRPipeline(settings)

        assert pipeline._settings == settings
        assert isinstance(pipeline._generator, TripoSRGenerator)

    def test_settings_property(self) -> None:
        """settingsプロパティのテスト。"""
        settings = TripoSRSettings(device="cpu")
        pipeline = TripoSRPipeline(settings)

        assert pipeline.settings == settings


class TestFactoryFunctions:
    """ファクトリ関数のテスト。"""

    def test_create_triposr_generator(self) -> None:
        """create_triposr_generator関数のテスト。"""
        settings = TripoSRSettings()
        generator = create_triposr_generator(settings)

        assert isinstance(generator, TripoSRGenerator)
        assert generator.settings == settings

    def test_create_triposr_pipeline(self) -> None:
        """create_triposr_pipeline関数のテスト。"""
        settings = TripoSRSettings()
        pipeline = create_triposr_pipeline(settings)

        assert isinstance(pipeline, TripoSRPipeline)
        assert pipeline.settings == settings


class TestCreatePipelineWithTripoSR:
    """create_pipelineでTripoSRモードを使用するテスト。"""

    def test_create_pipeline_triposr_mode(self) -> None:
        """model_mode='triposr'でTripoSRPipelineが返されるテスト。"""
        from shadowbox import create_pipeline

        settings = ShadowboxSettings()
        settings.model_mode = "triposr"

        pipeline = create_pipeline(settings)

        assert isinstance(pipeline, TripoSRPipeline)

    def test_create_pipeline_default_mode(self) -> None:
        """デフォルトでShadowboxPipelineが返されるテスト。"""
        from shadowbox import create_pipeline
        from shadowbox.core.pipeline import ShadowboxPipeline

        settings = ShadowboxSettings()
        # model_modeはデフォルトで"depth"

        pipeline = create_pipeline(settings, use_mock_depth=True)

        assert isinstance(pipeline, ShadowboxPipeline)


class TestTripoSRPipelineResult:
    """TripoSRPipelineResultデータクラスのテスト。"""

    def test_result_fields(self) -> None:
        """結果フィールドのテスト。"""
        # モックデータを作成
        original = np.zeros((100, 100, 3), dtype=np.uint8)
        mesh = MagicMock(spec=ShadowboxMesh)
        bbox = BoundingBox(x=10, y=20, width=50, height=60)

        result = TripoSRPipelineResult(
            original_image=original,
            mesh=mesh,
            bbox=bbox,
        )

        assert np.array_equal(result.original_image, original)
        assert result.mesh == mesh
        assert result.bbox == bbox

    def test_result_without_bbox(self) -> None:
        """bbox=Noneの場合のテスト。"""
        original = np.zeros((100, 100, 3), dtype=np.uint8)
        mesh = MagicMock(spec=ShadowboxMesh)

        result = TripoSRPipelineResult(
            original_image=original,
            mesh=mesh,
            bbox=None,
        )

        assert result.bbox is None


@pytest.mark.slow
@pytest.mark.integration
class TestTripoSRIntegration:
    """TripoSRの統合テスト（実際のモデルを使用）。

    これらのテストは時間がかかるため、通常のテスト実行では
    スキップされます。実行するには:
        pytest -m integration tests/test_triposr.py
    """

    @pytest.fixture
    def simple_image(self) -> Image.Image:
        """シンプルなテスト画像を作成。"""
        img = Image.new("RGB", (256, 256), color=(128, 128, 128))
        return img

    def test_full_triposr_pipeline(self, simple_image: Image.Image) -> None:
        """完全なTripoSRパイプラインをテスト。"""
        settings = TripoSRSettings(device="cpu")
        pipeline = create_triposr_pipeline(settings)

        result = pipeline.process(simple_image)

        assert isinstance(result, TripoSRPipelineResult)
        assert isinstance(result.mesh, ShadowboxMesh)
        assert result.mesh.num_layers >= 1
        assert result.mesh.total_vertices > 0

    def test_triposr_with_bbox(self, simple_image: Image.Image) -> None:
        """バウンディングボックス指定でのTripoSRパイプラインをテスト。"""
        settings = TripoSRSettings(device="cpu")
        pipeline = create_triposr_pipeline(settings)

        bbox = BoundingBox(x=50, y=50, width=100, height=100)
        result = pipeline.process(simple_image, bbox=bbox)

        assert result.bbox == bbox
        assert isinstance(result.mesh, ShadowboxMesh)
