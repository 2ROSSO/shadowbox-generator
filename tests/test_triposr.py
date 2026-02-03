"""TripoSRモジュールのテスト。

このモジュールは、TripoSRによる3Dメッシュ生成機能の
ユニットテストと統合テストを提供します。
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from shadowbox.config import BoundingBox, RenderSettings, ShadowboxSettings
from shadowbox.core.mesh import FrameMesh, ShadowboxMesh
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
        assert isinstance(pipeline._render_settings, RenderSettings)

    def test_init_with_render_settings(self) -> None:
        """RenderSettingsを渡して初期化するテスト。"""
        settings = TripoSRSettings()
        render_settings = RenderSettings(frame_depth=1.0)
        pipeline = TripoSRPipeline(settings, render_settings)

        assert pipeline._settings == settings
        assert pipeline._render_settings == render_settings
        assert pipeline._render_settings.frame_depth == 1.0

    def test_settings_property(self) -> None:
        """settingsプロパティのテスト。"""
        settings = TripoSRSettings(device="cpu")
        pipeline = TripoSRPipeline(settings)

        assert pipeline.settings == settings

    def test_render_settings_property(self) -> None:
        """render_settingsプロパティのテスト。"""
        settings = TripoSRSettings()
        render_settings = RenderSettings(frame_z=0.5)
        pipeline = TripoSRPipeline(settings, render_settings)

        assert pipeline.render_settings == render_settings
        assert pipeline.render_settings.frame_z == 0.5


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

    def test_create_triposr_pipeline_with_render_settings(self) -> None:
        """render_settingsを渡してcreate_triposr_pipelineするテスト。"""
        settings = TripoSRSettings()
        render_settings = RenderSettings(frame_depth=0.8)
        pipeline = create_triposr_pipeline(settings, render_settings)

        assert isinstance(pipeline, TripoSRPipeline)
        assert pipeline.settings == settings
        assert pipeline.render_settings == render_settings
        assert pipeline.render_settings.frame_depth == 0.8


class TestCreatePipelineWithTripoSR:
    """create_pipelineでTripoSRモードを使用するテスト。"""

    def test_create_pipeline_triposr_mode(self) -> None:
        """model_mode='triposr'でTripoSRPipelineが返されるテスト。"""
        from shadowbox import create_pipeline

        settings = ShadowboxSettings()
        settings.model_mode = "triposr"

        pipeline = create_pipeline(settings)

        assert isinstance(pipeline, TripoSRPipeline)

    def test_create_pipeline_triposr_mode_passes_render_settings(self) -> None:
        """model_mode='triposr'でRenderSettingsが渡されるテスト。"""
        from shadowbox import create_pipeline

        settings = ShadowboxSettings()
        settings.model_mode = "triposr"
        settings.render.frame_depth = 0.75

        pipeline = create_pipeline(settings)

        assert isinstance(pipeline, TripoSRPipeline)
        assert pipeline.render_settings.frame_depth == 0.75

    def test_create_pipeline_default_mode(self) -> None:
        """デフォルトでShadowboxPipelineが返されるテスト。"""
        from shadowbox import create_pipeline
        from shadowbox.core.pipeline import ShadowboxPipeline

        settings = ShadowboxSettings()
        # model_modeはデフォルトで"depth"

        pipeline = create_pipeline(settings, use_mock_depth=True)

        assert isinstance(pipeline, ShadowboxPipeline)


class TestTripoSRPipelineFrameGeneration:
    """TripoSRPipelineのフレーム生成機能のテスト。"""

    def test_create_frame(self) -> None:
        """_create_frameメソッドのテスト。"""
        settings = TripoSRSettings()
        pipeline = TripoSRPipeline(settings)

        frame = pipeline._create_frame(z_back=-1.0, z_front=0.5)

        assert isinstance(frame, FrameMesh)
        assert frame.has_walls is True
        assert frame.z_position == 0.5
        assert frame.z_back == -1.0
        # 12頂点: 前面外側4 + 前面内側4 + 背面外側4
        assert len(frame.vertices) == 12
        # 16三角形: 前面枠8 + 外壁8
        assert len(frame.faces) == 16
        # 色は暗い色（30, 30, 30）
        assert np.array_equal(frame.color, np.array([30, 30, 30], dtype=np.uint8))

    def test_create_frame_vertex_positions(self) -> None:
        """フレームの頂点位置が正しいかテスト。"""
        settings = TripoSRSettings()
        pipeline = TripoSRPipeline(settings)

        z_front = 0.0
        z_back = -0.5
        frame = pipeline._create_frame(z_back=z_back, z_front=z_front)

        # 前面の頂点はz_frontの位置
        assert frame.vertices[0, 2] == z_front
        assert frame.vertices[4, 2] == z_front
        # 背面の頂点はz_backの位置
        assert frame.vertices[8, 2] == z_back
        assert frame.vertices[11, 2] == z_back

    def test_add_frame_to_mesh(self) -> None:
        """_add_frame_to_meshメソッドのテスト。"""
        from shadowbox.core.mesh import LayerMesh

        settings = TripoSRSettings()
        pipeline = TripoSRPipeline(settings)

        # ダミーメッシュを作成
        vertices = np.array([
            [-0.5, -0.5, -0.3],
            [0.5, -0.5, 0.2],
            [0.5, 0.5, -0.1],
            [-0.5, 0.5, 0.1],
        ], dtype=np.float32)
        colors = np.array([
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
            [255, 255, 0],
        ], dtype=np.uint8)
        layer = LayerMesh(
            vertices=vertices,
            colors=colors,
            z_position=0.0,
            layer_index=0,
            pixel_indices=np.zeros((4, 2), dtype=np.int32),
        )
        original_mesh = ShadowboxMesh(
            layers=[layer],
            frame=None,
            bounds=(-0.5, 0.5, -0.5, 0.5, -0.3, 0.2),
        )

        # フレームを追加
        result_mesh = pipeline._add_frame_to_mesh(original_mesh)

        # フレームが追加されていること
        assert result_mesh.frame is not None
        assert isinstance(result_mesh.frame, FrameMesh)
        assert result_mesh.frame.has_walls is True
        # バウンズが再計算されていること
        assert result_mesh.bounds != original_mesh.bounds


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
