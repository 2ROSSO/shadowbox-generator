"""パイプラインモジュールのテスト。

このモジュールは、ShadowboxPipelineとcreate_pipeline関数の
ユニットテストを提供します。
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from shadowbox.config import BoundingBox, CardTemplate, ShadowboxSettings, YAMLConfigLoader
from shadowbox.core.pipeline import (
    BasePipelineResult,
    PipelineResult,
    ShadowboxPipeline,
    create_pipeline,
)


class TestCreatePipeline:
    """create_pipelineファクトリ関数のテスト。"""

    def test_create_with_defaults(self) -> None:
        """デフォルト設定でパイプラインを作成するテスト。"""
        pipeline = create_pipeline(use_mock_depth=True)

        assert isinstance(pipeline, ShadowboxPipeline)

    def test_create_with_custom_settings(self) -> None:
        """カスタム設定でパイプラインを作成するテスト。"""
        settings = ShadowboxSettings()
        settings.clustering.min_k = 4
        settings.clustering.max_k = 8

        pipeline = create_pipeline(settings, use_mock_depth=True)

        assert isinstance(pipeline, ShadowboxPipeline)

    def test_create_with_mock_depth(self) -> None:
        """モック深度推定器を使用するテスト。"""
        pipeline = create_pipeline(use_mock_depth=True)

        # モック推定器が使われていることを確認
        from shadowbox.core.depth import MockDepthEstimator
        assert isinstance(pipeline._depth_estimator, MockDepthEstimator)


class TestShadowboxPipeline:
    """ShadowboxPipelineのテスト。"""

    def test_process_basic(self, sample_image: Image.Image) -> None:
        """基本的なパイプライン処理をテスト。"""
        pipeline = create_pipeline(use_mock_depth=True)
        result = pipeline.process(sample_image)

        assert isinstance(result, PipelineResult)
        assert result.original_image.shape[:2] == sample_image.size[::-1]
        assert result.depth_map.shape == result.cropped_image.shape[:2]
        assert result.mesh is not None

    def test_process_with_custom_bbox(self, sample_image: Image.Image) -> None:
        """カスタムバウンディングボックスでの処理をテスト。"""
        pipeline = create_pipeline(use_mock_depth=True)

        bbox = BoundingBox(x=10, y=20, width=50, height=80)
        result = pipeline.process(sample_image, custom_bbox=bbox)

        assert result.bbox == bbox
        assert result.cropped_image.shape == (80, 50, 3)

    def test_process_with_specified_k(self, sample_image: Image.Image) -> None:
        """指定したレイヤー数での処理をテスト。"""
        settings = ShadowboxSettings()
        settings.render.back_panel = False
        settings.render.layer_interpolation = 0
        pipeline = create_pipeline(settings, use_mock_depth=True)

        result = pipeline.process(sample_image, k=5)

        assert result.optimal_k == 5
        assert len(result.centroids) == 5
        assert result.mesh.num_layers == 5

    def test_process_without_frame(self, sample_image: Image.Image) -> None:
        """フレームなしでの処理をテスト。"""
        pipeline = create_pipeline(use_mock_depth=True)

        result = pipeline.process(sample_image, include_frame=False)

        assert result.mesh.frame is None

    def test_process_with_numpy_array(self, sample_image_array: np.ndarray) -> None:
        """NumPy配列入力での処理をテスト。"""
        pipeline = create_pipeline(use_mock_depth=True)

        result = pipeline.process(sample_image_array)

        assert isinstance(result, PipelineResult)

    def test_process_with_template(self, sample_image: Image.Image) -> None:
        """テンプレートを使用した処理をテスト。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # テンプレートを作成
            loader = YAMLConfigLoader(Path(tmpdir))
            bbox = BoundingBox(x=5, y=10, width=80, height=100)
            template = CardTemplate(
                name="test_template",
                game="test",
                illustration_area=bbox,
                card_width=100,
                card_height=150,
            )
            loader.save_template(template)

            # テンプレートを使用してパイプラインを作成
            settings = ShadowboxSettings(templates_dir=Path(tmpdir))
            pipeline = create_pipeline(settings, use_mock_depth=True)

            result = pipeline.process(sample_image, template_name="test_template")

            assert result.bbox == bbox
            assert result.cropped_image.shape == (100, 80, 3)

    def test_process_with_nonexistent_template(self, sample_image: Image.Image) -> None:
        """存在しないテンプレートでエラーが発生することをテスト。"""
        pipeline = create_pipeline(use_mock_depth=True)

        with pytest.raises(ValueError, match="テンプレートが見つかりません"):
            pipeline.process(sample_image, template_name="nonexistent")


class TestPipelineResult:
    """PipelineResultデータクラスのテスト。"""

    def test_result_contains_all_fields(self, sample_image: Image.Image) -> None:
        """結果に全フィールドが含まれることをテスト。"""
        pipeline = create_pipeline(use_mock_depth=True)
        result = pipeline.process(sample_image)

        assert result.original_image is not None
        assert result.cropped_image is not None
        assert result.depth_map is not None
        assert result.labels is not None
        assert result.centroids is not None
        assert result.mesh is not None
        assert result.optimal_k > 0

    def test_result_types(self, sample_image: Image.Image) -> None:
        """結果の型が正しいことをテスト。"""
        pipeline = create_pipeline(use_mock_depth=True)
        result = pipeline.process(sample_image)

        assert result.original_image.dtype == np.uint8
        assert result.cropped_image.dtype == np.uint8
        assert result.depth_map.dtype == np.float32
        assert result.labels.dtype == np.int32
        assert result.centroids.dtype == np.float32


class TestPipelineIntegration:
    """パイプラインの統合テスト。"""

    def test_full_pipeline_flow(self) -> None:
        """完全なパイプラインフローをテスト。"""
        # テスト用の画像を作成（異なる深度を持つ領域）
        image = Image.new("RGB", (100, 100))
        pixels = image.load()

        # 3つの異なる色の領域
        for y in range(100):
            for x in range(100):
                if y < 33:
                    pixels[x, y] = (255, 0, 0)    # 赤（手前）
                elif y < 66:
                    pixels[x, y] = (0, 255, 0)    # 緑（中間）
                else:
                    pixels[x, y] = (0, 0, 255)    # 青（奥）

        settings = ShadowboxSettings()
        settings.render.back_panel = False
        settings.render.layer_interpolation = 0
        pipeline = create_pipeline(settings, use_mock_depth=True)
        result = pipeline.process(image, k=3)

        # 3つのレイヤーが生成される
        assert result.optimal_k == 3
        assert result.mesh.num_layers == 3

        # 各レイヤーに頂点がある
        for layer in result.mesh.layers:
            assert len(layer.vertices) > 0

    def test_pipeline_with_small_image(self) -> None:
        """小さい画像での処理をテスト。"""
        image = Image.new("RGB", (10, 10), color="red")

        # 非累積モードで全体の頂点数を確認
        settings = ShadowboxSettings()
        settings.render.cumulative_layers = False
        settings.render.back_panel = False
        settings.render.layer_interpolation = 0
        pipeline = create_pipeline(settings, use_mock_depth=True)
        result = pipeline.process(image, k=2)

        assert result.mesh.total_vertices == 100  # 10x10ピクセル

    def test_pipeline_preserves_colors(self) -> None:
        """色情報が保存されることをテスト。"""
        # 単色の画像
        color = (123, 45, 67)
        image = Image.new("RGB", (20, 20), color=color)

        pipeline = create_pipeline(use_mock_depth=True)
        result = pipeline.process(image, k=1)

        # 全頂点が同じ色を持つ
        expected_color = np.array(color, dtype=np.uint8)
        for layer in result.mesh.layers:
            for vertex_color in layer.colors:
                assert np.array_equal(vertex_color, expected_color)


class TestBasePipelineResult:
    """BasePipelineResultのテスト。"""

    def test_pipeline_result_inherits_base(self) -> None:
        """PipelineResultがBasePipelineResultを継承していることを確認。"""
        assert issubclass(PipelineResult, BasePipelineResult)

    def test_triposr_result_inherits_base(self) -> None:
        """TripoSRPipelineResultがBasePipelineResultを継承していることを確認。"""
        from shadowbox.triposr.pipeline import TripoSRPipelineResult

        assert issubclass(TripoSRPipelineResult, BasePipelineResult)

    def test_base_has_common_fields(self) -> None:
        """BasePipelineResultが共通フィールドを持つことを確認。"""
        import dataclasses

        fields = {f.name for f in dataclasses.fields(BasePipelineResult)}
        assert "original_image" in fields
        assert "mesh" in fields
        assert "bbox" in fields
