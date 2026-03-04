"""メッシュ生成モジュールのテスト。

このモジュールは、MeshGeneratorと関連データクラスの
ユニットテストを提供します。
"""

import numpy as np
import pytest

from shadowbox.config import RenderSettings
from shadowbox.core.mesh import (
    FrameMesh,
    LayerMesh,
    MeshGenerator,
    MeshGeneratorProtocol,
    ShadowboxMesh,
)


class TestMeshGenerator:
    """MeshGeneratorのテスト。"""

    def test_init(self) -> None:
        """初期化をテスト。"""
        settings = RenderSettings()
        generator = MeshGenerator(settings)

        assert generator._settings == settings

    def test_generate_basic(self) -> None:
        """基本的なメッシュ生成をテスト。"""
        settings = RenderSettings(back_panel=False, layer_interpolation=0)
        generator = MeshGenerator(settings)

        # テスト用の画像とラベル
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        image[:5, :] = [255, 0, 0]   # 上半分は赤
        image[5:, :] = [0, 0, 255]   # 下半分は青

        labels = np.zeros((10, 10), dtype=np.int32)
        labels[:5, :] = 0  # 上半分はレイヤー0
        labels[5:, :] = 1  # 下半分はレイヤー1

        centroids = np.array([0.2, 0.8], dtype=np.float32)

        mesh = generator.generate(image, labels, centroids, include_frame=True)

        # 基本的な検証
        assert isinstance(mesh, ShadowboxMesh)
        assert mesh.num_layers == 2
        assert mesh.frame is not None
        assert len(mesh.bounds) == 6

    def test_generate_without_frame(self) -> None:
        """フレームなしのメッシュ生成をテスト。"""
        settings = RenderSettings(back_panel=False, layer_interpolation=0, cumulative_layers=False)
        generator = MeshGenerator(settings)

        image = np.zeros((10, 10, 3), dtype=np.uint8)
        labels = np.zeros((10, 10), dtype=np.int32)
        centroids = np.array([0.5], dtype=np.float32)

        mesh = generator.generate(image, labels, centroids, include_frame=False)

        assert mesh.frame is None
        assert mesh.num_layers == 1

    def test_layer_z_positions(self) -> None:
        """レイヤーのZ座標が正しい順序で設定されることをテスト。"""
        settings = RenderSettings(
            back_panel=False, layer_interpolation=0, cumulative_layers=False
        )
        generator = MeshGenerator(settings)

        image = np.zeros((10, 10, 3), dtype=np.uint8)
        labels = np.zeros((10, 10), dtype=np.int32)
        labels[:3, :] = 0
        labels[3:6, :] = 1
        labels[6:, :] = 2

        centroids = np.array([0.1, 0.5, 0.9], dtype=np.float32)

        mesh = generator.generate(image, labels, centroids, include_frame=False)

        # 全レイヤーがZ座標が負（フレームより奥）
        for layer in mesh.layers:
            assert layer.z_position < 0

        # レイヤー0が最も手前、レイヤー2が最も奥
        assert mesh.layers[0].z_position > mesh.layers[1].z_position
        assert mesh.layers[1].z_position > mesh.layers[2].z_position

    def test_layer_with_gap(self) -> None:
        """レイヤー間のギャップが正しく反映されることをテスト。"""
        settings = RenderSettings(
            back_panel=False, layer_interpolation=0, cumulative_layers=False
        )
        generator = MeshGenerator(settings)

        image = np.zeros((10, 10, 3), dtype=np.uint8)
        labels = np.zeros((10, 10), dtype=np.int32)
        labels[:5, :] = 0
        labels[5:, :] = 1

        centroids = np.array([0.2, 0.8], dtype=np.float32)

        mesh = generator.generate(image, labels, centroids, include_frame=False)

        # レイヤー0が手前、レイヤー1が奥
        assert mesh.layers[0].z_position > mesh.layers[1].z_position
        # 両方とも負の値（フレームより奥）
        assert mesh.layers[0].z_position < 0
        assert mesh.layers[1].z_position < 0

    def test_vertex_coordinates_normalized(self) -> None:
        """頂点座標がアスペクト比を保持して正規化されることをテスト。"""
        settings = RenderSettings()
        generator = MeshGenerator(settings)

        # 10x20の画像（幅20、高さ10）→ 横長: X=[-1,1], Y=[-0.5,0.5]
        image = np.zeros((10, 20, 3), dtype=np.uint8)
        labels = np.zeros((10, 20), dtype=np.int32)
        centroids = np.array([0.5], dtype=np.float32)

        mesh = generator.generate(image, labels, centroids, include_frame=False)

        vertices = mesh.layers[0].vertices

        # X座標（長辺）は[-1, 1]の範囲
        assert vertices[:, 0].min() == pytest.approx(-1.0, abs=0.01)
        assert vertices[:, 0].max() == pytest.approx(1.0, abs=0.01)

        # Y座標（短辺）は[-0.5, 0.5]の範囲
        assert vertices[:, 1].min() == pytest.approx(-0.5, abs=0.01)
        assert vertices[:, 1].max() == pytest.approx(0.5, abs=0.01)

    def test_vertex_coordinates_portrait(self) -> None:
        """縦長画像でアスペクト比が保持されることをテスト。"""
        settings = RenderSettings()
        generator = MeshGenerator(settings)

        # 20x10の画像（幅10、高さ20）→ 縦長: X=[-0.5,0.5], Y=[-1,1]
        image = np.zeros((20, 10, 3), dtype=np.uint8)
        labels = np.zeros((20, 10), dtype=np.int32)
        centroids = np.array([0.5], dtype=np.float32)

        mesh = generator.generate(image, labels, centroids, include_frame=False)

        vertices = mesh.layers[0].vertices

        # X座標（短辺）は[-0.5, 0.5]の範囲
        assert vertices[:, 0].min() == pytest.approx(-0.5, abs=0.01)
        assert vertices[:, 0].max() == pytest.approx(0.5, abs=0.01)

        # Y座標（長辺）は[-1, 1]の範囲
        assert vertices[:, 1].min() == pytest.approx(-1.0, abs=0.01)
        assert vertices[:, 1].max() == pytest.approx(1.0, abs=0.01)

    def test_vertex_coordinates_square(self) -> None:
        """正方形画像では[-1,1]のまま変わらないことをテスト。"""
        settings = RenderSettings()
        generator = MeshGenerator(settings)

        image = np.zeros((10, 10, 3), dtype=np.uint8)
        labels = np.zeros((10, 10), dtype=np.int32)
        centroids = np.array([0.5], dtype=np.float32)

        mesh = generator.generate(image, labels, centroids, include_frame=False)

        vertices = mesh.layers[0].vertices

        assert vertices[:, 0].min() == pytest.approx(-1.0, abs=0.01)
        assert vertices[:, 0].max() == pytest.approx(1.0, abs=0.01)
        assert vertices[:, 1].min() == pytest.approx(-1.0, abs=0.01)
        assert vertices[:, 1].max() == pytest.approx(1.0, abs=0.01)

    def test_colors_preserved(self) -> None:
        """色情報が正しく保存されることをテスト。"""
        settings = RenderSettings()
        generator = MeshGenerator(settings)

        image = np.zeros((4, 4, 3), dtype=np.uint8)
        image[0, 0] = [255, 0, 0]    # 赤
        image[0, 1] = [0, 255, 0]    # 緑
        image[0, 2] = [0, 0, 255]    # 青

        labels = np.zeros((4, 4), dtype=np.int32)
        centroids = np.array([0.5], dtype=np.float32)

        mesh = generator.generate(image, labels, centroids, include_frame=False)

        colors = mesh.layers[0].colors

        # 色がuint8で保存されている
        assert colors.dtype == np.uint8

        # 全ピクセルの色が含まれている
        assert len(colors) == 16


class TestLayerMesh:
    """LayerMeshデータクラスのテスト。"""

    def test_create(self) -> None:
        """LayerMeshの作成をテスト。"""
        vertices = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)
        colors = np.array([[255, 0, 0], [0, 255, 0]], dtype=np.uint8)
        pixel_indices = np.array([[0, 0], [0, 1]], dtype=np.int32)

        layer = LayerMesh(
            vertices=vertices,
            colors=colors,
            z_position=-0.1,
            layer_index=0,
            pixel_indices=pixel_indices,
        )

        assert layer.z_position == -0.1
        assert layer.layer_index == 0
        assert len(layer.vertices) == 2


class TestFrameMesh:
    """FrameMeshデータクラスのテスト。"""

    def test_create(self) -> None:
        """FrameMeshの作成をテスト。"""
        vertices = np.zeros((8, 3), dtype=np.float32)
        faces = np.zeros((8, 3), dtype=np.int32)
        color = np.array([30, 30, 30], dtype=np.uint8)

        frame = FrameMesh(
            vertices=vertices,
            faces=faces,
            color=color,
            z_position=0.0,
        )

        assert frame.z_position == 0.0
        assert len(frame.vertices) == 8


class TestShadowboxMesh:
    """ShadowboxMeshデータクラスのテスト。"""

    def test_num_layers(self) -> None:
        """num_layersプロパティをテスト。"""
        layers = [
            LayerMesh(
                vertices=np.zeros((10, 3), dtype=np.float32),
                colors=np.zeros((10, 3), dtype=np.uint8),
                z_position=-0.1,
                layer_index=i,
                pixel_indices=np.zeros((10, 2), dtype=np.int32),
            )
            for i in range(3)
        ]

        mesh = ShadowboxMesh(layers=layers, frame=None, bounds=(0, 0, 0, 0, 0, 0))

        assert mesh.num_layers == 3

    def test_total_vertices(self) -> None:
        """total_verticesプロパティをテスト。"""
        layers = [
            LayerMesh(
                vertices=np.zeros((10, 3), dtype=np.float32),
                colors=np.zeros((10, 3), dtype=np.uint8),
                z_position=-0.1,
                layer_index=0,
                pixel_indices=np.zeros((10, 2), dtype=np.int32),
            ),
            LayerMesh(
                vertices=np.zeros((20, 3), dtype=np.float32),
                colors=np.zeros((20, 3), dtype=np.uint8),
                z_position=-0.2,
                layer_index=1,
                pixel_indices=np.zeros((20, 2), dtype=np.int32),
            ),
        ]

        mesh = ShadowboxMesh(layers=layers, frame=None, bounds=(0, 0, 0, 0, 0, 0))

        assert mesh.total_vertices == 30


class TestMeshGeneratorEdgeCases:
    """メッシュ生成のエッジケーステスト。"""

    def test_single_pixel_image(self) -> None:
        """1ピクセル画像のメッシュ生成をテスト。"""
        settings = RenderSettings(back_panel=False, layer_interpolation=0, cumulative_layers=False)
        generator = MeshGenerator(settings)

        image = np.array([[[255, 0, 0]]], dtype=np.uint8)  # 1x1の赤
        labels = np.array([[0]], dtype=np.int32)
        centroids = np.array([0.5], dtype=np.float32)

        mesh = generator.generate(image, labels, centroids, include_frame=False)

        assert mesh.num_layers == 1
        assert len(mesh.layers[0].vertices) == 1

    def test_empty_layer(self) -> None:
        """空のレイヤーが含まれる場合のテスト（非累積モード）。"""
        settings = RenderSettings()
        settings.cumulative_layers = False  # 従来の穴あきモード
        generator = MeshGenerator(settings)

        image = np.zeros((10, 10, 3), dtype=np.uint8)
        # 全ピクセルがレイヤー0、レイヤー1は空
        labels = np.zeros((10, 10), dtype=np.int32)
        centroids = np.array([0.2, 0.8], dtype=np.float32)

        mesh = generator.generate(image, labels, centroids, include_frame=False)

        # レイヤー0には頂点がある
        assert len(mesh.layers[0].vertices) == 100
        # レイヤー1は空（非累積モードなので空になる）
        assert len(mesh.layers[1].vertices) == 0

    def test_cumulative_layers(self) -> None:
        """累積レイヤーモードのテスト。"""
        settings = RenderSettings()
        settings.cumulative_layers = True  # 累積モード（デフォルト）
        generator = MeshGenerator(settings)

        image = np.zeros((10, 10, 3), dtype=np.uint8)
        # 全ピクセルがレイヤー0
        labels = np.zeros((10, 10), dtype=np.int32)
        centroids = np.array([0.2, 0.8], dtype=np.float32)

        mesh = generator.generate(image, labels, centroids, include_frame=False)

        # レイヤー0には頂点がある
        assert len(mesh.layers[0].vertices) == 100
        # レイヤー1も全ピクセルを含む（累積モードなので）
        assert len(mesh.layers[1].vertices) == 100

    def test_bounds_calculation(self) -> None:
        """バウンディングボックスの計算をテスト。"""
        settings = RenderSettings()
        generator = MeshGenerator(settings)

        image = np.zeros((10, 10, 3), dtype=np.uint8)
        labels = np.zeros((10, 10), dtype=np.int32)
        centroids = np.array([0.5], dtype=np.float32)

        mesh = generator.generate(image, labels, centroids, include_frame=True)

        min_x, max_x, min_y, max_y, min_z, max_z = mesh.bounds

        # X, Y座標は-1.05〜1.05（フレームのマージン込み）
        assert min_x == pytest.approx(-1.05, abs=0.01)
        assert max_x == pytest.approx(1.05, abs=0.01)
        assert min_y == pytest.approx(-1.05, abs=0.01)
        assert max_y == pytest.approx(1.05, abs=0.01)

        # Z座標は最奥のレイヤーから最前面のフレームまで
        assert min_z < 0  # レイヤーは負のZ
        assert max_z == pytest.approx(0.0, abs=0.01)  # フレームはz=0

    def test_generate_raw_depth(self) -> None:
        """生深度メッシュ生成をテスト。"""
        settings = RenderSettings(back_panel=False, layer_interpolation=0)
        generator = MeshGenerator(settings)

        # テスト用の画像と深度マップ
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        image[:, :] = [255, 128, 64]  # 均一な色

        # グラデーション深度マップ（0.0〜1.0）
        depth_map = np.linspace(0, 1, 100).reshape(10, 10).astype(np.float32)

        mesh = generator.generate_raw_depth(
            image, depth_map, include_frame=True, depth_scale=1.0
        )

        # 基本的な検証
        assert isinstance(mesh, ShadowboxMesh)
        assert mesh.num_layers == 1  # 生深度は1レイヤー
        assert mesh.total_vertices == 100  # 全ピクセル

        # Z座標が連続的に変化していることを確認
        z_coords = mesh.layers[0].vertices[:, 2]
        assert z_coords.min() < z_coords.max()  # Z座標に幅がある
        assert np.all(z_coords <= 0)  # すべて負（フレームより奥）

    def test_generate_raw_depth_without_frame(self) -> None:
        """フレームなしの生深度メッシュ生成をテスト。"""
        settings = RenderSettings(back_panel=False, layer_interpolation=0)
        generator = MeshGenerator(settings)

        image = np.zeros((5, 5, 3), dtype=np.uint8)
        depth_map = np.ones((5, 5), dtype=np.float32) * 0.5

        mesh = generator.generate_raw_depth(
            image, depth_map, include_frame=False
        )

        assert mesh.frame is None
        assert mesh.num_layers == 1


class TestNewFeatures:
    """新機能（back_panel, layer_interpolation等）のテスト。"""

    def test_back_panel_enabled(self) -> None:
        """背面パネルが有効な場合、レイヤーが1つ追加されることをテスト。"""
        settings = RenderSettings(back_panel=True, layer_interpolation=0, cumulative_layers=False)
        generator = MeshGenerator(settings)

        image = np.zeros((10, 10, 3), dtype=np.uint8)
        labels = np.zeros((10, 10), dtype=np.int32)
        labels[:5, :] = 0
        labels[5:, :] = 1
        centroids = np.array([0.2, 0.8], dtype=np.float32)

        mesh = generator.generate(image, labels, centroids, include_frame=False)

        # 2レイヤー + 背面パネル = 3レイヤー
        assert mesh.num_layers == 3

    def test_back_panel_disabled(self) -> None:
        """背面パネルが無効な場合のレイヤー数をテスト。"""
        settings = RenderSettings(back_panel=False, layer_interpolation=0, cumulative_layers=False)
        generator = MeshGenerator(settings)

        image = np.zeros((10, 10, 3), dtype=np.uint8)
        labels = np.zeros((10, 10), dtype=np.int32)
        labels[:5, :] = 0
        labels[5:, :] = 1
        centroids = np.array([0.2, 0.8], dtype=np.float32)

        mesh = generator.generate(image, labels, centroids, include_frame=False)

        # 2レイヤーのみ
        assert mesh.num_layers == 2

    def test_back_panel_is_at_back(self) -> None:
        """背面パネルが最も奥に配置されることをテスト。"""
        settings = RenderSettings(back_panel=True, layer_interpolation=0, cumulative_layers=False)
        generator = MeshGenerator(settings)

        image = np.zeros((10, 10, 3), dtype=np.uint8)
        labels = np.zeros((10, 10), dtype=np.int32)
        labels[:5, :] = 0
        labels[5:, :] = 1
        centroids = np.array([0.2, 0.8], dtype=np.float32)

        mesh = generator.generate(image, labels, centroids, include_frame=False)

        # 最後のレイヤー（背面パネル）が最も奥
        back_panel = mesh.layers[-1]
        other_layers = mesh.layers[:-1]

        for layer in other_layers:
            assert layer.z_position >= back_panel.z_position

    def test_frame_depth_setting(self) -> None:
        """frame_depth設定が正しく適用されることをテスト。"""
        settings = RenderSettings(
            frame_depth=1.0,
            back_panel=False,
            layer_interpolation=0,
            cumulative_layers=False,
        )
        generator = MeshGenerator(settings)

        image = np.zeros((10, 10, 3), dtype=np.uint8)
        labels = np.zeros((10, 10), dtype=np.int32)
        centroids = np.array([0.5], dtype=np.float32)

        mesh = generator.generate(image, labels, centroids, include_frame=True)

        # フレームの深さ範囲内にレイヤーが配置される
        min_z = min(layer.z_position for layer in mesh.layers)
        assert min_z >= -settings.frame_depth


class TestLayerSpacingMode:
    """layer_spacing_mode のテスト。"""

    def test_even_spacing_default(self) -> None:
        """デフォルト(even)モードで均等配置されることをテスト。"""
        settings = RenderSettings(
            frame_depth=0.5,
            back_panel=False,
            layer_interpolation=0,
            cumulative_layers=False,
        )
        generator = MeshGenerator(settings)

        image = np.zeros((10, 10, 3), dtype=np.uint8)
        labels = np.zeros((10, 10), dtype=np.int32)
        labels[:3, :] = 0
        labels[3:6, :] = 1
        labels[6:, :] = 2

        centroids = np.array([0.1, 0.5, 1.0], dtype=np.float32)

        mesh = generator.generate(image, labels, centroids, include_frame=False)

        # 均等配置: spacing = 0.5 / 3 ≈ 0.1667
        expected_spacing = 0.5 / 3
        assert mesh.layers[0].z_position == pytest.approx(-expected_spacing * 1, abs=1e-5)
        assert mesh.layers[1].z_position == pytest.approx(-expected_spacing * 2, abs=1e-5)
        assert mesh.layers[2].z_position == pytest.approx(-expected_spacing * 3, abs=1e-5)

    def test_proportional_spacing(self) -> None:
        """proportionalモードでcentroidに比例した配置をテスト。"""
        settings = RenderSettings(
            frame_depth=0.5,
            back_panel=False,
            layer_interpolation=0,
            cumulative_layers=False,
            layer_spacing_mode="proportional",
        )
        generator = MeshGenerator(settings)

        image = np.zeros((10, 10, 3), dtype=np.uint8)
        labels = np.zeros((10, 10), dtype=np.int32)
        labels[:3, :] = 0
        labels[3:6, :] = 1
        labels[6:, :] = 2

        centroids = np.array([0.1, 0.5, 1.0], dtype=np.float32)

        mesh = generator.generate(image, labels, centroids, include_frame=False)

        # proportional: z = -frame_depth * (c / max_c)
        # max_c = 1.0, frame_depth = 0.5
        assert mesh.layers[0].z_position == pytest.approx(-0.5 * 0.1, abs=1e-5)
        assert mesh.layers[1].z_position == pytest.approx(-0.5 * 0.5, abs=1e-5)
        assert mesh.layers[2].z_position == pytest.approx(-0.5 * 1.0, abs=1e-5)

    def test_proportional_with_close_centroids(self) -> None:
        """近接centroidでproportional配置が密になることをテスト。"""
        settings = RenderSettings(
            frame_depth=0.5,
            back_panel=False,
            layer_interpolation=0,
            cumulative_layers=False,
            layer_spacing_mode="proportional",
        )
        generator = MeshGenerator(settings)

        image = np.zeros((10, 10, 3), dtype=np.uint8)
        labels = np.zeros((10, 10), dtype=np.int32)
        labels[:3, :] = 0
        labels[3:6, :] = 1
        labels[6:8, :] = 2
        labels[8:, :] = 3

        # 手前に密集、奥は離れた centroid
        centroids = np.array([0.05, 0.10, 0.80, 0.95], dtype=np.float32)

        mesh = generator.generate(image, labels, centroids, include_frame=False)

        # max_c = 0.95
        z0 = mesh.layers[0].z_position
        z1 = mesh.layers[1].z_position
        z2 = mesh.layers[2].z_position
        z3 = mesh.layers[3].z_position

        # 手前2層の間隔は狭く、後方は広い
        gap_01 = abs(z1 - z0)
        gap_12 = abs(z2 - z1)
        assert gap_01 < gap_12

        # 最奥レイヤーが -frame_depth に最も近い
        assert z3 == pytest.approx(-0.5, abs=1e-5)

    def test_proportional_with_pop_out(self) -> None:
        """proportionalモードでpop_outオフセットが正しく適用されるテスト。"""
        settings = RenderSettings(
            frame_depth=0.5,
            back_panel=False,
            layer_interpolation=0,
            cumulative_layers=False,
            layer_spacing_mode="proportional",
            layer_pop_out=0.5,
        )
        generator = MeshGenerator(settings)

        image = np.zeros((10, 10, 3), dtype=np.uint8)
        labels = np.zeros((10, 10), dtype=np.int32)
        labels[:5, :] = 0
        labels[5:, :] = 1

        centroids = np.array([0.2, 1.0], dtype=np.float32)

        mesh = generator.generate(image, labels, centroids, include_frame=False)

        # pop_out_offset = 0.5 * 0.5 = 0.25
        # base z0 = -0.5 * (0.2 / 1.0) = -0.1, with pop_out: -0.1 + 0.25 = 0.15
        # base z1 = -0.5 * (1.0 / 1.0) = -0.5, with pop_out: -0.5 + 0.25 = -0.25
        assert mesh.layers[0].z_position == pytest.approx(0.15, abs=1e-5)
        assert mesh.layers[1].z_position == pytest.approx(-0.25, abs=1e-5)

    def test_proportional_zero_centroids_fallback(self) -> None:
        """centroidが全て0の場合、均等配置にフォールバックするテスト。"""
        settings = RenderSettings(
            frame_depth=0.5,
            back_panel=False,
            layer_interpolation=0,
            cumulative_layers=False,
            layer_spacing_mode="proportional",
        )
        generator = MeshGenerator(settings)

        image = np.zeros((10, 10, 3), dtype=np.uint8)
        labels = np.zeros((10, 10), dtype=np.int32)
        labels[:5, :] = 0
        labels[5:, :] = 1

        centroids = np.array([0.0, 0.0], dtype=np.float32)

        mesh = generator.generate(image, labels, centroids, include_frame=False)

        # フォールバック: 均等配置
        expected_spacing = 0.5 / 2
        assert mesh.layers[0].z_position == pytest.approx(-expected_spacing * 1, abs=1e-5)
        assert mesh.layers[1].z_position == pytest.approx(-expected_spacing * 2, abs=1e-5)


class TestContourMode:
    """等高線カット（contour）モードのテスト。"""

    def _make_generator(self, **kwargs) -> MeshGenerator:
        defaults = dict(
            layer_mask_mode="contour",
            back_panel=False,
            layer_interpolation=0,
            cumulative_layers=True,
            frame_depth=0.5,
        )
        defaults.update(kwargs)
        return MeshGenerator(RenderSettings(**defaults))

    def _make_depth_gradient(self, h: int = 10, w: int = 10) -> np.ndarray:
        """上が浅く(0)、下が深い(1)のグラデーション深度マップを生成。"""
        return np.linspace(0, 1, h).reshape(-1, 1).repeat(w, axis=1).astype(
            np.float32
        )

    def test_contour_basic(self) -> None:
        """contourモードで手前レイヤーのピクセル数 < 奥レイヤー。"""
        generator = self._make_generator()

        image = np.zeros((10, 10, 3), dtype=np.uint8)
        labels = np.zeros((10, 10), dtype=np.int32)
        labels[:5, :] = 0
        labels[5:, :] = 1
        centroids = np.array([0.2, 0.8], dtype=np.float32)
        depth_map = self._make_depth_gradient()

        mesh = generator.generate(
            image, labels, centroids, include_frame=False, depth_map=depth_map
        )

        # 手前レイヤー(layer 0)のピクセル数 < 奥レイヤー(layer 1)
        assert len(mesh.layers[0].vertices) < len(mesh.layers[1].vertices)

    def test_contour_back_layer_has_all_pixels(self) -> None:
        """最奥レイヤーが全イラストピクセルを含む。"""
        generator = self._make_generator()

        image = np.zeros((10, 10, 3), dtype=np.uint8)
        labels = np.zeros((10, 10), dtype=np.int32)
        labels[:5, :] = 0
        labels[5:, :] = 1
        centroids = np.array([0.2, 0.8], dtype=np.float32)
        depth_map = self._make_depth_gradient()

        mesh = generator.generate(
            image, labels, centroids, include_frame=False, depth_map=depth_map
        )

        # 最奥レイヤー (threshold = -back_z / frame_depth = 1.0)
        # depth_map <= 1.0 → 全ピクセル
        total_illustration = np.sum(labels >= 0)
        assert len(mesh.layers[-1].vertices) == total_illustration

    def test_contour_excludes_frame_pixels(self) -> None:
        """labels==-1 のフレームピクセルが contour マスクから除外される。"""
        generator = self._make_generator()

        image = np.zeros((10, 10, 3), dtype=np.uint8)
        labels = np.zeros((10, 10), dtype=np.int32)
        labels[:3, :] = -1  # フレームピクセル
        labels[3:7, :] = 0
        labels[7:, :] = 1
        centroids = np.array([0.2, 0.8], dtype=np.float32)
        depth_map = np.ones((10, 10), dtype=np.float32) * 0.1  # 全ピクセル浅い

        mesh = generator.generate(
            image, labels, centroids, include_frame=False, depth_map=depth_map
        )

        # フレームレイヤー (labels==-1) は先頭に作成される
        # イラストレイヤーはフレームピクセルを含まない
        illustration_layers = [
            l for l in mesh.layers if l.layer_index >= 0
        ]
        for layer in illustration_layers:
            # フレーム行(row 0-2)のピクセルが含まれないことを確認
            if len(layer.pixel_indices) > 0:
                assert np.all(layer.pixel_indices[:, 0] >= 3)

    def test_contour_with_interpolation(self) -> None:
        """補間レイヤーも等高線形状になる。"""
        generator = self._make_generator(layer_interpolation=1)

        image = np.zeros((10, 10, 3), dtype=np.uint8)
        labels = np.zeros((10, 10), dtype=np.int32)
        labels[:5, :] = 0
        labels[5:, :] = 1
        centroids = np.array([0.2, 0.8], dtype=np.float32)
        depth_map = self._make_depth_gradient()

        mesh = generator.generate(
            image, labels, centroids, include_frame=False, depth_map=depth_map
        )

        # 2元レイヤー + 補間レイヤーが存在
        # layer_interpolation=1 → 各レイヤー後に1補間 + 最終レイヤー後に1補間
        # total: 2 original + 2 interpolation = 4
        assert mesh.num_layers == 4

        # 補間レイヤーも contour 形状（ピクセル数がレイヤー0より多い）
        layer0_count = len(mesh.layers[0].vertices)
        interp_layer = mesh.layers[1]  # layer 0 の後の補間
        assert len(interp_layer.vertices) >= layer0_count

    def test_cluster_mode_unchanged(self) -> None:
        """cluster モードの既存動作が変わらない。"""
        generator_cluster = MeshGenerator(RenderSettings(
            layer_mask_mode="cluster",
            back_panel=False,
            layer_interpolation=0,
            cumulative_layers=True,
        ))

        image = np.zeros((10, 10, 3), dtype=np.uint8)
        labels = np.zeros((10, 10), dtype=np.int32)
        labels[:5, :] = 0
        labels[5:, :] = 1
        centroids = np.array([0.2, 0.8], dtype=np.float32)
        depth_map = self._make_depth_gradient()

        mesh = generator_cluster.generate(
            image, labels, centroids, include_frame=False, depth_map=depth_map
        )

        # cluster モードでは cumulative なので両レイヤーが100ピクセル
        assert len(mesh.layers[0].vertices) == 50  # layer 0 のみ
        assert len(mesh.layers[1].vertices) == 100  # layer 0 + 1


class TestMeshGeneratorProtocol:
    """MeshGeneratorProtocolのテスト。"""

    def test_mesh_generator_implements_protocol(self) -> None:
        """MeshGeneratorがプロトコルを満たすことを確認。"""
        settings = RenderSettings()
        generator = MeshGenerator(settings)

        # プロトコルで定義されたメソッドを持っていることを確認
        assert hasattr(generator, "generate")
        assert hasattr(generator, "generate_raw_depth")

        # 型チェックとしてプロトコル型に代入可能であることを確認
        protocol_generator: MeshGeneratorProtocol = generator
        assert protocol_generator is not None
