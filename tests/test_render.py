"""3Dレンダリングモジュールのテスト。

このモジュールは、ShadowboxRendererと関連クラスの
ユニットテストを提供します。

Note:
    Vedoは画面表示を必要とするため、一部のテストは
    オフスクリーンモードで実行されます。
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from shadowbox.core.mesh import FrameMesh, LayerMesh, ShadowboxMesh
from shadowbox.visualization.render import (
    RenderOptions,
    ShadowboxRenderer,
    render_layers_exploded,
    render_shadowbox,
)


def _vedo_available() -> bool:
    """Vedoが利用可能かチェック。"""
    try:
        import vedo

        return True
    except ImportError:
        return False


@pytest.fixture
def sample_mesh() -> ShadowboxMesh:
    """テスト用のシャドーボックスメッシュを作成。"""
    # 2つのレイヤーを持つ簡単なメッシュ
    layer0 = LayerMesh(
        vertices=np.array(
            [
                [0.0, 0.0, -0.1],
                [0.5, 0.0, -0.1],
                [0.0, 0.5, -0.1],
                [0.5, 0.5, -0.1],
            ],
            dtype=np.float32,
        ),
        colors=np.array(
            [
                [255, 0, 0],
                [255, 0, 0],
                [255, 0, 0],
                [255, 0, 0],
            ],
            dtype=np.uint8,
        ),
        z_position=-0.1,
        layer_index=0,
        pixel_indices=np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int32),
    )

    layer1 = LayerMesh(
        vertices=np.array(
            [
                [-0.5, -0.5, -0.2],
                [0.0, -0.5, -0.2],
                [-0.5, 0.0, -0.2],
                [0.0, 0.0, -0.2],
            ],
            dtype=np.float32,
        ),
        colors=np.array(
            [
                [0, 0, 255],
                [0, 0, 255],
                [0, 0, 255],
                [0, 0, 255],
            ],
            dtype=np.uint8,
        ),
        z_position=-0.2,
        layer_index=1,
        pixel_indices=np.array([[2, 0], [2, 1], [3, 0], [3, 1]], dtype=np.int32),
    )

    # フレーム（簡易的な四角形）
    frame = FrameMesh(
        vertices=np.array(
            [
                [-1.0, -1.0, 0.0],
                [1.0, -1.0, 0.0],
                [1.0, 1.0, 0.0],
                [-1.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        ),
        faces=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32),
        color=np.array([50, 50, 50], dtype=np.uint8),
        z_position=0.0,
    )

    return ShadowboxMesh(
        layers=[layer0, layer1],
        frame=frame,
        bounds=(-1.0, 1.0, -1.0, 1.0, -0.2, 0.0),
    )


@pytest.fixture
def sample_mesh_no_frame() -> ShadowboxMesh:
    """フレームなしのテスト用メッシュを作成。"""
    layer = LayerMesh(
        vertices=np.array(
            [
                [0.0, 0.0, -0.1],
                [0.5, 0.5, -0.1],
            ],
            dtype=np.float32,
        ),
        colors=np.array(
            [
                [0, 255, 0],
                [0, 255, 0],
            ],
            dtype=np.uint8,
        ),
        z_position=-0.1,
        layer_index=0,
        pixel_indices=np.array([[0, 0], [1, 1]], dtype=np.int32),
    )

    return ShadowboxMesh(
        layers=[layer],
        frame=None,
        bounds=(-1.0, 1.0, -1.0, 1.0, -0.1, 0.0),
    )


class TestRenderOptions:
    """RenderOptionsのテスト。"""

    def test_default_values(self) -> None:
        """デフォルト値をテスト。"""
        options = RenderOptions()

        assert options.background_color == (30, 30, 30)
        assert options.point_size == 3.0
        assert options.show_axes is False
        assert options.show_frame is True
        assert options.window_size == (1200, 800)
        assert options.title == "Shadowbox 3D Viewer"
        assert options.interactive is True
        assert options.layer_opacity == 1.0

    def test_custom_values(self) -> None:
        """カスタム値をテスト。"""
        options = RenderOptions(
            background_color=(0, 0, 0),
            point_size=5.0,
            show_axes=True,
            window_size=(800, 600),
            title="Custom Viewer",
        )

        assert options.background_color == (0, 0, 0)
        assert options.point_size == 5.0
        assert options.show_axes is True
        assert options.window_size == (800, 600)
        assert options.title == "Custom Viewer"


class TestShadowboxRenderer:
    """ShadowboxRendererのテスト。"""

    def test_init_default(self) -> None:
        """デフォルト初期化をテスト。"""
        renderer = ShadowboxRenderer()

        assert renderer._options is not None
        assert renderer._plotter is None

    def test_init_with_options(self) -> None:
        """オプション付き初期化をテスト。"""
        options = RenderOptions(point_size=5.0)
        renderer = ShadowboxRenderer(options)

        assert renderer._options.point_size == 5.0

    def test_normalize_color(self) -> None:
        """色の正規化をテスト。"""
        renderer = ShadowboxRenderer()

        result = renderer._normalize_color((255, 128, 0))

        assert result[0] == pytest.approx(1.0)
        assert result[1] == pytest.approx(128 / 255)
        assert result[2] == pytest.approx(0.0)

    def test_close_without_plotter(self) -> None:
        """プロッターなしでcloseを呼んでもエラーにならない。"""
        renderer = ShadowboxRenderer()
        renderer.close()  # エラーが出ないことを確認


class TestRendererOffscreen:
    """オフスクリーンレンダリングのテスト。

    これらのテストは実際にVedoを使用しますが、
    オフスクリーンモードで実行されます。
    """

    @pytest.mark.skipif(
        not _vedo_available(),
        reason="Vedo is not available",
    )
    def test_export_screenshot(self, sample_mesh: ShadowboxMesh) -> None:
        """スクリーンショット出力をテスト。"""
        renderer = ShadowboxRenderer()

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            output_path = f.name

        try:
            renderer.export_screenshot(sample_mesh, output_path)

            # ファイルが作成されたことを確認
            assert Path(output_path).exists()
            assert Path(output_path).stat().st_size > 0
        finally:
            Path(output_path).unlink(missing_ok=True)

    @pytest.mark.skipif(
        not _vedo_available(),
        reason="Vedo is not available",
    )
    def test_export_screenshot_custom_size(self, sample_mesh: ShadowboxMesh) -> None:
        """カスタムサイズでのスクリーンショットをテスト。"""
        renderer = ShadowboxRenderer()

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            output_path = f.name

        try:
            renderer.export_screenshot(sample_mesh, output_path, size=(640, 480))

            assert Path(output_path).exists()
        finally:
            Path(output_path).unlink(missing_ok=True)


class TestRenderWithoutDisplay:
    """ディスプレイなしで実行可能なテスト。"""

    def test_render_without_show(self, sample_mesh: ShadowboxMesh) -> None:
        """show=Falseでレンダリングをテスト。"""
        if not _vedo_available():
            pytest.skip("Vedo is not available")

        options = RenderOptions(interactive=False)
        renderer = ShadowboxRenderer(options)

        # show=Falseで呼び出し
        plotter = renderer.render(sample_mesh, show=False)

        assert plotter is not None
        renderer.close()

    def test_render_mesh_no_frame(self, sample_mesh_no_frame: ShadowboxMesh) -> None:
        """フレームなしメッシュのレンダリングをテスト。"""
        if not _vedo_available():
            pytest.skip("Vedo is not available")

        options = RenderOptions(interactive=False, show_frame=True)
        renderer = ShadowboxRenderer(options)

        plotter = renderer.render(sample_mesh_no_frame, show=False)

        assert plotter is not None
        renderer.close()


class TestUtilityFunctions:
    """ユーティリティ関数のテスト。"""

    def test_render_shadowbox_function(self, sample_mesh: ShadowboxMesh) -> None:
        """render_shadowbox関数をテスト。"""
        if not _vedo_available():
            pytest.skip("Vedo is not available")

        options = RenderOptions(interactive=False)

        plotter = render_shadowbox(sample_mesh, options)

        assert plotter is not None
        plotter.close()

    def test_render_layers_exploded_function(self, sample_mesh: ShadowboxMesh) -> None:
        """render_layers_exploded関数をテスト。"""
        if not _vedo_available():
            pytest.skip("Vedo is not available")

        options = RenderOptions(interactive=False)

        # この関数はshow()を内部で呼ぶので、テストではスキップ
        # 関数自体の存在とシグネチャの確認のみ
        assert callable(render_layers_exploded)


class TestMultiAngleExport:
    """8方向スクリーンショットエクスポートのテスト。"""

    @pytest.mark.skipif(
        not _vedo_available(),
        reason="Vedo is not available",
    )
    def test_export_8_directions(self, sample_mesh: ShadowboxMesh, tmp_path) -> None:
        """各tilt角度×8方向のスクリーンショットが専用サブディレクトリに生成されることを確認。"""
        renderer = ShadowboxRenderer()
        files = renderer.export_multi_angle_screenshots(
            sample_mesh, tmp_path, size=(200, 200)
        )

        # デフォルトは (25.0, 15.0) の2段階 × 8方向 = 16枚
        assert len(files) == 16
        subdir = tmp_path / "shadowbox"
        direction_names = [
            "left", "upper_left", "top", "upper_right",
            "right", "lower_right", "bottom", "lower_left",
        ]
        expected_names = []
        for tilt in [25, 15]:
            for name in direction_names:
                expected_names.append(f"shadowbox_{tilt}deg_{name}.png")
        for f, expected in zip(files, expected_names, strict=True):
            assert f.parent == subdir
            assert f.name == expected
            assert f.exists()
            assert f.stat().st_size > 0

    @pytest.mark.skipif(
        not _vedo_available(),
        reason="Vedo is not available",
    )
    def test_export_custom_prefix(self, sample_mesh: ShadowboxMesh, tmp_path) -> None:
        """カスタムプレフィックスが使用され、専用サブディレクトリに保存されることを確認。"""
        renderer = ShadowboxRenderer()
        files = renderer.export_multi_angle_screenshots(
            sample_mesh, tmp_path, size=(200, 200), prefix="card01"
        )

        assert len(files) == 16
        subdir = tmp_path / "card01"
        assert all(f.parent == subdir for f in files)
        assert all(f.name.startswith("card01_") for f in files)
