"""3Dレンダリングモジュール。

このモジュールは、Vedoを使用してシャドーボックスメッシュを
インタラクティブに3D表示する機能を提供します。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray

from shadowbox.core.mesh import ShadowboxMesh

if TYPE_CHECKING:
    import vedo

    from shadowbox.core.mesh import FrameMesh


def _is_jupyter() -> bool:
    """Jupyter環境で実行中かどうかを検出。

    Returns:
        Jupyter環境の場合True。
    """
    try:
        from IPython import get_ipython

        ipython = get_ipython()
        if ipython is None:
            return False

        # ZMQInteractiveShell (Jupyter Notebook/Lab) をチェック
        shell_class = ipython.__class__.__name__
        if "ZMQInteractiveShell" in shell_class:
            return True

        # ipykernel モジュールから実行されているかチェック
        if hasattr(ipython, "kernel"):
            return True

    except (ImportError, AttributeError, NameError):
        pass
    return False


@dataclass
class RenderOptions:
    """レンダリングオプション。

    Attributes:
        background_color: 背景色 (R, G, B) 各0-255。
        point_size: ポイントのサイズ（pointsモード時）。
        mesh_size: メッシュの各ポイントを表す四角形のサイズ（meshモード時）。
        show_axes: 軸を表示するかどうか。
        show_frame: フレームを表示するかどうか。
        window_size: ウィンドウサイズ (幅, 高さ)。
        title: ウィンドウタイトル。
        interactive: インタラクティブモードを有効にするか。
        layer_opacity: レイヤーの不透明度 (0.0-1.0)。
        render_mode: 描画モード ("points": ポイント, "mesh": メッシュ面)。
    """

    background_color: tuple[int, int, int] = (30, 30, 30)
    point_size: float = 3.0
    mesh_size: float = 0.008
    show_axes: bool = False
    show_frame: bool = True
    window_size: tuple[int, int] = (1200, 800)
    title: str = "Shadowbox 3D Viewer"
    interactive: bool = True
    layer_opacity: float = 1.0
    render_mode: Literal["points", "mesh"] = "points"


class ShadowboxRenderer:
    """シャドーボックスの3Dレンダラー。

    Vedoを使用してShadowboxMeshをインタラクティブに
    3D表示します。マウスでの回転・ズーム・パンに対応。

    Attributes:
        options: レンダリングオプション。

    Example:
        >>> from shadowbox.core.mesh import ShadowboxMesh
        >>> renderer = ShadowboxRenderer()
        >>> renderer.render(mesh)
    """

    def __init__(self, options: RenderOptions | None = None) -> None:
        """レンダラーを初期化。

        Args:
            options: レンダリングオプション。Noneの場合はデフォルト。
        """
        self._options = options or RenderOptions()
        self._plotter = None

    def render(
        self,
        mesh: ShadowboxMesh,
        show: bool = True,
    ) -> vedo.Plotter:
        """シャドーボックスメッシュを3Dレンダリング。

        Args:
            mesh: レンダリングするシャドーボックスメッシュ。
            show: 即座に表示するかどうか。

        Returns:
            VedoのPlotterオブジェクト。

        Example:
            >>> renderer = ShadowboxRenderer()
            >>> plotter = renderer.render(mesh, show=True)
        """
        import vedo

        # プロッターを作成
        self._plotter = vedo.Plotter(
            size=self._options.window_size,
            title=self._options.title,
            bg=self._normalize_color(self._options.background_color),
        )

        actors = []

        # 各レイヤーをポイントクラウドとして追加
        for layer in mesh.layers:
            if len(layer.vertices) == 0:
                continue

            # ポイントクラウドを作成
            points = vedo.Points(layer.vertices, r=self._options.point_size)

            # 色を設定（Vedoはpointdata["RGB"]で頂点ごとの色を設定）
            points.pointdata["RGB"] = layer.colors
            points.pointdata.select("RGB")
            points.alpha(self._options.layer_opacity)
            actors.append(points)

        # フレームを追加
        if self._options.show_frame and mesh.frame is not None:
            frame_mesh = self._create_frame_mesh(mesh.frame)
            if frame_mesh is not None:
                actors.append(frame_mesh)

        # 軸を表示
        if self._options.show_axes:
            axes = vedo.Axes(
                actors[0] if actors else None,
                xtitle="X",
                ytitle="Y",
                ztitle="Z (深度)",
            )
            actors.append(axes)

        # アクターを追加
        self._plotter.add(actors)

        # カメラ位置を調整（正面から少し斜めに）
        # K3Dバックエンドではカメラ操作がサポートされないためスキップ
        if self._plotter.camera is not None:
            self._plotter.camera.SetPosition(0, 0, 3)
            self._plotter.camera.SetFocalPoint(0, 0, -0.5)
            self._plotter.camera.SetViewUp(0, 1, 0)

        if show:
            self._plotter.show(interactive=self._options.interactive)

        return self._plotter

    def render_layers_separately(
        self,
        mesh: ShadowboxMesh,
        gap_multiplier: float = 2.0,
    ) -> vedo.Plotter:
        """レイヤーを分離して表示。

        デバッグ用に、各レイヤーを本来の位置より離して
        表示することで、レイヤー構造を確認しやすくします。

        Args:
            mesh: レンダリングするシャドーボックスメッシュ。
            gap_multiplier: レイヤー間隔の倍率。

        Returns:
            VedoのPlotterオブジェクト。

        Example:
            >>> renderer = ShadowboxRenderer()
            >>> plotter = renderer.render_layers_separately(mesh, gap_multiplier=3.0)
        """
        import vedo

        self._plotter = vedo.Plotter(
            size=self._options.window_size,
            title=f"{self._options.title} (レイヤー分離表示)",
            bg=self._normalize_color(self._options.background_color),
        )

        actors = []

        for i, layer in enumerate(mesh.layers):
            if len(layer.vertices) == 0:
                continue

            # Z座標を調整（より離す）
            vertices = layer.vertices.copy()
            original_z = layer.z_position
            vertices[:, 2] = original_z * gap_multiplier

            points = vedo.Points(vertices, r=self._options.point_size)
            points.pointdata["RGB"] = layer.colors
            points.pointdata.select("RGB")
            points.alpha(self._options.layer_opacity)
            actors.append(points)

            # レイヤー番号のラベルを追加
            label_pos = [0, 1.2, original_z * gap_multiplier]
            label = vedo.Text3D(
                f"Layer {i}",
                pos=label_pos,
                s=0.08,
                c="white",
                justify="center",
            )
            actors.append(label)

        self._plotter.add(actors)
        self._plotter.show(interactive=self._options.interactive)

        return self._plotter

    def export_screenshot(
        self,
        mesh: ShadowboxMesh,
        filename: str,
        size: tuple[int, int] | None = None,
    ) -> None:
        """レンダリング結果をスクリーンショットとして保存。

        Args:
            mesh: レンダリングするシャドーボックスメッシュ。
            filename: 保存先のファイルパス。
            size: 画像サイズ。Noneの場合はウィンドウサイズ。

        Example:
            >>> renderer = ShadowboxRenderer()
            >>> renderer.export_screenshot(mesh, "shadowbox.png")
        """
        import vedo

        # オフスクリーンレンダリング
        if size:
            orig_size = self._options.window_size
            self._options.window_size = size

        plotter = vedo.Plotter(
            size=self._options.window_size,
            offscreen=True,
            bg=self._normalize_color(self._options.background_color),
        )

        actors = []
        for layer in mesh.layers:
            if len(layer.vertices) == 0:
                continue

            points = vedo.Points(layer.vertices, r=self._options.point_size)
            points.pointdata["RGB"] = layer.colors
            points.pointdata.select("RGB")
            actors.append(points)

        if self._options.show_frame and mesh.frame is not None:
            frame_mesh = self._create_frame_mesh(mesh.frame)
            if frame_mesh is not None:
                actors.append(frame_mesh)

        plotter.add(actors)
        if plotter.camera is not None:
            plotter.camera.SetPosition(0, 0, 3)
            plotter.camera.SetFocalPoint(0, 0, -0.5)
            plotter.camera.SetViewUp(0, 1, 0)

        plotter.screenshot(filename)
        plotter.close()

        if size:
            self._options.window_size = orig_size

    def export_multi_angle_screenshots(
        self,
        mesh: ShadowboxMesh,
        output_dir: str | Path,
        size: tuple[int, int] = (800, 800),
        tilt_degrees_list: Sequence[float] = (25.0, 15.0),
        prefix: str = "shadowbox",
    ) -> list[Path]:
        """8方向からのスクリーンショットを一括エクスポート。

        各tilt角度ごとに8方向を出力する。

        Args:
            mesh: レンダリングするシャドーボックスメッシュ。
            output_dir: 出力ディレクトリ。
            size: 各画像のサイズ。
            tilt_degrees_list: 傾斜角度のリスト（度）。
            prefix: ファイル名のプレフィックス。

        Returns:
            生成されたファイルパスのリスト。
        """
        import math

        import vedo

        output_path = Path(output_dir) / prefix
        output_path.mkdir(parents=True, exist_ok=True)

        focal = np.array([0.0, 0.0, -0.5])
        distance = 4.5
        sqrt2 = math.sqrt(2.0)

        direction_names = [
            ("left", 315, 0.0),
            ("upper_left", 315, 1.0 / sqrt2),
            ("top", 0, 1.0),
            ("upper_right", 45, 1.0 / sqrt2),
            ("right", 45, 0.0),
            ("lower_right", 45, -1.0 / sqrt2),
            ("bottom", 0, -1.0),
            ("lower_left", 315, -1.0 / sqrt2),
        ]

        saved_files: list[Path] = []

        for tilt_degrees in tilt_degrees_list:
            tilt_label = f"{int(tilt_degrees)}deg"

            for name, azimuth_deg, elev_factor in direction_names:
                elevation_deg = tilt_degrees * elev_factor
                az = math.radians(azimuth_deg)
                el = math.radians(elevation_deg)

                cam_x = focal[0] + distance * math.cos(el) * math.sin(az)
                cam_y = focal[1] + distance * math.sin(el)
                cam_z = focal[2] + distance * math.cos(el) * math.cos(az)

                plotter = vedo.Plotter(
                    size=size,
                    offscreen=True,
                    bg=self._normalize_color(self._options.background_color),
                )

                actors = []
                for layer in mesh.layers:
                    if len(layer.vertices) == 0:
                        continue
                    points = vedo.Points(
                        layer.vertices, r=self._options.point_size
                    )
                    points.pointdata["RGB"] = layer.colors
                    points.pointdata.select("RGB")
                    points.alpha(self._options.layer_opacity)
                    actors.append(points)

                if self._options.show_frame and mesh.frame is not None:
                    frame_mesh = self._create_frame_mesh(mesh.frame)
                    if frame_mesh is not None:
                        actors.append(frame_mesh)

                plotter.add(actors)
                if plotter.camera is not None:
                    plotter.camera.SetPosition(cam_x, cam_y, cam_z)
                    plotter.camera.SetFocalPoint(*focal)
                    plotter.camera.SetViewUp(0, 1, 0)

                filepath = output_path / f"{prefix}_{tilt_label}_{name}.png"
                plotter.screenshot(str(filepath))
                plotter.close()
                saved_files.append(filepath)

        return saved_files

    def _create_frame_mesh(self, frame: FrameMesh) -> vedo.Mesh | None:
        """フレームメッシュをVedoメッシュに変換。

        Args:
            frame: フレームメッシュデータ。

        Returns:
            Vedoメッシュオブジェクト。頂点がない場合はNone。
        """
        import vedo

        if len(frame.vertices) == 0:
            return None

        color_normalized = frame.color.astype(np.float32) / 255.0

        mesh = vedo.Mesh([frame.vertices, frame.faces])
        mesh.c(color_normalized)
        mesh.alpha(0.9)

        return mesh

    def _normalize_color(
        self,
        color: tuple[int, int, int],
    ) -> tuple[float, float, float]:
        """色を0-255から0-1の範囲に正規化。

        Args:
            color: RGB色 (0-255)。

        Returns:
            正規化されたRGB色 (0-1)。
        """
        return (color[0] / 255.0, color[1] / 255.0, color[2] / 255.0)

    def close(self) -> None:
        """プロッターを閉じる。"""
        if self._plotter is not None:
            self._plotter.close()
            self._plotter = None


def _points_to_mesh_data(
    vertices: NDArray,
    colors: NDArray,
    size: float = 0.008,
) -> tuple[NDArray, NDArray, NDArray]:
    """ポイントをメッシュデータ（四角形）に変換。

    Args:
        vertices: 頂点座標 (N, 3)。
        colors: 頂点色 (N, 3)。
        size: 各ポイントを表す四角形のサイズ。

    Returns:
        (mesh_vertices, mesh_faces, face_colors) のタプル。
    """
    n = len(vertices)
    half = size / 2

    # 各ポイントを4頂点に展開
    mesh_vertices = np.zeros((n * 4, 3), dtype=np.float32)
    face_colors = np.zeros((n * 2, 3), dtype=np.uint8)
    faces = np.zeros((n * 2, 3), dtype=np.int32)

    for i, (v, c) in enumerate(zip(vertices, colors, strict=True)):
        base = i * 4
        # 四角形の4頂点
        mesh_vertices[base] = [v[0] - half, v[1] - half, v[2]]
        mesh_vertices[base + 1] = [v[0] + half, v[1] - half, v[2]]
        mesh_vertices[base + 2] = [v[0] + half, v[1] + half, v[2]]
        mesh_vertices[base + 3] = [v[0] - half, v[1] + half, v[2]]

        # 2つの三角形
        face_base = i * 2
        faces[face_base] = [base, base + 1, base + 2]
        faces[face_base + 1] = [base, base + 2, base + 3]

        # 面の色（2面とも同じ色）
        face_colors[face_base] = c
        face_colors[face_base + 1] = c

    return mesh_vertices, faces, face_colors


def _render_with_plotly(mesh: ShadowboxMesh, options: RenderOptions | None = None):
    """Plotlyを使用してJupyter内でインタラクティブ3D表示。

    Args:
        mesh: レンダリングするシャドーボックスメッシュ。
        options: レンダリングオプション。

    Returns:
        Plotly Figureオブジェクト。
    """
    import plotly.graph_objects as go

    opts = options or RenderOptions()
    traces = []

    if opts.render_mode == "mesh":
        # メッシュモード: 各ポイントを四角形（2三角形）として描画
        for i, layer in enumerate(mesh.layers):
            if len(layer.vertices) == 0:
                continue

            # ポイントをメッシュデータに変換
            mesh_verts, mesh_faces, face_colors = _points_to_mesh_data(
                layer.vertices, layer.colors, opts.mesh_size
            )

            # 面の色をintensityとcolorscaleで表現
            # Plotlyでは面ごとの色をfacecolorで指定
            face_color_strs = [
                f"rgb({c[0]},{c[1]},{c[2]})" for c in face_colors
            ]

            trace = go.Mesh3d(
                x=mesh_verts[:, 0],
                y=mesh_verts[:, 1],
                z=mesh_verts[:, 2],
                i=mesh_faces[:, 0],
                j=mesh_faces[:, 1],
                k=mesh_faces[:, 2],
                facecolor=face_color_strs,
                opacity=opts.layer_opacity,
                name=f"Layer {i}",
                hoverinfo="skip",
                flatshading=True,
            )
            traces.append(trace)
    else:
        # ポイントモード（従来）: Scatter3dで描画
        for i, layer in enumerate(mesh.layers):
            if len(layer.vertices) == 0:
                continue

            # RGB色を文字列に変換
            colors = [
                f"rgb({c[0]},{c[1]},{c[2]})"
                for c in layer.colors
            ]

            trace = go.Scatter3d(
                x=layer.vertices[:, 0],
                y=layer.vertices[:, 1],
                z=layer.vertices[:, 2],
                mode="markers",
                marker={
                    "size": opts.point_size,
                    "color": colors,
                    "opacity": opts.layer_opacity,
                },
                name=f"Layer {i}",
                hoverinfo="skip",
            )
            traces.append(trace)

    # フレームをMesh3dとして追加
    if opts.show_frame and mesh.frame is not None:
        frame = mesh.frame
        frame_color = f"rgb({frame.color[0]},{frame.color[1]},{frame.color[2]})"
        frame_trace = go.Mesh3d(
            x=frame.vertices[:, 0],
            y=frame.vertices[:, 1],
            z=frame.vertices[:, 2],
            i=frame.faces[:, 0],
            j=frame.faces[:, 1],
            k=frame.faces[:, 2],
            color=frame_color,
            opacity=0.9,
            name="Frame",
            hoverinfo="skip",
        )
        traces.append(frame_trace)

    # 背景色を正規化
    bg = opts.background_color
    bg_str = f"rgb({bg[0]},{bg[1]},{bg[2]})"

    fig = go.Figure(data=traces)
    axis_config = {
        "showbackground": False,
        "showgrid": False,
        "zeroline": False,
        "visible": False,
    }
    fig.update_layout(
        title=opts.title,
        width=opts.window_size[0],
        height=opts.window_size[1],
        scene={
            "xaxis": axis_config,
            "yaxis": axis_config,
            "zaxis": axis_config,
            "bgcolor": bg_str,
            "aspectmode": "data",
        },
        paper_bgcolor=bg_str,
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
    )

    # カメラを正面やや斜めに設定
    fig.update_layout(
        scene_camera={
            "eye": {"x": 0, "y": 0, "z": 2},
            "up": {"x": 0, "y": 1, "z": 0},
        }
    )

    return fig


def render_shadowbox(
    mesh: ShadowboxMesh,
    options: RenderOptions | None = None,
    render_mode: Literal["points", "mesh"] | None = None,
):
    """シャドーボックスを簡単にレンダリングするユーティリティ関数。

    Args:
        mesh: レンダリングするシャドーボックスメッシュ。
        options: レンダリングオプション。Noneの場合はデフォルト。
        render_mode: 描画モード。"points"でポイント、"mesh"でメッシュ面。
            Noneの場合はoptionsの設定を使用。

    Returns:
        Jupyter環境: Plotly Figureオブジェクト（インタラクティブ3D）。
        それ以外: VedoのPlotterオブジェクト。

    Note:
        Jupyter環境では自動的にPlotlyを使用し、
        セル内でインタラクティブな3D表示が可能になります。

        render_mode="mesh" を使用すると、Blenderのような
        きれいなメッシュ描画が可能です（頂点数が多いと重くなります）。

    Example:
        >>> from shadowbox import create_pipeline
        >>> from shadowbox.visualization import render_shadowbox
        >>>
        >>> pipeline = create_pipeline(use_mock_depth=True)
        >>> result = pipeline.process(image)
        >>> render_shadowbox(result.mesh)  # ポイントモード（デフォルト）
        >>> render_shadowbox(result.mesh, render_mode="mesh")  # メッシュモード
    """
    # render_modeが指定された場合、optionsを更新
    if render_mode is not None:
        if options is None:
            options = RenderOptions(render_mode=render_mode)
        else:
            # 既存のoptionsをコピーしてrender_modeを上書き
            from dataclasses import replace
            options = replace(options, render_mode=render_mode)

    if _is_jupyter():
        fig = _render_with_plotly(mesh, options)
        return fig  # Jupyter will auto-display the returned figure

    renderer = ShadowboxRenderer(options)
    return renderer.render(mesh)


def _render_layers_exploded_plotly(
    mesh: ShadowboxMesh,
    gap_multiplier: float = 2.5,
    options: RenderOptions | None = None,
):
    """Plotlyでレイヤー分離表示。

    Args:
        mesh: レンダリングするシャドーボックスメッシュ。
        gap_multiplier: レイヤー間隔の倍率。
        options: レンダリングオプション。

    Returns:
        Plotly Figureオブジェクト。
    """
    import plotly.graph_objects as go

    opts = options or RenderOptions()
    traces = []

    for i, layer in enumerate(mesh.layers):
        if len(layer.vertices) == 0:
            continue

        # Z座標を調整（より離す）
        vertices = layer.vertices.copy()
        original_z = layer.z_position
        vertices[:, 2] = original_z * gap_multiplier

        colors = [f"rgb({c[0]},{c[1]},{c[2]})" for c in layer.colors]

        trace = go.Scatter3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            mode="markers",
            marker={"size": opts.point_size, "color": colors, "opacity": opts.layer_opacity},
            name=f"Layer {i}",
            hoverinfo="skip",
        )
        traces.append(trace)

    bg = opts.background_color
    bg_str = f"rgb({bg[0]},{bg[1]},{bg[2]})"

    axis_cfg = {
        "showbackground": False,
        "showgrid": False,
        "zeroline": False,
        "visible": False,
    }
    fig = go.Figure(data=traces)
    fig.update_layout(
        title=f"{opts.title} (レイヤー分離表示)",
        width=opts.window_size[0],
        height=opts.window_size[1],
        scene={
            "xaxis": axis_cfg,
            "yaxis": axis_cfg,
            "zaxis": axis_cfg,
            "bgcolor": bg_str,
            "aspectmode": "data",
        },
        paper_bgcolor=bg_str,
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
    )

    return fig


def render_layers_exploded(
    mesh: ShadowboxMesh,
    gap_multiplier: float = 2.5,
    options: RenderOptions | None = None,
):
    """レイヤーを分解して表示するユーティリティ関数。

    Args:
        mesh: レンダリングするシャドーボックスメッシュ。
        gap_multiplier: レイヤー間隔の倍率。
        options: レンダリングオプション。Noneの場合はデフォルト。

    Returns:
        Jupyter環境: Plotly Figureオブジェクト（インタラクティブ3D）。
        それ以外: VedoのPlotterオブジェクト。

    Note:
        Jupyter環境では自動的にPlotlyを使用し、
        セル内でインタラクティブな3D表示が可能になります。

    Example:
        >>> render_layers_exploded(mesh, gap_multiplier=3.0)
    """
    if _is_jupyter():
        fig = _render_layers_exploded_plotly(mesh, gap_multiplier, options)
        return fig  # Jupyter will auto-display the returned figure

    renderer = ShadowboxRenderer(options)
    return renderer.render_layers_separately(mesh, gap_multiplier)
