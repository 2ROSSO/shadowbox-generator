"""3Dレンダリングモジュール。

このモジュールは、Vedoを使用してシャドーボックスメッシュを
インタラクティブに3D表示する機能を提供します。
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from shadowbox.core.mesh import ShadowboxMesh


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
        point_size: ポイントのサイズ。
        show_axes: 軸を表示するかどうか。
        show_frame: フレームを表示するかどうか。
        window_size: ウィンドウサイズ (幅, 高さ)。
        title: ウィンドウタイトル。
        interactive: インタラクティブモードを有効にするか。
        layer_opacity: レイヤーの不透明度 (0.0-1.0)。
    """

    background_color: Tuple[int, int, int] = (30, 30, 30)
    point_size: float = 3.0
    show_axes: bool = False
    show_frame: bool = True
    window_size: Tuple[int, int] = (1200, 800)
    title: str = "Shadowbox 3D Viewer"
    interactive: bool = True
    layer_opacity: float = 1.0


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

    def __init__(self, options: Optional[RenderOptions] = None) -> None:
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
    ) -> "vedo.Plotter":
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
    ) -> "vedo.Plotter":
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
        size: Optional[Tuple[int, int]] = None,
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

    def _create_frame_mesh(self, frame: "FrameMesh") -> Optional["vedo.Mesh"]:
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
        color: Tuple[int, int, int],
    ) -> Tuple[float, float, float]:
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


def _render_with_plotly(mesh: ShadowboxMesh, options: Optional[RenderOptions] = None):
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

    # 各レイヤーをScatter3dとして追加
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
            marker=dict(
                size=opts.point_size,
                color=colors,
                opacity=opts.layer_opacity,
            ),
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
    fig.update_layout(
        title=opts.title,
        width=opts.window_size[0],
        height=opts.window_size[1],
        scene=dict(
            xaxis=dict(showbackground=False, showgrid=False, zeroline=False, visible=False),
            yaxis=dict(showbackground=False, showgrid=False, zeroline=False, visible=False),
            zaxis=dict(showbackground=False, showgrid=False, zeroline=False, visible=False),
            bgcolor=bg_str,
            aspectmode="data",
        ),
        paper_bgcolor=bg_str,
        margin=dict(l=0, r=0, t=40, b=0),
    )

    # カメラを正面やや斜めに設定
    fig.update_layout(
        scene_camera=dict(
            eye=dict(x=0, y=0, z=2),
            up=dict(x=0, y=1, z=0),
        )
    )

    return fig


def render_shadowbox(
    mesh: ShadowboxMesh,
    options: Optional[RenderOptions] = None,
):
    """シャドーボックスを簡単にレンダリングするユーティリティ関数。

    Args:
        mesh: レンダリングするシャドーボックスメッシュ。
        options: レンダリングオプション。Noneの場合はデフォルト。

    Returns:
        Jupyter環境: Plotly Figureオブジェクト（インタラクティブ3D）。
        それ以外: VedoのPlotterオブジェクト。

    Note:
        Jupyter環境では自動的にPlotlyを使用し、
        セル内でインタラクティブな3D表示が可能になります。

    Example:
        >>> from shadowbox import create_pipeline
        >>> from shadowbox.visualization import render_shadowbox
        >>>
        >>> pipeline = create_pipeline(use_mock_depth=True)
        >>> result = pipeline.process(image)
        >>> render_shadowbox(result.mesh)
    """
    if _is_jupyter():
        fig = _render_with_plotly(mesh, options)
        return fig  # Jupyter will auto-display the returned figure

    renderer = ShadowboxRenderer(options)
    return renderer.render(mesh)


def _render_layers_exploded_plotly(
    mesh: ShadowboxMesh,
    gap_multiplier: float = 2.5,
    options: Optional[RenderOptions] = None,
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
            marker=dict(size=opts.point_size, color=colors, opacity=opts.layer_opacity),
            name=f"Layer {i}",
            hoverinfo="skip",
        )
        traces.append(trace)

    bg = opts.background_color
    bg_str = f"rgb({bg[0]},{bg[1]},{bg[2]})"

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=f"{opts.title} (レイヤー分離表示)",
        width=opts.window_size[0],
        height=opts.window_size[1],
        scene=dict(
            xaxis=dict(showbackground=False, showgrid=False, zeroline=False, visible=False),
            yaxis=dict(showbackground=False, showgrid=False, zeroline=False, visible=False),
            zaxis=dict(showbackground=False, showgrid=False, zeroline=False, visible=False),
            bgcolor=bg_str,
            aspectmode="data",
        ),
        paper_bgcolor=bg_str,
        margin=dict(l=0, r=0, t=40, b=0),
    )

    return fig


def render_layers_exploded(
    mesh: ShadowboxMesh,
    gap_multiplier: float = 2.5,
    options: Optional[RenderOptions] = None,
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
