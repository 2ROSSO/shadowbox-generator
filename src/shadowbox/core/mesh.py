"""3Dメッシュ生成モジュール。

このモジュールは、クラスタリングされた深度レイヤーから
3Dメッシュデータを生成する機能を提供します。
生成されたメッシュはVedoなどの3Dレンダラーで表示できます。
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from numpy.typing import NDArray

from shadowbox.config.settings import RenderSettings


@dataclass
class LayerMesh:
    """単一レイヤーの3Dメッシュデータ。

    Attributes:
        vertices: 頂点座標。shape (N, 3) で各行が(x, y, z)。
        colors: 各頂点の色。shape (N, 3) でRGB値(0-255)。
        z_position: レイヤーのZ座標位置。
        layer_index: レイヤーインデックス（0が最前面）。
        pixel_indices: 元画像でのピクセルインデックス。shape (N, 2)で(row, col)。

    Note:
        座標系は以下のように定義:
        - X: 左(-1)から右(+1)
        - Y: 下(-1)から上(+1)
        - Z: 奥(-n)から手前(0)
    """

    vertices: NDArray[np.float32]
    colors: NDArray[np.uint8]
    z_position: float
    layer_index: int
    pixel_indices: NDArray[np.int32]


@dataclass
class FrameMesh:
    """フレーム（枠）の3Dメッシュデータ。

    シャドーボックスの枠部分を表現します。
    壁モードが有効な場合、前面から背面まで繋がる3D壁を持ちます。

    Attributes:
        vertices: 頂点座標。shape (N, 3)。壁なしは8頂点、壁ありは12頂点。
        faces: 面のインデックス。shape (M, 3)。
        color: フレームの色。RGB値(0-255)。
        z_position: フレームのZ座標（最前面）。
        z_back: フレーム背面のZ座標（壁がある場合）。
        has_walls: 壁（3D厚み）があるかどうか。
    """

    vertices: NDArray[np.float32]
    faces: NDArray[np.int32]
    color: NDArray[np.uint8]
    z_position: float
    z_back: Optional[float] = None
    has_walls: bool = False


@dataclass
class ShadowboxMesh:
    """シャドーボックス全体の3Dメッシュデータ。

    全レイヤーとオプションのフレームを含みます。

    Attributes:
        layers: レイヤーメッシュのリスト。
        frame: フレームメッシュ（Noneの場合はフレームなし）。
        bounds: バウンディングボックス (min_x, max_x, min_y, max_y, min_z, max_z)。
        num_layers: レイヤー数。

    Example:
        >>> mesh = generator.generate(image, labels, centroids)
        >>> for layer in mesh.layers:
        ...     print(f"Layer {layer.layer_index}: {len(layer.vertices)} vertices")
    """

    layers: List[LayerMesh]
    frame: Optional[FrameMesh]
    bounds: tuple

    @property
    def num_layers(self) -> int:
        """レイヤー数を返す。"""
        return len(self.layers)

    @property
    def total_vertices(self) -> int:
        """全レイヤーの頂点数合計を返す。"""
        return sum(len(layer.vertices) for layer in self.layers)


class MeshGenerator:
    """深度レイヤーから3Dメッシュを生成するジェネレーター。

    クラスタリングされた深度マップと元画像から、
    各レイヤーの3Dポイントクラウドデータを生成します。

    Attributes:
        settings: レンダリング設定。

    Example:
        >>> settings = RenderSettings()
        >>> generator = MeshGenerator(settings)
        >>> mesh = generator.generate(image, labels, centroids)
    """

    def __init__(self, settings: RenderSettings) -> None:
        """ジェネレーターを初期化。

        Args:
            settings: レンダリング設定。
        """
        self._settings = settings

    def generate(
        self,
        image: NDArray[np.uint8],
        labels: NDArray[np.int32],
        centroids: NDArray[np.float32],
        include_frame: bool = True,
    ) -> ShadowboxMesh:
        """クラスタリング結果から完全なシャドーボックスメッシュを生成。

        Args:
            image: 元のRGB画像。shape (H, W, 3)。
            labels: 各ピクセルのレイヤーインデックス。shape (H, W)。
            centroids: ソート済みセントロイド。shape (k,)。
            include_frame: フレームを含めるかどうか。

        Returns:
            ShadowboxMeshオブジェクト。
        """
        layers = []
        num_layers = len(centroids)

        for i in range(num_layers):
            # Z位置: レイヤー0は-layer_thickness、レイヤーnは-(n+1)*layer_thickness
            # フレームがz=0なので、イラストはその奥
            z = -(i + 1) * (self._settings.layer_thickness + self._settings.layer_gap)

            layer_mesh = self._create_layer_mesh(
                image, labels, z, i,
                cumulative=self._settings.cumulative_layers,
            )
            layers.append(layer_mesh)

        # フレームの生成
        frame = None
        if include_frame:
            if self._settings.frame_wall_mode == "outer":
                frame = self._create_frame_mesh_with_walls(image.shape[:2], num_layers)
            else:
                frame = self._create_frame_mesh(image.shape[:2])

        # バウンディングボックスの計算
        bounds = self._calculate_bounds(layers, frame)

        return ShadowboxMesh(
            layers=layers,
            frame=frame,
            bounds=bounds,
        )

    def generate_raw_depth(
        self,
        image: NDArray[np.uint8],
        depth_map: NDArray[np.float32],
        include_frame: bool = True,
        depth_scale: float = 1.0,
    ) -> ShadowboxMesh:
        """生の深度マップから連続的な3Dメッシュを生成。

        クラスタリングを行わず、各ピクセルの深度値を直接Z座標として使用。
        より滑らかな深度表現が可能です。

        Args:
            image: 元のRGB画像。shape (H, W, 3)。
            depth_map: 正規化された深度マップ (0.0-1.0)。shape (H, W)。
            include_frame: フレームを含めるかどうか。
            depth_scale: 深度のスケール係数（大きいほど立体感が増す）。

        Returns:
            ShadowboxMeshオブジェクト（1レイヤーのみ）。
        """
        h, w = image.shape[:2]

        # 全ピクセルの座標を取得
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        y_coords = y_coords.flatten()
        x_coords = x_coords.flatten()

        # 座標を[-1, 1]の範囲に正規化
        vertices_x = (x_coords / (w - 1)) * 2 - 1 if w > 1 else np.zeros_like(x_coords)
        vertices_y = -((y_coords / (h - 1)) * 2 - 1) if h > 1 else np.zeros_like(y_coords)

        # Z座標: depth=0（近い）が手前、depth=1（遠い）が奥
        # フレームがz=0なので、イラストはその奥（負の値）
        z_base = -self._settings.layer_thickness
        z_range = -depth_scale * self._settings.layer_thickness * 5  # 深度の範囲
        vertices_z = z_base + depth_map.flatten() * z_range

        vertices = np.stack(
            [vertices_x, vertices_y, vertices_z], axis=1
        ).astype(np.float32)

        # 各頂点の色を取得
        colors = image.reshape(-1, 3).astype(np.uint8)

        # ピクセルインデックスを保存
        pixel_indices = np.stack([y_coords, x_coords], axis=1).astype(np.int32)

        # 単一レイヤーとして作成
        layer = LayerMesh(
            vertices=vertices,
            colors=colors,
            z_position=float(vertices_z.mean()),  # 平均Z位置
            layer_index=0,
            pixel_indices=pixel_indices,
        )

        # フレームの生成（深度範囲に合わせる）
        frame = None
        if include_frame:
            z_min = float(vertices_z.min())
            if self._settings.frame_wall_mode == "outer":
                frame = self._create_frame_mesh_with_walls_custom_depth(
                    image.shape[:2], z_min
                )
            else:
                frame = self._create_frame_mesh(image.shape[:2])

        # バウンディングボックスの計算
        bounds = self._calculate_bounds([layer], frame)

        return ShadowboxMesh(
            layers=[layer],
            frame=frame,
            bounds=bounds,
        )

    def _create_frame_mesh_with_walls_custom_depth(
        self,
        image_shape: tuple,
        z_back: float,
    ) -> FrameMesh:
        """カスタム深度で壁付きフレームを作成。

        Args:
            image_shape: 画像の形状 (H, W)。
            z_back: 背面のZ座標。

        Returns:
            FrameMeshオブジェクト（壁付き）。
        """
        margin = 0.05
        outer = 1.0 + margin
        inner = 1.0
        z_front = self._settings.frame_z

        vertices = np.array([
            [-outer, -outer, z_front],
            [+outer, -outer, z_front],
            [+outer, +outer, z_front],
            [-outer, +outer, z_front],
            [-inner, -inner, z_front],
            [+inner, -inner, z_front],
            [+inner, +inner, z_front],
            [-inner, +inner, z_front],
            [-outer, -outer, z_back],
            [+outer, -outer, z_back],
            [+outer, +outer, z_back],
            [-outer, +outer, z_back],
        ], dtype=np.float32)

        faces = np.array([
            [0, 1, 5], [0, 5, 4],
            [1, 2, 6], [1, 6, 5],
            [2, 3, 7], [2, 7, 6],
            [3, 0, 4], [3, 4, 7],
            [0, 8, 9], [0, 9, 1],
            [1, 9, 10], [1, 10, 2],
            [2, 10, 11], [2, 11, 3],
            [3, 11, 8], [3, 8, 0],
        ], dtype=np.int32)

        color = np.array([30, 30, 30], dtype=np.uint8)

        return FrameMesh(
            vertices=vertices,
            faces=faces,
            color=color,
            z_position=z_front,
            z_back=z_back,
            has_walls=True,
        )

    def _create_layer_mesh(
        self,
        image: NDArray[np.uint8],
        labels: NDArray[np.int32],
        z: float,
        layer_index: int,
        cumulative: bool = True,
    ) -> LayerMesh:
        """単一レイヤーのメッシュを作成。

        マスクされたピクセルのみを含むポイントクラウドを生成します。

        Args:
            image: 元のRGB画像。shape (H, W, 3)。
            labels: 各ピクセルのレイヤーインデックス。shape (H, W)。
            z: このレイヤーのZ座標。
            layer_index: レイヤーインデックス。
            cumulative: 累積レイヤーモード。Trueの場合、このレイヤー以下の
                すべてのピクセルを含む（奥のレイヤーほど完全な画像に近づく）。

        Returns:
            LayerMeshオブジェクト。
        """
        if cumulative:
            # 累積マスク: このレイヤー以下のすべてのピクセル
            # ただし、labels == -1（カードフレーム）は特別処理
            if layer_index == 0:
                # レイヤー0のみ: フレームピクセル(-1)を含む
                mask = (labels <= layer_index) | (labels == -1)
            else:
                # レイヤー1以降: フレームピクセルは除外
                mask = (labels <= layer_index) & (labels >= 0)
        else:
            # 従来: このレイヤーのピクセルのみ
            mask = labels == layer_index

        h, w = mask.shape

        # マスクされたピクセルの座標を取得
        y_coords, x_coords = np.where(mask)

        if len(y_coords) == 0:
            # 空のレイヤー
            return LayerMesh(
                vertices=np.array([], dtype=np.float32).reshape(0, 3),
                colors=np.array([], dtype=np.uint8).reshape(0, 3),
                z_position=z,
                layer_index=layer_index,
                pixel_indices=np.array([], dtype=np.int32).reshape(0, 2),
            )

        # 座標を[-1, 1]の範囲に正規化
        # X: 左端が-1、右端が+1
        vertices_x = (x_coords / (w - 1)) * 2 - 1 if w > 1 else np.zeros_like(x_coords)
        # Y: 上端が+1、下端が-1（画像座標系からOpenGL座標系への変換）
        vertices_y = -((y_coords / (h - 1)) * 2 - 1) if h > 1 else np.zeros_like(y_coords)
        # Z: 指定されたレイヤー位置
        vertices_z = np.full_like(vertices_x, z)

        vertices = np.stack([vertices_x, vertices_y, vertices_z], axis=1).astype(np.float32)

        # 各頂点の色を取得
        colors = image[y_coords, x_coords].astype(np.uint8)

        # ピクセルインデックスを保存
        pixel_indices = np.stack([y_coords, x_coords], axis=1).astype(np.int32)

        return LayerMesh(
            vertices=vertices,
            colors=colors,
            z_position=z,
            layer_index=layer_index,
            pixel_indices=pixel_indices,
        )

    def _create_frame_mesh(self, image_shape: tuple) -> FrameMesh:
        """フレーム（枠）メッシュを作成。

        イラストを囲む矩形フレームを生成します。

        Args:
            image_shape: 画像の形状 (H, W)。

        Returns:
            FrameMeshオブジェクト。
        """
        # フレームのサイズ（イラストより少し大きく）
        margin = 0.05
        outer = 1.0 + margin
        inner = 1.0

        # フレームの頂点（外側と内側の4隅ずつ）
        # 外側
        vertices = np.array([
            [-outer, -outer, self._settings.frame_z],  # 0: 外側左下
            [outer, -outer, self._settings.frame_z],   # 1: 外側右下
            [outer, outer, self._settings.frame_z],    # 2: 外側右上
            [-outer, outer, self._settings.frame_z],   # 3: 外側左上
            # 内側
            [-inner, -inner, self._settings.frame_z],  # 4: 内側左下
            [inner, -inner, self._settings.frame_z],   # 5: 内側右下
            [inner, inner, self._settings.frame_z],    # 6: 内側右上
            [-inner, inner, self._settings.frame_z],   # 7: 内側左上
        ], dtype=np.float32)

        # フレームの面（三角形）
        # 下辺
        faces = np.array([
            [0, 1, 5], [0, 5, 4],  # 下辺
            [1, 2, 6], [1, 6, 5],  # 右辺
            [2, 3, 7], [2, 7, 6],  # 上辺
            [3, 0, 4], [3, 4, 7],  # 左辺
        ], dtype=np.int32)

        # フレームの色（暗い色）
        color = np.array([30, 30, 30], dtype=np.uint8)

        return FrameMesh(
            vertices=vertices,
            faces=faces,
            color=color,
            z_position=self._settings.frame_z,
        )

    def _create_frame_mesh_with_walls(
        self,
        image_shape: tuple,
        num_layers: int,
    ) -> FrameMesh:
        """壁付きフレーム（枠）メッシュを作成。

        本物のシャドーボックスのように、前面から背面まで繋がる
        3D壁を持つフレームを生成します。

        Args:
            image_shape: 画像の形状 (H, W)。
            num_layers: レイヤー数（背面Z位置の計算用）。

        Returns:
            FrameMeshオブジェクト（壁付き）。
        """
        margin = 0.05
        outer = 1.0 + margin
        inner = 1.0
        z_front = self._settings.frame_z
        z_back = -num_layers * (self._settings.layer_thickness + self._settings.layer_gap)

        # 12頂点: 前面外側4 + 前面内側4 + 背面外側4
        vertices = np.array([
            # 前面外側 (0-3)
            [-outer, -outer, z_front],
            [+outer, -outer, z_front],
            [+outer, +outer, z_front],
            [-outer, +outer, z_front],
            # 前面内側 (4-7)
            [-inner, -inner, z_front],
            [+inner, -inner, z_front],
            [+inner, +inner, z_front],
            [-inner, +inner, z_front],
            # 背面外側 (8-11)
            [-outer, -outer, z_back],
            [+outer, -outer, z_back],
            [+outer, +outer, z_back],
            [-outer, +outer, z_back],
        ], dtype=np.float32)

        # 面の構成: 前面枠8三角形 + 外壁8三角形 = 16三角形
        faces = np.array([
            # 前面枠
            [0, 1, 5], [0, 5, 4],  # 下辺
            [1, 2, 6], [1, 6, 5],  # 右辺
            [2, 3, 7], [2, 7, 6],  # 上辺
            [3, 0, 4], [3, 4, 7],  # 左辺
            # 外壁（下/右/上/左）
            [0, 8, 9], [0, 9, 1],   # 下壁
            [1, 9, 10], [1, 10, 2],  # 右壁
            [2, 10, 11], [2, 11, 3],  # 上壁
            [3, 11, 8], [3, 8, 0],   # 左壁
        ], dtype=np.int32)

        # フレームの色（暗い色）
        color = np.array([30, 30, 30], dtype=np.uint8)

        return FrameMesh(
            vertices=vertices,
            faces=faces,
            color=color,
            z_position=z_front,
            z_back=z_back,
            has_walls=True,
        )

    def _calculate_bounds(
        self,
        layers: List[LayerMesh],
        frame: Optional[FrameMesh],
    ) -> tuple:
        """全メッシュのバウンディングボックスを計算。

        Args:
            layers: レイヤーメッシュのリスト。
            frame: フレームメッシュ（None可）。

        Returns:
            (min_x, max_x, min_y, max_y, min_z, max_z)のタプル。
        """
        all_vertices = []

        for layer in layers:
            if len(layer.vertices) > 0:
                all_vertices.append(layer.vertices)

        if frame is not None:
            all_vertices.append(frame.vertices)

        if not all_vertices:
            # 頂点がない場合はデフォルトのバウンズ
            return (-1.0, 1.0, -1.0, 1.0, -1.0, 0.0)

        combined = np.vstack(all_vertices)

        return (
            float(combined[:, 0].min()),
            float(combined[:, 0].max()),
            float(combined[:, 1].min()),
            float(combined[:, 1].max()),
            float(combined[:, 2].min()),
            float(combined[:, 2].max()),
        )
