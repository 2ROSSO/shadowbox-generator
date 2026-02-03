"""3Dメッシュ生成モジュール。

このモジュールは、クラスタリングされた深度レイヤーから
3Dメッシュデータを生成する機能を提供します。
生成されたメッシュはVedoなどの3Dレンダラーで表示できます。
"""

from __future__ import annotations

from dataclasses import dataclass

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
        faces: 三角形面のインデックス。shape (M, 3)。
            Noneの場合、エクスポート時に各頂点が小さな四角形に変換される。
            TripoSRなど直接メッシュを生成する場合に使用。

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
    faces: NDArray[np.int32] | None = None


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
    z_back: float | None = None
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

    layers: list[LayerMesh]
    frame: FrameMesh | None
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
        interp_count = self._settings.layer_interpolation
        pop_out = self._settings.layer_pop_out

        # 額縁の厚み（固定値）
        frame_depth = self._settings.frame_depth
        back_z = -frame_depth

        # レイヤー間隔を自動計算（額縁内に均等配置）
        layer_spacing = frame_depth / num_layers

        # 各レイヤーのZ位置を事前計算
        layer_z_positions = []
        for i in range(num_layers):
            z = -layer_spacing * (i + 1)
            layer_z_positions.append(z)

        # 飛び出しオフセットを計算（フレーム厚みに対する比率）
        pop_out_offset = frame_depth * pop_out if pop_out > 0 else 0

        # フレームの存在確認（labels == -1 があるか）
        has_card_frame = np.any(labels == -1)

        # フレームレイヤー（最前面、pop-outなし）
        if has_card_frame:
            frame_z = layer_z_positions[0]  # pop-outなし
            frame_layer = self._create_frame_only_layer(image, labels, frame_z, -1)
            if len(frame_layer.vertices) > 0:
                layers.append(frame_layer)

        # フレーム補間（labels == -1 のピクセルのみ、最前面から最背面まで）
        # フレームは飛び出しなし、レイヤー数×補間数で等間隔に埋める
        if interp_count > 0 and has_card_frame:
            frame_z_start = layer_z_positions[0]  # レイヤー0と同じ位置から開始
            frame_interp_count = num_layers * interp_count  # レイヤー数×補間数
            for j in range(1, frame_interp_count + 1):
                t = j / (frame_interp_count + 1)
                interp_z = frame_z_start + (back_z - frame_z_start) * t
                frame_layer = self._create_frame_only_layer(image, labels, interp_z, -1)
                if len(frame_layer.vertices) > 0:
                    layers.append(frame_layer)

        # 各レイヤーを生成（イラストレイヤーは飛び出しオフセット適用、フレーム除外）
        for i in range(num_layers):
            z = layer_z_positions[i] + pop_out_offset  # 飛び出し

            layer_mesh = self._create_layer_mesh(
                image, labels, z, i,
                cumulative=self._settings.cumulative_layers,
            )
            layers.append(layer_mesh)

            # レイヤー補間（すべてのレイヤーで次のレイヤーまで補間）
            if interp_count > 0:
                z_start = z
                if i + 1 < num_layers:
                    next_z = layer_z_positions[i + 1] + pop_out_offset
                else:
                    next_z = back_z + pop_out_offset
                z_end = next_z

                # N個の補間レイヤーを追加
                for j in range(1, interp_count + 1):
                    t = j / (interp_count + 1)
                    interp_z = z_start + (z_end - z_start) * t
                    interp_layer = self._create_layer_mesh(
                        image, labels, interp_z, i,
                        cumulative=self._settings.cumulative_layers,
                    )
                    layers.append(interp_layer)

        # 最背面パネル（カード全体画像）を追加
        # フレームと同じ位置（飛び出しなし）
        if self._settings.back_panel:
            back_panel_z = back_z  # pop-outなし
            back_panel = self._create_back_panel(image, back_panel_z, num_layers)
            layers.append(back_panel)

        frame_num_layers = num_layers

        # フレームの生成
        frame = None
        if include_frame:
            if self._settings.frame_wall_mode == "outer":
                frame = self._create_frame_mesh_with_walls(image.shape[:2], frame_num_layers)
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

        layers = [layer]

        # 最背面パネル（カード全体画像）を追加
        # 最深ポイントと同じ深さに配置
        z_min = float(vertices_z.min())
        if self._settings.back_panel:
            back_panel = self._create_back_panel(image, z_min, layer_index=1)
            layers.append(back_panel)

        # フレームの生成（深度範囲に合わせる）
        frame = None
        if include_frame:
            if self._settings.frame_wall_mode == "outer":
                frame = self._create_frame_mesh_with_walls_custom_depth(
                    image.shape[:2], z_min
                )
            else:
                frame = self._create_frame_mesh(image.shape[:2])

        # バウンディングボックスの計算
        bounds = self._calculate_bounds(layers, frame)

        return ShadowboxMesh(
            layers=layers,
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
        from shadowbox.core.frame_factory import FrameConfig, create_frame

        config = FrameConfig(
            z_front=self._settings.frame_z,
            z_back=z_back,
        )
        return create_frame(config)

    def _create_back_panel(
        self,
        image: NDArray[np.uint8],
        z: float,
        layer_index: int,
    ) -> LayerMesh:
        """最背面のパネル（カード全体画像）を作成。

        全ピクセルを含むレイヤーを生成します。

        Args:
            image: 元のRGB画像。shape (H, W, 3)。
            z: このレイヤーのZ座標。
            layer_index: レイヤーインデックス。

        Returns:
            LayerMeshオブジェクト（全ピクセル）。
        """
        from shadowbox.core.back_panel_factory import create_back_panel

        return create_back_panel(image, z, layer_index)

    def _create_frame_only_layer(
        self,
        image: NDArray[np.uint8],
        labels: NDArray[np.int32],
        z: float,
        layer_index: int,
    ) -> LayerMesh:
        """フレームピクセルのみのレイヤーを作成。

        カードフレーム（labels == -1）のピクセルのみを含むレイヤーを生成します。
        フレームの壁効果を作るための補間用。

        Args:
            image: 元のRGB画像。shape (H, W, 3)。
            labels: 各ピクセルのレイヤーインデックス。shape (H, W)。
            z: このレイヤーのZ座標。
            layer_index: レイヤーインデックス。

        Returns:
            LayerMeshオブジェクト（フレームピクセルのみ）。
        """
        # フレームピクセルのみ（labels == -1）
        mask = labels == -1

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
        vertices_x = (x_coords / (w - 1)) * 2 - 1 if w > 1 else np.zeros_like(x_coords)
        vertices_y = -((y_coords / (h - 1)) * 2 - 1) if h > 1 else np.zeros_like(y_coords)
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
        フレームピクセル（labels == -1）は常に除外されます。

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
            # フレーム(-1)は常に除外（別途フレームレイヤーとして作成）
            mask = (labels <= layer_index) & (labels >= 0)
        else:
            # 穴あきモード: このレイヤーのピクセルのみ
            # フレーム(-1)は除外
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
        from shadowbox.core.frame_factory import FrameConfig, create_frame

        config = FrameConfig(z_front=self._settings.frame_z)
        return create_frame(config)

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
            num_layers: レイヤー数（未使用、互換性のため残す）。

        Returns:
            FrameMeshオブジェクト（壁付き）。
        """
        from shadowbox.core.frame_factory import FrameConfig, create_frame

        config = FrameConfig(
            z_front=self._settings.frame_z,
            z_back=-self._settings.frame_depth,
        )
        return create_frame(config)

    def _calculate_bounds(
        self,
        layers: list[LayerMesh],
        frame: FrameMesh | None,
    ) -> tuple:
        """全メッシュのバウンディングボックスを計算。

        Args:
            layers: レイヤーメッシュのリスト。
            frame: フレームメッシュ（None可）。

        Returns:
            (min_x, max_x, min_y, max_y, min_z, max_z)のタプル。
        """
        from shadowbox.core.frame_factory import calculate_bounds

        return calculate_bounds(layers, frame)
