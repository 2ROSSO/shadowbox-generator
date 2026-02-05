"""TripoSR 3Dメッシュ生成器。

Stability AIのTripoSRモデルを使用して、
単一画像から3Dメッシュを生成します。
"""

from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

from shadowbox.core.mesh import LayerMesh, ShadowboxMesh
from shadowbox.triposr.settings import TripoSRSettings

if TYPE_CHECKING:
    from tsr.system import TSR


def _find_triposr_directory() -> str | None:
    """TripoSRディレクトリを自動検出。

    以下の場所を順番に探索:
    1. カレントディレクトリのTripoSR/
    2. プロジェクトルート（pyproject.tomlがある場所）のTripoSR/
    3. このファイルの親ディレクトリからの相対パス

    Returns:
        TripoSRディレクトリのパス、または見つからない場合はNone。
    """
    import sys
    from pathlib import Path

    # 既にsys.pathにある場合はスキップ
    try:
        import tsr  # noqa: F401
        return None  # 既にインポート可能
    except ImportError:
        pass

    candidates = []

    # 1. カレントディレクトリ
    candidates.append(Path.cwd() / "TripoSR")

    # 2. プロジェクトルート（pyproject.tomlを探す）
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            candidates.append(parent / "TripoSR")
            break

    # 3. このファイルから相対的に探索
    # src/shadowbox/triposr/generator.py → プロジェクトルート
    src_root = Path(__file__).resolve().parent.parent.parent.parent
    candidates.append(src_root / "TripoSR")

    # 候補をチェック
    for candidate in candidates:
        if candidate.exists() and (candidate / "tsr").is_dir():
            triposr_path = str(candidate)
            if triposr_path not in sys.path:
                sys.path.insert(0, triposr_path)
            return triposr_path

    return None


class TripoSRGenerator:
    """TripoSRによる3Dメッシュ生成器。

    単一画像から直接3Dメッシュを生成します。
    深度推定+クラスタリングのパイプラインとは異なり、
    ニューラルネットワークが直接3D形状を予測します。

    Note:
        このクラスを使用するには、TripoSR依存関係が必要です:
        pip install shadowbox[triposr]

    Example:
        >>> from shadowbox.triposr import TripoSRSettings, create_triposr_generator
        >>> settings = TripoSRSettings()
        >>> generator = create_triposr_generator(settings)
        >>> mesh = generator.generate(image)
    """

    def __init__(self, settings: TripoSRSettings) -> None:
        """TripoSR生成器を初期化。

        Args:
            settings: TripoSR設定。
        """
        self._settings = settings
        self._model: TSR | None = None
        self._device: str | None = None

    def _ensure_model(self) -> None:
        """モデルが初期化されていることを確認（遅延初期化）。"""
        if self._model is not None:
            return

        # TripoSRディレクトリをPYTHONPATHに追加（必要な場合）
        _find_triposr_directory()

        try:
            from tsr.system import TSR
        except ImportError as e:
            raise ImportError(
                "TripoSRを使用するには、以下の手順でインストールしてください:\n"
                "  1. git clone https://github.com/VAST-AI-Research/TripoSR.git\n"
                "  2. pip install -r TripoSR/requirements.txt\n"
                "  3. TripoSRディレクトリをPYTHONPATHに追加するか、\n"
                "     TripoSRディレクトリから実行してください"
            ) from e

        # デバイスを決定
        device = self._settings.device
        if device == "auto":
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device

        print(f"TripoSRモデルを読み込み中: {self._settings.model_id}")
        print(f"デバイス: {device}")

        self._model = TSR.from_pretrained(
            self._settings.model_id,
            config_name="config.yaml",
            weight_name="model.ckpt",
        )
        self._model.renderer.set_chunk_size(self._settings.chunk_size)
        self._model.to(device)

        print("TripoSRモデル読み込み完了")

    def generate(self, image: Image.Image) -> ShadowboxMesh:
        """画像から3Dメッシュを生成。

        Args:
            image: 入力画像。

        Returns:
            ShadowboxMesh: 生成された3Dメッシュ。
                単一レイヤーとして格納される。
        """
        self._ensure_model()
        assert self._model is not None
        assert self._device is not None

        # 背景除去（オプション）
        if self._settings.remove_background:
            import importlib.util

            if importlib.util.find_spec("rembg") is not None:
                image = self._remove_background(image)
            else:
                print("警告: rembgがインストールされていないため、背景除去をスキップします")

        # TripoSRで3Dメッシュ生成
        print("3Dメッシュを生成中...")
        with self._inference_context():
            scene_codes = self._model([image], device=self._device)
            meshes = self._model.extract_mesh(
                scene_codes,
                has_vertex_color=True,
                resolution=self._settings.mc_resolution,
            )

        # 最初のメッシュを取得
        mesh = meshes[0]
        print(f"メッシュ生成完了: 頂点数={len(mesh.vertices)}, 面数={len(mesh.faces)}")

        # ShadowboxMeshに変換
        return self._convert_to_shadowbox_mesh(mesh, image)

    def _remove_background(self, image: Image.Image) -> Image.Image:
        """背景を除去して前景をクロップ。"""
        import rembg

        # 背景除去
        image_rgba = rembg.remove(image)

        # アルファチャンネルで前景領域を検出
        alpha = np.array(image_rgba)[:, :, 3]
        y_indices, x_indices = np.where(alpha > 0)

        if len(y_indices) == 0:
            return image

        # バウンディングボックス
        y_min, y_max = y_indices.min(), y_indices.max()
        x_min, x_max = x_indices.min(), x_indices.max()

        # 前景領域をクロップ
        fg_width = x_max - x_min
        fg_height = y_max - y_min
        fg_size = max(fg_width, fg_height)

        # 余白を追加してクロップ
        target_size = int(fg_size / self._settings.foreground_ratio)
        pad_x = (target_size - fg_width) // 2
        pad_y = (target_size - fg_height) // 2

        # 新しい画像を作成（白背景）
        new_image = Image.new("RGB", (target_size, target_size), (255, 255, 255))

        # 前景をペースト
        crop = image_rgba.crop((x_min, y_min, x_max + 1, y_max + 1))
        new_image.paste(crop, (pad_x, pad_y), crop)

        return new_image

    def _inference_context(self):
        """推論用コンテキストマネージャ。"""
        import torch
        return torch.no_grad()

    def _convert_to_shadowbox_mesh(
        self,
        trimesh_mesh,
        original_image: Image.Image,
    ) -> ShadowboxMesh:
        """trimeshメッシュをShadowboxMeshに変換。

        Args:
            trimesh_mesh: TripoSRが生成したtrimeshメッシュ。
            original_image: 元の入力画像（テクスチャ用）。

        Returns:
            ShadowboxMesh: 変換されたメッシュ。
        """
        vertices = np.array(trimesh_mesh.vertices, dtype=np.float32)
        faces = np.array(trimesh_mesh.faces, dtype=np.int32)

        # 頂点カラーを取得（テクスチャがある場合）
        if hasattr(trimesh_mesh.visual, "vertex_colors"):
            colors = np.array(trimesh_mesh.visual.vertex_colors[:, :3], dtype=np.uint8)
        else:
            # デフォルトカラー（グレー）
            colors = np.full((len(vertices), 3), 128, dtype=np.uint8)

        # 正規化（-1〜1の範囲に）
        if len(vertices) > 0:
            center = (vertices.max(axis=0) + vertices.min(axis=0)) / 2
            vertices = vertices - center
            scale = np.abs(vertices).max()
            if scale > 0:
                vertices = vertices / scale

            # TripoSR座標系 → Shadowbox座標系の変換
            # TripoSR: X=奥行き(カメラ方向), Y=右, Z=上
            # Shadowbox: X=右, Y=上, Z=奥行き(正面)
            # 変換: (x, y, z) → (y, z, x)
            vertices = vertices[:, [1, 2, 0]]

        # バウンディングボックスを計算
        if len(vertices) > 0:
            min_coords = vertices.min(axis=0)
            max_coords = vertices.max(axis=0)
            bounds = (
                float(min_coords[0]), float(max_coords[0]),  # x
                float(min_coords[1]), float(max_coords[1]),  # y
                float(min_coords[2]), float(max_coords[2]),  # z
            )
        else:
            bounds = (-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)

        # ピクセルインデックス（TripoSRでは未使用だがLayerMeshの必須フィールド）
        pixel_indices = np.zeros((len(vertices), 2), dtype=np.int32)

        # 単一レイヤーとしてShadowboxMeshを作成
        layer = LayerMesh(
            layer_index=0,
            vertices=vertices,
            colors=colors,
            z_position=0.0,
            pixel_indices=pixel_indices,
            faces=faces,
        )

        return ShadowboxMesh(
            layers=[layer],
            frame=None,  # TripoSRメッシュにはフレームなし
            bounds=bounds,
        )

    @property
    def settings(self) -> TripoSRSettings:
        """現在の設定を取得。"""
        return self._settings
