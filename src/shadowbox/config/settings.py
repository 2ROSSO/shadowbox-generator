"""シャドーボックスジェネレータの設定モジュール。

このモジュールは、深度推定、クラスタリング、レンダリングの
各種設定を管理するデータクラスを提供します。
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass
class DepthEstimationSettings:
    """深度推定モデルの設定。

    深度推定に使用するモデルの種類、デバイス、バッチサイズなどを
    設定します。モデルは差し替え可能な設計になっています。

    Attributes:
        model_type: 使用する深度推定モデルの種類。
            - "depth_anything": Depth Anything v2 (デフォルト、軽量・高精度)
            - "midas": MiDaS (成熟したエコシステム)
            - "zoedepth": ZoeDepth (メトリック深度対応)
        model_name: Hugging Faceのモデル名またはパス。
        device: 推論を実行するデバイス。
            - "auto": 自動検出 (CUDA > MPS > CPU)
            - "cpu": CPU使用
            - "cuda": NVIDIA GPU使用
            - "mps": Apple Silicon使用
        batch_size: 推論時のバッチサイズ。

    Example:
        >>> # デフォルト設定を使用
        >>> settings = DepthEstimationSettings()
        >>>
        >>> # MiDaSモデルに切り替え
        >>> settings = DepthEstimationSettings(
        ...     model_type="midas",
        ...     model_name="Intel/dpt-large"
        ... )
    """

    model_type: Literal["depth_anything", "midas", "zoedepth"] = "depth_anything"
    model_name: str = "depth-anything/Depth-Anything-V2-Small-hf"
    device: Literal["cpu", "cuda", "mps", "auto"] = "auto"
    batch_size: int = 1
    invert_depth: bool = True  # Depth Anything v2 は大きい値=手前のため反転が必要


@dataclass
class ClusteringSettings:
    """深度レイヤーのクラスタリング設定。

    深度値を離散的なレイヤーに分割するためのクラスタリング
    パラメータを設定します。最適なクラスタ数(k)は自動で探索されます。

    Attributes:
        min_k: 探索するクラスタ数の最小値。
        max_k: 探索するクラスタ数の最大値。
        method: 最適なkを見つける方法。
            - "silhouette": シルエット分析 (デフォルト、推奨)
            - "elbow": エルボー法
        random_state: 再現性のための乱数シード。

    Example:
        >>> # デフォルト設定 (3〜10層で自動探索)
        >>> settings = ClusteringSettings()
        >>>
        >>> # 層数の範囲を狭める
        >>> settings = ClusteringSettings(min_k=4, max_k=7)
    """

    min_k: int = 3
    max_k: int = 10
    method: Literal["elbow", "silhouette"] = "silhouette"
    random_state: int = 42


@dataclass
class RenderSettings:
    """3Dレンダリングの設定。

    シャドーボックスの3D表示に関する設定を管理します。
    レイヤーの厚み、間隔、フレームの位置などを調整できます。

    Attributes:
        layer_thickness: 各レイヤーのZ軸方向の厚み。
        layer_gap: レイヤー間の隙間。0で密着した積み上げになります。
        frame_z: フレーム(枠)のZ位置。0が最前面です。
        background_color: 3Dシーンの背景色 (16進数カラーコード)。
        lighting_intensity: シーン照明の強度。
        point_size: ポイントクラウドレンダリング時の点のサイズ。
        frame_wall_mode: フレームの壁モード。
            - "none": 壁なし（従来の2D平面フレーム）
            - "outer": 外側の壁のみ（本物のシャドーボックス風）
        cumulative_layers: 累積レイヤーモード。
            Trueの場合、奥のレイヤーは手前のピクセルを継承し、
            最奥のレイヤーは完全な画像になります。

    Example:
        >>> # デフォルト設定
        >>> settings = RenderSettings()
        >>>
        >>> # レイヤー間に隙間を設ける
        >>> settings = RenderSettings(layer_gap=0.05)
        >>>
        >>> # 従来の穴あきレイヤーモード
        >>> settings = RenderSettings(cumulative_layers=False)
    """

    layer_thickness: float = 0.1
    layer_gap: float = 0.0
    frame_z: float = 0.0
    background_color: str = "#1a1a2e"
    lighting_intensity: float = 1.0
    point_size: int = 3
    frame_wall_mode: Literal["none", "outer"] = "outer"
    cumulative_layers: bool = True


@dataclass
class ShadowboxSettings:
    """シャドーボックスジェネレータのメイン設定コンテナ。

    深度推定、クラスタリング、レンダリングの全設定を
    一元管理するトップレベルの設定オブジェクトです。

    Attributes:
        depth: 深度推定の設定。
        clustering: クラスタリングの設定。
        render: レンダリングの設定。
        templates_dir: カードテンプレートYAMLファイルの保存ディレクトリ。

    Example:
        >>> # デフォルト設定でパイプラインを作成
        >>> settings = ShadowboxSettings()
        >>> pipeline = create_pipeline(settings)
        >>>
        >>> # モデルを変更
        >>> settings = ShadowboxSettings()
        >>> settings.depth.model_type = "midas"
        >>> pipeline = create_pipeline(settings)
    """

    depth: DepthEstimationSettings = field(default_factory=DepthEstimationSettings)
    clustering: ClusteringSettings = field(default_factory=ClusteringSettings)
    render: RenderSettings = field(default_factory=RenderSettings)
    templates_dir: Path = field(default_factory=lambda: Path("data/templates"))

    def __post_init__(self) -> None:
        """初期化後処理: templates_dirを文字列からPathに変換。"""
        if isinstance(self.templates_dir, str):
            self.templates_dir = Path(self.templates_dir)
