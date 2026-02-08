"""GUI設定とパイプライン設定の変換ブリッジ。

GuiSettings（プレーンPython型）をShadowboxSettings / RenderOptions /
パイプライン呼び出しkwargsに変換します。PyQt6 非依存のためテストが容易です。
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

_DEFAULTS_PATH = Path.home() / ".shadowbox" / "gui_defaults.json"


@dataclass
class GuiSettings:
    """GUI状態のスナップショット。全フィールドがプレーンPython型。"""

    # --- Processing ---
    model_mode: Literal["depth", "triposr"] = "depth"
    use_mock_depth: bool = False
    use_raw_depth: bool = False
    depth_scale: float = 1.0
    num_layers: int | None = 5  # None = auto
    max_resolution: int | None = 300  # None = unlimited
    detection_method: str = "auto"

    # --- Layers (RenderSettings) ---
    cumulative_layers: bool = True
    back_panel: bool = True
    layer_interpolation: int = 4
    layer_pop_out: float = 0.50
    layer_spacing_mode: Literal["even", "proportional"] = "proportional"
    layer_mask_mode: Literal["cluster", "contour"] = "contour"
    layer_thickness: float = 0.2

    # --- Frame ---
    include_frame: bool = True
    include_card_frame: bool = True
    frame_depth: float = 0.5
    frame_wall_mode: Literal["none", "outer"] = "outer"

    # --- Rendering (RenderOptions) ---
    render_mode: Literal["points", "mesh"] = "mesh"
    point_size: float = 3.0
    mesh_size: float = 0.008
    show_axes: bool = False
    show_frame_3d: bool = True
    layer_opacity: float = 1.0
    background_color: tuple[int, int, int] = field(default=(30, 30, 30))

    # --- Region (session state) ---
    region_image_path: str | None = None


def gui_to_shadowbox_settings(gs: GuiSettings):
    """GuiSettings → ShadowboxSettings を生成。

    Returns:
        ShadowboxSettings インスタンス。
    """
    from shadowbox.config.settings import RenderSettings, ShadowboxSettings

    render = RenderSettings(
        layer_thickness=gs.layer_thickness,
        frame_depth=gs.frame_depth,
        frame_wall_mode=gs.frame_wall_mode,
        cumulative_layers=gs.cumulative_layers,
        back_panel=gs.back_panel,
        layer_interpolation=gs.layer_interpolation,
        layer_pop_out=gs.layer_pop_out,
        layer_spacing_mode=gs.layer_spacing_mode,
        layer_mask_mode=gs.layer_mask_mode,
    )
    return ShadowboxSettings(
        model_mode=gs.model_mode,
        render=render,
    )


def gui_to_render_options(gs: GuiSettings):
    """GuiSettings → RenderOptions を生成。

    Returns:
        RenderOptions インスタンス。
    """
    from shadowbox.visualization.render import RenderOptions

    return RenderOptions(
        background_color=gs.background_color,
        point_size=gs.point_size,
        mesh_size=gs.mesh_size,
        show_axes=gs.show_axes,
        show_frame=gs.show_frame_3d,
        layer_opacity=gs.layer_opacity,
        render_mode=gs.render_mode,
        window_size=(1000, 800),
        title="TCG Shadowbox 3D View",
    )


def gui_to_process_kwargs(gs: GuiSettings) -> dict:
    """GuiSettings → pipeline.process() のキーワード引数を生成。

    Returns:
        pipeline.process() に展開できるdict。
    """
    kwargs: dict = {
        "include_frame": gs.include_frame,
        "include_card_frame": gs.include_card_frame,
        "use_raw_depth": gs.use_raw_depth,
        "depth_scale": gs.depth_scale,
    }
    if gs.num_layers is not None:
        kwargs["k"] = gs.num_layers
    if gs.max_resolution is not None:
        kwargs["max_resolution"] = gs.max_resolution
    return kwargs


def save_defaults(gs: GuiSettings) -> None:
    """GUI設定をJSONファイルに保存。

    Args:
        gs: 保存するGuiSettings。
    """
    data = asdict(gs)
    # tuple → list (JSON互換)
    data["background_color"] = list(data["background_color"])
    _DEFAULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    _DEFAULTS_PATH.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_defaults() -> GuiSettings | None:
    """保存済みGUI設定をJSONファイルから読込。

    Returns:
        GuiSettings。ファイルが無い/破損している場合は None。
    """
    if not _DEFAULTS_PATH.exists():
        return None
    try:
        raw = json.loads(_DEFAULTS_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        logger.warning("Failed to load GUI defaults from %s", _DEFAULTS_PATH)
        return None

    # 未知フィールドをフィルタリング（前方互換性）
    known = {f.name for f in fields(GuiSettings)}
    filtered = {k: v for k, v in raw.items() if k in known}

    # background_color: list → tuple
    if "background_color" in filtered and isinstance(
        filtered["background_color"], list
    ):
        filtered["background_color"] = tuple(filtered["background_color"])

    try:
        return GuiSettings(**filtered)
    except TypeError:
        logger.warning("Failed to construct GuiSettings from %s", _DEFAULTS_PATH)
        return None


# ---------------------------------------------------------------------------
# Per-card region history
# ---------------------------------------------------------------------------

_REGION_HISTORY_PATH = Path.home() / ".shadowbox" / "region_history.json"
_REGION_HISTORY_MAX = 200


def _load_region_history() -> dict[str, list[int]]:
    if not _REGION_HISTORY_PATH.exists():
        return {}
    try:
        data = json.loads(
            _REGION_HISTORY_PATH.read_text(encoding="utf-8")
        )
        if isinstance(data, dict):
            return data
    except (json.JSONDecodeError, OSError):
        logger.warning(
            "Failed to load region history from %s",
            _REGION_HISTORY_PATH,
        )
    return {}


def _save_region_history(history: dict[str, list[int]]) -> None:
    _REGION_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    _REGION_HISTORY_PATH.write_text(
        json.dumps(history, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def save_region(
    image_path: str, region: tuple[int, int, int, int]
) -> None:
    """画像パスに対応するリージョンを保存（LRU順）。"""
    history = _load_region_history()
    history.pop(image_path, None)
    history[image_path] = list(region)
    # LRU eviction: oldest entries first
    while len(history) > _REGION_HISTORY_MAX:
        oldest = next(iter(history))
        del history[oldest]
    _save_region_history(history)


def load_region(
    image_path: str,
) -> tuple[int, int, int, int] | None:
    """画像パスに対応するリージョンを読込。無ければ None。"""
    history = _load_region_history()
    value = history.get(image_path)
    if value is not None and isinstance(value, list) and len(value) == 4:
        return (value[0], value[1], value[2], value[3])
    return None


def remove_region(image_path: str) -> None:
    """画像パスに対応するリージョンを削除。"""
    history = _load_region_history()
    if image_path in history:
        del history[image_path]
        _save_region_history(history)
