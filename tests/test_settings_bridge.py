"""settings_bridge モジュールのテスト。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from shadowbox.gui.settings_bridge import (
    GuiSettings,
    gui_to_process_kwargs,
    load_defaults,
    save_defaults,
)


class TestGuiSettingsDefaults:
    """GuiSettings のデフォルト値を検証。"""

    def test_region_image_path_default(self) -> None:
        gs = GuiSettings()
        assert gs.region_image_path is None

    def test_region_selection_default(self) -> None:
        gs = GuiSettings()
        assert gs.region_selection is None

    def test_layer_interpolation_default(self) -> None:
        gs = GuiSettings()
        assert gs.layer_interpolation == 1

    def test_layer_pop_out_default(self) -> None:
        gs = GuiSettings()
        assert gs.layer_pop_out == 0.2

    def test_layer_spacing_mode_default(self) -> None:
        gs = GuiSettings()
        assert gs.layer_spacing_mode == "proportional"

    def test_layer_mask_mode_default(self) -> None:
        gs = GuiSettings()
        assert gs.layer_mask_mode == "contour"

    def test_layer_thickness_default(self) -> None:
        gs = GuiSettings()
        assert gs.layer_thickness == 0.2

    def test_include_card_frame_default(self) -> None:
        gs = GuiSettings()
        assert gs.include_card_frame is True

    def test_render_mode_default(self) -> None:
        gs = GuiSettings()
        assert gs.render_mode == "mesh"

    def test_detection_method_default(self) -> None:
        gs = GuiSettings()
        assert gs.detection_method == "auto"


class TestGuiToProcessKwargs:
    """gui_to_process_kwargs のテスト。"""

    def test_no_auto_detect_key(self) -> None:
        gs = GuiSettings()
        kwargs = gui_to_process_kwargs(gs)
        assert "auto_detect" not in kwargs

    def test_no_auto_detect_key_with_auto(self) -> None:
        gs = GuiSettings(detection_method="auto")
        kwargs = gui_to_process_kwargs(gs)
        assert "auto_detect" not in kwargs

    def test_no_auto_detect_key_with_none(self) -> None:
        gs = GuiSettings(detection_method="none")
        kwargs = gui_to_process_kwargs(gs)
        assert "auto_detect" not in kwargs

    def test_includes_basic_keys(self) -> None:
        gs = GuiSettings()
        kwargs = gui_to_process_kwargs(gs)
        assert "include_frame" in kwargs
        assert "include_card_frame" in kwargs

    def test_num_layers_omitted_when_none(self) -> None:
        gs = GuiSettings(num_layers=None)
        kwargs = gui_to_process_kwargs(gs)
        assert "k" not in kwargs

    def test_num_layers_included(self) -> None:
        gs = GuiSettings(num_layers=5)
        kwargs = gui_to_process_kwargs(gs)
        assert kwargs["k"] == 5


class TestSaveLoadDefaults:
    """save_defaults / load_defaults のラウンドトリップ。"""

    @pytest.fixture(autouse=True)
    def _use_tmp_path(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """一時ディレクトリを使って永続化テスト。"""
        test_path = tmp_path / "gui_defaults.json"
        monkeypatch.setattr(
            "shadowbox.gui.settings_bridge._DEFAULTS_PATH", test_path,
        )
        self._path = test_path

    def test_roundtrip(self) -> None:
        original = GuiSettings(
            layer_interpolation=3,
            render_mode="points",
            background_color=(10, 20, 30),
        )
        save_defaults(original)
        loaded = load_defaults()
        assert loaded is not None
        assert loaded.layer_interpolation == 3
        assert loaded.render_mode == "points"
        assert loaded.background_color == (10, 20, 30)

    def test_load_nonexistent_returns_none(self) -> None:
        assert load_defaults() is None

    def test_load_corrupt_json_returns_none(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text("{invalid json", encoding="utf-8")
        assert load_defaults() is None

    def test_load_unknown_fields_ignored(self) -> None:
        data = {"layer_interpolation": 2, "unknown_field": 42}
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(data), encoding="utf-8")
        loaded = load_defaults()
        assert loaded is not None
        assert loaded.layer_interpolation == 2

    def test_background_color_list_to_tuple(self) -> None:
        data = {"background_color": [100, 200, 50]}
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(data), encoding="utf-8")
        loaded = load_defaults()
        assert loaded is not None
        assert loaded.background_color == (100, 200, 50)
        assert isinstance(loaded.background_color, tuple)

    def test_roundtrip_region_selection(self) -> None:
        original = GuiSettings(
            region_image_path="/path/to/card.png",
            region_selection=(10, 20, 300, 400),
        )
        save_defaults(original)
        loaded = load_defaults()
        assert loaded is not None
        assert loaded.region_image_path == "/path/to/card.png"
        assert loaded.region_selection == (10, 20, 300, 400)

    def test_region_selection_none_roundtrip(self) -> None:
        original = GuiSettings()  # both region fields are None
        save_defaults(original)
        loaded = load_defaults()
        assert loaded is not None
        assert loaded.region_image_path is None
        assert loaded.region_selection is None

    def test_region_selection_list_to_tuple(self) -> None:
        data = {"region_selection": [10, 20, 300, 400]}
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(data), encoding="utf-8")
        loaded = load_defaults()
        assert loaded is not None
        assert loaded.region_selection == (10, 20, 300, 400)
        assert isinstance(loaded.region_selection, tuple)
