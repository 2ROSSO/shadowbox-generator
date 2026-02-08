"""settings_bridge モジュールのテスト。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from shadowbox.gui.settings_bridge import (
    GuiSettings,
    gui_to_process_kwargs,
    load_defaults,
    load_region,
    remove_region,
    save_defaults,
    save_region,
)


class TestGuiSettingsDefaults:
    """GuiSettings のデフォルト値を検証。"""

    def test_region_image_path_default(self) -> None:
        gs = GuiSettings()
        assert gs.region_image_path is None

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

    def test_region_image_path_roundtrip(self) -> None:
        original = GuiSettings(region_image_path="/path/to/card.png")
        save_defaults(original)
        loaded = load_defaults()
        assert loaded is not None
        assert loaded.region_image_path == "/path/to/card.png"

    def test_old_region_selection_key_ignored(self) -> None:
        """既存JSONにregion_selectionがあっても無視される。"""
        data = {"region_selection": [10, 20, 300, 400]}
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(data), encoding="utf-8")
        loaded = load_defaults()
        assert loaded is not None
        assert not hasattr(loaded, "region_selection")


class TestRegionHistory:
    """Per-card region history のテスト。"""

    @pytest.fixture(autouse=True)
    def _use_tmp_path(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        test_path = tmp_path / "region_history.json"
        monkeypatch.setattr(
            "shadowbox.gui.settings_bridge._REGION_HISTORY_PATH", test_path,
        )
        self._path = test_path

    def test_load_nonexistent_returns_none(self) -> None:
        assert load_region("/path/to/card.png") is None

    def test_save_load_roundtrip(self) -> None:
        save_region("/path/to/card.png", (10, 20, 300, 400))
        result = load_region("/path/to/card.png")
        assert result == (10, 20, 300, 400)

    def test_overwrite_existing(self) -> None:
        save_region("/path/to/card.png", (10, 20, 300, 400))
        save_region("/path/to/card.png", (50, 60, 100, 200))
        result = load_region("/path/to/card.png")
        assert result == (50, 60, 100, 200)

    def test_multiple_images_independent(self) -> None:
        save_region("/card_a.png", (1, 2, 3, 4))
        save_region("/card_b.png", (5, 6, 7, 8))
        assert load_region("/card_a.png") == (1, 2, 3, 4)
        assert load_region("/card_b.png") == (5, 6, 7, 8)

    def test_remove_region(self) -> None:
        save_region("/card.png", (10, 20, 30, 40))
        remove_region("/card.png")
        assert load_region("/card.png") is None

    def test_remove_nonexistent_no_error(self) -> None:
        remove_region("/no/such/card.png")

    def test_lru_eviction(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "shadowbox.gui.settings_bridge._REGION_HISTORY_MAX", 3,
        )
        save_region("/a.png", (1, 1, 1, 1))
        save_region("/b.png", (2, 2, 2, 2))
        save_region("/c.png", (3, 3, 3, 3))
        # Adding a 4th should evict the oldest (/a.png)
        save_region("/d.png", (4, 4, 4, 4))
        assert load_region("/a.png") is None
        assert load_region("/b.png") == (2, 2, 2, 2)
        assert load_region("/d.png") == (4, 4, 4, 4)

    def test_corrupt_json_returns_none(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text("{bad json", encoding="utf-8")
        assert load_region("/card.png") is None
