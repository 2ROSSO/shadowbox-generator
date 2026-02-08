"""settings_bridge の変換ロジックテスト（Qt不要）。"""

from shadowbox.gui.settings_bridge import (
    GuiSettings,
    gui_to_process_kwargs,
    gui_to_render_options,
    gui_to_shadowbox_settings,
)


class TestGuiToShadowboxSettings:
    def test_default_creates_depth_mode(self):
        gs = GuiSettings()
        ss = gui_to_shadowbox_settings(gs)
        assert ss.model_mode == "depth"

    def test_triposr_mode(self):
        gs = GuiSettings(model_mode="triposr")
        ss = gui_to_shadowbox_settings(gs)
        assert ss.model_mode == "triposr"

    def test_render_settings_mapped(self):
        gs = GuiSettings(
            layer_thickness=0.2,
            frame_depth=1.0,
            frame_wall_mode="none",
            cumulative_layers=False,
            back_panel=False,
            layer_interpolation=3,
            layer_pop_out=0.5,
            layer_spacing_mode="proportional",
            layer_mask_mode="contour",
        )
        ss = gui_to_shadowbox_settings(gs)
        r = ss.render
        assert r.layer_thickness == 0.2
        assert r.frame_depth == 1.0
        assert r.frame_wall_mode == "none"
        assert r.cumulative_layers is False
        assert r.back_panel is False
        assert r.layer_interpolation == 3
        assert r.layer_pop_out == 0.5
        assert r.layer_spacing_mode == "proportional"
        assert r.layer_mask_mode == "contour"


class TestGuiToRenderOptions:
    def test_default_values(self):
        gs = GuiSettings()
        ro = gui_to_render_options(gs)
        assert ro.render_mode == "mesh"
        assert ro.point_size == 3.0
        assert ro.background_color == (30, 30, 30)
        assert ro.show_axes is False
        assert ro.show_frame is True
        assert ro.layer_opacity == 1.0

    def test_custom_values(self):
        gs = GuiSettings(
            render_mode="mesh",
            mesh_size=0.02,
            show_axes=True,
            show_frame_3d=False,
            layer_opacity=0.5,
            background_color=(10, 20, 30),
        )
        ro = gui_to_render_options(gs)
        assert ro.render_mode == "mesh"
        assert ro.mesh_size == 0.02
        assert ro.show_axes is True
        assert ro.show_frame is False
        assert ro.layer_opacity == 0.5
        assert ro.background_color == (10, 20, 30)


class TestGuiToProcessKwargs:
    def test_default_kwargs(self):
        gs = GuiSettings()
        kw = gui_to_process_kwargs(gs)
        assert kw["include_frame"] is True
        assert kw["include_card_frame"] is True
        assert kw["use_raw_depth"] is False
        assert kw["depth_scale"] == 1.0
        assert "auto_detect" not in kw
        assert kw["k"] == 5
        assert kw["max_resolution"] == 300

    def test_num_layers_set(self):
        gs = GuiSettings(num_layers=5)
        kw = gui_to_process_kwargs(gs)
        assert kw["k"] == 5

    def test_max_resolution_set(self):
        gs = GuiSettings(max_resolution=200)
        kw = gui_to_process_kwargs(gs)
        assert kw["max_resolution"] == 200

    def test_detection_none(self):
        gs = GuiSettings(detection_method="none")
        kw = gui_to_process_kwargs(gs)
        assert "auto_detect" not in kw

    def test_raw_depth_kwargs(self):
        gs = GuiSettings(use_raw_depth=True, depth_scale=2.0)
        kw = gui_to_process_kwargs(gs)
        assert kw["use_raw_depth"] is True
        assert kw["depth_scale"] == 2.0
