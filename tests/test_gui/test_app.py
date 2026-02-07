"""MainWindow テスト。"""

from unittest.mock import patch

import pytest

pytestmark = pytest.mark.gui


class TestShadowboxAppInit:
    def test_window_title(self, qtbot):
        from shadowbox.gui.app import ShadowboxApp

        window = ShadowboxApp()
        qtbot.addWidget(window)
        assert window.windowTitle() == "TCG Shadowbox Generator"

    def test_minimum_size(self, qtbot):
        from shadowbox.gui.app import ShadowboxApp

        window = ShadowboxApp()
        qtbot.addWidget(window)
        assert window.minimumWidth() == 1000
        assert window.minimumHeight() == 700

    def test_has_settings_panel(self, qtbot):
        from shadowbox.gui.app import ShadowboxApp

        window = ShadowboxApp()
        qtbot.addWidget(window)
        assert window.settings_panel is not None

    def test_has_action_buttons(self, qtbot):
        from shadowbox.gui.app import ShadowboxApp

        window = ShadowboxApp()
        qtbot.addWidget(window)
        assert window.action_buttons is not None

    def test_has_image_preview(self, qtbot):
        from shadowbox.gui.app import ShadowboxApp

        window = ShadowboxApp()
        qtbot.addWidget(window)
        assert window.image_preview is not None

    def test_generate_button_disabled_initially(self, qtbot):
        from shadowbox.gui.app import ShadowboxApp

        window = ShadowboxApp()
        qtbot.addWidget(window)
        assert not window.action_buttons.generate_btn.isEnabled()

    def test_view_3d_button_disabled_initially(self, qtbot):
        from shadowbox.gui.app import ShadowboxApp

        window = ShadowboxApp()
        qtbot.addWidget(window)
        assert not window.action_buttons.view_3d_btn.isEnabled()

    def test_export_button_disabled_initially(self, qtbot):
        from shadowbox.gui.app import ShadowboxApp

        window = ShadowboxApp()
        qtbot.addWidget(window)
        assert not window.action_buttons.export_btn.isEnabled()

    def test_status_bar_initial_message(self, qtbot):
        from shadowbox.gui.app import ShadowboxApp

        window = ShadowboxApp()
        qtbot.addWidget(window)
        assert "画像を読み込んでください" in window.statusBar().currentMessage()

    def test_image_path_initially_none(self, qtbot):
        from shadowbox.gui.app import ShadowboxApp

        window = ShadowboxApp()
        qtbot.addWidget(window)
        assert window._image_path is None


class TestRegionPersistence:
    def test_open_image_sets_path(self, qtbot, tmp_path):
        from PIL import Image

        from shadowbox.gui.app import ShadowboxApp

        img_path = tmp_path / "card.png"
        Image.new("RGB", (100, 100), "red").save(img_path)

        window = ShadowboxApp()
        qtbot.addWidget(window)
        window._open_image(str(img_path))
        assert window._image_path == str(img_path)

    def test_close_saves_region(self, qtbot, tmp_path):
        from PyQt6.QtWidgets import QMessageBox as RealQMessageBox

        from shadowbox.config.template import BoundingBox
        from shadowbox.gui.app import ShadowboxApp

        window = ShadowboxApp()
        qtbot.addWidget(window)
        window._image_path = "/path/to/card.png"
        window._bbox = BoundingBox(x=10, y=20, width=300, height=400)
        # Make settings differ from initial so save prompt triggers
        window._initial_settings.region_image_path = None

        with patch(
            "shadowbox.gui.settings_bridge.save_defaults"
        ) as mock_save, patch(
            "shadowbox.gui.app.QMessageBox"
        ) as mock_msgbox:
            mock_msgbox.StandardButton = RealQMessageBox.StandardButton
            mock_msgbox.question.return_value = (
                RealQMessageBox.StandardButton.Yes
            )
            window.close()
            assert mock_save.called
            saved_gs = mock_save.call_args[0][0]
            assert saved_gs.region_image_path == "/path/to/card.png"
            assert saved_gs.region_selection == (10, 20, 300, 400)

    def test_close_saves_none_region(self, qtbot):
        from PyQt6.QtWidgets import QMessageBox as RealQMessageBox

        from shadowbox.gui.app import ShadowboxApp

        window = ShadowboxApp()
        qtbot.addWidget(window)
        window._image_path = None
        window._bbox = None
        # Make settings differ from initial so save prompt triggers
        window._initial_settings.layer_interpolation = 999

        with patch(
            "shadowbox.gui.settings_bridge.save_defaults"
        ) as mock_save, patch(
            "shadowbox.gui.app.QMessageBox"
        ) as mock_msgbox:
            mock_msgbox.StandardButton = RealQMessageBox.StandardButton
            mock_msgbox.question.return_value = (
                RealQMessageBox.StandardButton.Yes
            )
            window.close()
            assert mock_save.called
            saved_gs = mock_save.call_args[0][0]
            assert saved_gs.region_selection is None

    def test_restore_region_matching_path(self, qtbot, tmp_path):
        from PIL import Image

        from shadowbox.gui.app import ShadowboxApp
        from shadowbox.gui.settings_bridge import GuiSettings

        img_path = tmp_path / "card.png"
        Image.new("RGB", (200, 200), "blue").save(img_path)

        saved_settings = GuiSettings(
            region_image_path=str(img_path),
            region_selection=(10, 20, 50, 60),
        )

        with patch(
            "shadowbox.gui.settings_bridge.load_defaults",
            return_value=saved_settings,
        ):
            window = ShadowboxApp()
            qtbot.addWidget(window)
            # _restore_defaults auto-opened the image and restored region
            assert window._bbox is not None
            assert window._bbox.x == 10
            assert window._bbox.y == 20

    def test_restore_region_different_path(self, qtbot, tmp_path):
        from PIL import Image

        from shadowbox.gui.app import ShadowboxApp
        from shadowbox.gui.settings_bridge import GuiSettings

        img_path = tmp_path / "card.png"
        Image.new("RGB", (200, 200), "blue").save(img_path)

        saved_settings = GuiSettings(
            region_image_path="/other/path.png",  # different path
            region_selection=(10, 20, 50, 60),
        )

        with patch(
            "shadowbox.gui.settings_bridge.load_defaults",
            return_value=saved_settings,
        ):
            window = ShadowboxApp()
            qtbot.addWidget(window)
            # Path doesn't exist so auto-open won't trigger;
            # even if manually opened, path mismatch → no restore
            window._open_image(str(img_path))
            assert window._bbox is None

    def test_auto_open_on_startup(self, qtbot, tmp_path):
        from PIL import Image

        from shadowbox.gui.app import ShadowboxApp
        from shadowbox.gui.settings_bridge import GuiSettings

        img_path = tmp_path / "saved_card.png"
        Image.new("RGB", (100, 100), "green").save(img_path)

        saved_settings = GuiSettings(
            region_image_path=str(img_path),
        )

        with patch(
            "shadowbox.gui.settings_bridge.load_defaults",
            return_value=saved_settings,
        ):
            window = ShadowboxApp()
            qtbot.addWidget(window)
            # Auto-open should have loaded the image
            assert window._image is not None
            assert window._image_path == str(img_path)
