"""スタンドアロンGUIアプリケーション。

このモジュールは、PyQt6を使用したスタンドアロンの
シャドーボックス生成アプリケーションを提供します。

使用方法:
    python -m shadowbox.gui.app
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

# Windows: PyQt6より先にtorchをロードしないとDLL検索パスが競合し
# c10.dll のロードに失敗する (WinError 1114)
if sys.platform == "win32":
    try:
        import torch  # noqa: F401
    except ImportError:
        pass

from PIL import Image

if TYPE_CHECKING:
    from shadowbox.config.template import BoundingBox

try:
    from PyQt6.QtGui import QAction, QActionGroup
    from PyQt6.QtWidgets import (
        QApplication,
        QFileDialog,
        QHBoxLayout,
        QMainWindow,
        QMessageBox,
        QProgressBar,
        QStatusBar,
        QVBoxLayout,
        QWidget,
    )

    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False


class ShadowboxApp(QMainWindow):
    """シャドーボックス生成GUIアプリケーション。

    画像選択、領域設定、パラメータ調整、3D表示を統合した
    スタンドアロンアプリケーション。

    Example:
        >>> app = QApplication(sys.argv)
        >>> window = ShadowboxApp()
        >>> window.show()
        >>> sys.exit(app.exec())
    """

    def __init__(self) -> None:
        super().__init__()

        from shadowbox.gui.i18n import load_language_preference

        load_language_preference()

        self._image: Image.Image | None = None
        self._result = None
        self._bbox: BoundingBox | None = None
        self._thread = None

        self._init_ui()
        self._restore_defaults()

    def _init_ui(self) -> None:
        from shadowbox.gui.i18n import tr

        self.setWindowTitle("TCG Shadowbox Generator")
        self.setMinimumSize(1000, 700)

        self._create_menu_bar()

        # Central widget
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # Left: Image preview
        from shadowbox.gui.widgets.image_preview import ImagePreview

        self.image_preview = ImagePreview()
        self.image_preview.region_selected.connect(self._on_region_selected)
        self.image_preview.region_cleared.connect(self._on_region_cleared)
        main_layout.addWidget(self.image_preview, stretch=3)

        # Right: Settings + actions
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)

        from shadowbox.gui.widgets.action_buttons import ActionButtons
        from shadowbox.gui.widgets.settings_panel import SettingsPanel

        self.settings_panel = SettingsPanel()
        right_layout.addWidget(self.settings_panel, stretch=1)

        self.action_buttons = ActionButtons()
        self.action_buttons.generate_clicked.connect(self._process_image)
        self.action_buttons.view_3d_clicked.connect(self._show_3d_view)
        self.action_buttons.export_clicked.connect(self._export_mesh)
        right_layout.addWidget(self.action_buttons)

        main_layout.addWidget(right_panel, stretch=1)

        # Status bar
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)
        self._status_bar.showMessage(tr("status.load_image"))

        self._progress_bar = QProgressBar()
        self._progress_bar.setVisible(False)
        self._status_bar.addPermanentWidget(self._progress_bar)

    def _create_menu_bar(self) -> None:
        from shadowbox.gui.i18n import get_language, tr

        menubar = self.menuBar()

        # File menu
        self._file_menu = menubar.addMenu(tr("menu.file"))
        self._open_action = QAction(tr("menu.open"), self)
        self._open_action.setShortcut("Ctrl+O")
        self._open_action.triggered.connect(self._open_image)
        self._file_menu.addAction(self._open_action)
        self._file_menu.addSeparator()
        self._exit_action = QAction(tr("menu.exit"), self)
        self._exit_action.setShortcut("Ctrl+Q")
        self._exit_action.triggered.connect(self.close)
        self._file_menu.addAction(self._exit_action)

        # Language menu
        self._lang_menu = menubar.addMenu(tr("menu.language"))
        lang_group = QActionGroup(self)
        lang_group.setExclusive(True)

        self._lang_en = QAction("English", self)
        self._lang_en.setCheckable(True)
        self._lang_en.triggered.connect(lambda: self._change_language("en"))
        lang_group.addAction(self._lang_en)
        self._lang_menu.addAction(self._lang_en)

        self._lang_ja = QAction("日本語", self)
        self._lang_ja.setCheckable(True)
        self._lang_ja.triggered.connect(lambda: self._change_language("ja"))
        lang_group.addAction(self._lang_ja)
        self._lang_menu.addAction(self._lang_ja)

        current = get_language()
        if current == "ja":
            self._lang_ja.setChecked(True)
        else:
            self._lang_en.setChecked(True)

        # Help menu
        self._help_menu = menubar.addMenu(tr("menu.help"))
        self._about_action = QAction(tr("menu.about"), self)
        self._about_action.triggered.connect(self._show_about)
        self._help_menu.addAction(self._about_action)

    def _change_language(self, lang: str) -> None:
        from shadowbox.gui.i18n import save_language_preference, set_language

        set_language(lang)
        save_language_preference()
        self._retranslate_ui()

    def _retranslate_ui(self) -> None:
        from shadowbox.gui.i18n import tr

        # Menus
        self._file_menu.setTitle(tr("menu.file"))
        self._open_action.setText(tr("menu.open"))
        self._exit_action.setText(tr("menu.exit"))
        self._lang_menu.setTitle(tr("menu.language"))
        self._help_menu.setTitle(tr("menu.help"))
        self._about_action.setText(tr("menu.about"))

        # Child widgets
        self.image_preview.retranslate()
        self.settings_panel.retranslate()
        self.action_buttons.retranslate()

    # ---- Image loading ----

    def _open_image(self) -> None:
        from shadowbox.gui.i18n import tr

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            tr("dialog.open_image"),
            "",
            tr("dialog.image_filter"),
        )
        if file_path:
            try:
                self._image = Image.open(file_path).convert("RGB")
                self.image_preview.set_image(self._image)
                self.action_buttons.set_has_image(True)
                self._result = None
                self.action_buttons.set_has_result(False)
                self._bbox = None
                self._status_bar.showMessage(
                    tr("status.loaded", name=Path(file_path).name)
                )
            except Exception as e:
                QMessageBox.critical(
                    self,
                    tr("dialog.error"),
                    tr("dialog.load_failed", error=e),
                )

    # ---- Region selection ----

    def _on_region_selected(self, x: int, y: int, w: int, h: int) -> None:
        from shadowbox.config.template import BoundingBox
        from shadowbox.gui.i18n import tr

        self._bbox = BoundingBox(x=x, y=y, width=w, height=h)
        self._status_bar.showMessage(
            tr("status.region_selected", x=x, y=y, w=w, h=h)
        )

    def _on_region_cleared(self) -> None:
        from shadowbox.gui.i18n import tr

        self._bbox = None
        self._status_bar.showMessage(tr("status.region_cleared"))

    # ---- Processing ----

    def _process_image(self) -> None:
        if self._image is None:
            return

        self.action_buttons.generate_btn.setEnabled(False)
        self.action_buttons.set_has_result(False)
        self._progress_bar.setVisible(True)
        self._progress_bar.setRange(0, 0)

        from shadowbox.gui.processing import ProcessingThread

        gs = self.settings_panel.get_gui_settings()
        self._thread = ProcessingThread(
            image=self._image,
            settings=gs,
            bbox=self._bbox,
        )
        self._thread.finished.connect(self._on_processing_finished)
        self._thread.error.connect(self._on_processing_error)
        self._thread.progress.connect(self._status_bar.showMessage)
        self._thread.start()

    def _on_processing_finished(self, result) -> None:
        from shadowbox.gui.i18n import tr

        self._result = result
        self._progress_bar.setVisible(False)
        self.action_buttons.set_has_image(True)
        self.action_buttons.set_has_result(True)
        self._status_bar.showMessage(
            tr(
                "status.done",
                layers=result.mesh.num_layers,
                vertices=f"{result.mesh.total_vertices:,}",
            )
        )

        # Clear previous region selection and bbox
        self.image_preview._region_selector.clear_selection()
        self._bbox = None

        # Update preview tabs
        if hasattr(result, "depth_map"):
            self.image_preview.set_depth_map(result.depth_map)
        if hasattr(result, "labels") and hasattr(result, "cropped_image"):
            self.image_preview.set_labels(result.labels, result.cropped_image)

    def _on_processing_error(self, error_msg: str) -> None:
        from shadowbox.gui.i18n import tr

        self._progress_bar.setVisible(False)
        self.action_buttons.set_has_image(self._image is not None)
        QMessageBox.critical(
            self,
            tr("dialog.error"),
            tr("dialog.process_failed", error=error_msg),
        )
        self._status_bar.showMessage(tr("status.error"))

    # ---- 3D View ----

    def _show_3d_view(self) -> None:
        if self._result is None:
            return
        try:
            from shadowbox.gui.settings_bridge import gui_to_render_options
            from shadowbox.visualization import render_shadowbox

            gs = self.settings_panel.get_gui_settings()
            options = gui_to_render_options(gs)
            render_shadowbox(self._result.mesh, options)
        except Exception as e:
            from shadowbox.gui.i18n import tr

            QMessageBox.critical(
                self,
                tr("dialog.error"),
                tr("dialog.view3d_failed", error=e),
            )

    # ---- Export ----

    def _export_mesh(self) -> None:
        if self._result is None:
            return

        from shadowbox.gui.i18n import tr

        file_path, selected_filter = QFileDialog.getSaveFileName(
            self,
            tr("dialog.export"),
            "shadowbox",
            "STL (*.stl);;OBJ (*.obj);;PLY (*.ply)",
        )
        if not file_path:
            return

        try:
            from shadowbox.visualization.export import (
                export_to_obj,
                export_to_ply,
                export_to_stl,
            )

            if file_path.endswith(".obj") or "OBJ" in selected_filter:
                if not file_path.endswith(".obj"):
                    file_path += ".obj"
                export_to_obj(self._result.mesh, file_path)
            elif file_path.endswith(".ply") or "PLY" in selected_filter:
                if not file_path.endswith(".ply"):
                    file_path += ".ply"
                export_to_ply(self._result.mesh, file_path)
            else:
                if not file_path.endswith(".stl"):
                    file_path += ".stl"
                export_to_stl(self._result.mesh, file_path)

            self._status_bar.showMessage(tr("status.exported", path=file_path))
        except Exception as e:
            QMessageBox.critical(
                self,
                tr("dialog.error"),
                tr("dialog.export_failed", error=e),
            )

    # ---- About ----

    def _show_about(self) -> None:
        from shadowbox.gui.i18n import tr

        QMessageBox.about(
            self,
            "About TCG Shadowbox Generator",
            tr("dialog.about_text"),
        )

    # ---- Defaults persistence ----

    def _restore_defaults(self) -> None:
        from shadowbox.gui.settings_bridge import load_defaults

        loaded = load_defaults()
        if loaded is not None:
            self.settings_panel.set_gui_settings(loaded)
        self._initial_settings = self.settings_panel.get_gui_settings()

    def closeEvent(self, event) -> None:  # noqa: N802
        from dataclasses import asdict

        from shadowbox.gui.i18n import tr
        from shadowbox.gui.settings_bridge import save_defaults

        current = self.settings_panel.get_gui_settings()
        if asdict(current) != asdict(self._initial_settings):
            reply = QMessageBox.question(
                self,
                tr("dialog.save_settings"),
                tr("dialog.save_settings_q"),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                save_defaults(current)
        super().closeEvent(event)


def main() -> None:
    """アプリケーションのエントリーポイント。"""
    if not PYQT_AVAILABLE:
        print("Error: PyQt6 is not installed.")
        print("Install: uv pip install PyQt6")
        sys.exit(1)

    app = QApplication(sys.argv)

    from shadowbox.gui.theme import DARK_STYLESHEET

    app.setStyleSheet(DARK_STYLESHEET)

    window = ShadowboxApp()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
