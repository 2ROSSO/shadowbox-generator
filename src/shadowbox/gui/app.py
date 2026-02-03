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

import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from shadowbox.config.template import BoundingBox

try:
    from PyQt6.QtCore import Qt, QThread, pyqtSignal
    from PyQt6.QtGui import QAction, QImage, QPixmap
    from PyQt6.QtWidgets import (
        QApplication,
        QFileDialog,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QMainWindow,
        QMessageBox,
        QProgressBar,
        QPushButton,
        QSlider,
        QSpinBox,
        QStatusBar,
        QVBoxLayout,
        QWidget,
    )
    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False


class ProcessingThread(QThread):
    """バックグラウンドで画像処理を行うスレッド。"""

    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(
        self,
        image: Image.Image,
        k: int,
        use_mock: bool,
        bbox: BoundingBox | None = None,
    ):
        super().__init__()
        self._image = image
        self._k = k
        self._use_mock = use_mock
        self._bbox = bbox

    def run(self):
        """処理を実行。"""
        try:
            from shadowbox import create_pipeline

            self.progress.emit("パイプラインを作成中...")
            pipeline = create_pipeline(use_mock_depth=self._use_mock)

            self.progress.emit("深度推定中...")
            result = pipeline.process(
                self._image,
                custom_bbox=self._bbox,
                k=self._k,
                include_frame=True,
            )

            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


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

    def __init__(self):
        """アプリケーションを初期化。"""
        super().__init__()

        self._image: Image.Image | None = None
        self._result = None
        self._bbox: BoundingBox | None = None

        self._init_ui()

    def _init_ui(self):
        """UIを初期化。"""
        self.setWindowTitle("TCG Shadowbox Generator")
        self.setMinimumSize(800, 600)

        # メニューバー
        self._create_menu_bar()

        # メインウィジェット
        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        # メインレイアウト
        main_layout = QHBoxLayout(main_widget)

        # 左側: 画像表示
        left_panel = self._create_image_panel()
        main_layout.addWidget(left_panel, stretch=2)

        # 右側: コントロールパネル
        right_panel = self._create_control_panel()
        main_layout.addWidget(right_panel, stretch=1)

        # ステータスバー
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)
        self._status_bar.showMessage("画像を読み込んでください")

        # プログレスバー
        self._progress_bar = QProgressBar()
        self._progress_bar.setVisible(False)
        self._status_bar.addPermanentWidget(self._progress_bar)

    def _create_menu_bar(self):
        """メニューバーを作成。"""
        menubar = self.menuBar()

        # ファイルメニュー
        file_menu = menubar.addMenu("ファイル")

        open_action = QAction("画像を開く...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self._open_image)
        file_menu.addAction(open_action)

        file_menu.addSeparator()

        exit_action = QAction("終了", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # ヘルプメニュー
        help_menu = menubar.addMenu("ヘルプ")

        about_action = QAction("About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _create_image_panel(self) -> QWidget:
        """画像表示パネルを作成。"""
        panel = QGroupBox("画像プレビュー")
        layout = QVBoxLayout(panel)

        self._image_label = QLabel("画像が読み込まれていません")
        self._image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image_label.setMinimumSize(400, 400)
        self._image_label.setStyleSheet(
            "QLabel { background-color: #2a2a2a; color: #888; border: 1px solid #444; }"
        )
        layout.addWidget(self._image_label)

        return panel

    def _create_control_panel(self) -> QWidget:
        """コントロールパネルを作成。"""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # 画像読み込みボタン
        load_group = QGroupBox("画像")
        load_layout = QVBoxLayout(load_group)

        load_btn = QPushButton("画像を開く...")
        load_btn.clicked.connect(self._open_image)
        load_layout.addWidget(load_btn)

        layout.addWidget(load_group)

        # パラメータ設定
        param_group = QGroupBox("パラメータ")
        param_layout = QVBoxLayout(param_group)

        # レイヤー数
        layer_layout = QHBoxLayout()
        layer_layout.addWidget(QLabel("レイヤー数:"))
        self._layer_spin = QSpinBox()
        self._layer_spin.setRange(2, 10)
        self._layer_spin.setValue(5)
        layer_layout.addWidget(self._layer_spin)
        param_layout.addLayout(layer_layout)

        # ポイントサイズ
        point_layout = QHBoxLayout()
        point_layout.addWidget(QLabel("ポイントサイズ:"))
        self._point_slider = QSlider(Qt.Orientation.Horizontal)
        self._point_slider.setRange(1, 10)
        self._point_slider.setValue(4)
        self._point_label = QLabel("4")
        self._point_slider.valueChanged.connect(
            lambda v: self._point_label.setText(str(v))
        )
        point_layout.addWidget(self._point_slider)
        point_layout.addWidget(self._point_label)
        param_layout.addLayout(point_layout)

        layout.addWidget(param_group)

        # 処理ボタン
        process_group = QGroupBox("処理")
        process_layout = QVBoxLayout(process_group)

        self._process_btn = QPushButton("シャドーボックス生成")
        self._process_btn.setEnabled(False)
        self._process_btn.clicked.connect(self._process_image)
        process_layout.addWidget(self._process_btn)

        self._view_btn = QPushButton("3Dビューを開く")
        self._view_btn.setEnabled(False)
        self._view_btn.clicked.connect(self._show_3d_view)
        process_layout.addWidget(self._view_btn)

        layout.addWidget(process_group)

        # スペーサー
        layout.addStretch()

        return panel

    def _open_image(self):
        """画像ファイルを開く。"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "画像を開く",
            "",
            "画像ファイル (*.png *.jpg *.jpeg *.gif *.bmp);;すべてのファイル (*.*)",
        )

        if file_path:
            try:
                self._image = Image.open(file_path).convert("RGB")
                self._display_image(self._image)
                self._process_btn.setEnabled(True)
                self._status_bar.showMessage(f"読み込み完了: {Path(file_path).name}")
            except Exception as e:
                QMessageBox.critical(self, "エラー", f"画像の読み込みに失敗しました:\n{e}")

    def _display_image(self, image: Image.Image):
        """画像を表示。"""
        # PILからQPixmapに変換
        img_array = np.array(image)
        height, width, channel = img_array.shape
        bytes_per_line = 3 * width

        q_image = QImage(
            img_array.data,
            width,
            height,
            bytes_per_line,
            QImage.Format.Format_RGB888,
        )
        pixmap = QPixmap.fromImage(q_image)

        # ラベルに合わせてスケール
        scaled_pixmap = pixmap.scaled(
            self._image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._image_label.setPixmap(scaled_pixmap)

    def _process_image(self):
        """画像を処理してシャドーボックスを生成。"""
        if self._image is None:
            return

        # ボタンを無効化
        self._process_btn.setEnabled(False)
        self._view_btn.setEnabled(False)
        self._progress_bar.setVisible(True)
        self._progress_bar.setRange(0, 0)  # インデターミネートモード

        # バックグラウンドで処理
        self._thread = ProcessingThread(
            self._image,
            k=self._layer_spin.value(),
            use_mock=True,  # デモ用にモックを使用
            bbox=self._bbox,
        )
        self._thread.finished.connect(self._on_processing_finished)
        self._thread.error.connect(self._on_processing_error)
        self._thread.progress.connect(self._status_bar.showMessage)
        self._thread.start()

    def _on_processing_finished(self, result):
        """処理完了時のコールバック。"""
        self._result = result
        self._progress_bar.setVisible(False)
        self._process_btn.setEnabled(True)
        self._view_btn.setEnabled(True)
        self._status_bar.showMessage(
            f"処理完了: {result.mesh.num_layers}レイヤー, {result.mesh.total_vertices}頂点"
        )

    def _on_processing_error(self, error_msg):
        """処理エラー時のコールバック。"""
        self._progress_bar.setVisible(False)
        self._process_btn.setEnabled(True)
        QMessageBox.critical(self, "エラー", f"処理に失敗しました:\n{error_msg}")
        self._status_bar.showMessage("処理エラー")

    def _show_3d_view(self):
        """3Dビューを表示。"""
        if self._result is None:
            return

        try:
            from shadowbox.visualization import RenderOptions, render_shadowbox

            options = RenderOptions(
                point_size=float(self._point_slider.value()),
                window_size=(1000, 800),
                title="TCG Shadowbox 3D View",
            )
            render_shadowbox(self._result.mesh, options)
        except Exception as e:
            QMessageBox.critical(self, "エラー", f"3D表示に失敗しました:\n{e}")

    def _show_about(self):
        """Aboutダイアログを表示。"""
        QMessageBox.about(
            self,
            "About TCG Shadowbox Generator",
            "TCG Shadowbox Generator\n\n"
            "TCGカードのイラストを深度推定とクラスタリングで階層化し、\n"
            "インタラクティブな3Dシャドーボックスとして表示するツール。\n\n"
            "https://github.com/2ROSSO/shadowbox-generator",
        )


def main():
    """アプリケーションのエントリーポイント。"""
    if not PYQT_AVAILABLE:
        print("エラー: PyQt6がインストールされていません。")
        print("インストール方法: uv pip install PyQt6")
        sys.exit(1)

    app = QApplication(sys.argv)

    # ダークテーマ風のスタイルシート
    app.setStyleSheet("""
        QMainWindow {
            background-color: #1e1e1e;
        }
        QGroupBox {
            color: #ddd;
            border: 1px solid #444;
            border-radius: 5px;
            margin-top: 10px;
            padding-top: 10px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 0 5px;
        }
        QLabel {
            color: #ddd;
        }
        QPushButton {
            background-color: #3a3a3a;
            color: #ddd;
            border: 1px solid #555;
            border-radius: 3px;
            padding: 5px 15px;
        }
        QPushButton:hover {
            background-color: #4a4a4a;
        }
        QPushButton:pressed {
            background-color: #2a2a2a;
        }
        QPushButton:disabled {
            background-color: #2a2a2a;
            color: #666;
        }
        QSpinBox, QSlider {
            background-color: #3a3a3a;
            color: #ddd;
            border: 1px solid #555;
        }
        QMenuBar {
            background-color: #2a2a2a;
            color: #ddd;
        }
        QMenuBar::item:selected {
            background-color: #3a3a3a;
        }
        QMenu {
            background-color: #2a2a2a;
            color: #ddd;
        }
        QMenu::item:selected {
            background-color: #3a3a3a;
        }
        QStatusBar {
            background-color: #2a2a2a;
            color: #888;
        }
    """)

    window = ShadowboxApp()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
