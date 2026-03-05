"""アクションボタンパネル。"""

from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QPushButton, QVBoxLayout, QWidget

from shadowbox.gui.i18n import tr


class ActionButtons(QWidget):
    """生成・3Dビュー・エクスポートのアクションボタン群。

    Signals:
        generate_clicked: 生成ボタンが押されたとき。
        view_3d_clicked: 3Dビューボタンが押されたとき。
        export_clicked: エクスポートボタンが押されたとき。
        export_8dir_clicked: 8方向画像出力ボタンが押されたとき。
    """

    generate_clicked = pyqtSignal()
    view_3d_clicked = pyqtSignal()
    export_clicked = pyqtSignal()
    export_8dir_clicked = pyqtSignal()
    ai_convert_clicked = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.ai_convert_btn = QPushButton(tr("btn.ai_convert"))
        self.ai_convert_btn.setEnabled(False)
        self.ai_convert_btn.clicked.connect(self.ai_convert_clicked)
        layout.addWidget(self.ai_convert_btn)

        self.generate_btn = QPushButton(tr("btn.generate"))
        self.generate_btn.setObjectName("primary")
        self.generate_btn.setEnabled(False)
        self.generate_btn.clicked.connect(self.generate_clicked)
        layout.addWidget(self.generate_btn)

        self.view_3d_btn = QPushButton(tr("btn.view_3d"))
        self.view_3d_btn.setEnabled(False)
        self.view_3d_btn.clicked.connect(self.view_3d_clicked)
        layout.addWidget(self.view_3d_btn)

        self.export_btn = QPushButton(tr("btn.export_3d"))
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self.export_clicked)
        layout.addWidget(self.export_btn)

        self.export_8dir_btn = QPushButton(tr("btn.export_8dir"))
        self.export_8dir_btn.setEnabled(False)
        self.export_8dir_btn.clicked.connect(self.export_8dir_clicked)
        layout.addWidget(self.export_8dir_btn)

    def set_has_image(self, has_image: bool) -> None:
        """画像読み込み状態に応じてボタンを有効化。"""
        self.ai_convert_btn.setEnabled(has_image)
        self.generate_btn.setEnabled(has_image)

    def set_has_result(self, has_result: bool) -> None:
        """処理結果の有無に応じてボタンを有効化。"""
        self.view_3d_btn.setEnabled(has_result)
        self.export_btn.setEnabled(has_result)
        self.export_8dir_btn.setEnabled(has_result)

    def retranslate(self) -> None:
        """言語変更時にUI文字列を更新。"""
        self.ai_convert_btn.setText(tr("btn.ai_convert"))
        self.generate_btn.setText(tr("btn.generate"))
        self.view_3d_btn.setText(tr("btn.view_3d"))
        self.export_btn.setText(tr("btn.export_3d"))
        self.export_8dir_btn.setText(tr("btn.export_8dir"))
