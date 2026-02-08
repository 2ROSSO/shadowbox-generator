"""処理設定タブ。"""

from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from shadowbox.gui.i18n import tr


class ProcessingTab(QWidget):
    """処理パラメータ設定タブ。

    Signals:
        settings_changed: 設定が変更されたとき。
    """

    settings_changed = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._init_ui()
        self._connect_signals()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        # Model Mode
        row = QHBoxLayout()
        self._lbl_model_mode = QLabel(tr("proc.model_mode"))
        row.addWidget(self._lbl_model_mode)
        self.model_mode = QComboBox()
        self.model_mode.addItems(["depth", "triposr"])
        row.addWidget(self.model_mode)
        layout.addLayout(row)

        # Detection Method
        row = QHBoxLayout()
        self._lbl_detection = QLabel(tr("proc.detection"))
        row.addWidget(self._lbl_detection)
        self.detection_method = QComboBox()
        from shadowbox.detection.region import DETECTION_METHODS

        self.detection_method.addItems(["auto", "none"] + DETECTION_METHODS)
        row.addWidget(self.detection_method)
        layout.addLayout(row)

        # Mock Depth
        self.mock_depth = QCheckBox(tr("proc.mock_depth"))
        layout.addWidget(self.mock_depth)

        # Raw Depth
        self.raw_depth = QCheckBox(tr("proc.raw_depth"))
        layout.addWidget(self.raw_depth)

        # Depth Scale
        row = QHBoxLayout()
        self._lbl_depth_scale = QLabel(tr("proc.depth_scale"))
        row.addWidget(self._lbl_depth_scale)
        self.depth_scale = QDoubleSpinBox()
        self.depth_scale.setRange(0.1, 5.0)
        self.depth_scale.setSingleStep(0.1)
        self.depth_scale.setValue(1.0)
        self.depth_scale.setEnabled(False)
        row.addWidget(self.depth_scale)
        layout.addLayout(row)

        # Num Layers
        row = QHBoxLayout()
        self._lbl_num_layers = QLabel(tr("proc.num_layers"))
        row.addWidget(self._lbl_num_layers)
        self.num_layers_auto = QCheckBox(tr("proc.auto"))
        self.num_layers_auto.setChecked(False)
        row.addWidget(self.num_layers_auto)
        self.num_layers = QSpinBox()
        self.num_layers.setRange(1, 10)
        self.num_layers.setValue(5)
        self.num_layers.setEnabled(True)
        row.addWidget(self.num_layers)
        layout.addLayout(row)

        # Max Resolution
        row = QHBoxLayout()
        self._lbl_max_res = QLabel(tr("proc.max_resolution"))
        row.addWidget(self._lbl_max_res)
        self.max_res_unlimited = QCheckBox(tr("proc.unlimited"))
        self.max_res_unlimited.setChecked(False)
        row.addWidget(self.max_res_unlimited)
        self.max_resolution = QSpinBox()
        self.max_resolution.setRange(50, 2000)
        self.max_resolution.setValue(300)
        self.max_resolution.setEnabled(True)
        row.addWidget(self.max_resolution)
        layout.addLayout(row)

        layout.addStretch()

    def _connect_signals(self) -> None:
        self.raw_depth.toggled.connect(self._on_raw_depth_toggled)
        self.num_layers_auto.toggled.connect(self._on_auto_layers_toggled)
        self.max_res_unlimited.toggled.connect(self._on_unlimited_res_toggled)

        # Emit settings_changed for all controls
        self.model_mode.currentIndexChanged.connect(self.settings_changed)
        self.detection_method.currentIndexChanged.connect(self.settings_changed)
        self.mock_depth.toggled.connect(self.settings_changed)
        self.raw_depth.toggled.connect(self.settings_changed)
        self.depth_scale.valueChanged.connect(self.settings_changed)
        self.num_layers_auto.toggled.connect(self.settings_changed)
        self.num_layers.valueChanged.connect(self.settings_changed)
        self.max_res_unlimited.toggled.connect(self.settings_changed)
        self.max_resolution.valueChanged.connect(self.settings_changed)

    def _on_raw_depth_toggled(self, checked: bool) -> None:
        self.depth_scale.setEnabled(checked)
        self.num_layers.setEnabled(not checked and not self.num_layers_auto.isChecked())
        self.num_layers_auto.setEnabled(not checked)

    def _on_auto_layers_toggled(self, checked: bool) -> None:
        self.num_layers.setEnabled(not checked and not self.raw_depth.isChecked())

    def _on_unlimited_res_toggled(self, checked: bool) -> None:
        self.max_resolution.setEnabled(not checked)

    def retranslate(self) -> None:
        """言語変更時にUI文字列を更新。"""
        self._lbl_model_mode.setText(tr("proc.model_mode"))
        self._lbl_detection.setText(tr("proc.detection"))
        self.mock_depth.setText(tr("proc.mock_depth"))
        self.raw_depth.setText(tr("proc.raw_depth"))
        self._lbl_depth_scale.setText(tr("proc.depth_scale"))
        self._lbl_num_layers.setText(tr("proc.num_layers"))
        self.num_layers_auto.setText(tr("proc.auto"))
        self._lbl_max_res.setText(tr("proc.max_resolution"))
        self.max_res_unlimited.setText(tr("proc.unlimited"))

    def set_values(self, values: dict) -> None:
        """辞書から設定値を復元。"""
        if "model_mode" in values:
            self.model_mode.setCurrentText(values["model_mode"])
        if "detection_method" in values:
            self.detection_method.setCurrentText(values["detection_method"])
        if "use_mock_depth" in values:
            self.mock_depth.setChecked(values["use_mock_depth"])
        if "use_raw_depth" in values:
            self.raw_depth.setChecked(values["use_raw_depth"])
        if "depth_scale" in values:
            self.depth_scale.setValue(values["depth_scale"])
        if "num_layers" in values:
            if values["num_layers"] is None:
                self.num_layers_auto.setChecked(True)
            else:
                self.num_layers_auto.setChecked(False)
                self.num_layers.setValue(values["num_layers"])
        if "max_resolution" in values:
            if values["max_resolution"] is None:
                self.max_res_unlimited.setChecked(True)
            else:
                self.max_res_unlimited.setChecked(False)
                self.max_resolution.setValue(values["max_resolution"])

    def get_values(self) -> dict:
        """現在の設定値を辞書で返す。"""
        return {
            "model_mode": self.model_mode.currentText(),
            "detection_method": self.detection_method.currentText(),
            "use_mock_depth": self.mock_depth.isChecked(),
            "use_raw_depth": self.raw_depth.isChecked(),
            "depth_scale": self.depth_scale.value(),
            "num_layers": None if self.num_layers_auto.isChecked() else self.num_layers.value(),
            "max_resolution": (
                None if self.max_res_unlimited.isChecked() else self.max_resolution.value()
            ),
        }
