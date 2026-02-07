"""レイヤー設定タブ。"""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from shadowbox.gui.i18n import tr


class LayersTab(QWidget):
    """レイヤーパラメータ設定タブ。

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

        # Cumulative Layers
        self.cumulative_layers = QCheckBox(tr("layer.cumulative"))
        self.cumulative_layers.setChecked(True)
        layout.addWidget(self.cumulative_layers)

        # Back Panel
        self.back_panel = QCheckBox(tr("layer.back_panel"))
        self.back_panel.setChecked(True)
        layout.addWidget(self.back_panel)

        # Layer Interpolation
        row = QHBoxLayout()
        self._lbl_interpolation = QLabel(tr("layer.interpolation"))
        row.addWidget(self._lbl_interpolation)
        self.layer_interpolation = QSpinBox()
        self.layer_interpolation.setRange(0, 5)
        self.layer_interpolation.setValue(1)
        row.addWidget(self.layer_interpolation)
        layout.addLayout(row)

        # Layer Pop Out
        row = QHBoxLayout()
        self._lbl_pop_out = QLabel(tr("layer.pop_out"))
        row.addWidget(self._lbl_pop_out)
        self.pop_out_slider = QSlider(Qt.Orientation.Horizontal)
        self.pop_out_slider.setRange(0, 100)
        self.pop_out_slider.setValue(20)
        row.addWidget(self.pop_out_slider)
        self.pop_out_spin = QDoubleSpinBox()
        self.pop_out_spin.setRange(0.0, 1.0)
        self.pop_out_spin.setSingleStep(0.05)
        self.pop_out_spin.setValue(0.2)
        self.pop_out_spin.setFixedWidth(70)
        row.addWidget(self.pop_out_spin)
        layout.addLayout(row)

        # Spacing Mode
        row = QHBoxLayout()
        self._lbl_spacing = QLabel(tr("layer.spacing_mode"))
        row.addWidget(self._lbl_spacing)
        self.spacing_mode = QComboBox()
        self.spacing_mode.addItems(["even", "proportional"])
        self.spacing_mode.setCurrentText("proportional")
        row.addWidget(self.spacing_mode)
        layout.addLayout(row)

        # Mask Mode
        row = QHBoxLayout()
        self._lbl_mask = QLabel(tr("layer.mask_mode"))
        row.addWidget(self._lbl_mask)
        self.mask_mode = QComboBox()
        self.mask_mode.addItems(["cluster", "contour"])
        self.mask_mode.setCurrentText("contour")
        row.addWidget(self.mask_mode)
        layout.addLayout(row)

        # Layer Thickness
        row = QHBoxLayout()
        self._lbl_thickness = QLabel(tr("layer.thickness"))
        row.addWidget(self._lbl_thickness)
        self.layer_thickness = QDoubleSpinBox()
        self.layer_thickness.setRange(0.01, 1.0)
        self.layer_thickness.setSingleStep(0.01)
        self.layer_thickness.setValue(0.2)
        row.addWidget(self.layer_thickness)
        layout.addLayout(row)

        layout.addStretch()

    def _connect_signals(self) -> None:
        # Slider ↔ SpinBox sync
        self.pop_out_slider.valueChanged.connect(
            lambda v: self.pop_out_spin.setValue(v / 100.0)
        )
        self.pop_out_spin.valueChanged.connect(
            lambda v: self.pop_out_slider.setValue(int(v * 100))
        )

        # settings_changed
        self.cumulative_layers.toggled.connect(self.settings_changed)
        self.back_panel.toggled.connect(self.settings_changed)
        self.layer_interpolation.valueChanged.connect(self.settings_changed)
        self.pop_out_spin.valueChanged.connect(self.settings_changed)
        self.spacing_mode.currentIndexChanged.connect(self.settings_changed)
        self.mask_mode.currentIndexChanged.connect(self.settings_changed)
        self.layer_thickness.valueChanged.connect(self.settings_changed)

    def retranslate(self) -> None:
        """言語変更時にUI文字列を更新。"""
        self.cumulative_layers.setText(tr("layer.cumulative"))
        self.back_panel.setText(tr("layer.back_panel"))
        self._lbl_interpolation.setText(tr("layer.interpolation"))
        self._lbl_pop_out.setText(tr("layer.pop_out"))
        self._lbl_spacing.setText(tr("layer.spacing_mode"))
        self._lbl_mask.setText(tr("layer.mask_mode"))
        self._lbl_thickness.setText(tr("layer.thickness"))

    def set_values(self, values: dict) -> None:
        """辞書から設定値を復元。"""
        if "cumulative_layers" in values:
            self.cumulative_layers.setChecked(values["cumulative_layers"])
        if "back_panel" in values:
            self.back_panel.setChecked(values["back_panel"])
        if "layer_interpolation" in values:
            self.layer_interpolation.setValue(values["layer_interpolation"])
        if "layer_pop_out" in values:
            self.pop_out_spin.setValue(values["layer_pop_out"])
        if "layer_spacing_mode" in values:
            self.spacing_mode.setCurrentText(values["layer_spacing_mode"])
        if "layer_mask_mode" in values:
            self.mask_mode.setCurrentText(values["layer_mask_mode"])
        if "layer_thickness" in values:
            self.layer_thickness.setValue(values["layer_thickness"])

    def get_values(self) -> dict:
        """現在の設定値を辞書で返す。"""
        return {
            "cumulative_layers": self.cumulative_layers.isChecked(),
            "back_panel": self.back_panel.isChecked(),
            "layer_interpolation": self.layer_interpolation.value(),
            "layer_pop_out": self.pop_out_spin.value(),
            "layer_spacing_mode": self.spacing_mode.currentText(),
            "layer_mask_mode": self.mask_mode.currentText(),
            "layer_thickness": self.layer_thickness.value(),
        }
