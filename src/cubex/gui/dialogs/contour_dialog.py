"""
Module containing the dialog for configuring contour overlays.

This dialog allows users to specify contour level modes (RMS, Linear, Log,
Percentage), visual styling (color, width, style), and smoothing parameters 
for individual panel overlays.
"""
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QLineEdit, QComboBox, QDoubleSpinBox, 
                             QRadioButton, QSpinBox, QTableWidget, QTableWidgetItem, 
                             QHeaderView, QMessageBox, QStackedWidget, QWidget,
                             QScrollArea, QGridLayout, QFileDialog, QGroupBox, QCheckBox,
                             QTabWidget)
import numpy as np
import pyqtgraph as pg
import matplotlib.pyplot as plt

class ContourDialog(QDialog):
    """
    Dialog window for configuring and applying contour overlays on 2D panels.

    Attributes
    ----------
    target_tab : ExplorerView or None
        The parent explorer tab where the contours will be drawn.
    target_id : int or str or None
        The identifier of the panel ('channel' or integer 0-2) receiving the contours.
    action : str
        The final action taken by the user ('apply', 'clear', or 'cancel').
    """
    _LINE_STYLES = {'Solid': Qt.SolidLine, 'Dashed': Qt.DashLine, 'Dotted': Qt.DotLine}

    def __init__(self, parent, current_params, title, target_tab=None, target_id=None):
        """
        Initialize the ContourDialog.

        Parameters
        ----------
        parent : PyQt5.QtWidgets.QWidget
            The parent widget.
        current_params : dict or None
            Existing contour configuration parameters to pre-populate the UI.
        title : str
            The window title.
        target_tab : ExplorerView, optional
            The explorer tab containing the target panel, by default None.
        target_id : int or str, optional
            The identifier for the specific panel to update, by default None.
        """
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(500)
        self.setMinimumHeight(520)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)

        self.target_tab = target_tab
        self.target_id = target_id
        self.action = 'cancel'

        opts = current_params or {}

        layout = QVBoxLayout(self)

        mode_group = QGroupBox("Levels Type")
        mode_layout = QVBoxLayout(mode_group)

        self.rad_rms = QRadioButton("Sigma / RMS List")
        self.rad_linear = QRadioButton("Linear Range")
        self.rad_log = QRadioButton("Logarithmic Range")
        self.rad_pct = QRadioButton("Percentage Levels (Recommended)")
        rad_style = ("QRadioButton { color: white; } "
                     "QRadioButton::indicator { width: 14px; height: 14px; border-radius: 7px; "
                     "border: 2px solid #888; background-color: #2a2a2a; } "
                     "QRadioButton::indicator:checked { border: 2px solid white; background-color: #111; }")
        self.rad_rms.setStyleSheet(rad_style)
        self.rad_linear.setStyleSheet(rad_style)
        self.rad_log.setStyleSheet(rad_style)
        self.rad_pct.setStyleSheet(rad_style)
        mode_layout.addWidget(self.rad_rms)
        mode_layout.addWidget(self.rad_linear)
        mode_layout.addWidget(self.rad_log)
        mode_layout.addWidget(self.rad_pct)
        layout.addWidget(mode_group)

        is_vel = False
        if self.target_tab is not None and self.target_id is not None and self.target_id != 'channel':
            try:
                mtype = self.target_tab.panels[self.target_id]['combo'].currentText()
                if "Moment 1" in mtype or "Moment 9" in mtype:
                    is_vel = True
            except Exception:
                pass

        if is_vel:
            self.rad_rms.setEnabled(False)
            self.rad_log.setEnabled(False)
            disabled_style = ("QRadioButton { color: gray; } "
                              "QRadioButton::indicator { width: 14px; height: 14px; border-radius: 7px; "
                              "border: 2px solid #555; background-color: #2a2a2a; } ")
            self.rad_rms.setStyleSheet(disabled_style)
            self.rad_log.setStyleSheet(disabled_style)
            if opts.get('mode', 'percent') in ['log', 'rms']:
                opts['mode'] = 'percent'
                
        self.stack = QStackedWidget()

        page_rms = QWidget()
        rms_layout = QVBoxLayout(page_rms)
        r1 = QHBoxLayout()
        r1.addWidget(QLabel("RMS / σ value:"))
        self.spin_rms = QDoubleSpinBox()
        self.spin_rms.setRange(1e-12, 1e12)
        self.spin_rms.setDecimals(6)
        self.spin_rms.setValue(float(opts.get('rms', 0.001)))
        r1.addWidget(self.spin_rms)
        rms_layout.addLayout(r1)
        r2 = QHBoxLayout()
        r2.addWidget(QLabel("Multipliers (comma-separated):"))
        self.edit_mult = QLineEdit()
        self.edit_mult.setText(opts.get('multipliers_str', '3, 5, 10, 20, 40'))
        self.edit_mult.setPlaceholderText("e.g., 3, 5, 10, 20, 40")
        r2.addWidget(self.edit_mult)
        rms_layout.addLayout(r2)
        self.stack.addWidget(page_rms)

        page_lin = QWidget()
        lin_layout = QVBoxLayout(page_lin)
        l1 = QHBoxLayout()
        l1.addWidget(QLabel("Min value:"))
        self.spin_lin_min = QDoubleSpinBox()
        self.spin_lin_min.setRange(-1e12, 1e12)
        self.spin_lin_min.setValue(float(opts.get('lin_min', 0.0)))
        l1.addWidget(self.spin_lin_min)
        lin_layout.addLayout(l1)
        l2 = QHBoxLayout()
        l2.addWidget(QLabel("Max value:"))
        self.spin_lin_max = QDoubleSpinBox()
        self.spin_lin_max.setRange(-1e12, 1e12)
        self.spin_lin_max.setValue(float(opts.get('lin_max', 10.0)))
        l2.addWidget(self.spin_lin_max)
        lin_layout.addLayout(l2)
        l3 = QHBoxLayout()
        l3.addWidget(QLabel("Number of levels:"))
        self.spin_lin_n = QSpinBox()
        self.spin_lin_n.setRange(1, 200)
        self.spin_lin_n.setValue(int(opts.get('n_levels', 5)))
        l3.addWidget(self.spin_lin_n)
        lin_layout.addLayout(l3)
        self.stack.addWidget(page_lin)

        page_log = QWidget()
        log_layout = QVBoxLayout(page_log)
        g1 = QHBoxLayout()
        g1.addWidget(QLabel("Min value:"))
        self.spin_log_min = QDoubleSpinBox()
        self.spin_log_min.setRange(1e-12, 1e12)
        self.spin_log_min.setDecimals(6)
        self.spin_log_min.setValue(float(opts.get('log_min', 0.001)))
        g1.addWidget(self.spin_log_min)
        log_layout.addLayout(g1)
        g2 = QHBoxLayout()
        g2.addWidget(QLabel("Max value:"))
        self.spin_log_max = QDoubleSpinBox()
        self.spin_log_max.setRange(1e-12, 1e12)
        self.spin_log_max.setDecimals(6)
        self.spin_log_max.setValue(float(opts.get('log_max', 10.0)))
        g2.addWidget(self.spin_log_max)
        log_layout.addLayout(g2)
        g3 = QHBoxLayout()
        g3.addWidget(QLabel("Number of levels:"))
        self.spin_log_n = QSpinBox()
        self.spin_log_n.setRange(1, 200)
        self.spin_log_n.setValue(int(opts.get('n_levels', 5)))
        g3.addWidget(self.spin_log_n)
        log_layout.addLayout(g3)
        g4 = QHBoxLayout()
        g4.addWidget(QLabel("Log base:"))
        self.spin_log_base = QDoubleSpinBox()
        self.spin_log_base.setRange(1.1, 1000.0)
        self.spin_log_base.setValue(float(opts.get('log_base', 10.0)))
        g4.addWidget(self.spin_log_base)
        log_layout.addLayout(g4)
        self.stack.addWidget(page_log)

        page_pct = QWidget()
        pct_layout = QVBoxLayout(page_pct)
        p1 = QHBoxLayout()
        p1.addWidget(QLabel("Percentages of peak (comma-separated):"))
        self.edit_pct = QLineEdit()
        self.edit_pct.setText(opts.get('percentages_str', '10, 30, 50, 70, 90'))
        self.edit_pct.setPlaceholderText("e.g., 10, 30, 50, 70, 90")
        p1.addWidget(self.edit_pct)
        pct_layout.addLayout(p1)
        self.stack.addWidget(page_pct)

        layout.addWidget(self.stack)

        style_group = QGroupBox("Visual Styling")
        style_layout = QGridLayout(style_group)

        style_layout.addWidget(QLabel("Line Color:"), 0, 0)
        self.combo_color = QComboBox()
        self.combo_color.addItems(["White", "Red", "Cyan", "Magenta", "Green",
                                    "Yellow", "Lime", "Orange", "Pink", "Aquamarine", "Gold"])
        color_val = opts.get('color', 'cyan').capitalize()
        color_idx = self.combo_color.findText(color_val)
        if color_idx >= 0:
            self.combo_color.setCurrentIndex(color_idx)
        style_layout.addWidget(self.combo_color, 0, 1)

        style_layout.addWidget(QLabel("Line Width:"), 1, 0)
        self.spin_lw = QDoubleSpinBox()
        self.spin_lw.setRange(0.1, 10.0)
        self.spin_lw.setSingleStep(0.1)
        self.spin_lw.setValue(float(opts.get('line_width', 1.5)))
        style_layout.addWidget(self.spin_lw, 1, 1)

        style_layout.addWidget(QLabel("Line Style:"), 2, 0)
        self.combo_style = QComboBox()
        self.combo_style.addItems(["Solid", "Dashed", "Dotted"])
        style_name = opts.get('line_style', 'solid')
        style_idx = self.combo_style.findText(style_name.capitalize())
        if style_idx >= 0:
            self.combo_style.setCurrentIndex(style_idx)
        style_layout.addWidget(self.combo_style, 2, 1)

        layout.addWidget(style_group)

        smooth_group = QGroupBox("Data Smoothing")
        smooth_layout = QVBoxLayout(smooth_group)
        sh1 = QHBoxLayout()
        self.chk_smooth = QCheckBox("Apply Gaussian blur to contour data")
        self.chk_smooth.setChecked(bool(opts.get('smooth', False)))
        sh1.addWidget(self.chk_smooth)
        smooth_layout.addLayout(sh1)
        sh2 = QHBoxLayout()
        sh2.addWidget(QLabel("Kernel size (pixels):"))
        self.spin_kernel = QSpinBox()
        self.spin_kernel.setRange(1, 7)
        self.spin_kernel.setSingleStep(2)
        self.spin_kernel.setValue(int(opts.get('smooth_kernel', 3)))
        if self.spin_kernel.value() % 2 == 0:
            self.spin_kernel.setValue(3)
        self.spin_kernel.setEnabled(self.chk_smooth.isChecked())
        sh2.addWidget(self.spin_kernel)
        sh2.addStretch()
        smooth_layout.addLayout(sh2)
        layout.addWidget(smooth_group)

        self.chk_smooth.toggled.connect(lambda checked: self.spin_kernel.setEnabled(checked))

        btn_layout = QHBoxLayout()
        btn_apply = QPushButton("Apply Contours")
        btn_apply.setStyleSheet("background-color: #27ae60; font-weight: bold; color: white;")
        btn_clear = QPushButton("Clear Contours")
        btn_clear.setStyleSheet("background-color: #c0392b; color: white;")
        btn_cancel = QPushButton("Cancel")

        btn_apply.clicked.connect(self._on_apply)
        btn_clear.clicked.connect(self._on_clear)
        btn_cancel.clicked.connect(self.reject)

        btn_layout.addWidget(btn_apply)
        btn_layout.addWidget(btn_clear)
        btn_layout.addWidget(btn_cancel)
        layout.addLayout(btn_layout)

        mode = opts.get('mode', 'percent')
        if mode == 'linear':
            self.rad_linear.setChecked(True)
            self.stack.setCurrentIndex(1)
        elif mode == 'log':
            self.rad_log.setChecked(True)
            self.stack.setCurrentIndex(2)
        elif mode == 'percent':
            self.rad_pct.setChecked(True)
            self.stack.setCurrentIndex(3)
        else:
            self.rad_rms.setChecked(True)
            self.stack.setCurrentIndex(0)

        self.rad_rms.toggled.connect(lambda: self.stack.setCurrentIndex(0))
        self.rad_linear.toggled.connect(lambda: self.stack.setCurrentIndex(1))
        self.rad_log.toggled.connect(lambda: self.stack.setCurrentIndex(2))
        self.rad_pct.toggled.connect(lambda: self.stack.setCurrentIndex(3))

    def _on_apply(self):
        """
        Gather settings from the UI and apply the contour configuration to the target panel.
        """
        opts = {}
        if self.rad_rms.isChecked():
            opts['mode'] = 'rms'
        elif self.rad_linear.isChecked():
            opts['mode'] = 'linear'
        elif self.rad_log.isChecked():
            opts['mode'] = 'log'
        elif self.rad_pct.isChecked():
            opts['mode'] = 'percent'

        opts['rms'] = self.spin_rms.value()
        opts['multipliers_str'] = self.edit_mult.text()
        opts['lin_min'] = self.spin_lin_min.value()
        opts['lin_max'] = self.spin_lin_max.value()
        opts['n_levels'] = (self.spin_lin_n.value() if opts['mode'] == 'linear'
                            else self.spin_log_n.value() if opts['mode'] == 'log'
                            else 5)
        opts['log_min'] = self.spin_log_min.value()
        opts['log_max'] = self.spin_log_max.value()
        opts['log_base'] = self.spin_log_base.value()
        opts['percentages_str'] = self.edit_pct.text()
        opts['color'] = self.combo_color.currentText()
        opts['line_width'] = self.spin_lw.value()
        opts['line_style'] = self.combo_style.currentText()
        opts['smooth'] = self.chk_smooth.isChecked()
        opts['smooth_kernel'] = self.spin_kernel.value()

        if self.target_tab is not None and self.target_id is not None:
            self.target_tab.contour_params[self.target_id] = opts
            if self.target_id == 'channel':
                self.target_tab.update_channel_map()
            else:
                self.target_tab.update_moment_maps()

    def _on_clear(self):
        """
        Remove contour configurations and clear them from the target panel.
        """
        if self.target_tab is not None and self.target_id is not None:
            self.target_tab.contour_params[self.target_id] = None
            if self.target_id == 'channel':
                self.target_tab.update_channel_map()
            else:
                self.target_tab.update_moment_maps()
