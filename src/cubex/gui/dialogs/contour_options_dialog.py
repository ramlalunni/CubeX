"""
Module containing the dialog for managing multiple contour overlays.

This dialog allows users to edit properties (levels, styling, smoothing) 
for all active cross-file contour overlays on the main channel map, or 
clear them individually.
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

class ContourOptionsDialog(QDialog):
    """
    Dialog window for editing settings of multiple active contour overlays.

    Attributes
    ----------
    tab : ExplorerView
        The parent explorer tab containing the overlays.
    overlays : list of dict
        List of dictionaries containing data and configurations for each active overlay.
    tab_widget : PyQt5.QtWidgets.QTabWidget
        The tabbed widget grouping configurations for each individual overlay.
    """
    _LINE_STYLES = {'Solid': Qt.SolidLine, 'Dashed': Qt.DashLine, 'Dotted': Qt.DotLine}

    def __init__(self, parent, contour_overlays):
        """
        Initialize the ContourOptionsDialog.

        Parameters
        ----------
        parent : ExplorerView
            The parent explorer tab instance.
        contour_overlays : list of dict
            List of dictionaries containing data and configurations for each active overlay.
        """
        super().__init__(parent)
        self.setWindowTitle("Contour Overlay Options")
        self.setMinimumWidth(500)
        self.setMinimumHeight(520)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)

        self.tab = parent
        self.overlays = contour_overlays

        layout = QVBoxLayout(self)

        self.tab_widget = QTabWidget()

        for idx, ov in enumerate(self.overlays):
            tab_page = self._create_overlay_tab(idx, ov)
            self.tab_widget.addTab(tab_page, f"{ov['name']} ({ov['color']})")

        layout.addWidget(self.tab_widget)

        btn_layout = QHBoxLayout()
        btn_apply = QPushButton("Apply All")
        btn_apply.setStyleSheet("background-color: #27ae60; font-weight: bold; color: white;")
        btn_clear_all = QPushButton("Clear All Overlays")
        btn_clear_all.setStyleSheet("background-color: #c0392b; color: white;")
        btn_cancel = QPushButton("Cancel")

        btn_apply.clicked.connect(self._on_apply)
        btn_clear_all.clicked.connect(self._on_clear_all)
        btn_cancel.clicked.connect(self.reject)

        btn_layout.addWidget(btn_apply)
        btn_layout.addWidget(btn_clear_all)
        btn_layout.addWidget(btn_cancel)
        layout.addLayout(btn_layout)

    def _on_apply(self):
        """
        Apply configuration changes to all active overlays and trigger redraws.
        """
        for i in range(self.tab_widget.count()):
            new_opts = self._collect_tab_options(i)
            if 0 <= i < len(self.tab.contour_overlays):
                self.tab.contour_overlays[i]['options'] = new_opts
        self.tab.draw_overlay_contours()
        self.tab.update_spectrum()
        self.tab.update_spatial_analysis()

    def _on_clear_all(self):
        """
        Remove all active contour overlays from the channel map.
        """
        self.tab.close_overlay()
        self.close()

    def _on_clear_single(self, idx):
        """
        Remove a single specific contour overlay from the channel map.

        Parameters
        ----------
        idx : int
            The index of the overlay to remove.
        """
        self.tab.close_overlay(idx)
        self.tab_widget.removeTab(idx)
        if self.tab_widget.count() == 0:
            self.close()
        else:
            self.overlays = self.tab.contour_overlays

    def _create_overlay_tab(self, idx, ov):
        """
        Construct a configuration tab page for a given overlay.

        Parameters
        ----------
        idx : int
            The index of the overlay in the master list.
        ov : dict
            The overlay dictionary containing data and configuration.

        Returns
        -------
        PyQt5.QtWidgets.QWidget
            The generated configuration widget.
        """
        widget = QWidget()
        layout = QVBoxLayout(widget)
        opts = ov['options']

        mode_group = QGroupBox("Levels Type")
        mode_layout = QVBoxLayout(mode_group)

        rad_rms = QRadioButton("Sigma / RMS List")
        rad_linear = QRadioButton("Linear Range")
        rad_log = QRadioButton("Logarithmic Range")
        rad_pct = QRadioButton("Percentage Levels (Recommended)")
        rad_style = ("QRadioButton { color: white; } "
                     "QRadioButton::indicator { width: 14px; height: 14px; border-radius: 7px; "
                     "border: 2px solid #888; background-color: #2a2a2a; } "
                     "QRadioButton::indicator:checked { border: 2px solid white; background-color: #111; }")
        rad_rms.setStyleSheet(rad_style)
        rad_linear.setStyleSheet(rad_style)
        rad_log.setStyleSheet(rad_style)
        rad_pct.setStyleSheet(rad_style)
        mode_layout.addWidget(rad_rms)
        mode_layout.addWidget(rad_linear)
        mode_layout.addWidget(rad_log)
        mode_layout.addWidget(rad_pct)
        layout.addWidget(mode_group)

        stack = QStackedWidget()

        page_rms = QWidget()
        rms_layout = QVBoxLayout(page_rms)
        r1 = QHBoxLayout()
        r1.addWidget(QLabel("RMS / σ value:"))
        spin_rms = QDoubleSpinBox()
        spin_rms.setRange(1e-12, 1e12)
        spin_rms.setDecimals(6)
        spin_rms.setValue(float(opts.get('rms', 0.001)))
        r1.addWidget(spin_rms)
        rms_layout.addLayout(r1)
        r2 = QHBoxLayout()
        r2.addWidget(QLabel("Multipliers (comma-separated):"))
        edit_mult = QLineEdit()
        edit_mult.setText(opts.get('multipliers_str', '3, 5, 10, 20, 40'))
        edit_mult.setPlaceholderText("e.g., 3, 5, 10, 20, 40")
        r2.addWidget(edit_mult)
        rms_layout.addLayout(r2)
        stack.addWidget(page_rms)

        page_lin = QWidget()
        lin_layout = QVBoxLayout(page_lin)
        l1 = QHBoxLayout()
        l1.addWidget(QLabel("Min value:"))
        spin_lin_min = QDoubleSpinBox()
        spin_lin_min.setRange(-1e12, 1e12)
        spin_lin_min.setValue(float(opts.get('lin_min', 0.0)))
        l1.addWidget(spin_lin_min)
        lin_layout.addLayout(l1)
        l2 = QHBoxLayout()
        l2.addWidget(QLabel("Max value:"))
        spin_lin_max = QDoubleSpinBox()
        spin_lin_max.setRange(-1e12, 1e12)
        spin_lin_max.setValue(float(opts.get('lin_max', 10.0)))
        l2.addWidget(spin_lin_max)
        lin_layout.addLayout(l2)
        l3 = QHBoxLayout()
        l3.addWidget(QLabel("Number of levels:"))
        spin_lin_n = QSpinBox()
        spin_lin_n.setRange(1, 200)
        spin_lin_n.setValue(int(opts.get('n_levels', 5)))
        l3.addWidget(spin_lin_n)
        lin_layout.addLayout(l3)
        stack.addWidget(page_lin)

        page_log = QWidget()
        log_layout = QVBoxLayout(page_log)
        g1 = QHBoxLayout()
        g1.addWidget(QLabel("Min value:"))
        spin_log_min = QDoubleSpinBox()
        spin_log_min.setRange(1e-12, 1e12)
        spin_log_min.setDecimals(6)
        spin_log_min.setValue(float(opts.get('log_min', 0.001)))
        g1.addWidget(spin_log_min)
        log_layout.addLayout(g1)
        g2 = QHBoxLayout()
        g2.addWidget(QLabel("Max value:"))
        spin_log_max = QDoubleSpinBox()
        spin_log_max.setRange(1e-12, 1e12)
        spin_log_max.setDecimals(6)
        spin_log_max.setValue(float(opts.get('log_max', 10.0)))
        g2.addWidget(spin_log_max)
        log_layout.addLayout(g2)
        g3 = QHBoxLayout()
        g3.addWidget(QLabel("Number of levels:"))
        spin_log_n = QSpinBox()
        spin_log_n.setRange(1, 200)
        spin_log_n.setValue(int(opts.get('n_levels', 5)))
        g3.addWidget(spin_log_n)
        log_layout.addLayout(g3)
        g4 = QHBoxLayout()
        g4.addWidget(QLabel("Log base:"))
        spin_log_base = QDoubleSpinBox()
        spin_log_base.setRange(1.1, 1000.0)
        spin_log_base.setValue(float(opts.get('log_base', 10.0)))
        g4.addWidget(spin_log_base)
        log_layout.addLayout(g4)
        stack.addWidget(page_log)

        page_pct = QWidget()
        pct_layout = QVBoxLayout(page_pct)
        p1 = QHBoxLayout()
        p1.addWidget(QLabel("Percentages of peak (comma-separated):"))
        edit_pct = QLineEdit()
        edit_pct.setText(opts.get('percentages_str', '10, 30, 50, 70, 90'))
        edit_pct.setPlaceholderText("e.g., 10, 30, 50, 70, 90")
        p1.addWidget(edit_pct)
        pct_layout.addLayout(p1)
        stack.addWidget(page_pct)

        layout.addWidget(stack)

        style_group = QGroupBox("Visual Styling")
        style_layout = QGridLayout(style_group)

        style_layout.addWidget(QLabel("Line Color:"), 0, 0)
        combo_color = QComboBox()
        combo_color.addItems(["White", "Black", "Red", "Cyan", "Magenta", "Green",
                              "Yellow", "Lime", "Orange", "Pink", "Aquamarine", "Gold"])
        color_val = opts.get('color', 'Cyan').capitalize()
        color_idx = combo_color.findText(color_val)
        if color_idx >= 0:
            combo_color.setCurrentIndex(color_idx)
        style_layout.addWidget(combo_color, 0, 1)

        style_layout.addWidget(QLabel("Line Width:"), 1, 0)
        spin_lw = QDoubleSpinBox()
        spin_lw.setRange(0.1, 10.0)
        spin_lw.setSingleStep(0.1)
        spin_lw.setValue(float(opts.get('line_width', 1.5)))
        style_layout.addWidget(spin_lw, 1, 1)

        style_layout.addWidget(QLabel("Line Style:"), 2, 0)
        combo_style = QComboBox()
        combo_style.addItems(["Solid", "Dashed", "Dotted"])
        style_name = opts.get('line_style', 'Solid')
        style_idx = combo_style.findText(style_name.capitalize())
        if style_idx >= 0:
            combo_style.setCurrentIndex(style_idx)
        style_layout.addWidget(combo_style, 2, 1)

        layout.addWidget(style_group)

        smooth_group = QGroupBox("Data Smoothing")
        smooth_layout = QVBoxLayout(smooth_group)
        sh1 = QHBoxLayout()
        chk_smooth = QCheckBox("Apply Gaussian blur to contour data")
        chk_smooth.setChecked(bool(opts.get('smooth', False)))
        sh1.addWidget(chk_smooth)
        smooth_layout.addLayout(sh1)
        sh2 = QHBoxLayout()
        sh2.addWidget(QLabel("Kernel size (pixels):"))
        spin_kernel = QSpinBox()
        spin_kernel.setRange(1, 7)
        spin_kernel.setSingleStep(2)
        spin_kernel.setValue(int(opts.get('smooth_kernel', 3)))
        if spin_kernel.value() % 2 == 0:
            spin_kernel.setValue(3)
        spin_kernel.setEnabled(chk_smooth.isChecked())
        sh2.addWidget(spin_kernel)
        sh2.addStretch()
        smooth_layout.addLayout(sh2)
        layout.addWidget(smooth_group)

        chk_smooth.toggled.connect(lambda checked, s=spin_kernel: s.setEnabled(checked))

        btn_clear_this = QPushButton(f"Remove This Overlay")
        btn_clear_this.setStyleSheet("background-color: #c0392b; color: white;")
        btn_clear_this.clicked.connect(lambda checked, i=idx: self._on_clear_single(i))
        layout.addWidget(btn_clear_this)

        layout.addStretch()

        mode = opts.get('mode', 'percent')
        if mode == 'linear':
            rad_linear.setChecked(True)
            stack.setCurrentIndex(1)
        elif mode == 'log':
            rad_log.setChecked(True)
            stack.setCurrentIndex(2)
        elif mode == 'percent':
            rad_pct.setChecked(True)
            stack.setCurrentIndex(3)
        else:
            rad_rms.setChecked(True)
            stack.setCurrentIndex(0)

        rad_rms.toggled.connect(lambda: stack.setCurrentIndex(0))
        rad_linear.toggled.connect(lambda: stack.setCurrentIndex(1))
        rad_log.toggled.connect(lambda: stack.setCurrentIndex(2))
        rad_pct.toggled.connect(lambda: stack.setCurrentIndex(3))

        widget._controls = {
            'rad_rms': rad_rms, 'rad_linear': rad_linear, 'rad_log': rad_log, 'rad_pct': rad_pct,
            'spin_rms': spin_rms, 'edit_mult': edit_mult,
            'spin_lin_min': spin_lin_min, 'spin_lin_max': spin_lin_max, 'spin_lin_n': spin_lin_n,
            'spin_log_min': spin_log_min, 'spin_log_max': spin_log_max, 'spin_log_n': spin_log_n,
            'spin_log_base': spin_log_base,
            'edit_pct': edit_pct,
            'combo_color': combo_color, 'spin_lw': spin_lw, 'combo_style': combo_style,
            'chk_smooth': chk_smooth, 'spin_kernel': spin_kernel,
        }

        return widget

    def _collect_tab_options(self, tab_idx):
        """
        Extract the user-defined settings from a specific overlay tab.

        Parameters
        ----------
        tab_idx : int
            The index of the tab page.

        Returns
        -------
        dict
            A dictionary of the updated contour configuration settings.
        """
        tab = self.tab_widget.widget(tab_idx)
        c = tab._controls

        opts = {}
        if c['rad_rms'].isChecked():
            opts['mode'] = 'rms'
        elif c['rad_linear'].isChecked():
            opts['mode'] = 'linear'
        elif c['rad_log'].isChecked():
            opts['mode'] = 'log'
        elif c['rad_pct'].isChecked():
            opts['mode'] = 'percent'

        opts['rms'] = c['spin_rms'].value()
        opts['multipliers_str'] = c['edit_mult'].text()
        opts['lin_min'] = c['spin_lin_min'].value()
        opts['lin_max'] = c['spin_lin_max'].value()
        opts['n_levels'] = (c['spin_lin_n'].value() if opts['mode'] == 'linear'
                            else c['spin_log_n'].value() if opts['mode'] == 'log'
                            else 5)
        opts['log_min'] = c['spin_log_min'].value()
        opts['log_max'] = c['spin_log_max'].value()
        opts['log_base'] = c['spin_log_base'].value()
        opts['percentages_str'] = c['edit_pct'].text()
        opts['color'] = c['combo_color'].currentText().lower()
        opts['line_width'] = c['spin_lw'].value()
        opts['line_style'] = c['combo_style'].currentText().lower()
        opts['smooth'] = c['chk_smooth'].isChecked()
        opts['smooth_kernel'] = c['spin_kernel'].value()

        return opts