from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QLineEdit, QComboBox, QDoubleSpinBox, 
                             QRadioButton, QSpinBox, QTableWidget, QTableWidgetItem, 
                             QHeaderView, QMessageBox, QStackedWidget, QWidget,
                             QScrollArea, QGridLayout, QFileDialog, QGroupBox, QCheckBox,
                             QTabWidget)
import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# LINE CATALOG DIALOG
# ==============================================================================
class LineCatalogDialog(QDialog):
    """
    Dialog to input parameters for querying the Splatalogue database.
    """
    def __init__(self, parent, obs_fmin, obs_fmax):
        super().__init__(parent)
        self.setWindowTitle("Molecular Line Catalog Query")
        self.setMinimumWidth(420)
        self.fmin, self.fmax = obs_fmin, obs_fmax

        layout = QVBoxLayout(self)
        
        info = QLabel(f"<b>Selected Obs Range:</b> {self.fmin:.3f} to {self.fmax:.3f} GHz<br><i>(Rest frequency range will adjust based on V_sys)</i>")
        layout.addWidget(info)
        
        form = QVBoxLayout()
        
        h1 = QHBoxLayout()
        h1.addWidget(QLabel("Source Systemic Velocity ($v_{sys}$):"))
        self.spin_vsys = QDoubleSpinBox()
        self.spin_vsys.setRange(-1000, 1000)
        self.spin_vsys.setDecimals(2)
        self.spin_vsys.setValue(0.00)
        h1.addWidget(self.spin_vsys)
        h1.addWidget(QLabel("km/s"))
        form.addLayout(h1)
        
        h2 = QHBoxLayout()
        h2.addWidget(QLabel("Database:"))
        self.combo_db = QComboBox()
        self.combo_db.addItems(["CDMS", "JPL", "CDMS & JPL"])
        h2.addWidget(self.combo_db)
        form.addLayout(h2)
        
        h3 = QHBoxLayout()
        h3.addWidget(QLabel("Max Upper State Energy ($E_{up}$):"))
        self.spin_eup = QDoubleSpinBox()
        self.spin_eup.setRange(10, 5000)
        self.spin_eup.setValue(500.0)
        h3.addWidget(self.spin_eup)
        h3.addWidget(QLabel("K"))
        form.addLayout(h3)
        
        h4 = QHBoxLayout()
        h4.addWidget(QLabel("Species Filter:"))
        self.edit_species = QLineEdit()
        self.edit_species.setPlaceholderText("e.g., CS, CO, HCN (leave blank for all)")
        h4.addWidget(self.edit_species)
        form.addLayout(h4)

        layout.addLayout(form)
        layout.addSpacing(10)

        btn_layout = QHBoxLayout()
        self.btn_query = QPushButton("Query Splatalogue")
        self.btn_query.setStyleSheet("background-color: #27ae60; font-weight: bold; color: white;")
        self.btn_clear = QPushButton("Clear Existing Lines")
        self.btn_cancel = QPushButton("Close")

        self.btn_query.clicked.connect(self.accept)
        self.btn_clear.clicked.connect(self.clear_lines)
        self.btn_cancel.clicked.connect(self.reject)

        btn_layout.addWidget(self.btn_query)
        btn_layout.addWidget(self.btn_clear)
        btn_layout.addWidget(self.btn_cancel)
        layout.addLayout(btn_layout)

        self.action = 'cancel'

    def accept(self):
        self.action = 'query'
        super().accept()
        
    def clear_lines(self):
        self.action = 'clear'
        super().accept()


# ==============================================================================
# LINE SELECTION DIALOG
# ==============================================================================
class LineSelectionDialog(QDialog):
    """
    Dialog to display fetched Splatalogue lines and allow the user to select 
    which ones to plot on the spectrum.
    """
    def __init__(self, parent, parsed_data, v_sys):
        super().__init__(parent)
        self.setWindowTitle("Select Lines to Overlay")
        self.resize(900, 450)
        self.parsed_data = parsed_data
        self.v_sys = v_sys
        self.selected_rows = []

        layout = QVBoxLayout(self)
        info_lbl = QLabel(f"Found {len(parsed_data)} transitions. Select the ones you want to plot:")
        layout.addWidget(info_lbl)

        self.table = QTableWidget(len(parsed_data), 6)
        self.table.setHorizontalHeaderLabels(["Draw", "Molecule", "Name", "Transition QNs", "Rest Freq (GHz)", "E_up (K)"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents) 
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents) 
        self.table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeToContents)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)

        for i, row in enumerate(parsed_data):
            chk = QTableWidgetItem()
            chk.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            chk.setCheckState(Qt.Unchecked)
            self.table.setItem(i, 0, chk)

            form = str(row.get('formula', ''))
            sp = str(row.get('molecule_name', 'Unknown'))
            qn = str(row.get('QN', ''))
            fq = f"{row.get('restfreq', 0.0):.5f}"
            eu = f"{row.get('Eup(K)', 0.0):.1f}"

            self.table.setItem(i, 1, QTableWidgetItem(form))
            self.table.setItem(i, 2, QTableWidgetItem(sp))
            self.table.setItem(i, 3, QTableWidgetItem(qn))
            self.table.setItem(i, 4, QTableWidgetItem(fq))
            self.table.setItem(i, 5, QTableWidgetItem(eu))

        layout.addWidget(self.table)

        btn_layout = QHBoxLayout()
        btn_all = QPushButton("Select All")
        btn_species = QPushButton("Select all lines of chosen species")
        btn_none = QPushButton("Clear Selection")
        btn_draw = QPushButton("Plot Selected")
        btn_draw.setStyleSheet("background-color: #27ae60; font-weight: bold; color: white;")
        btn_cancel = QPushButton("Cancel")

        btn_all.clicked.connect(lambda: self.toggle_all(Qt.Checked))
        btn_none.clicked.connect(lambda: self.toggle_all(Qt.Unchecked))
        btn_species.clicked.connect(self.select_species)
        btn_draw.clicked.connect(self.accept)
        btn_cancel.clicked.connect(self.reject)

        btn_layout.addWidget(btn_all)
        btn_layout.addWidget(btn_species)
        btn_layout.addWidget(btn_none)
        btn_layout.addStretch()
        btn_layout.addWidget(btn_draw)
        btn_layout.addWidget(btn_cancel)
        layout.addLayout(btn_layout)

    def toggle_all(self, state):
        for i in range(self.table.rowCount()):
            self.table.item(i, 0).setCheckState(state)

    def select_species(self):
        target_molecules = set()
        for i in range(self.table.rowCount()):
            if self.table.item(i, 0).checkState() == Qt.Checked:
                target_molecules.add(self.table.item(i, 1).text())
                
        if not target_molecules:
            for item in self.table.selectedItems():
                row = item.row()
                target_molecules.add(self.table.item(row, 1).text())
                
        if target_molecules:
            for i in range(self.table.rowCount()):
                if self.table.item(i, 1).text() in target_molecules:
                    self.table.item(i, 0).setCheckState(Qt.Checked)

    def accept(self):
        self.selected_rows = []
        for i in range(self.table.rowCount()):
            if self.table.item(i, 0).checkState() == Qt.Checked:
                self.selected_rows.append(self.parsed_data[i])
        super().accept()


# ==============================================================================
# CONTOUR DIALOG
# ==============================================================================
class ContourDialog(QDialog):
    _LINE_STYLES = {'Solid': Qt.SolidLine, 'Dashed': Qt.DashLine, 'Dotted': Qt.DotLine}

    def __init__(self, parent, current_params, title, target_tab=None, target_id=None):
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
        if self.target_tab is not None and self.target_id is not None:
            self.target_tab.contour_params[self.target_id] = None
            if self.target_id == 'channel':
                self.target_tab.update_channel_map()
            else:
                self.target_tab.update_moment_maps()

# ==============================================================================
# REGION PROPERTIES DIALOG
# ==============================================================================
import pyqtgraph as pg

class RegionPropertiesDialog(QDialog):
    """
    Dialog to adjust the properties (center, size, P.A.) of Ellipse and Rectangle ROIs.
    Supports Image and World coordinates (if WCS is available).
    """
    def __init__(self, roi, explorer_tab, parent=None, roi_dict=None):
        super().__init__(parent or explorer_tab)
        self.roi = roi
        self.roi_dict = roi_dict
        self.tab = explorer_tab
        self.tool = roi_dict.get("type", roi_dict.get("tool", "Unknown")) if roi_dict else "Unknown"
        self.is_ellipse = isinstance(roi, pg.EllipseROI) or self.tool == "Ellipse"
        self.is_line = isinstance(roi, pg.LineSegmentROI) or self.tool == "Line"
        self.is_polyline = (isinstance(roi, pg.PolyLineROI) and not self.is_ellipse) or self.tool == "Custom Polygon"
        self.is_point = self.tool == "Point" or self.tool == "Point (Beam)"
        self.is_rect = (isinstance(roi, pg.RectROI) and not self.is_ellipse) or self.tool == "Rectangle"
        self.setWindowTitle("Region Properties")
        self.setMinimumWidth(400)
        
        self.wcs = self.tab.wcs_2d
        self.use_world = self.wcs is not None
        
        self.initUI()
        self.update_from_roi()
        self.roi.sigRegionChanged.connect(self.update_from_roi)

    def initUI(self):
        layout = QVBoxLayout(self)
        
        if self.roi_dict is not None:
            name_layout = QHBoxLayout()
            name_layout.addWidget(QLabel("Region Name:"))
            self.edit_name = QLineEdit()
            self.edit_name.setText(self.roi_dict.get("name", ""))
            name_layout.addWidget(self.edit_name)
            layout.addLayout(name_layout)
            self.edit_name.editingFinished.connect(self.apply_name)
            
        if self.is_polyline:
            layout.addStretch()
            self._updating = False
            return
            
        # Coordinate selection
        coord_layout = QHBoxLayout()
        coord_layout.addWidget(QLabel("Coordinate:"))
        self.rad_image = QRadioButton("Image")
        self.rad_world = QRadioButton("World")
        self.rad_image.setStyleSheet("color: white;")
        self.rad_world.setStyleSheet("color: white;")
        coord_layout.addWidget(self.rad_image)
        coord_layout.addWidget(self.rad_world)
        coord_layout.addStretch()
        
        if self.use_world:
            self.rad_world.setChecked(True)
        else:
            self.rad_image.setChecked(True)
            self.rad_world.setEnabled(False)
            
        self.rad_image.toggled.connect(self.toggle_coord_mode)
        self.rad_world.toggled.connect(self.toggle_coord_mode)
        layout.addLayout(coord_layout)
        
        # Center
        center_layout = QHBoxLayout()
        center_layout.addWidget(QLabel("Center:"))
        self.edit_cx = QLineEdit()
        self.edit_cy = QLineEdit()
        center_layout.addWidget(self.edit_cx)
        center_layout.addWidget(self.edit_cy)
        layout.addLayout(center_layout)
        
        self.lbl_center_img = QLabel("Image: ()")
        self.lbl_center_img.setStyleSheet("font-family: monospace; font-size: 11px; color: #aaa;")
        layout.addWidget(self.lbl_center_img)
        

        if self.is_line:
            # Add Start/End point fields for Line
            line_layout = QVBoxLayout()
            p1_layout = QHBoxLayout()
            p1_layout.addWidget(QLabel("Start Point:"))
            self.edit_p1x = QLineEdit(); self.edit_p1y = QLineEdit()
            p1_layout.addWidget(self.edit_p1x); p1_layout.addWidget(self.edit_p1y)
            line_layout.addLayout(p1_layout)
            
            p2_layout = QHBoxLayout()
            p2_layout.addWidget(QLabel("End Point:"))
            self.edit_p2x = QLineEdit(); self.edit_p2y = QLineEdit()
            p2_layout.addWidget(self.edit_p2x); p2_layout.addWidget(self.edit_p2y)
            line_layout.addLayout(p2_layout)
            layout.addLayout(line_layout)
            
            self.edit_p1x.editingFinished.connect(self.apply_to_line)
            self.edit_p1y.editingFinished.connect(self.apply_to_line)
            self.edit_p2x.editingFinished.connect(self.apply_to_line)
            self.edit_p2y.editingFinished.connect(self.apply_to_line)
            
        # Size / Semi-axes
        if not self.is_line:
            size_layout = QHBoxLayout()
            size_label = "Semi-axes:" if self.is_ellipse else "Size:"
            size_layout.addWidget(QLabel(size_label))
            self.edit_w = QLineEdit()
            self.edit_h = QLineEdit()
            if self.is_point:
                self.edit_w.setReadOnly(True)
                self.edit_h.setReadOnly(True)
            size_layout.addWidget(self.edit_w)
            size_layout.addWidget(self.edit_h)
            layout.addLayout(size_layout)
            
            self.lbl_size_img = QLabel("Image: ()")
            self.lbl_size_img.setStyleSheet("font-family: monospace; font-size: 11px; color: #aaa;")
            layout.addWidget(self.lbl_size_img)
        
        # Bottom-left and Top-right for Rectangle
        if self.is_rect:
            bl_layout = QHBoxLayout()
            bl_layout.addWidget(QLabel("Bottom-left:"))
            self.edit_bl_x = QLineEdit()
            self.edit_bl_y = QLineEdit()
            bl_layout.addWidget(self.edit_bl_x)
            bl_layout.addWidget(self.edit_bl_y)
            layout.addLayout(bl_layout)
            
            self.lbl_bl_img = QLabel("Image: ()")
            self.lbl_bl_img.setStyleSheet("font-family: monospace; font-size: 11px; color: #aaa;")
            layout.addWidget(self.lbl_bl_img)
            
            tr_layout = QHBoxLayout()
            tr_layout.addWidget(QLabel("Top-right:"))
            self.edit_tr_x = QLineEdit()
            self.edit_tr_y = QLineEdit()
            tr_layout.addWidget(self.edit_tr_x)
            tr_layout.addWidget(self.edit_tr_y)
            layout.addLayout(tr_layout)
            
            self.lbl_tr_img = QLabel("Image: ()")
            self.lbl_tr_img.setStyleSheet("font-family: monospace; font-size: 11px; color: #aaa;")
            layout.addWidget(self.lbl_tr_img)
        
        # P.A. (deg)
        pa_layout = QHBoxLayout()
        pa_layout.addWidget(QLabel("P.A. (deg):"))
        self.edit_pa = QLineEdit()
        if self.is_point:
            self.edit_pa.setReadOnly(True)
        pa_layout.addWidget(self.edit_pa)
        pa_layout.addStretch()
        layout.addLayout(pa_layout)

        if self.roi_dict and self.roi_dict.get("tool") == "PV Cut":
            width_layout = QHBoxLayout()
            width_layout.addWidget(QLabel("Width (pixels):"))
            self.spin_width = QSpinBox()
            self.spin_width.setMinimum(1)
            self.spin_width.setMaximum(101)
            self.spin_width.setSingleStep(2)
            self.spin_width.setToolTip("Number of pixels averaged perpendicular to the cut")
            pv_cut_dict = self.roi_dict.get("pv_cut_dict", {})
            w_val = pv_cut_dict.get("width", 1)
            if w_val % 2 == 0:
                w_val += 1
            self.spin_width.setValue(w_val)
            self.spin_width.valueChanged.connect(self.apply_width)
            width_layout.addWidget(self.spin_width)
            width_layout.addStretch()
            layout.addLayout(width_layout)
        
        # Connect editing finished
        self.edit_cx.editingFinished.connect(self.apply_to_roi)
        self.edit_cy.editingFinished.connect(self.apply_to_roi)
        if not self.is_point and not self.is_line:
            self.edit_w.editingFinished.connect(self.apply_to_roi)
            self.edit_h.editingFinished.connect(self.apply_to_roi)
        if not self.is_point:
            self.edit_pa.editingFinished.connect(self.apply_to_roi)
        if self.is_rect:
            self.edit_bl_x.editingFinished.connect(self.apply_to_roi_from_bounds)
            self.edit_bl_y.editingFinished.connect(self.apply_to_roi_from_bounds)
            self.edit_tr_x.editingFinished.connect(self.apply_to_roi_from_bounds)
            self.edit_tr_y.editingFinished.connect(self.apply_to_roi_from_bounds)
            
        layout.addStretch()
        self._updating = False

    def toggle_coord_mode(self):
        self.use_world = self.rad_world.isChecked()
        self.update_from_roi()

    def update_from_roi(self):
        if self._updating: return
        self._updating = True
        try:
            if self.is_polyline: return
            pos = self.roi.pos()
            size = self.roi.size()
            from src.gui.custom import get_casa_pa
            angle = get_casa_pa(self.roi)
            
            w, h = size.x(), size.y()
            cx = pos.x() + w / 2.0
            cy = pos.y() + h / 2.0
            
            from astropy.coordinates import SkyCoord
            import astropy.units as u

            if self.is_line:
                # Update line endpoints
                hndls = self.roi.getHandles()
                if len(hndls) >= 2:
                    p1_p = self.roi.mapToParent(hndls[0].pos())
                    p2_p = self.roi.mapToParent(hndls[1].pos())
                    
                    # Calculate true center as midpoint of endpoints
                    cx = (p1_p.x() + p2_p.x()) / 2.0
                    cy = (p1_p.y() + p2_p.y()) / 2.0
                    
                    if self.use_world and self.wcs:
                        px1 = (self.tab.nx / 2) - (p1_p.x() / self.tab.pix_scale_arcsec)
                        py1 = (p1_p.y() / self.tab.pix_scale_arcsec) + (self.tab.ny / 2)
                        ra1, dec1 = self.wcs.pixel_to_world_values(px1, py1)
                        sc1 = SkyCoord(ra1, dec1, unit='deg')
                        self.edit_p1x.setText(sc1.ra.to_string(unit=u.hour, sep=':', precision=6))
                        self.edit_p1y.setText(sc1.dec.to_string(unit=u.deg, sep=':', precision=6))
                        
                        px2 = (self.tab.nx / 2) - (p2_p.x() / self.tab.pix_scale_arcsec)
                        py2 = (p2_p.y() / self.tab.pix_scale_arcsec) + (self.tab.ny / 2)
                        ra2, dec2 = self.wcs.pixel_to_world_values(px2, py2)
                        sc2 = SkyCoord(ra2, dec2, unit='deg')
                        self.edit_p2x.setText(sc2.ra.to_string(unit=u.hour, sep=':', precision=6))
                        self.edit_p2y.setText(sc2.dec.to_string(unit=u.deg, sep=':', precision=6))
                        
                        # Calculate astronomical PA (East of North)
                        angle = sc1.position_angle(sc2).deg
                    else:
                        self.edit_p1x.setText(f"{p1_p.x():.5f}")
                        self.edit_p1y.setText(f"{p1_p.y():.5f}")
                        self.edit_p2x.setText(f"{p2_p.x():.5f}")
                        self.edit_p2y.setText(f"{p2_p.y():.5f}")
                        
                        # Calculate PA in image coordinates (North UP, East RIGHT)
                        import numpy as np
                        angle = np.degrees(np.arctan2(p2_p.x() - p1_p.x(), p2_p.y() - p1_p.y())) % 360

            self.edit_pa.setText(f"{angle:.5f}")
            
            if self.use_world and self.wcs:
                px = (self.tab.nx / 2) - (cx / self.tab.pix_scale_arcsec)
                py = (cy / self.tab.pix_scale_arcsec) + (self.tab.ny / 2)
                ra, dec = self.wcs.pixel_to_world_values(px, py)
                sc = SkyCoord(ra, dec, unit='deg')
                self.edit_cx.setText(sc.ra.to_string(unit=u.hour, sep=':', precision=7))
                self.edit_cy.setText(sc.dec.to_string(unit=u.deg, sep=':', precision=6))
                
                if not self.is_line:
                    self.edit_w.setText(f"{w:.5f}\"")
                    self.edit_h.setText(f"{h:.5f}\"")
                    self.lbl_size_img.setText(f"Image: ({w / self.tab.pix_scale_arcsec:.3f} px, {h / self.tab.pix_scale_arcsec:.3f} px)")
                
                self.lbl_center_img.setText(f"Image: ({px:.3f} px, {py:.3f} px)")
                
                if self.is_rect:
                    bl_px = (self.tab.nx / 2) - (pos.x() / self.tab.pix_scale_arcsec)
                    bl_py = (pos.y() / self.tab.pix_scale_arcsec) + (self.tab.ny / 2)
                    bl_ra, bl_dec = self.wcs.pixel_to_world_values(bl_px, bl_py)
                    sc_bl = SkyCoord(bl_ra, bl_dec, unit='deg')
                    self.edit_bl_x.setText(sc_bl.ra.to_string(unit=u.hour, sep=':', precision=6))
                    self.edit_bl_y.setText(sc_bl.dec.to_string(unit=u.deg, sep=':', precision=6))
                    self.lbl_bl_img.setText(f"Image: ({bl_px:.3f} px, {bl_py:.3f} px)")
                    
                    tr_x = pos.x() + w
                    tr_y = pos.y() + h
                    tr_px = (self.tab.nx / 2) - (tr_x / self.tab.pix_scale_arcsec)
                    tr_py = (tr_y / self.tab.pix_scale_arcsec) + (self.tab.ny / 2)
                    tr_ra, tr_dec = self.wcs.pixel_to_world_values(tr_px, tr_py)
                    sc_tr = SkyCoord(tr_ra, tr_dec, unit='deg')
                    self.edit_tr_x.setText(sc_tr.ra.to_string(unit=u.hour, sep=':', precision=6))
                    self.edit_tr_y.setText(sc_tr.dec.to_string(unit=u.deg, sep=':', precision=6))
                    self.lbl_tr_img.setText(f"Image: ({tr_px:.3f} px, {tr_py:.3f} px)")

            else:
                self.edit_cx.setText(f"{cx:.5f}")
                self.edit_cy.setText(f"{cy:.5f}")
                if not self.is_line:
                    self.edit_w.setText(f"{w:.5f}")
                    self.edit_h.setText(f"{h:.5f}")
                    self.lbl_size_img.setText(f"Image: ({w:.3f} arcsec, {h:.3f} arcsec)")
                self.lbl_center_img.setText(f"Image: ({cx:.3f} arcsec, {cy:.3f} arcsec)")
                
                if self.is_rect:
                    self.edit_bl_x.setText(f"{pos.x():.5f}")
                    self.edit_bl_y.setText(f"{pos.y():.5f}")
                    self.lbl_bl_img.setText("")
                    self.edit_tr_x.setText(f"{pos.x()+w:.5f}")
                    self.edit_tr_y.setText(f"{pos.y()+h:.5f}")
                    self.lbl_tr_img.setText("")
                    
        finally:
            self._updating = False

    def _parse_coord(self, val_str, is_ra=True):
        from astropy.coordinates import Angle
        import astropy.units as u
        if ':' in val_str:
            return Angle(val_str, unit=u.hourangle if is_ra else u.deg).deg
        return float(val_str)

    def apply_to_roi(self):
        if self._updating: return
        self._updating = True
        try:
            cx_str = self.edit_cx.text()
            cy_str = self.edit_cy.text()
            
            if self.use_world and self.wcs:
                ra = self._parse_coord(cx_str, is_ra=True)
                dec = self._parse_coord(cy_str, is_ra=False)
                px, py = self.wcs.world_to_pixel_values(ra, dec)
                cx = (self.tab.nx / 2 - px) * self.tab.pix_scale_arcsec
                cy = (py - self.tab.ny / 2) * self.tab.pix_scale_arcsec
            else:
                cx, cy = float(cx_str), float(cy_str)
                
            if self.is_point:
                w, h = self.roi.size().x(), self.roi.size().y()
                pos_x = cx - w / 2.0
                pos_y = cy - h / 2.0
                self.roi.blockSignals(True)
                self.roi.setPos([pos_x, pos_y])
                self.roi.blockSignals(False)
            elif self.is_line:
                hndls = self.roi.getHandles()
                if len(hndls) >= 2:
                    p1_p = self.roi.mapToParent(hndls[0].pos())
                    p2_p = self.roi.mapToParent(hndls[1].pos())
                    
                    import numpy as np
                    L = np.hypot(p2_p.x() - p1_p.x(), p2_p.y() - p1_p.y())
                    pa_val = float(self.edit_pa.text())
                    
                    rad = np.radians(pa_val)
                    dx = (L / 2.0) * np.sin(rad)
                    dy = (L / 2.0) * np.cos(rad)
                    
                    new_p1 = [cx - dx, cy - dy]
                    new_p2 = [cx + dx, cy + dy]
                    
                    self.roi.blockSignals(True)
                    self.roi.movePoint(hndls[0], new_p1, finish=False)
                    self.roi.movePoint(hndls[1], new_p2, finish=True)
                    self.roi.blockSignals(False)
            else:
                w_val = float(self.edit_w.text().replace('"', ''))
                h_val = float(self.edit_h.text().replace('"', ''))
                pa_val = float(self.edit_pa.text())
                
                pos_x = cx - w_val / 2.0
                pos_y = cy - h_val / 2.0
                
                self.roi.blockSignals(True)
                self.roi.setPos([pos_x, pos_y])
                self.roi.setSize([w_val, h_val])
                from src.gui.custom import get_pyqt_angle
                self.roi.setAngle(get_pyqt_angle(pa_val), center=[0.5, 0.5])
                self.roi.blockSignals(False)
                
            self.roi.sigRegionChanged.emit(self.roi)
        except Exception:
            pass
        finally:
            self._updating = False
            self.update_from_roi()

    def apply_to_roi_from_bounds(self):
        if self._updating: return
        self._updating = True
        try:
            bl_x_str = self.edit_bl_x.text()
            bl_y_str = self.edit_bl_y.text()
            tr_x_str = self.edit_tr_x.text()
            tr_y_str = self.edit_tr_y.text()
            
            if self.use_world and self.wcs:
                ra_bl = self._parse_coord(bl_x_str, is_ra=True)
                dec_bl = self._parse_coord(bl_y_str, is_ra=False)
                px_bl, py_bl = self.wcs.world_to_pixel_values(ra_bl, dec_bl)
                bl_x = (self.tab.nx / 2 - px_bl) * self.tab.pix_scale_arcsec
                bl_y = (py_bl - self.tab.ny / 2) * self.tab.pix_scale_arcsec
                
                ra_tr = self._parse_coord(tr_x_str, is_ra=True)
                dec_tr = self._parse_coord(tr_y_str, is_ra=False)
                px_tr, py_tr = self.wcs.world_to_pixel_values(ra_tr, dec_tr)
                tr_x = (self.tab.nx / 2 - px_tr) * self.tab.pix_scale_arcsec
                tr_y = (py_tr - self.tab.ny / 2) * self.tab.pix_scale_arcsec
            else:
                bl_x, bl_y = float(bl_x_str), float(bl_y_str)
                tr_x, tr_y = float(tr_x_str), float(tr_y_str)
                
            w = tr_x - bl_x
            h = tr_y - bl_y
            
            self.roi.blockSignals(True)
            self.roi.setPos([bl_x, bl_y])
            self.roi.setSize([w, h])
            self.roi.blockSignals(False)
            self.roi.sigRegionChanged.emit(self.roi)
        except ValueError:
            pass
        finally:
            self._updating = False
            self.update_from_roi()

    def apply_name(self):
        if self.roi_dict:
            new_name = self.edit_name.text().strip()
            if new_name:
                old_name = self.roi_dict["name"]
                if new_name != old_name:
                    self.roi_dict["name"] = new_name
                    is_pv_cut = (self.roi_dict.get("tool") == "PV Cut")

                    if "checkbox" in self.roi_dict:
                        self.roi_dict["checkbox"].setText(new_name)

                    if hasattr(self.tab, 'combo_spatial_regions'):
                        idx = self.tab.combo_spatial_regions.findText(old_name)
                        if idx >= 0:
                            self.tab.combo_spatial_regions.setItemText(idx, new_name)

                    if old_name in self.tab.spectrum_curves:
                        self.tab.spectrum_curves[new_name] = self.tab.spectrum_curves.pop(old_name)
                        if hasattr(self.tab.spectrum_curves[new_name], 'opts'):
                            self.tab.spectrum_curves[new_name].opts['name'] = new_name
                    if hasattr(self.tab, 'spectrum_curves_smooth') and old_name in self.tab.spectrum_curves_smooth:
                        self.tab.spectrum_curves_smooth[new_name] = self.tab.spectrum_curves_smooth.pop(old_name)
                        if hasattr(self.tab.spectrum_curves_smooth[new_name], 'opts'):
                            self.tab.spectrum_curves_smooth[new_name].opts['name'] = new_name

                    if "text_item" in self.roi_dict:
                        self.roi_dict["text_item"].setText(new_name)

                    if is_pv_cut:
                        self.tab.refresh_all_pv_cut_combos()
                    else:
                        self.tab.update_spectrum()
                        self.tab.update_spectrum_region_calc()
                        self.tab.refresh_spectral_stats_apertures()

    def apply_width(self, value):
        if value % 2 == 0:
            value += 1
            self.spin_width.blockSignals(True)
            self.spin_width.setValue(value)
            self.spin_width.blockSignals(False)
        pv_cut_dict = self.roi_dict.get("pv_cut_dict")
        if pv_cut_dict is not None:
            if pv_cut_dict.get("width") != value:
                pv_cut_dict["width"] = value
                if "update_annotations" in pv_cut_dict:
                    pv_cut_dict["update_annotations"]()
                self.tab.update_moment_maps()

    def apply_to_line(self):
        if self._updating: return
        self._updating = True
        try:
            from astropy.coordinates import SkyCoord
            import astropy.units as u
            
            p1x_str = self.edit_p1x.text()
            p1y_str = self.edit_p1y.text()
            p2x_str = self.edit_p2x.text()
            p2y_str = self.edit_p2y.text()
            
            if self.use_world and self.wcs:
                ra1 = self._parse_coord(p1x_str, is_ra=True)
                dec1 = self._parse_coord(p1y_str, is_ra=False)
                px1, py1 = self.wcs.world_to_pixel_values(ra1, dec1)
                x1 = (self.tab.nx / 2 - px1) * self.tab.pix_scale_arcsec
                y1 = (py1 - self.tab.ny / 2) * self.tab.pix_scale_arcsec
                
                ra2 = self._parse_coord(p2x_str, is_ra=True)
                dec2 = self._parse_coord(p2y_str, is_ra=False)
                px2, py2 = self.wcs.world_to_pixel_values(ra2, dec2)
                x2 = (self.tab.nx / 2 - px2) * self.tab.pix_scale_arcsec
                y2 = (py2 - self.tab.ny / 2) * self.tab.pix_scale_arcsec
            else:
                x1, y1 = float(p1x_str), float(p1y_str)
                x2, y2 = float(p2x_str), float(p2y_str)
                
            handles = self.roi.getHandles()
            self.roi.blockSignals(True)
            self.roi.movePoint(handles[0], [x1, y1], finish=False)
            self.roi.movePoint(handles[1], [x2, y2], finish=True)
            self.roi.blockSignals(False)
            self.roi.sigRegionChanged.emit(self.roi)
        except Exception:
            pass
        finally:
            self._updating = False
            self.update_from_roi()


# ==============================================================================
# SPECTRAL SMOOTHING DIALOG
# ==============================================================================
class SpectralSmoothingDialog(QDialog):
    """
    Dialog to select spectral smoothing method and parameters.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Spectral Smoothing")
        self.setMinimumWidth(350)
        self.initUI()
        
    def initUI(self):
        layout = QVBoxLayout(self)
        
        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel("Method:"))
        self.combo_method = QComboBox()
        self.combo_method.addItems(["Boxcar", "Gaussian", "Savitzky-Golay"])
        method_layout.addWidget(self.combo_method)
        layout.addLayout(method_layout)
        
        self.param_stack = QStackedWidget()
        
        self.widget_boxcar = QWidget()
        boxcar_layout = QHBoxLayout(self.widget_boxcar)
        boxcar_layout.setContentsMargins(0, 0, 0, 0)
        boxcar_layout.addWidget(QLabel("Window Size (pixels):"))
        self.spin_boxcar_w = QSpinBox()
        self.spin_boxcar_w.setRange(2, 1000)
        self.spin_boxcar_w.setValue(3)
        boxcar_layout.addWidget(self.spin_boxcar_w)
        boxcar_layout.addStretch()
        self.param_stack.addWidget(self.widget_boxcar)
        
        self.widget_gauss = QWidget()
        gauss_layout = QHBoxLayout(self.widget_gauss)
        gauss_layout.setContentsMargins(0, 0, 0, 0)
        gauss_layout.addWidget(QLabel("Sigma (pixels):"))
        self.spin_gauss_sigma = QDoubleSpinBox()
        self.spin_gauss_sigma.setRange(0.1, 500.0)
        self.spin_gauss_sigma.setValue(1.0)
        self.spin_gauss_sigma.setSingleStep(0.5)
        gauss_layout.addWidget(self.spin_gauss_sigma)
        gauss_layout.addStretch()
        self.param_stack.addWidget(self.widget_gauss)
        
        self.widget_savgol = QWidget()
        savgol_layout = QHBoxLayout(self.widget_savgol)
        savgol_layout.setContentsMargins(0, 0, 0, 0)
        savgol_layout.addWidget(QLabel("Window Size:"))
        self.spin_savgol_w = QSpinBox()
        self.spin_savgol_w.setRange(3, 999)
        self.spin_savgol_w.setSingleStep(2)
        self.spin_savgol_w.setValue(5)
        savgol_layout.addWidget(self.spin_savgol_w)
        
        savgol_layout.addWidget(QLabel("Poly Order:"))
        self.spin_savgol_p = QSpinBox()
        self.spin_savgol_p.setRange(1, 10)
        self.spin_savgol_p.setValue(2)
        savgol_layout.addWidget(self.spin_savgol_p)
        savgol_layout.addStretch()
        self.param_stack.addWidget(self.widget_savgol)
        
        layout.addWidget(self.param_stack)
        
        self.combo_method.currentIndexChanged.connect(self.param_stack.setCurrentIndex)
        self.spin_savgol_w.valueChanged.connect(self.validate_savgol)
        
        btn_layout = QHBoxLayout()
        btn_apply = QPushButton("Apply")
        btn_apply.setStyleSheet("background-color: #27ae60; font-weight: bold; color: white;")
        btn_cancel = QPushButton("Cancel")
        btn_apply.clicked.connect(self.accept)
        btn_cancel.clicked.connect(self.reject)
        btn_layout.addStretch()
        btn_layout.addWidget(btn_apply)
        btn_layout.addWidget(btn_cancel)
        layout.addLayout(btn_layout)
        
    def validate_savgol(self, value):
        if value % 2 == 0:
            self.spin_savgol_w.setValue(value + 1)
            
    def get_params(self):
        method = self.combo_method.currentText()
        if method == "Boxcar":
            return {"method": "boxcar", "window": self.spin_boxcar_w.value()}
        elif method == "Gaussian":
            return {"method": "gaussian", "sigma": self.spin_gauss_sigma.value()}
        elif method == "Savitzky-Golay":
            w = self.spin_savgol_w.value()
            p = self.spin_savgol_p.value()
            if p >= w:
                p = w - 1
            return {"method": "savgol", "window": w, "polyorder": p}
        return None


# ==============================================================================
# CHANNEL GRID DIALOG
# ==============================================================================
class ChannelGridDialog(QDialog):
    """
    Dialog to display a grid of channel maps corresponding to the selected
    velocity range in the spectrum panel.
    """
    def __init__(self, explorer_tab, parent=None):
        super().__init__(parent or explorer_tab)
        self.tab = explorer_tab
        self.setWindowTitle("Channel Grid")
        self.setMinimumSize(800, 600)
        
        # Set dark theme
        self.setStyleSheet("background-color: #1a1a1a; color: #e0e0e0;")
        
        layout = QHBoxLayout(self)
        
        # Left side: Scroll area + hover label
        left_layout = QVBoxLayout()
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setStyleSheet("QScrollArea { border: none; }")
        self.grid_widget = pg.GraphicsLayoutWidget()
        self.grid_widget.setBackground('#1a1a1a')
        self.scroll.setWidget(self.grid_widget)
        left_layout.addWidget(self.scroll, stretch=1)
        
        self.lbl_hover = QLabel("Hover over a tile to see details")
        self.lbl_hover.setStyleSheet("font-family: monospace; font-size: 11.5px; color: #aaa; padding: 5px;")
        left_layout.addWidget(self.lbl_hover)
        
        layout.addLayout(left_layout, stretch=4)
        
        # Right side: Colorbar (Histogram) + Export Button
        right_layout = QVBoxLayout()
        self.hist = pg.HistogramLUTWidget()
        right_layout.addWidget(self.hist, stretch=1)
        
        btn_layout = QHBoxLayout()
        
        from PyQt5.QtWidgets import QComboBox
        self.combo_cmap = QComboBox()
        self.combo_cmap.addItems(['Turbo', 'Inferno', 'Viridis', 'Plasma', 'Magma', 'Cubehelix', 'Grey'])
        self.combo_cmap.currentTextChanged.connect(self.change_cmap)
        self.combo_cmap.setStyleSheet("background-color: #34495e; color: white; padding: 5px;")
        btn_layout.addWidget(self.combo_cmap)
        
        self.btn_reset_zoom = QPushButton("Reset zoom")
        self.btn_reset_zoom.clicked.connect(self.reset_zoom)
        self.btn_reset_zoom.setStyleSheet("background-color: #34495e; color: white; padding: 5px;")
        btn_layout.addWidget(self.btn_reset_zoom)
        
        self.btn_export = QPushButton("Export to PDF")
        self.btn_export.clicked.connect(self.export_to_pdf)
        self.btn_export.setStyleSheet("background-color: #2c3e50; color: white; padding: 5px;")
        btn_layout.addWidget(self.btn_export)
        
        right_layout.addLayout(btn_layout)
        
        layout.addLayout(right_layout, stretch=1)
        
        self.images = []
        self.view_boxes = []
        self.pos_tup = None
        self.scale_tup = None
        self.grid_widget.scene().sigMouseMoved.connect(self.on_mouse_moved)
        
        # Connect histogram signals to update all images
        self.hist.sigLevelsChanged.connect(self.on_hist_levels_changed)
        self.hist.sigLookupTableChanged.connect(self.on_hist_lut_changed)
        
        self.update_grid()
        
    def update_grid(self):
        # Clear existing grid
        self.grid_widget.clear()
        self.images.clear()
        self.view_boxes.clear()
        
        cube, v_axis, minX, maxX = self.tab.get_velocity_subset(use_full_range=False)
        if cube is None or len(v_axis) == 0:
            self.grid_widget.addLabel("No channels in selected range.", col=0, row=0)
            return
            
        n_channels = len(v_axis)
        cols = int(np.ceil(np.sqrt(n_channels)))
        rows = int(np.ceil(n_channels / cols))
        
        # Set a fixed size. Without axes, grid size matches data aspect ratio exactly. Add 2px for margin.
        base_w = 200
        base_h = int(200 * (self.tab.ny / self.tab.nx)) if self.tab.nx > 0 else 200
        self.grid_widget.setFixedSize(int(cols * base_w) + 2, int(rows * base_h) + 2)
        
        # Get current histogram state from main channel map
        main_hist = self.tab.view_channel.ui.histogram
        levels = main_hist.getLevels()
        
        # Update dialog's histogram gradient to match main one
        self.hist.gradient.restoreState(main_hist.gradient.saveState())
        lut = self.hist.gradient.getLookupTable(256)
        
        # Adapt histogram range to be reasonable (0 to max positive data)
        cube_max = float(np.nanmax(cube)) if np.nanmax(cube) > 0 else 1.0
        new_levels = [0, cube_max]
        
        pos_tup = ((self.tab.nx / 2) * self.tab.pix_scale_arcsec, -(self.tab.ny / 2) * self.tab.pix_scale_arcsec)
        scale_tup = (-self.tab.pix_scale_arcsec, self.tab.pix_scale_arcsec)
        
        self.pos_tup = pos_tup
        self.scale_tup = scale_tup
        
        self.grid_widget.ci.layout.setSpacing(0)
        self.grid_widget.ci.layout.setContentsMargins(1, 1, 1, 1)
        
        first_plot = None
        for idx in range(n_channels):
            r, c = divmod(idx, cols)
            
            p = self.grid_widget.addPlot(row=r, col=c)
            p.setAspectLocked(True)
            p.invertY(False)
            p.invertX(True)
            p.hideButtons()
            p.layout.setContentsMargins(0, 0, 0, 0)
            
            vb = p.getViewBox()
            vb.setDefaultPadding(0.0)
            vb.setBorder(pg.mkPen(color='w', width=1))
            
            if first_plot is None:
                first_plot = p
            else:
                p.setXLink(first_plot)
                p.setYLink(first_plot)
                
            # Hide all axes
            p.hideAxis('left')
            p.hideAxis('bottom')
            p.hideAxis('right')
            p.hideAxis('top')
            
            img = pg.ImageItem()
            p.addItem(img)
            
            # Set data
            img.setImage(cube[idx, :, :], scale=scale_tup, pos=pos_tup)
            img.setLevels(new_levels)
            img.setLookupTable(lut)
            
            # Add velocity text at top left corner of the ViewBox (screen coordinates)
            vel_text = pg.TextItem(f"{v_axis[idx]:.2f} km/s", color='w', anchor=(0, 0), fill=pg.mkBrush(0, 0, 0, 150))
            vel_text.setParentItem(p.getViewBox())
            vel_text.setPos(5, 5)
            vel_text.setZValue(100)
            
            self.images.append(img)
            self.view_boxes.append({'vb': vb, 'img': img, 'vel': v_axis[idx], 'data': cube[idx, :, :]})
            
        # Link the histogram to the first image to make it active
        if self.images:
            self.hist.setImageItem(self.images[0])
            # Set the levels and visible bounds again because setImageItem overrides them
            self.hist.setLevels(new_levels[0], new_levels[1])
            self.hist.setHistogramRange(0, cube_max)
            
            # Sync the colormap with the dropdown selection (which includes fallbacks for non-native ones like cubehelix)
            self.change_cmap(self.combo_cmap.currentText())
            
            # Delay the autoRange reset to ensure layout has fully updated after adding/removing tiles
            pg.QtCore.QTimer.singleShot(100, self.reset_zoom)
                
    def change_cmap(self, cmap_name):
        cmap_name = cmap_name.lower()
        try:
            self.hist.gradient.loadPreset(cmap_name)
        except KeyError:
            import matplotlib.pyplot as plt
            import numpy as np
            cmap = plt.get_cmap(cmap_name)
            pos = np.linspace(0.0, 1.0, 64)
            colors = cmap(pos) * 255
            self.hist.gradient.setColorMap(pg.ColorMap(pos, colors.astype(np.ubyte)))
            
        self.on_hist_lut_changed()

    def reset_zoom(self):
        if self.view_boxes:
            vb = self.view_boxes[0]['vb']
            vb.autoRange(padding=0)
            
    def on_mouse_moved(self, pos):
        if not hasattr(self, 'view_boxes') or not self.view_boxes:
            return
            
        for item in self.view_boxes:
            vb = item['vb']
            if vb.sceneBoundingRect().contains(pos):
                mouse_point = vb.mapSceneToView(pos)
                x = mouse_point.x()
                y = mouse_point.y()
                
                img = item['img']
                local_pos = img.mapFromView(mouse_point)
                px = int(local_pos.x())
                py = int(local_pos.y())
                
                data = item['data']
                if 0 <= px < data.shape[0] and 0 <= py < data.shape[1]:
                    val = data[px, py]
                    # Calculate offsets manually from the local image pixel coordinate to guarantee
                    # perfectly zeroed centers regardless of ViewBox projection linking
                    ra_offset = (data.shape[0] / 2.0 - local_pos.x()) * self.tab.pix_scale_arcsec
                    dec_offset = (local_pos.y() - data.shape[1] / 2.0) * self.tab.pix_scale_arcsec
                    
                    self.lbl_hover.setText(f"{item['vel']:.2f} km/s  |  ({px}, {py})  |  RA: {ra_offset:.2f}\"  |  DEC: {dec_offset:.2f}\"  |  {val:.4e} {self.tab.display_unit}")
                    self.lbl_hover.setStyleSheet("font-family: monospace; font-size: 11.5px; color: #3498db; font-weight: bold; padding: 5px;")
                else:
                    self.lbl_hover.setText(f"{item['vel']:.2f} km/s  |  RA: --  |  DEC: --  |  --")
                    self.lbl_hover.setStyleSheet("font-family: monospace; font-size: 11.5px; color: #aaa; padding: 5px;")
                return
                
        # If we reach here, we are not hovering over any valid tile
        self.lbl_hover.setText("Hover over a tile to see details")
        self.lbl_hover.setStyleSheet("font-family: monospace; font-size: 11.5px; color: #aaa; padding: 5px;")
                
    def on_hist_levels_changed(self):
        levels = self.hist.getLevels()
        for img in self.images:
            img.setLevels(levels)
            
    def on_hist_lut_changed(self):
        lut = self.hist.gradient.getLookupTable(256)
        for img in self.images:
            img.setLookupTable(lut)
            
    def update_from_main_hist(self):
        main_hist = self.tab.view_channel.ui.histogram
        levels = main_hist.getLevels()
        self.hist.setLevels(levels[0], levels[1])
            
    def update_from_main_lut(self):
        main_hist = self.tab.view_channel.ui.histogram
        self.hist.gradient.restoreState(main_hist.gradient.saveState())
        
    def export_to_pdf(self):
        filename, _ = QFileDialog.getSaveFileName(self, "Save PDF", "channel_grid.pdf", "PDF Files (*.pdf)")
        if filename:
            try:
                cube, v_axis, minX, maxX = self.tab.get_velocity_subset(use_full_range=False)
                if cube is None or len(v_axis) == 0:
                    return
                    
                n_channels = len(v_axis)
                cols = int(np.ceil(np.sqrt(n_channels)))
                rows = int(np.ceil(n_channels / cols))
                
                fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3), squeeze=False)
                
                levels = self.hist.getLevels()
                cmap_name = self.tab.parent_window.current_cmap
                
                extent = [self.tab.nx/2 * self.tab.pix_scale_arcsec, -self.tab.nx/2 * self.tab.pix_scale_arcsec, 
                          -self.tab.ny/2 * self.tab.pix_scale_arcsec, self.tab.ny/2 * self.tab.pix_scale_arcsec]
                          
                for idx in range(rows * cols):
                    r = idx // cols
                    c = idx % cols
                    ax = axes[r, c]
                    
                    if idx < n_channels:
                        plot_data = cube[idx, :, :].T
                        
                        im = ax.imshow(plot_data, origin='lower', cmap=cmap_name, 
                                       vmin=levels[0], vmax=levels[1], extent=extent)
                                       
                        ax.text(0.05, 0.95, f"{v_axis[idx]:.2f} km/s", transform=ax.transAxes,
                                color='white', verticalalignment='top', bbox=dict(facecolor='black', alpha=0.5, pad=1))
                                
                        if c == 0:
                            ax.set_ylabel('DEC offset (arcsec)')
                        else:
                            ax.set_yticklabels([])
                            
                        if r == rows - 1 or idx + cols >= n_channels:
                            ax.set_xlabel('RA offset (arcsec)')
                        else:
                            ax.set_xticklabels([])
                    else:
                        ax.axis('off')
                        
                cbar = fig.colorbar(im, ax=axes.ravel().tolist(), orientation='vertical', shrink=0.8)
                cbar.set_label(f"Flux ({self.tab.display_unit})")
                
                plt.savefig(filename, format='pdf', bbox_inches='tight')
                plt.close(fig)
                QMessageBox.information(self, "Success", f"Saved PDF: {filename}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save PDF:\n{str(e)}")


class ContourOptionsDialog(QDialog):
    _LINE_STYLES = {'Solid': Qt.SolidLine, 'Dashed': Qt.DashLine, 'Dotted': Qt.DotLine}

    def __init__(self, parent, contour_overlays):
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
        for i in range(self.tab_widget.count()):
            new_opts = self._collect_tab_options(i)
            if 0 <= i < len(self.tab.contour_overlays):
                self.tab.contour_overlays[i]['options'] = new_opts
        self.tab.draw_overlay_contours()
        self.tab.update_spectrum()
        self.tab.update_spatial_analysis()

    def _on_clear_all(self):
        self.tab.close_overlay()
        self.close()

    def _on_clear_single(self, idx):
        self.tab.close_overlay(idx)
        self.tab_widget.removeTab(idx)
        if self.tab_widget.count() == 0:
            self.close()
        else:
            self.overlays = self.tab.contour_overlays

    def _create_overlay_tab(self, idx, ov):
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