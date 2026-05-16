from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QLineEdit, QComboBox, QDoubleSpinBox, 
                             QRadioButton, QSpinBox, QTableWidget, QTableWidgetItem, 
                             QHeaderView, QMessageBox, QStackedWidget, QWidget,
                             QScrollArea, QGridLayout, QFileDialog)
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
    """
    Dialog to allow the user to specify auto-generated or manual contour levels 
    for the image panels.
    """
    def __init__(self, parent, current_params, title):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(350)
        self.params = current_params or {'mode': 'auto', 'n': 5, 'levels': []}

        layout = QVBoxLayout(self)

        self.rad_auto = QRadioButton("Auto-determine optimal levels")
        self.rad_auto.setChecked(self.params['mode'] == 'auto')
        self.spin_n = QSpinBox()
        self.spin_n.setRange(1, 50)
        self.spin_n.setValue(self.params['n'])

        h1 = QHBoxLayout()
        h1.addWidget(self.rad_auto)
        h1.addWidget(QLabel("Number of levels:"))
        h1.addWidget(self.spin_n)
        layout.addLayout(h1)

        self.rad_manual = QRadioButton("Manual contour levels")
        self.rad_manual.setChecked(self.params['mode'] == 'manual')
        self.edit_manual = QLineEdit()
        self.edit_manual.setPlaceholderText("e.g., 0.1, 0.5, 1.2, 5.0")
        if self.params['levels']:
            self.edit_manual.setText(", ".join(map(str, self.params['levels'])))

        h2 = QHBoxLayout()
        h2.addWidget(self.rad_manual)
        h2.addWidget(self.edit_manual)
        layout.addLayout(h2)

        btn_layout = QHBoxLayout()
        btn_apply = QPushButton("Apply Contours")
        btn_clear = QPushButton("Clear Contours")
        btn_cancel = QPushButton("Cancel")

        btn_apply.clicked.connect(self.apply)
        btn_clear.clicked.connect(self.clear)
        btn_cancel.clicked.connect(self.reject)

        btn_layout.addWidget(btn_apply)
        btn_layout.addWidget(btn_clear)
        btn_layout.addWidget(btn_cancel)
        layout.addLayout(btn_layout)

        self.result_params = None
        self.action = 'cancel'

    def apply(self):
        if self.rad_auto.isChecked():
            self.result_params = {'mode': 'auto', 'n': self.spin_n.value(), 'levels': []}
        else:
            try:
                lvls = [float(x.strip()) for x in self.edit_manual.text().split(',') if x.strip()]
                self.result_params = {'mode': 'manual', 'n': 0, 'levels': lvls}
            except ValueError:
                QMessageBox.warning(self, "Error", "Invalid manual levels. Please use comma-separated numbers.")
                return
        self.action = 'apply'
        self.accept()

    def clear(self):
        self.action = 'clear'
        self.accept()

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
        self.tool = roi_dict.get("tool", "Unknown") if roi_dict else "Unknown"
        self.is_ellipse = isinstance(roi, pg.EllipseROI) or self.tool == "Ellipse"
        self.is_line = isinstance(roi, pg.LineSegmentROI) or self.tool == "Line"
        self.is_polyline = (isinstance(roi, pg.PolyLineROI) and not self.is_ellipse) or self.tool == "Custom Polygon"
        self.is_point = self.tool == "Point"
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
        
        if self.is_point:
            layout.addStretch()
            self._updating = False
            return
            
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
        pa_layout.addWidget(self.edit_pa)
        pa_layout.addStretch()
        layout.addLayout(pa_layout)
        
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
            angle = self.roi.angle()
            
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

            if not self.is_point:
                self.edit_pa.setText(f"{angle:.5f}")
            
            if self.use_world and self.wcs:
                px = (self.tab.nx / 2) - (cx / self.tab.pix_scale_arcsec)
                py = (cy / self.tab.pix_scale_arcsec) + (self.tab.ny / 2)
                ra, dec = self.wcs.pixel_to_world_values(px, py)
                sc = SkyCoord(ra, dec, unit='deg')
                self.edit_cx.setText(sc.ra.to_string(unit=u.hour, sep=':', precision=7))
                self.edit_cy.setText(sc.dec.to_string(unit=u.deg, sep=':', precision=6))
                
                if not self.is_point and not self.is_line:
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
                if not self.is_point and not self.is_line:
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
                self.roi.setAngle(pa_val)
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
                    
                    # Update Checkbox if it exists (Spectrum ROIs)
                    if "checkbox" in self.roi_dict:
                        self.roi_dict["checkbox"].setText(new_name)
                    
                    # Update Dropdown if it exists (Spatial ROIs)
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
                        
                    self.tab.update_spectrum()
                    self.tab.update_spectrum_region_calc()
                    # Refresh spectral statistics popup if open
                    self.tab.refresh_spectral_stats_apertures()

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
        
        # Left side: Scroll area for grid
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setStyleSheet("QScrollArea { border: none; }")
        self.grid_widget = QWidget()
        self.grid_layout = QGridLayout(self.grid_widget)
        self.grid_layout.setSpacing(2) # Minimize spacing between tiles
        self.grid_layout.setContentsMargins(2, 2, 2, 2)
        self.scroll.setWidget(self.grid_widget)
        layout.addWidget(self.scroll, stretch=4)
        
        # Right side: Colorbar (Histogram) + Export Button
        right_layout = QVBoxLayout()
        self.hist = pg.HistogramLUTWidget()
        right_layout.addWidget(self.hist, stretch=1)
        
        self.btn_export = QPushButton("Export to PDF")
        self.btn_export.clicked.connect(self.export_to_pdf)
        self.btn_export.setStyleSheet("background-color: #2c3e50; color: white; padding: 5px;")
        right_layout.addWidget(self.btn_export)
        
        layout.addLayout(right_layout, stretch=1)
        
        self.images = []
        
        # Connect histogram signals to update all images
        self.hist.sigLevelsChanged.connect(self.on_hist_levels_changed)
        self.hist.sigLookupTableChanged.connect(self.on_hist_lut_changed)
        
        self.update_grid()
        
    def update_grid(self):
        # Clear existing grid
        while self.grid_layout.count():
            item = self.grid_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.images.clear()
        
        cube, v_axis, minX, maxX = self.tab.get_velocity_subset(use_full_range=False)
        if cube is None or len(v_axis) == 0:
            lbl = QLabel("No channels in selected range.")
            self.grid_layout.addWidget(lbl, 0, 0)
            return
            
        n_channels = len(v_axis)
        cols = int(np.ceil(np.sqrt(n_channels)))
        rows = int(np.ceil(n_channels / cols))
        
        # Get current histogram state from main channel map
        main_hist = self.tab.view_channel.ui.histogram
        levels = main_hist.getLevels()
        
        # Update dialog's histogram to match main one
        self.hist.setLevels(levels[0], levels[1])
        self.hist.gradient.restoreState(main_hist.gradient.saveState())
        lut = self.hist.gradient.getLookupTable(256)
        
        # Adapt histogram range to be reasonable (slightly larger than levels)
        margin = (levels[1] - levels[0]) * 0.1
        self.hist.setHistogramRange(levels[0] - margin, levels[1] + margin)
        
        pos_tup = ((self.tab.nx / 2) * self.tab.pix_scale_arcsec, -(self.tab.ny / 2) * self.tab.pix_scale_arcsec)
        scale_tup = (-self.tab.pix_scale_arcsec, self.tab.pix_scale_arcsec)
        
        for idx in range(n_channels):
            r, c = divmod(idx, cols)
            
            pw = pg.PlotWidget(background='#1a1a1a')
            pw.setAspectLocked(True)
            pw.invertY(False)
            pw.invertX(True)
            
            # Show axes only on left and bottom edges
            if c == 0:
                pw.getAxis('left').show()
                pw.setLabel('left', 'Dec offset (arcsec)')
            else:
                pw.getAxis('left').hide()
                
            if idx + cols >= n_channels:
                pw.getAxis('bottom').show()
                pw.setLabel('bottom', 'RA offset (arcsec)')
            else:
                pw.getAxis('bottom').hide()
            
            img = pg.ImageItem()
            pw.addItem(img)
            
            # Set data
            img.setImage(cube[idx, :, :], scale=scale_tup, pos=pos_tup)
            img.setLevels(levels)
            img.setLookupTable(lut)
            
            # Add velocity text at top left corner using image coordinates
            vel_text = pg.TextItem(f"{v_axis[idx]:.2f} km/s", color='w', anchor=(0, 0))
            vel_text.setParentItem(img)
            vel_text.setPos(0, self.tab.ny)
            pw.addItem(vel_text)
            
            pw.setFixedSize(150, 150)
            
            self.grid_layout.addWidget(pw, r, c)
            self.images.append(img)
            
        # Link the histogram to the first image to make it active
        if self.images:
            self.hist.setImageItem(self.images[0])
            
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
                            ax.set_ylabel('Dec offset (arcsec)')
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
                QMessageBox.critical(self, "Error", f"Failed to save PDF:\\n{str(e)}")