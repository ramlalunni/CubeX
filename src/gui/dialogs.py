from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QLineEdit, QComboBox, QDoubleSpinBox, 
                             QRadioButton, QSpinBox, QTableWidget, QTableWidgetItem, 
                             QHeaderView, QMessageBox)

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
    def __init__(self, roi, explorer_tab, parent=None):
        super().__init__(parent or explorer_tab)
        self.roi = roi
        self.tab = explorer_tab
        self.is_ellipse = isinstance(roi, pg.EllipseROI)
        self.setWindowTitle("Region Properties")
        self.setMinimumWidth(400)
        
        self.wcs = self.tab.wcs_2d
        self.use_world = self.wcs is not None
        
        self.initUI()
        self.update_from_roi()
        self.roi.sigRegionChanged.connect(self.update_from_roi)

    def initUI(self):
        layout = QVBoxLayout(self)
        
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
        
        # Size / Semi-axes
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
        if not self.is_ellipse:
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
        self.edit_w.editingFinished.connect(self.apply_to_roi)
        self.edit_h.editingFinished.connect(self.apply_to_roi)
        self.edit_pa.editingFinished.connect(self.apply_to_roi)
        if not self.is_ellipse:
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
            pos = self.roi.pos()
            size = self.roi.size()
            angle = self.roi.angle()
            
            w, h = size.x(), size.y()
            cx = pos.x() + w / 2.0
            cy = pos.y() + h / 2.0
            
            self.edit_pa.setText(f"{angle:.5f}")
            
            if self.use_world and self.wcs:
                from astropy.coordinates import SkyCoord
                import astropy.units as u
                
                px = (self.tab.nx / 2) - (cx / self.tab.pix_scale_arcsec)
                py = (cy / self.tab.pix_scale_arcsec) + (self.tab.ny / 2)
                ra, dec = self.wcs.pixel_to_world_values(px, py)
                sc = SkyCoord(ra, dec, unit='deg')
                self.edit_cx.setText(sc.ra.to_string(unit=u.hour, sep=':', precision=7))
                self.edit_cy.setText(sc.dec.to_string(unit=u.deg, sep=':', precision=6))
                
                self.edit_w.setText(f"{w:.5f}\"")
                self.edit_h.setText(f"{h:.5f}\"")
                
                self.lbl_center_img.setText(f"Image: ({px:.3f} px, {py:.3f} px)")
                self.lbl_size_img.setText(f"Image: ({w / self.tab.pix_scale_arcsec:.3f} px, {h / self.tab.pix_scale_arcsec:.3f} px)")
                
                if not self.is_ellipse:
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
                self.edit_w.setText(f"{w:.5f}")
                self.edit_h.setText(f"{h:.5f}")
                self.lbl_center_img.setText(f"Image: ({cx:.3f} arcsec, {cy:.3f} arcsec)")
                self.lbl_size_img.setText(f"Image: ({w:.3f} arcsec, {h:.3f} arcsec)")
                
                if not self.is_ellipse:
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
            w_val = float(self.edit_w.text().replace('"', ''))
            h_val = float(self.edit_h.text().replace('"', ''))
            pa_val = float(self.edit_pa.text())
            
            if self.use_world and self.wcs:
                ra = self._parse_coord(cx_str, is_ra=True)
                dec = self._parse_coord(cy_str, is_ra=False)
                px, py = self.wcs.world_to_pixel_values(ra, dec)
                cx = (self.tab.nx / 2 - px) * self.tab.pix_scale_arcsec
                cy = (py - self.tab.ny / 2) * self.tab.pix_scale_arcsec
                w, h = w_val, h_val
            else:
                cx, cy = float(cx_str), float(cy_str)
                w, h = w_val, h_val
                
            pos_x = cx - w / 2.0
            pos_y = cy - h / 2.0
            
            self.roi.blockSignals(True)
            self.roi.setPos([pos_x, pos_y])
            self.roi.setSize([w, h])
            self.roi.setAngle(pa_val)
            self.roi.blockSignals(False)
            self.roi.sigRegionChanged.emit(self.roi)
        except ValueError:
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