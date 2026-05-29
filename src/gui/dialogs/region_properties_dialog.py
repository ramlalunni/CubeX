"""
Module containing the dialog for editing geometric properties of Regions of Interest (ROIs).

This dialog provides exact numerical control over shapes (center, size, angle) 
in both pixel/image and astronomical world coordinate systems.
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

class RegionPropertiesDialog(QDialog):
    """
    Dialog to adjust the properties (center, size, P.A.) of ROIs on the map.

    Supports Image and World coordinate modifications (if WCS is available) 
    for shapes like Ellipse, Rectangle, Point, and Line.

    Attributes
    ----------
    roi : pyqtgraph.ROI
        The actual pyqtgraph Region of Interest object being edited.
    roi_dict : dict
        The associated metadata dictionary tracking this ROI in the main app.
    tab : ExplorerView
        The parent explorer tab instance.
    use_world : bool
        Flag indicating if coordinates should be displayed/edited in WCS world coords.
    wcs : astropy.wcs.WCS or None
        The associated WCS object for the map if loaded.
    """
    def __init__(self, roi, explorer_view, parent=None, roi_dict=None):
        """
        Initialize the RegionPropertiesDialog.

        Parameters
        ----------
        roi : pyqtgraph.ROI
            The target pyqtgraph ROI object to modify.
        explorer_view : ExplorerView
            The parent explorer tab containing the 2D map.
        parent : PyQt5.QtWidgets.QWidget, optional
            The parent widget, by default None.
        roi_dict : dict, optional
            The dictionary representing ROI metadata, by default None.
        """
        super().__init__(parent or explorer_view)
        
        self.roi = roi
        self.roi_dict = roi_dict
        self.tab = explorer_view
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
        """
        Build the UI elements dynamically based on the ROI type.
        """
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
        """
        Switch between Image (Pixel/Arcsec offset) and World (RA/Dec) coordinate systems.
        """
        self.use_world = self.rad_world.isChecked()
        self.update_from_roi()

    def update_from_roi(self):
        """
        Pull spatial coordinates and dimensions from the ROI object to populate the dialog text fields.
        """
        if self._updating: return
        self._updating = True
        try:
            if self.is_polyline: return
            pos = self.roi.pos()
            size = self.roi.size()
            from src.gui.components.custom_widgets import get_casa_pa
            angle = get_casa_pa(self.roi)
            
            w, h = size.x(), size.y()
            cx = pos.x() + w / 2.0
            cy = pos.y() + h / 2.0
            
            from src.utils.wcs_helpers import get_ra_dec_str, calculate_position_angle

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
                        ra_str, dec_str, _ = get_ra_dec_str(self.wcs, px1, py1)
                        self.edit_p1x.setText(ra_str)
                        self.edit_p1y.setText(dec_str)
                        
                        px2 = (self.tab.nx / 2) - (p2_p.x() / self.tab.pix_scale_arcsec)
                        py2 = (p2_p.y() / self.tab.pix_scale_arcsec) + (self.tab.ny / 2)
                        ra_str, dec_str, _ = get_ra_dec_str(self.wcs, px2, py2)
                        self.edit_p2x.setText(ra_str)
                        self.edit_p2y.setText(dec_str)
                        
                        # Calculate astronomical PA (East of North)
                        angle = calculate_position_angle(self.wcs, px1, py1, px2, py2)
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
                ra_str, dec_str, _ = get_ra_dec_str(self.wcs, px, py)
                self.edit_cx.setText(ra_str)
                self.edit_cy.setText(dec_str)
                
                if not self.is_line:
                    self.edit_w.setText(f"{w:.5f}\"")
                    self.edit_h.setText(f"{h:.5f}\"")
                    self.lbl_size_img.setText(f"Image: ({w / self.tab.pix_scale_arcsec:.3f} px, {h / self.tab.pix_scale_arcsec:.3f} px)")
                
                self.lbl_center_img.setText(f"Image: ({px:.3f} px, {py:.3f} px)")
                
                if self.is_rect:
                    bl_px = (self.tab.nx / 2) - (pos.x() / self.tab.pix_scale_arcsec)
                    bl_py = (pos.y() / self.tab.pix_scale_arcsec) + (self.tab.ny / 2)
                    ra_str, dec_str, _ = get_ra_dec_str(self.wcs, bl_px, bl_py)
                    self.edit_bl_x.setText(ra_str)
                    self.edit_bl_y.setText(dec_str)
                    self.lbl_bl_img.setText(f"Image: ({bl_px:.3f} px, {bl_py:.3f} px)")
                    
                    tr_x = pos.x() + w
                    tr_y = pos.y() + h
                    tr_px = (self.tab.nx / 2) - (tr_x / self.tab.pix_scale_arcsec)
                    tr_py = (tr_y / self.tab.pix_scale_arcsec) + (self.tab.ny / 2)
                    ra_str, dec_str, _ = get_ra_dec_str(self.wcs, tr_px, tr_py)
                    self.edit_tr_x.setText(ra_str)
                    self.edit_tr_y.setText(dec_str)
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
        """
        Parse string coordinates into decimal degrees.

        Parameters
        ----------
        val_str : str
            The coordinate string (e.g. '12h34m56s' or '12:34:56').
        is_ra : bool, optional
            Whether this string is a Right Ascension coordinate, by default True.

        Returns
        -------
        float
            Decimal coordinate.
        """
        from src.utils.wcs_helpers import parse_coord_string
        return parse_coord_string(val_str, is_ra)

    def apply_to_roi(self):
        """
        Push user edits from the center/size/angle fields back to the ROI object.
        """
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
                from src.gui.components.custom_widgets import get_pyqt_angle
                self.roi.setAngle(get_pyqt_angle(pa_val), center=[0.5, 0.5])
                self.roi.blockSignals(False)
                
            self.roi.sigRegionChanged.emit(self.roi)
        except Exception:
            pass
        finally:
            self._updating = False
            self.update_from_roi()

    def apply_to_roi_from_bounds(self):
        """
        Push user edits from the bottom-left/top-right bounding box fields back to a Rectangle ROI.
        """
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
        """
        Update the name of the ROI in all relevant UI lists and dictionaries.
        """
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
        """
        Update the spatial integration width for a PV Cut ROI.

        Parameters
        ----------
        value : int
            The new width in pixels (must be odd).
        """
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
        """
        Push user edits from the endpoint coordinate fields back to a Line ROI.
        """
        if self._updating: return
        self._updating = True
        try:

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
