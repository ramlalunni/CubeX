"""
Module defining the controller for the overarching KinematicExplorerApp.

This handles application-wide actions such as file loading, export routines,
and communicating state changes across multiple tabs.
"""
import os
import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QDialog, QApplication
from astropy.wcs import WCS

class MainController:
    """
    Controller for the overarching KinematicExplorerApp (Main Window).

    Handles global file loading logic, application state, and cross-tab 
    export actions. Delegates tab-specific actions to the active `ExplorerView`.

    Attributes
    ----------
    view : KinematicExplorerApp
        The main application window view.
    """
    def __init__(self, view):
        """
        Initialize the MainController.

        Parameters
        ----------
        view : KinematicExplorerApp
            The main application window instance.
        """
        self.view = view

    def load_file(self):
        """
        Open a file dialog to load a new FITS cube.

        If the current tab is empty, the file is loaded into the active tab.
        Otherwise, a new tab is spawned.

        Returns
        -------
        None
        """
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self.view, "Open FITS Cube", "", "FITS (*.fits *.fits.gz);;All (*)", options=options)
        if file_name:
            tab = self.view.get_active_tab()
            self.view.statusBar().showMessage(f"Loading: {file_name.split('/')[-1]}...")
            QApplication.processEvents()
            
            if tab.cube_clean is None:
                success = tab.load_file(file_name)
            else:
                self.view.add_new_tab()
                tab = self.view.get_active_tab()
                success = tab.load_file(file_name)
                
            if success:
                short_name = file_name.split('/')[-1]
                self.view.tabs.setTabText(self.view.tabs.currentIndex(), short_name)
                self.view.setWindowTitle(f"CubeX - {short_name}")
                self.view.statusBar().showMessage("File loaded successfully.")
            else:
                self.view.statusBar().showMessage("Load failed.")

    def load_overlay_file(self):
        """Prompt the user to select and load a FITS file as a contour overlay on the active tab."""
        tab = self.view.get_active_tab()
        if not tab or tab.cube_clean is None:
            QMessageBox.warning(self.view, "No Data", "Please load a primary cube first.")
            return

        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self.view, "Open Contour Overlay FITS", "",
                                                     "FITS (*.fits *.fits.gz);;All (*)", options=options)
        if file_name:
            self.view.statusBar().showMessage(f"Loading overlay: {file_name.split('/')[-1]}...")
            QApplication.processEvents()
            tab.load_overlay_file(file_name)

    def close_cube(self):
        """Close the currently active cube and reset the tab state to default."""
        tab = self.view.get_active_tab()
        if tab:
            tab.close_file()
            self.view.tabs.setTabText(self.view.tabs.currentIndex(), "Untitled")
            self.view.statusBar().showMessage("Cube closed.")

    def _get_active_spectrum_curves(self, tab):
        """
        Retrieve a dictionary of the currently active (visible) 1D spectral curves in the provided tab.
        
        Parameters
        ----------
        tab : ExplorerView
            The tab to query for active spectrum curves.
            
        Returns
        -------
        dict
            A dictionary mapping region names to their active pyqtgraph curve objects.
        """
        is_smooth = False
        if getattr(tab, 'spectrum_tabs', None) is not None and getattr(tab, 'plot_widget_smooth', None) is not None:
            if tab.spectrum_tabs.currentWidget() == tab.plot_widget_smooth:
                is_smooth = True
                
        active_spatial = [r_dict for r_dict in getattr(tab, 'spectrum_spatial_rois', []) if r_dict["checkbox"].isChecked()]
        active_names = [r["name"] for r in active_spatial]
        if not active_spatial:
            active_names = ["Whole Map"]
            
        curves = {}
        for name in active_names:
            if name == "Whole Map":
                c = getattr(tab, 'spectrum_curve_smooth', tab.spectrum_curve) if is_smooth else tab.spectrum_curve
                if c and c.yData is not None:
                    curves[name] = c
            else:
                curves_dict = getattr(tab, 'spectrum_curves_smooth', getattr(tab, 'spectrum_curves', {})) if is_smooth else getattr(tab, 'spectrum_curves', {})
                if name in curves_dict and curves_dict[name].yData is not None:
                    curves[name] = curves_dict[name]
        return curves

    def _format_roi_props(self, roi):
        """
        Format the spatial properties (center, axes, position angle) of a given Region of Interest (ROI) into a descriptive string.
        
        Parameters
        ----------
        roi : pyqtgraph.ROI or None
            The ROI object to describe, or None if representing the whole map.
            
        Returns
        -------
        str
            A human-readable string describing the ROI geometry.
        """
        import pyqtgraph as pg
        if not roi: return "Whole Map"
        if isinstance(roi, pg.EllipseROI) or isinstance(roi, pg.RectROI):
            c = roi.pos() + roi.size()/2
            a, b = roi.size()[0]/2, roi.size()[1]/2
            pa = roi.angle()
            rtype = "Ellipse" if isinstance(roi, pg.EllipseROI) else "Rectangle"
            return f"{rtype} - Center({c.x():.2f}, {c.y():.2f}), a={a:.2f}, b={b:.2f}, PA={pa:.1f}"
        else:
            return "Custom Polygon"

    def _get_save_pdf_filename_with_title_option(self, default_filename):
        """
        Open a save file dialog tailored for exporting plots to PDF, including a custom checkbox for adding a title.
        
        Parameters
        ----------
        default_filename : str
            The suggested initial filename.
            
        Returns
        -------
        tuple
            A tuple `(filename, include_title)` where `filename` is the selected path (or None) and `include_title` is a boolean.
        """
        from PyQt5.QtWidgets import QCheckBox
        dialog = QFileDialog(self.view, "Save PDF", default_filename, "PDF Files (*.pdf)")
        dialog.setOption(QFileDialog.DontUseNativeDialog, False)
        dialog.setAcceptMode(QFileDialog.AcceptSave)
        dialog.setDefaultSuffix("pdf")
        
        layout = dialog.layout()
        chk_title = QCheckBox("Include Plot Title in Export")
        chk_title.setChecked(False)
        if layout:
            try:
                layout.addWidget(chk_title, layout.rowCount(), 0, 1, layout.columnCount())
            except Exception:
                layout.addWidget(chk_title)
                
        if dialog.exec_() == QFileDialog.Accepted:
            files = dialog.selectedFiles()
            if files:
                return files[0], chk_title.isChecked()
        return None, False

    def export_spectrum(self):
        """
        Export the selected 1D spectral regions to a CSV file.

        Returns
        -------
        None
        """
        from cubex.core.exporters import export_spectrum_csv_core
        # Circular import resolution for dialogs
        from cubex.gui.dialogs import ExportRegionsDialog
        tab = self.view.get_active_tab()
        if tab and tab.cube_clean is not None:
            curves = self._get_active_spectrum_curves(tab)
            if not curves:
                QMessageBox.warning(self.view, "No Data", "No spectrum data to export.")
                return
                
            regions_to_export = list(curves.keys())
            if len(curves) > 1:
                dlg = ExportRegionsDialog(self.view, curves, "Export Spectra (CSV)")
                if dlg.exec_() == QDialog.Accepted:
                    regions_to_export = dlg.get_selected_regions()
                    if not regions_to_export: return
                else:
                    return

            parent_filename = "cube"
            if getattr(tab, 'current_file_name', None):
                parent_filename = os.path.basename(tab.current_file_name)
                
            base_filename = os.path.splitext(parent_filename)[0]
            region_name = regions_to_export[0].replace(' ', '_').lower() if regions_to_export else 'whole_map'
            default_filename = f"{base_filename}_spectrum_{region_name}.csv"
            
            options = QFileDialog.Options()
            save_path, _ = QFileDialog.getSaveFileName(self.view, "Save Spectrum CSV", default_filename, "CSV Files (*.csv)", options=options)
            if save_path:
                try:
                    sort_idx = np.argsort(tab.v_axis)
                    v_sorted = tab.v_axis[sort_idx]
                    
                    if save_path.endswith('.csv'):
                        save_path = save_path[:-4]
                    
                    active_spatial = getattr(tab, 'spectrum_spatial_rois', [])
                    roi_dict = {r["name"]: self._format_roi_props(r.get("roi", None)) for r in active_spatial}
                    curves_data = {name: curves[name].yData for name in regions_to_export}
                    
                    export_spectrum_csv_core(save_path, parent_filename, regions_to_export, curves_data, v_sorted, roi_dict, tab.spec_unit)
                    self.view.statusBar().showMessage("Spectra saved successfully.")
                except Exception as e:
                    QMessageBox.critical(self.view, "Error", f"Failed to save file:\n{str(e)}")
        else:
            QMessageBox.warning(self.view, "No Data", "No cube loaded to export.")

    def export_spectrum_fits(self):
        """
        Export the selected 1D spectral regions to a 1D FITS file.

        Returns
        -------
        None
        """
        from cubex.core.exporters import export_spectrum_fits_core
        from cubex.gui.dialogs import ExportRegionsDialog
        tab = self.view.get_active_tab()
        if not tab or tab.cube_clean is None: return
        
        curves = self._get_active_spectrum_curves(tab)
        if not curves: return
        
        regions_to_export = list(curves.keys())
        if len(curves) > 1:
            dlg = ExportRegionsDialog(self.view, curves, "Export Spectra (FITS)")
            if dlg.exec_() == QDialog.Accepted:
                regions_to_export = dlg.get_selected_regions()
                if not regions_to_export: return
            else:
                return
                
        options = QFileDialog.Options()
        base_filename, _ = QFileDialog.getSaveFileName(self.view, "Save Spectrum FITS", "spectrum.fits", "FITS Files (*.fits)", options=options)
        if base_filename:
            try:
                sort_idx = np.argsort(tab.v_axis)
                v_sorted = tab.v_axis[sort_idx]
                
                if base_filename.endswith('.fits'):
                    base_filename = base_filename[:-5]
                    
                curves_data = {name: curves[name].yData for name in regions_to_export}
                restfrq = tab.raw_header.get('RESTFRQ', None) if tab.raw_header else None
                export_spectrum_fits_core(base_filename, regions_to_export, curves_data, v_sorted, tab.spec_unit, restfrq)
                self.view.statusBar().showMessage("Saved Spectra FITS successfully.")
            except Exception as e:
                QMessageBox.critical(self.view, "Error", f"Failed to save FITS:\n{str(e)}")

    def export_spectrum_pdf(self):
        """
        Export the selected 1D spectral plots to a PDF document.

        Returns
        -------
        None
        """
        from cubex.core.exporters import export_spectrum_pdf_core
        from cubex.gui.dialogs import ExportRegionsDialog
        tab = self.view.get_active_tab()
        if not tab or tab.cube_clean is None: return
        
        curves = self._get_active_spectrum_curves(tab)
        if not curves: return
        
        regions_to_export = list(curves.keys())
        is_single_file = True
        if len(curves) > 1:
            dlg = ExportRegionsDialog(self.view, curves, "Export Spectra (PDF)", is_pdf=True)
            if dlg.exec_() == QDialog.Accepted:
                regions_to_export = dlg.get_selected_regions()
                is_single_file = dlg.is_single_file()
                if not regions_to_export: return
            else:
                return
                
        parent_filename = "cube"
        if getattr(tab, 'current_file_name', None):
            parent_filename = os.path.basename(tab.current_file_name)
        base_filename = os.path.splitext(parent_filename)[0]
        region_name = regions_to_export[0].replace(' ', '_').lower() if regions_to_export else 'whole_map'
        default_filename = f"{base_filename}_spectrum_{region_name}.pdf"
        
        save_path, include_title = self._get_save_pdf_filename_with_title_option(default_filename)
        if save_path:
            try:
                sort_idx = np.argsort(tab.v_axis)
                v_sorted = tab.v_axis[sort_idx]
                
                if save_path.endswith('.pdf'):
                    save_path = save_path[:-4]
                    
                color_map = {}
                if "Whole Map" in curves: color_map["Whole Map"] = '#3498db'
                for r_dict in getattr(tab, 'spectrum_spatial_rois', []):
                    color_map[r_dict["name"]] = r_dict["color"]
                    
                curves_data = {name: curves[name].yData for name in regions_to_export}
                catalog_items = []
                for item in getattr(tab, 'catalog_overlay_items', []):
                    if isinstance(item, pg.InfiniteLine):
                        catalog_items.append({'type': 'line', 'x': item.pos().x()})
                    elif isinstance(item, pg.TextItem):
                        catalog_items.append({'type': 'text', 'x': item.pos().x(), 'text': item.toPlainText()})
                        
                export_spectrum_pdf_core(save_path, regions_to_export, curves_data, v_sorted, is_single_file, include_title, base_filename, tab.spec_unit, color_map, catalog_items)
                self.view.statusBar().showMessage("Saved Spectra PDF successfully.")
            except Exception as e:
                QMessageBox.critical(self.view, "Error", f"Failed to save PDF:\n{str(e)}")

    def export_fits_active(self):
        """
        Export the currently active 2D panel to a FITS file.

        Returns
        -------
        None
        """
        from cubex.core.exporters import export_fits_active_core
        tab = self.view.get_active_tab()
        if not tab or tab.cube_clean is None: return
        target_id = tab.last_clicked_panel_id
        
        if target_id == 'spectrum': return
        
        if target_id == 'channel':
            data = tab.get_current_channel_data()
            name = "Channel_Map"
            bunit = tab.display_unit
        else:
            data = tab.panels[target_id]['current_data']
            name = tab.panels[target_id]['combo'].currentText().split('(')[0].strip().replace(" ", "_")
            bunit = tab.panels[target_id].get('unit', tab.display_unit)
            
        if data is None: return
        
        options = QFileDialog.Options()
        filename, _ = QFileDialog.getSaveFileName(self.view, "Save FITS", f"{name}.fits", "FITS Files (*.fits)", options=options)
        if filename:
            try:
                export_data = data.T 
                panel_type = name.replace("_", " ") if target_id != 'channel' else "Channel Map"
                new_wcs_header = None
                
                if tab.raw_header is not None:
                    orig_wcs = WCS(tab.raw_header)
                    if panel_type == "PV Diagram":
                        new_wcs = WCS(naxis=2)
                        new_wcs.wcs.ctype = ['OFFSET', orig_wcs.wcs.ctype[2]]
                        if hasattr(tab, 'panels') and target_id in tab.panels:
                            dx = tab.panels[target_id].get('dx', 1.0)
                            dv = tab.panels[target_id].get('dv', 1.0)
                            offsets = tab.panels[target_id].get('offsets', [0])
                            v_sorted = tab.panels[target_id].get('v_sorted', [0])
                            new_wcs.wcs.cdelt = [dx, dv]
                            new_wcs.wcs.crpix = [1, 1]
                            new_wcs.wcs.crval = [offsets[0] * 3600.0 if orig_wcs.wcs.cunit[0] == 'deg' else offsets[0], v_sorted[0]]
                            new_wcs.wcs.cunit = ['arcsec', orig_wcs.wcs.cunit[2] if len(orig_wcs.wcs.cunit) > 2 else 'km/s']
                        new_wcs_header = new_wcs.to_header()
                    else:
                        new_wcs_header = orig_wcs.celestial.to_header()
                    
                if panel_type == "Moment 0": bunit = f"{tab.display_unit} km/s"
                elif panel_type in ["Moment 1", "Moment 2", "Moment 9"]: bunit = "km/s"
                elif panel_type in ["Channel Map", "Moment 8", "PV Diagram"]: bunit = tab.display_unit
                
                bmaj, bmin, bpa = None, None, None
                if panel_type != "PV Diagram":
                    if hasattr(tab, 'bmaj_array') and tab.bmaj_array is not None:
                        if len(tab.bmaj_array) > 1:
                            if panel_type == "Channel Map":
                                c_idx = tab.slider_channel.value()
                                bmaj = tab.bmaj_array[c_idx]
                                bmin = tab.bmin_array[c_idx]
                                bpa = tab.bpa_array[c_idx] if tab.bpa_array is not None else 0.0
                            else:
                                minX, maxX = tab.region.getRegion()
                                search_axis = tab.v_axis if tab.v_axis[0] < tab.v_axis[-1] else tab.v_axis[::-1]
                                idx_min = np.searchsorted(search_axis, minX)
                                idx_max = np.searchsorted(search_axis, maxX)
                                if tab.v_axis[0] > tab.v_axis[-1]:
                                    idx_min, idx_max = len(tab.v_axis) - idx_max, len(tab.v_axis) - idx_min
                                if idx_max > idx_min:
                                    bmaj = float(np.nanmedian(tab.bmaj_array[idx_min:idx_max]))
                                    bmin = float(np.nanmedian(tab.bmin_array[idx_min:idx_max]))
                                    bpa_vals = tab.bpa_array[idx_min:idx_max] if tab.bpa_array is not None else [0.0]
                                    bpa = float(np.nanmedian(bpa_vals))
                                else:
                                    idx_safe = min(idx_min, len(tab.bmaj_array)-1)
                                    bmaj = float(tab.bmaj_array[idx_safe])
                                    bmin = float(tab.bmin_array[idx_safe])
                                    bpa = float(tab.bpa_array[idx_safe]) if tab.bpa_array is not None else 0.0
                        else:
                            bmaj, bmin = float(tab.bmaj_array[0]), float(tab.bmin_array[0])
                            bpa = float(tab.bpa_array[0]) if tab.bpa_array is not None else 0.0
                            
                export_fits_active_core(filename, export_data, panel_type, bunit, tab.raw_header, new_wcs_header, bmaj, bmin, bpa)
                self.view.statusBar().showMessage(f"Saved FITS: {filename}")
            except Exception as e:
                QMessageBox.critical(self.view, "Error", f"Failed to save FITS:\n{str(e)}")

    def export_pdf_active(self):
        """
        Export the currently active 2D panel to a PDF document.

        Returns
        -------
        None
        """
        from cubex.core.exporters import export_pdf_active_core
        tab = self.view.get_active_tab()
        if not tab or tab.cube_clean is None: return
        target_id = tab.last_clicked_panel_id
        
        if target_id == 'spectrum': return
        
        parent_filename = "cube"
        if getattr(tab, 'current_file_name', None):
            parent_filename = os.path.basename(tab.current_file_name)
        base_filename = os.path.splitext(parent_filename)[0]
        
        if target_id == 'channel':
            data = tab.get_current_channel_data()
            task = "channel"
            cbar_label = f"Flux ({tab.display_unit})"
            cmap_name = self.view.current_cmap
            levels = tab.ch_levels
        else:
            p = tab.panels[target_id]
            data = p['current_data']
            mtype = p['combo'].currentText()
            task = mtype.replace(' ', '_').lower()
            cbar_label = p['view'].ui.histogram.axis.labelText
            is_vel = ("Moment 1" in mtype) or ("Moment 9" in mtype)
            cmap_name = 'bwr' if is_vel else self.view.current_cmap
            levels = p['view'].ui.histogram.getLevels()

        if data is None: return
        
        default_filename = f"{base_filename}_{task}.pdf"
        filename, include_title = self._get_save_pdf_filename_with_title_option(default_filename)
        if filename:
            try:
                plot_data = data.T
                extent = [tab.nx/2 * tab.pix_scale_arcsec, -tab.nx/2 * tab.pix_scale_arcsec, 
                          -tab.ny/2 * tab.pix_scale_arcsec, tab.ny/2 * tab.pix_scale_arcsec]
                contour_params = tab.contour_params.get(target_id) if hasattr(tab, 'contour_params') else None
                
                export_pdf_active_core(filename, plot_data, task, cbar_label, cmap_name, levels, include_title, base_filename, tab.wcs_2d, contour_params, self.view.is_absolute_wcs, extent)
                self.view.statusBar().showMessage(f"Saved PDF: {filename}")
            except Exception as e:
                QMessageBox.critical(self.view, "Error", f"Failed to save PDF:\n{str(e)}")

    def show_header(self):
        """Display the primary FITS header of the active cube in a popup dialog."""
        from PyQt5.QtWidgets import QVBoxLayout, QTextEdit
        tab = self.view.get_active_tab()
        if tab and tab.cube_clean is not None:
            dlg = QDialog(self.view)
            dlg.setWindowTitle("FITS Header")
            dlg.resize(600, 800)
            layout = QVBoxLayout(dlg)
            text_edit = QTextEdit()
            text_edit.setReadOnly(True)
            text_edit.setFontFamily("monospace")
            text_edit.setText(tab.fits_header_text)
            layout.addWidget(text_edit)
            dlg.exec_()
        else:
            QMessageBox.warning(self.view, "No Data", "Please load a cube first.")

    def reset_views(self):
        """Auto-range all 2D image and 1D spectrum plots in the active tab to fit their current data."""
        tab = self.view.get_active_tab()
        if tab:
            tab.view_channel.autoRange()
            tab.plot_widget.autoRange() 
            for p in getattr(tab, 'panels', []):
                p['view'].autoRange()

    def show_contour_dialog(self):
        """Open the contour configuration dialog for the most recently clicked 2D plot panel."""
        from cubex.gui.dialogs import ContourDialog
        from PyQt5.QtCore import Qt
        tab = self.view.get_active_tab()
        if not tab or tab.cube_clean is None: 
            QMessageBox.warning(self.view, "No Data", "Please load a cube first.")
            return

        target_id = tab.last_clicked_panel_id
        if target_id == 'spectrum':
            QMessageBox.warning(self.view, "Invalid Panel", "Contours cannot be drawn on the 1D spectrum.")
            return

        if target_id == 'channel':
            name = "Channel Map"
        else:
            name = f"Bottom Panel {target_id + 1} ({tab.panels[target_id]['combo'].currentText()})"

        dlg = ContourDialog(self.view, tab.contour_params.get(target_id), f"Contours - {name}",
                            target_tab=tab, target_id=target_id)
        dlg.setAttribute(Qt.WA_DeleteOnClose, True)
        dlg.show()

    def open_line_catalog(self):
        """Open the spectral line catalog tool for the active tab to overlay known transition lines."""
        tab = self.view.get_active_tab()
        if tab: tab.open_line_catalog()

    def set_colormap(self, cmap_name):
        """
        Apply a new colormap to all active 2D images (channel maps and moment maps) globally.
        
        Parameters
        ----------
        cmap_name : str
            The name of the colormap to apply.
        """
        self.view.current_cmap = cmap_name
        tab = self.view.get_active_tab()
        if tab and tab.cube_clean is not None:
            if hasattr(tab, 'controller'):
                tab.controller.update_moment_maps()
                tab.controller.update_channel_map()
            else:
                tab.update_moment_maps()
                tab.update_channel_map()

    def toggle_wcs(self, checked):
        """
        Toggle between absolute WCS coordinates (RA/Dec) and relative offset coordinates (arcsec) for all spatial axes.
        
        Parameters
        ----------
        checked : bool
            True if absolute WCS coordinates should be used, False for relative offsets.
        """
        self.view.is_absolute_wcs = checked
        for i in range(self.view.tabs.count()):
            tab = self.view.tabs.widget(i)
            if hasattr(tab, 'controller'):
                tab.controller.update_wcs_mode(checked)
            elif hasattr(tab, 'update_wcs_mode'):
                tab.update_wcs_mode(checked)

