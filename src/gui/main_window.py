import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import pyqtgraph as pg

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import (QApplication, QMainWindow, QAction, QTabWidget, 
                             QFileDialog, QMessageBox, QMenu, QDialog, 
                             QVBoxLayout, QTextEdit)

# Import the tab environment we built
from src.gui.explorer_tab import ExplorerTab

# ==============================================================================
# MAIN WINDOW APP
# ==============================================================================
class KinematicExplorerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CubeX")
        self.setGeometry(50, 50, 1600, 950) 
        self.current_cmap = 'turbo' 
        self.is_absolute_wcs = False 

        self.tabs = QTabWidget()
        self.tabs.setTabsClosable(True)
        self.tabs.tabCloseRequested.connect(self.close_tab)
        self.tabs.currentChanged.connect(self.update_menu_states)
        self.setCentralWidget(self.tabs)
        
        self.statusBar().showMessage("Ready.")
        self.init_menu()
        self.add_new_tab() 

    def init_menu(self):
        menubar = self.menuBar()
        
        # --- File Menu ---
        file_menu = menubar.addMenu('File')
        
        action_load = QAction('Open FITS Cube', self)
        action_load.triggered.connect(self.load_file)
        file_menu.addAction(action_load)

        action_close = QAction('Close FITS Cube', self)
        action_close.triggered.connect(self.close_cube)
        file_menu.addAction(action_close)

        file_menu.addSeparator()

        action_new_tab = QAction('New Tab', self)
        action_new_tab.triggered.connect(self.add_new_tab)
        file_menu.addAction(action_new_tab)

        action_new_win = QAction('New Window', self)
        action_new_win.triggered.connect(self.spawn_new_window)
        file_menu.addAction(action_new_win)

        file_menu.addSeparator()

        action_exit = QAction('Exit', self)
        action_exit.triggered.connect(self.close)
        file_menu.addAction(action_exit)

        # --- View Menu ---
        view_menu = menubar.addMenu('View')
        
        action_header = QAction('File Header', self)
        action_header.triggered.connect(self.show_header)
        view_menu.addAction(action_header)

        action_reset = QAction('Reset Zoom/Views', self)
        action_reset.triggered.connect(self.reset_views)
        view_menu.addAction(action_reset)
        
        view_menu.addSeparator()
        
        self.action_wcs = QAction('Use Absolute WCS Coordinates', self, checkable=True)
        self.action_wcs.setChecked(False)
        self.action_wcs.triggered.connect(self.toggle_wcs)
        view_menu.addAction(self.action_wcs)
        
        view_menu.addSeparator()
        
        theme_menu = QMenu('Theme (Global)', self)
        action_theme_dark = QAction('Dark Theme (Default)', self)
        action_theme_dark.triggered.connect(lambda: self.set_theme('dark'))
        action_theme_light = QAction('Light Theme', self)
        action_theme_light.triggered.connect(lambda: self.set_theme('light'))
        theme_menu.addAction(action_theme_dark)
        theme_menu.addAction(action_theme_light)
        view_menu.addMenu(theme_menu)

        # --- Tools Menu ---
        tools_menu = menubar.addMenu('Tools')
        
        action_lines = QAction('Query Molecular Line Database...', self)
        action_lines.triggered.connect(self.open_line_catalog)
        tools_menu.addAction(action_lines)

        action_contour = QAction('Draw Contours on Active Panel...', self)
        action_contour.triggered.connect(self.show_contour_dialog)
        tools_menu.addAction(action_contour)
        tools_menu.addSeparator()

        action_clear_roi = QAction('Clear Spatial ROIs', self)
        action_clear_roi.triggered.connect(self.clear_roi)
        tools_menu.addAction(action_clear_roi)

        action_clear_pv = QAction('Clear PV Cuts', self)
        action_clear_pv.triggered.connect(self.clear_pv_cuts)
        tools_menu.addAction(action_clear_pv)

        action_clear_spec_regions = QAction('Clear Spectrum Regions', self)
        action_clear_spec_regions.triggered.connect(self.clear_spectrum_regions)
        tools_menu.addAction(action_clear_spec_regions)

        cmap_menu = QMenu('Moment Intensity Colormap', self)
        for c in ['turbo', 'inferno', 'viridis', 'plasma', 'magma', 'grey']:
            act = QAction(c.capitalize(), self)
            act.triggered.connect(lambda checked, cm=c: self.set_colormap(cm))
            cmap_menu.addAction(act)
        tools_menu.addMenu(cmap_menu)

        # --- Export Menu ---
        export_menu = menubar.addMenu('Export')
        
        self.action_save_fits = QAction('Export active panel to FITS...', self)
        self.action_save_fits.triggered.connect(self.export_fits_active)
        export_menu.addAction(self.action_save_fits)

        self.action_save_pdf = QAction('Save active panel as PDF...', self)
        self.action_save_pdf.triggered.connect(self.export_pdf_active)
        export_menu.addAction(self.action_save_pdf)
        
        export_menu.addSeparator()

        self.action_export_spec_fits = QAction('Export spectrum as FITS...', self)
        self.action_export_spec_fits.triggered.connect(self.export_spectrum_fits)
        export_menu.addAction(self.action_export_spec_fits)

        action_export_csv = QAction('Export spectrum as CSV...', self)
        action_export_csv.triggered.connect(self.export_spectrum)
        export_menu.addAction(action_export_csv)
        
        self.action_export_spec_pdf = QAction('Save spectrum as PDF...', self)
        self.action_export_spec_pdf.triggered.connect(self.export_spectrum_pdf)
        export_menu.addAction(self.action_export_spec_pdf)

        # --- Help Menu ---
        help_menu = menubar.addMenu('Help')
        
        action_manual = QAction('Manual', self)
        action_manual.triggered.connect(self.show_manual)
        help_menu.addAction(action_manual)

        action_shortcuts = QAction('Controls & Shortcuts', self)
        action_shortcuts.triggered.connect(self.show_shortcuts)
        help_menu.addAction(action_shortcuts)

        help_menu.addSeparator()
        action_about = QAction('About CubeX', self)
        action_about.triggered.connect(self.show_about)
        help_menu.addAction(action_about)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            tab = self.get_active_tab()
            if tab and getattr(tab, 'roi_selected', False) and tab.current_roi is not None:
                tab.clear_roi()
        super().keyPressEvent(event)

    def update_menu_states(self):
        tab = self.get_active_tab()
        if tab:
            is_image = tab.last_clicked_panel_id != 'spectrum'
            self.action_save_fits.setEnabled(is_image)
            self.action_save_pdf.setEnabled(is_image)

    def toggle_wcs(self, checked):
        self.is_absolute_wcs = checked
        for i in range(self.tabs.count()):
            tab = self.tabs.widget(i)
            if hasattr(tab, 'update_wcs_mode'):
                tab.update_wcs_mode(checked)

    def get_active_tab(self): return self.tabs.currentWidget()

    def add_new_tab(self):
        tab = ExplorerTab(self)
        idx = self.tabs.addTab(tab, "Untitled")
        self.tabs.setCurrentIndex(idx)
        self.update_menu_states()

    def close_tab(self, index):
        if self.tabs.count() > 1:
            self.tabs.widget(index).deleteLater()
            self.tabs.removeTab(index)
        else:
            self.get_active_tab().close_file()
            self.tabs.setTabText(0, "Untitled")
            self.setWindowTitle("CubeX")
        self.update_menu_states()

    def spawn_new_window(self):
        self.new_win = KinematicExplorerApp()
        self.new_win.show()

    def clear_roi(self):
        tab = self.get_active_tab()
        if tab: tab.clear_roi()

    def clear_spectrum_regions(self):
        tab = self.get_active_tab()
        if tab: tab.clear_spectrum_regions()

    def clear_pv_cuts(self):
        tab = self.get_active_tab()
        if tab: tab.clear_pv_cuts()

    def load_file(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog 
        file_name, _ = QFileDialog.getOpenFileName(self, "Open FITS Cube", "", "FITS (*.fits *.fits.gz);;All (*)", options=options)
        if file_name:
            tab = self.get_active_tab()
            self.statusBar().showMessage(f"Loading: {file_name.split('/')[-1]}...")
            QApplication.processEvents()
            
            if tab.cube_clean is None:
                success = tab.load_file(file_name)
            else:
                self.add_new_tab()
                tab = self.get_active_tab()
                success = tab.load_file(file_name)
                
            if success:
                short_name = file_name.split('/')[-1]
                self.tabs.setTabText(self.tabs.currentIndex(), short_name)
                self.setWindowTitle(f"CubeX - {short_name}")
                self.statusBar().showMessage("File loaded successfully.")
            else:
                self.statusBar().showMessage("Load failed.")

    def close_cube(self):
        tab = self.get_active_tab()
        if tab:
            tab.close_file()
            self.tabs.setTabText(self.tabs.currentIndex(), "Untitled")
            self.statusBar().showMessage("Cube closed.")

    def export_spectrum(self):
        tab = self.get_active_tab()
        if tab and tab.cube_clean is not None:
            options = QFileDialog.Options() | QFileDialog.DontUseNativeDialog
            file_name, _ = QFileDialog.getSaveFileName(self, "Save Spectrum CSV", "spectrum.csv", "CSV Files (*.csv)", options=options)
            if file_name:
                try:
                    sort_idx = np.argsort(tab.v_axis)
                    v_sorted = tab.v_axis[sort_idx]
                    spec_sorted = tab.spectrum_curve.yData 
                    with open(file_name, mode='w', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow(["Velocity (km/s)", f"Flux ({tab.spec_unit})"])
                        for v, f in zip(v_sorted, spec_sorted): writer.writerow([v, f])
                    self.statusBar().showMessage(f"Spectrum saved to {file_name}")
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Failed to save file:\n{str(e)}")
        else:
            QMessageBox.warning(self, "No Data", "No cube loaded to export.")

    def export_spectrum_fits(self):
        tab = self.get_active_tab()
        if not tab or tab.cube_clean is None: return
        
        options = QFileDialog.Options() | QFileDialog.DontUseNativeDialog
        filename, _ = QFileDialog.getSaveFileName(self, "Save Spectrum FITS", "spectrum.fits", "FITS Files (*.fits)", options=options)
        if filename:
            try:
                sort_idx = np.argsort(tab.v_axis)
                v_sorted = tab.v_axis[sort_idx]
                spec_sorted = tab.spectrum_curve.yData 
                
                dv = v_sorted[1] - v_sorted[0] if len(v_sorted) > 1 else 1.0
                
                hdu = fits.PrimaryHDU(spec_sorted)
                hdu.header['BUNIT'] = tab.spec_unit
                hdu.header['CTYPE1'] = 'VRAD'
                hdu.header['CUNIT1'] = 'km/s'
                hdu.header['CRPIX1'] = 1
                hdu.header['CRVAL1'] = v_sorted[0]
                hdu.header['CDELT1'] = dv
                
                if tab.raw_header is not None and 'RESTFRQ' in tab.raw_header:
                    hdu.header['RESTFRQ'] = tab.raw_header['RESTFRQ']
                    
                hdu.writeto(filename, overwrite=True)
                self.statusBar().showMessage(f"Saved Spectrum FITS: {filename}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save FITS:\n{str(e)}")

    def export_spectrum_pdf(self):
        tab = self.get_active_tab()
        if not tab or tab.cube_clean is None: return
        
        options = QFileDialog.Options() | QFileDialog.DontUseNativeDialog
        filename, _ = QFileDialog.getSaveFileName(self, "Save Spectrum PDF", "spectrum.pdf", "PDF Files (*.pdf)", options=options)
        if filename:
            try:
                sort_idx = np.argsort(tab.v_axis)
                v_sorted = tab.v_axis[sort_idx]
                spec_sorted = tab.spectrum_curve.yData 
                
                # Standard Matplotlib PDF Export
                fig, ax = plt.subplots(figsize=(10, 5))
                
                ax.step(v_sorted, spec_sorted, color='#3498db', where='mid', linewidth=1.5)
                ax.set_xlabel('Radio Velocity (km/s)')
                ax.set_ylabel(f'Flux ({tab.spec_unit})')
                
                for item in tab.catalog_overlay_items:
                    if isinstance(item, pg.InfiniteLine):
                        ax.axvline(x=item.pos().x(), color='#e74c3c', linestyle='--', linewidth=1)
                    elif isinstance(item, pg.TextItem):
                        ax.text(item.pos().x(), np.nanmax(spec_sorted), item.toPlainText(), 
                                color='#e74c3c', rotation=90, verticalalignment='top', horizontalalignment='right')
                
                plt.savefig(filename, format='pdf', bbox_inches='tight')
                plt.close(fig)
                self.statusBar().showMessage(f"Saved Spectrum PDF: {filename}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save PDF:\n{str(e)}")

    def export_fits_active(self):
        tab = self.get_active_tab()
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
        
        options = QFileDialog.Options() | QFileDialog.DontUseNativeDialog
        filename, _ = QFileDialog.getSaveFileName(self, "Save FITS", f"{name}.fits", "FITS Files (*.fits)", options=options)
        if filename:
            try:
                export_data = data.T 
                hdu = fits.PrimaryHDU(export_data)
                hdu.header['BUNIT'] = bunit
                if tab.raw_header is not None:
                    for key in ['CTYPE1', 'CRVAL1', 'CDELT1', 'CRPIX1', 'CUNIT1', 
                                'CTYPE2', 'CRVAL2', 'CDELT2', 'CRPIX2', 'CUNIT2', 'RESTFRQ', 'OBJECT']:
                        if key in tab.raw_header:
                            hdu.header[key] = tab.raw_header[key]
                hdu.writeto(filename, overwrite=True)
                self.statusBar().showMessage(f"Saved FITS: {filename}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save FITS:\n{str(e)}")

    def export_pdf_active(self):
        tab = self.get_active_tab()
        if not tab or tab.cube_clean is None: return
        target_id = tab.last_clicked_panel_id
        
        if target_id == 'spectrum': return
        
        if target_id == 'channel':
            data = tab.get_current_channel_data()
            title = f"Channel Map ({tab.slider_channel.value()})"
            cbar_label = f"Flux ({tab.display_unit})"
            cmap_name = self.current_cmap
            levels = tab.ch_levels
        else:
            p = tab.panels[target_id]
            data = p['current_data']
            mtype = p['combo'].currentText()
            title = mtype
            cbar_label = p['view'].ui.histogram.axis.labelText
            is_vel = ("Moment 1" in mtype) or ("Moment 9" in mtype)
            cmap_name = 'bwr' if is_vel else self.current_cmap
            levels = p['view'].ui.histogram.getLevels()

        if data is None: return
        
        options = QFileDialog.Options() | QFileDialog.DontUseNativeDialog
        filename, _ = QFileDialog.getSaveFileName(self, "Save PDF", f"export.pdf", "PDF Files (*.pdf)", options=options)
        if filename:
            try:
                plot_data = data.T
                
                # Standard Matplotlib PDF Export
                if tab.is_absolute_wcs and tab.wcs_2d is not None:
                    fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={'projection': tab.wcs_2d})
                    
                    im = ax.imshow(plot_data, origin='lower', cmap=cmap_name, 
                                   vmin=levels[0], vmax=levels[1])
                                   
                    ax.set_xlabel('Right Ascension')
                    ax.set_ylabel('Declination')
                    
                    if target_id in tab.contour_params and tab.contour_params[target_id]:
                        params = tab.contour_params[target_id]
                        if params['mode'] == 'auto':
                            valid = data[~np.isnan(data) & ~np.isinf(data)]
                            if len(valid) > 0:
                                min_v, max_v = np.nanmin(valid), np.nanmax(valid)
                                c_levels = np.linspace(min_v, max_v, params['n'] + 2)[1:-1]
                                ax.contour(plot_data, levels=c_levels, colors='#2ecc71', linewidths=1.0)
                        else:
                            ax.contour(plot_data, levels=params['levels'], colors='#2ecc71', linewidths=1.0)

                else:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    
                    extent = [tab.nx/2 * tab.pix_scale_arcsec, -tab.nx/2 * tab.pix_scale_arcsec, 
                              -tab.ny/2 * tab.pix_scale_arcsec, tab.ny/2 * tab.pix_scale_arcsec]
                    
                    im = ax.imshow(plot_data, origin='lower', cmap=cmap_name, 
                                   vmin=levels[0], vmax=levels[1], extent=extent)
                                   
                    ax.set_xlabel('RA offset (arcsec)')
                    ax.set_ylabel('Dec offset (arcsec)')
                    
                    if target_id in tab.contour_params and tab.contour_params[target_id]:
                        params = tab.contour_params[target_id]
                        if params['mode'] == 'auto':
                            valid = data[~np.isnan(data) & ~np.isinf(data)]
                            if len(valid) > 0:
                                min_v, max_v = np.nanmin(valid), np.nanmax(valid)
                                c_levels = np.linspace(min_v, max_v, params['n'] + 2)[1:-1]
                                ax.contour(plot_data, levels=c_levels, colors='#2ecc71', linewidths=1.0, extent=extent)
                        else:
                            ax.contour(plot_data, levels=params['levels'], colors='#2ecc71', linewidths=1.0, extent=extent)

                cbar = fig.colorbar(im, ax=ax)
                cbar.set_label(cbar_label)
                
                plt.savefig(filename, format='pdf', bbox_inches='tight')
                plt.close(fig)
                self.statusBar().showMessage(f"Saved PDF: {filename}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save PDF:\n{str(e)}")

    def show_header(self):
        tab = self.get_active_tab()
        if tab and tab.cube_clean is not None:
            dlg = QDialog(self)
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
            QMessageBox.warning(self, "No Data", "Please load a cube first.")

    def reset_views(self):
        tab = self.get_active_tab()
        if tab:
            tab.view_channel.autoRange()
            tab.plot_widget.autoRange() 
            for p in tab.panels: p['view'].autoRange()

    def show_contour_dialog(self):
        tab = self.get_active_tab()
        if not tab or tab.cube_clean is None: 
            QMessageBox.warning(self, "No Data", "Please load a cube first.")
            return

        target_id = tab.last_clicked_panel_id
        if target_id == 'spectrum':
            QMessageBox.warning(self, "Invalid Panel", "Contours cannot be drawn on the 1D spectrum.")
            return

        if target_id == 'channel':
            name = "Channel Map"
        else:
            name = f"Bottom Panel {target_id + 1} ({tab.panels[target_id]['combo'].currentText()})"

        dlg = ContourDialog(self, tab.contour_params.get(target_id), f"Contours - {name}")
        if dlg.exec_():
            if dlg.action == 'clear':
                tab.contour_params[target_id] = None
            elif dlg.action == 'apply':
                tab.contour_params[target_id] = dlg.result_params

            if target_id == 'channel':
                tab.update_channel_map()
            else:
                tab.update_moment_maps()

    def open_line_catalog(self):
        tab = self.get_active_tab()
        if tab: tab.open_line_catalog()

    def set_colormap(self, cmap_name):
        self.current_cmap = cmap_name
        tab = self.get_active_tab()
        if tab and tab.cube_clean is not None: tab.update_moment_maps() 

    def set_theme(self, theme):
        if theme == 'dark':
            pg.setConfigOption('background', '#121212')
            pg.setConfigOption('foreground', '#e0e0e0')
        else:
            pg.setConfigOption('background', 'w')
            pg.setConfigOption('foreground', 'k')
        QMessageBox.information(self, "Theme Changed", f"Global theme set to {theme}. Open a new tab or window to see changes.")

    def show_manual(self):
        man = """
        <h3>CubeX User Manual</h3>
        <b>1. Loading Data:</b> Use File -> Open to load an ALMA FITS cube.<br><br>
        <b>2. Channel Map:</b> The top-left panel shows individual velocity slices. Use the media controls, slider, or type a velocity to navigate.<br><br>
        <b>3. Extracting Spectra:</b> Select a shape from the 'Extraction Region' dropdown. Draw it on the channel map to see the local spectrum in the top-right panel. <b>Click the shape to select it (turns yellow), then press ESC to delete.</b><br><br>
        <b>4. Building PV Diagrams:</b> Set any bottom-panel dropdown to PV Diagram, then hold CTRL and drag a line on the channel map. Use that panel's Cut selector to choose which slice to display, and the Range selector to switch between the selected spectrum range and the full cube. The default PV range is the selected spectrum range.<br><br>
        <b>5. Line Catalog:</b> Select a velocity range in the Spectrum using the blue handles. Go to Tools -> Query Molecular Line Database to dynamically fetch species from Splatalogue and overlay them on the spectrum.<br><br>
        <b>6. Generating Moments:</b> The bottom panels can show moment maps or PV diagrams. Moment products use the velocity range selected by the blue slider in the Spectrum plot.<br><br>
        <b>7. Threshold Masking (Important!):</b> To clean up noise in your Velocity maps, click the Dropper icon on a moment panel, then click on the dark background in the raw Moment 0 map. This extracts that background noise level and applies it as a 3D cutoff mask!
        """
        QMessageBox.information(self, "Manual", man)

    def show_shortcuts(self):
        sc = """
        <b>Mouse Controls:</b>
        <ul>
        <li><b>Left Click + Drag:</b> Pan around the maps.</li>
        <li><b>Right Click + Drag:</b> Zoom in and out.</li>
        <li><b>Middle Click (or 'A' key):</b> Auto-reset zoom.</li>
        <li><b>Left Click on Spectrum:</b> Instantly jump the channel map to that velocity.</li>
        <li><b>Left Click on Panel:</b> Sets the "Active View" (used for drawing Contours or PDF Export, highlighted in blue).</li>
        <li><b>CTRL + Drag on Channel Map while any bottom panel is set to PV Diagram:</b> Draw a new PV cut.</li>
        </ul>
        <b>Keyboard Controls:</b>
        <ul>
        <li><b>ESC:</b> Deletes the currently selected spatial ROI or PV cut.</li>
        </ul>
        """
        QMessageBox.information(self, "Controls & Shortcuts", sc)

    def show_about(self):
        QMessageBox.about(self, "About CubeX", "<b>CubeX</b><br>A lightweight, real-time ALMA data visualization tool.<br>Powered by PyQt5, PyQtGraph, Matplotlib, Astroquery, and Astropy.")
