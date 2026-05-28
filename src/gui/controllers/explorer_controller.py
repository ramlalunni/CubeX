import numpy as np
from PyQt5.QtCore import QObject, Qt
from PyQt5.QtWidgets import QMessageBox, QCheckBox, QLabel
import pyqtgraph as pg
from spectral_cube import SpectralCube
import astropy.units as u
from astropy.io import fits
from astropy.wcs import WCS
import astropy.constants as const
from src.gui.controllers.workers import MomentWorker
from src.gui.components.graph_panels import make_roi_rotatable_with_ctrl
from src.gui.dialogs import ContourDialog

class ExplorerController(QObject):
    def __init__(self, view):
        super().__init__()
        self.view = view

    def get_velocity_subset(self, use_full_range=False):
        if self.view.cube_clean is None:
            return None, None, None, None
        if use_full_range:
            return self.view.cube_clean, self.view.v_axis, float(np.nanmin(self.view.v_axis)), float(np.nanmax(self.view.v_axis))

        minX, maxX = self.view.region.getRegion()
        search_axis = self.view.v_axis if self.view.v_axis[0] < self.view.v_axis[-1] else self.view.v_axis[::-1]
        idx_min = np.searchsorted(search_axis, minX)
        idx_max = np.searchsorted(search_axis, maxX)
        if self.view.v_axis[0] > self.view.v_axis[-1]:
            idx_min, idx_max = len(self.view.v_axis) - idx_max, len(self.view.v_axis) - idx_min
        if idx_max <= idx_min:
            return None, None, minX, maxX
        return self.view.cube_clean[idx_min:idx_max, :, :], self.view.v_axis[idx_min:idx_max], minX, maxX

    def update_moment_maps(self):
        if self.view.cube_clean is None:
            return
        if getattr(self.view, '_region_dragging', False):
            return

        selected_cube, sub_v, minX, maxX = self.get_velocity_subset(use_full_range=False)
        if selected_cube is None or sub_v is None:
            return

        if self.view._moment_worker is not None and self.view._moment_worker.isRunning():
            self.view._moment_worker.cancel()
            try:
                self.view._moment_worker.result_ready.disconnect(self._on_moment_result)
            except TypeError:
                pass
            self.view._moment_worker.finished.connect(self.view._moment_worker.deleteLater)
            self.view._pending_workers.append(self.view._moment_worker)
            
        self._purge_finished_workers()
        self.view._moment_generation += 1
        current_gen = self.view._moment_generation

        thresh = []
        for i in range(3):
            try:
                thresh.append(float(self.view.panels[i]['input_thresh'].text()))
            except ValueError:
                thresh.append(0.0)

        for i, p in enumerate(self.view.panels):
            mtype = p['combo'].currentText()
            self.view.configure_bottom_panel_controls(p, mtype)
            if mtype != 'PV Diagram':
                self.view.configure_bottom_panel_axes(p, is_pv=False)
                p['plot_item'].setTitle('')
                is_vel = ('Moment 1' in mtype) or ('Moment 9' in mtype)
                self.view.apply_cmap(p['view'], is_vel)
            self.view.update_beam_visualizers('moment', panel_id=i)

        panel_configs = []
        for i, p in enumerate(self.view.panels):
            mtype = p['combo'].currentText()
            cfg = {'mtype': mtype, 'threshold': thresh[i]}

            if mtype == 'PV Diagram':
                cut_name  = p['combo_pv_cut'].currentText()
                active_item = self.view.get_pv_cut_by_name(cut_name)
                if active_item is not None:
                    points = self.view.get_line_roi_points(active_item['roi'])
                    use_full = p['combo_pv_range'].currentText() == 'Full Cube'
                    cfg['pv_points'] = points
                    cfg['pv_cube']   = self.view.cube_clean if use_full else selected_cube
                    cfg['pv_sub_v']  = self.view.v_axis     if use_full else sub_v
                    cfg['pv_width']  = active_item.get('width', 1)
                else:
                    cfg['pv_points'] = None

            panel_configs.append(cfg)

        worker_params = {
            'selected_cube':  selected_cube,
            'sub_v':          sub_v,
            'minX':           minX,
            'maxX':           maxX,
            'nx':             self.view.nx,
            'ny':             self.view.ny,
            'pix_scale_arcsec': self.view.pix_scale_arcsec,
            'display_unit':   self.view.display_unit,
            'panel_configs':  panel_configs,
        }
        self.view._moment_worker = MomentWorker(worker_params, current_gen)
        self.view._moment_worker.result_ready.connect(self._on_moment_result)
        self.view._moment_worker.start()

    def _purge_finished_workers(self):
        alive = []
        for w in self.view._pending_workers:
            try:
                if w.isRunning():
                    alive.append(w)
            except RuntimeError:
                pass
        self.view._pending_workers = alive

    def _on_moment_result(self, results: dict):
        if results['generation'] != self.view._moment_generation:
            return
        if self.view.cube_clean is None:
            return

        self.view.current_m0_raw = results['m0_raw']
        minX = results['minX']
        maxX = results['maxX']

        pos_tup   = ((self.view.nx / 2) * self.view.pix_scale_arcsec, -(self.view.ny / 2) * self.view.pix_scale_arcsec)
        scale_tup = (-self.view.pix_scale_arcsec, self.view.pix_scale_arcsec)

        for p, pr in zip(self.view.panels, results['panel_results']):
            mtype = pr['mtype']
            panel_id = self.view.panels.index(p)

            if mtype == 'PV Diagram':
                if pr.get('data') is None:
                    self.view.clear_panel_pv_diagram(p)
                else:
                    pv_sorted = pr['data']
                    offsets   = pr['offsets']
                    v_sorted  = pr['v_sorted']
                    levels    = pr['levels']
                    dx, dv    = pr['dx'], pr['dv']

                    self.view.configure_bottom_panel_axes(p, is_pv=True)
                    p['view'].ui.histogram.gradient.loadPreset('turbo')
                    p['view'].ui.histogram.axis.setLabel(f"Flux ({self.view.display_unit})")
                    p['plot_item'].setTitle('PV Diagram')

                    p['current_data']       = pv_sorted
                    p['pv_offset_axis']     = offsets
                    p['pv_velocity_axis']   = v_sorted
                    p['unit']               = self.view.display_unit

                    p['view'].setImage(
                        pv_sorted,
                        autoLevels=False,
                        autoHistogramRange=False,
                        levels=levels,
                        scale=(dx, dv),
                        pos=(0.0, v_sorted[0]),
                    )
                    self.view.draw_contours(panel_id, p['view'], None)

            else:
                data     = pr.get('data')
                levels   = pr.get('levels', (0.0, 1.0))
                unit_str = pr.get('unit_str', '')

                view = p['view']
                if 'Moment -1' in mtype:
                    view.ui.histogram.axis.setLabel(f"Mean Flux ({unit_str})")
                elif 'Moment 0' in mtype:
                    view.ui.histogram.axis.setLabel(f"Flux ({unit_str})")
                elif 'Moment 1' in mtype:
                    view.ui.histogram.axis.setLabel('Velocity (km/s)')
                elif 'Moment 2' in mtype:
                    view.ui.histogram.axis.setLabel('Dispersion (km/s)')
                elif 'Moment 8' in mtype:
                    view.ui.histogram.axis.setLabel(f"Peak Flux ({unit_str})")
                elif 'Moment 9' in mtype:
                    view.ui.histogram.axis.setLabel('Peak Velocity (km/s)')

                p['current_data'] = data
                p['unit']         = unit_str

                if data is not None:
                    view.setImage(
                        data,
                        autoLevels=False,
                        autoHistogramRange=False,
                        levels=levels,
                        scale=scale_tup,
                        pos=pos_tup,
                    )
                    self.view.draw_contours(panel_id, view, data)
                    self.view.update_beam_visualizers('moment', panel_id=panel_id)

    def load_file(self, file_name):
        try:
            self.view.current_file_name = file_name
            self.view.is_2d_image = False
            try:
                # Step 1: Read the cube without immediate conversion
                sc = SpectralCube.read(file_name)
                
                # Step 2: Implement Central Frequency Fallback
                self.view.rest_freq_hz = sc.header.get('RESTFRQ', sc.header.get('RESTFREQ', None))
                
                if self.view.rest_freq_hz is None:
                    raw_axis = sc.spectral_axis
                    central_idx = len(raw_axis) // 2
                    try:
                        # Convert to Hz safely if axis is frequency
                        self.view.rest_freq_hz = raw_axis[central_idx].to(u.Hz).value
                    except Exception:
                        # Blind fallback if axis cannot be explicitly converted
                        self.view.rest_freq_hz = raw_axis[central_idx].value
                    
                    # Inject into the header to prevent downstream crashes
                    if hasattr(sc, '_header'):
                        sc._header['RESTFRQ'] = self.view.rest_freq_hz
                        
                # Sync UI Text Box
                if hasattr(self.view, 'input_ref_freq') and self.view.rest_freq_hz is not None:
                    self.view.input_ref_freq.blockSignals(True)
                    self.view.input_ref_freq.setText(f"{self.view.rest_freq_hz / 1e9:.6f}")
                    self.view.input_ref_freq.setCursorPosition(0)
                    self.view.input_ref_freq.blockSignals(False)
                        
                # Step 3: Now safely convert using the rest_value argument
                sc = sc.with_spectral_unit(u.km / u.s, velocity_convention='radio', rest_value=self.view.rest_freq_hz * u.Hz)
            except Exception as e:
                with fits.open(file_name) as hdul:
                    data = np.squeeze(hdul[0].data)
                    if data.ndim == 2:
                        self.view.is_2d_image = True
                        class MockData:
                            def __init__(self, d): self._d = d
                            @property
                            def value(self): return np.expand_dims(self._d, axis=0)
                            def __getitem__(self, key): return self
                        class MockAxis:
                            @property
                            def value(self): return np.array([0.0])
                        class MockCube:
                            def __init__(self, d, h):
                                self.header = h
                                self.filled_data = MockData(d)
                                self.spectral_axis = MockAxis()
                        sc = MockCube(data, hdul[0].header)
                    else:
                        raise e
            
            raw_bunit = sc.header.get('BUNIT', 'Unknown').strip()
            self.view.display_unit = raw_bunit
            self.view.spec_unit = raw_bunit
            self.view.raw_header = sc.header.copy() 
            self.view.fits_header_text = sc.header.tostring(sep='\n')
            
            self.view.rest_freq_hz = sc.header.get('RESTFRQ', sc.header.get('RESTFREQ', None))
            
            try: self.view.wcs_2d = WCS(self.view.raw_header).celestial
            except Exception: self.view.wcs_2d = None
                
            cdelt2 = sc.header.get('CDELT2', None)
            cdelt1 = sc.header.get('CDELT1', None)
            self.view.pix_scale_arcsec = abs(float(cdelt2)) * 3600.0 if cdelt2 else 1.0 
            
            raw_cube = sc.filled_data[:].value
            self.view.v_axis = sc.spectral_axis.value
            
            self.view.cube_clean = np.transpose(raw_cube, (0, 2, 1))
            self.view.nx, self.view.ny = self.view.cube_clean.shape[1], self.view.cube_clean.shape[2]
            
            # Multi-Beam Header Parsing
            self.view.bmaj_array = None
            self.view.bmin_array = None
            self.view.bpa_array = None
            self.view.pixels_per_beam_array = None
            self.view.beam_omega_array = None
            self.view.freq_array = None
            self.view.can_convert_units = True
            
            # Extract frequency array
            try:
                if sc.header.get('CTYPE3', '').startswith('FREQ'):
                    freq_crval = sc.header.get('CRVAL3')
                    freq_cdelt = sc.header.get('CDELT3')
                    freq_crpix = sc.header.get('CRPIX3')
                    self.view.freq_array = freq_crval + (np.arange(len(self.view.v_axis)) - (freq_crpix - 1)) * freq_cdelt
                elif sc.header.get('RESTFRQ') or sc.header.get('RESTFREQ'):
                    rf = sc.header.get('RESTFRQ', sc.header.get('RESTFREQ'))
                    self.view.freq_array = rf * (1.0 - (self.view.v_axis * 1000.0) / const.c.value)
                else:
                    self.view.can_convert_units = False
            except Exception:
                self.view.can_convert_units = False

            try:
                with fits.open(file_name) as hdul:
                    is_multibeam = sc.header.get('CASAMBM', 'F') == 'T' or 'BEAMS' in hdul
                    
                    if is_multibeam and 'BEAMS' in hdul:
                        beams_data = hdul['BEAMS'].data
                        bmaj_raw = beams_data['BMAJ']
                        bmin_raw = beams_data['BMIN']
                        try:
                            bpa_raw = beams_data['BPA']
                        except KeyError:
                            bpa_raw = None
                        
                        if len(bmaj_raw) == len(self.view.v_axis):
                            bmaj_unit = hdul['BEAMS'].columns['BMAJ'].unit
                            if bmaj_unit and 'deg' in str(bmaj_unit).lower():
                                self.view.bmaj_array = bmaj_raw
                                self.view.bmin_array = bmin_raw
                            else:
                                self.view.bmaj_array = bmaj_raw / 3600.0
                                self.view.bmin_array = bmin_raw / 3600.0
                            if bpa_raw is not None:
                                self.view.bpa_array = bpa_raw
                            else:
                                bpa = sc.header.get('BPA', 0.0)
                                self.view.bpa_array = np.full(len(self.view.v_axis), bpa)
                        else:
                            bmaj = sc.header.get('BMAJ')
                            bmin = sc.header.get('BMIN')
                            if bmaj and bmin:
                                self.view.bmaj_array = np.full(len(self.view.v_axis), bmaj)
                                self.view.bmin_array = np.full(len(self.view.v_axis), bmin)
                            else:
                                self.view.can_convert_units = False
                            bpa = sc.header.get('BPA', 0.0)
                            self.view.bpa_array = np.full(len(self.view.v_axis), bpa)
                    else:
                        bmaj = sc.header.get('BMAJ')
                        bmin = sc.header.get('BMIN')
                        if bmaj and bmin:
                            self.view.bmaj_array = np.full(len(self.view.v_axis), bmaj)
                            self.view.bmin_array = np.full(len(self.view.v_axis), bmin)
                            bpa = sc.header.get('BPA', 0.0)
                            self.view.bpa_array = np.full(len(self.view.v_axis), bpa)
                        else:
                            self.view.can_convert_units = False
            except Exception:
                bmaj = sc.header.get('BMAJ')
                bmin = sc.header.get('BMIN')
                if bmaj and bmin:
                    self.view.bmaj_array = np.full(len(self.view.v_axis), bmaj)
                    self.view.bmin_array = np.full(len(self.view.v_axis), bmin)
                    bpa = sc.header.get('BPA', 0.0)
                    self.view.bpa_array = np.full(len(self.view.v_axis), bpa)
                else:
                    self.view.can_convert_units = False
                    
            if self.view.bmaj_array is not None and self.view.bmin_array is not None and cdelt1 and cdelt2:
                omega_pix = abs(cdelt1 * cdelt2) * (u.deg ** 2)
                self.view.omega_pix_sr = omega_pix.to(u.sr)
                
                omega_beam = (np.pi * self.view.bmaj_array * self.view.bmin_array) / (4.0 * np.log(2.0)) * (u.deg ** 2)
                self.view.omega_beam_sr = omega_beam.to(u.sr)
                
                self.view.n_beam_array = self.view.omega_beam_sr / self.view.omega_pix_sr
                self.view.pixels_per_beam = self.view.n_beam_array[0].value
            else:
                self.view.omega_pix_sr = 1.0 * u.sr
                self.view.omega_beam_sr = np.ones(len(self.view.v_axis)) * u.sr
                self.view.n_beam_array = np.ones(len(self.view.v_axis)) * u.dimensionless_unscaled
                self.view.pixels_per_beam = 1.0
                self.view.can_convert_units = False
            
            native_label = f"Native ({raw_bunit})" if raw_bunit != 'Unknown' else "Native"
            unit_lower = raw_bunit.replace(" ", "").lower()
            
            new_units = [native_label, "Jy"]
            if not ("k" == unit_lower or "kelvin" in unit_lower):
                new_units.append("K")
            if not ("jy" in unit_lower and "pixel" not in unit_lower and "pix" not in unit_lower):
                new_units.append("Jy/beam")
                
            self.view.combo_spec_unit.blockSignals(True)
            self.view.combo_spec_unit.clear()
            self.view.combo_spec_unit.addItems(new_units)
            self.view.combo_spec_unit.blockSignals(False)
            
            if not self.view.can_convert_units:
                self.view.combo_spec_unit.blockSignals(True)
                self.view.combo_spec_unit.setCurrentIndex(0)
                self.view.combo_spec_unit.blockSignals(False)
                
                for i in range(1, self.view.combo_spec_unit.count()):
                    self.view.combo_spec_unit.model().item(i).setEnabled(False)
                self.view.combo_spec_unit.setToolTip("Conversion disabled: Missing beam or frequency metadata in FITS.")
                
                sum_idx = self.view.combo_spec_stat.findText("Flux Density")
                if sum_idx != -1:
                    self.view.combo_spec_stat.model().item(sum_idx).setEnabled(False)
            else:
                self.view.combo_spec_unit.setToolTip("")
                sum_idx = self.view.combo_spec_stat.findText("Flux Density")
                if sum_idx != -1:
                    self.view.combo_spec_stat.model().item(sum_idx).setEnabled(True)
            
            self.view.combo_spec_stat.blockSignals(True)
            self.view.combo_spec_stat.setCurrentText("Mean" if not self.view.can_convert_units else "Mean")
            self.view.combo_spec_stat.blockSignals(False)
            self.view._update_spectrum_state_machine()

            self.view.plot_widget.setLabel('left', f'Mean Flux ({self.view.display_unit})')
            self.view.view_channel.ui.histogram.axis.setLabel(f"Flux ({self.view.display_unit})")
            
            peak_flux = np.nanmax(self.view.cube_clean)
            self.view.ch_levels = (0, peak_flux if peak_flux > 0 else 1.0)

            self.view.slider_channel.setRange(0, len(self.view.v_axis) - 1)
            mean_spectrum = np.nanmean(self.view.cube_clean, axis=(1, 2))
            brightest_ch = int(np.nanargmax(mean_spectrum))
            self.view.slider_channel.setValue(brightest_ch)
            self.view.update_channel_map()
            self.view.v_line.show()
            if hasattr(self.view, 'smooth_active_line'):
                self.view.smooth_active_line.show()

            v_min, v_max = np.nanmin(self.view.v_axis), np.nanmax(self.view.v_axis)
            peak_vel = self.view.v_axis[brightest_ch]
            half_span = 0.1 * (v_max - v_min)
            r_lo = max(v_min, peak_vel - half_span)
            r_hi = min(v_max, peak_vel + half_span)
            if r_hi - r_lo < 0.02 * (v_max - v_min):
                r_lo = v_min + 0.4 * (v_max - v_min)
                r_hi = v_min + 0.6 * (v_max - v_min)
            self.view.region.setRegion([r_lo, r_hi])
            self.view.region.show()
            if hasattr(self.view, 'smooth_velocity_region'):
                self.view.smooth_velocity_region.show()
            self.view.combo_roi.blockSignals(True)
            self.view.combo_roi.setCurrentText("Whole Map")
            self.view.combo_roi.blockSignals(False)
            self.view.roi_selected = False
            self.view.change_roi("Whole Map")

            self.view.combo_spec_stat.blockSignals(True)
            self.view.combo_spec_stat.setCurrentText("Mean")
            self.view.combo_spec_stat.blockSignals(False)
            self.view.smoothing_params = None
            if hasattr(self.view, 'spectrum_tabs'):
                idx = self.view.spectrum_tabs.indexOf(self.view.plot_widget_smooth)
                if idx != -1:
                    self.view.spectrum_tabs.removeTab(idx)
                self.view.spectrum_tabs.tabBar().hide()
                self.view.spectrum_tabs.setCurrentWidget(self.view.plot_widget)
            self.view.spectrum_curve_smooth.setData([], [])
            for p in self.view.panels:
                p['input_thresh'].setText("0.000")

            self.view.update_moment_maps()
            
            self.view.update_wcs_mode(self.view.parent_window.is_absolute_wcs)
            self.view.set_2d_ui_state(self.view.is_2d_image)
            self.view.parent_window.update_menu_states()
            
            return True
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load cube:\n{str(e)}")
            return False


    def _run_spectral_stats_calc(self, popup):
        selected_rois_1d = self.view._get_popup_selected_boxes(popup)
        selected_apertures = self.view._get_popup_selected_apertures(popup)

        # Determine which spatial apertures to calculate for — completely independent of panel
        if not selected_apertures:
            # Whole map
            apertures_to_calc = [{"name": "Whole Map", "roi": None}]
        else:
            apertures_to_calc = [{"name": r["name"], "roi": r["roi"]} for r in selected_apertures]

        if not selected_rois_1d:
            popup.lbl_result.setText("---")
            return

        calc_types = [name for name, cb in popup.stat_checkboxes.items() if cb.isChecked()]
        unit = getattr(self.view, 'spec_unit', '')
        results_html = []


        # Determine which plot widget is active to grab the correct curves
        is_smooth_active = False
        if getattr(self.view, 'spectrum_tabs', None) is not None and getattr(self.view, 'plot_widget_smooth', None) is not None:
            if self.view.spectrum_tabs.currentWidget() == self.view.plot_widget_smooth:
                is_smooth_active = True

        for ap in apertures_to_calc:
            name = ap["name"]
            
            # Extract spectrum directly from the active plot's pre-converted data
            curve = None
            if name == "Whole Map":
                curve = self.view.spectrum_curve_smooth if is_smooth_active else getattr(self.view, 'spectrum_curve', None)
            else:
                curves_dict = self.view.spectrum_curves_smooth if is_smooth_active else getattr(self.view, 'spectrum_curves', {})
                curve = curves_dict.get(name)
                
            if curve is None:
                results_html.append(f"<b>{name}:</b> Could not locate plot data")
                continue
                
            x_edges, flux = curve.getData()
            if x_edges is None or flux is None or len(flux) == 0:
                results_html.append(f"<b>{name}:</b> Plot data is empty")
                continue
                
            # Recover the centers from the stepMode edges (which exactly matches the UI's sorted v_axis)
            v_axis = (x_edges[:-1] + x_edges[1:]) / 2.0

            # Hardcode physics to Radio Velocity for dx and peak location calculations
            try:
                sort_idx = np.argsort(self.view.v_axis)
                freq_q = self.view.freq_array[sort_idx] * u.Hz
                rest_q = self.view.rest_freq_hz * u.Hz
                radio_v_axis = freq_q.to(u.km / u.s, equivalencies=u.doppler_radio(rest_q)).value
            except Exception:
                radio_v_axis = v_axis

            # Build mask from selected 1D velocity boxes (using displayed axis for UI mapping)
            combined_mask = np.zeros_like(v_axis, dtype=bool)
            for roi in selected_rois_1d:
                if hasattr(roi, 'getData'):
                    x_data, _ = roi.getData()
                    if x_data is None or len(x_data) < 2:
                        continue
                    min_v, max_v = min(x_data), max(x_data)
                else:
                    pos = roi.pos(); size = roi.size()
                    min_v, max_v = pos.x(), pos.x() + size.x()
                if min_v > max_v: min_v, max_v = max_v, min_v
                combined_mask |= (v_axis >= min_v) & (v_axis <= max_v)

            valid_flux = flux[combined_mask]
            if len(valid_flux) == 0:
                results_html.append(f"<b>{name}:</b> No data in selected range")
                continue

            stats_lines = [f"<b style='color:#89b4fa'>{name}</b>"]
            valid_v = radio_v_axis[combined_mask]
            dv = abs(valid_v[1] - valid_v[0]) if len(valid_v) > 1 else 1.0

            for calc in calc_types:
                calc = calc.strip()
                if calc == "Integrated Intensity":
                    val = np.nansum(valid_flux) * dv
                    stats_lines.append(f"&nbsp;&nbsp;Integrated Intensity: <b>{val:.4f}</b> {unit} km/s")
                elif calc == "RMS":
                    val = np.sqrt(np.nanmean(valid_flux**2))
                    stats_lines.append(f"&nbsp;&nbsp;RMS: <b>{val:.4f}</b> {unit}")
                elif calc == "Peak (Max)":
                    val = np.nanmax(valid_flux)
                    vpeak = valid_v[np.nanargmax(valid_flux)]
                    stats_lines.append(f"&nbsp;&nbsp;Peak: <b>{val:.4f}</b> {unit} @ {vpeak:.2f} km/s")
                elif calc == "Min":
                    val = np.nanmin(valid_flux)
                    vmin = valid_v[np.nanargmin(valid_flux)]
                    stats_lines.append(f"&nbsp;&nbsp;Min: <b>{val:.4f}</b> {unit} @ {vmin:.2f} km/s")
                elif calc == "Mean":
                    val = np.nanmean(valid_flux)
                    stats_lines.append(f"&nbsp;&nbsp;Mean: <b>{val:.4f}</b> {unit}")
                elif calc == "Median":
                    val = np.nanmedian(valid_flux)
                    stats_lines.append(f"&nbsp;&nbsp;Median: <b>{val:.4f}</b> {unit}")
                elif calc == "SNR (Peak/RMS)":
                    peak = np.nanmax(np.abs(valid_flux))
                    rms = np.sqrt(np.nanmean(valid_flux**2))
                    snr = peak / rms if rms > 0 else float('nan')
                    stats_lines.append(f"&nbsp;&nbsp;SNR: <b>{snr:.2f}</b>")
                elif calc == "Sum":
                    val = np.nansum(valid_flux)
                    stats_lines.append(f"&nbsp;&nbsp;Sum: <b>{val:.4f}</b> {unit}")
                elif calc == "Std. Deviation":
                    val = np.nanstd(valid_flux)
                    stats_lines.append(f"&nbsp;&nbsp;Std. Dev.: <b>{val:.4f}</b> {unit}")

            results_html.append("<br>".join(stats_lines))

        if len(apertures_to_calc) <= 1 and self.view.contour_overlays:
            aperture_roi = apertures_to_calc[0]["roi"]
            for ov in self.view.contour_overlays:
                if ov['is_static'] or ov['v_axis'] is None:
                    continue
                ov_name = ov['name']
                ov_color = ov['options']['color']
                ov_unit = getattr(self.view, 'display_unit', '')

                if aperture_roi is None:
                    ov_sub_data = ov['cube']
                else:
                    ov_sub_data = aperture_roi.getArrayRegion(ov['cube'], self.view.view_channel.getImageItem(), axes=(1, 2))

                if "Max" in self.view.combo_spec_stat.currentText():
                    ov_spec = np.nanmax(ov_sub_data, axis=(1, 2))
                elif "Sum" in self.view.combo_spec_stat.currentText() or "Flux Density" in self.view.combo_spec_stat.currentText():
                    ov_spec = np.nansum(ov_sub_data, axis=(1, 2))
                else:
                    ov_spec = np.nanmean(ov_sub_data, axis=(1, 2))

                ov_v = ov['v_axis']
                ov_sort = np.argsort(ov_v)
                ov_vs, ov_ss = ov_v[ov_sort], ov_spec[ov_sort]

                ov_combined_mask = np.zeros_like(ov_vs, dtype=bool)
                for roi in selected_rois_1d:
                    pos = roi.pos(); size = roi.size()
                    min_v, max_v = pos.x(), pos.x() + size.x()
                    if min_v > max_v: min_v, max_v = max_v, min_v
                    ov_combined_mask |= (ov_vs >= min_v) & (ov_vs <= max_v)

                ov_valid_flux = ov_ss[ov_combined_mask]
                if len(ov_valid_flux) == 0:
                    continue

                ov_dv = abs(ov_vs[1] - ov_vs[0]) if len(ov_vs) > 1 else 1.0
                ov_valid_v = ov_vs[ov_combined_mask]

                ov_stats_lines = [f"<b style='color:{ov_color}'>{ov_name} (overlay)</b>"]
                for calc in calc_types:
                    calc = calc.strip()
                    if calc == "Integrated Intensity":
                        val = np.nansum(ov_valid_flux) * ov_dv
                        ov_stats_lines.append(f"&nbsp;&nbsp;Integrated Intensity: <b>{val:.4f}</b> {ov_unit} km/s")
                    elif calc == "RMS":
                        val = np.sqrt(np.nanmean(ov_valid_flux**2))
                        ov_stats_lines.append(f"&nbsp;&nbsp;RMS: <b>{val:.4f}</b> {ov_unit}")
                    elif calc == "Peak (Max)":
                        val = np.nanmax(ov_valid_flux)
                        vpeak = ov_valid_v[np.nanargmax(ov_valid_flux)]
                        ov_stats_lines.append(f"&nbsp;&nbsp;Peak: <b>{val:.4f}</b> {ov_unit} @ {vpeak:.2f} km/s")
                    elif calc == "Min":
                        val = np.nanmin(ov_valid_flux)
                        vmin = ov_valid_v[np.nanargmin(ov_valid_flux)]
                        ov_stats_lines.append(f"&nbsp;&nbsp;Min: <b>{val:.4f}</b> {ov_unit} @ {vmin:.2f} km/s")
                    elif calc == "Mean":
                        val = np.nanmean(ov_valid_flux)
                        ov_stats_lines.append(f"&nbsp;&nbsp;Mean: <b>{val:.4f}</b> {ov_unit}")
                    elif calc == "Median":
                        val = np.nanmedian(ov_valid_flux)
                        ov_stats_lines.append(f"&nbsp;&nbsp;Median: <b>{val:.4f}</b> {ov_unit}")
                    elif calc == "SNR (Peak/RMS)":
                        peak = np.nanmax(np.abs(ov_valid_flux))
                        rms_ov = np.sqrt(np.nanmean(ov_valid_flux**2))
                        snr = peak / rms_ov if rms_ov > 0 else float('nan')
                        ov_stats_lines.append(f"&nbsp;&nbsp;SNR: <b>{snr:.2f}</b>")
                    elif calc == "Sum":
                        val = np.nansum(ov_valid_flux)
                        ov_stats_lines.append(f"&nbsp;&nbsp;Sum: <b>{val:.4f}</b> {ov_unit}")
                    elif calc == "Std. Deviation":
                        val = np.nanstd(ov_valid_flux)
                        ov_stats_lines.append(f"&nbsp;&nbsp;Std. Dev.: <b>{val:.4f}</b> {ov_unit}")

                results_html.append("<br>".join(ov_stats_lines))

        popup.lbl_result.setText("<br><br>".join(results_html) if results_html else "---")
        self.view.lbl_region_result.setText("---")


    def get_pv_cut_by_name(self, name):
        for item in self.view.pv_cuts:
            if item["name"] == name:
                return item
        return None

    def get_selected_pv_cut_name(self):
        if not self.view.pv_cuts_to_delete:
            return None
        for item in self.view.pv_cuts:
            if item["roi"] == self.view.pv_cuts_to_delete[-1]:
                return item["name"]
        return None

    def set_selected_pv_cut(self, name):
        self.view.pv_cuts_to_delete.clear()
        for item in self.view.pv_cuts:
            is_selected = item["name"] == name
            if is_selected:
                self.view.pv_cuts_to_delete.append(item["roi"])
            item["roi"].setPen(pg.mkPen('m', width=3) if is_selected else pg.mkPen('c', width=2))
            direction_item = item.get("direction_item")
            if direction_item is not None:
                direction_item.setPen(pg.mkPen('#f1c40f' if is_selected else '#f7dc6f', width=3 if is_selected else 2))

    def refresh_all_pv_cut_combos(self):
        cut_names = [item["name"] for item in self.view.pv_cuts]
        combos = []
        if hasattr(self.view, 'combo_pv_cuts'):
            combos.append(self.view.combo_pv_cuts)
        combos.extend(panel['combo_pv_cut'] for panel in self.view.panels)

        for combo in combos:
            current = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            combo.addItem("None")
            for name in cut_names:
                combo.addItem(name)
            combo.setCurrentText(current if current in cut_names else "None")
            combo.blockSignals(False)

    def on_panel_pv_cut_selected(self, panel_id):
        name = self.view.panels[panel_id]['combo_pv_cut'].currentText()
        if name != "None":
            self.view.set_selected_pv_cut(name)
        self.view.update_moment_maps()

    def delete_panel_pv_cut(self, panel_id):
        name = self.view.panels[panel_id]['combo_pv_cut'].currentText()
        if name == "None":
            return
        self.view.set_selected_pv_cut(name)
        self.view.delete_selected_pv_cuts()

    def clear_panel_pv_diagram(self, panel):
        panel['current_data'] = None
        panel['pv_offset_axis'] = None
        panel['pv_velocity_axis'] = None
        panel['unit'] = self.view.display_unit
        panel['view'].clear()
        panel['lbl_hover'].setText("")
        self.view.draw_contours(panel['id'], panel['view'], None)

    def update_panel_pv_diagram(self, panel):
        self.view.configure_bottom_panel_axes(panel, is_pv=True)
        panel['view'].ui.histogram.gradient.loadPreset('turbo')
        panel['view'].ui.histogram.axis.setLabel(f"Flux ({self.view.display_unit})")
        panel['plot_item'].setTitle("PV Diagram")

        cut_name = panel['combo_pv_cut'].currentText()
        active_item = self.view.get_pv_cut_by_name(cut_name)
        if active_item is None:
            self.view.clear_panel_pv_diagram(panel)
            return

        use_full_range = panel['combo_pv_range'].currentText() == "Full Cube"
        cube_data, velocity_axis, _, _ = self.view.get_velocity_subset(use_full_range=use_full_range)
        if cube_data is None or velocity_axis is None:
            self.view.clear_panel_pv_diagram(panel)
            return

        offsets, pv_data = self.view.sample_cube_along_line(active_item["roi"], cube_data,
                                                         width=active_item.get('width', 1))
        if offsets is None or pv_data is None or pv_data.size == 0:
            self.view.clear_panel_pv_diagram(panel)
            return

        sort_idx = np.argsort(velocity_axis)
        v_sorted = velocity_axis[sort_idx]
        pv_sorted = pv_data[:, sort_idx]
        valid = pv_sorted[np.isfinite(pv_sorted)]
        if valid.size > 0:
            levels = (float(np.nanmin(valid)), float(np.nanmax(valid)))
            if levels[0] == levels[1]:
                levels = (levels[0], levels[0] + 1.0)
        else:
            levels = (0.0, 1.0)

        dx = offsets[1] - offsets[0] if len(offsets) > 1 else 1.0
        dv = v_sorted[1] - v_sorted[0] if len(v_sorted) > 1 else 1.0

        panel['current_data'] = pv_sorted
        panel['pv_offset_axis'] = offsets
        panel['pv_velocity_axis'] = v_sorted
        panel['unit'] = self.view.display_unit
        panel['view'].setImage(
            pv_sorted,
            autoLevels=False,
            autoHistogramRange=False,
            levels=levels,
            scale=(dx, dv),
            pos=(0.0, v_sorted[0]),
        )
        self.view.draw_contours(panel['id'], panel['view'], None)

    def change_spatial_tool(self, tool, auto_draw=True):
        if self.view.cube_clean is None: return
        
        if tool == "None":
            self.view.plot_spatial_1.hide()
            self.view.plot_spatial_2.hide()
            self.view.stacked_spatial_info.setCurrentIndex(0)
            self.view.lbl_spatial_stats.setText("Choose a tool to begin analysis.")
            self.view.stacked_spatial_info.show()
            return

        if tool == "Point":
            self.view.plot_spatial_1.show()
            self.view.plot_spatial_1.setTitle("X Profile")
            self.view.plot_spatial_2.show()
            self.view.stacked_spatial_info.setCurrentIndex(0)
            self.view.stacked_spatial_info.hide()
        elif tool == "Line":
            self.view.plot_spatial_1.show()
            self.view.plot_spatial_1.setTitle("Spatial Profile")
            self.view.plot_spatial_2.hide()
            self.view.stacked_spatial_info.setCurrentIndex(0)
            self.view.stacked_spatial_info.hide()
        else:
            self.view.plot_spatial_1.hide()
            self.view.plot_spatial_2.hide()
            self.view.stacked_spatial_info.setCurrentIndex(1)
            self.view.spatial_stats_scroll.show()
            self.view.stacked_spatial_info.show()

        if not auto_draw:
            return

        # Auto-draw default shape
        sz = self.view.nx * self.view.pix_scale_arcsec * 0.15
        num = len(self.view.spatial_rois)
        cx, cy = num * sz * 0.1, num * sz * 0.1
        
        new_roi = None
        if tool == "Point":
            sz_pt = self.view.pix_scale_arcsec * 0.1
            new_roi = pg.ROI([cx - sz_pt/2, cy - sz_pt/2], [sz_pt, sz_pt], pen='c')
        elif tool == "Line":
            new_roi = pg.LineSegmentROI([[cx, cy], [cx + sz, cy + sz]], pen='c')
        elif tool == "Rectangle":
            new_roi = pg.RectROI([cx, cy], [sz, sz], pen='c')
            new_roi.addScaleHandle([0, 0], [1, 1]); new_roi.addScaleHandle([1, 1], [0, 0])
            new_roi.addScaleHandle([0, 1], [1, 0]); new_roi.addScaleHandle([1, 0], [0, 1])
            new_roi.addScaleHandle([0.5, 0], [0.5, 1]); new_roi.addScaleHandle([0.5, 1], [0.5, 0])
            new_roi.addScaleHandle([0, 0.5], [1, 0.5]); new_roi.addScaleHandle([1, 0.5], [0, 0.5])
            make_roi_rotatable_with_ctrl(new_roi)
        elif tool == "Ellipse":
            new_roi = pg.EllipseROI([cx, cy], [sz, sz], pen='c')
            new_roi.addScaleHandle([0, 0], [1, 1]); new_roi.addScaleHandle([1, 1], [0, 0])
            new_roi.addScaleHandle([0, 1], [1, 0]); new_roi.addScaleHandle([1, 0], [0, 1])
            new_roi.addScaleHandle([0.5, 0], [0.5, 1]); new_roi.addScaleHandle([0.5, 1], [0.5, 0])
            new_roi.addScaleHandle([0, 0.5], [1, 0.5]); new_roi.addScaleHandle([1, 0.5], [0, 0.5])
            make_roi_rotatable_with_ctrl(new_roi)

        if new_roi:
            self.view.view_channel.addItem(new_roi)
            self.view.add_spatial_region(new_roi, tool)
            self.view.select_spatial_region(new_roi)

    def add_spatial_region(self, roi, tool):
        e_count = sum(1 for item in self.view.spatial_rois if item.get("tool") == "Ellipse")
        r_count = sum(1 for item in self.view.spatial_rois if item.get("tool") == "Rectangle")
        l_count = sum(1 for item in self.view.spatial_rois if item.get("tool") == "Line")
        p_count = sum(1 for item in self.view.spatial_rois if item.get("tool") == "Point")
        if tool == "Ellipse":
            label = f"E{e_count + 1}"
            name = f"Ellipse {e_count + 1}"
        elif tool == "Rectangle":
            label = f"R{r_count + 1}"
            name = f"Rectangle {r_count + 1}"
        elif tool == "Line":
            label = f"L{l_count + 1}"
            name = f"Line {l_count + 1}"
        elif tool == "Point":
            label = f"P{p_count + 1}"
            name = f"Point {p_count + 1}"
        else:
            label = None
            name = f"{tool} {len(self.view.spatial_rois) + 1}"

        self.view.spatial_rois.append({"name": name, "roi": roi, "tool": tool})

        self.view.combo_spatial_regions.blockSignals(True)
        self.view.combo_spatial_regions.addItem(name)
        self.view.combo_spatial_regions.setCurrentText(name)
        self.view.combo_spatial_regions.blockSignals(False)

        roi.sigRegionChanged.connect(self.view.update_spatial_analysis)

        if label is not None:
            text_item = pg.TextItem(text=label, color=(255, 255, 255, 200), anchor=(0, 1))
            text_item.setZValue(30)
            self.view.plot_channel.addItem(text_item)

            def update_spatial_label(r=roi, t=text_item):
                try:
                    pos = r.pos()
                    size = r.size()
                    max_x = max(pos.x(), pos.x() + size.x())
                    max_y = max(pos.y(), pos.y() + size.y())
                    t.setPos(max_x, max_y)
                except Exception:
                    pass

            roi.sigRegionChanged.connect(update_spatial_label)
            update_spatial_label()
            active_item = self.view.spatial_rois[-1]
            active_item["text_item"] = text_item
            active_item["update_spatial_label"] = update_spatial_label
        
        if tool == "Line":
            direction_item = pg.PlotDataItem(
                [], [], connect='finite',
                pen=pg.mkPen('#f7dc6f', width=3),
            )
            direction_item.setZValue(20)
            self.view.plot_channel.addItem(direction_item)

            def update_spatial_arrow(r=roi, a=direction_item):
                points = self.view.get_line_roi_points(r)
                if points is None:
                    a.setData([], [])
                    return
                p1, p2 = points
                vec = p2 - p1
                length = np.hypot(vec[0], vec[1])
                if length <= 0:
                    a.setData([], [])
                    return
                unit = vec / length
                normal = np.array([-unit[1], unit[0]], dtype=float)
                tip = p1 + 0.62 * vec
                head_len = min(max(6.0 * self.view.pix_scale_arcsec, 0.18 * length), 0.32 * length)
                head_width = 0.75 * head_len
                base_center = tip - unit * head_len
                left = base_center + normal * (0.5 * head_width)
                right = base_center - normal * (0.5 * head_width)
                a.setData(
                    [left[0], tip[0], np.nan, right[0], tip[0]],
                    [left[1], tip[1], np.nan, right[1], tip[1]],
                )

            roi.sigRegionChanged.connect(update_spatial_arrow)
            update_spatial_arrow()
            active_item = self.view.spatial_rois[-1]
            active_item["direction_item"] = direction_item
            active_item["update_spatial_arrow"] = update_spatial_arrow

        for item in self.view.spatial_rois:
            if item["roi"] != roi:
                item["roi"].setPen(pg.mkPen('c', width=2))
        roi.setPen(pg.mkPen('y', width=3))
        self.view.spatial_rois_to_delete = [roi]
        
        self.view.update_spatial_analysis()

    def line_roi_hit_test(self, roi, scene_pos, tolerance=10.0):
        pts = roi.getSceneHandlePositions()
        if len(pts) < 2:
            return False

        p = np.array([scene_pos.x(), scene_pos.y()])
        for (_, p1_scene), (_, p2_scene) in zip(pts[:-1], pts[1:]):
            p1 = np.array([p1_scene.x(), p1_scene.y()])
            p2 = np.array([p2_scene.x(), p2_scene.y()])
            seg_len_sq = np.sum((p2 - p1) ** 2)
            if seg_len_sq == 0:
                proj = p1
            else:
                t = max(0.0, min(1.0, np.dot(p - p1, p2 - p1) / seg_len_sq))
                proj = p1 + t * (p2 - p1)
            if np.linalg.norm(p - proj) < tolerance:
                return True
        return False

    def on_spatial_region_selected(self, name):
        for item in self.view.spatial_rois:
            if item["name"] == name:
                self.view.select_spatial_region(item["roi"])
                break

    def delete_selected_spatial_via_button(self):
        self.view.delete_selected_spatial_regions()

    def select_spatial_region(self, roi):
        self.view.spatial_rois_to_delete = [roi]
        for item in self.view.spatial_rois:
            if item["roi"] == roi:
                item["roi"].setPen(pg.mkPen('y', width=3))
                self.view.combo_spatial_regions.blockSignals(True)
                self.view.combo_spatial_regions.setCurrentText(item["name"])
                self.view.combo_spatial_regions.blockSignals(False)
                
                # Show edit button
                if item["tool"] in ["Ellipse", "Rectangle", "Point", "Line"]:
                    self.view.btn_edit_region.show()
                else:
                    self.view.btn_edit_region.hide()
            else:
                item["roi"].setPen(pg.mkPen('c', width=2))
        self.view.update_spatial_analysis()

    def delete_selected_spatial_regions(self):
        for roi in list(self.view.spatial_rois_to_delete):
            for item in self.view.spatial_rois:
                if item["roi"] == roi:
                    di = item.get("direction_item")
                    if di is not None:
                        try:
                            self.view.plot_channel.removeItem(di)
                        except Exception:
                            pass
                        di.setData([], [])
                    if "update_spatial_arrow" in item and item["roi"] is not None:
                        try:
                            item["roi"].sigRegionChanged.disconnect(item["update_spatial_arrow"])
                        except Exception:
                            pass
                    ti = item.get("text_item")
                    if ti is not None:
                        try:
                            self.view.plot_channel.removeItem(ti)
                        except Exception:
                            pass
                    if "update_spatial_label" in item and item["roi"] is not None:
                        try:
                            item["roi"].sigRegionChanged.disconnect(item["update_spatial_label"])
                        except Exception:
                            pass
                    break

            if roi.scene():
                roi.scene().removeItem(roi)
            else:
                try:
                    self.view.view_channel.getView().removeItem(roi)
                except:
                    pass
            
            self.view.spatial_rois = [item for item in self.view.spatial_rois if item["roi"] != roi]
        self.view.spatial_rois_to_delete.clear()

        e_idx = 0
        r_idx = 0
        l_idx = 0
        p_idx = 0
        for item in self.view.spatial_rois:
            tool = item.get("tool", "")
            if tool == "Ellipse":
                e_idx += 1
                new_label = f"E{e_idx}"
                new_name = f"Ellipse {e_idx}"
            elif tool == "Rectangle":
                r_idx += 1
                new_label = f"R{r_idx}"
                new_name = f"Rectangle {r_idx}"
            elif tool == "Line":
                l_idx += 1
                new_label = f"L{l_idx}"
                new_name = f"Line {l_idx}"
            elif tool == "Point":
                p_idx += 1
                new_label = f"P{p_idx}"
                new_name = f"Point {p_idx}"
            else:
                continue
            item["name"] = new_name
            ti = item.get("text_item")
            if ti is not None:
                ti.setText(new_label)

        self.view.combo_spatial_regions.blockSignals(True)
        self.view.combo_spatial_regions.clear()
        self.view.combo_spatial_regions.addItem("None")
        for item in self.view.spatial_rois:
            self.view.combo_spatial_regions.addItem(item["name"])
        self.view.combo_spatial_regions.blockSignals(False)

        if self.view.spatial_rois:
            self.view.combo_spatial_regions.setCurrentText(self.view.spatial_rois[-1]["name"])
        else:
            self.view.combo_spatial_regions.setCurrentText("None")
            
        self.view.update_spatial_analysis()

    def add_pv_cut(self, roi):
        name = f"Cut {len(self.view.pv_cuts) + 1}"
        cut_info = {"name": name, "roi": roi, "width": 1}
        self.view.pv_cuts.append(cut_info)

        text_item = pg.TextItem(text=name, color=(220, 220, 220, 180), anchor=(0, 1))
        self.view.plot_channel.addItem(text_item)
        direction_item = pg.PlotDataItem(
            [],
            [],
            connect='finite',
            pen=pg.mkPen('#f7dc6f', width=3),
        )
        direction_item.setZValue(20)
        self.view.plot_channel.addItem(direction_item)

        from PyQt5.QtWidgets import QGraphicsPolygonItem
        width_item = QGraphicsPolygonItem()
        width_item.setBrush(pg.mkBrush(255, 255, 255, 40))
        width_item.setPen(pg.mkPen(255, 255, 255, 100, width=1))
        width_item.setZValue(19)
        self.view.plot_channel.addItem(width_item)

        def update_annotations(r=roi, t=text_item, a=direction_item, w_item=width_item, c_info=cut_info):
            points = self.view.get_line_roi_points(r)
            if points is None:
                return
            p1, p2 = points
            vec = p2 - p1
            length = np.hypot(vec[0], vec[1])
            if length <= 0:
                a.setData([], [])
                from PyQt5.QtGui import QPolygonF
                w_item.setPolygon(QPolygonF())
                return

            unit = vec / length
            normal = np.array([-unit[1], unit[0]], dtype=float)
            tip = p1 + 0.62 * vec
            head_len = min(max(6.0 * self.view.pix_scale_arcsec, 0.18 * length), 0.32 * length)
            head_width = 0.75 * head_len
            base_center = tip - unit * head_len
            left = base_center + normal * (0.5 * head_width)
            right = base_center - normal * (0.5 * head_width)

            t.setPos(p2[0], p2[1])
            a.setData(
                [left[0], tip[0], np.nan, right[0], tip[0]],
                [left[1], tip[1], np.nan, right[1], tip[1]],
            )

            # Draw width polygon
            cut_width_pixels = c_info.get("width", 1)
            if cut_width_pixels <= 1:
                w_item.hide()
            else:
                w_item.show()
                width_arcsec = cut_width_pixels * self.view.pix_scale_arcsec
                hw = width_arcsec / 2.0
                p1_left = p1 + normal * hw
                p1_right = p1 - normal * hw
                p2_left = p2 + normal * hw
                p2_right = p2 - normal * hw
                
                from PyQt5.QtGui import QPolygonF
                from PyQt5.QtCore import QPointF
                poly = QPolygonF([
                    QPointF(p1_left[0], p1_left[1]),
                    QPointF(p2_left[0], p2_left[1]),
                    QPointF(p2_right[0], p2_right[1]),
                    QPointF(p1_right[0], p1_right[1])
                ])
                w_item.setPolygon(poly)

        roi.sigRegionChanged.connect(update_annotations)
        roi.sigRegionChanged.connect(self.view.update_moment_maps)
        update_annotations()

        cut_info["text_item"] = text_item
        cut_info["direction_item"] = direction_item
        cut_info["width_item"] = width_item
        cut_info["update_annotations"] = update_annotations

        self.view.refresh_all_pv_cut_combos()
        self.view.set_selected_pv_cut(name)
        for panel in self.view.panels:
            if panel['combo'].currentText() == "PV Diagram" and panel['combo_pv_cut'].currentText() == "None":
                panel['combo_pv_cut'].setCurrentText(name)
        self.view.update_moment_maps()

    def on_pv_cut_selected(self, name):
        self.view.set_selected_pv_cut(name)
        self.view.update_moment_maps()

    def open_edit_pv_cut_dialog(self):
        if not self.view.pv_cuts_to_delete:
            return
        selected_roi = self.view.pv_cuts_to_delete[-1]
        cut_dict = next((item for item in self.view.pv_cuts if item["roi"] == selected_roi), None)
        if cut_dict is None:
            return
        from src.gui.dialogs import RegionPropertiesDialog
        if getattr(self.view, '_pv_edit_dialog', None) and self.view._pv_edit_dialog.isVisible():
            self.view._pv_edit_dialog.raise_()
            self.view._pv_edit_dialog.activateWindow()
            return
        roi_dict = {"name": cut_dict["name"], "roi": cut_dict["roi"], "tool": "PV Cut",
                     "text_item": cut_dict.get("text_item"), "pv_cut_dict": cut_dict}
        dlg = RegionPropertiesDialog(cut_dict["roi"], self.view, parent=self.view.window(), roi_dict=roi_dict)
        self.view._pv_edit_dialog = dlg
        dlg.show()

    def select_pv_cut(self, roi):
        for item in self.view.pv_cuts:
            if item["roi"] == roi:
                self.view.set_selected_pv_cut(item["name"])
                active_panel_id = getattr(self.view, 'last_clicked_panel_id', None)
                if isinstance(active_panel_id, int):
                    panel = self.view.panels[active_panel_id]
                    if panel['combo'].currentText() == "PV Diagram":
                        panel['combo_pv_cut'].setCurrentText(item["name"])
                return

    def delete_selected_pv_via_button(self):
        self.view.delete_selected_pv_cuts()

    def delete_selected_pv_cuts(self):
        for roi in list(self.view.pv_cuts_to_delete):
            if roi.scene():
                roi.scene().removeItem(roi)
            else:
                try:
                    self.view.view_channel.getView().removeItem(roi)
                except Exception:
                    pass

            for i, item in enumerate(self.view.pv_cuts):
                if item["roi"] == roi:
                    text_item = item.get("text_item")
                    if text_item is not None:
                        if text_item.scene():
                            text_item.scene().removeItem(text_item)
                        else:
                            self.view.plot_channel.removeItem(text_item)
                    direction_item = item.get("direction_item")
                    if direction_item is not None:
                        if direction_item.scene():
                            direction_item.scene().removeItem(direction_item)
                        else:
                            self.view.plot_channel.removeItem(direction_item)
                    width_item = item.get("width_item")
                    if width_item is not None:
                        if width_item.scene():
                            width_item.scene().removeItem(width_item)
                        else:
                            self.view.plot_channel.removeItem(width_item)
                    self.view.pv_cuts.pop(i)
                    break

        self.view.pv_cuts_to_delete.clear()
        for idx, item in enumerate(self.view.pv_cuts, start=1):
            item["name"] = f"Cut {idx}"
            if "text_item" in item:
                item["text_item"].setText(item["name"])
        self.view.refresh_all_pv_cut_combos()
        if self.view.pv_cuts:
            self.view.set_selected_pv_cut(self.view.pv_cuts[-1]["name"])
        else:
            if hasattr(self.view, 'combo_pv_cuts'):
                self.view.combo_pv_cuts.blockSignals(True)
                self.view.combo_pv_cuts.setCurrentText("None")
                self.view.combo_pv_cuts.blockSignals(False)
            for panel in self.view.panels:
                panel['combo_pv_cut'].blockSignals(True)
                panel['combo_pv_cut'].setCurrentText("None")
                panel['combo_pv_cut'].blockSignals(False)
                self.view.clear_panel_pv_diagram(panel)
            self.view.pv_data = None
            self.view.pv_offset_axis = None
            self.view.pv_velocity_axis = None
            self.view.pv_view.clear()
            self.view.lbl_hover_pv.setText("")
        self.view.update_moment_maps()

    def clear_pv_cuts(self):
        self.view.pv_cuts_to_delete = [item["roi"] for item in self.view.pv_cuts]
        self.view.delete_selected_pv_cuts()

    def get_line_roi_points(self, roi):
        pts = roi.getSceneHandlePositions()
        if len(pts) < 2:
            return None
        p1 = self.view.plot_channel.vb.mapSceneToView(pts[0][1])
        p2 = self.view.plot_channel.vb.mapSceneToView(pts[1][1])
        return np.array([p1.x(), p1.y()], dtype=float), np.array([p2.x(), p2.y()], dtype=float)

    def change_roi(self, roi_type, cx=None, cy=None):
        if self.view.cube_clean is None: return
        
        if roi_type == "Whole Map":
            for r_dict in getattr(self.view, 'spectrum_spatial_rois', []):
                r_dict["checkbox"].blockSignals(True)
                r_dict["checkbox"].setChecked(False)
                r_dict["checkbox"].blockSignals(False)
            self.view.update_spectrum()
            return
            
        num_rois = len(getattr(self.view, 'spectrum_spatial_rois', []))
        sz = self.view.nx * self.view.pix_scale_arcsec * 0.2
        offset = num_rois * sz * 0.15
        
        if cx is None:
            cx = offset
        if cy is None:
            cy = offset
        
        if hasattr(self.view, 'btn_edit_region'):
            if roi_type in ["Ellipse", "Rectangle", "Point (Beam)", "Custom Polygon"]:
                self.view.btn_edit_region.show()
            else:
                self.view.btn_edit_region.hide()
                
        new_roi = None
        if roi_type == "Point (Beam)": 
            has_beam = False
            bmaj_deg = 0.0
            bmin_deg = 0.0
            bpa_deg = 0.0
            
            if getattr(self.view, 'bmaj_array', None) is not None and getattr(self.view, 'bmin_array', None) is not None:
                has_beam = True
                bmaj_deg = np.median(self.view.bmaj_array) if isinstance(self.view.bmaj_array, (list, np.ndarray)) else self.view.bmaj_array
                bmin_deg = np.median(self.view.bmin_array) if isinstance(self.view.bmin_array, (list, np.ndarray)) else self.view.bmin_array
                
                bpa_array = getattr(self.view, 'bpa_array', None)
                if bpa_array is not None:
                    bpa_deg = np.median(bpa_array) if isinstance(bpa_array, (list, np.ndarray)) else float(bpa_array)
                elif self.view.raw_header is not None:
                    bpa_val = self.view.raw_header.get('BPA', 0.0)
                    bpa_deg = float(bpa_val)

            if has_beam:
                bmaj_arcsec = bmaj_deg * 3600.0
                bmin_arcsec = bmin_deg * 3600.0
                
                from src.gui.components.custom_widgets import get_pyqt_angle
                new_roi = pg.EllipseROI([cx, cy], [bmaj_arcsec, bmin_arcsec], pen='#f1c40f')
                new_roi.setAngle(get_pyqt_angle(bpa_deg), center=[0.5, 0.5])
                for handle in list(new_roi.getHandles()):
                    handle.hide()
                    handle.setParentItem(None)
                    new_roi.removeHandle(handle)
            else:
                print("WARNING: No beam information found in FITS header. Falling back to single-pixel point extraction. Integrated Flux Density (Jy) calculations are invalid.")
                try:
                    new_roi = pg.PointROI([cx, cy], pen='#f1c40f')
                except AttributeError:
                    new_roi = pg.RectROI([cx, cy], [self.view.pix_scale_arcsec, self.view.pix_scale_arcsec], pen='#f1c40f')
                    for handle in list(new_roi.getHandles()):
                        handle.hide()
                        handle.setParentItem(None)
                        new_roi.removeHandle(handle)
        elif roi_type == "Ellipse": 
            new_roi = pg.EllipseROI([cx, cy], [sz, sz], pen='#f1c40f')
            new_roi.addScaleHandle([0, 0], [1, 1])
            new_roi.addScaleHandle([1, 1], [0, 0])
            new_roi.addScaleHandle([0, 1], [1, 0])
            new_roi.addScaleHandle([1, 0], [0, 1])
            new_roi.addScaleHandle([0.5, 0], [0.5, 1])
            new_roi.addScaleHandle([0.5, 1], [0.5, 0])
            new_roi.addScaleHandle([0, 0.5], [1, 0.5])
            new_roi.addScaleHandle([1, 0.5], [0, 0.5])
            make_roi_rotatable_with_ctrl(new_roi)
        elif roi_type == "Rectangle": 
            new_roi = pg.RectROI([cx, cy], [sz, sz], pen='#f1c40f')
            new_roi.addScaleHandle([0, 0], [1, 1])
            new_roi.addScaleHandle([1, 1], [0, 0])
            new_roi.addScaleHandle([0, 1], [1, 0])
            new_roi.addScaleHandle([1, 0], [0, 1])
            new_roi.addScaleHandle([0.5, 0], [0.5, 1])
            new_roi.addScaleHandle([0.5, 1], [0.5, 0])
            new_roi.addScaleHandle([0, 0.5], [1, 0.5])
            new_roi.addScaleHandle([1, 0.5], [0, 0.5])
            make_roi_rotatable_with_ctrl(new_roi)
        elif roi_type == "Custom Polygon": 
            # Interactive drawing mode
            self.view.is_drawing_polygon = True
            self.view.polygon_points = []
            if self.view.polygon_preview_line is not None:
                self.view.plot_channel.vb.removeItem(self.view.polygon_preview_line)
            self.view.polygon_preview_line = pg.PlotDataItem([], [], pen=pg.mkPen('y', width=2, style=Qt.DashLine))
            self.view.plot_channel.vb.addItem(self.view.polygon_preview_line)
            # Notify the user via status bar if available, or just start
            return # Don't add ROI yet
        
        if new_roi is not None:
            self.view._finish_roi_addition(new_roi, roi_type)

    def _finish_roi_addition(self, new_roi, roi_type):
        col = self.view.region_colors[len(self.view.spectrum_spatial_rois) % len(self.view.region_colors)]
        new_roi.setPen(pg.mkPen(col, width=3))
        self.view.view_channel.addItem(new_roi)
        new_roi.sigRegionChanged.connect(self.view.update_spectrum)
        
        # Uncheck existing
        for r_dict in self.view.spectrum_spatial_rois:
            r_dict["checkbox"].blockSignals(True)
            r_dict["checkbox"].setChecked(False)
            r_dict["checkbox"].blockSignals(False)
            r_dict["roi"].setPen(pg.mkPen(r_dict["color"], width=2))
            
        name = f"SR{len(self.view.spectrum_spatial_rois) + 1}"
        cb = QCheckBox(name)
        cb.setChecked(True)
        cb.setStyleSheet(f"color: {col}; font-weight: bold;")
        cb.toggled.connect(self.view.update_spectrum)
        self.view.box_regions_layout.addWidget(cb)
        
        text_item = pg.TextItem(text=name, color=col, anchor=(0, 1))
        text_item.setZValue(30)
        self.view.view_channel.addItem(text_item)
        
        def update_spectrum_region_label(r=new_roi, t=text_item):
            try:
                br = r.boundingRect()
                pos = r.pos()
                t.setPos(pos.x() + br.right(), pos.y() + br.bottom())
            except Exception:
                pass

        new_roi.sigRegionChanged.connect(update_spectrum_region_label)
        update_spectrum_region_label()
        
        self.view.spectrum_spatial_rois.append({
            "name": name,
            "roi": new_roi,
            "checkbox": cb,
            "color": col,
            "type": roi_type,
            "text_item": text_item,
            "update_label": update_spectrum_region_label
        })
        self.view.roi_selected = True
        self.view.active_spatial_spectrum_roi = new_roi
        
        if len(self.view.spectrum_spatial_rois) > 1:
            self.view.box_regions.show()
        self.view.refresh_spectral_stats_apertures()
        self.view.update_spectrum()
        
    def add_spectrum_region(self, roi):
        active_rois = self.view.get_active_spectrum_rois()
        region_name = f"Box {len(active_rois) + 1}"
        roi_info = {"name": region_name, "roi": roi}
        active_rois.append(roi_info)
        
        self.view.update_region_ui_visibility()
        
        text_item = pg.TextItem(text=region_name, color=(200, 200, 200, 150), anchor=(1, 1))
        self.view.get_active_spectrum_plot().addItem(text_item)
        
        def update_text_pos(r=roi, t=text_item):
            try:
                if hasattr(r, 'getData'):
                    x_data, _ = r.getData()
                    if x_data is not None and len(x_data) > 0:
                        t.setPos(max(x_data), 0)
                else:
                    pos = r.pos()
                    size = r.size()
                    max_x = max(pos.x(), pos.x() + size.x())
                    max_y = max(pos.y(), pos.y() + size.y())
                    t.setPos(max_x, max_y)
            except Exception:
                pass
            
        if hasattr(roi, 'sigRegionChanged'):
            roi.sigRegionChanged.connect(update_text_pos)
            roi.sigRegionChanged.connect(self.view.update_spectrum_region_calc)
        update_text_pos()
        
        roi_info["text_item"] = text_item
        roi_info["update_text_pos"] = update_text_pos
        
        self.view.rename_regions()
        self.view.combo_regions.blockSignals(True)
        self.view.combo_regions.setCurrentText(region_name)
        self.view.combo_regions.blockSignals(False)
        self.view.on_region_selected()

    def remove_spatial_spectrum_roi(self, roi):
        for i, r_dict in enumerate(self.view.spectrum_spatial_rois):
            if r_dict["roi"] == roi:
                # Remove from view
                if roi.scene():
                    roi.scene().removeItem(roi)
                else:
                    self.view.view_channel.getView().removeItem(roi)
                # Remove checkbox
                cb = r_dict["checkbox"]
                self.view.box_regions_layout.removeWidget(cb)
                cb.deleteLater()
                
                if "text_item" in r_dict:
                    text_item = r_dict["text_item"]
                    if text_item.scene():
                        text_item.scene().removeItem(text_item)
                    else:
                        self.view.view_channel.getView().removeItem(text_item)
                        
                # Remove from dicts and plot items
                if r_dict["name"] in self.view.spectrum_curves:
                    c = self.view.spectrum_curves.pop(r_dict["name"])
                    if c.scene(): c.scene().removeItem(c)
                    else: self.view.plot_widget.removeItem(c)
                
                if hasattr(self.view, 'spectrum_curves_smooth') and r_dict["name"] in self.view.spectrum_curves_smooth:
                    c = self.view.spectrum_curves_smooth.pop(r_dict["name"])
                    if c.scene(): c.scene().removeItem(c)
                    else: getattr(self.view, 'plot_widget_smooth', self.view.plot_widget).removeItem(c)
                
                self.view.spectrum_spatial_rois.pop(i)
                break
                
        # Rename remaining to be contiguous
        for i, r_dict in enumerate(self.view.spectrum_spatial_rois):
            new_name = f"SR{i + 1}"
            old_name = r_dict["name"]
            r_dict["name"] = new_name
            r_dict["checkbox"].setText(new_name)
            
            if "text_item" in r_dict:
                r_dict["text_item"].setText(new_name)
            
            if old_name in self.view.spectrum_curves:
                self.view.spectrum_curves[new_name] = self.view.spectrum_curves.pop(old_name)
            if hasattr(self.view, 'spectrum_curves_smooth') and old_name in self.view.spectrum_curves_smooth:
                self.view.spectrum_curves_smooth[new_name] = self.view.spectrum_curves_smooth.pop(old_name)
                
        if len(self.view.spectrum_spatial_rois) <= 1:
            self.view.box_regions.hide()

        self.view.refresh_spectral_stats_apertures()
        self.view.update_spectrum()

    def clear_roi(self):
        self.view.delete_nr_roi()
        if hasattr(self.view, 'combo_roi'):
            self.view.combo_roi.blockSignals(True)
            self.view.combo_roi.setCurrentText("Whole Map")
            self.view.combo_roi.blockSignals(False)
            
        for r_dict in list(getattr(self.view, 'spectrum_spatial_rois', [])):
            self.remove_spatial_spectrum_roi(r_dict["roi"])
            
        if hasattr(self.view, 'spatial_rois_to_delete'):
            self.view.spatial_rois_to_delete = [item["roi"] for item in getattr(self.view, 'spatial_rois', [])]
            self.delete_selected_spatial_regions()

    # --- DYNAMIC LINE CATALOG ENGINE ---

    def update_nr_rms(self):
        if getattr(self.view, 'nr_roi', None) is None:
            return
        ch_idx = self.view.slider_channel.value()
        if self.view.cube_clean is None or ch_idx < 0 or ch_idx >= self.view.cube_clean.shape[0]:
            return
            
        current_slice = self.view.cube_clean[ch_idx, :, :]
        r_pos = self.view.nr_roi.pos()
        r_size = self.view.nr_roi.size()
        
        start_x = (self.view.nx / 2) * self.view.pix_scale_arcsec
        start_y = -(self.view.ny / 2) * self.view.pix_scale_arcsec
        
        min_x_scene = min(r_pos.x(), r_pos.x() + r_size.x())
        max_x_scene = max(r_pos.x(), r_pos.x() + r_size.x())
        min_y_scene = min(r_pos.y(), r_pos.y() + r_size.y())
        max_y_scene = max(r_pos.y(), r_pos.y() + r_size.y())
        
        x1_idx = int((min_x_scene - start_x) / (-self.view.pix_scale_arcsec))
        x2_idx = int((max_x_scene - start_x) / (-self.view.pix_scale_arcsec))
        y1_idx = int((min_y_scene - start_y) / self.view.pix_scale_arcsec)
        y2_idx = int((max_y_scene - start_y) / self.view.pix_scale_arcsec)
        
        x_min = max(0, min(x1_idx, x2_idx))
        x_max = min(self.view.nx, max(x1_idx, x2_idx) + 1)
        y_min = max(0, min(y1_idx, y2_idx))
        y_max = min(self.view.ny, max(y1_idx, y2_idx) + 1)
        
        if x_min >= x_max or y_min >= y_max:
            return
            
        extracted_data = current_slice[x_min:x_max, y_min:y_max]
        rms_val = 3.0 * float(np.nanstd(extracted_data))
        
        if np.isnan(rms_val):
            val_str = "NaN"
        else:
            val_str = f"{rms_val:.4e}" if rms_val < 1e-3 else f"{rms_val:.4f}"
            
        if getattr(self.view, 'nr_label', None) is not None:
            self.view.nr_label.setText(f"NR (3σ = {val_str})")
            
        pid = getattr(self.view.nr_roi, 'target_panel_id', None)
        if pid is not None and 0 <= pid < len(self.view.panels):
            target_panel = self.view.panels[pid]
            if not np.isnan(rms_val):
                target_panel['input_thresh'].setText(val_str)
                self.view.update_moment_maps()

    def update_pv_diagram(self, _=None):
        self.view.update_moment_maps()

    def update_wcs_mode(self, is_absolute):
        self.view.is_absolute_wcs = is_absolute
        x_label = 'Right Ascension (J2000)' if is_absolute else 'RA offset (arcsec)'
        y_label = 'Declination (J2000)' if is_absolute else 'Dec offset (arcsec)'

        self.view.plot_channel.setLabel('bottom', x_label)
        self.view.plot_channel.setLabel('left', y_label)
        self.view.plot_channel.getAxis('bottom').update_wcs(self.view.wcs_2d, self.view.nx, self.view.ny, self.view.pix_scale_arcsec, is_absolute)
        self.view.plot_channel.getAxis('left').update_wcs(self.view.wcs_2d, self.view.nx, self.view.ny, self.view.pix_scale_arcsec, is_absolute)
        for panel in self.view.panels:
            self.view.configure_bottom_panel_axes(panel, panel['combo'].currentText() == "PV Diagram")

    def draw_selected_lines(self, selected_rows, v_sys):
        c_kms = const.c.to(u.km / u.s).value
        ref_freq_ghz = self.view.rest_freq_hz / 1e9

        drawn_count = 0
        for row in selected_rows:
            f_cat_ghz = row.get('restfreq', 0.0)
            if np.isnan(f_cat_ghz) or f_cat_ghz == 0.0: continue
            
            label_text = str(row.get('formula', row.get('molecule_name', 'Unknown')))
            
            v_offset = c_kms * (ref_freq_ghz - f_cat_ghz) / ref_freq_ghz
            v_plot = v_offset + v_sys
            
            line = pg.InfiniteLine(angle=90, movable=False, pos=v_plot, pen=pg.mkPen('#e74c3c', width=1.5, style=Qt.DashLine))
            label = pg.TextItem(text=label_text, color='#e74c3c', anchor=(0, 1), angle=-90)
            label.setPos(v_plot, np.nanmax(self.view.spectrum_curve.yData) if self.view.spectrum_curve.yData is not None else 1.0)
            
            self.view.plot_widget.addItem(line)
            self.view.plot_widget.addItem(label)
            self.view.catalog_overlay_items.extend([line, label])
            drawn_count += 1

        self.view.parent_window.statusBar().showMessage(f"Overlaid {drawn_count} selected molecular lines.")

    def draw_overlay_contours(self):
        self.view._clear_overlay_contours()
        if not self.view.contour_overlays or self.view.cube_clean is None:
            return

        for overlay_dict in self.view.contour_overlays:
            overlay_slice = self.view._get_overlay_slice_for_channel(overlay_dict)
            if overlay_slice is None:
                continue

            reprojected = self.view._reproject_overlay_slice(overlay_dict, overlay_slice)
            if reprojected is None:
                continue

            overlay_dict['_reproj_raw'] = reprojected
            overlay_dict['_reproj_channel'] = self.view.slider_channel.value()

            opts = overlay_dict['options']
            if opts.get('smooth', False):
                k = int(opts.get('smooth_kernel', 3))
                if k % 2 == 0:
                    k += 1
                if k >= 3:
                    from scipy.ndimage import gaussian_filter
                    mask = np.isfinite(reprojected)
                    smooth_data = np.where(mask, reprojected, 0.0)
                    smooth_data = gaussian_filter(smooth_data, sigma=k / 3.0)
                    smooth_data[~mask] = np.nan
                    reprojected = smooth_data

            levels = self.view._compute_contour_levels(reprojected, opts)
            if not levels:
                continue

            min_val = float(np.nanmin(reprojected))
            max_val = float(np.nanmax(reprojected))
            if not (np.isfinite(min_val) and np.isfinite(max_val)):
                continue
            data_range = max(abs(max_val - min_val), 1e-12)
            nan_fill = min_val - 10.0 * data_range
            reprojected = np.where(np.isfinite(reprojected), reprojected, nan_fill)

            color = opts.get('color', 'white')
            lw = opts.get('line_width', 1.5)
            style_str = opts.get('line_style', 'solid')

            style_map = {'solid': Qt.SolidLine, 'dashed': Qt.DashLine, 'dotted': Qt.DotLine}
            pen_style = style_map.get(style_str, Qt.SolidLine)

            for lvl in levels:
                iso = pg.IsocurveItem(data=reprojected, level=lvl, pen=pg.mkPen(color, width=lw, style=pen_style))
                iso.setParentItem(self.view.view_channel.getImageItem())
                iso.setZValue(10)
                overlay_dict['iso_items'].append(iso)

    def draw_contours(self, target_id, view, data):
        for iso in self.view.active_contours.get(target_id, []):
            iso.setParentItem(None)
            if iso.scene() is not None:
                view.getView().removeItem(iso)
        self.view.active_contours[target_id] = []

        params = self.view.contour_params.get(target_id)
        if not params or data is None or np.isnan(data).all(): return

        smoothed = data
        if params.get('smooth', False):
            k = int(params.get('smooth_kernel', 3))
            if k % 2 == 0: k += 1
            from scipy.ndimage import gaussian_filter
            smoothed = gaussian_filter(np.where(np.isfinite(data), data, 0.0).astype(np.float64), sigma=k / 2.355)
            smooth_mask = gaussian_filter(np.isfinite(data).astype(np.float64), sigma=k / 2.355)
            mask = smooth_mask > 0.01
            smoothed = np.where(mask, smoothed / np.where(mask, smooth_mask, 1.0), np.nan)

        levels = self.view._compute_contour_levels(smoothed, params, target_id=target_id)
        if not levels:
            return

        min_level = min(levels)
        fill_value = min_level - 999.0
        clean_data = np.copy(smoothed)
        clean_data[~np.isfinite(clean_data)] = fill_value

        color_name = params.get('color', 'cyan').lower()
        qcolor = pg.mkColor(color_name)
        if not qcolor.isValid():
            qcolor = pg.mkColor('c')
        lw = float(params.get('line_width', 1.5))
        style_name = params.get('line_style', 'solid')
        style = ContourDialog._LINE_STYLES.get(style_name.capitalize(), Qt.SolidLine)
        pen = pg.mkPen(qcolor, width=lw, style=style)

        for lvl in levels:
            iso = pg.IsocurveItem(data=clean_data, level=lvl, pen=pen)
            iso.setParentItem(view.getImageItem())
            iso.setZValue(10)
            self.view.active_contours[target_id].append(iso)

    def update_channel_map(self):
        if self.view.cube_clean is None: return
        idx = self.view.slider_channel.value()
        self.view.input_channel_vel.setText(f"{self.view.v_axis[idx]:.2f}")
        self.view.v_line.setPos(self.view.v_axis[idx])
        
        pos_tup = ((self.view.nx / 2) * self.view.pix_scale_arcsec, -(self.view.ny / 2) * self.view.pix_scale_arcsec)
        scale_tup = (-self.view.pix_scale_arcsec, self.view.pix_scale_arcsec)
        
        slice_data = self.view.cube_clean[idx]
        self.view.view_channel.setImage(slice_data, autoLevels=False, levels=getattr(self.view, 'ch_levels', (0, 1)), autoHistogramRange=True, scale=scale_tup, pos=pos_tup)
        self.view.view_channel.ui.histogram.gradient.loadPreset(self.view.parent_window.current_cmap)
        self.view.draw_contours('channel', self.view.view_channel, slice_data)
        self.view.draw_overlay_contours()
        self.view.update_spatial_analysis()
        self.view.update_beam_visualizers('channel')
        if hasattr(self.view, 'update_nr_rms'):
            self.view.update_nr_rms()
        
        grad = self.view.view_channel.ui.histogram.gradient
        ticks = list(grad.ticks.keys())
        if len(ticks) > 3:
            sorted_ticks = sorted(ticks, key=lambda t: grad.ticks[t])
            keep = [sorted_ticks[0], sorted_ticks[len(sorted_ticks)//2], sorted_ticks[-1]]
            for t in sorted_ticks:
                if t not in keep:
                    t.hide()

    def update_spectral_axis(self):
        """Dynamically recalculates the spectrum X-axis based on dropdown selections and custom rest frequencies."""
        if getattr(self.view, 'freq_array', None) is None:
            return
            
        # 1. Parse custom rest frequency if provided
        ref_freq_str = self.view.input_ref_freq.text().strip()
        if ref_freq_str:
            try:
                self.view.rest_freq_hz = float(ref_freq_str) * 1e9
            except ValueError:
                pass
                
        if getattr(self.view, 'rest_freq_hz', None) is None:
            return
            
        axis_type = self.view.combo_axis_type.currentText()
        
        # 2. Capture old state for synchronization
        old_v_axis = getattr(self.view, 'v_axis', None)
        old_view_range = None
        old_vline_pos = None
        old_region_bounds = None
        old_stats_roi_bounds = {}
        
        if old_v_axis is not None:
            try:
                old_view_range = self.view.plot_widget.viewRange()[0]
                if hasattr(self.view, 'v_line'): old_vline_pos = self.view.v_line.value()
                if hasattr(self.view, 'region'): old_region_bounds = self.view.region.getRegion()
                for r in self.view.get_active_spectrum_rois():
                    roi = r["roi"]
                    if hasattr(roi, 'getRegion'):
                        old_stats_roi_bounds[r["name"]] = roi.getRegion()
                    elif hasattr(roi, 'getData'):
                        x_data, _ = roi.getData()
                        if x_data is not None and len(x_data) >= 2:
                            old_stats_roi_bounds[r["name"]] = (min(x_data), max(x_data))
                    else:
                        pos = roi.pos()
                        size = roi.size()
                        old_stats_roi_bounds[r["name"]] = (pos.x(), pos.x() + size.x(), pos.y(), size.y())
            except Exception:
                pass

        # 3. Use astropy units for robust conversion
        freq_q = self.view.freq_array * u.Hz
        rest_q = self.view.rest_freq_hz * u.Hz
        
        try:
            # 4. Generate New X-Axis based on type
            if axis_type == "Radio Velocity":
                vel_q = freq_q.to(u.km / u.s, equivalencies=u.doppler_radio(rest_q))
                self.view.v_axis = vel_q.value
                self.view.plot_widget.setLabel('bottom', 'Radio Velocity (km/s)')
                
            elif axis_type == "Optical Velocity":
                vel_q = freq_q.to(u.km / u.s, equivalencies=u.doppler_optical(rest_q))
                self.view.v_axis = vel_q.value
                self.view.plot_widget.setLabel('bottom', 'Optical Velocity (km/s)')
                
            elif axis_type == "Frequency":
                if np.nanmax(freq_q.value) > 1e9:
                    self.view.v_axis = freq_q.to(u.GHz).value
                    self.view.plot_widget.setLabel('bottom', 'Frequency (GHz)')
                elif np.nanmax(freq_q.value) > 1e6:
                    self.view.v_axis = freq_q.to(u.MHz).value
                    self.view.plot_widget.setLabel('bottom', 'Frequency (MHz)')
                else:
                    self.view.v_axis = freq_q.value
                    self.view.plot_widget.setLabel('bottom', 'Frequency (Hz)')
                    
            elif axis_type == "Wavelength":
                wave_q = freq_q.to(u.m, equivalencies=u.spectral())
                if np.nanmean(wave_q.value) < 1e-3:
                    self.view.v_axis = wave_q.to(u.um).value
                    self.view.plot_widget.setLabel('bottom', 'Wavelength (μm)')
                else:
                    self.view.v_axis = wave_q.to(u.mm).value
                    self.view.plot_widget.setLabel('bottom', 'Wavelength (mm)')
                    
            elif axis_type == "Channel":
                self.view.v_axis = np.arange(len(self.view.freq_array))
                self.view.plot_widget.setLabel('bottom', 'Channel')
                
            # Sync the smoothing tab's label if it exists
            if hasattr(self.view, 'plot_widget_smooth'):
                self.view.plot_widget_smooth.setLabel('bottom', self.view.plot_widget.getAxis('bottom').labelText)
                
            # 5. Synchronize UI markers and view limits to the new axis
            if old_v_axis is not None and len(old_v_axis) > 1 and len(old_v_axis) == len(self.view.v_axis):
                sort_idx = np.argsort(old_v_axis)
                old_v_sorted = old_v_axis[sort_idx]
                new_v_sorted = self.view.v_axis[sort_idx]
                
                def map_val(val):
                    channel_interp = np.interp(val, old_v_sorted, np.arange(len(old_v_sorted)))
                    return float(np.interp(channel_interp, np.arange(len(new_v_sorted)), new_v_sorted))
                    
                if old_view_range is not None:
                    nv_min, nv_max = map_val(old_view_range[0]), map_val(old_view_range[1])
                    self.view.plot_widget.setXRange(min(nv_min, nv_max), max(nv_min, nv_max), padding=0)
                    if hasattr(self.view, 'plot_widget_smooth'):
                        self.view.plot_widget_smooth.setXRange(min(nv_min, nv_max), max(nv_min, nv_max), padding=0)
                    
                if hasattr(self.view, 'v_line') and old_vline_pos is not None:
                    new_pos = map_val(old_vline_pos)
                    self.view.v_line.setValue(new_pos)
                    if hasattr(self.view, 'smooth_active_line'):
                        self.view.smooth_active_line.setValue(new_pos)
                    
                if hasattr(self.view, 'region') and old_region_bounds is not None:
                    nr_min, nr_max = map_val(old_region_bounds[0]), map_val(old_region_bounds[1])
                    new_region = [min(nr_min, nr_max), max(nr_min, nr_max)]
                    self.view.region.setRegion(new_region)
                    if hasattr(self.view, 'smooth_velocity_region'):
                        self.view.smooth_velocity_region.setRegion(new_region)
                    
                for r in self.view.get_active_spectrum_rois():
                    if r["name"] in old_stats_roi_bounds:
                        b = old_stats_roi_bounds[r["name"]]
                        nr_min, nr_max = map_val(b[0]), map_val(b[1])
                        roi = r["roi"]
                        if hasattr(roi, 'setRegion'):
                            roi.setRegion([min(nr_min, nr_max), max(nr_min, nr_max)])
                        elif hasattr(roi, 'setData'):
                            roi.setData([min(nr_min, nr_max), max(nr_min, nr_max)], [0, 0])
                            if "update_text_pos" in r:
                                r["update_text_pos"]()
                        else:
                            roi.blockSignals(True)
                            new_x = min(nr_min, nr_max)
                            new_w = abs(nr_max - nr_min)
                            roi.setPos([new_x, b[2]])
                            roi.setSize([new_w, b[3]])
                            roi.blockSignals(False)
                            if "update_text_pos" in r:
                                r["update_text_pos"]()
                    
            # 6. Redraw spectrum with the new axis array
            self.view.update_spectrum()
            
        except Exception as e:
            print(f"Error updating spectral axis: {e}")

    def update_spectrum(self):
        if self.view.cube_clean is None or getattr(self.view, 'is_2d_image', False): return
        stat = self.view.combo_spec_stat.currentText()
        
        active_rois = [r_dict for r_dict in self.view.spectrum_spatial_rois if r_dict["checkbox"].isChecked()]
        if not active_rois:
            rois_to_plot = [{"name": "Whole Map", "roi": None, "color": "w"}]
        else:
            rois_to_plot = active_rois
            
        active_names = [r["name"] for r in rois_to_plot]
        for name in list(self.view.spectrum_curves.keys()):
            if name not in active_names:
                c = self.view.spectrum_curves.pop(name)
                if c.scene(): c.scene().removeItem(c)
                else: self.view.plot_widget.removeItem(c)
                
                if hasattr(self.view, 'spectrum_curves_smooth') and name in self.view.spectrum_curves_smooth:
                    c_s = self.view.spectrum_curves_smooth.pop(name)
                    if c_s.scene(): c_s.scene().removeItem(c_s)
                    else: getattr(self.view, 'plot_widget_smooth', self.view.plot_widget).removeItem(c_s)
                    
        if "Whole Map" not in active_names:
            self.view.spectrum_curve.setData([], [])
            if hasattr(self.view, 'spectrum_curve_smooth'):
                self.view.spectrum_curve_smooth.setData([], [])
                
        ymax_global = -np.inf
        
        stat = self.view.combo_spec_stat.currentText()
        unit_sel = self.view.combo_spec_unit.currentText()
        
        is_rj_active = False
        with np.errstate(invalid='ignore', divide='ignore'):
            for r_dict in rois_to_plot:
                roi = r_dict["roi"]
                name = r_dict["name"]
                color = r_dict["color"]
                
                if roi is None:
                    sub_data = self.view.cube_clean
                else:
                    sub_data = roi.getArrayRegion(self.view.cube_clean, self.view.view_channel.getImageItem(), axes=(1, 2))
                    
                    # Create precise boolean mask using PyQtGraph rasterization
                    dummy_ones = np.ones((self.view.nx, self.view.ny))
                    roi_mask = roi.getArrayRegion(dummy_ones, self.view.view_channel.getImageItem(), axes=(0, 1))
                    
                    # Safely set background bounding-box pixels to np.nan
                    sub_data[:, roi_mask == 0] = np.nan
                    
                # Print valid pixels count for diagnostic purposes
                # if len(sub_data) > 0:
                #     n_valid_pixels = np.count_nonzero(~np.isnan(sub_data[0]))
                #     print(f"[{stat}] Valid pixels in mask for region '{name}': {n_valid_pixels}")
                    
                # Phase 2: Spatial Collapse
                if "Max" in stat:
                    raw_array = np.nanmax(sub_data, axis=(1, 2))
                elif "Sum" in stat or "Flux Density" in stat:
                    raw_array = np.nansum(sub_data, axis=(1, 2))
                elif "Median" in stat:
                    raw_array = np.nanmedian(sub_data, axis=(1, 2))
                else:
                    raw_array = np.nanmean(sub_data, axis=(1, 2))
                
                # Phase 3: The 4 Conversion Paths
                unit_lower = self.view.display_unit.replace(" ", "").lower()
                
                if "Native" in unit_sel or not self.view.can_convert_units:
                    # Path 4: Native matches Target
                    final_array = raw_array
                    y_label = f"{stat} ({self.view.display_unit})"
                    self.view.spec_unit = self.view.display_unit
                elif unit_sel == "Jy":
                    if "jy" in unit_lower:
                        # Path 1: Native Jy/beam, Target Jy (Statistic: Sum)
                        if "pixel" in unit_lower or "pix" in unit_lower:
                            flux_array_jy = raw_array
                        else:
                            flux_array_jy = raw_array / self.view.n_beam_array.value
                        final_array = flux_array_jy
                    elif "k" == unit_lower or "kelvin" in unit_lower:
                        # Path 3: Native K, Target Jy (Statistic: Sum)
                        is_rj_active = True
                        freq_hz = self.view.freq_array
                        jy_sr_per_kelvin = (1 * u.K).to(u.Jy / u.sr, equivalencies=u.brightness_temperature(freq_hz * u.Hz))
                        flux_array_jy = raw_array * jy_sr_per_kelvin.value * self.view.omega_pix_sr.value
                        final_array = flux_array_jy
                    else:
                        final_array = raw_array
                    y_label = f"{stat} (Jy)"
                    self.view.spec_unit = "Jy"
                elif unit_sel == "K":
                    if "k" == unit_lower or "kelvin" in unit_lower:
                        # Path 4: Native K, Target K
                        final_array = raw_array
                    elif "jy" in unit_lower:
                        # Path 2: Native Jy/beam, Target K (Statistic: Mean, Median, Max)
                        is_rj_active = True
                        omega = self.view.omega_pix_sr if ("pixel" in unit_lower or "pix" in unit_lower) else self.view.omega_beam_sr
                        surface_brightness = (raw_array * u.Jy) / omega
                        freq_hz = self.view.freq_array
                        tb_array_k = surface_brightness.to(u.K, equivalencies=u.brightness_temperature(freq_hz * u.Hz))
                        final_array = tb_array_k.value
                    else:
                        final_array = raw_array
                    y_label = f"{stat} (K)"
                    self.view.spec_unit = "K"
                elif unit_sel == "Jy/beam":
                    if "jy" in unit_lower and ("pixel" not in unit_lower and "pix" not in unit_lower):
                        # Path 4: Native Jy/beam, Target Jy/beam
                        final_array = raw_array
                    elif "k" == unit_lower or "kelvin" in unit_lower:
                        # Path 5: Native K, Target Jy/beam
                        is_rj_active = True
                        freq_hz = self.view.freq_array
                        jy_sr_per_kelvin = (1 * u.K).to(u.Jy / u.sr, equivalencies=u.brightness_temperature(freq_hz * u.Hz))
                        surface_brightness_jy_sr = raw_array * jy_sr_per_kelvin.value
                        final_array = surface_brightness_jy_sr * self.view.omega_beam_sr.value
                    else:
                        final_array = raw_array
                    y_label = f"{stat} (Jy/beam)"
                    self.view.spec_unit = "Jy/beam"
                    
                spec = final_array
                    
                self.view.plot_widget.setLabel('left', y_label)
                if hasattr(self.view, 'plot_widget_smooth'):
                    self.view.plot_widget_smooth.setLabel('left', y_label)
                
                sort_idx = np.argsort(self.view.v_axis)
                vs, ss = self.view.v_axis[sort_idx], spec[sort_idx]
                ve = np.zeros(len(vs) + 1)
                dv = np.diff(vs)
                if len(dv) > 0:
                    ve[:-1] = vs - np.append(dv, dv[-1])/2
                    ve[-1] = vs[-1] + dv[-1]/2
                else: ve = np.array([vs[0]-1, vs[0]+1])
                
                if ss is not None and len(ss) > 0:
                    ymax_global = max(ymax_global, np.nanmax(ss))
                
                if name == "Whole Map":
                    self.view.spectrum_curve.setData(x=ve, y=ss)
                else:
                    if name not in self.view.spectrum_curves:
                        c = pg.PlotDataItem([], [], stepMode="center", pen=pg.mkPen(color, width=2), name=name)
                        self.view.spectrum_curves[name] = c
                        self.view.plot_widget.addItem(c)
                    self.view.spectrum_curves[name].setData(x=ve, y=ss)
                    
                if getattr(self.view, 'smoothing_params', None) is not None and getattr(self.view, 'spectrum_tabs', None) is not None:
                    if self.view.spectrum_tabs.indexOf(self.view.plot_widget_smooth) != -1:
                        method = self.view.smoothing_params['method']
                        ss_smooth = ss.copy()
                        try:
                            if method == 'boxcar':
                                from scipy.ndimage import uniform_filter1d
                                w = self.view.smoothing_params['window']
                                ss_smooth = uniform_filter1d(ss_smooth, size=w)
                            elif method == 'gaussian':
                                from scipy.ndimage import gaussian_filter1d
                                sigma = self.view.smoothing_params['sigma']
                                ss_smooth = gaussian_filter1d(ss_smooth, sigma=sigma)
                            elif method == 'savgol':
                                from scipy.signal import savgol_filter
                                w = self.view.smoothing_params['window']
                                p = self.view.smoothing_params['polyorder']
                                if len(ss_smooth) > w:
                                    ss_smooth = savgol_filter(ss_smooth, window_length=w, polyorder=p)
                        except Exception:
                            pass
                            
                        if name == "Whole Map":
                            self.view.spectrum_curve_smooth.setData(x=ve, y=ss_smooth)
                        else:
                            if name not in self.view.spectrum_curves_smooth:
                                c_s = pg.PlotDataItem([], [], stepMode="center", pen=pg.mkPen(color, width=2), name=name)
                                self.view.spectrum_curves_smooth[name] = c_s
                                self.view.plot_widget_smooth.addItem(c_s)
                            self.view.spectrum_curves_smooth[name].setData(x=ve, y=ss_smooth)

        if hasattr(self.view, 'lbl_rj_warning'):
            self.view.lbl_rj_warning.setVisible(is_rj_active)

        num_spatial_regions = len(active_rois)
        active_spatial_roi = active_rois[0]["roi"] if active_rois else None
        if num_spatial_regions <= 1 and self.view.contour_overlays:
            for ov in self.view.contour_overlays:
                if ov['is_static'] or ov['v_axis'] is None:
                    continue
                ov_name = ov['name']
                ov_color = ov['options']['color']

                if active_spatial_roi is None:
                    ov_sub_data = ov['cube']
                else:
                    ov_sub_data = active_spatial_roi.getArrayRegion(ov['cube'], self.view.view_channel.getImageItem(), axes=(1, 2))

                if "Max" in stat:
                    ov_spec = np.nanmax(ov_sub_data, axis=(1, 2))
                elif "Sum" in stat or "Flux Density" in stat:
                    ov_spec = np.nansum(ov_sub_data, axis=(1, 2))
                else:
                    ov_spec = np.nanmean(ov_sub_data, axis=(1, 2))

                # Inject Jy and K conversion math for the overlay
                unit_sel = self.view.combo_spec_unit.currentText()
                unit_lower = ov.get('display_unit', 'Unknown').replace(" ", "").lower()
                bmaj_o = ov.get('bmaj_array')
                bmin_o = ov.get('bmin_array')
                cd1 = ov.get('cdelt1')
                cd2 = ov.get('cdelt2')
                freq_o = ov.get('freq_array')
                
                can_convert = bmaj_o is not None and bmin_o is not None and cd1 and cd2
                
                with np.errstate(invalid='ignore', divide='ignore'):
                    if unit_sel == "Jy" and "jy" in unit_lower:
                        if can_convert and not ("pixel" in unit_lower or "pix" in unit_lower):
                            om_pix = abs(cd1 * cd2) * (u.deg ** 2)
                            om_beam = (np.pi * bmaj_o * bmin_o) / (4.0 * np.log(2.0)) * (u.deg ** 2)
                            n_beam_arr = om_beam.to(u.sr) / om_pix.to(u.sr)
                            ov_spec = ov_spec / n_beam_arr.value
                        elif not can_convert and not ("pixel" in unit_lower or "pix" in unit_lower):
                            print(f"WARNING: Overlay '{ov_name}' lacks beam metadata. Plotting in native units.")
                    elif unit_sel == "K" and "jy" in unit_lower:
                        if can_convert and freq_o is not None:
                            om_pix = abs(cd1 * cd2) * (u.deg ** 2)
                            om_beam = (np.pi * bmaj_o * bmin_o) / (4.0 * np.log(2.0)) * (u.deg ** 2)
                            omega = om_pix if ("pixel" in unit_lower or "pix" in unit_lower) else om_beam
                            sb = (ov_spec * u.Jy) / omega.to(u.sr)
                            tb_array = sb.to(u.K, equivalencies=u.brightness_temperature(freq_o * u.Hz))
                            ov_spec = tb_array.value
                        else:
                            print(f"WARNING: Overlay '{ov_name}' lacks beam metadata. Plotting in native units.")

                ov_v = ov['v_axis']
                ov_sort = np.argsort(ov_v)
                ov_vs, ov_ss = ov_v[ov_sort], ov_spec[ov_sort]

                from scipy.interpolate import interp1d
                try:
                    interp = interp1d(ov_vs, ov_ss, kind='linear', bounds_error=False,
                                      fill_value=(ov_ss[0], ov_ss[-1]))
                    ss_resampled = interp(vs)
                except Exception:
                    ss_resampled = np.full_like(vs, np.nan)

                display_name = f"{ov_name} (overlay)"
                curve_color = pg.mkPen(ov_color, width=2, style=Qt.DashLine)

                if display_name not in self.view.overlay_spectrum_curves:
                    c = pg.PlotDataItem([], [], stepMode="center", pen=curve_color, name=display_name)
                    self.view.overlay_spectrum_curves[display_name] = c
                    self.view.plot_widget.addItem(c)
                self.view.overlay_spectrum_curves[display_name].setPen(curve_color)
                self.view.overlay_spectrum_curves[display_name].setData(x=ve, y=ss_resampled)

                if getattr(self.view, 'smoothing_params', None) is not None and getattr(self.view, 'spectrum_tabs', None) is not None:
                    if self.view.spectrum_tabs.indexOf(self.view.plot_widget_smooth) != -1:
                        method = self.view.smoothing_params['method']
                        ss_ov_smooth = ss_resampled.copy()
                        try:
                            if method == 'boxcar':
                                from scipy.ndimage import uniform_filter1d
                                ss_ov_smooth = uniform_filter1d(ss_ov_smooth, size=self.view.smoothing_params['window'])
                            elif method == 'gaussian':
                                from scipy.ndimage import gaussian_filter1d
                                ss_ov_smooth = gaussian_filter1d(ss_ov_smooth, sigma=self.view.smoothing_params['sigma'])
                            elif method == 'savgol':
                                from scipy.signal import savgol_filter
                                w = self.view.smoothing_params['window']
                                p = self.view.smoothing_params['polyorder']
                                if len(ss_ov_smooth) > w:
                                    ss_ov_smooth = savgol_filter(ss_ov_smooth, window_length=w, polyorder=p)
                        except Exception:
                            pass

                        if display_name not in self.view.overlay_spectrum_curves_smooth:
                            c_s = pg.PlotDataItem([], [], stepMode="center", pen=curve_color, name=display_name)
                            self.view.overlay_spectrum_curves_smooth[display_name] = c_s
                            self.view.plot_widget_smooth.addItem(c_s)
                        self.view.overlay_spectrum_curves_smooth[display_name].setPen(curve_color)
                        self.view.overlay_spectrum_curves_smooth[display_name].setData(x=ve, y=ss_ov_smooth)
        else:
            self.view._clear_all_overlay_spectrum_curves()

        self.view._cleanup_removed_overlay_curves()

        self.view.plot_widget.autoRange()
        if hasattr(self.view, 'plot_widget_smooth'):
            self.view.plot_widget_smooth.autoRange()

        # Update legends
        has_overlay = bool(self.view.contour_overlays)
        base_prefix = "Base: " if has_overlay else ""

        if getattr(self.view.plot_widget, 'plotItem', None) is not None and self.view.plot_widget.plotItem.legend is not None:
            self.view.plot_widget.plotItem.legend.clear()
            if "Whole Map" in active_names:
                self.view.plot_widget.plotItem.legend.addItem(self.view.spectrum_curve, f"{base_prefix}Whole Map")
            for n, c in self.view.spectrum_curves.items():
                self.view.plot_widget.plotItem.legend.addItem(c, f"{base_prefix}{n}")
            for n, c in self.view.overlay_spectrum_curves.items():
                clean_name = n.replace(" (overlay)", "")
                self.view.plot_widget.plotItem.legend.addItem(c, f"Overlay: {clean_name}")

        if getattr(self.view, 'plot_widget_smooth', None) is not None and self.view.plot_widget_smooth.plotItem.legend is not None:
            self.view.plot_widget_smooth.plotItem.legend.clear()
            if "Whole Map" in active_names:
                self.view.plot_widget_smooth.plotItem.legend.addItem(self.view.spectrum_curve_smooth, f"{base_prefix}Whole Map")
            for n, c_s in getattr(self.view, 'spectrum_curves_smooth', {}).items():
                self.view.plot_widget_smooth.plotItem.legend.addItem(c_s, f"{base_prefix}{n}")
            for n, c_s in getattr(self.view, 'overlay_spectrum_curves_smooth', {}).items():
                clean_name = n.replace(" (overlay)", "")
                self.view.plot_widget_smooth.plotItem.legend.addItem(c_s, f"Overlay: {clean_name}")

        if self.view.catalog_overlay_items:
            ymax = ymax_global if ymax_global != -np.inf else 1.0
            for item in self.view.catalog_overlay_items:
                if isinstance(item, pg.TextItem):
                    item.setPos(item.pos().x(), ymax)
                    
        # Update popup if open
        if getattr(self.view, '_spectral_stats_popup', None) is not None and self.view._spectral_stats_popup.isVisible():
            self.view._run_spectral_stats_calc(self.view._spectral_stats_popup)

    def update_text_from_region(self):
        if self.view.cube_clean is None: return
        minX, maxX = self.view.region.getRegion()
        
        axis_str = self.view.combo_axis_type.currentText() if hasattr(self.view, 'combo_axis_type') else ""
        
        if axis_str == "Channel":
            self.view.input_vmin.setText(f"{int(round(minX))}")
            self.view.input_vmax.setText(f"{int(round(maxX))}")
        elif axis_str in ["Frequency", "Wavelength"]:
            self.view.input_vmin.setText(f"{minX:.6f}")
            self.view.input_vmax.setText(f"{maxX:.6f}")
        else:
            self.view.input_vmin.setText(f"{minX:.2f}")
            self.view.input_vmax.setText(f"{maxX:.2f}")
            
        self.view.input_vmin.setCursorPosition(0)
        self.view.input_vmax.setCursorPosition(0)

    def update_region_from_text(self):
        if self.view.cube_clean is None: return
        try:
            minX, maxX = float(self.view.input_vmin.text()), float(self.view.input_vmax.text())
            if minX < maxX: self.view.region.setRegion([minX, maxX])
        except ValueError: pass 

    def update_beam_visualizers(self, panel_type, panel_id=None):
        if self.view.cube_clean is None:
            return

        target_plot = self.view.plot_channel if panel_type == 'channel' else self.view.panels[panel_id]['plot_item']
        
        if not hasattr(self.view, 'beam_visualizer_items'):
            self.view.beam_visualizer_items = {}
        
        dict_key = 'channel' if panel_type == 'channel' else f'moment_{panel_id}'
        for item in self.view.beam_visualizer_items.get(dict_key, []):
            try:
                target_plot.vb.removeItem(item)
            except Exception:
                pass
        self.view.beam_visualizer_items[dict_key] = []
        
        if panel_type == 'moment':
            mtype = self.view.panels[panel_id]['combo'].currentText()
            if 'PV Diagram' in mtype:
                return
        
        beams_to_draw = []
        
        def get_beam_for_cube(bmaj_arr, bmin_arr, bpa_arr, bmaj_s, bmin_s, bpa_s):
            if bmaj_arr is not None and bmin_arr is not None:
                if panel_type == 'channel':
                    idx = self.view.slider_channel.value()
                    return bmaj_arr[idx], bmin_arr[idx], (bpa_arr[idx] if bpa_arr is not None else 0.0)
                elif panel_type == 'moment':
                    if getattr(self.view, 'slider_velocity', None) is not None:
                        rg = self.view.slider_velocity.getRegion()
                        v_min, v_max = rg
                        mask = (self.view.v_axis >= v_min) & (self.view.v_axis <= v_max) if self.view.v_axis[0] < self.view.v_axis[-1] else (self.view.v_axis <= v_min) & (self.view.v_axis >= v_max)
                        mask_idx = np.where(mask)[0]
                        if len(mask_idx) > 0:
                            b_ma = np.nanmedian(bmaj_arr[mask_idx])
                            b_mi = np.nanmedian(bmin_arr[mask_idx])
                            b_pa = np.nanmedian(bpa_arr[mask_idx]) if bpa_arr is not None else 0.0
                            return b_ma, b_mi, b_pa
                    return np.nanmedian(bmaj_arr), np.nanmedian(bmin_arr), np.nanmedian(bpa_arr) if bpa_arr is not None else 0.0
            return bmaj_s, bmin_s, bpa_s if bpa_s is not None else 0.0

        base_bmaj, base_bmin, base_bpa = get_beam_for_cube(
            self.view.bmaj_array, self.view.bmin_array, getattr(self.view, 'bpa_array', None),
            self.view.raw_header.get('BMAJ') if self.view.raw_header else None,
            self.view.raw_header.get('BMIN') if self.view.raw_header else None,
            self.view.raw_header.get('BPA', 0.0) if self.view.raw_header else 0.0
        )
        
        if base_bmaj and base_bmin:
            beams_to_draw.append({'bmaj': base_bmaj, 'bmin': base_bmin, 'bpa': base_bpa, 'color': 'white'})
        else:
            print(f"WARNING: No beam info found for base cube. Beam visualizer hidden.")
            
        processed_files = [self.view.current_file_name] if hasattr(self.view, 'current_file_name') else []
        for ov in self.view.contour_overlays:
            if panel_type == 'moment' and ov.get('_reproj_raw') is None:
                continue
            if ov['file'] in processed_files:
                continue
            processed_files.append(ov['file'])
            
            bmaj_o, bmin_o, bpa_o = get_beam_for_cube(
                ov.get('bmaj_array'), ov.get('bmin_array'), None,
                None, None, 0.0
            )
            if bmaj_o and bmin_o:
                beams_to_draw.append({'bmaj': bmaj_o, 'bmin': bmin_o, 'bpa': bpa_o, 'color': ov['color']})
                
        if not beams_to_draw:
            return
            
        t = np.linspace(0, 2*np.pi, 60)
        for b in beams_to_draw:
            bmaj_arcsec = b['bmaj'] * 3600.0
            bmin_arcsec = b['bmin'] * 3600.0
            bpa = b['bpa']
            
            angle_rad = np.radians(90.0 - bpa)
            cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
            
            x_el = (bmaj_arcsec / 2.0) * np.cos(t)
            y_el = (bmin_arcsec / 2.0) * np.sin(t)
            x_rot = x_el * cos_a - y_el * sin_a
            y_rot = x_el * sin_a + y_el * cos_a
            
            x_maj = np.array([-bmaj_arcsec/2.0, bmaj_arcsec/2.0])
            y_maj = np.array([0, 0])
            x_maj_rot = x_maj * cos_a - y_maj * sin_a
            y_maj_rot = x_maj * sin_a + y_maj * cos_a
            
            x_min = np.array([0, 0])
            y_min = np.array([-bmin_arcsec/2.0, bmin_arcsec/2.0])
            x_min_rot = x_min * cos_a - y_min * sin_a
            y_min_rot = x_min * sin_a + y_min * cos_a
            
            pen = pg.mkPen(b['color'], width=1.5)
            
            item_el = pg.PlotDataItem(x=x_rot, y=y_rot, pen=pen)
            item_maj = pg.PlotDataItem(x=x_maj_rot, y=y_maj_rot, pen=pen)
            item_min = pg.PlotDataItem(x=x_min_rot, y=y_min_rot, pen=pen)
            
            item_el.setZValue(10)
            item_maj.setZValue(10)
            item_min.setZValue(10)
            
            target_plot.vb.addItem(item_el, ignoreBounds=True)
            target_plot.vb.addItem(item_maj, ignoreBounds=True)
            target_plot.vb.addItem(item_min, ignoreBounds=True)
            
            self.view.beam_visualizer_items[dict_key].extend([item_el, item_maj, item_min])

        self.view.update_beam_positions(target_plot.vb)

    def update_beam_positions(self, view_box, view_range=None):
        if not hasattr(self.view, 'beam_visualizer_items'): return
        
        target_key = None
        if view_box == self.view.plot_channel.vb:
            target_key = 'channel'
        else:
            for i, p in enumerate(self.view.panels):
                if view_box == p['plot_item'].vb:
                    target_key = f'moment_{i}'
                    break
                    
        if target_key not in self.view.beam_visualizer_items: return
        items = self.view.beam_visualizer_items[target_key]
        if not items: return
        
        view_range = view_box.viewRange()
        v_x_min, v_x_max = view_range[0]
        v_y_min, v_y_max = view_range[1]

        cam_left_x = max(v_x_min, v_x_max)
        cam_bottom_y = min(v_y_min, v_y_max)
        
        img_left_x = (self.view.nx / 2.0) * self.view.pix_scale_arcsec
        img_bottom_y = -(self.view.ny / 2.0) * self.view.pix_scale_arcsec
        
        hud_left_x = min(cam_left_x, img_left_x)
        hud_bottom_y = max(cam_bottom_y, img_bottom_y)

        pad_x = abs(v_x_max - v_x_min) * 0.03
        pad_y = abs(v_y_max - v_y_min) * 0.03

        target_x = hud_left_x - pad_x
        target_y = hud_bottom_y + pad_y
        
        for item in items:
            item.setPos(target_x, target_y)


    def update_region_ui_visibility(self):
        active_rois = self.view.get_active_spectrum_rois()
        has_boxes = len(active_rois) >= 1
        self.view.btn_spectral_stats.setVisible(has_boxes)
        # Also refresh popup if open
        if self.view._spectral_stats_popup and self.view._spectral_stats_popup.isVisible():
            self.view.refresh_spectral_stats_popup()

    def apply_cmap(self, view, is_velocity):
        if is_velocity:
            pos = np.array([0.0, 0.5, 1.0])
            colors = np.array([[0, 0, 255, 255], [255, 255, 255, 255], [255, 0, 0, 255]], dtype=np.ubyte)
            view.setColorMap(pg.ColorMap(pos, colors))
        else: 
            view.ui.histogram.gradient.loadPreset(self.view.parent_window.current_cmap)
            
        grad = view.ui.histogram.gradient
        ticks = list(grad.ticks.keys())
        if len(ticks) > 3:
            sorted_ticks = sorted(ticks, key=lambda t: grad.ticks[t])
            keep = [sorted_ticks[0], sorted_ticks[len(sorted_ticks)//2], sorted_ticks[-1]]
            for t in sorted_ticks:
                if t not in keep:
                    t.hide()

    def configure_bottom_panel_controls(self, panel, mode):
        is_pv = mode == "PV Diagram"
        panel['aux_stack'].setCurrentWidget(panel['pv_controls_widget'] if is_pv else panel['thresh_widget'])
        if is_pv:
            panel['aux_stack'].show()
            if panel['combo_pv_cut'].currentText() == "None" and self.view.pv_cuts:
                preferred = self.view.get_selected_pv_cut_name() or self.view.pv_cuts[-1]["name"]
                panel['combo_pv_cut'].blockSignals(True)
                panel['combo_pv_cut'].setCurrentText(preferred)
                panel['combo_pv_cut'].blockSignals(False)
            if self.view.active_picker_panel == panel['id']:
                panel['btn_pick'].setChecked(False)
                self.view.active_picker_panel = None
        else:
            panel['aux_stack'].setVisible(True)

    def configure_bottom_panel_axes(self, panel, is_pv):
        plot_item = panel['plot_item']
        plot_item.invertX(not is_pv)
        plot_item.invertY(False)

        bottom_axis = plot_item.getAxis('bottom')
        left_axis = plot_item.getAxis('left')
        if is_pv:
            plot_item.setLabel('bottom', 'Offset along cut (arcsec)')
            plot_item.setLabel('left', 'Radio Velocity (km/s)')
            if hasattr(bottom_axis, 'update_wcs'):
                bottom_axis.update_wcs(None, self.view.nx, self.view.ny, self.view.pix_scale_arcsec, False)
            if hasattr(left_axis, 'update_wcs'):
                left_axis.update_wcs(None, self.view.nx, self.view.ny, self.view.pix_scale_arcsec, False)
        else:
            x_label = 'Right Ascension (J2000)' if self.view.parent_window.is_absolute_wcs else 'RA offset (arcsec)'
            y_label = 'Declination (J2000)' if self.view.parent_window.is_absolute_wcs else 'Dec offset (arcsec)'
            plot_item.setLabel('bottom', x_label)
            plot_item.setLabel('left', y_label)
            if hasattr(bottom_axis, 'update_wcs'):
                bottom_axis.update_wcs(self.view.wcs_2d, self.view.nx, self.view.ny, self.view.pix_scale_arcsec, self.view.parent_window.is_absolute_wcs)
            if hasattr(left_axis, 'update_wcs'):
                left_axis.update_wcs(self.view.wcs_2d, self.view.nx, self.view.ny, self.view.pix_scale_arcsec, self.view.parent_window.is_absolute_wcs)

    def _update_overlay_spatial_curve(self, plot_num, ov_name, x, y, color):
        x = np.asarray(x, dtype=float).ravel()
        y = np.asarray(y, dtype=float).ravel()
        if len(x) != len(y) or len(x) == 0:
            return
        attr_name = f'overlay_spatial_curves_{plot_num}'
        if not hasattr(self.view, attr_name):
            setattr(self.view, attr_name, {})
        curves = getattr(self.view, attr_name)
        if ov_name not in curves:
            c = self.view.plot_spatial_1.plot([], [], pen=pg.mkPen(color, width=2, style=Qt.DashLine), name=ov_name) if plot_num == 1 else \
                self.view.plot_spatial_2.plot([], [], pen=pg.mkPen(color, width=2, style=Qt.DashLine), name=ov_name)
            curves[ov_name] = c
        curves[ov_name].setPen(pg.mkPen(color, width=2, style=Qt.DashLine))
        curves[ov_name].setData(x, y)

    def _update_contour_options_rms_for(self, overlay_dict):
        slice_data = self.view._get_overlay_slice_for_channel(overlay_dict)
        if slice_data is None:
            return
        valid = slice_data[np.isfinite(slice_data)]
        if len(valid) > 1:
            rms = float(np.std(valid))
            if rms > 0:
                overlay_dict['options']['rms'] = rms
                overlay_dict['options']['lin_min'] = rms * 3
                overlay_dict['options']['lin_max'] = rms * 40
                overlay_dict['options']['log_min'] = max(rms, 1e-12)
                peak = float(np.nanmax(np.abs(valid)))
                overlay_dict['options']['log_max'] = max(peak, rms * 10)
        overlay_dict['options']['multipliers_str'] = '3, 5, 10, 20, 40'

    def _update_spectrum_state_machine(self):
        stat = self.view.combo_spec_stat.currentText()
        if stat == "Flux Density":
            for i in range(self.view.combo_spec_unit.count()):
                if self.view.combo_spec_unit.itemText(i) == "Jy":
                    self.view.combo_spec_unit.model().item(i).setEnabled(True)
                else:
                    self.view.combo_spec_unit.model().item(i).setEnabled(False)
            
            if self.view.combo_spec_unit.currentText() != "Jy":
                self.view.combo_spec_unit.setCurrentIndex(1) # Auto-switch to Jy
        else:
            for i in range(self.view.combo_spec_unit.count()):
                if "Native" in self.view.combo_spec_unit.itemText(i) or self.view.combo_spec_unit.itemText(i) in ["K", "Jy/beam"]:
                    self.view.combo_spec_unit.model().item(i).setEnabled(True)
                else:
                    self.view.combo_spec_unit.model().item(i).setEnabled(False)
            
            if self.view.combo_spec_unit.currentText() == "Jy":
                self.view.combo_spec_unit.setCurrentIndex(0) # Auto-switch to Native

    def _cleanup_stale_overlay_spatial_curves(self, active_names):
        for pnum in [1, 2]:
            attr = f'overlay_spatial_curves_{pnum}'
            if not hasattr(self.view, attr):
                continue
            curves = getattr(self.view, attr)
            plot = self.view.plot_spatial_1 if pnum == 1 else self.view.plot_spatial_2
            for name in list(curves.keys()):
                if name not in active_names:
                    c = curves.pop(name)
                    try:
                        plot.removeItem(c)
                    except Exception:
                        pass
                    c.setData([], [])

    def _cleanup_stale_overlay_spatial_curves_plot1(self, active_names):
        self.view._cleanup_stale_overlay_spatial_curves(active_names)
        curves = getattr(self.view, 'overlay_spatial_curves_2', {})
        for curve_name in list(curves.keys()):
            c = curves.pop(curve_name)
            try:
                self.view.plot_spatial_2.removeItem(c)
            except Exception:
                pass
            c.setData([], [])

    def _clear_all_overlay_spatial_curves(self):
        for pnum in [1, 2]:
            attr = f'overlay_spatial_curves_{pnum}'
            plot = self.view.plot_spatial_1 if pnum == 1 else self.view.plot_spatial_2
            if hasattr(self.view, attr):
                for c in getattr(self.view, attr).values():
                    try:
                        plot.removeItem(c)
                    except Exception:
                        pass
                    c.setData([], [])
                setattr(self.view, attr, {})

    def _clear_spatial_stats_panels(self):
        while self.view.spatial_stats_layout.count():
            item = self.view.spatial_stats_layout.takeAt(0)
            w = item.widget()
            if w:
                w.hide()
                w.setParent(None)
                w.deleteLater()


    def _clear_overlay_contours(self, overlay_dict=None):
        if overlay_dict is None:
            for ov in self.view.contour_overlays:
                self.view._clear_overlay_contours(ov)
            return
        for iso in overlay_dict.get('iso_items', []):
            iso.setParentItem(None)
            if iso.scene() is not None:
                self.view.view_channel.getView().removeItem(iso)
        overlay_dict['iso_items'] = []

    def _cleanup_removed_overlay_curves(self):
        active_names = set()
        for ov in self.view.contour_overlays:
            if not ov['is_static'] and ov['v_axis'] is not None:
                active_names.add(f"{ov['name']} (overlay)")
        for name in list(self.view.overlay_spectrum_curves.keys()):
            if name not in active_names:
                c = self.view.overlay_spectrum_curves.pop(name)
                if c.scene():
                    c.scene().removeItem(c)
                else:
                    self.view.plot_widget.removeItem(c)
        for name in list(self.view.overlay_spectrum_curves_smooth.keys()):
            if name not in active_names:
                c = self.view.overlay_spectrum_curves_smooth.pop(name)
                if c.scene():
                    c.scene().removeItem(c)
                else:
                    self.view.plot_widget_smooth.removeItem(c)

    def _clear_all_overlay_spectrum_curves(self):
        for name in list(self.view.overlay_spectrum_curves.keys()):
            c = self.view.overlay_spectrum_curves.pop(name)
            if c.scene():
                c.scene().removeItem(c)
            else:
                self.view.plot_widget.removeItem(c)
        for name in list(self.view.overlay_spectrum_curves_smooth.keys()):
            c = self.view.overlay_spectrum_curves_smooth.pop(name)
            if c.scene():
                c.scene().removeItem(c)
            else:
                self.view.plot_widget_smooth.removeItem(c)

    def refresh_spectral_stats_popup(self):
        popup = self.view._spectral_stats_popup
        if popup is None: return

        # ── Rebuild boxes (auto-flow into 3 cols) ────────────────────────
        while popup.boxes_grid.count():
            item = popup.boxes_grid.takeAt(0)
            if item.widget(): item.widget().deleteLater()
        active_rois = self.view.get_active_spectrum_rois()
        if not active_rois:
            popup.boxes_grid.addWidget(QLabel("No boxes drawn yet."), 0, 0)
        for idx, item in enumerate(active_rois):
            cb = QCheckBox(item["name"])
            cb.setChecked(False)
            cb.toggled.connect(lambda checked, p=popup: self.view._run_spectral_stats_calc(p))
            popup.boxes_grid.addWidget(cb, *divmod(idx, 3))

        # ── Rebuild apertures (auto-flow into 3 cols) ────────────────────
        while popup.apertures_grid.count():
            it = popup.apertures_grid.takeAt(0)
            if it.widget(): it.widget().deleteLater()
        spatial_rois = getattr(self.view, 'spectrum_spatial_rois', [])
        if not spatial_rois:
            popup.apertures_grid.addWidget(QLabel("No spatial regions defined."), 0, 0)
        for idx, r_dict in enumerate(spatial_rois):
            cb = QCheckBox(r_dict["name"])
            cb.setStyleSheet(f"color: {r_dict['color']};")
            cb.toggled.connect(lambda checked, p=popup: self.view._run_spectral_stats_calc(p))
            popup.apertures_grid.addWidget(cb, *divmod(idx, 3))

        popup.adjustSize()


    def refresh_spectral_stats_apertures(self):
        """Called when spatial regions are added/removed to refresh only the apertures section."""
        if self.view._spectral_stats_popup and self.view._spectral_stats_popup.isVisible():
            self.view.refresh_spectral_stats_popup()

    def _refresh_spatial_legend(self, plot_num):
        plot = self.view.plot_spatial_1 if plot_num == 1 else self.view.plot_spatial_2
        if not hasattr(plot, 'plotItem') or plot.plotItem.legend is None:
            return
        legend = plot.plotItem.legend
        legend.clear()
        base_curve = self.view.curve_spatial_1 if plot_num == 1 else self.view.curve_spatial_2
        if base_curve.xData is not None and len(base_curve.xData) > 0:
            legend.addItem(base_curve, "Base")
        curves = getattr(self.view, f'overlay_spatial_curves_{plot_num}', {})
        for name, c in curves.items():
            if c.xData is not None and len(c.xData) > 0:
                legend.addItem(c, name)

    def update_spectrum_region_calc(self, _=None):
        """Legacy stub: calculation is now handled by the popup widget."""
        if self.view._spectral_stats_popup and self.view._spectral_stats_popup.isVisible():
            self.view._run_spectral_stats_calc(self.view._spectral_stats_popup)
