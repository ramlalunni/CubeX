import csv
import numpy as np
import pyqtgraph as pg
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
import astropy.constants as const
import astropy.units as u
from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPainterPath
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QFileDialog, QMessageBox, QLineEdit, 
                             QComboBox, QFrame, QStackedWidget, QSizePolicy, QTabWidget,
                             QGroupBox, QCheckBox, QDialog, QScrollArea, QGridLayout)

try:
    from PyQt5.QtWidgets import FlowLayout
except ImportError:
    pass # we will use QHBoxLayout if FlowLayout is not easily available
from spectral_cube import SpectralCube

# Optional Numba acceleration (graceful fallback to NumPy when not installed)
try:
    import numba
    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False

# Import our modularized components
from src.core.splatalogue import SplatalogueWorker
from src.gui.custom import JumpSlider, fix_axis_scaling, WCSAxisItem
from src.gui.dialogs import LineCatalogDialog, LineSelectionDialog, ContourDialog

# ==============================================================================
# BILINEAR INTERPOLATION KERNEL (Numba-accelerated when available)
# ==============================================================================
# The kernel fuses the 4-gather bilinear interpolation into a single pass.
# Benefits vs. plain NumPy:
#   • Zero intermediate allocation (no v00/v10/v01/v11 temporaries)
#   • Parallelised over spectral channels via prange
#   • cache=True: compiled once, stored in __pycache__, instant on next run

if _NUMBA_AVAILABLE:
    @numba.njit(parallel=True, fastmath=True, cache=True)
    def _bilinear_interp_numba(cube, x0, y0, x1, y1, fx, fy, out):
        """Fill out[nv, n_valid] via bilinear interpolation; no temporaries."""
        nv = cube.shape[0]
        n  = x0.shape[0]
        for v in numba.prange(nv):          # parallel over spectral channels
            for s in range(n):
                w00 = (1.0 - fx[s]) * (1.0 - fy[s])
                w10 =        fx[s]  * (1.0 - fy[s])
                w01 = (1.0 - fx[s]) *        fy[s]
                w11 =        fx[s]  *        fy[s]
                out[v, s] = (
                    w00 * cube[v, x0[s], y0[s]]
                    + w10 * cube[v, x1[s], y0[s]]
                    + w01 * cube[v, x0[s], y1[s]]
                    + w11 * cube[v, x1[s], y1[s]]
                )

def _bilinear_interp_numpy(cube, x0, y0, x1, y1, fx, fy, out):
    """Pure-NumPy fallback: same result as the Numba kernel."""
    out[:] = (
        (1.0 - fx) * (1.0 - fy) * cube[:, x0, y0]
        + fx * (1.0 - fy) * cube[:, x1, y0]
        + (1.0 - fx) * fy * cube[:, x0, y1]
        + fx * fy * cube[:, x1, y1]
    )

if _NUMBA_AVAILABLE:
    _bilinear_interp = _bilinear_interp_numba
else:
    _bilinear_interp = _bilinear_interp_numpy

# ---- Warmup: trigger JIT compilation at import time with a tiny dummy ----
# cache=True means this only blocks on the very first run after installation;
# subsequent runs load pre-compiled native code and return in microseconds.
if _NUMBA_AVAILABLE:
    _wup_cube = np.zeros((2, 2, 2), dtype=np.float64)
    _wup_x0   = np.array([0], dtype=np.int64)
    _wup_y0   = np.array([0], dtype=np.int64)
    _wup_x1   = np.array([1], dtype=np.int64)
    _wup_y1   = np.array([1], dtype=np.int64)
    _wup_fx   = np.array([0.5], dtype=np.float64)
    _wup_fy   = np.array([0.5], dtype=np.float64)
    _wup_out  = np.zeros((2, 1), dtype=np.float64)
    _bilinear_interp_numba(_wup_cube, _wup_x0, _wup_y0, _wup_x1, _wup_y1,
                           _wup_fx, _wup_fy, _wup_out)
    del _wup_cube, _wup_x0, _wup_y0, _wup_x1, _wup_y1, _wup_fx, _wup_fy, _wup_out


# ==============================================================================
# FUSED MOMENT 1 / MOMENT 2 KERNEL (Numba-accelerated when available)
# ==============================================================================
# Computes M1 (velocity field) and M2 (velocity dispersion) in a single pass
# over the spectral axis, accumulating sum_w, sum_wv, sum_wvv per pixel.
# Eliminates three separate nansum calls and two (Nv,Nx,Ny) intermediate arrays.

if _NUMBA_AVAILABLE:
    @numba.njit(parallel=True, fastmath=True, cache=True)
    def _compute_moments_12_numba(mc, v_axis):
        """Single-pass kernel: returns (m1_map, m2_map) each (Nx, Ny).

        mc:     (Nv, Nx, Ny) float64 — NaN where below intensity threshold.
        v_axis: (Nv,) float64        — velocity values in km/s.
        """
        nv, nx, ny = mc.shape
        m1_out = np.full((nx, ny), np.nan)
        m2_out = np.full((nx, ny), np.nan)
        for x in numba.prange(nx):          # parallel over RA pixels
            for y in range(ny):
                sw = 0.0; swv = 0.0; swvv = 0.0
                for v in range(nv):
                    val = mc[v, x, y]
                    if not np.isnan(val):
                        vv    = v_axis[v]
                        sw   += val
                        swv  += val * vv
                        swvv += val * vv * vv
                if sw != 0.0:
                    m1         = swv / sw
                    variance   = swvv / sw - m1 * m1
                    m1_out[x, y] = m1
                    m2_out[x, y] = np.sqrt(variance if variance >= 0.0 else 0.0)
        return m1_out, m2_out

def _compute_moments_12_numpy(mc, v_axis):
    """NumPy fallback using tensordot for BLAS-accelerated weighted sums.

    Strategy vs. the old nansum approach:
      • Replace NaN→0 once (np.where) instead of letting each nansum scan for NaN.
      • Use np.tensordot to contract the spectral axis: NumPy internally reshapes
        mc_nz to (Nv, Nx*Ny) and calls a BLAS dgemm — much faster than
        nansum(mc * v_broad, axis=0) which allocates a full (Nv,Nx,Ny) intermediate.
      • Moment 2 via the computational variance formula  Var = E[v²] – E[v]²,
        eliminating two more (Nv,Nx,Ny) intermediates (v_broad–m1 and mc*(…)²).
    """
    mc_nz = np.where(np.isnan(mc), 0.0, mc)          # NaN→0, single pass
    with np.errstate(invalid='ignore', divide='ignore'):
        m0      = mc_nz.sum(axis=0)                   # sum(axis=0) uses SIMD/BLAS
        m0_safe = np.where(m0 != 0, m0, np.nan)
        # tensordot contracts along spectral axis (dim 0 of mc_nz, dim 0 of v_axis)
        sum_wv  = np.tensordot(v_axis,          mc_nz, axes=([0], [0]))  # (Nx,Ny)
        sum_wv2 = np.tensordot(v_axis ** 2,     mc_nz, axes=([0], [0]))  # (Nx,Ny)
        m1 = sum_wv  / m0_safe
        m2 = np.sqrt(np.maximum(sum_wv2 / m0_safe - m1 ** 2, 0.0))
    return m1, m2

if _NUMBA_AVAILABLE:
    _compute_moments_12 = _compute_moments_12_numba
else:
    _compute_moments_12 = _compute_moments_12_numpy

# Warmup — fires at import; cache=True makes this instant on subsequent runs.
if _NUMBA_AVAILABLE:
    _wup2_mc = np.zeros((2, 2, 2), dtype=np.float64)
    _wup2_v  = np.array([0.0, 1.0], dtype=np.float64)
    _compute_moments_12_numba(_wup2_mc, _wup2_v)
    del _wup2_mc, _wup2_v


# ==============================================================================
# BACKGROUND WORKER — moment maps & PV diagrams
# ==============================================================================

class MomentWorker(QThread):
    """
    Computes all moment maps and PV diagram data in a background thread.
    No Qt or PyQtGraph calls are made here — only plain NumPy.
    Emits result_ready(dict) when finished, or returns silently if cancelled.
    """
    result_ready = pyqtSignal(dict)

    def __init__(self, params: dict, generation: int):
        super().__init__()
        self.params = params
        self.generation = generation
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    # ------------------------------------------------------------------
    def run(self):
        p = self.params
        selected_cube = p['selected_cube']   # (Nv, Nx, Ny) numpy array
        sub_v         = p['sub_v']
        minX, maxX    = p['minX'], p['maxX']
        nx, ny        = p['nx'], p['ny']
        pix_scale     = p['pix_scale_arcsec']
        display_unit  = p['display_unit']
        panel_configs = p['panel_configs']

        span     = maxX - minX if maxX > minX else 1.0
        sub_v_f64 = sub_v.astype(np.float64)  # 1-D velocity axis for Numba kernels

        with np.errstate(invalid='ignore', divide='ignore'):
            m0_raw = np.nansum(selected_cube, axis=0)

        if self._cancelled:
            return

        panel_results = []
        for cfg in panel_configs:
            if self._cancelled:
                return

            mtype = cfg['mtype']

            # ---- PV Diagram ----
            if mtype == 'PV Diagram':
                pv_points  = cfg.get('pv_points')
                pv_cube    = cfg.get('pv_cube')
                pv_sub_v   = cfg.get('pv_sub_v')

                if pv_points is None or pv_cube is None or pv_sub_v is None:
                    panel_results.append({'mtype': mtype, 'data': None})
                    continue

                p1, p2 = pv_points
                offsets, pv_data = MomentWorker._sample_along_line(
                    p1, p2, pv_cube, nx, ny, pix_scale
                )
                if offsets is None or pv_data is None or pv_data.size == 0:
                    panel_results.append({'mtype': mtype, 'data': None})
                    continue

                sort_idx = np.argsort(pv_sub_v)
                v_sorted = pv_sub_v[sort_idx]
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

                panel_results.append({
                    'mtype':    mtype,
                    'data':     pv_sorted,
                    'offsets':  offsets,
                    'v_sorted': v_sorted,
                    'levels':   levels,
                    'dx': dx, 'dv': dv,
                })
                continue

            # ---- Moment maps ----
            t    = cfg['threshold']
            mask = (m0_raw > t)[np.newaxis, :, :]
            mc   = np.where(mask, selected_cube, np.nan)
            is_all_nan = np.isnan(mc).all()

            with np.errstate(invalid='ignore', divide='ignore'):
                if 'Moment 0' in mtype:
                    data = m0_raw.copy()
                    data[data == 0] = np.nan
                    levels   = (0, float(np.nanmax(data)) if not np.isnan(data).all() else 1.0)
                    unit_str = f"{display_unit} km/s"

                elif 'Moment 1' in mtype:
                    if is_all_nan:
                        data = np.full(m0_raw.shape, np.nan)
                    else:
                        mc_f64 = np.ascontiguousarray(mc, dtype=np.float64)
                        data, _ = _compute_moments_12(mc_f64, sub_v_f64)
                    levels   = (minX, maxX)
                    unit_str = 'km/s'

                elif 'Moment 2' in mtype:
                    if is_all_nan:
                        data = np.full(m0_raw.shape, np.nan)
                    else:
                        mc_f64 = np.ascontiguousarray(mc, dtype=np.float64)
                        _, data = _compute_moments_12(mc_f64, sub_v_f64)
                    levels   = (0, span / 2)
                    unit_str = 'km/s'

                elif 'Moment 8' in mtype:
                    if is_all_nan:
                        data = np.full(m0_raw.shape, np.nan)
                    else:
                        data = np.nanmax(mc, axis=0)
                    levels   = (0, float(np.nanmax(data)) if not np.isnan(data).all() else 1.0)
                    unit_str = display_unit

                elif 'Moment 9' in mtype:
                    if is_all_nan:
                        data = np.full(m0_raw.shape, np.nan)
                    else:
                        safe = np.copy(mc)
                        safe[np.isnan(safe)] = -np.inf
                        pidx = np.argmax(safe, axis=0)
                        data = sub_v[pidx]
                        m0   = np.nansum(mc, axis=0)
                        data[m0 == 0] = np.nan
                    levels   = (minX, maxX)
                    unit_str = 'km/s'

                else:
                    data     = np.full(m0_raw.shape, np.nan)
                    levels   = (0.0, 1.0)
                    unit_str = ''

            panel_results.append({
                'mtype':    mtype,
                'data':     data,
                'levels':   levels,
                'unit_str': unit_str,
            })

        if self._cancelled:
            return

        self.result_ready.emit({
            'generation':    self.generation,
            'm0_raw':        m0_raw,
            'panel_results': panel_results,
            'minX': minX, 'maxX': maxX,
        })

    # ------------------------------------------------------------------
    @staticmethod
    def _sample_along_line(p1, p2, cube_data, nx, ny, pix_scale_arcsec):
        """
        Pure-NumPy bilinear interpolation along a line through the cube.
        p1, p2 are world-coordinate arrays [x_arcsec, y_arcsec].
        Returns (offsets, samples.T) exactly as the Qt-dependent
        sample_cube_along_line does, but with no Qt dependencies.
        """
        dx_w = p2[0] - p1[0]
        dy_w = p2[1] - p1[1]
        length_arcsec = np.hypot(dx_w, dy_w)
        n_samples = max(int(np.ceil(length_arcsec / max(pix_scale_arcsec, 1e-6))) + 1, 2)

        xs = np.linspace(p1[0], p2[0], n_samples)
        ys = np.linspace(p1[1], p2[1], n_samples)
        offsets = np.linspace(0.0, length_arcsec, n_samples)

        # world_to_pixel (inline)
        start_x = (nx / 2) * pix_scale_arcsec
        start_y = -(ny / 2) * pix_scale_arcsec
        x_pix = (start_x - xs) / pix_scale_arcsec
        y_pix = (ys - start_y) / pix_scale_arcsec

        valid = (
            (x_pix >= 0.0) & (x_pix <= nx - 1) &
            (y_pix >= 0.0) & (y_pix <= ny - 1)
        )
        nv = cube_data.shape[0]
        samples = np.full((nv, n_samples), np.nan, dtype=np.float64)
        if np.any(valid):
            x0 = np.floor(x_pix[valid]).astype(np.int64)
            y0 = np.floor(y_pix[valid]).astype(np.int64)
            x1 = np.clip(x0 + 1, 0, nx - 1).astype(np.int64)
            y1 = np.clip(y0 + 1, 0, ny - 1).astype(np.int64)
            fx = (x_pix[valid] - x0).astype(np.float64)
            fy = (y_pix[valid] - y0).astype(np.float64)
            buf = np.ascontiguousarray(cube_data, dtype=np.float64)
            out = np.empty((nv, int(valid.sum())), dtype=np.float64)
            _bilinear_interp(buf, x0, y0, x1, y1, fx, fy, out)
            samples[:, valid] = out
        return offsets, samples.T


# ==============================================================================
# INDIVIDUAL EXPLORER TAB
# ==============================================================================

def make_roi_rotatable_with_ctrl(roi):
    original_move_point = roi.movePoint
    def custom_move_point(handle, pos, modifiers=Qt.NoModifier, finish=True, coords='parent'):
        h_dict = next((h for h in roi.handles if h['item'] == handle), None)
        if h_dict:
            if 'orig_center' not in h_dict:
                h_dict['orig_center'] = h_dict['center']
                
            if modifiers & Qt.ControlModifier:
                h_dict['type'] = 'r'
                h_dict['center'] = pg.Point(0.5, 0.5)
                handle.setCursor(Qt.ClosedHandCursor)
            else:
                h_dict['type'] = 's'
                h_dict['center'] = h_dict['orig_center']
                handle.setCursor(Qt.CrossCursor)
        original_move_point(handle, pos, modifiers, finish, coords)
    roi.movePoint = custom_move_point

class ChannelMapViewBox(pg.ViewBox):
    def __init__(self, *args, **kwds):
        super().__init__(*args, **kwds)
        self.drag_start = None
        self.current_roi = None
        self.parent_tab = None

    def mouseDragEvent(self, ev, axis=None):
        if ev.modifiers() == Qt.ControlModifier and self.parent_tab:
            mode = self.parent_tab.combo_panel_mode.currentText()
            if mode == "Spatial Analysis":
                tool = self.parent_tab.combo_spatial_tool.currentText()
                if tool == "Point":
                    ev.ignore()
                    return
                if ev.isStart():
                    self.drag_start = self.mapSceneToView(ev.buttonDownScenePos())
                    if tool == "Line":
                        self.current_roi = pg.LineSegmentROI([[self.drag_start.x(), self.drag_start.y()], [self.drag_start.x() + 0.1, self.drag_start.y() + 0.1]], pen=pg.mkPen('c', width=2))
                    elif tool == "Rectangle":
                        self.current_roi = pg.RectROI([self.drag_start.x(), self.drag_start.y()], [1e-5, 1e-5], pen=pg.mkPen('c', width=2))
                        self.current_roi.addScaleHandle([0, 0], [1, 1])
                        self.current_roi.addScaleHandle([1, 1], [0, 0])
                        self.current_roi.addScaleHandle([0, 1], [1, 0])
                        self.current_roi.addScaleHandle([1, 0], [0, 1])
                        self.current_roi.addScaleHandle([0.5, 0], [0.5, 1])
                        self.current_roi.addScaleHandle([0.5, 1], [0.5, 0])
                        self.current_roi.addScaleHandle([0, 0.5], [1, 0.5])
                        self.current_roi.addScaleHandle([1, 0.5], [0, 0.5])
                        make_roi_rotatable_with_ctrl(self.current_roi)
                    elif tool == "Ellipse":
                        self.current_roi = pg.EllipseROI([self.drag_start.x(), self.drag_start.y()], [1e-5, 1e-5], pen=pg.mkPen('c', width=2))
                        self.current_roi.addScaleHandle([0, 0], [1, 1])
                        self.current_roi.addScaleHandle([1, 1], [0, 0])
                        self.current_roi.addScaleHandle([0, 1], [1, 0])
                        self.current_roi.addScaleHandle([1, 0], [0, 1])
                        self.current_roi.addScaleHandle([0.5, 0], [0.5, 1])
                        self.current_roi.addScaleHandle([0.5, 1], [0.5, 0])
                        self.current_roi.addScaleHandle([0, 0.5], [1, 0.5])
                        self.current_roi.addScaleHandle([1, 0.5], [0, 0.5])
                        make_roi_rotatable_with_ctrl(self.current_roi)
                    
                    if self.current_roi:
                        self.addItem(self.current_roi)
                        ev.accept()
                elif ev.isFinish():
                    if self.current_roi:
                        self.parent_tab.add_spatial_region(self.current_roi, tool)
                        self.current_roi = None
                    ev.accept()
                else:
                    if self.current_roi:
                        current_pos = self.mapSceneToView(ev.scenePos())
                        if tool == "Line":
                            handles = self.current_roi.getHandles()
                            if len(handles) > 1:
                                self.current_roi.movePoint(handles[1], current_pos)
                        else:
                            w = current_pos.x() - self.drag_start.x()
                            h = current_pos.y() - self.drag_start.y()
                            self.current_roi.setSize([w, h])
                    ev.accept()
            elif self.parent_tab.is_pv_drawing_mode():
                if ev.isStart():
                    self.drag_start = self.mapSceneToView(ev.buttonDownScenePos())
                    self.current_roi = pg.LineSegmentROI(
                        [
                            [self.drag_start.x(), self.drag_start.y()],
                            [self.drag_start.x() + 0.1, self.drag_start.y() + 0.1],
                        ],
                        pen=pg.mkPen('m', width=2),
                    )
                    self.addItem(self.current_roi)
                    ev.accept()
                elif ev.isFinish():
                    if self.current_roi:
                        self.parent_tab.add_pv_cut(self.current_roi)
                        self.current_roi = None
                    ev.accept()
                else:
                    if self.current_roi:
                        current_pos = self.mapSceneToView(ev.scenePos())
                        handles = self.current_roi.getHandles()
                        if len(handles) > 1:
                            self.current_roi.movePoint(handles[1], current_pos)
                    ev.accept()
            else:
                super().mouseDragEvent(ev, axis)
        else:
            super().mouseDragEvent(ev, axis)

    def mouseClickEvent(self, ev):
        if ev.modifiers() == Qt.ControlModifier and self.parent_tab:
            mode = self.parent_tab.combo_panel_mode.currentText()
            if mode == "Spatial Analysis":
                tool = self.parent_tab.combo_spatial_tool.currentText()
                pos = self.mapSceneToView(ev.scenePos())
                
                hit = False
                for item in self.parent_tab.spatial_rois:
                    roi = item["roi"]
                    if hasattr(roi, 'shape'):
                        if roi.shape().contains(roi.mapFromScene(ev.scenePos())):
                            self.parent_tab.select_spatial_region(roi)
                            hit = True
                            
                if not hit:
                    for item in self.parent_tab.spatial_rois:
                        roi = item["roi"]
                        if isinstance(roi, pg.LineSegmentROI) and self.parent_tab.line_roi_hit_test(roi, ev.scenePos()):
                            self.parent_tab.select_spatial_region(roi)
                            hit = True
                            break
                if hit:
                    ev.accept()
                    return
                
                if tool == "Point":
                    # Use a small ROI for point since PointROI doesn't exist
                    sz = self.parent_tab.pix_scale_arcsec * 0.1 if hasattr(self.parent_tab, 'pix_scale_arcsec') else 0.1
                    roi = pg.ROI([pos.x() - sz/2, pos.y() - sz/2], [sz, sz], pen=pg.mkPen('c', width=2))
                    self.addItem(roi)
                    self.parent_tab.add_spatial_region(roi, "Point")
                    ev.accept()
                    return
            elif self.parent_tab.is_pv_drawing_mode():
                for item in self.parent_tab.pv_cuts:
                    if self.parent_tab.line_roi_hit_test(item["roi"], ev.scenePos()):
                        self.parent_tab.select_pv_cut(item["roi"])
                        ev.accept()
                        return
                
        super().mouseClickEvent(ev)

    def keyPressEvent(self, ev):
        if ev.key() == Qt.Key_Escape:
            if self.parent_tab and hasattr(self.parent_tab, 'spatial_rois_to_delete') and self.parent_tab.spatial_rois_to_delete:
                self.parent_tab.delete_selected_spatial_regions()
                ev.accept()
                return
            if self.parent_tab and hasattr(self.parent_tab, 'pv_cuts_to_delete') and self.parent_tab.pv_cuts_to_delete:
                self.parent_tab.delete_selected_pv_cuts()
                ev.accept()
                return
        super().keyPressEvent(ev)

class SpectrumViewBox(pg.ViewBox):
    def __init__(self, *args, **kwds):
        super().__init__(*args, **kwds)
        self.drag_start = None
        self.current_roi = None
        self.parent_tab = None

    def mouseDragEvent(self, ev, axis=None):
        if ev.modifiers() == Qt.ControlModifier:
            if ev.isStart():
                self.drag_start = self.mapSceneToView(ev.buttonDownScenePos())
                self.current_roi = pg.ROI([self.drag_start.x(), self.drag_start.y()], [0, 0], pen=pg.mkPen('c', width=2))
                self.current_roi.addScaleHandle([0, 0], [1, 1])
                self.current_roi.addScaleHandle([1, 1], [0, 0])
                self.current_roi.addScaleHandle([0, 1], [1, 0])
                self.current_roi.addScaleHandle([1, 0], [0, 1])
                self.current_roi.addScaleHandle([0.5, 0], [0.5, 1])
                self.current_roi.addScaleHandle([0.5, 1], [0.5, 0])
                self.current_roi.addScaleHandle([0, 0.5], [1, 0.5])
                self.current_roi.addScaleHandle([1, 0.5], [0, 0.5])
                self.addItem(self.current_roi)
                ev.accept()
            elif ev.isFinish():
                if self.parent_tab and self.current_roi:
                    pos = self.current_roi.pos()
                    size = self.current_roi.size()
                    nx = pos.x() + min(0, size.x())
                    ny = pos.y() + min(0, size.y())
                    nw = abs(size.x())
                    nh = abs(size.y())
                    self.current_roi.setPos([nx, ny])
                    self.current_roi.setSize([nw, nh])
                    self.parent_tab.add_spectrum_region(self.current_roi)
                self.current_roi = None
                ev.accept()
            else:
                if self.current_roi:
                    current_pos = self.mapSceneToView(ev.scenePos())
                    w = current_pos.x() - self.drag_start.x()
                    h = current_pos.y() - self.drag_start.y()
                    self.current_roi.setSize([w, h])
                ev.accept()
        else:
            super().mouseDragEvent(ev, axis)

    def mouseClickEvent(self, ev):
        if ev.modifiers() == Qt.ControlModifier:
            pos = self.mapSceneToView(ev.scenePos())
            if self.parent_tab:
                hit = False
                for item in self.parent_tab.spectrum_rois:
                    roi = item["roi"]
                    r_pos = roi.pos()
                    r_size = roi.size()
                    min_x = min(r_pos.x(), r_pos.x() + r_size.x())
                    max_x = max(r_pos.x(), r_pos.x() + r_size.x())
                    
                    # Spectrum ROIs are conceptually 1D velocity bands. 
                    # We allow clicking anywhere in the vertical column (ignoring Y bounds)
                    # so the user doesn't have to click exactly inside a potentially flat box.
                    if min_x <= pos.x() <= max_x:
                        self.parent_tab.select_region_for_deletion(roi)
                        hit = True
                        
                if hit:
                    ev.accept()
                    return
        super().mouseClickEvent(ev)

    def keyPressEvent(self, ev):
        if ev.key() == Qt.Key_Escape:
            if self.parent_tab and hasattr(self.parent_tab, 'rois_to_delete') and self.parent_tab.rois_to_delete:
                self.parent_tab.delete_selected_regions()
                ev.accept()
                return
        super().keyPressEvent(ev)

class ExplorerTab(QWidget):
    def __init__(self, parent_window):
        super().__init__()
        self.parent_window = parent_window 
        
        self.cube_clean = None 
        self.v_axis = None
        self.display_unit = "Unknown"
        self.spec_unit = "Unknown"
        self.pix_scale_arcsec = 1.0
        self.pixels_per_beam = 1.0
        self.nx = 1
        self.ny = 1
        self.raw_header = None
        self.fits_header_text = ""
        self.ch_levels = (0, 1) 
        
        self.wcs_2d = None
        self.is_absolute_wcs = False
        
        self.rest_freq_hz = None
        self.catalog_overlay_items = []
        self.spectrum_spatial_rois = [] # List of {"name": str, "roi": ROI, "checkbox": QCheckBox, "color": str}
        self.active_spatial_spectrum_roi = None
        self.roi_selected = False

        # Polygon drawing state
        self.is_drawing_polygon = False
        self.polygon_points = []
        self.polygon_preview_line = None
        
        self.region_colors = ['#2ecc71', '#fd6b6b', '#9b59b6', '#e74c3c', '#e67e22', '#1abc9c', '#e84393', '#fd79a8', '#00b894', '#a29bfe']
        self.roi_selected = False
        self.current_m0_raw = None
        self.active_picker_panel = None 
        self.pv_data = None
        self.pv_offset_axis = None
        self.pv_velocity_axis = None
        
        self.last_clicked_panel_id = 'channel' 
        self.contour_params = {'channel': None, 0: None, 1: None, 2: None}
        self.active_contours = {'channel': [], 0: [], 1: [], 2: []}
        
        self.playback_timer = QTimer()
        self.playback_timer.timeout.connect(self.step_channel)
        self.play_direction = 1

        # Background worker for moment / PV computation
        self._moment_worker = None
        self._moment_generation = 0

        self.initUI()

    def initUI(self):
        main_layout = QVBoxLayout(self)
        self.frames = {} 

        # ==================== TOP HALF ====================
        top_half = QHBoxLayout()

        # --- Channel Map ---
        self.frame_channel = QFrame()
        self.frame_channel.setObjectName("PanelFrame")
        channel_layout = QVBoxLayout(self.frame_channel)
        
        lbl_ch_title = QLabel("Channel Map")
        lbl_ch_title.setStyleSheet("font-weight: bold; color: #3498db; font-size: 13px;")
        channel_layout.addWidget(lbl_ch_title)

        ch_bottom = WCSAxisItem(orientation='bottom')
        ch_left = WCSAxisItem(orientation='left')
        self.channel_viewbox = ChannelMapViewBox()
        self.channel_viewbox.parent_tab = self
        self.plot_channel = pg.PlotItem(viewBox=self.channel_viewbox, axisItems={'bottom': ch_bottom, 'left': ch_left})
        
        self.plot_channel.invertX(True)
        self.plot_channel.setLabel('bottom', 'RA offset (arcsec)')
        self.plot_channel.setLabel('left', 'Dec offset (arcsec)')
        
        self.view_channel = pg.ImageView(view=self.plot_channel)
        self.plot_channel.invertY(False) 
        
        self.view_channel.ui.roiBtn.hide()
        self.view_channel.ui.menuBtn.hide()
        self.view_channel.ui.histogram.gradient.loadPreset('turbo')
        self.view_channel.ui.histogram.setFixedWidth(160) 
        fix_axis_scaling(self.view_channel.ui.histogram.axis) 
        channel_layout.addWidget(self.view_channel, stretch=1)

        self.lbl_hover_ch = QLabel("")
        self.lbl_hover_ch.setStyleSheet("color: #aaa; font-size: 9.5px;")
        channel_layout.addWidget(self.lbl_hover_ch)

        ctrl_layout = QHBoxLayout()
        ctrl_layout.setContentsMargins(0, 0, 0, 0)
        self.btn_prev = QPushButton("<")
        self.btn_play_rev = QPushButton("<<")
        self.btn_stop = QPushButton("■")
        self.btn_play_fwd = QPushButton(">>")
        self.btn_next = QPushButton(">")
        
        for btn in [self.btn_prev, self.btn_play_rev, self.btn_stop, self.btn_play_fwd, self.btn_next]:
            btn.setFixedWidth(40)
            btn.setFixedHeight(22)
            ctrl_layout.addWidget(btn)
        
        self.btn_prev.clicked.connect(lambda _=False: self.step_channel(-1))
        self.btn_next.clicked.connect(lambda _=False: self.step_channel(1))
        self.btn_play_rev.clicked.connect(lambda _=False: self.start_playback(-1))
        self.btn_play_fwd.clicked.connect(lambda _=False: self.start_playback(1))
        self.btn_stop.clicked.connect(self.stop_playback)
        
        self.slider_channel = JumpSlider(Qt.Horizontal)
        self.slider_channel.valueChanged.connect(self.update_channel_map)
        ctrl_layout.addWidget(self.slider_channel, stretch=1)
        
        ctrl_layout.addWidget(QLabel("Vel:"))
        self.input_channel_vel = QLineEdit("")
        self.input_channel_vel.setFixedWidth(55)
        self.input_channel_vel.setFixedHeight(22)
        self.input_channel_vel.editingFinished.connect(self.set_channel_from_text)
        ctrl_layout.addWidget(self.input_channel_vel)
        ctrl_layout.addWidget(QLabel("km/s"))
        
        channel_layout.addLayout(ctrl_layout)

        roi_layout = QHBoxLayout()
        roi_layout.setContentsMargins(0, 0, 0, 0)
        self.lbl_combo_roi = QLabel("Spectrum Region:")
        roi_layout.addWidget(self.lbl_combo_roi)
        self.combo_roi = QComboBox()
        self.combo_roi.setFixedHeight(22)
        self.combo_roi.addItems(["Whole Map", "Point (Beam)", "Ellipse", "Rectangle", "Custom Polygon"])
        self.combo_roi.activated[str].connect(self.change_roi)
        roi_layout.addWidget(self.combo_roi)
        
        self.btn_edit_region = QPushButton("Edit region")
        self.btn_edit_region.setFixedHeight(22)
        self.btn_edit_region.setStyleSheet("font-size: 11px; padding: 0px 4px;")
        self.btn_edit_region.hide()
        self.btn_edit_region.clicked.connect(self.open_edit_region_dialog)
        roi_layout.addWidget(self.btn_edit_region)

        self.lbl_spatial_tool = QLabel("Spatial Analysis Tool:")
        roi_layout.addWidget(self.lbl_spatial_tool)
        self.combo_spatial_tool = QComboBox()
        self.combo_spatial_tool.setFixedHeight(22)
        self.combo_spatial_tool.addItems(["None", "Point", "Line", "Rectangle", "Ellipse"])
        self.combo_spatial_tool.currentTextChanged.connect(self.change_spatial_tool)
        roi_layout.addWidget(self.combo_spatial_tool)
        
        self.lbl_spatial_tool.hide()
        self.combo_spatial_tool.hide()
        roi_layout.addStretch()
        channel_layout.addLayout(roi_layout)

        top_half.addWidget(self.frame_channel, stretch=4)
        self.frames['channel'] = self.frame_channel

        # --- Spectrum / Spatial ---
        self.frame_spectrum = QFrame()
        self.frame_spectrum.setObjectName("PanelFrame")
        self.panel_layout = QVBoxLayout(self.frame_spectrum)
        
        top_bar = QHBoxLayout()
        self.combo_panel_mode = QComboBox()
        self.combo_panel_mode.addItems(["Spectrum", "Spatial Analysis"])
        self.combo_panel_mode.setStyleSheet("font-weight: bold; color: #3498db; font-size: 13px;")
        self.combo_panel_mode.currentTextChanged.connect(self.switch_panel_mode)
        top_bar.addWidget(self.combo_panel_mode)
        top_bar.addStretch()
        self.panel_layout.addLayout(top_bar)
        
        self.stacked_panel = QStackedWidget()
        self.panel_layout.addWidget(self.stacked_panel)
        
        self.spectrum_widget = QWidget()
        spectrum_layout = QVBoxLayout(self.spectrum_widget)
        spectrum_layout.setContentsMargins(0,0,0,0)
        self.stacked_panel.addWidget(self.spectrum_widget)
        
        self.spatial_widget = QWidget()
        spatial_layout = QVBoxLayout(self.spatial_widget)
        spatial_layout.setContentsMargins(0,0,0,0)
        
        spatial_controls_layout = QHBoxLayout()
        self.lbl_spatial_region_sel = QLabel("Select Region:")
        self.combo_spatial_regions = QComboBox()
        self.combo_spatial_regions.addItem("None")
        self.combo_spatial_regions.currentTextChanged.connect(self.on_spatial_region_selected)
        
        self.btn_delete_spatial = QPushButton("Delete Selected")
        self.btn_delete_spatial.clicked.connect(self.delete_selected_spatial_via_button)
        
        spatial_controls_layout.addWidget(self.lbl_spatial_region_sel)
        spatial_controls_layout.addWidget(self.combo_spatial_regions)
        spatial_controls_layout.addWidget(self.btn_delete_spatial)
        spatial_controls_layout.addStretch()
        
        spatial_layout.addLayout(spatial_controls_layout)
        
        self.plot_spatial_1 = pg.PlotWidget(title="X Profile / Spatial Profile")
        self.plot_spatial_1.showGrid(x=True, y=True, alpha=0.3)
        self.plot_spatial_1.setLabel('bottom', 'Offset (arcsec)')
        self.plot_spatial_1.setLabel('left', 'Flux')
        self.curve_spatial_1 = self.plot_spatial_1.plot([], [], pen=pg.mkPen('w', width=2))
        
        self.plot_spatial_2 = pg.PlotWidget(title="Y Profile")
        self.plot_spatial_2.showGrid(x=True, y=True, alpha=0.3)
        self.plot_spatial_2.setLabel('bottom', 'Offset (arcsec)')
        self.plot_spatial_2.setLabel('left', 'Flux')
        self.curve_spatial_2 = self.plot_spatial_2.plot([], [], pen=pg.mkPen('w', width=2))
        
        self.lbl_spatial_stats = QLabel("Draw a region to see statistics.")
        self.lbl_spatial_stats.setAlignment(Qt.AlignCenter)
        self.lbl_spatial_stats.setStyleSheet("font-size: 13px; color: #aaa; background-color: #1a1a1a; border: 1px solid #333; border-radius: 4px; padding: 10px;")
        
        spatial_layout.addWidget(self.plot_spatial_1, stretch=1)
        spatial_layout.addWidget(self.plot_spatial_2, stretch=1)
        spatial_layout.addWidget(self.lbl_spatial_stats)
        self.stacked_panel.addWidget(self.spatial_widget)

        self.pv_widget = QWidget()
        pv_layout = QVBoxLayout(self.pv_widget)
        pv_layout.setContentsMargins(0, 0, 0, 0)

        pv_controls_layout = QHBoxLayout()
        self.lbl_pv_cut_sel = QLabel("Select Cut:")
        self.combo_pv_cuts = QComboBox()
        self.combo_pv_cuts.addItem("None")
        self.combo_pv_cuts.currentTextChanged.connect(self.on_pv_cut_selected)
        self.btn_delete_pv = QPushButton("Delete Selected")
        self.btn_delete_pv.clicked.connect(self.delete_selected_pv_via_button)
        pv_controls_layout.addWidget(self.lbl_pv_cut_sel)
        pv_controls_layout.addWidget(self.combo_pv_cuts)
        pv_controls_layout.addWidget(self.btn_delete_pv)
        pv_controls_layout.addStretch()
        pv_layout.addLayout(pv_controls_layout)

        self.lbl_pv_help = QLabel("Ctrl+drag on the channel map to draw a PV cut.")
        self.lbl_pv_help.setAlignment(Qt.AlignCenter)
        self.lbl_pv_help.setStyleSheet("font-size: 12px; color: #aaa;")
        pv_layout.addWidget(self.lbl_pv_help)

        self.pv_plot_item = pg.PlotItem(title="PV Diagram")
        self.pv_plot_item.showGrid(x=True, y=True, alpha=0.3)
        self.pv_plot_item.setLabel('bottom', 'Offset along cut (arcsec)')
        self.pv_plot_item.setLabel('left', 'Radio Velocity (km/s)')
        self.pv_view = pg.ImageView(view=self.pv_plot_item)
        self.pv_view.ui.roiBtn.hide()
        self.pv_view.ui.menuBtn.hide()
        self.pv_view.ui.histogram.gradient.loadPreset('turbo')
        self.pv_view.ui.histogram.setFixedWidth(160)
        fix_axis_scaling(self.pv_view.ui.histogram.axis)
        pv_layout.addWidget(self.pv_view, stretch=1)

        self.lbl_hover_pv = QLabel("")
        self.lbl_hover_pv.setStyleSheet("color: #aaa; font-size: 9.5px;")
        pv_layout.addWidget(self.lbl_hover_pv)
        self.stacked_panel.addWidget(self.pv_widget)
        
        self.spectrum_viewbox = SpectrumViewBox()
        self.spectrum_viewbox.parent_tab = self
        self.plot_widget = pg.PlotWidget(viewBox=self.spectrum_viewbox)
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        fix_axis_scaling(self.plot_widget.getAxis('left')) 
        self.plot_widget.setLabel('bottom', 'Radio Velocity (km/s)')
        self.plot_widget.setLabel('left', 'Flux') 
        self.plot_widget.addLegend(offset=(10, 10))
        
        self.spectrum_curves = {} # mapping from region name to PlotDataItem
        # Original curve (fallback for Whole Map)
        self.spectrum_curve = pg.PlotDataItem([], [], stepMode="center", pen=pg.mkPen('w', width=2))
        self.plot_widget.addItem(self.spectrum_curve)
        
        self.v_line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('g', width=2, style=Qt.DashLine))
        self.v_line.hide()
        self.plot_widget.addItem(self.v_line)
        
        self.region = pg.LinearRegionItem([0, 1], brush=pg.mkBrush(52, 152, 219, 40))
        self.region.setZValue(10)
        self.region.hide()
        for line in self.region.lines:
            line.setPen(pg.mkPen(color='#3498db', width=3))
            line.setHoverPen(pg.mkPen(color='#f1c40f', width=5))
        self.plot_widget.addItem(self.region)
        
        self.box_regions = QGroupBox("Regions")
        self.box_regions.setStyleSheet("QGroupBox { color: white; font-weight: bold; } QCheckBox { color: white; }")
        self.box_regions_layout = QHBoxLayout(self.box_regions)
        self.box_regions_layout.setContentsMargins(5, 10, 5, 5)
        self.box_regions.hide()
        spectrum_layout.addWidget(self.box_regions)
        
        self.spectrum_tabs = QTabWidget()
        self.spectrum_tabs.addTab(self.plot_widget, "Original")
        self.spectrum_tabs.tabBar().hide()
        spectrum_layout.addWidget(self.spectrum_tabs)
        
        self.spectrum_viewbox_smooth = SpectrumViewBox()
        self.spectrum_viewbox_smooth.parent_tab = self
        self.plot_widget_smooth = pg.PlotWidget(viewBox=self.spectrum_viewbox_smooth)
        self.plot_widget_smooth.showGrid(x=True, y=True, alpha=0.3)
        fix_axis_scaling(self.plot_widget_smooth.getAxis('left'))
        self.plot_widget_smooth.setLabel('bottom', 'Radio Velocity (km/s)')
        self.plot_widget_smooth.setLabel('left', 'Flux')
        self.plot_widget_smooth.addLegend(offset=(10, 10))
        
        self.spectrum_curves_smooth = {}
        self.spectrum_curve_smooth = pg.PlotDataItem([], [], stepMode="center", pen=pg.mkPen('c', width=2))
        self.plot_widget_smooth.addItem(self.spectrum_curve_smooth)
        
        self.smoothing_params = None
        self.spectrum_tabs.currentChanged.connect(self._on_spectrum_tab_changed)
        
        self.lbl_hover_spec = QLabel("")
        self.lbl_hover_spec.setStyleSheet("color: #aaa; font-size: 9.5px;")
        spectrum_layout.addWidget(self.lbl_hover_spec)

        input_layout = QHBoxLayout()
        input_layout.addWidget(QLabel("Statistic:"))
        self.combo_spec_stat = QComboBox()
        self.combo_spec_stat.addItems(["Mean", "Max", "Sum"])
        self.combo_spec_stat.currentTextChanged.connect(self.update_spectrum)
        self.combo_spec_stat.currentTextChanged.connect(lambda: self.lbl_region_result.setText("---"))
        input_layout.addWidget(self.combo_spec_stat)
        
        self.btn_smooth = QPushButton("Smooth")
        self.btn_smooth.clicked.connect(self.open_smoothing_dialog)
        input_layout.addWidget(self.btn_smooth)
        
        self.btn_remove_smooth = QPushButton("Remove Smoothed")
        self.btn_remove_smooth.clicked.connect(self.remove_smoothed_spectrum)
        self.btn_remove_smooth.hide()
        input_layout.addWidget(self.btn_remove_smooth)
        
        input_layout.addStretch()
        input_layout.addWidget(QLabel("Min Vel:"))
        self.input_vmin = QLineEdit("0.00")
        self.input_vmin.setMinimumWidth(80)
        input_layout.addWidget(self.input_vmin)
        
        input_layout.addWidget(QLabel("Max Vel:"))
        self.input_vmax = QLineEdit("1.00")
        self.input_vmax.setMinimumWidth(80)
        input_layout.addWidget(self.input_vmax)
        
        # Hidden backing widgets (used by update_spectrum_region_calc)
        self.combo_regions = QComboBox(); self.combo_regions.addItem("None")
        self.combo_regions_2 = QComboBox(); self.combo_regions_2.addItem("None")
        self.combo_regions_3 = QComboBox(); self.combo_regions_3.addItem("None")
        self.combo_region_calc = QComboBox()
        self.combo_region_calc.addItems(["Integrated intensity", "RMS"])
        self.lbl_region_result = QLabel("---")
        self.lbl_regions = QLabel("")
        self.lbl_plus1 = QLabel("")
        self.lbl_plus2 = QLabel("")
        self.lbl_calc = QLabel("")

        # Spectral Statistics popup button
        self.btn_spectral_stats = QPushButton("📊 Spectral Statistics")
        self.btn_spectral_stats.setToolTip("Open spectral statistics panel for drawn velocity boxes")
        self.btn_spectral_stats.clicked.connect(self.open_spectral_stats_popup)
        self.btn_spectral_stats.hide()
        input_layout.addWidget(self.btn_spectral_stats)

        self.spectrum_rois = []
        self.rois_to_delete = []
        self._spectral_stats_popup = None
        
        spectrum_layout.addLayout(input_layout)

        self.input_vmin.editingFinished.connect(self.update_region_from_text)
        self.input_vmax.editingFinished.connect(self.update_region_from_text)
        self.region.sigRegionChanged.connect(self.update_text_from_region)
        self.region.sigRegionChanged.connect(self._on_region_drag_start)
        self.region.sigRegionChangeFinished.connect(self._on_region_drag_end)
        self._region_dragging = False

        top_half.addWidget(self.frame_spectrum, stretch=7)
        self.spatial_rois = []
        self.spatial_rois_to_delete = []
        self.pv_cuts = []
        self.pv_cuts_to_delete = []
        self.frames['spectrum'] = self.frame_spectrum
        main_layout.addLayout(top_half, stretch=1)

        # ==================== BOTTOM HALF ====================
        self.bottom_half = QHBoxLayout()
        self.bottom_half.setSpacing(6)
        self.panels = []

        moment_options = ["Moment 0 (Integrated Intensity)", "Moment 1 (Velocity Field)",
                          "Moment 2 (Velocity Dispersion)", "Moment 8 (Peak Intensity)",
                          "Moment 9 (Peak Velocity)", "PV Diagram"]

        for i, default_option in enumerate([moment_options[0], moment_options[1], moment_options[2]]):
            panel = {}
            panel_frame = QFrame()
            panel_frame.setObjectName("PanelFrame")
            panel_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            panel_frame.setMinimumWidth(0)
            
            panel_layout = QVBoxLayout(panel_frame)
            
            top_ctrl_layout = QHBoxLayout()
            combo = QComboBox()
            combo.addItems(moment_options)
            combo.setCurrentText(default_option)
            combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(14)
            top_ctrl_layout.addWidget(combo, stretch=1)
            
            aux_stack = QStackedWidget()
            aux_stack.setContentsMargins(0, 0, 0, 0)
            aux_stack.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

            thresh_widget = QWidget()
            thresh_layout = QHBoxLayout(thresh_widget)
            thresh_layout.setContentsMargins(5, 0, 0, 0)

            thresh_layout.addWidget(QLabel("Min. Intensity:"))
            input_thresh = QLineEdit("0.000")
            input_thresh.setMinimumWidth(80)
            thresh_layout.addWidget(input_thresh)
            
            btn_pick = QPushButton("💧")
            btn_pick.setToolTip("Pick min. intensity threshold from raw map")
            btn_pick.setFixedWidth(35)
            btn_pick.setCheckable(True)
            btn_pick.setStyleSheet("QPushButton:checked { background-color: #d35400; font-weight: bold; }")
            thresh_layout.addWidget(btn_pick)
            thresh_layout.addStretch()
            
            pv_controls_widget = QWidget()
            pv_controls_layout = QHBoxLayout(pv_controls_widget)
            pv_controls_layout.setContentsMargins(5, 0, 0, 0)
            pv_controls_layout.setSpacing(2)
            pv_controls_layout.addWidget(QLabel("Cut:"))
            combo_pv_cut = QComboBox()
            combo_pv_cut.addItem("None")
            combo_pv_cut.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            combo_pv_cut.setMinimumContentsLength(6)
            combo_pv_cut.setFixedWidth(80)
            pv_controls_layout.addWidget(combo_pv_cut, stretch=1)
            pv_controls_layout.addWidget(QLabel("Range:"))
            combo_pv_range = QComboBox()
            combo_pv_range.addItems(["Selected Range", "Full Cube"])
            combo_pv_range.setCurrentText("Selected Range")
            combo_pv_range.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            combo_pv_range.setMinimumContentsLength(9)
            combo_pv_range.setFixedWidth(98)
            pv_controls_layout.addWidget(combo_pv_range)
            btn_delete_pv = QPushButton("Del")
            btn_delete_pv.setFixedWidth(42)
            pv_controls_layout.addWidget(btn_delete_pv)

            aux_stack.addWidget(thresh_widget)
            aux_stack.addWidget(pv_controls_widget)
            top_ctrl_layout.addWidget(aux_stack, stretch=1)
            panel_layout.addLayout(top_ctrl_layout)
            
            p_bottom = WCSAxisItem(orientation='bottom')
            p_left = WCSAxisItem(orientation='left')
            plot_item = pg.PlotItem(axisItems={'bottom': p_bottom, 'left': p_left})
            
            plot_item.invertX(True)
            plot_item.setLabel('bottom', 'RA offset (arcsec)')
            plot_item.setLabel('left', 'Dec offset (arcsec)')
            
            view = pg.ImageView(view=plot_item)
            plot_item.invertY(False)

            view.ui.roiBtn.hide()
            view.ui.menuBtn.hide()
            view.ui.histogram.setFixedWidth(160) 
            fix_axis_scaling(view.ui.histogram.axis) 
            panel_layout.addWidget(view, stretch=1)
            
            lbl_hover = QLabel("")
            lbl_hover.setStyleSheet("color: #aaa; font-size: 9.5px;")
            panel_layout.addWidget(lbl_hover)

            self.bottom_half.addWidget(panel_frame, stretch=1)
            self.frames[i] = panel_frame
            
            panel['combo'] = combo
            panel['view'] = view
            panel['plot_item'] = plot_item
            panel['aux_stack'] = aux_stack
            panel['thresh_widget'] = thresh_widget
            panel['input_thresh'] = input_thresh
            panel['btn_pick'] = btn_pick
            panel['pv_controls_widget'] = pv_controls_widget
            panel['combo_pv_cut'] = combo_pv_cut
            panel['combo_pv_range'] = combo_pv_range
            panel['btn_delete_pv'] = btn_delete_pv
            panel['lbl_hover'] = lbl_hover
            panel['current_data'] = None
            panel['pv_offset_axis'] = None
            panel['pv_velocity_axis'] = None
            panel['id'] = i
            panel['unit'] = ''
            self.panels.append(panel)

            combo.currentTextChanged.connect(self.update_moment_maps)
            input_thresh.editingFinished.connect(self.update_moment_maps)
            combo_pv_cut.currentTextChanged.connect(lambda _text, p_id=i: self.on_panel_pv_cut_selected(p_id))
            combo_pv_range.currentTextChanged.connect(self.update_moment_maps)
            btn_delete_pv.clicked.connect(lambda _checked=False, p_id=i: self.delete_panel_pv_cut(p_id))
            plot_item.scene().sigMouseMoved.connect(lambda pos, p=panel: self.hover_panel(pos, p))
            btn_pick.clicked.connect(lambda checked, p_id=i: self.set_active_picker(checked, p_id))

        main_layout.addLayout(self.bottom_half, stretch=1)

        self.set_active_panel('channel')

        self.plot_channel.scene().sigMouseMoved.connect(lambda pos: self.hover_event(pos, self.plot_channel, self.get_current_channel_data(), self.lbl_hover_ch, 'channel'))
        self.plot_widget.scene().sigMouseMoved.connect(lambda pos: self.hover_spectrum(pos, self.plot_widget))
        self.plot_widget_smooth.scene().sigMouseMoved.connect(lambda pos: self.hover_spectrum(pos, self.plot_widget_smooth))
        self.pv_plot_item.scene().sigMouseMoved.connect(self.hover_pv)
        
        self.plot_widget.scene().sigMouseClicked.connect(lambda event: self.universal_click_handler(event, self.plot_widget))
        self.plot_widget_smooth.scene().sigMouseClicked.connect(lambda event: self.universal_click_handler(event, self.plot_widget_smooth))
        self.plot_channel.scene().sigMouseClicked.connect(lambda event: self.universal_click_handler(event, self.plot_channel))
        self.pv_plot_item.scene().sigMouseClicked.connect(lambda _event: self.set_active_panel('spectrum'))
        for p in self.panels:
            p['plot_item'].scene().sigMouseClicked.connect(lambda event, view=p['plot_item']: self.universal_click_handler(event, view))


    def switch_panel_mode(self, mode):
        if mode == "Spectrum":
            self.stacked_panel.setCurrentWidget(self.spectrum_widget)
            self.lbl_combo_roi.show()
            self.combo_roi.show()
            self.lbl_spatial_tool.hide()
            self.combo_spatial_tool.hide()
        elif mode == "Spatial Analysis":
            self.stacked_panel.setCurrentWidget(self.spatial_widget)
            self.lbl_combo_roi.hide()
            self.combo_roi.hide()
            self.lbl_spatial_tool.show()
            self.combo_spatial_tool.show()
            self.change_spatial_tool(self.combo_spatial_tool.currentText())

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            # Delete Spatial Analysis region
            if getattr(self, 'spatial_rois_to_delete', []):
                self.delete_selected_spatial_regions()
            # Delete Spectrum region (if highlighted)
            if getattr(self, 'active_spatial_spectrum_roi', None):
                self.remove_spatial_spectrum_roi(self.active_spatial_spectrum_roi)
                self.active_spatial_spectrum_roi = None
        super().keyPressEvent(event)

    def open_smoothing_dialog(self):
        # Mutual exclusion: warn if Edit Region dialog is open
        if getattr(self, '_region_dialog', None) and self._region_dialog.isVisible():
            msg = QMessageBox(self.window())
            msg.setWindowTitle("Dialog Open")
            msg.setText("Please close the 'Edit Region' dialog before opening Smooth.")
            msg.setIcon(QMessageBox.Information)
            msg.setWindowFlags(msg.windowFlags() | Qt.WindowStaysOnTopHint)
            msg.exec_()
            self._region_dialog.raise_()
            self._region_dialog.activateWindow()
            return
        from src.gui.dialogs import SpectralSmoothingDialog
        self._smooth_dialog_active = True
        try:
            dialog = SpectralSmoothingDialog(self.window())
            dialog.setWindowModality(Qt.WindowModal)
            if dialog.exec_():
                params = dialog.get_params()
                if params:
                    self.smoothing_params = params
                    if self.spectrum_tabs.indexOf(self.plot_widget_smooth) == -1:
                        self.spectrum_tabs.addTab(self.plot_widget_smooth, "Smoothed")
                    self.spectrum_tabs.tabBar().show()
                    self.spectrum_tabs.setCurrentWidget(self.plot_widget_smooth)
                    self.btn_remove_smooth.show()
                    self.update_spectrum()
        finally:
            self._smooth_dialog_active = False

    def remove_smoothed_spectrum(self):
        self.smoothing_params = None
        idx = self.spectrum_tabs.indexOf(self.plot_widget_smooth)
        if idx != -1:
            self.spectrum_tabs.removeTab(idx)
        self.spectrum_tabs.tabBar().hide()
        self.spectrum_tabs.setCurrentWidget(self.plot_widget)
        self.btn_remove_smooth.hide()
        self._on_spectrum_tab_changed()
        self.update_spectrum()

    def get_active_spectrum_plot(self):
        if getattr(self, 'spectrum_tabs', None) is not None and getattr(self, 'plot_widget_smooth', None) is not None:
            if self.spectrum_tabs.currentWidget() == self.plot_widget_smooth:
                return self.plot_widget_smooth
        return getattr(self, 'plot_widget', None)

    def get_active_spectrum_rois(self):
        if getattr(self, 'spectrum_tabs', None) is not None and getattr(self, 'plot_widget_smooth', None) is not None:
            if self.spectrum_tabs.currentWidget() == self.plot_widget_smooth:
                if not hasattr(self, 'spectrum_rois_smooth'):
                    self.spectrum_rois_smooth = []
                return self.spectrum_rois_smooth
        if not hasattr(self, 'spectrum_rois'):
            self.spectrum_rois = []
        return self.spectrum_rois

    def _on_spectrum_tab_changed(self):
        current_widget = self.spectrum_tabs.currentWidget()
        if not current_widget: return
        
        if current_widget == getattr(self, 'plot_widget_smooth', None):
            self.btn_remove_smooth.show()
        else:
            self.btn_remove_smooth.hide()
            
        self.update_region_ui_visibility()
        self.rename_regions()
        self.update_spectrum_region_calc()

    def any_pv_panels_active(self):
        return any(panel['combo'].currentText() == "PV Diagram" for panel in self.panels)

    def is_pv_drawing_mode(self):
        return self.combo_panel_mode.currentText() != "Spatial Analysis" and self.any_pv_panels_active()

    def get_velocity_subset(self, use_full_range=False):
        if self.cube_clean is None:
            return None, None, None, None
        if use_full_range:
            return self.cube_clean, self.v_axis, float(np.nanmin(self.v_axis)), float(np.nanmax(self.v_axis))

        minX, maxX = self.region.getRegion()
        search_axis = self.v_axis if self.v_axis[0] < self.v_axis[-1] else self.v_axis[::-1]
        idx_min = np.searchsorted(search_axis, minX)
        idx_max = np.searchsorted(search_axis, maxX)
        if self.v_axis[0] > self.v_axis[-1]:
            idx_min, idx_max = len(self.v_axis) - idx_max, len(self.v_axis) - idx_min
        if idx_max <= idx_min:
            return None, None, minX, maxX
        return self.cube_clean[idx_min:idx_max, :, :], self.v_axis[idx_min:idx_max], minX, maxX

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
                bottom_axis.update_wcs(None, self.nx, self.ny, self.pix_scale_arcsec, False)
            if hasattr(left_axis, 'update_wcs'):
                left_axis.update_wcs(None, self.nx, self.ny, self.pix_scale_arcsec, False)
        else:
            x_label = 'Right Ascension (J2000)' if self.parent_window.is_absolute_wcs else 'RA offset (arcsec)'
            y_label = 'Declination (J2000)' if self.parent_window.is_absolute_wcs else 'Dec offset (arcsec)'
            plot_item.setLabel('bottom', x_label)
            plot_item.setLabel('left', y_label)
            if hasattr(bottom_axis, 'update_wcs'):
                bottom_axis.update_wcs(self.wcs_2d, self.nx, self.ny, self.pix_scale_arcsec, self.parent_window.is_absolute_wcs)
            if hasattr(left_axis, 'update_wcs'):
                left_axis.update_wcs(self.wcs_2d, self.nx, self.ny, self.pix_scale_arcsec, self.parent_window.is_absolute_wcs)

    def configure_bottom_panel_controls(self, panel, mode):
        is_pv = mode == "PV Diagram"
        panel['aux_stack'].setCurrentWidget(panel['pv_controls_widget'] if is_pv else panel['thresh_widget'])
        if is_pv:
            panel['aux_stack'].show()
            if panel['combo_pv_cut'].currentText() == "None" and self.pv_cuts:
                preferred = self.get_selected_pv_cut_name() or self.pv_cuts[-1]["name"]
                panel['combo_pv_cut'].blockSignals(True)
                panel['combo_pv_cut'].setCurrentText(preferred)
                panel['combo_pv_cut'].blockSignals(False)
            if self.active_picker_panel == panel['id']:
                panel['btn_pick'].setChecked(False)
                self.active_picker_panel = None
        else:
            panel['aux_stack'].setVisible("Moment 0" not in mode)

    def get_pv_cut_by_name(self, name):
        for item in self.pv_cuts:
            if item["name"] == name:
                return item
        return None

    def get_selected_pv_cut_name(self):
        if not self.pv_cuts_to_delete:
            return None
        for item in self.pv_cuts:
            if item["roi"] == self.pv_cuts_to_delete[-1]:
                return item["name"]
        return None

    def set_selected_pv_cut(self, name):
        self.pv_cuts_to_delete.clear()
        for item in self.pv_cuts:
            is_selected = item["name"] == name
            if is_selected:
                self.pv_cuts_to_delete.append(item["roi"])
            item["roi"].setPen(pg.mkPen('m', width=3) if is_selected else pg.mkPen('c', width=2))
            direction_item = item.get("direction_item")
            if direction_item is not None:
                direction_item.setPen(pg.mkPen('#f1c40f' if is_selected else '#f7dc6f', width=3 if is_selected else 2))

    def refresh_all_pv_cut_combos(self):
        cut_names = [item["name"] for item in self.pv_cuts]
        combos = []
        if hasattr(self, 'combo_pv_cuts'):
            combos.append(self.combo_pv_cuts)
        combos.extend(panel['combo_pv_cut'] for panel in self.panels)

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
        name = self.panels[panel_id]['combo_pv_cut'].currentText()
        if name != "None":
            self.set_selected_pv_cut(name)
        self.update_moment_maps()

    def delete_panel_pv_cut(self, panel_id):
        name = self.panels[panel_id]['combo_pv_cut'].currentText()
        if name == "None":
            return
        self.set_selected_pv_cut(name)
        self.delete_selected_pv_cuts()

    def clear_panel_pv_diagram(self, panel):
        panel['current_data'] = None
        panel['pv_offset_axis'] = None
        panel['pv_velocity_axis'] = None
        panel['unit'] = self.display_unit
        panel['view'].clear()
        panel['lbl_hover'].setText("")
        self.draw_contours(panel['id'], panel['view'], None)

    def update_panel_pv_diagram(self, panel):
        self.configure_bottom_panel_axes(panel, is_pv=True)
        panel['view'].ui.histogram.gradient.loadPreset('turbo')
        panel['view'].ui.histogram.axis.setLabel(f"Flux ({self.display_unit})")
        panel['plot_item'].setTitle("PV Diagram")

        cut_name = panel['combo_pv_cut'].currentText()
        active_item = self.get_pv_cut_by_name(cut_name)
        if active_item is None:
            self.clear_panel_pv_diagram(panel)
            return

        use_full_range = panel['combo_pv_range'].currentText() == "Full Cube"
        cube_data, velocity_axis, _, _ = self.get_velocity_subset(use_full_range=use_full_range)
        if cube_data is None or velocity_axis is None:
            self.clear_panel_pv_diagram(panel)
            return

        offsets, pv_data = self.sample_cube_along_line(active_item["roi"], cube_data)
        if offsets is None or pv_data is None or pv_data.size == 0:
            self.clear_panel_pv_diagram(panel)
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
        panel['unit'] = self.display_unit
        panel['view'].setImage(
            pv_sorted,
            autoLevels=False,
            autoHistogramRange=False,
            levels=levels,
            scale=(dx, dv),
            pos=(0.0, v_sorted[0]),
        )
        self.draw_contours(panel['id'], panel['view'], None)

    def hover_panel(self, pos, panel):
        if panel['combo'].currentText() == "PV Diagram":
            self.hover_panel_pv(pos, panel)
        else:
            self.hover_event(pos, panel['plot_item'], panel['current_data'], panel['lbl_hover'], panel['id'])

    def hover_panel_pv(self, pos, panel):
        self.clear_all_hover_labels()
        if panel['current_data'] is None or panel['pv_offset_axis'] is None or panel['pv_velocity_axis'] is None:
            return
        if not panel['plot_item'].sceneBoundingRect().contains(pos):
            return

        mp = panel['plot_item'].vb.mapSceneToView(pos)
        x_idx = int(np.abs(panel['pv_offset_axis'] - mp.x()).argmin())
        y_idx = int(np.abs(panel['pv_velocity_axis'] - mp.y()).argmin())
        if 0 <= x_idx < panel['current_data'].shape[0] and 0 <= y_idx < panel['current_data'].shape[1]:
            val = panel['current_data'][x_idx, y_idx]
            val_str = f"{val:.3e}" if (np.isfinite(val) and abs(val) < 1e-3 and abs(val) > 0) else f"{val:.4g}" if np.isfinite(val) else "NaN"
            panel['lbl_hover'].setText(
                f"Offset: {panel['pv_offset_axis'][x_idx]:.2f} arcsec | Vel: {panel['pv_velocity_axis'][y_idx]:.2f} km/s | {val_str} {self.display_unit}"
            )
            panel['lbl_hover'].setStyleSheet("color: #3498db; font-weight: bold; font-size: 9.5px;")

    def change_spatial_tool(self, tool):
        if self.cube_clean is None: return
        
        if tool == "None":
            self.plot_spatial_1.hide()
            self.plot_spatial_2.hide()
            self.lbl_spatial_stats.setText("Choose a tool to begin analysis.")
            return

        if tool == "Point":
            self.plot_spatial_1.show()
            self.plot_spatial_1.setTitle("X Profile")
            self.plot_spatial_2.show()
            self.lbl_spatial_stats.hide()
        elif tool == "Line":
            self.plot_spatial_1.show()
            self.plot_spatial_1.setTitle("Spatial Profile")
            self.plot_spatial_2.hide()
            self.lbl_spatial_stats.hide()
        else:
            self.plot_spatial_1.hide()
            self.plot_spatial_2.hide()
            self.lbl_spatial_stats.show()

        # Auto-draw default shape
        sz = self.nx * self.pix_scale_arcsec * 0.15
        num = len(self.spatial_rois)
        cx, cy = num * sz * 0.1, num * sz * 0.1
        
        new_roi = None
        if tool == "Point":
            sz_pt = self.pix_scale_arcsec * 0.1
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
            self.view_channel.addItem(new_roi)
            self.add_spatial_region(new_roi, tool)
            self.select_spatial_region(new_roi)

    def add_spatial_region(self, roi, tool):
        name = f"{tool} {len(self.spatial_rois) + 1}"
        self.spatial_rois.append({"name": name, "roi": roi, "tool": tool})
        
        self.combo_spatial_regions.blockSignals(True)
        self.combo_spatial_regions.addItem(name)
        self.combo_spatial_regions.setCurrentText(name)
        self.combo_spatial_regions.blockSignals(False)
        
        roi.sigRegionChanged.connect(self.update_spatial_analysis)
        
        for item in self.spatial_rois:
            if item["roi"] != roi:
                item["roi"].setPen(pg.mkPen('c', width=2))
        roi.setPen(pg.mkPen('y', width=3))
        self.spatial_rois_to_delete = [roi]
        
        self.update_spatial_analysis()

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
        for item in self.spatial_rois:
            if item["name"] == name:
                self.select_spatial_region(item["roi"])
                break

    def delete_selected_spatial_via_button(self):
        self.delete_selected_spatial_regions()

    def select_spatial_region(self, roi):
        self.spatial_rois_to_delete = [roi]
        for item in self.spatial_rois:
            if item["roi"] == roi:
                item["roi"].setPen(pg.mkPen('y', width=3))
                self.combo_spatial_regions.blockSignals(True)
                self.combo_spatial_regions.setCurrentText(item["name"])
                self.combo_spatial_regions.blockSignals(False)
                
                # Show edit button
                if item["tool"] in ["Ellipse", "Rectangle", "Point", "Line"]:
                    self.btn_edit_region.show()
                else:
                    self.btn_edit_region.hide()
            else:
                item["roi"].setPen(pg.mkPen('c', width=2))
        self.update_spatial_analysis()

    def delete_selected_spatial_regions(self):
        for roi in list(self.spatial_rois_to_delete):
            if roi.scene():
                roi.scene().removeItem(roi)
            else:
                try:
                    self.view_channel.removeItem(roi)
                except:
                    pass
            
            self.spatial_rois = [item for item in self.spatial_rois if item["roi"] != roi]
        self.spatial_rois_to_delete.clear()
        
        self.combo_spatial_regions.blockSignals(True)
        self.combo_spatial_regions.clear()
        self.combo_spatial_regions.addItem("None")
        for item in self.spatial_rois:
            self.combo_spatial_regions.addItem(item["name"])
        self.combo_spatial_regions.blockSignals(False)
        
        if self.spatial_rois:
            self.combo_spatial_regions.setCurrentText(self.spatial_rois[-1]["name"])
        else:
            self.combo_spatial_regions.setCurrentText("None")
            
        self.update_spatial_analysis()

    def add_pv_cut(self, roi):
        name = f"Cut {len(self.pv_cuts) + 1}"
        cut_info = {"name": name, "roi": roi}
        self.pv_cuts.append(cut_info)

        text_item = pg.TextItem(text=name, color=(220, 220, 220, 180), anchor=(0, 1))
        self.plot_channel.addItem(text_item)
        direction_item = pg.PlotDataItem(
            [],
            [],
            connect='finite',
            pen=pg.mkPen('#f7dc6f', width=3),
        )
        direction_item.setZValue(20)
        self.plot_channel.addItem(direction_item)

        def update_annotations(r=roi, t=text_item, a=direction_item):
            points = self.get_line_roi_points(r)
            if points is None:
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
            head_len = min(max(6.0 * self.pix_scale_arcsec, 0.18 * length), 0.32 * length)
            head_width = 0.75 * head_len
            base_center = tip - unit * head_len
            left = base_center + normal * (0.5 * head_width)
            right = base_center - normal * (0.5 * head_width)

            t.setPos(p2[0], p2[1])
            a.setData(
                [left[0], tip[0], np.nan, right[0], tip[0]],
                [left[1], tip[1], np.nan, right[1], tip[1]],
            )

        roi.sigRegionChanged.connect(update_annotations)
        roi.sigRegionChanged.connect(self.update_moment_maps)
        update_annotations()

        cut_info["text_item"] = text_item
        cut_info["direction_item"] = direction_item
        cut_info["update_annotations"] = update_annotations

        self.refresh_all_pv_cut_combos()
        self.set_selected_pv_cut(name)
        for panel in self.panels:
            if panel['combo'].currentText() == "PV Diagram" and panel['combo_pv_cut'].currentText() == "None":
                panel['combo_pv_cut'].setCurrentText(name)
        self.update_moment_maps()

    def on_pv_cut_selected(self, name):
        self.set_selected_pv_cut(name)
        self.update_moment_maps()

    def select_pv_cut(self, roi):
        for item in self.pv_cuts:
            if item["roi"] == roi:
                self.set_selected_pv_cut(item["name"])
                active_panel_id = getattr(self, 'last_clicked_panel_id', None)
                if isinstance(active_panel_id, int):
                    panel = self.panels[active_panel_id]
                    if panel['combo'].currentText() == "PV Diagram":
                        panel['combo_pv_cut'].setCurrentText(item["name"])
                return

    def delete_selected_pv_via_button(self):
        self.delete_selected_pv_cuts()

    def delete_selected_pv_cuts(self):
        for roi in list(self.pv_cuts_to_delete):
            if roi.scene():
                roi.scene().removeItem(roi)
            else:
                try:
                    self.view_channel.removeItem(roi)
                except Exception:
                    pass

            for i, item in enumerate(self.pv_cuts):
                if item["roi"] == roi:
                    text_item = item.get("text_item")
                    if text_item is not None:
                        if text_item.scene():
                            text_item.scene().removeItem(text_item)
                        else:
                            self.plot_channel.removeItem(text_item)
                    direction_item = item.get("direction_item")
                    if direction_item is not None:
                        if direction_item.scene():
                            direction_item.scene().removeItem(direction_item)
                        else:
                            self.plot_channel.removeItem(direction_item)
                    self.pv_cuts.pop(i)
                    break

        self.pv_cuts_to_delete.clear()
        for idx, item in enumerate(self.pv_cuts, start=1):
            item["name"] = f"Cut {idx}"
            if "text_item" in item:
                item["text_item"].setText(item["name"])
        self.refresh_all_pv_cut_combos()
        if self.pv_cuts:
            self.set_selected_pv_cut(self.pv_cuts[-1]["name"])
        else:
            if hasattr(self, 'combo_pv_cuts'):
                self.combo_pv_cuts.blockSignals(True)
                self.combo_pv_cuts.setCurrentText("None")
                self.combo_pv_cuts.blockSignals(False)
            for panel in self.panels:
                panel['combo_pv_cut'].blockSignals(True)
                panel['combo_pv_cut'].setCurrentText("None")
                panel['combo_pv_cut'].blockSignals(False)
                self.clear_panel_pv_diagram(panel)
            self.pv_data = None
            self.pv_offset_axis = None
            self.pv_velocity_axis = None
            self.pv_view.clear()
            self.lbl_hover_pv.setText("")
        self.update_moment_maps()

    def clear_pv_cuts(self):
        self.pv_cuts_to_delete = [item["roi"] for item in self.pv_cuts]
        self.delete_selected_pv_cuts()

    def get_line_roi_points(self, roi):
        pts = roi.getSceneHandlePositions()
        if len(pts) < 2:
            return None
        p1 = self.plot_channel.vb.mapSceneToView(pts[0][1])
        p2 = self.plot_channel.vb.mapSceneToView(pts[1][1])
        return np.array([p1.x(), p1.y()], dtype=float), np.array([p2.x(), p2.y()], dtype=float)

    def world_to_pixel(self, x_world, y_world):
        start_x = (self.nx / 2) * self.pix_scale_arcsec
        start_y = -(self.ny / 2) * self.pix_scale_arcsec
        x_pix = (start_x - x_world) / self.pix_scale_arcsec
        y_pix = (y_world - start_y) / self.pix_scale_arcsec
        return x_pix, y_pix

    def sample_cube_along_line(self, roi, cube_data=None):
        points = self.get_line_roi_points(roi)
        if points is None:
            return None, None
        if cube_data is None:
            cube_data = self.cube_clean

        p1, p2 = points
        dx_world = p2[0] - p1[0]
        dy_world = p2[1] - p1[1]
        length_arcsec = np.hypot(dx_world, dy_world)
        n_samples = max(int(np.ceil(length_arcsec / max(self.pix_scale_arcsec, 1e-6))) + 1, 2)

        xs = np.linspace(p1[0], p2[0], n_samples)
        ys = np.linspace(p1[1], p2[1], n_samples)
        offsets = np.linspace(0.0, length_arcsec, n_samples)

        x_pix, y_pix = self.world_to_pixel(xs, ys)
        valid = (
            (x_pix >= 0.0)
            & (x_pix <= self.nx - 1)
            & (y_pix >= 0.0)
            & (y_pix <= self.ny - 1)
        )

        nv = cube_data.shape[0]
        samples = np.full((nv, n_samples), np.nan, dtype=np.float64)
        if np.any(valid):
            x0 = np.floor(x_pix[valid]).astype(np.int64)
            y0 = np.floor(y_pix[valid]).astype(np.int64)
            x1 = np.clip(x0 + 1, 0, self.nx - 1).astype(np.int64)
            y1 = np.clip(y0 + 1, 0, self.ny - 1).astype(np.int64)
            fx = (x_pix[valid] - x0).astype(np.float64)
            fy = (y_pix[valid] - y0).astype(np.float64)
            buf = np.ascontiguousarray(cube_data, dtype=np.float64)
            out = np.empty((nv, int(valid.sum())), dtype=np.float64)
            _bilinear_interp(buf, x0, y0, x1, y1, fx, fy, out)
            samples[:, valid] = out

        return offsets, samples.T

    def update_pv_diagram(self, _=None):
        self.update_moment_maps()

    def update_spatial_analysis(self, _=None):
        if self.cube_clean is None: return
        data = self.get_current_channel_data()
        if data is None: return
        
        active_item = None
        if self.spatial_rois_to_delete:
            for item in self.spatial_rois:
                if item["roi"] == self.spatial_rois_to_delete[-1]:
                    active_item = item
                    break
        elif self.spatial_rois:
            active_item = self.spatial_rois[-1]
            
        if not active_item:
            self.curve_spatial_1.setData([], [])
            self.curve_spatial_2.setData([], [])
            self.lbl_spatial_stats.setText("Draw a region to see statistics.")
            return
            
        roi = active_item["roi"]
        tool = active_item["tool"]
        
        try:
            if tool == "Point":
                pos = roi.pos()
                size = roi.size()
                cx, cy = pos.x() + size.x()/2, pos.y() + size.y()/2
                
                start_x = (self.nx / 2) * self.pix_scale_arcsec
                start_y = -(self.ny / 2) * self.pix_scale_arcsec
                x_idx = int((cx - start_x) / (-self.pix_scale_arcsec))
                y_idx = int((cy - start_y) / self.pix_scale_arcsec)
                
                if 0 <= x_idx < self.nx and 0 <= y_idx < self.ny:
                    x_profile = data[:, y_idx]
                    y_profile = data[x_idx, :]
                    
                    x_axis = (self.nx / 2 - np.arange(self.nx)) * self.pix_scale_arcsec
                    y_axis = (np.arange(self.ny) - self.ny / 2) * self.pix_scale_arcsec
                    
                    self.curve_spatial_1.setData(x_axis, x_profile)
                    self.plot_spatial_1.setLabel('left', f'Flux ({self.display_unit})')
                    self.plot_spatial_1.setLabel('bottom', 'RA offset (arcsec)')
                    
                    self.curve_spatial_2.setData(y_axis, y_profile)
                    self.plot_spatial_2.setLabel('left', f'Flux ({self.display_unit})')
                    self.plot_spatial_2.setLabel('bottom', 'Dec offset (arcsec)')
                    

            elif tool == "Line":
                offsets, profile_2d = self.sample_cube_along_line(roi, data[np.newaxis, :, :])
                if profile_2d is not None and profile_2d.size > 0:
                    profile = profile_2d[:, 0]
                    self.curve_spatial_1.setData(offsets, profile)
                    self.curve_spatial_2.setData([], [])
                    self.plot_spatial_1.setLabel('left', f'Flux ({self.display_unit})')
                    self.plot_spatial_1.setLabel('bottom', 'Distance (arcsec)')
                    self.lbl_spatial_stats.setText("Line profile plotted.")
                    
            elif tool in ["Rectangle", "Ellipse"]:
                sub_data = roi.getArrayRegion(data, self.view_channel.getImageItem())
                if sub_data is not None and sub_data.size > 0:
                    valid = sub_data[~np.isnan(sub_data)]
                    if len(valid) > 0:
                        mean_val = np.mean(valid)
                        sum_val = np.sum(valid)
                        max_val = np.max(valid)
                        min_val = np.min(valid)
                        rms_val = np.sqrt(np.mean(valid**2))
                        std_val = np.std(valid)
                        
                        stats_text = (
                            f"<b>Statistics</b><br><br>"
                            f"<table style='width:100%'>"
                            f"<tr><td>Mean:</td><td>{mean_val:.4g} {self.display_unit}</td></tr>"
                            f"<tr><td>Sum:</td><td>{sum_val:.4g} {self.display_unit}</td></tr>"
                            f"<tr><td>Peak:</td><td>{max_val:.4g} {self.display_unit}</td></tr>"
                            f"<tr><td>Min:</td><td>{min_val:.4g} {self.display_unit}</td></tr>"
                            f"<tr><td>RMS:</td><td>{rms_val:.4g} {self.display_unit}</td></tr>"
                            f"<tr><td>Std Dev:</td><td>{std_val:.4g} {self.display_unit}</td></tr>"
                            f"</table>"
                        )
                        self.lbl_spatial_stats.setText(stats_text)
                    else:
                        self.lbl_spatial_stats.setText("No valid data in region.")


        except Exception as e:
            with open("line_debug.txt", "a") as dbg:
                dbg.write(f"Exception: {e}\n")
            print(f"Error in update_spatial_analysis: {e}")


    def update_wcs_mode(self, is_absolute):
        self.is_absolute_wcs = is_absolute
        x_label = 'Right Ascension (J2000)' if is_absolute else 'RA offset (arcsec)'
        y_label = 'Declination (J2000)' if is_absolute else 'Dec offset (arcsec)'

        self.plot_channel.setLabel('bottom', x_label)
        self.plot_channel.setLabel('left', y_label)
        self.plot_channel.getAxis('bottom').update_wcs(self.wcs_2d, self.nx, self.ny, self.pix_scale_arcsec, is_absolute)
        self.plot_channel.getAxis('left').update_wcs(self.wcs_2d, self.nx, self.ny, self.pix_scale_arcsec, is_absolute)
        for panel in self.panels:
            self.configure_bottom_panel_axes(panel, panel['combo'].currentText() == "PV Diagram")

    def set_active_panel(self, panel_id):
        self.last_clicked_panel_id = panel_id
        for pid, frame in self.frames.items():
            if pid == panel_id:
                frame.setStyleSheet("QFrame#PanelFrame { border: 2px solid #3498db; border-radius: 6px; background-color: #1a1a1a; }")
            else:
                frame.setStyleSheet("QFrame#PanelFrame { border: 1px solid #333; border-radius: 6px; background-color: #121212; }")
        
        self.parent_window.update_menu_states()

    def set_active_picker(self, checked, panel_id):
        for i, p in enumerate(self.panels):
            if i != panel_id: p['btn_pick'].setChecked(False)
        self.active_picker_panel = panel_id if checked else None

    def universal_click_handler(self, event, source_plot):
        if self.cube_clean is None: return
        
        try:
            mp = source_plot.vb.mapSceneToView(event.scenePos())
        except Exception:
            if hasattr(source_plot, 'plotItem'):
                mp = source_plot.plotItem.vb.mapSceneToView(event.scenePos())
            else:
                return

        if source_plot == self.plot_channel:
            self.set_active_panel('channel')
        elif source_plot == self.plot_widget:
            self.set_active_panel('spectrum')
        else:
            for i, p in enumerate(self.panels):
                if source_plot == p['plot_item']:
                    self.set_active_panel(i)
                    break

        if source_plot == self.plot_channel and getattr(self, 'is_drawing_polygon', False):
            if event.button() == Qt.LeftButton:
                mp = source_plot.vb.mapSceneToView(event.scenePos())
                
                # Check for closing condition: double click OR clicking near first point
                is_closing = False
                if len(self.polygon_points) >= 3:
                    p0 = self.polygon_points[0]
                    dist = np.sqrt((mp.x() - p0[0])**2 + (mp.y() - p0[1])**2)
                    # Tolerance: 3% of view width
                    view_range = source_plot.vb.viewRect()
                    tol = view_range.width() * 0.03
                    if event.double() or dist < tol:
                        is_closing = True
                
                if is_closing:
                    self.finalize_polygon()
                else:
                    self.polygon_points.append([mp.x(), mp.y()])
                    # Update preview
                    pts = np.array(self.polygon_points)
                    self.polygon_preview_line.setData(pts[:,0], pts[:,1], symbol='o', symbolSize=5)
            elif event.button() == Qt.RightButton:
                self.cancel_polygon()
            return

        if source_plot == self.plot_widget:
            if event.button() == Qt.LeftButton and event.modifiers() == Qt.NoModifier:
                idx = (np.abs(self.v_axis - mp.x())).argmin()
                self.slider_channel.setValue(idx)
            return

        if self.active_picker_panel is not None and self.current_m0_raw is not None:
            if event.button() == Qt.LeftButton:
                pos = event.scenePos()
                if source_plot.sceneBoundingRect().contains(pos):
                    start_x = (self.nx / 2) * self.pix_scale_arcsec
                    start_y = -(self.ny / 2) * self.pix_scale_arcsec
                    x_idx = int((mp.x() - start_x) / (-self.pix_scale_arcsec))
                    y_idx = int((mp.y() - start_y) / self.pix_scale_arcsec)
                    
                    if 0 <= x_idx < self.nx and 0 <= y_idx < self.ny:
                        val = self.current_m0_raw[x_idx, y_idx]
                        if not np.isnan(val):
                            target_panel = self.panels[self.active_picker_panel]
                            target_panel['input_thresh'].setText(f"{val:.4f}")
                            target_panel['btn_pick'].setChecked(False)
                            self.active_picker_panel = None
                            self.update_moment_maps()
            return 
            
        if source_plot == self.plot_channel:
            hit = False
            mode = self.combo_panel_mode.currentText()
            
            # Priority: check Spatial Analysis ROIs if in that mode
            if mode == "Spatial Analysis" and self.spatial_rois:
                for item in self.spatial_rois:
                    roi = item["roi"]
                    is_clicked = False
                    if hasattr(roi, 'shape'):
                         is_clicked = roi.shape().contains(roi.mapFromScene(event.scenePos()))
                    elif isinstance(roi, pg.LineSegmentROI):
                         is_clicked = self.line_roi_hit_test(roi, event.scenePos())
                    elif isinstance(roi, pg.ROI):
                         # For Points (small ROI) and others
                         r_pos = roi.pos()
                         r_size = roi.size()
                         # Buffer for points to make them clickable
                         buf = 0.5 * self.pix_scale_arcsec if r_size.x() < self.pix_scale_arcsec else 0
                         min_x = min(r_pos.x(), r_pos.x() + r_size.x()) - buf
                         max_x = max(r_pos.x(), r_pos.x() + r_size.x()) + buf
                         min_y = min(r_pos.y(), r_pos.y() + r_size.y()) - buf
                         max_y = max(r_pos.y(), r_pos.y() + r_size.y()) + buf
                         is_clicked = (min_x <= mp.x() <= max_x) and (min_y <= mp.y() <= max_y)

                    if is_clicked:
                        hit = True
                        self.select_spatial_region(roi)
                        break

            # Then check Spectrum ROIs
            if not hit and self.spectrum_spatial_rois:
                for r_dict in self.spectrum_spatial_rois:
                    roi = r_dict["roi"]
                    is_clicked = False
                    if isinstance(roi, pg.PolyLineROI):
                        path = QPainterPath()
                        pts = [roi.mapToParent(h.pos()) for h in roi.getHandles()]
                        if pts:
                            path.moveTo(pts[0])
                            for p in pts[1:]:
                                path.lineTo(p)
                            path.closeSubpath()
                        is_clicked = path.contains(mp)
                    else:
                        r_pos = roi.pos()
                        r_size = roi.size()
                        min_x = min(r_pos.x(), r_pos.x() + r_size.x())
                        max_x = max(r_pos.x(), r_pos.x() + r_size.x())
                        min_y = min(r_pos.y(), r_pos.y() + r_size.y())
                        max_y = max(r_pos.y(), r_pos.y() + r_size.y())
                        is_clicked = (min_x <= mp.x() <= max_x) and (min_y <= mp.y() <= max_y)
                    
                    if is_clicked:
                        hit = True
                        roi.setPen(pg.mkPen('y', width=3))
                        self.active_spatial_spectrum_roi = roi
                        
                        # Sync UI tools with selected ROI type
                        roi_type = r_dict.get("type", "Ellipse")
                        self.combo_roi.blockSignals(True)
                        self.combo_roi.setCurrentText(roi_type)
                        self.combo_roi.blockSignals(False)
                        
                        if roi_type in ["Ellipse", "Rectangle", "Point (Beam)", "Custom Polygon"]:
                            self.btn_edit_region.show()
                        else:
                            self.btn_edit_region.hide()
                    else:
                        roi.setPen(pg.mkPen(r_dict["color"], width=3 if r_dict["checkbox"].isChecked() else 2))
                
                self.roi_selected = hit
                if not hit:
                    self.active_spatial_spectrum_roi = None
                    self.btn_edit_region.hide()


    # ==================== DATA LOADING ====================
    def load_file(self, file_name):
        try:
            sc = SpectralCube.read(file_name).with_spectral_unit(u.km / u.s, velocity_convention='radio')
            
            raw_bunit = sc.header.get('BUNIT', 'Unknown').strip()
            self.display_unit = raw_bunit
            self.spec_unit = raw_bunit
            self.raw_header = sc.header.copy() 
            self.fits_header_text = sc.header.tostring(sep='\n')
            
            self.rest_freq_hz = sc.header.get('RESTFRQ', sc.header.get('RESTFREQ', None))
            
            try: self.wcs_2d = WCS(self.raw_header).celestial
            except Exception: self.wcs_2d = None
                
            cdelt2 = sc.header.get('CDELT2', None)
            cdelt1 = sc.header.get('CDELT1', None)
            self.pix_scale_arcsec = abs(float(cdelt2)) * 3600.0 if cdelt2 else 1.0 
            
            bmaj = sc.header.get('BMAJ')
            bmin = sc.header.get('BMIN')
            if bmaj and bmin and cdelt1 and cdelt2:
                beam_area = (np.pi * bmaj * bmin) / (4.0 * np.log(2.0))
                pix_area = abs(cdelt1 * cdelt2)
                self.pixels_per_beam = beam_area / pix_area
            else:
                self.pixels_per_beam = 1.0

            raw_cube = sc.filled_data[:].value
            self.v_axis = sc.spectral_axis.value
            
            self.cube_clean = np.transpose(raw_cube, (0, 2, 1))
            self.nx, self.ny = self.cube_clean.shape[1], self.cube_clean.shape[2]

            self.plot_widget.setLabel('left', f'Mean Flux ({self.display_unit})')
            self.view_channel.ui.histogram.axis.setLabel(f"Flux ({self.display_unit})")
            
            peak_flux = np.nanmax(self.cube_clean)
            self.ch_levels = (0, peak_flux if peak_flux > 0 else 1.0)

            self.slider_channel.setRange(0, len(self.v_axis) - 1)
            self.slider_channel.setValue(len(self.v_axis) // 2)
            self.v_line.show()
            
            v_min, v_max = np.nanmin(self.v_axis), np.nanmax(self.v_axis)
            self.region.setRegion([v_min + 0.4*(v_max-v_min), v_min + 0.6*(v_max-v_min)])
            self.region.show()
            self.change_roi(self.combo_roi.currentText())
            self.update_moment_maps()
            
            self.update_wcs_mode(self.parent_window.is_absolute_wcs)
            self.parent_window.update_menu_states()
            
            return True
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load cube:\n{str(e)}")
            return False

    def close_file(self):
        self.cube_clean = None
        self.fits_header_text = ""
        self.raw_header = None
        self.wcs_2d = None
        self.rest_freq_hz = None
        self.view_channel.clear()
        self.spectrum_curve.setData([], [])
        for roi_info in getattr(self, 'spatial_rois', []):
            try:
                self.view_channel.removeItem(roi_info['roi'])
            except:
                pass
        self.spatial_rois = []
        self.spatial_rois_to_delete = []
        for cut_info in getattr(self, 'pv_cuts', []):
            try:
                self.view_channel.removeItem(cut_info['roi'])
            except Exception:
                pass
            text_item = cut_info.get('text_item')
            if text_item is not None:
                try:
                    self.plot_channel.removeItem(text_item)
                except Exception:
                    pass
            direction_item = cut_info.get('direction_item')
            if direction_item is not None:
                try:
                    self.plot_channel.removeItem(direction_item)
                except Exception:
                    pass
        self.pv_cuts = []
        self.pv_cuts_to_delete = []
        self.pv_data = None
        self.pv_offset_axis = None
        self.pv_velocity_axis = None
        if hasattr(self, 'pv_view'):
            self.pv_view.clear()
            self.lbl_hover_pv.setText("")
            self.combo_pv_cuts.blockSignals(True)
            self.combo_pv_cuts.clear()
            self.combo_pv_cuts.addItem("None")
            self.combo_pv_cuts.blockSignals(False)
        if hasattr(self, 'curve_spatial_1'):
            self.curve_spatial_1.setData([], [])
            self.curve_spatial_2.setData([], [])
            self.lbl_spatial_stats.setText("Draw a region to see statistics.")
        self.v_line.hide()
        self.region.hide()
        for p in self.panels:
            p['combo_pv_cut'].blockSignals(True)
            p['combo_pv_cut'].clear()
            p['combo_pv_cut'].addItem("None")
            p['combo_pv_cut'].blockSignals(False)
            p['combo_pv_range'].setCurrentText("Selected Range")
            p['current_data'] = None
            p['pv_offset_axis'] = None
            p['pv_velocity_axis'] = None
            p['view'].clear()
        self.clear_all_hover_labels()
        self.contour_params = {'channel': None, 0: None, 1: None, 2: None}
        for k in self.active_contours:
            for iso in self.active_contours[k]: iso.setParentItem(None)
            self.active_contours[k] = []
        self.clear_catalog_lines()
        self.parent_window.update_menu_states()

    # --- DYNAMIC LINE CATALOG ENGINE ---
    def open_line_catalog(self):
        if self.cube_clean is None or self.rest_freq_hz is None:
            QMessageBox.warning(self, "Missing Data", "Please load a cube containing RESTFRQ in its header first.")
            return

        min_v, max_v = self.region.getRegion()
        c_kms = const.c.to(u.km / u.s).value
        f_ref_ghz = self.rest_freq_hz / 1e9
        
        f1 = f_ref_ghz * (1.0 - min_v / c_kms)
        f2 = f_ref_ghz * (1.0 - max_v / c_kms)
        
        v_pad = 50.0 
        f_pad = (v_pad / c_kms) * f_ref_ghz
        
        obs_fmin = min(f1, f2) - f_pad
        obs_fmax = max(f1, f2) + f_pad

        dlg = LineCatalogDialog(self, obs_fmin, obs_fmax)
        if dlg.exec_():
            if dlg.action == 'clear':
                self.clear_catalog_lines()
            elif dlg.action == 'query':
                self.clear_catalog_lines()
                self.parent_window.statusBar().showMessage("Querying Splatalogue database... please wait.")
                
                v_sys = dlg.spin_vsys.value()
                e_max = dlg.spin_eup.value()
                species = dlg.edit_species.text()
                
                db_choice = dlg.combo_db.currentText()
                if db_choice == "CDMS": catalogs = ['CDMS']
                elif db_choice == "JPL": catalogs = ['JPL']
                else: catalogs = ['CDMS', 'JPL']

                rest_fmin = obs_fmin / (1.0 - (v_sys / c_kms))
                rest_fmax = obs_fmax / (1.0 - (v_sys / c_kms))

                self.worker = SplatalogueWorker(rest_fmin, rest_fmax, catalogs, v_sys, e_max, species)
                self.worker.finished.connect(lambda t: self.process_splatalogue_results(t, v_sys))
                self.worker.error.connect(self.show_splatalogue_error)
                self.worker.start()

    def process_splatalogue_results(self, parsed_data, v_sys):
        self.parent_window.statusBar().showMessage("Ready.")
        if parsed_data is None or len(parsed_data) == 0:
            QMessageBox.information(self, "No Results", "No molecular lines found matching those parameters.")
            return

        dlg = LineSelectionDialog(self, parsed_data, v_sys)
        if dlg.exec_():
            if dlg.selected_rows:
                self.draw_selected_lines(dlg.selected_rows, v_sys)

    def draw_selected_lines(self, selected_rows, v_sys):
        c_kms = const.c.to(u.km / u.s).value
        ref_freq_ghz = self.rest_freq_hz / 1e9

        drawn_count = 0
        for row in selected_rows:
            f_cat_ghz = row.get('restfreq', 0.0)
            if np.isnan(f_cat_ghz) or f_cat_ghz == 0.0: continue
            
            label_text = str(row.get('formula', row.get('molecule_name', 'Unknown')))
            
            v_offset = c_kms * (ref_freq_ghz - f_cat_ghz) / ref_freq_ghz
            v_plot = v_offset + v_sys
            
            line = pg.InfiniteLine(angle=90, movable=False, pos=v_plot, pen=pg.mkPen('#e74c3c', width=1.5, style=Qt.DashLine))
            label = pg.TextItem(text=label_text, color='#e74c3c', anchor=(0, 1), angle=-90)
            label.setPos(v_plot, np.nanmax(self.spectrum_curve.yData) if self.spectrum_curve.yData is not None else 1.0)
            
            self.plot_widget.addItem(line)
            self.plot_widget.addItem(label)
            self.catalog_overlay_items.extend([line, label])
            drawn_count += 1

        self.parent_window.statusBar().showMessage(f"Overlaid {drawn_count} selected molecular lines.")

    def show_splatalogue_error(self, err_msg):
        self.parent_window.statusBar().showMessage("Query failed.")
        QMessageBox.critical(self, "Splatalogue Error", err_msg)

    def clear_catalog_lines(self):
        for item in self.catalog_overlay_items:
            self.plot_widget.removeItem(item)
        self.catalog_overlay_items = []

    # ==================== MAP & ROI FUNCTIONS ====================
    def draw_contours(self, target_id, view, data):
        for iso in self.active_contours.get(target_id, []):
            iso.setParentItem(None)
            if iso.scene() is not None:
                view.getView().removeItem(iso)
        self.active_contours[target_id] = []

        params = self.contour_params.get(target_id)
        if not params or data is None or np.isnan(data).all(): return

        if params['mode'] == 'auto':
            valid = data[~np.isnan(data) & ~np.isinf(data)]
            if len(valid) == 0: return
            min_v, max_v = np.nanmin(valid), np.nanmax(valid)
            levels = np.linspace(min_v, max_v, params['n'] + 2)[1:-1]
        else:
            levels = params['levels']

        for lvl in levels:
            iso = pg.IsocurveItem(data=data, level=lvl, pen=pg.mkPen('#2ecc71', width=1.5))
            iso.setParentItem(view.getImageItem())
            iso.setZValue(10)
            self.active_contours[target_id].append(iso)

    def update_channel_map(self):
        if self.cube_clean is None: return
        idx = self.slider_channel.value()
        self.input_channel_vel.setText(f"{self.v_axis[idx]:.2f}")
        self.v_line.setPos(self.v_axis[idx])
        
        pos_tup = ((self.nx / 2) * self.pix_scale_arcsec, -(self.ny / 2) * self.pix_scale_arcsec)
        scale_tup = (-self.pix_scale_arcsec, self.pix_scale_arcsec)
        
        slice_data = self.cube_clean[idx]
        self.view_channel.setImage(slice_data, autoLevels=False, levels=getattr(self, 'ch_levels', (0, 1)), autoHistogramRange=True, scale=scale_tup, pos=pos_tup)
        self.draw_contours('channel', self.view_channel, slice_data)
        self.update_spatial_analysis()
        
        grad = self.view_channel.ui.histogram.gradient
        ticks = list(grad.ticks.keys())
        if len(ticks) > 3:
            sorted_ticks = sorted(ticks, key=lambda t: grad.ticks[t])
            keep = [sorted_ticks[0], sorted_ticks[len(sorted_ticks)//2], sorted_ticks[-1]]
            for t in sorted_ticks:
                if t not in keep:
                    t.hide()

    def set_channel_from_text(self):
        if self.cube_clean is None: return
        try:
            target_v = float(self.input_channel_vel.text())
            idx = (np.abs(self.v_axis - target_v)).argmin()
            self.slider_channel.setValue(idx)
        except ValueError: self.update_channel_map()

    def get_current_channel_data(self): return None if self.cube_clean is None else self.cube_clean[self.slider_channel.value()]
    
    def start_playback(self, d): 
        self.play_direction = d
        self.playback_timer.start(150)
        
    def stop_playback(self): 
        self.playback_timer.stop()
        
    def step_channel(self, d=None):
        if self.cube_clean is None: return
        shift = d if type(d) is int else self.play_direction
        nv = self.slider_channel.value() + shift
        if 0 <= nv < len(self.v_axis): 
            self.slider_channel.setValue(nv)
        else: 
            self.stop_playback()

    def change_roi(self, roi_type):
        if self.cube_clean is None: return
        
        if roi_type == "Whole Map":
            for r_dict in getattr(self, 'spectrum_spatial_rois', []):
                r_dict["checkbox"].blockSignals(True)
                r_dict["checkbox"].setChecked(False)
                r_dict["checkbox"].blockSignals(False)
            self.update_spectrum()
            return
            
        num_rois = len(getattr(self, 'spectrum_spatial_rois', []))
        sz = self.nx * self.pix_scale_arcsec * 0.2
        offset = num_rois * sz * 0.15
        cx, cy = offset, offset
        
        if hasattr(self, 'btn_edit_region'):
            if roi_type in ["Ellipse", "Rectangle", "Point (Beam)", "Custom Polygon"]:
                self.btn_edit_region.show()
            else:
                self.btn_edit_region.hide()
                
        new_roi = None
        if roi_type == "Point (Beam)": new_roi = pg.CircleROI([cx, cy], [self.pix_scale_arcsec*3, self.pix_scale_arcsec*3], pen='#f1c40f')
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
            self.is_drawing_polygon = True
            self.polygon_points = []
            if self.polygon_preview_line is not None:
                self.plot_channel.vb.removeItem(self.polygon_preview_line)
            self.polygon_preview_line = pg.PlotDataItem([], [], pen=pg.mkPen('y', width=2, style=Qt.DashLine))
            self.plot_channel.vb.addItem(self.polygon_preview_line)
            # Notify the user via status bar if available, or just start
            return # Don't add ROI yet
        
        if new_roi is not None:
            self._finish_roi_addition(new_roi, roi_type)

    def finalize_polygon(self):
        if len(self.polygon_points) < 3:
            self.cancel_polygon()
            return
            
        pts = self.polygon_points
        self.cancel_polygon()
        
        new_roi = pg.PolyLineROI(pts, closed=True, pen='#f1c40f')
        def custom_shape(roi=new_roi):
            p = QPainterPath()
            if not roi.handles: return p
            p.moveTo(roi.handles[0]['item'].pos())
            for h in roi.handles[1:]:
                p.lineTo(h['item'].pos())
            p.closeSubpath()
            return p
        new_roi.shape = custom_shape
        
        self._finish_roi_addition(new_roi, "Custom Polygon")
        self.update_spectrum()

    def cancel_polygon(self):
        self.is_drawing_polygon = False
        self.polygon_points = []
        if self.polygon_preview_line:
            self.plot_channel.vb.removeItem(self.polygon_preview_line)
            self.polygon_preview_line = None

    def _finish_roi_addition(self, new_roi, roi_type):
        col = self.region_colors[len(self.spectrum_spatial_rois) % len(self.region_colors)]
        new_roi.setPen(pg.mkPen(col, width=3))
        self.view_channel.addItem(new_roi)
        new_roi.sigRegionChanged.connect(self.update_spectrum)
        
        # Uncheck existing
        for r_dict in self.spectrum_spatial_rois:
            r_dict["checkbox"].blockSignals(True)
            r_dict["checkbox"].setChecked(False)
            r_dict["checkbox"].blockSignals(False)
            r_dict["roi"].setPen(pg.mkPen(r_dict["color"], width=2))
            
        name = f"Region {len(self.spectrum_spatial_rois) + 1}"
        cb = QCheckBox(name)
        cb.setChecked(True)
        cb.setStyleSheet(f"color: {col}; font-weight: bold;")
        cb.toggled.connect(self.update_spectrum)
        self.box_regions_layout.addWidget(cb)
        
        self.spectrum_spatial_rois.append({
            "name": name,
            "roi": new_roi,
            "checkbox": cb,
            "color": col,
            "type": roi_type
        })
        self.roi_selected = True
        self.active_spatial_spectrum_roi = new_roi
        
        if len(self.spectrum_spatial_rois) > 1:
            self.box_regions.show()
        self.refresh_spectral_stats_apertures()
        self.update_spectrum()
        
    def remove_spatial_spectrum_roi(self, roi):
        for i, r_dict in enumerate(self.spectrum_spatial_rois):
            if r_dict["roi"] == roi:
                # Remove from view
                if roi.scene():
                    roi.scene().removeItem(roi)
                else:
                    self.view_channel.removeItem(roi)
                # Remove checkbox
                cb = r_dict["checkbox"]
                self.box_regions_layout.removeWidget(cb)
                cb.deleteLater()
                # Remove from dicts and plot items
                if r_dict["name"] in self.spectrum_curves:
                    c = self.spectrum_curves.pop(r_dict["name"])
                    if c.scene(): c.scene().removeItem(c)
                    else: self.plot_widget.removeItem(c)
                
                if hasattr(self, 'spectrum_curves_smooth') and r_dict["name"] in self.spectrum_curves_smooth:
                    c = self.spectrum_curves_smooth.pop(r_dict["name"])
                    if c.scene(): c.scene().removeItem(c)
                    else: getattr(self, 'plot_widget_smooth', self.plot_widget).removeItem(c)
                
                self.spectrum_spatial_rois.pop(i)
                break
                
        # Rename remaining to be contiguous
        for i, r_dict in enumerate(self.spectrum_spatial_rois):
            new_name = f"Region {i + 1}"
            old_name = r_dict["name"]
            r_dict["name"] = new_name
            r_dict["checkbox"].setText(new_name)
            
            if old_name in self.spectrum_curves:
                self.spectrum_curves[new_name] = self.spectrum_curves.pop(old_name)
            if hasattr(self, 'spectrum_curves_smooth') and old_name in self.spectrum_curves_smooth:
                self.spectrum_curves_smooth[new_name] = self.spectrum_curves_smooth.pop(old_name)
                
        if len(self.spectrum_spatial_rois) <= 1:
            self.box_regions.hide()

        self.refresh_spectral_stats_apertures()
        self.update_spectrum()

    def clear_roi(self):
        self.combo_roi.blockSignals(True)
        self.combo_roi.setCurrentText("Whole Map")
        self.combo_roi.blockSignals(False)
        for r_dict in list(self.spectrum_spatial_rois):
            self.remove_spatial_spectrum_roi(r_dict["roi"])
            
        if hasattr(self, 'spatial_rois_to_delete'):
            self.spatial_rois_to_delete = [item["roi"] for item in getattr(self, 'spatial_rois', [])]
            self.delete_selected_spatial_regions()

    def open_edit_region_dialog(self):
        roi = getattr(self, 'active_spatial_spectrum_roi', None)
        roi_dict = None
        
        if roi:
            roi_dict = next((r for r in getattr(self, 'spectrum_spatial_rois', []) if r["roi"] == roi), None)
        else:
            if getattr(self, 'spatial_rois_to_delete', []):
                roi = self.spatial_rois_to_delete[-1]
                roi_dict = next((r for r in getattr(self, 'spatial_rois', []) if r["roi"] == roi), None)
                
        if not roi: return
        
        # Mutual exclusion: warn if Smooth dialog is currently open
        if getattr(self, '_smooth_dialog_active', False):
            msg = QMessageBox(self.window())
            msg.setWindowTitle("Dialog Open")
            msg.setText("Please close the 'Smooth' dialog before opening Edit Region.")
            msg.setIcon(QMessageBox.Information)
            msg.setWindowFlags(msg.windowFlags() | Qt.WindowStaysOnTopHint)
            msg.exec_()
            return
        from src.gui.dialogs import RegionPropertiesDialog
        # If an edit dialog is already open for this ROI, just raise it
        if getattr(self, '_region_dialog', None) and self._region_dialog.isVisible():
            self._region_dialog.raise_()
            self._region_dialog.activateWindow()
            return
        dlg = RegionPropertiesDialog(roi, self, parent=self.window(), roi_dict=roi_dict)
        self._region_dialog = dlg
        dlg.show()

    def update_spectrum(self):
        if self.cube_clean is None: return
        stat = self.combo_spec_stat.currentText()
        
        active_rois = [r_dict for r_dict in self.spectrum_spatial_rois if r_dict["checkbox"].isChecked()]
        if not active_rois:
            rois_to_plot = [{"name": "Whole Map", "roi": None, "color": "w"}]
        else:
            rois_to_plot = active_rois
            
        active_names = [r["name"] for r in rois_to_plot]
        for name in list(self.spectrum_curves.keys()):
            if name not in active_names:
                c = self.spectrum_curves.pop(name)
                if c.scene(): c.scene().removeItem(c)
                else: self.plot_widget.removeItem(c)
                
                if hasattr(self, 'spectrum_curves_smooth') and name in self.spectrum_curves_smooth:
                    c_s = self.spectrum_curves_smooth.pop(name)
                    if c_s.scene(): c_s.scene().removeItem(c_s)
                    else: getattr(self, 'plot_widget_smooth', self.plot_widget).removeItem(c_s)
                    
        if "Whole Map" not in active_names:
            self.spectrum_curve.setData([], [])
            if hasattr(self, 'spectrum_curve_smooth'):
                self.spectrum_curve_smooth.setData([], [])
                
        ymax_global = -np.inf
        
        with np.errstate(invalid='ignore', divide='ignore'):
            for r_dict in rois_to_plot:
                roi = r_dict["roi"]
                name = r_dict["name"]
                color = r_dict["color"]
                
                if roi is None:
                    sub_data = self.cube_clean
                else:
                    sub_data = roi.getArrayRegion(self.cube_clean, self.view_channel.getImageItem(), axes=(1, 2))
                    
                if "Max" in stat:
                    spec = np.nanmax(sub_data, axis=(1, 2))
                    y_label = f"Max Flux ({self.display_unit})"
                    self.spec_unit = self.display_unit
                elif "Sum" in stat:
                    spec = np.nansum(sub_data, axis=(1, 2))
                    if self.pixels_per_beam > 1.0:
                        spec /= self.pixels_per_beam
                        self.spec_unit = self.display_unit.replace('/beam', '')
                    else:
                        self.spec_unit = f"{self.display_unit} * pix"
                    y_label = f"Sum Flux ({self.spec_unit})"
                else:
                    spec = np.nanmean(sub_data, axis=(1, 2))
                    y_label = f"Mean Flux ({self.display_unit})"
                    self.spec_unit = self.display_unit
                    
                self.plot_widget.setLabel('left', y_label)
                if hasattr(self, 'plot_widget_smooth'):
                    self.plot_widget_smooth.setLabel('left', y_label)
                
                sort_idx = np.argsort(self.v_axis)
                vs, ss = self.v_axis[sort_idx], spec[sort_idx]
                ve = np.zeros(len(vs) + 1)
                dv = np.diff(vs)
                if len(dv) > 0:
                    ve[:-1] = vs - np.append(dv, dv[-1])/2
                    ve[-1] = vs[-1] + dv[-1]/2
                else: ve = np.array([vs[0]-1, vs[0]+1])
                
                if ss is not None and len(ss) > 0:
                    ymax_global = max(ymax_global, np.nanmax(ss))
                
                if name == "Whole Map":
                    self.spectrum_curve.setData(x=ve, y=ss)
                else:
                    if name not in self.spectrum_curves:
                        c = pg.PlotDataItem([], [], stepMode="center", pen=pg.mkPen(color, width=2), name=name)
                        self.spectrum_curves[name] = c
                        self.plot_widget.addItem(c)
                    self.spectrum_curves[name].setData(x=ve, y=ss)
                    
                if getattr(self, 'smoothing_params', None) is not None and getattr(self, 'spectrum_tabs', None) is not None:
                    if self.spectrum_tabs.indexOf(self.plot_widget_smooth) != -1:
                        method = self.smoothing_params['method']
                        ss_smooth = ss.copy()
                        try:
                            if method == 'boxcar':
                                from scipy.ndimage import uniform_filter1d
                                w = self.smoothing_params['window']
                                ss_smooth = uniform_filter1d(ss_smooth, size=w)
                            elif method == 'gaussian':
                                from scipy.ndimage import gaussian_filter1d
                                sigma = self.smoothing_params['sigma']
                                ss_smooth = gaussian_filter1d(ss_smooth, sigma=sigma)
                            elif method == 'savgol':
                                from scipy.signal import savgol_filter
                                w = self.smoothing_params['window']
                                p = self.smoothing_params['polyorder']
                                if len(ss_smooth) > w:
                                    ss_smooth = savgol_filter(ss_smooth, window_length=w, polyorder=p)
                        except Exception:
                            pass
                            
                        if name == "Whole Map":
                            self.spectrum_curve_smooth.setData(x=ve, y=ss_smooth)
                        else:
                            if name not in self.spectrum_curves_smooth:
                                c_s = pg.PlotDataItem([], [], stepMode="center", pen=pg.mkPen(color, width=2), name=name)
                                self.spectrum_curves_smooth[name] = c_s
                                self.plot_widget_smooth.addItem(c_s)
                            self.spectrum_curves_smooth[name].setData(x=ve, y=ss_smooth)

        # Update legends
        if getattr(self.plot_widget, 'plotItem', None) is not None and self.plot_widget.plotItem.legend is not None:
            self.plot_widget.plotItem.legend.clear()
            if "Whole Map" in active_names:
                self.plot_widget.plotItem.legend.addItem(self.spectrum_curve, "Whole Map")
            for n, c in self.spectrum_curves.items():
                self.plot_widget.plotItem.legend.addItem(c, n)
                
        if getattr(self, 'plot_widget_smooth', None) is not None and self.plot_widget_smooth.plotItem.legend is not None:
            self.plot_widget_smooth.plotItem.legend.clear()
            if "Whole Map" in active_names:
                self.plot_widget_smooth.plotItem.legend.addItem(self.spectrum_curve_smooth, "Whole Map")
            for n, c_s in getattr(self, 'spectrum_curves_smooth', {}).items():
                self.plot_widget_smooth.plotItem.legend.addItem(c_s, n)

        if self.catalog_overlay_items:
            ymax = ymax_global if ymax_global != -np.inf else 1.0
            for item in self.catalog_overlay_items:
                if isinstance(item, pg.TextItem):
                    item.setPos(item.pos().x(), ymax)

    def update_text_from_region(self):
        if self.cube_clean is None: return
        minX, maxX = self.region.getRegion()
        self.input_vmin.setText(f"{minX:.2f}"); self.input_vmax.setText(f"{maxX:.2f}")

    def update_region_from_text(self):
        if self.cube_clean is None: return
        try:
            minX, maxX = float(self.input_vmin.text()), float(self.input_vmax.text())
            if minX < maxX: self.region.setRegion([minX, maxX])
        except ValueError: pass 

    def apply_cmap(self, view, is_velocity):
        if is_velocity:
            pos = np.array([0.0, 0.5, 1.0])
            colors = np.array([[0, 0, 255, 255], [255, 255, 255, 255], [255, 0, 0, 255]], dtype=np.ubyte)
            view.setColorMap(pg.ColorMap(pos, colors))
        else: 
            view.ui.histogram.gradient.loadPreset(self.parent_window.current_cmap)
            
        grad = view.ui.histogram.gradient
        ticks = list(grad.ticks.keys())
        if len(ticks) > 3:
            sorted_ticks = sorted(ticks, key=lambda t: grad.ticks[t])
            keep = [sorted_ticks[0], sorted_ticks[len(sorted_ticks)//2], sorted_ticks[-1]]
            for t in sorted_ticks:
                if t not in keep:
                    t.hide()

    def _on_region_drag_start(self):
        """Called on every sigRegionChanged — marks that a drag is in progress."""
        self._region_dragging = True

    def _on_region_drag_end(self):
        """Called on sigRegionChangeFinished — clears the drag flag and recomputes."""
        self._region_dragging = False
        self.update_moment_maps()


    def update_moment_maps(self):
        """Entry point (runs on the Qt main thread).

        Gathers all Qt-side inputs, performs cheap UI configuration, then
        dispatches the heavy NumPy work to MomentWorker running in a
        background thread.  Results are delivered via _on_moment_result.
        """
        if self.cube_clean is None:
            return
        # Skip while the velocity region handle is being actively dragged;
        # _on_region_drag_end fires update_moment_maps once on release.
        if self._region_dragging:
            return

        selected_cube, sub_v, minX, maxX = self.get_velocity_subset(use_full_range=False)
        if selected_cube is None or sub_v is None:
            return

        # ---- Cancel any in-flight worker --------------------------------
        if self._moment_worker is not None and self._moment_worker.isRunning():
            self._moment_worker.cancel()
        self._moment_generation += 1
        current_gen = self._moment_generation

        # ---- Read thresholds from widgets (Qt, main thread) -------------
        thresh = []
        for i in range(3):
            try:
                thresh.append(float(self.panels[i]['input_thresh'].text()))
            except ValueError:
                thresh.append(0.0)

        # ---- Cheap UI configuration (axes, colormaps) -------------------
        for i, p in enumerate(self.panels):
            mtype = p['combo'].currentText()
            self.configure_bottom_panel_controls(p, mtype)
            if mtype != 'PV Diagram':
                self.configure_bottom_panel_axes(p, is_pv=False)
                p['plot_item'].setTitle('')
                is_vel = ('Moment 1' in mtype) or ('Moment 9' in mtype)
                self.apply_cmap(p['view'], is_vel)

        # ---- Extract ROI world-coordinates for PV panels ----------------
        # (get_line_roi_points uses Qt, must stay on main thread)
        panel_configs = []
        for i, p in enumerate(self.panels):
            mtype = p['combo'].currentText()
            cfg = {'mtype': mtype, 'threshold': thresh[i]}

            if mtype == 'PV Diagram':
                cut_name  = p['combo_pv_cut'].currentText()
                active_item = self.get_pv_cut_by_name(cut_name)
                if active_item is not None:
                    points = self.get_line_roi_points(active_item['roi'])
                    use_full = p['combo_pv_range'].currentText() == 'Full Cube'
                    cfg['pv_points'] = points           # (p1, p2) numpy arrays or None
                    cfg['pv_cube']   = self.cube_clean if use_full else selected_cube
                    cfg['pv_sub_v']  = self.v_axis     if use_full else sub_v
                else:
                    cfg['pv_points'] = None

            panel_configs.append(cfg)

        # ---- Dispatch to background worker ------------------------------
        worker_params = {
            'selected_cube':  selected_cube,
            'sub_v':          sub_v,
            'minX':           minX,
            'maxX':           maxX,
            'nx':             self.nx,
            'ny':             self.ny,
            'pix_scale_arcsec': self.pix_scale_arcsec,
            'display_unit':   self.display_unit,
            'panel_configs':  panel_configs,
        }
        self._moment_worker = MomentWorker(worker_params, current_gen)
        self._moment_worker.result_ready.connect(self._on_moment_result)
        self._moment_worker.start()

    def _on_moment_result(self, results: dict):
        """Receives computed moment/PV data from MomentWorker and updates the UI.
        Runs on the Qt main thread via the signal/slot mechanism.
        """
        # Discard results from a superseded (cancelled) worker.
        if results['generation'] != self._moment_generation:
            return
        if self.cube_clean is None:
            return

        self.current_m0_raw = results['m0_raw']
        minX = results['minX']
        maxX = results['maxX']

        pos_tup   = ((self.nx / 2) * self.pix_scale_arcsec, -(self.ny / 2) * self.pix_scale_arcsec)
        scale_tup = (-self.pix_scale_arcsec, self.pix_scale_arcsec)

        for p, pr in zip(self.panels, results['panel_results']):
            mtype = pr['mtype']
            panel_id = self.panels.index(p)

            if mtype == 'PV Diagram':
                if pr.get('data') is None:
                    self.clear_panel_pv_diagram(p)
                else:
                    pv_sorted = pr['data']
                    offsets   = pr['offsets']
                    v_sorted  = pr['v_sorted']
                    levels    = pr['levels']
                    dx, dv    = pr['dx'], pr['dv']

                    self.configure_bottom_panel_axes(p, is_pv=True)
                    p['view'].ui.histogram.gradient.loadPreset('turbo')
                    p['view'].ui.histogram.axis.setLabel(f"Flux ({self.display_unit})")
                    p['plot_item'].setTitle('PV Diagram')

                    p['current_data']       = pv_sorted
                    p['pv_offset_axis']     = offsets
                    p['pv_velocity_axis']   = v_sorted
                    p['unit']               = self.display_unit

                    p['view'].setImage(
                        pv_sorted,
                        autoLevels=False,
                        autoHistogramRange=False,
                        levels=levels,
                        scale=(dx, dv),
                        pos=(0.0, v_sorted[0]),
                    )
                    self.draw_contours(panel_id, p['view'], None)

            else:
                data     = pr.get('data')
                levels   = pr.get('levels', (0.0, 1.0))
                unit_str = pr.get('unit_str', '')

                # Histogram axis label
                view = p['view']
                if 'Moment 0' in mtype:
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
                    self.draw_contours(panel_id, view, data)

    def clear_all_hover_labels(self):
        for lbl in [self.lbl_hover_ch, self.lbl_hover_spec, self.lbl_hover_pv] + [p['lbl_hover'] for p in self.panels]:
            lbl.setText("")

    def hover_event(self, pos, plot_item, data_array, active_label, panel_id='channel'):
        self.clear_all_hover_labels()
        if data_array is None or self.cube_clean is None: return

        if getattr(self, 'is_drawing_polygon', False) and panel_id == 'channel':
            if self.polygon_points and self.polygon_preview_line:
                mp = plot_item.vb.mapSceneToView(pos)
                pts = list(self.polygon_points)
                pts.append([mp.x(), mp.y()])
                arr = np.array(pts)
                self.polygon_preview_line.setData(arr[:,0], arr[:,1], symbol='o', symbolSize=5)
        if plot_item.sceneBoundingRect().contains(pos):
            mp = plot_item.vb.mapSceneToView(pos)
            
            start_x = (self.nx / 2) * self.pix_scale_arcsec
            start_y = -(self.ny / 2) * self.pix_scale_arcsec
            x_idx = int((mp.x() - start_x) / (-self.pix_scale_arcsec))
            y_idx = int((mp.y() - start_y) / self.pix_scale_arcsec)
            
            if 0 <= x_idx < self.nx and 0 <= y_idx < self.ny:
                val = data_array[x_idx, y_idx]
                unit_str = self.display_unit if panel_id == 'channel' else self.panels[panel_id].get('unit', '')
                
                # Format the flux value
                val_str = f"{val:.3e}" if (not np.isnan(val) and abs(val) < 1e-3 and abs(val)>0) else f"{val:.4g}" if not np.isnan(val) else "NaN"

                # Calculate Absolute RA and Dec
                if self.wcs_2d is not None:
                    coord = self.wcs_2d.pixel_to_world(x_idx, y_idx)
                    ra_str = coord.ra.to_string(unit=u.hourangle, sep=':', precision=2, pad=True)
                    dec_str = coord.dec.to_string(unit=u.degree, sep=':', precision=1, alwayssign=True, pad=True)
                    coord_text = f"RA: {ra_str}, Dec: {dec_str}"
                else:
                    # Fallback to pixels if WCS fails to load
                    coord_text = f"Pix: ({x_idx}, {y_idx})"
                
                # Set the final label text (Pixels + Absolute RA/Dec + Value)
                active_label.setText(f"Pix: ({x_idx}, {y_idx}) | {coord_text} | {val_str} {unit_str}")
                active_label.setStyleSheet("color: #3498db; font-weight: bold; font-size: 9.5px;")
                return
        active_label.setText("")

    def hover_spectrum(self, pos, widget=None):
        self.clear_all_hover_labels()
        if self.cube_clean is None or self.v_axis is None: return
        
        # If widget is not provided, try to find which one contains the mouse
        if widget is None:
            if self.plot_widget.sceneBoundingRect().contains(pos):
                widget = self.plot_widget
            elif self.plot_widget_smooth.sceneBoundingRect().contains(pos):
                widget = self.plot_widget_smooth
            else:
                return
        elif not widget.sceneBoundingRect().contains(pos):
            return

        is_smooth = (widget == self.plot_widget_smooth)
        mp = widget.plotItem.vb.mapSceneToView(pos)
        
        # Find closest velocity index
        idx = (np.abs(self.v_axis - mp.x())).argmin()
        if idx >= len(self.v_axis): return
        
        v_val = self.v_axis[idx]
        
        # Collect values from all active curves in this plot
        all_vals = [] # List of (sort_key, display_name, value_str)
        
        # Check Whole Map curve
        main_curve = self.spectrum_curve_smooth if is_smooth else self.spectrum_curve
        if main_curve.yData is not None and len(main_curve.yData) > idx:
            if len(main_curve.yData) > 0:
                y = main_curve.yData[idx]
                if np.isfinite(y):
                    val_str = f'{y:.3e}' if (abs(y) < 1e-3 and abs(y) > 0) else f'{y:.4g}'
                    # Sort key -1 to ensure it's first
                    all_vals.append((-1, "", val_str))
        
        # Check region curves
        region_curves = getattr(self, 'spectrum_curves_smooth' if is_smooth else 'spectrum_curves', {})
        for name, c in region_curves.items():
            if c.yData is not None and len(c.yData) > idx:
                y = c.yData[idx]
                if np.isfinite(y):
                    val_str = f'{y:.3e}' if (abs(y) < 1e-3 and abs(y) > 0) else f'{y:.4g}'
                    
                    if name.startswith("Region"):
                        try:
                            # Extract number, handling "Region 1" or "Region1"
                            num_str = name.replace("Region", "").strip()
                            num = int(num_str)
                            display = f"R{num}"
                            sort_key = (1, num) # Group 1 for Regions
                        except:
                            display = name[:2].upper()
                            sort_key = (0, display) # Group 0 for Custom
                    else:
                        display = name[:2].upper()
                        sort_key = (0, display)
                        
                    all_vals.append((sort_key, display, val_str))
        
        # Sort values based on the key
        all_vals.sort(key=lambda x: x[0])
        
        if all_vals:
            # Format the strings
            display_parts = []
            for _, disp, v_str in all_vals:
                if disp == "":
                    display_parts.append(v_str)
                else:
                    display_parts.append(f"{disp}: {v_str}")
            
            unit = getattr(self, 'spec_unit', '')
            self.lbl_hover_spec.setText(f"Ch: {idx} | {v_val:.2f} km/s | " + " | ".join(display_parts) + f" {unit}")
            self.lbl_hover_spec.setStyleSheet("color: #3498db; font-weight: bold; font-size: 9.5px;")



    def hover_pv(self, pos):
        self.clear_all_hover_labels()
        if self.pv_data is None or self.pv_offset_axis is None or self.pv_velocity_axis is None:
            return
        if not self.pv_plot_item.sceneBoundingRect().contains(pos):
            return

        mp = self.pv_plot_item.vb.mapSceneToView(pos)
        x_idx = int(np.abs(self.pv_offset_axis - mp.x()).argmin())
        y_idx = int(np.abs(self.pv_velocity_axis - mp.y()).argmin())
        if 0 <= x_idx < self.pv_data.shape[0] and 0 <= y_idx < self.pv_data.shape[1]:
            val = self.pv_data[x_idx, y_idx]
            val_str = f"{val:.3e}" if (np.isfinite(val) and abs(val) < 1e-3 and abs(val) > 0) else f"{val:.4g}" if np.isfinite(val) else "NaN"
            self.lbl_hover_pv.setText(
                f"Offset: {self.pv_offset_axis[x_idx]:.2f} arcsec | Vel: {self.pv_velocity_axis[y_idx]:.2f} km/s | {val_str} {self.display_unit}"
            )
            self.lbl_hover_pv.setStyleSheet("color: #3498db; font-weight: bold; font-size: 9.5px;")

    def update_region_ui_visibility(self):
        active_rois = self.get_active_spectrum_rois()
        has_boxes = len(active_rois) >= 1
        self.btn_spectral_stats.setVisible(has_boxes)
        # Also refresh popup if open
        if self._spectral_stats_popup and self._spectral_stats_popup.isVisible():
            self.refresh_spectral_stats_popup()

    def rename_regions(self):
        self.combo_regions.blockSignals(True)
        self.combo_regions_2.blockSignals(True)
        self.combo_regions_3.blockSignals(True)
        
        self.combo_regions.clear()
        self.combo_regions_2.clear()
        self.combo_regions_3.clear()
        
        self.combo_regions.addItem("None")
        self.combo_regions_2.addItem("None")
        self.combo_regions_3.addItem("None")
        
        active_rois = self.get_active_spectrum_rois()
        for i, item in enumerate(active_rois):
            new_name = f"Box {i + 1}"
            item["name"] = new_name
            item["text_item"].setText(new_name)
            
            self.combo_regions.addItem(new_name)
            self.combo_regions_2.addItem(new_name)
            self.combo_regions_3.addItem(new_name)
            
            item["roi"].setPen(pg.mkPen('c', width=2))
            
        self.combo_regions.blockSignals(False)
        self.combo_regions_2.blockSignals(False)
        self.combo_regions_3.blockSignals(False)
        
        self.on_region_selected()
        # Refresh popup box list
        if self._spectral_stats_popup and self._spectral_stats_popup.isVisible():
            self.refresh_spectral_stats_popup()

    def delete_region(self, roi):
        if roi.scene():
            roi.scene().removeItem(roi)
        else:
            self.get_active_spectrum_plot().removeItem(roi)
            
        active_rois = self.get_active_spectrum_rois()
        for i, item in enumerate(active_rois):
            if item["roi"] == roi:
                ti = item["text_item"]
                if ti.scene():
                    ti.scene().removeItem(ti)
                else:
                    self.get_active_spectrum_plot().removeItem(ti)
                active_rois.pop(i)
                break

    def delete_selected_regions(self):
        for roi in list(self.rois_to_delete):
            self.delete_region(roi)
        self.rois_to_delete.clear()
        self.update_region_ui_visibility()
        self.rename_regions()

    def clear_spectrum_regions(self):
        active_rois = self.get_active_spectrum_rois()
        for item in list(active_rois):
            self.delete_region(item["roi"])
        self.rois_to_delete.clear()
        self.update_region_ui_visibility()
        self.rename_regions()

    def select_region_for_deletion(self, roi):
        if roi in self.rois_to_delete:
            self.rois_to_delete.remove(roi)
        else:
            self.rois_to_delete.append(roi)
        self.on_region_selected()

    def add_spectrum_region(self, roi):
        active_rois = self.get_active_spectrum_rois()
        region_name = f"Box {len(active_rois) + 1}"
        roi_info = {"name": region_name, "roi": roi}
        active_rois.append(roi_info)
        
        self.update_region_ui_visibility()
        
        text_item = pg.TextItem(text=region_name, color=(200, 200, 200, 150), anchor=(1, 1))
        self.get_active_spectrum_plot().addItem(text_item)
        
        def update_text_pos(r=roi, t=text_item):
            try:
                pos = r.pos()
                size = r.size()
                max_x = max(pos.x(), pos.x() + size.x())
                max_y = max(pos.y(), pos.y() + size.y())
                t.setPos(max_x, max_y)
            except Exception:
                pass
            
        roi.sigRegionChanged.connect(update_text_pos)
        update_text_pos()
        
        roi_info["text_item"] = text_item
        roi_info["update_text_pos"] = update_text_pos
        
        roi.sigRegionChanged.connect(self.update_spectrum_region_calc)
        
        self.rename_regions()
        self.combo_regions.blockSignals(True)
        self.combo_regions.setCurrentText(region_name)
        self.combo_regions.blockSignals(False)
        self.on_region_selected()

    def open_spectral_stats_popup(self):
        if self._spectral_stats_popup is None or not self._spectral_stats_popup.isVisible():

            main_win = self.window()

            popup = QDialog(main_win)
            popup.setWindowTitle("Spectral Statistics")
            popup.setMinimumWidth(480)
            popup.setWindowFlags(
                Qt.Window | Qt.WindowCloseButtonHint | Qt.WindowMinimizeButtonHint
            )
            popup.setAttribute(Qt.WA_DeleteOnClose, False)

            layout = QVBoxLayout(popup)
            layout.setSpacing(8)
            layout.setContentsMargins(10, 10, 10, 10)

            _BTN_STYLE = "font-size: 9px; padding: 1px 6px;"

            # ── Velocity Boxes ────────────────────────────────────────────
            popup.grp_boxes = QGroupBox("Velocity Boxes (drawn on spectrum)")
            boxes_outer = QVBoxLayout(popup.grp_boxes)
            boxes_outer.setSpacing(4)
            sel_row_b = QHBoxLayout()
            btn_b_all  = QPushButton("Select All");   btn_b_all.setStyleSheet(_BTN_STYLE);  btn_b_all.setFixedHeight(20)
            btn_b_none = QPushButton("Deselect All"); btn_b_none.setStyleSheet(_BTN_STYLE); btn_b_none.setFixedHeight(20)
            sel_row_b.addWidget(btn_b_all); sel_row_b.addWidget(btn_b_none); sel_row_b.addStretch()
            boxes_outer.addLayout(sel_row_b)
            popup.boxes_grid = QGridLayout(); popup.boxes_grid.setHorizontalSpacing(8); popup.boxes_grid.setVerticalSpacing(2)
            boxes_outer.addLayout(popup.boxes_grid)

            def _sel_all_boxes(_, p=popup):
                for i in range(p.boxes_grid.count()):
                    w = p.boxes_grid.itemAt(i).widget()
                    if isinstance(w, QCheckBox): w.setChecked(True)
            def _sel_none_boxes(_, p=popup):
                for i in range(p.boxes_grid.count()):
                    w = p.boxes_grid.itemAt(i).widget()
                    if isinstance(w, QCheckBox): w.setChecked(False)
            btn_b_all.clicked.connect(_sel_all_boxes)
            btn_b_none.clicked.connect(_sel_none_boxes)
            layout.addWidget(popup.grp_boxes)

            # ── Spatial Aperture Regions ──────────────────────────────────
            popup.grp_apertures = QGroupBox("Spatial Aperture Regions")
            ap_outer = QVBoxLayout(popup.grp_apertures)
            ap_outer.setSpacing(4)
            sel_row_a = QHBoxLayout()
            btn_a_all  = QPushButton("Select All");   btn_a_all.setStyleSheet(_BTN_STYLE);  btn_a_all.setFixedHeight(20)
            btn_a_none = QPushButton("Deselect All"); btn_a_none.setStyleSheet(_BTN_STYLE); btn_a_none.setFixedHeight(20)
            ap_note = QLabel("(all unchecked = Whole Map)"); ap_note.setStyleSheet("font-style: italic; font-size: 10px;")
            sel_row_a.addWidget(btn_a_all); sel_row_a.addWidget(btn_a_none); sel_row_a.addWidget(ap_note); sel_row_a.addStretch()
            ap_outer.addLayout(sel_row_a)
            popup.apertures_grid = QGridLayout(); popup.apertures_grid.setHorizontalSpacing(8); popup.apertures_grid.setVerticalSpacing(2)
            ap_outer.addLayout(popup.apertures_grid)

            def _sel_all_ap(_, p=popup):
                for i in range(p.apertures_grid.count()):
                    w = p.apertures_grid.itemAt(i).widget()
                    if isinstance(w, QCheckBox): w.setChecked(True)
            def _sel_none_ap(_, p=popup):
                for i in range(p.apertures_grid.count()):
                    w = p.apertures_grid.itemAt(i).widget()
                    if isinstance(w, QCheckBox): w.setChecked(False)
            btn_a_all.clicked.connect(_sel_all_ap)
            btn_a_none.clicked.connect(_sel_none_ap)
            layout.addWidget(popup.grp_apertures)

            # ── Statistics to compute (3-column grid) ────────────────────
            grp_stats = QGroupBox("Statistics to compute")
            stats_grid = QGridLayout(grp_stats)
            stats_grid.setHorizontalSpacing(12)
            _STAT_OPTIONS = [
                ("Integrated Intensity", False), ("RMS", False),        ("Peak (Max)", False),
                ("Min", False),                 ("Mean", False),       ("Median", False),
                ("Std. Deviation", False),      ("SNR (Peak/RMS)", False), ("Sum", False),
            ]
            popup.stat_checkboxes = {}
            for idx, (stat_name, default_on) in enumerate(_STAT_OPTIONS):
                cb = QCheckBox(stat_name)
                cb.setChecked(default_on)
                cb.toggled.connect(lambda checked, p=popup: self._run_spectral_stats_calc(p))
                stats_grid.addWidget(cb, *divmod(idx, 3))
                popup.stat_checkboxes[stat_name] = cb
            layout.addWidget(grp_stats)

            # ── Results (scrollable, dark bg + green text) ───────────────
            results_group = QGroupBox("Results")
            results_inner = QVBoxLayout(results_group)
            results_inner.setContentsMargins(4, 4, 4, 4)
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setMinimumHeight(90)
            scroll.setMaximumHeight(200)
            scroll.setFrameShape(QScrollArea.NoFrame)
            popup.lbl_result = QLabel("---")
            popup.lbl_result.setWordWrap(True)
            popup.lbl_result.setAlignment(Qt.AlignLeft | Qt.AlignTop)
            popup.lbl_result.setTextFormat(Qt.RichText)
            popup.lbl_result.setContentsMargins(6, 6, 6, 6)
            popup.lbl_result.setStyleSheet(
                "background-color: #0d0d0d; color: #2ecc71; font-weight: bold; font-size: 11px;"
            )
            scroll.setWidget(popup.lbl_result)
            results_inner.addWidget(scroll)
            layout.addWidget(results_group)
            # No separate Calculate button — all calculations run live on toggle

            self._spectral_stats_popup = popup
            self.refresh_spectral_stats_popup()
            btn_pos = self.btn_spectral_stats.mapToGlobal(self.btn_spectral_stats.rect().topRight())
            popup.move(btn_pos)
            popup.show()
        else:
            self._spectral_stats_popup.raise_()
            self._spectral_stats_popup.activateWindow()


    def refresh_spectral_stats_popup(self):
        popup = self._spectral_stats_popup
        if popup is None: return

        # ── Rebuild boxes (auto-flow into 3 cols) ────────────────────────
        while popup.boxes_grid.count():
            item = popup.boxes_grid.takeAt(0)
            if item.widget(): item.widget().deleteLater()
        active_rois = self.get_active_spectrum_rois()
        if not active_rois:
            popup.boxes_grid.addWidget(QLabel("No boxes drawn yet."), 0, 0)
        for idx, item in enumerate(active_rois):
            cb = QCheckBox(item["name"])
            cb.setChecked(False)
            cb.toggled.connect(lambda checked, p=popup: self._run_spectral_stats_calc(p))
            popup.boxes_grid.addWidget(cb, *divmod(idx, 3))

        # ── Rebuild apertures (auto-flow into 3 cols) ────────────────────
        while popup.apertures_grid.count():
            it = popup.apertures_grid.takeAt(0)
            if it.widget(): it.widget().deleteLater()
        spatial_rois = getattr(self, 'spectrum_spatial_rois', [])
        if not spatial_rois:
            popup.apertures_grid.addWidget(QLabel("No spatial regions defined."), 0, 0)
        for idx, r_dict in enumerate(spatial_rois):
            cb = QCheckBox(r_dict["name"])
            cb.setStyleSheet(f"color: {r_dict['color']};")
            cb.toggled.connect(lambda checked, p=popup: self._run_spectral_stats_calc(p))
            popup.apertures_grid.addWidget(cb, *divmod(idx, 3))

        popup.adjustSize()


    def refresh_spectral_stats_apertures(self):
        """Called when spatial regions are added/removed to refresh only the apertures section."""
        if self._spectral_stats_popup and self._spectral_stats_popup.isVisible():
            self.refresh_spectral_stats_popup()

    def _get_popup_selected_boxes(self, popup):
        boxes = []
        active_rois = self.get_active_spectrum_rois()
        for i in range(popup.boxes_grid.count()):
            widget = popup.boxes_grid.itemAt(i).widget()
            if isinstance(widget, QCheckBox) and widget.isChecked():
                name = widget.text()
                for item in active_rois:
                    if item["name"] == name:
                        boxes.append(item["roi"])
                        break
        return boxes

    def _get_popup_selected_apertures(self, popup):
        """Returns list of r_dict for selected apertures; empty means Whole Map."""
        selected = []
        spatial_rois = getattr(self, 'spectrum_spatial_rois', [])
        for i in range(popup.apertures_grid.count()):
            widget = popup.apertures_grid.itemAt(i).widget()
            if isinstance(widget, QCheckBox) and widget.isChecked():
                name = widget.text()
                for r_dict in spatial_rois:
                    if r_dict["name"] == name:
                        selected.append(r_dict)
                        break
        return selected

    def _extract_spectrum_for_stats(self, spatial_roi):
        """Extract mean spectrum from cube_clean for a given spatial ROI (or whole map if None).
        Returns (v_axis_sorted, flux_sorted) or (None, None) on failure."""
        if self.cube_clean is None: return None, None
        stat = self.combo_spec_stat.currentText()
        try:
            if spatial_roi is None:
                sub_data = self.cube_clean
            else:
                sub_data = spatial_roi.getArrayRegion(
                    self.cube_clean, self.view_channel.getImageItem(), axes=(1, 2))
            if "Max" in stat:
                spec = np.nanmax(sub_data, axis=(1, 2))
            elif "Sum" in stat:
                spec = np.nansum(sub_data, axis=(1, 2))
                if self.pixels_per_beam > 1.0:
                    spec /= self.pixels_per_beam
            else:
                spec = np.nanmean(sub_data, axis=(1, 2))
            sort_idx = np.argsort(self.v_axis)
            return self.v_axis[sort_idx], spec[sort_idx]
        except Exception:
            return None, None

    def _run_spectral_stats_calc(self, popup):
        selected_rois_1d = self._get_popup_selected_boxes(popup)
        selected_apertures = self._get_popup_selected_apertures(popup)

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
        unit = getattr(self, 'spec_unit', '')
        results_html = []


        for ap in apertures_to_calc:
            name = ap["name"]
            # Extract spectrum independently from cube – ignores what the panel shows
            v_axis, flux = self._extract_spectrum_for_stats(ap["roi"])
            if v_axis is None or flux is None:
                results_html.append(f"<b>{name}:</b> Could not extract data")
                continue

            # Build mask from selected 1D velocity boxes
            combined_mask = np.zeros_like(v_axis, dtype=bool)
            for roi in selected_rois_1d:
                pos = roi.pos(); size = roi.size()
                min_v, max_v = pos.x(), pos.x() + size.x()
                if min_v > max_v: min_v, max_v = max_v, min_v
                combined_mask |= (v_axis >= min_v) & (v_axis <= max_v)

            valid_flux = flux[combined_mask]
            if len(valid_flux) == 0:
                results_html.append(f"<b>{name}:</b> No data in selected range")
                continue

            stats_lines = [f"<b style='color:#89b4fa'>{name}</b>"]
            dv = abs(v_axis[1] - v_axis[0]) if len(v_axis) > 1 else 1.0
            valid_v = v_axis[combined_mask]

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

        popup.lbl_result.setText("<br><br>".join(results_html) if results_html else "---")
        self.lbl_region_result.setText("---")

    def on_region_selected(self, _=None):
        selected_names = [
            self.combo_regions.currentText(),
            self.combo_regions_2.currentText(),
            self.combo_regions_3.currentText()
        ]
        
        active_rois = self.get_active_spectrum_rois()
        for item in active_rois:
            roi = item["roi"]
            if roi in getattr(self, 'rois_to_delete', []):
                roi.setPen(pg.mkPen('r', width=3))
            elif item["name"] in selected_names and item["name"] != "None":
                roi.setPen(pg.mkPen('y', width=3))
            else:
                roi.setPen(pg.mkPen('c', width=2))
        # Trigger recalculation in popup if open
        if self._spectral_stats_popup and self._spectral_stats_popup.isVisible():
            self._run_spectral_stats_calc(self._spectral_stats_popup)

    def update_spectrum_region_calc(self, _=None):
        """Legacy stub: calculation is now handled by the popup widget."""
        if self._spectral_stats_popup and self._spectral_stats_popup.isVisible():
            self._run_spectral_stats_calc(self._spectral_stats_popup)
