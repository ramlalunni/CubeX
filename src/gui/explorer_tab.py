import csv
import warnings
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
                             QPushButton, QMessageBox, QLineEdit, 
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

import qtawesome as qta

# Import our modularized components
from src.core.splatalogue import SplatalogueWorker
from src.gui.custom import JumpSlider, fix_axis_scaling, WCSAxisItem
from src.gui.dialogs import LineCatalogDialog, LineSelectionDialog, ContourDialog, ChannelGridDialog, ContourOptionsDialog

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

# TODO: Numba temporarily bypassed due to catastrophic cancellation precision issues in Moment 2 variance math. Fix in next update.
if False: # _NUMBA_AVAILABLE:
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
                pv_width   = cfg.get('pv_width', 1)

                if pv_points is None or pv_cube is None or pv_sub_v is None:
                    panel_results.append({'mtype': mtype, 'data': None})
                    continue

                p1, p2 = pv_points
                offsets, pv_data = MomentWorker._sample_along_line(
                    p1, p2, pv_cube, nx, ny, pix_scale, width=pv_width
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
            # CRITICAL FIX: Evaluate the threshold against the 3D sub-cube voxel-by-voxel.
            mask = selected_cube > t
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
                        # Filter out physically impossible centroids caused by division by near-zero sums
                        data[(data < minX - 1e-5) | (data > maxX + 1e-5)] = np.nan
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
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
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
    def _sample_along_line(p1, p2, cube_data, nx, ny, pix_scale_arcsec, width=1):
        """
        Pure-NumPy bilinear interpolation along a line through the cube.
        p1, p2 are world-coordinate arrays [x_arcsec, y_arcsec].
        width is the number of pixels averaged perpendicular to the cut
        (1 = no averaging).
        Returns (offsets, samples.T).
        """
        if width < 1:
            width = 1

        dx_w = p2[0] - p1[0]
        dy_w = p2[1] - p1[1]
        length_arcsec = np.hypot(dx_w, dy_w)
        n_samples = max(int(np.ceil(length_arcsec / max(pix_scale_arcsec, 1e-6))) + 1, 2)

        if width == 1:
            xs = np.linspace(p1[0], p2[0], n_samples)
            ys = np.linspace(p1[1], p2[1], n_samples)
            offsets = np.linspace(0.0, length_arcsec, n_samples)

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

        all_samples = []
        offsets_k = np.linspace(-(width - 1) / 2.0, (width - 1) / 2.0, width)
        for k in offsets_k:
            off_x = k * pix_scale_arcsec * dy_w / max(length_arcsec, 1e-6)
            off_y = -k * pix_scale_arcsec * dx_w / max(length_arcsec, 1e-6)
            off_p1 = np.array([p1[0] + off_x, p1[1] + off_y], dtype=float)
            off_p2 = np.array([p2[0] + off_x, p2[1] + off_y], dtype=float)

            _, samples = MomentWorker._sample_along_line(
                off_p1, off_p2, cube_data, nx, ny, pix_scale_arcsec, width=1
            )
            if samples is not None:
                all_samples.append(samples)

        if not all_samples:
            offsets = np.linspace(0.0, length_arcsec, n_samples)
            return offsets, np.full((n_samples, cube_data.shape[0]), np.nan, dtype=np.float64)

        offsets = np.linspace(0.0, length_arcsec, n_samples)
        stacked = np.dstack(all_samples)
        with np.errstate(all='ignore'):
            avg = np.nanmean(stacked, axis=2)
        return offsets, avg


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
        if ((ev.modifiers() == Qt.ControlModifier and ev.isStart()) or self.current_roi is not None) and self.parent_tab:
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
            elif mode == "Spectrum":
                tool = self.parent_tab.combo_roi.currentText()
                if tool in ["Whole Map", "Point (Beam)", "Custom Polygon"]:
                    ev.ignore()
                    return
                
                if ev.isStart():
                    self.drag_start = self.mapSceneToView(ev.buttonDownScenePos())
                    if tool == "Rectangle":
                        self.current_roi = pg.RectROI([self.drag_start.x(), self.drag_start.y()], [1e-5, 1e-5], pen=pg.mkPen('#f1c40f'))
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
                        self.current_roi = pg.EllipseROI([self.drag_start.x(), self.drag_start.y()], [1e-5, 1e-5], pen=pg.mkPen('#f1c40f'))
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
                        self.parent_tab._finish_roi_addition(self.current_roi, tool)
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
            elif mode == "Spectrum":
                tool = self.parent_tab.combo_roi.currentText()
                if tool == "Point (Beam)":
                    pos = self.mapSceneToView(ev.scenePos())
                    self.parent_tab.change_roi(tool, cx=pos.x(), cy=pos.y())
                    ev.accept()
                    return
                
        super().mouseClickEvent(ev)

class SpectrumViewBox(pg.ViewBox):
    def __init__(self, *args, **kwds):
        super().__init__(*args, **kwds)
        self.drag_start = None
        self.current_roi = None
        self.parent_tab = None

    def mouseDragEvent(self, ev, axis=None):
        if (ev.modifiers() == Qt.ControlModifier and ev.isStart()) or self.current_roi is not None:
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
        super().mouseClickEvent(ev)

class ExplorerTab(QWidget):
    def __init__(self, parent_window):
        super().__init__()
        self.parent_window = parent_window 
        self.is_2d_image = False
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

        self.is_drawing_polygon = False
        self.polygon_points = []
        self.roi_selected = False
        self.catalog_overlay_items = []
        self.spectrum_spatial_rois = [] # List of {"name": str, "roi": ROI, "checkbox": QCheckBox, "color": str}
        self.active_spatial_spectrum_roi = None
        self.roi_selected = False
        
        from PyQt5.QtWidgets import QShortcut
        from PyQt5.QtGui import QKeySequence
        self.esc_shortcut = QShortcut(QKeySequence(Qt.Key_Escape), self)
        self.esc_shortcut.setContext(Qt.WidgetWithChildrenShortcut)
        self.esc_shortcut.activated.connect(self.handle_escape)

        # Polygon drawing state
        self.is_drawing_polygon = False
        self.polygon_points = []
        self.polygon_preview_line = None
        
        self.region_colors = ['#8A2BE2', '#FF0000', '#FFA500', '#228B22', '#FF00FF', '#FFFF00', '#DDA0DD', '#E9967A', '#0000CD', '#D2691E']
        self.roi_selected = False
        self.current_m0_raw = None
        self.active_picker_panel = None
        self.pv_data = None
        self.pv_offset_axis = None
        self.pv_velocity_axis = None
        
        self.last_clicked_panel_id = 'channel' 
        self.contour_params = {'channel': None, 0: None, 1: None, 2: None}
        self.active_contours = {'channel': [], 0: [], 1: [], 2: []}

        self.contour_overlays = []
        self.overlay_color_palette = ['cyan', 'magenta', 'yellow', 'lime', 'orange', 'pink', 'aquamarine', 'gold', 'salmon', 'skyblue']

        self.contour_overlay_file = None
        self.contour_overlay_cube = None
        self.contour_overlay_wcs = None
        self.contour_overlay_v_axis = None
        self.contour_overlay_is_static = False
        self.contour_overlay_2d = None
        self.contour_overlay_nx = 0
        self.contour_overlay_ny = 0
        self.contour_overlay_pix_scale = 1.0
        self.contour_overlay_iso_items = []
        self._overlay_reproject_cache = None
        self.contour_options = {
            'mode': 'rms',
            'rms': 0.001,
            'multipliers_str': '3, 5, 10, 20, 40',
            'lin_min': 0.0,
            'lin_max': 10.0,
            'n_levels': 5,
            'log_min': 0.001,
            'log_max': 10.0,
            'log_base': 10.0,
            'percentages_str': '10, 30, 50, 70, 90',
            'color': 'white',
            'line_width': 1.5,
            'line_style': 'solid',
            'smooth': False,
            'smooth_kernel': 3,
        }

        self.overlay_spectrum_curves = {}
        self.overlay_spectrum_curves_smooth = {}
        
        self.playback_timer = QTimer()
        self.playback_timer.timeout.connect(self.step_channel)
        self.play_direction = 1

        # Background worker for moment / PV computation
        self._moment_worker = None
        self._moment_generation = 0
        self._pending_workers = []

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
        self.plot_channel.vb.sigRangeChanged.connect(self.update_beam_positions)
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
        
        self.lbl_spatial_tool = QLabel("Spatial Analysis Tool:")
        roi_layout.addWidget(self.lbl_spatial_tool)
        self.combo_spatial_tool = QComboBox()
        self.combo_spatial_tool.setFixedHeight(22)
        self.combo_spatial_tool.addItems(["None", "Point", "Line", "Rectangle", "Ellipse"])
        self.combo_spatial_tool.currentTextChanged.connect(self.change_spatial_tool)
        roi_layout.addWidget(self.combo_spatial_tool)
        
        self.btn_edit_region = QPushButton("Edit region")
        self.btn_edit_region.setFixedHeight(22)
        self.btn_edit_region.setStyleSheet("font-size: 11px; padding: 0px 4px;")
        self.btn_edit_region.hide()
        self.btn_edit_region.clicked.connect(self.open_edit_region_dialog)
        roi_layout.addWidget(self.btn_edit_region)
        
        self.lbl_spatial_tool.hide()
        self.combo_spatial_tool.hide()
        roi_layout.addStretch()
        
        # Draw a reliable square grid icon using QPainter to avoid any font or XPM string issues
        from PyQt5.QtGui import QIcon, QPixmap, QPainter, QPen, QColor
        pix = QPixmap(16, 16)
        pix.fill(Qt.transparent)
        painter = QPainter(pix)
        painter.setPen(QPen(QColor(255, 255, 255), 1))
        painter.drawRect(2, 2, 11, 11)
        painter.drawLine(6, 2, 6, 13)
        painter.drawLine(10, 2, 10, 13)
        painter.drawLine(2, 6, 13, 6)
        painter.drawLine(2, 10, 13, 10)
        painter.end()
        self.btn_contour_overlay = QPushButton("Contour Options")
        self.btn_contour_overlay.setToolTip("Configure contour overlay from external FITS file")
        self.btn_contour_overlay.setFixedHeight(22)
        self.btn_contour_overlay.setStyleSheet("font-size: 11px; padding: 0px 6px;")
        self.btn_contour_overlay.clicked.connect(self.open_contour_options)
        self.btn_contour_overlay.setEnabled(False)
        self.btn_contour_overlay.hide()
        roi_layout.addWidget(self.btn_contour_overlay)

        self.btn_grid = QPushButton()
        self.btn_grid.setIcon(QIcon(pix))
        self.btn_grid.setToolTip("View channel grid")
        self.btn_grid.setFixedWidth(30)
        self.btn_grid.setFixedHeight(22)
        self.btn_grid.clicked.connect(self.open_channel_grid_popup)
        roi_layout.addWidget(self.btn_grid)
        
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
        self.combo_spatial_regions.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.combo_spatial_regions.setMinimumContentsLength(14)
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
        self.plot_spatial_1.addLegend(offset=(1, 1))
        self.curve_spatial_1 = self.plot_spatial_1.plot([], [], pen=pg.mkPen('w', width=2), name="Base")
        
        self.plot_spatial_2 = pg.PlotWidget(title="Y Profile")
        self.plot_spatial_2.showGrid(x=True, y=True, alpha=0.3)
        self.plot_spatial_2.setLabel('bottom', 'Offset (arcsec)')
        self.plot_spatial_2.setLabel('left', 'Flux')
        self.plot_spatial_2.addLegend(offset=(1, 1))
        self.curve_spatial_2 = self.plot_spatial_2.plot([], [], pen=pg.mkPen('w', width=2), name="Base")
        
        self.lbl_spatial_stats = QLabel("Draw a region to see statistics.")
        self.lbl_spatial_stats.setAlignment(Qt.AlignCenter)
        self.lbl_spatial_stats.setStyleSheet("font-size: 13px; color: #aaa; background-color: #1a1a1a; border: 1px solid #333; border-radius: 4px; padding: 10px;")

        self.spatial_stats_scroll = QScrollArea()
        self.spatial_stats_scroll.setWidgetResizable(True)
        self.spatial_stats_scroll.setStyleSheet("QScrollArea { border: 1px solid #333; border-radius: 4px; background-color: #1a1a1a; }")
        self.spatial_stats_container = QWidget()
        self.spatial_stats_container.setStyleSheet("background-color: #1a1a1a;")
        self.spatial_stats_layout = QHBoxLayout(self.spatial_stats_container)
        self.spatial_stats_layout.setContentsMargins(5, 5, 5, 5)
        self.spatial_stats_scroll.setWidget(self.spatial_stats_container)
        self.spatial_stats_scroll.hide()

        self.stacked_spatial_info = QStackedWidget()
        self.stacked_spatial_info.addWidget(self.lbl_spatial_stats)
        self.stacked_spatial_info.addWidget(self.spatial_stats_scroll)

        spatial_layout.addWidget(self.plot_spatial_1, stretch=1)
        spatial_layout.addWidget(self.plot_spatial_2, stretch=1)
        spatial_layout.addWidget(self.stacked_spatial_info)
        self.stacked_panel.addWidget(self.spatial_widget)

        self.pv_widget = QWidget()
        pv_layout = QVBoxLayout(self.pv_widget)
        pv_layout.setContentsMargins(0, 0, 0, 0)

        pv_controls_layout = QHBoxLayout()
        self.lbl_pv_cut_sel = QLabel("Select Cut:")
        self.combo_pv_cuts = QComboBox()
        self.combo_pv_cuts.addItem("None")
        self.combo_pv_cuts.currentTextChanged.connect(self.on_pv_cut_selected)
        self.btn_edit_pv = QPushButton("Edit")
        self.btn_edit_pv.setFixedHeight(22)
        self.btn_edit_pv.setStyleSheet("font-size: 11px; padding: 0px 4px;")
        self.btn_edit_pv.clicked.connect(self.open_edit_pv_cut_dialog)
        self.btn_delete_pv = QPushButton("Delete Selected")
        self.btn_delete_pv.clicked.connect(self.delete_selected_pv_via_button)
        pv_controls_layout.addWidget(self.lbl_pv_cut_sel)
        pv_controls_layout.addWidget(self.combo_pv_cuts)
        pv_controls_layout.addWidget(self.btn_edit_pv)
        pv_controls_layout.addWidget(self.btn_delete_pv)
        pv_controls_layout.addStretch()
        pv_layout.addLayout(pv_controls_layout)

        self.lbl_pv_help = QLabel("Ctrl+drag on the channel map to draw a PV cut.")
        self.lbl_pv_help.setAlignment(Qt.AlignCenter)
        self.lbl_pv_help.setStyleSheet("font-size: 12px; color: #aaa;")
        pv_layout.addWidget(self.lbl_pv_help)

        self.pv_plot_item = pg.PlotItem(title="PV Diagram")
        self.pv_hover_line_main = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('m', width=1.5, style=Qt.DashLine))
        self.pv_hover_line_main.hide()
        self.pv_plot_item.addItem(self.pv_hover_line_main)
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
        self.spectrum_tabs.setTabsClosable(True)
        self.spectrum_tabs.tabCloseRequested.connect(self._on_spectrum_tab_close_requested)
        from PyQt5.QtWidgets import QTabBar
        self.spectrum_tabs.tabBar().setTabButton(0, QTabBar.RightSide, None)
        self.spectrum_tabs.tabBar().setTabButton(0, QTabBar.LeftSide, None)
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
        
        hover_layout = QHBoxLayout()
        hover_layout.addWidget(self.lbl_hover_spec)
        hover_layout.addStretch()
        
        self.lbl_rj_warning = QLabel("Warning: Brightness converted using the Rayleigh–Jeans approximation.")
        self.lbl_rj_warning.setStyleSheet("color: orange; font-weight: normal;")
        self.lbl_rj_warning.hide()
        hover_layout.addWidget(self.lbl_rj_warning)
        
        spectrum_layout.addLayout(hover_layout)

        input_layout = QHBoxLayout()
        input_layout.addWidget(QLabel("Statistic:"))
        self.combo_spec_stat = QComboBox()
        self.combo_spec_stat.addItems(["Flux Density", "Mean", "Median", "Max (Peak)"])
        self.combo_spec_stat.setCurrentText("Mean")
        self.combo_spec_stat.currentTextChanged.connect(self._update_spectrum_state_machine)
        self.combo_spec_stat.currentTextChanged.connect(self.update_spectrum)
        self.combo_spec_stat.currentTextChanged.connect(lambda: self.lbl_region_result.setText("---"))
        input_layout.addWidget(self.combo_spec_stat)
        
        input_layout.addWidget(QLabel("Unit:"))
        self.combo_spec_unit = QComboBox()
        self.combo_spec_unit.addItems(["Native", "Jy", "K", "Jy/beam"])
        self.combo_spec_unit.currentTextChanged.connect(self.update_spectrum)
        input_layout.addWidget(self.combo_spec_unit)
        
        self.btn_smooth = QPushButton("Smooth")
        self.btn_smooth.clicked.connect(self.open_smoothing_dialog)
        input_layout.addWidget(self.btn_smooth)
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
        self.btn_spectral_stats = QPushButton("Spectral Statistics")
        self.btn_spectral_stats.setToolTip("Open spectral statistics panel for drawn velocity boxes")
        self.btn_spectral_stats.clicked.connect(self.open_spectral_stats_popup)
        self.btn_spectral_stats.hide()
        input_layout.addWidget(self.btn_spectral_stats)

        self.spectrum_rois = []
        self.rois_to_delete = []
        self._spectral_stats_popup = None
        self._channel_grid_popup = None
        
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
        self.toggle_bottom_btn = QPushButton("▼ Show Moment/PV Panels")
        self.toggle_bottom_btn.setStyleSheet("QPushButton { border: none; color: #3498db; text-align: left; padding: 5px; font-weight: bold; background: transparent; } QPushButton:hover { color: #5dade2; }")
        self.toggle_bottom_btn.clicked.connect(self.toggle_bottom_pane)
        main_layout.addWidget(self.toggle_bottom_btn)

        self.bottom_container = QWidget()
        self.bottom_half = QHBoxLayout(self.bottom_container)
        self.bottom_half.setContentsMargins(0, 0, 0, 0)
        self.bottom_half.setSpacing(6)
        self.panels = []

        moment_options = ["Moment 0 (Integrated Intensity)", "Moment 1 (Velocity Field)",
                          "Moment 2 (Velocity Dispersion)", "Moment 8 (Peak Intensity)",
                          "Moment 9 (Peak Velocity)", "PV Diagram"]

        for i, default_option in enumerate([moment_options[0], moment_options[3], moment_options[1]]):
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
            combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
            combo.setMinimumContentsLength(10)
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
            combo_pv_range.addItems(["Selected", "Full Cube"])
            combo_pv_range.setCurrentText("Selected")
            combo_pv_range.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            combo_pv_range.setMinimumContentsLength(9)
            combo_pv_range.setFixedWidth(98)
            pv_controls_layout.addWidget(combo_pv_range)
            btn_edit_pv = QPushButton()
            edit_icon = qta.icon('fa5s.pen')  # Creates a solid pencil icon
            btn_edit_pv.setIcon(edit_icon)
            btn_edit_pv.setToolTip("Edit PV cut")
            btn_edit_pv.setFixedHeight(22)
            btn_edit_pv.setStyleSheet("font-size: 11px; padding: 0px 4px;")
            pv_controls_layout.addWidget(btn_edit_pv)
            btn_delete_pv = QPushButton()
            trash_icon = qta.icon('mdi.trash-can-outline')
            btn_delete_pv.setIcon(trash_icon)
            btn_delete_pv.setToolTip("Remove PV Diagram Plot")
            btn_delete_pv.setFixedWidth(30)
            btn_delete_pv.setFixedHeight(22)
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
            view.ui.histogram.setMaximumWidth(160)
            view.ui.histogram.setMinimumWidth(80)
            fix_axis_scaling(view.ui.histogram.axis) 
            plot_item.vb.sigRangeChanged.connect(self.update_beam_positions)
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
            panel['btn_edit_pv'] = btn_edit_pv
            panel['btn_delete_pv'] = btn_delete_pv
            panel['lbl_hover'] = lbl_hover
            panel['current_data'] = None
            panel['pv_offset_axis'] = None
            panel['pv_velocity_axis'] = None
            panel['id'] = i
            panel['unit'] = ''
            panel['pv_hover_line'] = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('m', width=1.5, style=Qt.DashLine))
            panel['pv_hover_line'].hide()
            plot_item.addItem(panel['pv_hover_line'])
            self.panels.append(panel)

            combo.currentTextChanged.connect(self.update_moment_maps)
            input_thresh.editingFinished.connect(self.update_moment_maps)
            combo_pv_cut.currentTextChanged.connect(lambda _text, p_id=i: self.on_panel_pv_cut_selected(p_id))
            combo_pv_range.currentTextChanged.connect(self.update_moment_maps)
            btn_delete_pv.clicked.connect(lambda _checked=False, p_id=i: self.delete_panel_pv_cut(p_id))
            btn_edit_pv.clicked.connect(self.open_edit_pv_cut_dialog)
            plot_item.scene().sigMouseMoved.connect(lambda pos, p=panel: self.hover_panel(pos, p))
            btn_pick.clicked.connect(lambda checked, p_id=i: self.set_active_picker(checked, p_id))

        self.bottom_container.setVisible(False)
        main_layout.addWidget(self.bottom_container, stretch=1)

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

    def toggle_bottom_pane(self):
        is_visible = self.bottom_container.isVisible()
        main_win = self.window()
        
        from PyQt5.QtWidgets import QApplication
        screen_height = QApplication.desktop().availableGeometry(main_win).height()
        
        if is_visible:
            # Hiding the bottom pane
            self.bottom_container.setVisible(False)
            self.toggle_bottom_btn.setText("▼ Show Moment/PV Panels")
            
            if main_win.isMaximized():
                main_win.showNormal()
                
            if hasattr(main_win, 'startup_width') and hasattr(main_win, 'startup_height'):
                main_win.resize(main_win.startup_width, main_win.startup_height)
            else:
                main_win.resize(main_win.width(), main_win.height() // 2)
        else:
            # Showing the bottom pane
            target_height = int(main_win.height() * 1.6)
            
            if screen_height > target_height:
                main_win.resize(main_win.width(), target_height)
            else:
                main_win.showMaximized()
                
            self.bottom_container.setVisible(True)
            self.toggle_bottom_btn.setText("▲ Hide Moment/PV Panels")

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
            
            if not getattr(self, 'spatial_rois', []):
                self.combo_spatial_tool.blockSignals(True)
                self.combo_spatial_tool.setCurrentText("None")
                self.combo_spatial_tool.blockSignals(False)
            
            self.change_spatial_tool(self.combo_spatial_tool.currentText(), auto_draw=False)

    def delete_nr_roi(self):
        if getattr(self, 'nr_roi', None) is not None:
            try:
                if self.nr_roi.scene() is not None:
                    self.view_channel.getView().removeItem(self.nr_roi)
            except Exception: pass
            self.nr_roi = None
        if getattr(self, 'nr_label', None) is not None:
            try:
                if self.nr_label.scene() is not None:
                    self.view_channel.getView().removeItem(self.nr_label)
            except Exception: pass
            self.nr_label = None
        self.nr_roi_selected = False

    def update_nr_rms(self):
        if getattr(self, 'nr_roi', None) is None:
            return
        ch_idx = self.slider_channel.value()
        if self.cube_clean is None or ch_idx < 0 or ch_idx >= self.cube_clean.shape[0]:
            return
            
        current_slice = self.cube_clean[ch_idx, :, :]
        r_pos = self.nr_roi.pos()
        r_size = self.nr_roi.size()
        
        start_x = (self.nx / 2) * self.pix_scale_arcsec
        start_y = -(self.ny / 2) * self.pix_scale_arcsec
        
        min_x_scene = min(r_pos.x(), r_pos.x() + r_size.x())
        max_x_scene = max(r_pos.x(), r_pos.x() + r_size.x())
        min_y_scene = min(r_pos.y(), r_pos.y() + r_size.y())
        max_y_scene = max(r_pos.y(), r_pos.y() + r_size.y())
        
        x1_idx = int((min_x_scene - start_x) / (-self.pix_scale_arcsec))
        x2_idx = int((max_x_scene - start_x) / (-self.pix_scale_arcsec))
        y1_idx = int((min_y_scene - start_y) / self.pix_scale_arcsec)
        y2_idx = int((max_y_scene - start_y) / self.pix_scale_arcsec)
        
        x_min = max(0, min(x1_idx, x2_idx))
        x_max = min(self.nx, max(x1_idx, x2_idx) + 1)
        y_min = max(0, min(y1_idx, y2_idx))
        y_max = min(self.ny, max(y1_idx, y2_idx) + 1)
        
        if x_min >= x_max or y_min >= y_max:
            return
            
        extracted_data = current_slice[x_min:x_max, y_min:y_max]
        rms_val = 3.0 * float(np.nanstd(extracted_data))
        
        if np.isnan(rms_val):
            val_str = "NaN"
        else:
            val_str = f"{rms_val:.4e}" if rms_val < 1e-3 else f"{rms_val:.4f}"
            
        if getattr(self, 'nr_label', None) is not None:
            self.nr_label.setText(f"NR (3σ = {val_str})")
            
        pid = getattr(self.nr_roi, 'target_panel_id', None)
        if pid is not None and 0 <= pid < len(self.panels):
            target_panel = self.panels[pid]
            if not np.isnan(rms_val):
                target_panel['input_thresh'].setText(val_str)
                self.update_moment_maps()

    def handle_escape(self):
        if getattr(self, 'active_picker_panel', None) is not None:
            pid = self.active_picker_panel
            self.set_active_picker(False, pid)
            if hasattr(self, 'panels') and pid < len(self.panels):
                self.panels[pid]['btn_pick'].setChecked(False)
            return
        if getattr(self, 'nr_roi_selected', False):
            self.delete_nr_roi()
            return
            
        if getattr(self, 'spatial_rois_to_delete', []):
            self.delete_selected_spatial_regions()
        if getattr(self, 'pv_cuts_to_delete', []):
            self.delete_selected_pv_cuts()
        if getattr(self, 'rois_to_delete', []):
            self.delete_selected_regions()
        if getattr(self, 'active_spatial_spectrum_roi', None):
            self.remove_spatial_spectrum_roi(self.active_spatial_spectrum_roi)
            self.active_spatial_spectrum_roi = None
    def open_smoothing_dialog(self):
        # Prevent multiple instances
        if getattr(self, '_smooth_dialog_active', False) and hasattr(self, '_smooth_dialog') and self._smooth_dialog:
            self._smooth_dialog.raise_()
            self._smooth_dialog.activateWindow()
            return
            
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
        
        self._smooth_dialog = SpectralSmoothingDialog(self.window())
        self._smooth_dialog.setWindowModality(Qt.NonModal)
        
        def on_apply(params):
            self.smoothing_params = params
            if self.spectrum_tabs.indexOf(self.plot_widget_smooth) == -1:
                self.spectrum_tabs.addTab(self.plot_widget_smooth, "Smoothed")
            self.spectrum_tabs.tabBar().show()
            self.spectrum_tabs.setCurrentWidget(self.plot_widget_smooth)
            self.update_spectrum()
            
        def on_close():
            self._smooth_dialog_active = False
            self._smooth_dialog = None
            
        self._smooth_dialog.apply_clicked.connect(on_apply)
        self._smooth_dialog.finished.connect(on_close)
        self._smooth_dialog.show()

    def _on_spectrum_tab_close_requested(self, index):
        if self.spectrum_tabs.widget(index) == self.plot_widget_smooth:
            self.remove_smoothed_spectrum()

    def remove_smoothed_spectrum(self):
        self.smoothing_params = None
        idx = self.spectrum_tabs.indexOf(self.plot_widget_smooth)
        if idx != -1:
            self.spectrum_tabs.removeTab(idx)
        self.spectrum_tabs.tabBar().hide()
        self.spectrum_tabs.setCurrentWidget(self.plot_widget)
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

        offsets, pv_data = self.sample_cube_along_line(active_item["roi"], cube_data,
                                                         width=active_item.get('width', 1))
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

    def change_spatial_tool(self, tool, auto_draw=True):
        if self.cube_clean is None: return
        
        if tool == "None":
            self.plot_spatial_1.hide()
            self.plot_spatial_2.hide()
            self.stacked_spatial_info.setCurrentIndex(0)
            self.lbl_spatial_stats.setText("Choose a tool to begin analysis.")
            self.stacked_spatial_info.show()
            return

        if tool == "Point":
            self.plot_spatial_1.show()
            self.plot_spatial_1.setTitle("X Profile")
            self.plot_spatial_2.show()
            self.stacked_spatial_info.setCurrentIndex(0)
            self.stacked_spatial_info.hide()
        elif tool == "Line":
            self.plot_spatial_1.show()
            self.plot_spatial_1.setTitle("Spatial Profile")
            self.plot_spatial_2.hide()
            self.stacked_spatial_info.setCurrentIndex(0)
            self.stacked_spatial_info.hide()
        else:
            self.plot_spatial_1.hide()
            self.plot_spatial_2.hide()
            self.stacked_spatial_info.setCurrentIndex(1)
            self.spatial_stats_scroll.show()
            self.stacked_spatial_info.show()

        if not auto_draw:
            return

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
        e_count = sum(1 for item in self.spatial_rois if item.get("tool") == "Ellipse")
        r_count = sum(1 for item in self.spatial_rois if item.get("tool") == "Rectangle")
        l_count = sum(1 for item in self.spatial_rois if item.get("tool") == "Line")
        p_count = sum(1 for item in self.spatial_rois if item.get("tool") == "Point")
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
            name = f"{tool} {len(self.spatial_rois) + 1}"

        self.spatial_rois.append({"name": name, "roi": roi, "tool": tool})

        self.combo_spatial_regions.blockSignals(True)
        self.combo_spatial_regions.addItem(name)
        self.combo_spatial_regions.setCurrentText(name)
        self.combo_spatial_regions.blockSignals(False)

        roi.sigRegionChanged.connect(self.update_spatial_analysis)

        if label is not None:
            text_item = pg.TextItem(text=label, color=(255, 255, 255, 200), anchor=(0, 1))
            text_item.setZValue(30)
            self.plot_channel.addItem(text_item)

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
            active_item = self.spatial_rois[-1]
            active_item["text_item"] = text_item
            active_item["update_spatial_label"] = update_spatial_label
        
        if tool == "Line":
            direction_item = pg.PlotDataItem(
                [], [], connect='finite',
                pen=pg.mkPen('#f7dc6f', width=3),
            )
            direction_item.setZValue(20)
            self.plot_channel.addItem(direction_item)

            def update_spatial_arrow(r=roi, a=direction_item):
                points = self.get_line_roi_points(r)
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
                head_len = min(max(6.0 * self.pix_scale_arcsec, 0.18 * length), 0.32 * length)
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
            active_item = self.spatial_rois[-1]
            active_item["direction_item"] = direction_item
            active_item["update_spatial_arrow"] = update_spatial_arrow

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
            for item in self.spatial_rois:
                if item["roi"] == roi:
                    di = item.get("direction_item")
                    if di is not None:
                        try:
                            self.plot_channel.removeItem(di)
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
                            self.plot_channel.removeItem(ti)
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
                    self.view_channel.getView().removeItem(roi)
                except:
                    pass
            
            self.spatial_rois = [item for item in self.spatial_rois if item["roi"] != roi]
        self.spatial_rois_to_delete.clear()

        e_idx = 0
        r_idx = 0
        l_idx = 0
        p_idx = 0
        for item in self.spatial_rois:
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
        cut_info = {"name": name, "roi": roi, "width": 1}
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

        from PyQt5.QtWidgets import QGraphicsPolygonItem
        width_item = QGraphicsPolygonItem()
        width_item.setBrush(pg.mkBrush(255, 255, 255, 40))
        width_item.setPen(pg.mkPen(255, 255, 255, 100, width=1))
        width_item.setZValue(19)
        self.plot_channel.addItem(width_item)

        def update_annotations(r=roi, t=text_item, a=direction_item, w_item=width_item, c_info=cut_info):
            points = self.get_line_roi_points(r)
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

            # Draw width polygon
            cut_width_pixels = c_info.get("width", 1)
            if cut_width_pixels <= 1:
                w_item.hide()
            else:
                w_item.show()
                width_arcsec = cut_width_pixels * self.pix_scale_arcsec
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
        roi.sigRegionChanged.connect(self.update_moment_maps)
        update_annotations()

        cut_info["text_item"] = text_item
        cut_info["direction_item"] = direction_item
        cut_info["width_item"] = width_item
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

    def open_edit_pv_cut_dialog(self):
        if not self.pv_cuts_to_delete:
            return
        selected_roi = self.pv_cuts_to_delete[-1]
        cut_dict = next((item for item in self.pv_cuts if item["roi"] == selected_roi), None)
        if cut_dict is None:
            return
        from src.gui.dialogs import RegionPropertiesDialog
        if getattr(self, '_pv_edit_dialog', None) and self._pv_edit_dialog.isVisible():
            self._pv_edit_dialog.raise_()
            self._pv_edit_dialog.activateWindow()
            return
        roi_dict = {"name": cut_dict["name"], "roi": cut_dict["roi"], "tool": "PV Cut",
                     "text_item": cut_dict.get("text_item"), "pv_cut_dict": cut_dict}
        dlg = RegionPropertiesDialog(cut_dict["roi"], self, parent=self.window(), roi_dict=roi_dict)
        self._pv_edit_dialog = dlg
        dlg.show()

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
                    self.view_channel.getView().removeItem(roi)
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
                    width_item = item.get("width_item")
                    if width_item is not None:
                        if width_item.scene():
                            width_item.scene().removeItem(width_item)
                        else:
                            self.plot_channel.removeItem(width_item)
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

    def sample_cube_along_line(self, roi, cube_data=None, width=1):
        points = self.get_line_roi_points(roi)
        if points is None:
            return None, None
        if cube_data is None:
            cube_data = self.cube_clean
        if width < 1 or width % 2 == 0:
            width = 1

        p1, p2 = points
        dx_world = p2[0] - p1[0]
        dy_world = p2[1] - p1[1]
        length_arcsec = np.hypot(dx_world, dy_world)
        n_samples = max(int(np.ceil(length_arcsec / max(self.pix_scale_arcsec, 1e-6))) + 1, 2)

        if width == 1:
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

        half_w = width // 2
        all_samples = []
        for k in range(-half_w, half_w + 1):
            off_x = k * self.pix_scale_arcsec * dy_world / max(length_arcsec, 1e-6)
            off_y = -k * self.pix_scale_arcsec * dx_world / max(length_arcsec, 1e-6)

            off_p1 = np.array([p1[0] + off_x, p1[1] + off_y], dtype=float)
            off_p2 = np.array([p2[0] + off_x, p2[1] + off_y], dtype=float)

            off_xs = np.linspace(off_p1[0], off_p2[0], n_samples)
            off_ys = np.linspace(off_p1[1], off_p2[1], n_samples)

            x_pix, y_pix = self.world_to_pixel(off_xs, off_ys)
            valid = (
                (x_pix >= 0.0)
                & (x_pix <= self.nx - 1)
                & (y_pix >= 0.0)
                & (y_pix <= self.ny - 1)
            )

            nv = cube_data.shape[0]
            row = np.full((nv, n_samples), np.nan, dtype=np.float64)
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
                row[:, valid] = out

            all_samples.append(row)

        if not all_samples:
            offsets = np.linspace(0.0, length_arcsec, n_samples)
            return offsets, np.full((n_samples, cube_data.shape[0]), np.nan, dtype=np.float64)

        offsets = np.linspace(0.0, length_arcsec, n_samples)
        stacked = np.dstack(all_samples)
        with np.errstate(all='ignore'):
            avg = np.nanmean(stacked, axis=2)
        return offsets, avg

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
            self._clear_all_overlay_spatial_curves()
            self.stacked_spatial_info.setCurrentIndex(0)
            self.spatial_stats_scroll.hide()
            self.lbl_spatial_stats.setText("Draw a region to see statistics.")
            return

        roi = active_item["roi"]
        tool = active_item["tool"]

        if tool == "Point":
            self.plot_spatial_1.show()
            self.plot_spatial_1.setTitle("X Profile")
            self.plot_spatial_2.show()
        elif tool == "Line":
            self.plot_spatial_1.show()
            self.plot_spatial_1.setTitle("Spatial Profile")
            self.plot_spatial_2.hide()
        else:
            self.plot_spatial_1.hide()
            self.plot_spatial_2.hide()

        overlay_repr = self._get_overlay_reprojections_current()

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

                    self.stacked_spatial_info.setCurrentIndex(0)
                    self.spatial_stats_scroll.hide()

                    for ov_name, ov_data in overlay_repr.items():
                        ov_x_profile = ov_data[:, y_idx]
                        ov_y_profile = ov_data[x_idx, :]
                        ov_color = self._overlay_color_by_name(ov_name)

                        self._update_overlay_spatial_curve(1, ov_name, x_axis, ov_x_profile, ov_color)
                        self._update_overlay_spatial_curve(2, ov_name, y_axis, ov_y_profile, ov_color)

                    self._cleanup_stale_overlay_spatial_curves(overlay_repr.keys())
                    self._refresh_spatial_legend(1)
                    self._refresh_spatial_legend(2)

            elif tool == "Line":
                offsets, profile_2d = self.sample_cube_along_line(roi, data[np.newaxis, :, :])
                if profile_2d is not None and profile_2d.size > 0:
                    profile = profile_2d[:, 0]
                    self.curve_spatial_1.setData(offsets, profile)
                    self.curve_spatial_2.setData([], [])
                    self.plot_spatial_1.setLabel('left', f'Flux ({self.display_unit})')
                    self.plot_spatial_1.setLabel('bottom', 'Distance (arcsec)')

                    points = self.get_line_roi_points(roi)
                    if points is not None:
                        p1, p2 = points
                        dx_w = p2[0] - p1[0]
                        dy_w = p2[1] - p1[1]
                        length_arcsec = np.hypot(dx_w, dy_w)
                        n_samples = max(int(np.ceil(length_arcsec / max(self.pix_scale_arcsec, 1e-6))) + 1, 2)

                        xs = np.linspace(p1[0], p2[0], n_samples)
                        ys = np.linspace(p1[1], p2[1], n_samples)

                        x_pix, y_pix = self.world_to_pixel(xs, ys)
                        valid = (x_pix >= 0) & (x_pix <= self.nx - 1) & (y_pix >= 0) & (y_pix <= self.ny - 1)

                        for ov_name, ov_data in overlay_repr.items():
                            ov_samples = np.full(n_samples, np.nan, dtype=np.float64)
                            if np.any(valid):
                                x0 = np.floor(x_pix[valid]).astype(np.int64)
                                y0 = np.floor(y_pix[valid]).astype(np.int64)
                                x1 = np.clip(x0 + 1, 0, self.nx - 1).astype(np.int64)
                                y1 = np.clip(y0 + 1, 0, self.ny - 1).astype(np.int64)
                                fx = (x_pix[valid] - x0).astype(np.float64)
                                fy = (y_pix[valid] - y0).astype(np.float64)
                                ov_buf = np.ascontiguousarray(ov_data[np.newaxis, :, :], dtype=np.float64)
                                ov_out = np.empty((1, int(valid.sum())), dtype=np.float64)
                                _bilinear_interp(ov_buf, x0, y0, x1, y1, fx, fy, ov_out)
                                ov_samples[valid] = ov_out[0]

                            ov_color = self._overlay_color_by_name(ov_name)
                            self._update_overlay_spatial_curve(1, ov_name, offsets, ov_samples, ov_color)

                    self.curve_spatial_2.setData([], [])
                    for curve_name in list(getattr(self, 'overlay_spatial_curves_2', {}).keys()):
                        getattr(self, 'overlay_spatial_curves_2')[curve_name].setData([], [])


                    self._cleanup_stale_overlay_spatial_curves_plot1(overlay_repr.keys())
                    self._refresh_spatial_legend(1)
                    self.stacked_spatial_info.setCurrentIndex(0)
                    self.spatial_stats_scroll.hide()
                    self.lbl_spatial_stats.setText("Line profile plotted.")

            elif tool in ["Rectangle", "Ellipse"]:
                if isinstance(roi, (pg.EllipseROI, pg.PolyLineROI)):
                    sub_data = roi.getArrayRegion(data, self.view_channel.getImageItem())
                    dummy_ones = np.ones_like(data)
                    roi_mask = roi.getArrayRegion(dummy_ones, self.view_channel.getImageItem())
                    if sub_data is not None and roi_mask is not None:
                        sub_data[roi_mask == 0] = np.nan
                elif isinstance(roi, pg.RectROI):
                    sub_data = roi.getArrayRegion(data, self.view_channel.getImageItem())
                    roi_mask = None
                elif isinstance(roi, (pg.LineSegmentROI, pg.LineROI, getattr(pg, 'PointROI', type('Dummy', (), {})))):
                    sub_data = roi.getArrayRegion(data, self.view_channel.getImageItem())
                    roi_mask = None
                else:
                    sub_data = roi.getArrayRegion(data, self.view_channel.getImageItem())
                    roi_mask = None

                if sub_data is not None and sub_data.size > 0:
                    valid = sub_data[~np.isnan(sub_data)]
                    if len(valid) > 0:
                        mean_val = np.mean(valid)
                        sum_val = np.sum(valid)
                        max_val = np.max(valid)
                        min_val = np.min(valid)
                        rms_val = np.sqrt(np.mean(valid**2))
                        std_val = np.std(valid)

                        self._clear_spatial_stats_panels()

                        base_panel = self._make_stats_panel("Base Cube", "white", [
                            ("Mean", f"{mean_val:.4g} {self.display_unit}"),
                            ("Sum", f"{sum_val:.4g} {self.display_unit}"),
                            ("Peak", f"{max_val:.4g} {self.display_unit}"),
                            ("Min", f"{min_val:.4g} {self.display_unit}"),
                            ("RMS", f"{rms_val:.4g} {self.display_unit}"),
                            ("Std Dev", f"{std_val:.4g} {self.display_unit}"),
                        ])
                        self.spatial_stats_layout.addWidget(base_panel)

                        for ov_name, ov_data in overlay_repr.items():
                            ov_sub = roi.getArrayRegion(ov_data, self.view_channel.getImageItem())
                            if ov_sub is not None and roi_mask is not None:
                                ov_sub[roi_mask == 0] = np.nan
                            if ov_sub is not None and ov_sub.size > 0:
                                ov_valid = ov_sub[~np.isnan(ov_sub)]
                                if len(ov_valid) > 0:
                                    ov_mean = np.mean(ov_valid)
                                    ov_sum = np.sum(ov_valid)
                                    ov_max = np.max(ov_valid)
                                    ov_min = np.min(ov_valid)
                                    ov_rms = np.sqrt(np.mean(ov_valid**2))
                                    ov_std = np.std(ov_valid)
                                    ov_color = self._overlay_color_by_name(ov_name)
                                    ov_panel = self._make_stats_panel(ov_name, ov_color, [
                                        ("Mean", f"{ov_mean:.4g}"),
                                        ("Sum", f"{ov_sum:.4g}"),
                                        ("Peak", f"{ov_max:.4g}"),
                                        ("Min", f"{ov_min:.4g}"),
                                        ("RMS", f"{ov_rms:.4g}"),
                                        ("Std Dev", f"{ov_std:.4g}"),
                                    ])
                                    self.spatial_stats_layout.addWidget(ov_panel)

                        self.spatial_stats_layout.addStretch()
                        self.stacked_spatial_info.setCurrentIndex(1)
                        self.spatial_stats_scroll.show()
                        self.curve_spatial_1.setData([], [])
                        self.curve_spatial_2.setData([], [])
                        self._clear_all_overlay_spatial_curves()
                    else:
                        self.stacked_spatial_info.setCurrentIndex(0)
                        self.spatial_stats_scroll.hide()
                        self.lbl_spatial_stats.setText("No valid data in region.")

        except Exception as e:
            with open("line_debug.txt", "a") as dbg:
                dbg.write(f"Exception: {e}\n")
            print(f"Error in update_spatial_analysis: {e}")

    def _get_overlay_reprojections_current(self):
        result = {}
        if not self.contour_overlays or self.cube_clean is None:
            return result
        current_ch = self.slider_channel.value()
        for ov in self.contour_overlays:
            if ov.get('_reproj_raw') is not None and ov.get('_reproj_channel') == current_ch:
                result[ov['name']] = ov['_reproj_raw']
                continue
            overlay_slice = self._get_overlay_slice_for_channel(ov)
            if overlay_slice is None:
                continue
            reprojected = self._reproject_overlay_slice(ov, overlay_slice)
            if reprojected is not None:
                result[ov['name']] = reprojected
                ov['_reproj_raw'] = reprojected
                ov['_reproj_channel'] = current_ch
        return result

    def _overlay_color_by_name(self, name):
        for ov in self.contour_overlays:
            if ov['name'] == name:
                return ov['options']['color']
        return 'white'

    def _update_overlay_spatial_curve(self, plot_num, ov_name, x, y, color):
        x = np.asarray(x, dtype=float).ravel()
        y = np.asarray(y, dtype=float).ravel()
        if len(x) != len(y) or len(x) == 0:
            return
        attr_name = f'overlay_spatial_curves_{plot_num}'
        if not hasattr(self, attr_name):
            setattr(self, attr_name, {})
        curves = getattr(self, attr_name)
        if ov_name not in curves:
            c = self.plot_spatial_1.plot([], [], pen=pg.mkPen(color, width=2, style=Qt.DashLine), name=ov_name) if plot_num == 1 else \
                self.plot_spatial_2.plot([], [], pen=pg.mkPen(color, width=2, style=Qt.DashLine), name=ov_name)
            curves[ov_name] = c
        curves[ov_name].setPen(pg.mkPen(color, width=2, style=Qt.DashLine))
        curves[ov_name].setData(x, y)

    def _cleanup_stale_overlay_spatial_curves(self, active_names):
        for pnum in [1, 2]:
            attr = f'overlay_spatial_curves_{pnum}'
            if not hasattr(self, attr):
                continue
            curves = getattr(self, attr)
            plot = self.plot_spatial_1 if pnum == 1 else self.plot_spatial_2
            for name in list(curves.keys()):
                if name not in active_names:
                    c = curves.pop(name)
                    try:
                        plot.removeItem(c)
                    except Exception:
                        pass
                    c.setData([], [])

    def _cleanup_stale_overlay_spatial_curves_plot1(self, active_names):
        self._cleanup_stale_overlay_spatial_curves(active_names)
        curves = getattr(self, 'overlay_spatial_curves_2', {})
        for curve_name in list(curves.keys()):
            c = curves.pop(curve_name)
            try:
                self.plot_spatial_2.removeItem(c)
            except Exception:
                pass
            c.setData([], [])

    def _clear_all_overlay_spatial_curves(self):
        for pnum in [1, 2]:
            attr = f'overlay_spatial_curves_{pnum}'
            plot = self.plot_spatial_1 if pnum == 1 else self.plot_spatial_2
            if hasattr(self, attr):
                for c in getattr(self, attr).values():
                    try:
                        plot.removeItem(c)
                    except Exception:
                        pass
                    c.setData([], [])
                setattr(self, attr, {})

    def _refresh_spatial_legend(self, plot_num):
        plot = self.plot_spatial_1 if plot_num == 1 else self.plot_spatial_2
        if not hasattr(plot, 'plotItem') or plot.plotItem.legend is None:
            return
        legend = plot.plotItem.legend
        legend.clear()
        base_curve = self.curve_spatial_1 if plot_num == 1 else self.curve_spatial_2
        if base_curve.xData is not None and len(base_curve.xData) > 0:
            legend.addItem(base_curve, "Base")
        curves = getattr(self, f'overlay_spatial_curves_{plot_num}', {})
        for name, c in curves.items():
            if c.xData is not None and len(c.xData) > 0:
                legend.addItem(c, name)

    def _make_stats_panel(self, title, color, rows):
        panel = QWidget()
        panel.setStyleSheet("background-color: #2a2a2a; border: 1px solid #555; border-radius: 6px; padding: 0px;")
        panel.setFixedWidth(220)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(3)
        title_lbl = QLabel(title)
        title_lbl.setStyleSheet(f"font-weight: bold; color: {color}; font-size: 13px; border: none; padding: 0px; background: transparent;")
        title_lbl.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_lbl)
        layout.addSpacing(5)
        for label, value in rows:
            row_layout = QHBoxLayout()
            row_layout.setSpacing(4)
            lbl_name = QLabel(label + ":")
            lbl_name.setStyleSheet("color: #999; font-size: 11px; border: none; padding: 0px; background: transparent;")
            lbl_val = QLabel(value)
            lbl_val.setStyleSheet("color: #eee; font-size: 11px; font-weight: bold; border: none; padding: 0px; background: transparent;")
            lbl_val.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            row_layout.addWidget(lbl_name)
            row_layout.addWidget(lbl_val, stretch=1)
            layout.addLayout(row_layout)
        return panel

    def _clear_spatial_stats_panels(self):
        while self.spatial_stats_layout.count():
            item = self.spatial_stats_layout.takeAt(0)
            w = item.widget()
            if w:
                w.hide()
                w.setParent(None)
                w.deleteLater()


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
        
        if checked:
            self.plot_channel.vb.setCursor(Qt.CrossCursor)
        else:
            self.plot_channel.vb.setCursor(Qt.ArrowCursor)

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

        if source_plot in (self.plot_widget, getattr(self, 'plot_widget_smooth', None)):
            if event.button() == Qt.LeftButton:
                if event.modifiers() == Qt.NoModifier:
                    idx = (np.abs(self.v_axis - mp.x())).argmin()
                    self.slider_channel.setValue(idx)
                elif event.modifiers() == Qt.ControlModifier:
                    hit = False
                    if hasattr(self, 'spectrum_rois') and self.spectrum_rois:
                        for item in self.spectrum_rois:
                            roi = item["roi"]
                            r_pos = roi.pos()
                            r_size = roi.size()
                            min_x = min(r_pos.x(), r_pos.x() + r_size.x())
                            max_x = max(r_pos.x(), r_pos.x() + r_size.x())
                            if min_x <= mp.x() <= max_x:
                                self.select_region_for_deletion(roi)
                                hit = True
                                break
                    if hit:
                        event.accept()
            return

        if self.active_picker_panel is not None:
            if event.button() == Qt.LeftButton and source_plot == self.plot_channel:
                self.plot_channel.vb.setCursor(Qt.ArrowCursor)
                self.delete_nr_roi()
                
                w = 10.0 * abs(self.pix_scale_arcsec)
                h = 10.0 * abs(self.pix_scale_arcsec)
                
                self.nr_roi = pg.RectROI([mp.x() - w/2, mp.y() - h/2], [w, h], pen=pg.mkPen('#00FFFF', width=2))
                self.nr_roi.addScaleHandle([0, 0], [1, 1]); self.nr_roi.addScaleHandle([1, 1], [0, 0])
                self.nr_roi.addScaleHandle([0, 1], [1, 0]); self.nr_roi.addScaleHandle([1, 0], [0, 1])
                self.nr_roi.addScaleHandle([0.5, 0], [0.5, 1]); self.nr_roi.addScaleHandle([0.5, 1], [0.5, 0])
                self.nr_roi.addScaleHandle([0, 0.5], [1, 0.5]); self.nr_roi.addScaleHandle([1, 0.5], [0, 0.5])
                
                make_roi_rotatable_with_ctrl(self.nr_roi)
                    
                self.nr_label = pg.TextItem("NR", color='#00FFFF', anchor=(0, 1))
                self.nr_label.setParentItem(self.nr_roi)
                
                self.view_channel.getView().addItem(self.nr_roi)
                
                self.nr_roi.target_panel_id = self.active_picker_panel
                self.panels[self.active_picker_panel]['btn_pick'].setChecked(False)
                self.active_picker_panel = None
                
                self.nr_roi.sigRegionChangeFinished.connect(self.update_nr_rms)
                self.update_nr_rms()
            return 
            
        if source_plot == self.plot_channel:
            hit = False
            
            if getattr(self, 'nr_roi', None) is not None:
                r_pos = self.nr_roi.pos()
                r_size = self.nr_roi.size()
                min_x = min(r_pos.x(), r_pos.x() + r_size.x())
                max_x = max(r_pos.x(), r_pos.x() + r_size.x())
                min_y = min(r_pos.y(), r_pos.y() + r_size.y())
                max_y = max(r_pos.y(), r_pos.y() + r_size.y())
                
                if min_x <= mp.x() <= max_x and min_y <= mp.y() <= max_y:
                    self.nr_roi.setPen(pg.mkPen('y', width=3))
                    self.nr_roi_selected = True
                    hit = True
                    event.accept()
                    return
                    
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
                    if hasattr(roi, 'shape'):
                        is_clicked = roi.shape().contains(roi.mapFromScene(event.scenePos()))
                    elif isinstance(roi, pg.LineSegmentROI):
                        is_clicked = self.line_roi_hit_test(roi, event.scenePos())
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
            self.current_file_name = file_name
            self.is_2d_image = False
            try:
                sc = SpectralCube.read(file_name).with_spectral_unit(u.km / u.s, velocity_convention='radio')
            except Exception as e:
                with fits.open(file_name) as hdul:
                    data = np.squeeze(hdul[0].data)
                    if data.ndim == 2:
                        self.is_2d_image = True
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
            
            raw_cube = sc.filled_data[:].value
            self.v_axis = sc.spectral_axis.value
            
            self.cube_clean = np.transpose(raw_cube, (0, 2, 1))
            self.nx, self.ny = self.cube_clean.shape[1], self.cube_clean.shape[2]
            
            # Multi-Beam Header Parsing
            self.bmaj_array = None
            self.bmin_array = None
            self.bpa_array = None
            self.pixels_per_beam_array = None
            self.beam_omega_array = None
            self.freq_array = None
            self.can_convert_units = True
            
            # Extract frequency array
            try:
                if sc.header.get('CTYPE3', '').startswith('FREQ'):
                    freq_crval = sc.header.get('CRVAL3')
                    freq_cdelt = sc.header.get('CDELT3')
                    freq_crpix = sc.header.get('CRPIX3')
                    self.freq_array = freq_crval + (np.arange(len(self.v_axis)) - (freq_crpix - 1)) * freq_cdelt
                elif sc.header.get('RESTFRQ') or sc.header.get('RESTFREQ'):
                    rf = sc.header.get('RESTFRQ', sc.header.get('RESTFREQ'))
                    self.freq_array = rf * (1.0 - (self.v_axis * 1000.0) / const.c.value)
                else:
                    self.can_convert_units = False
            except Exception:
                self.can_convert_units = False

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
                        
                        if len(bmaj_raw) == len(self.v_axis):
                            bmaj_unit = hdul['BEAMS'].columns['BMAJ'].unit
                            if bmaj_unit and 'deg' in str(bmaj_unit).lower():
                                self.bmaj_array = bmaj_raw
                                self.bmin_array = bmin_raw
                            else:
                                self.bmaj_array = bmaj_raw / 3600.0
                                self.bmin_array = bmin_raw / 3600.0
                            if bpa_raw is not None:
                                self.bpa_array = bpa_raw
                            else:
                                bpa = sc.header.get('BPA', 0.0)
                                self.bpa_array = np.full(len(self.v_axis), bpa)
                        else:
                            bmaj = sc.header.get('BMAJ')
                            bmin = sc.header.get('BMIN')
                            if bmaj and bmin:
                                self.bmaj_array = np.full(len(self.v_axis), bmaj)
                                self.bmin_array = np.full(len(self.v_axis), bmin)
                            else:
                                self.can_convert_units = False
                            bpa = sc.header.get('BPA', 0.0)
                            self.bpa_array = np.full(len(self.v_axis), bpa)
                    else:
                        bmaj = sc.header.get('BMAJ')
                        bmin = sc.header.get('BMIN')
                        if bmaj and bmin:
                            self.bmaj_array = np.full(len(self.v_axis), bmaj)
                            self.bmin_array = np.full(len(self.v_axis), bmin)
                            bpa = sc.header.get('BPA', 0.0)
                            self.bpa_array = np.full(len(self.v_axis), bpa)
                        else:
                            self.can_convert_units = False
            except Exception:
                bmaj = sc.header.get('BMAJ')
                bmin = sc.header.get('BMIN')
                if bmaj and bmin:
                    self.bmaj_array = np.full(len(self.v_axis), bmaj)
                    self.bmin_array = np.full(len(self.v_axis), bmin)
                    bpa = sc.header.get('BPA', 0.0)
                    self.bpa_array = np.full(len(self.v_axis), bpa)
                else:
                    self.can_convert_units = False
                    
            if self.bmaj_array is not None and self.bmin_array is not None and cdelt1 and cdelt2:
                omega_pix = abs(cdelt1 * cdelt2) * (u.deg ** 2)
                self.omega_pix_sr = omega_pix.to(u.sr)
                
                omega_beam = (np.pi * self.bmaj_array * self.bmin_array) / (4.0 * np.log(2.0)) * (u.deg ** 2)
                self.omega_beam_sr = omega_beam.to(u.sr)
                
                self.n_beam_array = self.omega_beam_sr / self.omega_pix_sr
                self.pixels_per_beam = self.n_beam_array[0].value
            else:
                self.omega_pix_sr = 1.0 * u.sr
                self.omega_beam_sr = np.ones(len(self.v_axis)) * u.sr
                self.n_beam_array = np.ones(len(self.v_axis)) * u.dimensionless_unscaled
                self.pixels_per_beam = 1.0
                self.can_convert_units = False
            
            native_label = f"Native ({raw_bunit})" if raw_bunit != 'Unknown' else "Native"
            unit_lower = raw_bunit.replace(" ", "").lower()
            
            new_units = [native_label, "Jy"]
            if not ("k" == unit_lower or "kelvin" in unit_lower):
                new_units.append("K")
            if not ("jy" in unit_lower and "pixel" not in unit_lower and "pix" not in unit_lower):
                new_units.append("Jy/beam")
                
            self.combo_spec_unit.blockSignals(True)
            self.combo_spec_unit.clear()
            self.combo_spec_unit.addItems(new_units)
            self.combo_spec_unit.blockSignals(False)
            
            if not self.can_convert_units:
                self.combo_spec_unit.blockSignals(True)
                self.combo_spec_unit.setCurrentIndex(0)
                self.combo_spec_unit.blockSignals(False)
                
                for i in range(1, self.combo_spec_unit.count()):
                    self.combo_spec_unit.model().item(i).setEnabled(False)
                self.combo_spec_unit.setToolTip("Conversion disabled: Missing beam or frequency metadata in FITS.")
                
                sum_idx = self.combo_spec_stat.findText("Flux Density")
                if sum_idx != -1:
                    self.combo_spec_stat.model().item(sum_idx).setEnabled(False)
            else:
                self.combo_spec_unit.setToolTip("")
                sum_idx = self.combo_spec_stat.findText("Flux Density")
                if sum_idx != -1:
                    self.combo_spec_stat.model().item(sum_idx).setEnabled(True)
            
            self.combo_spec_stat.blockSignals(True)
            self.combo_spec_stat.setCurrentText("Mean" if not self.can_convert_units else "Mean")
            self.combo_spec_stat.blockSignals(False)
            self._update_spectrum_state_machine()

            self.plot_widget.setLabel('left', f'Mean Flux ({self.display_unit})')
            self.view_channel.ui.histogram.axis.setLabel(f"Flux ({self.display_unit})")
            
            peak_flux = np.nanmax(self.cube_clean)
            self.ch_levels = (0, peak_flux if peak_flux > 0 else 1.0)

            self.slider_channel.setRange(0, len(self.v_axis) - 1)
            mean_spectrum = np.nanmean(self.cube_clean, axis=(1, 2))
            brightest_ch = int(np.nanargmax(mean_spectrum))
            self.slider_channel.setValue(brightest_ch)
            self.update_channel_map()
            self.v_line.show()

            v_min, v_max = np.nanmin(self.v_axis), np.nanmax(self.v_axis)
            peak_vel = self.v_axis[brightest_ch]
            half_span = 0.1 * (v_max - v_min)
            r_lo = max(v_min, peak_vel - half_span)
            r_hi = min(v_max, peak_vel + half_span)
            if r_hi - r_lo < 0.02 * (v_max - v_min):
                r_lo = v_min + 0.4 * (v_max - v_min)
                r_hi = v_min + 0.6 * (v_max - v_min)
            self.region.setRegion([r_lo, r_hi])
            self.region.show()
            self.combo_roi.blockSignals(True)
            self.combo_roi.setCurrentText("Whole Map")
            self.combo_roi.blockSignals(False)
            self.roi_selected = False
            self.change_roi("Whole Map")

            self.combo_spec_stat.blockSignals(True)
            self.combo_spec_stat.setCurrentText("Mean")
            self.combo_spec_stat.blockSignals(False)
            self.smoothing_params = None
            if hasattr(self, 'spectrum_tabs'):
                idx = self.spectrum_tabs.indexOf(self.plot_widget_smooth)
                if idx != -1:
                    self.spectrum_tabs.removeTab(idx)
                self.spectrum_tabs.tabBar().hide()
                self.spectrum_tabs.setCurrentWidget(self.plot_widget)
            self.spectrum_curve_smooth.setData([], [])
            for p in self.panels:
                p['input_thresh'].setText("0.000")

            self.update_moment_maps()
            
            self.update_wcs_mode(self.parent_window.is_absolute_wcs)
            self.set_2d_ui_state(self.is_2d_image)
            self.parent_window.update_menu_states()
            
            return True
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load cube:\n{str(e)}")
            return False

    def set_2d_ui_state(self, is_2d):
        enable = not is_2d
        
        for btn in [self.btn_prev, self.btn_play_rev, self.btn_stop, self.btn_play_fwd, self.btn_next]:
            btn.setEnabled(enable)
            
        self.slider_channel.setEnabled(enable)
        self.input_channel_vel.setEnabled(enable)
        self.btn_grid.setEnabled(enable)
        self.toggle_bottom_btn.setEnabled(enable)
        
        if is_2d and getattr(self, 'bottom_container', None) is not None and self.bottom_container.isVisible():
            if "Hide" in self.toggle_bottom_btn.text():
                self.toggle_bottom_pane()
            
        model = self.combo_panel_mode.model()
        spectrum_index = self.combo_panel_mode.findText("Spectrum")
        if spectrum_index >= 0:
            item = model.item(spectrum_index)
            if item:
                item.setEnabled(enable)
                
        if is_2d:
            self.combo_panel_mode.setCurrentText("Spatial Analysis")

    def close_file(self):
        self.set_2d_ui_state(False)
        self.is_2d_image = False
        self.cube_clean = None
        self.fits_header_text = ""
        self.raw_header = None
        self.wcs_2d = None
        self.rest_freq_hz = None
        
        if hasattr(self, 'beam_visualizer_items'):
            for key, items_list in self.beam_visualizer_items.items():
                for item in items_list:
                    try:
                        if item.scene():
                            item.scene().removeItem(item)
                    except Exception: pass
            self.beam_visualizer_items.clear()

        self.is_drawing_polygon = False
        self.polygon_points = []
        if self.polygon_preview_line is not None:
            try:
                self.plot_channel.removeItem(self.polygon_preview_line)
            except Exception:
                pass
            self.polygon_preview_line = None
        self.roi_selected = False

        for item in getattr(self, 'spatial_rois', []):
            di = item.get("direction_item")
            if di is not None:
                try:
                    dscene = di.scene()
                    if dscene is not None: dscene.removeItem(di)
                    else: self.plot_channel.removeItem(di)
                except Exception: pass
                di.setData([], [])
            if "update_spatial_arrow" in item and item.get("roi") is not None:
                try: item["roi"].sigRegionChanged.disconnect(item["update_spatial_arrow"])
                except Exception: pass
            ti = item.get("text_item")
            if ti is not None:
                try:
                    tscene = ti.scene()
                    if tscene is not None: tscene.removeItem(ti)
                    else: self.plot_channel.removeItem(ti)
                except Exception: pass
            if "update_spatial_label" in item and item.get("roi") is not None:
                try: item["roi"].sigRegionChanged.disconnect(item["update_spatial_label"])
                except Exception: pass
            roi = item['roi']
            try: roi.sigRegionChanged.disconnect()
            except Exception: pass
            s = roi.scene()
            if s is not None:
                try: s.removeItem(roi)
                except Exception: pass
        self.spatial_rois = []
        self.spatial_rois_to_delete = []
        self.combo_spatial_regions.blockSignals(True)
        self.combo_spatial_regions.clear()
        self.combo_spatial_regions.addItem("None")
        self.combo_spatial_regions.blockSignals(False)

        for r_dict in list(getattr(self, 'spectrum_spatial_rois', [])):
            roi = r_dict["roi"]
            try:
                if roi.scene():
                    roi.scene().removeItem(roi)
                else:
                    self.view_channel.getView().removeItem(roi)
            except Exception: pass
            cb = r_dict.get("checkbox")
            if cb is not None:
                try: self.box_regions_layout.removeWidget(cb)
                except Exception: pass
                cb.deleteLater()
            name = r_dict.get("name", "")
            if name in getattr(self, 'spectrum_curves', {}):
                c = self.spectrum_curves.pop(name)
                try:
                    if c.scene(): c.scene().removeItem(c)
                    else: self.plot_widget.removeItem(c)
                except Exception: pass
            if name in getattr(self, 'spectrum_curves_smooth', {}):
                c = self.spectrum_curves_smooth.pop(name)
                try:
                    if c.scene(): c.scene().removeItem(c)
                    else: self.plot_widget_smooth.removeItem(c)
                except Exception: pass
        self.spectrum_spatial_rois = []

        active_rois = getattr(self, 'get_active_spectrum_rois', lambda: [])()
        for item in list(active_rois):
            roi = item["roi"]
            try:
                if roi.scene(): roi.scene().removeItem(roi)
                else: self.plot_widget.removeItem(roi)
            except Exception: pass
            ti = item.get("text_item")
            if ti is not None:
                try:
                    if ti.scene(): ti.scene().removeItem(ti)
                    else: self.plot_widget.removeItem(ti)
                except Exception: pass
        if hasattr(self, '_spectrum_regions'):
            self._spectrum_regions = []
        self.rois_to_delete.clear()

        self.combo_regions.blockSignals(True)
        self.combo_regions_2.blockSignals(True)
        self.combo_regions_3.blockSignals(True)
        self.combo_regions.clear()
        self.combo_regions_2.clear()
        self.combo_regions_3.clear()
        self.combo_regions.addItem("None")
        self.combo_regions_2.addItem("None")
        self.combo_regions_3.addItem("None")
        self.combo_regions.blockSignals(False)
        self.combo_regions_2.blockSignals(False)
        self.combo_regions_3.blockSignals(False)

        for name in list(getattr(self, 'spectrum_curves', {}).keys()):
            c = self.spectrum_curves.pop(name)
            try:
                if c.scene(): c.scene().removeItem(c)
                else: self.plot_widget.removeItem(c)
            except Exception: pass
        for name in list(getattr(self, 'spectrum_curves_smooth', {}).keys()):
            c = self.spectrum_curves_smooth.pop(name)
            try:
                if c.scene(): c.scene().removeItem(c)
                else: self.plot_widget_smooth.removeItem(c)
            except Exception: pass

        for cut_info in list(getattr(self, 'pv_cuts', [])):
            roi = cut_info['roi']
            try:
                roi.removeHandle(0)
            except Exception:
                pass
            try:
                roi.removeHandle(1)
            except Exception:
                pass
            try:
                roi.sigRegionChanged.disconnect()
            except Exception:
                pass
            s = roi.scene()
            if s is not None:
                try:
                    s.removeItem(roi)
                except Exception:
                    pass
            text_item = cut_info.get('text_item')
            if text_item is not None:
                try:
                    tscene = text_item.scene()
                    if tscene is not None:
                        tscene.removeItem(text_item)
                    else:
                        self.plot_channel.removeItem(text_item)
                except Exception:
                    pass
            direction_item = cut_info.get('direction_item')
            if direction_item is not None:
                try:
                    dscene = direction_item.scene()
                    if dscene is not None:
                        dscene.removeItem(direction_item)
                    else:
                        self.plot_channel.removeItem(direction_item)
                except Exception:
                    pass
            width_item = cut_info.get('width_item')
            if width_item is not None:
                try:
                    wscene = width_item.scene()
                    if wscene is not None:
                        wscene.removeItem(width_item)
                    else:
                        self.plot_channel.removeItem(width_item)
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
            self.stacked_spatial_info.setCurrentIndex(0)
            self.spatial_stats_scroll.hide()
            self.lbl_spatial_stats.setText("Draw a region to see statistics.")

        self.lbl_region_result.setText("---")

        self.input_channel_vel.setText("")
        self.input_vmin.setText("0.00")
        self.input_vmax.setText("1.00")
        self.combo_spec_stat.blockSignals(True)
        self.combo_spec_stat.setCurrentText("Mean")
        self.combo_spec_stat.blockSignals(False)
        self.smoothing_params = None
        self.spectrum_curve_smooth.setData([], [])
        if hasattr(self, 'spectrum_tabs'):
            idx = self.spectrum_tabs.indexOf(self.plot_widget_smooth)
            if idx != -1:
                self.spectrum_tabs.removeTab(idx)
            self.spectrum_tabs.tabBar().hide()
            self.spectrum_tabs.setCurrentWidget(self.plot_widget)
        for p in self.panels:
            p['input_thresh'].setText("0.000")

        self.view_channel.getImageItem().clear()
        self.spectrum_curve.setData([], [])
        self.plot_widget.setLabel('left', 'Flux')
        if hasattr(self, 'plot_widget_smooth'):
            self.plot_widget_smooth.setLabel('left', 'Flux')

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
        self._clear_overlay_contours()
        for name in list(self.overlay_spectrum_curves.keys()):
            c = self.overlay_spectrum_curves.pop(name)
            try:
                if c.scene(): c.scene().removeItem(c)
                else: self.plot_widget.removeItem(c)
            except Exception: pass
        for name in list(self.overlay_spectrum_curves_smooth.keys()):
            c = self.overlay_spectrum_curves_smooth.pop(name)
            try:
                if c.scene(): c.scene().removeItem(c)
                else: self.plot_widget_smooth.removeItem(c)
            except Exception: pass
        self.contour_overlays = []
        self._clear_all_overlay_spatial_curves()
        if hasattr(self.plot_widget, 'plotItem') and self.plot_widget.plotItem.legend is not None:
            self.plot_widget.plotItem.legend.clear()
        if hasattr(self, 'plot_widget_smooth') and hasattr(self.plot_widget_smooth, 'plotItem') and self.plot_widget_smooth.plotItem.legend is not None:
            self.plot_widget_smooth.plotItem.legend.clear()
        self.btn_contour_overlay.setEnabled(False)
        self.btn_contour_overlay.hide()
        self.parent_window.update_menu_states()

    def clear_roi(self):
        self.delete_nr_roi()
        for item in getattr(self, 'spatial_rois', []):
            if "update_spatial_label" in item and getattr(self, 'plot_channel', None) is not None:
                try: item["roi"].sigRegionChanged.disconnect(item["update_spatial_label"])
                except Exception: pass
            roi = item['roi']
            try: roi.sigRegionChanged.disconnect()
            except Exception: pass
            s = roi.scene()
            if s is not None:
                try: s.removeItem(roi)
                except Exception: pass
        self.spatial_rois = []
        self.spatial_rois_to_delete = []
        self.combo_spatial_regions.blockSignals(True)
        self.combo_spatial_regions.clear()
        self.combo_spatial_regions.addItem("None")
        self.combo_spatial_regions.blockSignals(False)
        self.lbl_region_result.setText("---")

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

    def load_overlay_file(self, file_name):
        try:
            overlay_cube = None
            overlay_header = None
            overlay_v = None
            is_static = False
            static_2d = None
            is_2d_image = False
            sc = None

            try:
                sc = SpectralCube.read(file_name).with_spectral_unit(u.km / u.s, velocity_convention='radio')
                overlay_cube = sc.filled_data[:].value
                overlay_cube = np.transpose(overlay_cube, (0, 2, 1))
                overlay_header = sc.header
            except (ValueError, Exception):
                hdul = fits.open(file_name)
                data_2d = hdul[0].data
                if data_2d is None or data_2d.ndim < 2:
                    hdul.close()
                    QMessageBox.critical(self, "Format Error",
                        "The selected FITS file does not contain 2D or 3D image data.")
                    return False
                if data_2d.ndim == 2:
                    is_2d_image = True
                    overlay_cube = np.transpose(data_2d, (1, 0))[np.newaxis, :, :]
                    overlay_header = hdul[0].header
                    is_static = True
                    static_2d = overlay_cube[0]
                elif data_2d.ndim == 3:
                    overlay_cube = np.transpose(data_2d, (0, 2, 1))
                    overlay_header = hdul[0].header
                    nv_o = overlay_cube.shape[0]
                    if nv_o == 1:
                        is_static = True
                        static_2d = overlay_cube[0]
                    else:
                        is_static = True
                        static_2d = overlay_cube[0]
                        QMessageBox.information(self, "Spectral Axis Missing",
                            "This FITS file has multiple channels but no spectral axis "
                            "information. Using the first channel as a static 2D overlay.")
                else:
                    hdul.close()
                    QMessageBox.critical(self, "Format Error",
                        "Unexpected data dimensions. Expected 2D or 3D.")
                    return False
                hdul.close()

            try:
                overlay_wcs = WCS(overlay_header).celestial
            except Exception:
                QMessageBox.critical(self, "WCS Error",
                    "The selected FITS file does not contain valid spatial WCS information.")
                return False

            primary_wcs = self.wcs_2d
            if primary_wcs is None:
                QMessageBox.critical(self, "WCS Error",
                    "The primary cube has no valid WCS. Please reload it.")
                return False

            nv_o = overlay_cube.shape[0]
            ny_o = overlay_cube.shape[1]
            nx_o = overlay_cube.shape[2]

            if nv_o == 1:
                is_static = True
                static_2d = overlay_cube[0]
            elif not is_2d_image and not is_static:
                try:
                    overlay_v = sc.spectral_axis.value
                except Exception:
                    pass
                if overlay_v is None and nv_o > 1:
                    is_static = True
                    static_2d = overlay_cube[0]

            cdelt2_o = overlay_header.get('CDELT2', None)
            cdelt1_o = overlay_header.get('CDELT1', None)
            pix_scale_o = abs(float(cdelt2_o)) * 3600.0 if cdelt2_o else 1.0
            
            bmaj_array_o = None
            bmin_array_o = None
            freq_array_o = None
            try:
                if overlay_header.get('CTYPE3', '').startswith('FREQ'):
                    freq_crval = overlay_header.get('CRVAL3')
                    freq_cdelt = overlay_header.get('CDELT3')
                    freq_crpix = overlay_header.get('CRPIX3')
                    freq_array_o = freq_crval + (np.arange(nv_o) - (freq_crpix - 1)) * freq_cdelt
                else:
                    rf = overlay_header.get('RESTFRQ', overlay_header.get('RESTFREQ'))
                    if rf and overlay_v is not None:
                        freq_array_o = rf * (1.0 - (overlay_v * 1000.0) / const.c.value)

                with fits.open(file_name) as hdul:
                    is_mb = overlay_header.get('CASAMBM', 'F') == 'T' or 'BEAMS' in hdul
                    if is_mb and 'BEAMS' in hdul:
                        b_data = hdul['BEAMS'].data
                        bmaj_raw = b_data['BMAJ']
                        bmin_raw = b_data['BMIN']
                        if len(bmaj_raw) == nv_o:
                            bmaj_unit = hdul['BEAMS'].columns['BMAJ'].unit
                            if bmaj_unit and 'deg' in str(bmaj_unit).lower():
                                bmaj_array_o = bmaj_raw
                                bmin_array_o = bmin_raw
                            else:
                                bmaj_array_o = bmaj_raw / 3600.0
                                bmin_array_o = bmin_raw / 3600.0
                        else:
                            bmaj = overlay_header.get('BMAJ')
                            bmin = overlay_header.get('BMIN')
                            if bmaj and bmin:
                                bmaj_array_o = np.full(nv_o, bmaj)
                                bmin_array_o = np.full(nv_o, bmin)
                    else:
                        bmaj = overlay_header.get('BMAJ')
                        bmin = overlay_header.get('BMIN')
                        if bmaj and bmin:
                            bmaj_array_o = np.full(nv_o, bmaj)
                            bmin_array_o = np.full(nv_o, bmin)
            except Exception:
                bmaj = overlay_header.get('BMAJ')
                bmin = overlay_header.get('BMIN')
                if bmaj and bmin:
                    bmaj_array_o = np.full(nv_o, bmaj)
                    bmin_array_o = np.full(nv_o, bmin)

            corners_px = np.array([[0, 0], [self.nx - 1, 0], [0, self.ny - 1], [self.nx - 1, self.ny - 1]], dtype=float)
            ra_primary, dec_primary = primary_wcs.wcs_pix2world(corners_px, 0).T
            ra_min_p, ra_max_p = ra_primary.min(), ra_primary.max()
            dec_min_p, dec_max_p = dec_primary.min(), dec_primary.max()

            corners_o_px = np.array([[0, 0], [nx_o - 1, 0], [0, ny_o - 1], [nx_o - 1, ny_o - 1]], dtype=float)
            ra_overlay, dec_overlay = overlay_wcs.wcs_pix2world(corners_o_px, 0).T
            ra_min_o, ra_max_o = ra_overlay.min(), ra_overlay.max()
            dec_min_o, dec_max_o = dec_overlay.min(), dec_overlay.max()

            ra_overlap = (ra_min_p <= ra_max_o and ra_max_p >= ra_min_o)
            dec_overlap = (dec_min_p <= dec_max_o and dec_max_p >= dec_min_o)
            if not ra_overlap or not dec_overlap:
                QMessageBox.critical(self, "No Overlap",
                    "No spatial overlap between the two cubes.\n"
                    "The contour file covers a different region of the sky.")
                return False

            if not is_static and overlay_v is not None:
                v_min_p, v_max_p = np.nanmin(self.v_axis), np.nanmax(self.v_axis)
                v_min_o, v_max_o = np.nanmin(overlay_v), np.nanmax(overlay_v)
                if v_max_p < v_min_o or v_max_o < v_min_p:
                    QMessageBox.critical(self, "No Overlap",
                        "No spectral overlap between the two cubes.\n"
                        "The contour cube covers a different velocity range.")
                    return False

            color_idx = len(self.contour_overlays) % len(self.overlay_color_palette)
            assigned_color = self.overlay_color_palette[color_idx]

            default_opts = {
                'mode': 'percent',
                'rms': 0.001,
                'multipliers_str': '3, 5, 10, 20, 40',
                'lin_min': 0.0,
                'lin_max': 10.0,
                'n_levels': 5,
                'log_min': 0.001,
                'log_max': 10.0,
                'log_base': 10.0,
                'percentages_str': '10, 30, 50, 70, 90',
                'color': assigned_color,
                'line_width': 1.5,
                'line_style': 'solid',
                'smooth': False,
                'smooth_kernel': 3,
            }
            overlay_name = file_name.split('/')[-1]
            if len(overlay_name) > 20:
                overlay_name = overlay_name[:17] + '...'

            overlay_dict = {
                'file': file_name,
                'name': overlay_name,
                'cube': overlay_cube,
                'wcs': overlay_wcs,
                'v_axis': overlay_v,
                'is_static': is_static,
                '2d': static_2d,
                'nx': nx_o,
                'ny': ny_o,
                'pix_scale': pix_scale_o,
                'cdelt1': cdelt1_o,
                'cdelt2': cdelt2_o,
                'bmaj_array': bmaj_array_o,
                'bmin_array': bmin_array_o,
                'freq_array': freq_array_o,
                'display_unit': overlay_header.get('BUNIT', 'Unknown').strip(),
                'iso_items': [],
                'reproject_cache': None,
                'options': default_opts.copy(),
                'color': assigned_color,
            }

            self._update_contour_options_rms_for(overlay_dict)
            self.contour_overlays.append(overlay_dict)

            self.btn_contour_overlay.setEnabled(True)
            self.btn_contour_overlay.show()
            self.update_channel_map()
            self.update_spectrum()

            v_info = "static" if is_static else f"{nv_o} channels"
            self.parent_window.statusBar().showMessage(
                f"Overlay {len(self.contour_overlays)} loaded: {overlay_name} ({v_info}, {assigned_color})")
            return True

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load overlay file:\n{str(e)}")
            return False

    def _make_default_overlay_options(self, color):
        return {
            'mode': 'percent', 'rms': 0.001,
            'multipliers_str': '3, 5, 10, 20, 40',
            'lin_min': 0.0, 'lin_max': 10.0, 'n_levels': 5,
            'log_min': 0.001, 'log_max': 10.0, 'log_base': 10.0,
            'percentages_str': '10, 30, 50, 70, 90',
            'color': color, 'line_width': 1.5, 'line_style': 'solid',
            'smooth': False, 'smooth_kernel': 3,
        }

    def _update_contour_options_rms_for(self, overlay_dict):
        slice_data = self._get_overlay_slice_for_channel(overlay_dict)
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

    def _get_overlay_slice_for_channel(self, overlay_dict):
        if overlay_dict is None:
            return None
        if overlay_dict['is_static']:
            return overlay_dict['2d']

        idx = self.slider_channel.value()
        if self.v_axis is None or idx >= len(self.v_axis):
            return None
        target_v = self.v_axis[idx]

        overlay_v = overlay_dict['v_axis']
        if overlay_v is None:
            return None

        if target_v < np.nanmin(overlay_v) or target_v > np.nanmax(overlay_v):
            return None

        nearest_idx = int(np.argmin(np.abs(overlay_v - target_v)))
        if nearest_idx < 0 or nearest_idx >= overlay_dict['cube'].shape[0]:
            return None
        return overlay_dict['cube'][nearest_idx]

    def _get_overlay_slice_for_current_channel(self):
        if self.contour_overlay_cube is None:
            return None
        return self._get_overlay_slice_for_channel({
            'cube': self.contour_overlay_cube,
            'is_static': self.contour_overlay_is_static,
            '2d': self.contour_overlay_2d,
            'v_axis': self.contour_overlay_v_axis,
        })

    def _reproject_overlay_slice(self, overlay_dict, overlay_slice):
        primary_wcs = self.wcs_2d
        overlay_wcs = overlay_dict['wcs']
        if primary_wcs is None or overlay_wcs is None or overlay_slice is None:
            return None

        cache = overlay_dict.get('reproject_cache')
        if cache is None:
            px = np.arange(self.nx, dtype=float)
            py = np.arange(self.ny, dtype=float)
            PX, PY = np.meshgrid(px, py, indexing='ij')
            pts_pix = np.column_stack([PX.ravel(), PY.ravel()])

            ra_arr, dec_arr = primary_wcs.wcs_pix2world(pts_pix, 0).T
            ox, oy = overlay_wcs.wcs_world2pix(np.column_stack([ra_arr, dec_arr]), 0).T

            ox = ox.reshape(self.nx, self.ny)
            oy = oy.reshape(self.nx, self.ny)

            valid = (ox >= 0) & (ox < overlay_dict['nx'] - 1) & (oy >= 0) & (oy < overlay_dict['ny'] - 1)
            if not np.any(valid):
                result = np.full((self.nx, self.ny), np.nan, dtype=np.float64)
                cache = {'valid': valid, 'x0': None, 'y0': None, 'x1': None, 'y1': None, 'fx': None, 'fy': None}
                overlay_dict['reproject_cache'] = cache
                return result

            vx = ox[valid]
            vy = oy[valid]
            x0 = np.floor(vx).astype(np.int64)
            y0 = np.floor(vy).astype(np.int64)
            x1 = np.clip(x0 + 1, 0, overlay_dict['nx'] - 1).astype(np.int64)
            y1 = np.clip(y0 + 1, 0, overlay_dict['ny'] - 1).astype(np.int64)
            fx = (vx - x0.astype(float)).astype(np.float64)
            fy = (vy - y0.astype(float)).astype(np.float64)

            cache = {'valid': valid, 'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1, 'fx': fx, 'fy': fy}
            overlay_dict['reproject_cache'] = cache
        elif cache.get('x0') is None:
            return np.full((self.nx, self.ny), np.nan, dtype=np.float64)

        result = np.full((self.nx, self.ny), np.nan, dtype=np.float64)
        valid = cache['valid']
        x0 = cache['x0']
        y0 = cache['y0']
        x1 = cache['x1']
        y1 = cache['y1']
        fx = cache['fx']
        fy = cache['fy']

        vals = ((1.0 - fx) * (1.0 - fy) * overlay_slice[x0, y0]
                + fx * (1.0 - fy) * overlay_slice[x1, y0]
                + (1.0 - fx) * fy * overlay_slice[x0, y1]
                + fx * fy * overlay_slice[x1, y1])
        result[valid] = vals
        return result

    def _compute_contour_levels(self, data, options, target_id=None):
        if data is None:
            return []
        valid = data[np.isfinite(data)]
        if len(valid) == 0:
            return []
            
        mode = options.get('mode', 'rms')
        lo = float(np.nanmin(valid))
        hi = float(np.nanmax(valid))
        
        is_vel = False
        if target_id is not None and target_id != 'channel':
            try:
                mtype = self.panels[target_id]['combo'].currentText()
                if "Moment 1" in mtype or "Moment 9" in mtype:
                    is_vel = True
            except:
                pass

        if mode == 'rms':
            rms = options.get('rms', 0.001)
            mean_val = float(np.nanmean(valid))
            try:
                mults = [float(x.strip()) for x in options.get('multipliers_str', '3,5,10,20,40').split(',') if x.strip()]
            except ValueError:
                mults = [3, 5, 10, 20]
            levels = [mean_val + m * rms for m in mults]
        elif mode == 'linear':
            u_lo = options.get('lin_min', lo)
            u_hi = options.get('lin_max', hi)
            n = max(int(options.get('n_levels', 5)), 1)
            levels = np.linspace(u_lo, u_hi, n).tolist()
        elif mode == 'log':
            pos_data = valid[valid > 0]
            if len(pos_data) == 0 or is_vel:
                n = max(int(options.get('n_levels', 5)), 1)
                levels = np.linspace(lo, hi, n).tolist()
            else:
                min_pos = np.nanmin(pos_data)
                n = max(int(options.get('n_levels', 5)), 1)
                base = options.get('log_base', 10.0)
                log_lo = np.log(min_pos) / np.log(base)
                log_hi = np.log(hi) / np.log(base)
                levels = np.logspace(log_lo, log_hi, n, base=base).tolist()
        elif mode == 'percent':
            try:
                pcts = [float(x.strip()) for x in options.get('percentages_str', '10,30,50,70,90').split(',') if x.strip()]
            except ValueError:
                pcts = [10, 30, 50, 70, 90]
                
            if is_vel:
                levels = [lo + (hi - lo) * (p / 100.0) for p in pcts]
            else:
                levels = [max(0, hi) * (p / 100.0) for p in pcts]
        else:
            levels = []

        return sorted([float(lv) for lv in levels if np.isfinite(lv)])

    def _clear_overlay_contours(self, overlay_dict=None):
        if overlay_dict is None:
            for ov in self.contour_overlays:
                self._clear_overlay_contours(ov)
            return
        for iso in overlay_dict.get('iso_items', []):
            iso.setParentItem(None)
            if iso.scene() is not None:
                self.view_channel.getView().removeItem(iso)
        overlay_dict['iso_items'] = []

    def draw_overlay_contours(self):
        self._clear_overlay_contours()
        if not self.contour_overlays or self.cube_clean is None:
            return

        for overlay_dict in self.contour_overlays:
            overlay_slice = self._get_overlay_slice_for_channel(overlay_dict)
            if overlay_slice is None:
                continue

            reprojected = self._reproject_overlay_slice(overlay_dict, overlay_slice)
            if reprojected is None:
                continue

            overlay_dict['_reproj_raw'] = reprojected
            overlay_dict['_reproj_channel'] = self.slider_channel.value()

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

            levels = self._compute_contour_levels(reprojected, opts)
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
                iso.setParentItem(self.view_channel.getImageItem())
                iso.setZValue(10)
                overlay_dict['iso_items'].append(iso)

    def close_overlay(self, index=None):
        removed_name = None
        if index is not None:
            if 0 <= index < len(self.contour_overlays):
                ov = self.contour_overlays.pop(index)
                removed_name = ov['name']
                self._clear_overlay_contours(ov)
        else:
            self._clear_overlay_contours()
            self.contour_overlays = []
        if removed_name is not None:
            suffix = " (overlay)"
            to_remove = [n for n in self.overlay_spectrum_curves if n == f"{removed_name}{suffix}"]
            for name in to_remove:
                c = self.overlay_spectrum_curves.pop(name)
                try:
                    if c.scene():
                        c.scene().removeItem(c)
                    else:
                        self.plot_widget.removeItem(c)
                except Exception:
                    pass
            to_remove_s = [n for n in self.overlay_spectrum_curves_smooth if n == f"{removed_name}{suffix}"]
            for name in to_remove_s:
                c = self.overlay_spectrum_curves_smooth.pop(name)
                try:
                    if c.scene():
                        c.scene().removeItem(c)
                    else:
                        self.plot_widget_smooth.removeItem(c)
                except Exception:
                    pass
        else:
            for name in list(self.overlay_spectrum_curves.keys()):
                c = self.overlay_spectrum_curves.pop(name)
                try:
                    if c.scene():
                        c.scene().removeItem(c)
                    else: self.plot_widget.removeItem(c)
                except Exception: pass
            for name in list(self.overlay_spectrum_curves_smooth.keys()):
                c = self.overlay_spectrum_curves_smooth.pop(name)
                try:
                    if c.scene():
                        c.scene().removeItem(c)
                    else: self.plot_widget_smooth.removeItem(c)
                except Exception: pass
        if not self.contour_overlays:
            self.btn_contour_overlay.setEnabled(False)
            self.btn_contour_overlay.hide()
        self.parent_window.statusBar().showMessage("Contour overlay removed.")
        self.update_channel_map()
        self.update_spectrum()

    def open_contour_options(self):
        if not self.contour_overlays:
            QMessageBox.warning(self, "No Overlay", "Please load a contour overlay file first.")
            return

        if hasattr(self, '_contour_options_dlg') and self._contour_options_dlg is not None and self._contour_options_dlg.isVisible():
            self._contour_options_dlg.raise_()
            self._contour_options_dlg.activateWindow()
            return

        dlg = ContourOptionsDialog(self, self.contour_overlays)
        self._contour_options_dlg = dlg
        dlg.setAttribute(Qt.WA_DeleteOnClose, True)
        dlg.destroyed.connect(lambda: setattr(self, '_contour_options_dlg', None))
        dlg.show()

    # ==================== MAP & ROI FUNCTIONS ====================
    def draw_contours(self, target_id, view, data):
        for iso in self.active_contours.get(target_id, []):
            iso.setParentItem(None)
            if iso.scene() is not None:
                view.getView().removeItem(iso)
        self.active_contours[target_id] = []

        params = self.contour_params.get(target_id)
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

        levels = self._compute_contour_levels(smoothed, params, target_id=target_id)
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
        self.view_channel.ui.histogram.gradient.loadPreset(self.parent_window.current_cmap)
        self.draw_contours('channel', self.view_channel, slice_data)
        self.draw_overlay_contours()
        self.update_spatial_analysis()
        self.update_beam_visualizers('channel')
        if hasattr(self, 'update_nr_rms'):
            self.update_nr_rms()
        
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

    def change_roi(self, roi_type, cx=None, cy=None):
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
        
        if cx is None:
            cx = offset
        if cy is None:
            cy = offset
        
        if hasattr(self, 'btn_edit_region'):
            if roi_type in ["Ellipse", "Rectangle", "Point (Beam)", "Custom Polygon"]:
                self.btn_edit_region.show()
            else:
                self.btn_edit_region.hide()
                
        new_roi = None
        if roi_type == "Point (Beam)": 
            has_beam = False
            bmaj_deg = 0.0
            bmin_deg = 0.0
            bpa_deg = 0.0
            
            if getattr(self, 'bmaj_array', None) is not None and getattr(self, 'bmin_array', None) is not None:
                has_beam = True
                bmaj_deg = np.median(self.bmaj_array) if isinstance(self.bmaj_array, (list, np.ndarray)) else self.bmaj_array
                bmin_deg = np.median(self.bmin_array) if isinstance(self.bmin_array, (list, np.ndarray)) else self.bmin_array
                
                bpa_array = getattr(self, 'bpa_array', None)
                if bpa_array is not None:
                    bpa_deg = np.median(bpa_array) if isinstance(bpa_array, (list, np.ndarray)) else float(bpa_array)
                elif self.raw_header is not None:
                    bpa_val = self.raw_header.get('BPA', 0.0)
                    bpa_deg = float(bpa_val)

            if has_beam:
                bmaj_arcsec = bmaj_deg * 3600.0
                bmin_arcsec = bmin_deg * 3600.0
                
                from src.gui.custom import get_pyqt_angle
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
                    new_roi = pg.RectROI([cx, cy], [self.pix_scale_arcsec, self.pix_scale_arcsec], pen='#f1c40f')
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
            
        name = f"SR{len(self.spectrum_spatial_rois) + 1}"
        cb = QCheckBox(name)
        cb.setChecked(True)
        cb.setStyleSheet(f"color: {col}; font-weight: bold;")
        cb.toggled.connect(self.update_spectrum)
        self.box_regions_layout.addWidget(cb)
        
        text_item = pg.TextItem(text=name, color=col, anchor=(0, 1))
        text_item.setZValue(30)
        self.view_channel.addItem(text_item)
        
        def update_spectrum_region_label(r=new_roi, t=text_item):
            try:
                br = r.boundingRect()
                pos = r.pos()
                t.setPos(pos.x() + br.right(), pos.y() + br.bottom())
            except Exception:
                pass

        new_roi.sigRegionChanged.connect(update_spectrum_region_label)
        update_spectrum_region_label()
        
        self.spectrum_spatial_rois.append({
            "name": name,
            "roi": new_roi,
            "checkbox": cb,
            "color": col,
            "type": roi_type,
            "text_item": text_item,
            "update_label": update_spectrum_region_label
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
                    self.view_channel.getView().removeItem(roi)
                # Remove checkbox
                cb = r_dict["checkbox"]
                self.box_regions_layout.removeWidget(cb)
                cb.deleteLater()
                
                if "text_item" in r_dict:
                    text_item = r_dict["text_item"]
                    if text_item.scene():
                        text_item.scene().removeItem(text_item)
                    else:
                        self.view_channel.getView().removeItem(text_item)
                        
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
            new_name = f"SR{i + 1}"
            old_name = r_dict["name"]
            r_dict["name"] = new_name
            r_dict["checkbox"].setText(new_name)
            
            if "text_item" in r_dict:
                r_dict["text_item"].setText(new_name)
            
            if old_name in self.spectrum_curves:
                self.spectrum_curves[new_name] = self.spectrum_curves.pop(old_name)
            if hasattr(self, 'spectrum_curves_smooth') and old_name in self.spectrum_curves_smooth:
                self.spectrum_curves_smooth[new_name] = self.spectrum_curves_smooth.pop(old_name)
                
        if len(self.spectrum_spatial_rois) <= 1:
            self.box_regions.hide()

        self.refresh_spectral_stats_apertures()
        self.update_spectrum()

    def clear_roi(self):
        self.delete_nr_roi()
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

    def _update_spectrum_state_machine(self):
        stat = self.combo_spec_stat.currentText()
        if stat == "Flux Density":
            for i in range(self.combo_spec_unit.count()):
                if self.combo_spec_unit.itemText(i) == "Jy":
                    self.combo_spec_unit.model().item(i).setEnabled(True)
                else:
                    self.combo_spec_unit.model().item(i).setEnabled(False)
            
            if self.combo_spec_unit.currentText() != "Jy":
                self.combo_spec_unit.setCurrentIndex(1) # Auto-switch to Jy
        else:
            for i in range(self.combo_spec_unit.count()):
                if "Native" in self.combo_spec_unit.itemText(i) or self.combo_spec_unit.itemText(i) in ["K", "Jy/beam"]:
                    self.combo_spec_unit.model().item(i).setEnabled(True)
                else:
                    self.combo_spec_unit.model().item(i).setEnabled(False)
            
            if self.combo_spec_unit.currentText() == "Jy":
                self.combo_spec_unit.setCurrentIndex(0) # Auto-switch to Native

    def update_spectrum(self):
        if self.cube_clean is None or getattr(self, 'is_2d_image', False): return
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
        
        stat = self.combo_spec_stat.currentText()
        unit_sel = self.combo_spec_unit.currentText()
        
        is_rj_active = False
        with np.errstate(invalid='ignore', divide='ignore'):
            for r_dict in rois_to_plot:
                roi = r_dict["roi"]
                name = r_dict["name"]
                color = r_dict["color"]
                
                if roi is None:
                    sub_data = self.cube_clean
                else:
                    sub_data = roi.getArrayRegion(self.cube_clean, self.view_channel.getImageItem(), axes=(1, 2))
                    
                    # Create precise boolean mask using PyQtGraph rasterization
                    dummy_ones = np.ones((self.nx, self.ny))
                    roi_mask = roi.getArrayRegion(dummy_ones, self.view_channel.getImageItem(), axes=(0, 1))
                    
                    # Safely set background bounding-box pixels to np.nan
                    sub_data[:, roi_mask == 0] = np.nan
                    
                # Print valid pixels count for diagnostic purposes
                # if len(sub_data) > 0:
                #     n_valid_pixels = np.count_nonzero(~np.isnan(sub_data[0]))
                #     print(f"[{stat}] Valid pixels in mask for region '{name}': {n_valid_pixels}")
                    
                # Phase 2: Spatial Collapse
                if "Max" in stat:
                    raw_array = np.nanmax(sub_data, axis=(1, 2))
                elif "Sum" in stat:
                    raw_array = np.nansum(sub_data, axis=(1, 2))
                elif "Median" in stat:
                    raw_array = np.nanmedian(sub_data, axis=(1, 2))
                else:
                    raw_array = np.nanmean(sub_data, axis=(1, 2))
                
                # Phase 3: The 4 Conversion Paths
                unit_lower = self.display_unit.replace(" ", "").lower()
                
                if "Native" in unit_sel or not self.can_convert_units:
                    # Path 4: Native matches Target
                    final_array = raw_array
                    y_label = f"{stat} ({self.display_unit})"
                    self.spec_unit = self.display_unit
                elif unit_sel == "Jy":
                    if "jy" in unit_lower:
                        # Path 1: Native Jy/beam, Target Jy (Statistic: Sum)
                        if "pixel" in unit_lower or "pix" in unit_lower:
                            flux_array_jy = raw_array
                        else:
                            flux_array_jy = raw_array / self.n_beam_array.value
                        final_array = flux_array_jy
                    elif "k" == unit_lower or "kelvin" in unit_lower:
                        # Path 3: Native K, Target Jy (Statistic: Sum)
                        is_rj_active = True
                        freq_hz = self.freq_array
                        jy_sr_per_kelvin = (1 * u.K).to(u.Jy / u.sr, equivalencies=u.brightness_temperature(freq_hz * u.Hz))
                        flux_array_jy = raw_array * jy_sr_per_kelvin.value * self.omega_pix_sr.value
                        final_array = flux_array_jy
                    else:
                        final_array = raw_array
                    y_label = f"{stat} (Jy)"
                    self.spec_unit = "Jy"
                elif unit_sel == "K":
                    if "k" == unit_lower or "kelvin" in unit_lower:
                        # Path 4: Native K, Target K
                        final_array = raw_array
                    elif "jy" in unit_lower:
                        # Path 2: Native Jy/beam, Target K (Statistic: Mean, Median, Max)
                        is_rj_active = True
                        omega = self.omega_pix_sr if ("pixel" in unit_lower or "pix" in unit_lower) else self.omega_beam_sr
                        surface_brightness = (raw_array * u.Jy) / omega
                        freq_hz = self.freq_array
                        tb_array_k = surface_brightness.to(u.K, equivalencies=u.brightness_temperature(freq_hz * u.Hz))
                        final_array = tb_array_k.value
                    else:
                        final_array = raw_array
                    y_label = f"{stat} (K)"
                    self.spec_unit = "K"
                elif unit_sel == "Jy/beam":
                    if "jy" in unit_lower and ("pixel" not in unit_lower and "pix" not in unit_lower):
                        # Path 4: Native Jy/beam, Target Jy/beam
                        final_array = raw_array
                    elif "k" == unit_lower or "kelvin" in unit_lower:
                        # Path 5: Native K, Target Jy/beam
                        is_rj_active = True
                        freq_hz = self.freq_array
                        jy_sr_per_kelvin = (1 * u.K).to(u.Jy / u.sr, equivalencies=u.brightness_temperature(freq_hz * u.Hz))
                        surface_brightness_jy_sr = raw_array * jy_sr_per_kelvin.value
                        final_array = surface_brightness_jy_sr * self.omega_beam_sr.value
                    else:
                        final_array = raw_array
                    y_label = f"{stat} (Jy/beam)"
                    self.spec_unit = "Jy/beam"
                    
                spec = final_array
                    
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

        if hasattr(self, 'lbl_rj_warning'):
            self.lbl_rj_warning.setVisible(is_rj_active)

        num_spatial_regions = len(active_rois)
        active_spatial_roi = active_rois[0]["roi"] if active_rois else None
        if num_spatial_regions <= 1 and self.contour_overlays:
            for ov in self.contour_overlays:
                if ov['is_static'] or ov['v_axis'] is None:
                    continue
                ov_name = ov['name']
                ov_color = ov['options']['color']

                if active_spatial_roi is None:
                    ov_sub_data = ov['cube']
                else:
                    ov_sub_data = active_spatial_roi.getArrayRegion(ov['cube'], self.view_channel.getImageItem(), axes=(1, 2))

                if "Max" in stat:
                    ov_spec = np.nanmax(ov_sub_data, axis=(1, 2))
                elif "Sum" in stat:
                    ov_spec = np.nansum(ov_sub_data, axis=(1, 2))
                else:
                    ov_spec = np.nanmean(ov_sub_data, axis=(1, 2))

                # Inject Jy and K conversion math for the overlay
                unit_sel = self.combo_spec_unit.currentText()
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

                if display_name not in self.overlay_spectrum_curves:
                    c = pg.PlotDataItem([], [], stepMode="center", pen=curve_color, name=display_name)
                    self.overlay_spectrum_curves[display_name] = c
                    self.plot_widget.addItem(c)
                self.overlay_spectrum_curves[display_name].setPen(curve_color)
                self.overlay_spectrum_curves[display_name].setData(x=ve, y=ss_resampled)

                if getattr(self, 'smoothing_params', None) is not None and getattr(self, 'spectrum_tabs', None) is not None:
                    if self.spectrum_tabs.indexOf(self.plot_widget_smooth) != -1:
                        method = self.smoothing_params['method']
                        ss_ov_smooth = ss_resampled.copy()
                        try:
                            if method == 'boxcar':
                                from scipy.ndimage import uniform_filter1d
                                ss_ov_smooth = uniform_filter1d(ss_ov_smooth, size=self.smoothing_params['window'])
                            elif method == 'gaussian':
                                from scipy.ndimage import gaussian_filter1d
                                ss_ov_smooth = gaussian_filter1d(ss_ov_smooth, sigma=self.smoothing_params['sigma'])
                            elif method == 'savgol':
                                from scipy.signal import savgol_filter
                                w = self.smoothing_params['window']
                                p = self.smoothing_params['polyorder']
                                if len(ss_ov_smooth) > w:
                                    ss_ov_smooth = savgol_filter(ss_ov_smooth, window_length=w, polyorder=p)
                        except Exception:
                            pass

                        if display_name not in self.overlay_spectrum_curves_smooth:
                            c_s = pg.PlotDataItem([], [], stepMode="center", pen=curve_color, name=display_name)
                            self.overlay_spectrum_curves_smooth[display_name] = c_s
                            self.plot_widget_smooth.addItem(c_s)
                        self.overlay_spectrum_curves_smooth[display_name].setPen(curve_color)
                        self.overlay_spectrum_curves_smooth[display_name].setData(x=ve, y=ss_ov_smooth)
        else:
            self._clear_all_overlay_spectrum_curves()

        self._cleanup_removed_overlay_curves()

        self.plot_widget.autoRange()
        if hasattr(self, 'plot_widget_smooth'):
            self.plot_widget_smooth.autoRange()

        # Update legends
        has_overlay = bool(self.contour_overlays)
        base_prefix = "Base: " if has_overlay else ""

        if getattr(self.plot_widget, 'plotItem', None) is not None and self.plot_widget.plotItem.legend is not None:
            self.plot_widget.plotItem.legend.clear()
            if "Whole Map" in active_names:
                self.plot_widget.plotItem.legend.addItem(self.spectrum_curve, f"{base_prefix}Whole Map")
            for n, c in self.spectrum_curves.items():
                self.plot_widget.plotItem.legend.addItem(c, f"{base_prefix}{n}")
            for n, c in self.overlay_spectrum_curves.items():
                clean_name = n.replace(" (overlay)", "")
                self.plot_widget.plotItem.legend.addItem(c, f"Overlay: {clean_name}")

        if getattr(self, 'plot_widget_smooth', None) is not None and self.plot_widget_smooth.plotItem.legend is not None:
            self.plot_widget_smooth.plotItem.legend.clear()
            if "Whole Map" in active_names:
                self.plot_widget_smooth.plotItem.legend.addItem(self.spectrum_curve_smooth, f"{base_prefix}Whole Map")
            for n, c_s in getattr(self, 'spectrum_curves_smooth', {}).items():
                self.plot_widget_smooth.plotItem.legend.addItem(c_s, f"{base_prefix}{n}")
            for n, c_s in getattr(self, 'overlay_spectrum_curves_smooth', {}).items():
                clean_name = n.replace(" (overlay)", "")
                self.plot_widget_smooth.plotItem.legend.addItem(c_s, f"Overlay: {clean_name}")

        if self.catalog_overlay_items:
            ymax = ymax_global if ymax_global != -np.inf else 1.0
            for item in self.catalog_overlay_items:
                if isinstance(item, pg.TextItem):
                    item.setPos(item.pos().x(), ymax)

    def _cleanup_removed_overlay_curves(self):
        active_names = set()
        for ov in self.contour_overlays:
            if not ov['is_static'] and ov['v_axis'] is not None:
                active_names.add(f"{ov['name']} (overlay)")
        for name in list(self.overlay_spectrum_curves.keys()):
            if name not in active_names:
                c = self.overlay_spectrum_curves.pop(name)
                if c.scene():
                    c.scene().removeItem(c)
                else:
                    self.plot_widget.removeItem(c)
        for name in list(self.overlay_spectrum_curves_smooth.keys()):
            if name not in active_names:
                c = self.overlay_spectrum_curves_smooth.pop(name)
                if c.scene():
                    c.scene().removeItem(c)
                else:
                    self.plot_widget_smooth.removeItem(c)

    def _clear_all_overlay_spectrum_curves(self):
        for name in list(self.overlay_spectrum_curves.keys()):
            c = self.overlay_spectrum_curves.pop(name)
            if c.scene():
                c.scene().removeItem(c)
            else:
                self.plot_widget.removeItem(c)
        for name in list(self.overlay_spectrum_curves_smooth.keys()):
            c = self.overlay_spectrum_curves_smooth.pop(name)
            if c.scene():
                c.scene().removeItem(c)
            else:
                self.plot_widget_smooth.removeItem(c)

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
        if self._channel_grid_popup and self._channel_grid_popup.isVisible():
            self._channel_grid_popup.update_grid()

    def update_beam_visualizers(self, panel_type, panel_id=None):
        if self.cube_clean is None:
            return

        target_plot = self.plot_channel if panel_type == 'channel' else self.panels[panel_id]['plot_item']
        
        if not hasattr(self, 'beam_visualizer_items'):
            self.beam_visualizer_items = {}
        
        dict_key = 'channel' if panel_type == 'channel' else f'moment_{panel_id}'
        for item in self.beam_visualizer_items.get(dict_key, []):
            try:
                target_plot.vb.removeItem(item)
            except Exception:
                pass
        self.beam_visualizer_items[dict_key] = []
        
        if panel_type == 'moment':
            mtype = self.panels[panel_id]['combo'].currentText()
            if 'PV Diagram' in mtype:
                return
        
        beams_to_draw = []
        
        def get_beam_for_cube(bmaj_arr, bmin_arr, bpa_arr, bmaj_s, bmin_s, bpa_s):
            if bmaj_arr is not None and bmin_arr is not None:
                if panel_type == 'channel':
                    idx = self.slider_channel.value()
                    return bmaj_arr[idx], bmin_arr[idx], (bpa_arr[idx] if bpa_arr is not None else 0.0)
                elif panel_type == 'moment':
                    if getattr(self, 'slider_velocity', None) is not None:
                        rg = self.slider_velocity.getRegion()
                        v_min, v_max = rg
                        mask = (self.v_axis >= v_min) & (self.v_axis <= v_max) if self.v_axis[0] < self.v_axis[-1] else (self.v_axis <= v_min) & (self.v_axis >= v_max)
                        mask_idx = np.where(mask)[0]
                        if len(mask_idx) > 0:
                            b_ma = np.nanmedian(bmaj_arr[mask_idx])
                            b_mi = np.nanmedian(bmin_arr[mask_idx])
                            b_pa = np.nanmedian(bpa_arr[mask_idx]) if bpa_arr is not None else 0.0
                            return b_ma, b_mi, b_pa
                    return np.nanmedian(bmaj_arr), np.nanmedian(bmin_arr), np.nanmedian(bpa_arr) if bpa_arr is not None else 0.0
            return bmaj_s, bmin_s, bpa_s if bpa_s is not None else 0.0

        base_bmaj, base_bmin, base_bpa = get_beam_for_cube(
            self.bmaj_array, self.bmin_array, getattr(self, 'bpa_array', None),
            self.raw_header.get('BMAJ') if self.raw_header else None,
            self.raw_header.get('BMIN') if self.raw_header else None,
            self.raw_header.get('BPA', 0.0) if self.raw_header else 0.0
        )
        
        if base_bmaj and base_bmin:
            beams_to_draw.append({'bmaj': base_bmaj, 'bmin': base_bmin, 'bpa': base_bpa, 'color': 'white'})
        else:
            print(f"WARNING: No beam info found for base cube. Beam visualizer hidden.")
            
        processed_files = [self.current_file_name] if hasattr(self, 'current_file_name') else []
        for ov in self.contour_overlays:
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
            
            self.beam_visualizer_items[dict_key].extend([item_el, item_maj, item_min])

        self.update_beam_positions(target_plot.vb)

    def update_beam_positions(self, view_box, view_range=None):
        if not hasattr(self, 'beam_visualizer_items'): return
        
        target_key = None
        if view_box == self.plot_channel.vb:
            target_key = 'channel'
        else:
            for i, p in enumerate(self.panels):
                if view_box == p['plot_item'].vb:
                    target_key = f'moment_{i}'
                    break
                    
        if target_key not in self.beam_visualizer_items: return
        items = self.beam_visualizer_items[target_key]
        if not items: return
        
        view_range = view_box.viewRange()
        v_x_min, v_x_max = view_range[0]
        v_y_min, v_y_max = view_range[1]

        cam_left_x = max(v_x_min, v_x_max)
        cam_bottom_y = min(v_y_min, v_y_max)
        
        img_left_x = (self.nx / 2.0) * self.pix_scale_arcsec
        img_bottom_y = -(self.ny / 2.0) * self.pix_scale_arcsec
        
        hud_left_x = min(cam_left_x, img_left_x)
        hud_bottom_y = max(cam_bottom_y, img_bottom_y)

        pad_x = abs(v_x_max - v_x_min) * 0.03
        pad_y = abs(v_y_max - v_y_min) * 0.03

        target_x = hud_left_x - pad_x
        target_y = hud_bottom_y + pad_y
        
        for item in items:
            item.setPos(target_x, target_y)


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
            try:
                self._moment_worker.result_ready.disconnect(self._on_moment_result)
            except TypeError:
                pass
            self._moment_worker.finished.connect(self._moment_worker.deleteLater)
            self._pending_workers.append(self._moment_worker)
        self._purge_finished_workers()
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
                is_vel = ('Moment 1' in mtype) or ('Moment 2' in mtype) or ('Moment 9' in mtype)
                self.apply_cmap(p['view'], is_vel)
            self.update_beam_visualizers('moment', panel_id=i)

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
                    cfg['pv_width']  = active_item.get('width', 1)
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

    def _purge_finished_workers(self):
        alive = []
        for w in self._pending_workers:
            try:
                if w.isRunning():
                    alive.append(w)
            except RuntimeError:
                pass
        self._pending_workers = alive

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
                    self.update_beam_visualizers('moment', panel_id=panel_id)

    def clear_all_hover_labels(self):
        for lbl in [self.lbl_hover_ch, self.lbl_hover_spec, self.lbl_hover_pv] + [p['lbl_hover'] for p in self.panels]:
            lbl.setText("")

    def hover_event(self, pos, plot_item, data_array, active_label, panel_id='channel'):
        self.clear_all_hover_labels()

        hovered_pv_cut_name = None
        hovered_offset = None
        
        if panel_id == 'channel' and plot_item.sceneBoundingRect().contains(pos):
            mp = plot_item.vb.mapSceneToView(pos)
            mouse_pt = np.array([mp.x(), mp.y()])
            
            from PyQt5.QtCore import QPointF
            for item in self.pv_cuts:
                hit = False
                w_item = item.get("width_item")
                if w_item is not None and w_item.isVisible() and w_item.contains(QPointF(mp.x(), mp.y())):
                    hit = True
                elif item.get("roi") is not None and self.line_roi_hit_test(item["roi"], pos, tolerance=10.0):
                    hit = True
                    
                if hit:
                    pts = self.get_line_roi_points(item["roi"])
                    if pts is not None:
                        p1, p2 = pts
                        vec = p2 - p1
                        length = np.hypot(vec[0], vec[1])
                        if length > 0:
                            unit = vec / length
                            offset = np.dot(mouse_pt - p1, unit)
                            offset = max(0.0, min(length, offset))
                            hovered_pv_cut_name = item["name"]
                            hovered_offset = offset
                            break

        if hasattr(self, 'pv_hover_line_main'):
            if self.combo_panel_mode.currentText() != "Spatial Analysis" and hovered_pv_cut_name is not None and getattr(self, 'combo_pv_cuts', None) and self.combo_pv_cuts.currentText() == hovered_pv_cut_name:
                self.pv_hover_line_main.setPos(hovered_offset)
                self.pv_hover_line_main.show()
                self.lbl_hover_pv.setText(f"Offset: {hovered_offset:.2f} arcsec")
                self.lbl_hover_pv.setStyleSheet("color: #3498db; font-weight: bold; font-size: 9.5px;")
            else:
                self.pv_hover_line_main.hide()

        for panel in self.panels:
            if 'pv_hover_line' in panel:
                if panel['combo'].currentText() == "PV Diagram" and hovered_pv_cut_name is not None and panel['combo_pv_cut'].currentText() == hovered_pv_cut_name:
                    panel['pv_hover_line'].setPos(hovered_offset)
                    panel['pv_hover_line'].show()
                    panel['lbl_hover'].setText(f"Offset: {hovered_offset:.2f} arcsec")
                    panel['lbl_hover'].setStyleSheet("color: #3498db; font-weight: bold; font-size: 9.5px;")
                else:
                    panel['pv_hover_line'].hide()

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
                    coord_text = f"RA: {ra_str}, DEC: {dec_str}"
                else:
                    # Fallback to pixels if WCS fails to load
                    coord_text = f"({x_idx}, {y_idx})"
                
                # Set the final label text (Pixels + Absolute RA/Dec + Value)
                active_label.setText(f"({x_idx}, {y_idx}) | {coord_text} | {val_str} {unit_str}")
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
                    
                    if name.startswith("SR"):
                        try:
                            # Extract number, handling "SR 1" or "SR1"
                            num_str = name.replace("SR", "").strip()
                            num = int(num_str)
                            display = f"SR{num}"
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

    def open_channel_grid_popup(self):
        if self._channel_grid_popup is None:
            self._channel_grid_popup = ChannelGridDialog(self)
            
        self._channel_grid_popup.update_grid()
        self._channel_grid_popup.show()
        self._channel_grid_popup.raise_()
        self._channel_grid_popup.activateWindow()


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

        if len(apertures_to_calc) <= 1 and self.contour_overlays:
            aperture_roi = apertures_to_calc[0]["roi"]
            for ov in self.contour_overlays:
                if ov['is_static'] or ov['v_axis'] is None:
                    continue
                ov_name = ov['name']
                ov_color = ov['options']['color']
                ov_unit = getattr(self, 'display_unit', '')

                if aperture_roi is None:
                    ov_sub_data = ov['cube']
                else:
                    ov_sub_data = aperture_roi.getArrayRegion(ov['cube'], self.view_channel.getImageItem(), axes=(1, 2))

                if "Max" in self.combo_spec_stat.currentText():
                    ov_spec = np.nanmax(ov_sub_data, axis=(1, 2))
                elif "Sum" in self.combo_spec_stat.currentText():
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
