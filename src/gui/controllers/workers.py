"""
Module defining background worker threads for CubeX.

These workers execute expensive data processing tasks (like moment map and PV 
diagram generation) off the main Qt thread to keep the UI responsive.
"""
import numpy as np
import warnings
from PyQt5.QtCore import QThread, pyqtSignal
from src.core.math_kernels import _compute_moments_12, _bilinear_interp

class MomentWorker(QThread):
    """
    Computes all moment maps and PV diagram data in a background thread.

    No Qt GUI or PyQtGraph calls are made here — only plain NumPy and Numba.
    Emits `result_ready` with a dictionary payload when finished, or returns 
    silently if cancelled.

    Attributes
    ----------
    result_ready : PyQt5.QtCore.pyqtSignal
        Signal emitted when computation is complete, containing the results dict.
    params : dict
        The configuration parameters for the current calculation batch.
    generation : int
        An ID tag to ensure out-of-order background threads are ignored by the UI.
    """
    result_ready = pyqtSignal(dict)

    def __init__(self, params: dict, generation: int):
        """
        Initialize the MomentWorker.

        Parameters
        ----------
        params : dict
            Dictionary containing the data cube, threshold settings, and panel configs.
        generation : int
            The generation counter from the controller to track obsolete runs.
        """
        super().__init__()
        self.params = params
        self.generation = generation
        self._cancelled = False

    def cancel(self):
        """
        Cancel the ongoing calculation at the next available checkpoint.

        Returns
        -------
        None
        """
        self._cancelled = True

    # ------------------------------------------------------------------
    def run(self):
        """
        Execute the moment map and PV diagram calculations.

        Iterates over the provided panel configurations, evaluating thresholded
        moment products (M0, M1, M2, M8, M9) and PV slices using the underlying
        math kernels.

        Returns
        -------
        None
        """
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
                if 'Moment -1' in mtype:
                    if is_all_nan:
                        data = np.full(m0_raw.shape, np.nan)
                    else:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            data = np.nanmean(mc, axis=0)
                    levels   = (0, float(np.nanmax(data)) if not np.isnan(data).all() else 1.0)
                    unit_str = display_unit

                elif 'Moment 0' in mtype:
                    if is_all_nan:
                        data = np.full(m0_raw.shape, np.nan)
                    else:
                        dv = np.abs(sub_v[1] - sub_v[0]) if len(sub_v) > 1 else 1.0
                        data = np.nansum(mc, axis=0) * dv
                        data[np.nansum(np.isfinite(mc), axis=0) == 0] = np.nan
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
        Extract a Position-Velocity slice using bilinear interpolation.

        Parameters
        ----------
        p1 : array_like
            Starting world-coordinate point [x_arcsec, y_arcsec].
        p2 : array_like
            Ending world-coordinate point [x_arcsec, y_arcsec].
        cube_data : numpy.ndarray
            The 3D data cube to sample from (Nv, Nx, Ny).
        nx : int
            Number of pixels in the spatial X axis.
        ny : int
            Number of pixels in the spatial Y axis.
        pix_scale_arcsec : float
            The pixel scale in arcseconds.
        width : int, optional
            Number of pixels to average perpendicular to the cut, by default 1.

        Returns
        -------
        tuple
            A 2-tuple containing:
            - offsets (numpy.ndarray): 1D array of spatial offsets along the cut.
            - pv_data (numpy.ndarray): 2D array of extracted fluxes (Offsets, Velocity).
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
