"""
Module providing high-performance mathematical kernels.

Contains JIT-compiled Numba and vectorized NumPy functions for computationally
heavy tasks like interpolation and spectral moment map generation.
"""
import numpy as np
import warnings

# Optional Numba acceleration (graceful fallback to NumPy when not installed)
try:
    import numba
    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False

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
        """
        Perform parallelized bilinear interpolation on a 3D data cube.

        Parameters
        ----------
        cube : numpy.ndarray
            The 3D data cube with shape (Nv, Nx, Ny).
        x0 : numpy.ndarray
            1D integer array of floor x-coordinates.
        y0 : numpy.ndarray
            1D integer array of floor y-coordinates.
        x1 : numpy.ndarray
            1D integer array of ceiling x-coordinates.
        y1 : numpy.ndarray
            1D integer array of ceiling y-coordinates.
        fx : numpy.ndarray
            1D float array of fractional weights in x.
        fy : numpy.ndarray
            1D float array of fractional weights in y.
        out : numpy.ndarray
            2D float array to store the interpolated output with shape (Nv, Npoints).

        Returns
        -------
        None

        Notes
        -----
        This function uses Numba JIT compilation for multi-core parallelism across 
        the spectral axis (`Nv`). The spatial interpolation is a classic 4-point 
        bilinear gather operation weighted by `fx` and `fy`.
        """
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
    """
    Perform bilinear interpolation on a 3D data cube using vectorized NumPy operations.

    Parameters
    ----------
    cube : numpy.ndarray
        The 3D data cube with shape (Nv, Nx, Ny).
    x0 : numpy.ndarray
        1D integer array of floor x-coordinates.
    y0 : numpy.ndarray
        1D integer array of floor y-coordinates.
    x1 : numpy.ndarray
        1D integer array of ceiling x-coordinates.
    y1 : numpy.ndarray
        1D integer array of ceiling y-coordinates.
    fx : numpy.ndarray
        1D float array of fractional weights in x.
    fy : numpy.ndarray
        1D float array of fractional weights in y.
    out : numpy.ndarray
        2D float array to store the interpolated output with shape (Nv, Npoints).

    Returns
    -------
    None

    Notes
    -----
    This is a pure-NumPy fallback for environments without Numba. It evaluates 
    the bilinear interpolation formula across all spatial sample points simultaneously 
    via advanced indexing and array broadcasting over the spectral axis.
    """
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
        """
        Compute Moment 1 (velocity) and Moment 2 (dispersion) in a single pass.

        Parameters
        ----------
        mc : numpy.ndarray
            Masked 3D data cube (float64) of shape (Nv, Nx, Ny). NaN values signify 
            voxels below the intensity threshold.
        v_axis : numpy.ndarray
            1D float array (float64) of velocity values in km/s of length Nv.

        Returns
        -------
        tuple of numpy.ndarray
            A tuple containing:
            - m1_out : 2D float array (Nx, Ny) of intensity-weighted velocities.
            - m2_out : 2D float array (Nx, Ny) of intensity-weighted velocity dispersions.

        Notes
        -----
        This Numba-accelerated kernel computes both moments simultaneously to minimize 
        memory overhead and cache misses. The math relies on the standard definitions:
        M1 = sum(I * v) / sum(I)
        M2 = sqrt(sum(I * v^2) / sum(I) - M1^2)
        Note that this naive single-pass variance calculation can suffer from 
        catastrophic cancellation precision issues for narrow spectral lines.
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
    """
    Compute Moment 1 and Moment 2 using vectorized NumPy tensor contractions.

    Parameters
    ----------
    mc : numpy.ndarray
        Masked 3D data cube of shape (Nv, Nx, Ny). NaN values represent masked voxels.
    v_axis : numpy.ndarray
        1D float array of velocity values in km/s.

    Returns
    -------
    tuple of numpy.ndarray
        A tuple containing:
        - m1 : 2D float array (Nx, Ny) representing the Moment 1 map.
        - m2 : 2D float array (Nx, Ny) representing the Moment 2 map.

    Notes
    -----
    This pure-NumPy implementation utilizes BLAS-accelerated `numpy.tensordot` 
    for fast weighted sum contractions along the spectral axis. The mathematical 
    formulation is mathematically identical to the loop-based M1 and M2 definitions, 
    but evaluates the full 3D array in a batched memory operation.
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
