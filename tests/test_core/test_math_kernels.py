import numpy as np
from cubex.core.math_kernels import _compute_moments_12, _bilinear_interp

def test_compute_moments_12_basic():
    """Verify standard moment 1 and 2 calculations."""
    mc = np.array([
        [[1.0, 0.0], [0.0, 1.0]],
        [[0.0, 1.0], [1.0, 0.0]]
    ]) # shape: (Nv=2, Nx=2, Ny=2)
    v_axis = np.array([10.0, 20.0])
    
    m1, m2 = _compute_moments_12(mc, v_axis)
    
    assert m1.shape == (2, 2)
    assert m2.shape == (2, 2)
    
    # m1 = sum(I * v) / sum(I)
    np.testing.assert_array_equal(m1, np.array([[10.0, 20.0], [20.0, 10.0]]))
    
    # m2 (variance) should be 0 because all flux is in exactly one channel for each pixel
    np.testing.assert_array_equal(m2, np.zeros((2, 2)))

def test_compute_moments_12_all_zeros():
    """Verify that an entirely zero-filled cube results in NaN moments (avoids ZeroDivisionError)."""
    mc = np.zeros((3, 4, 4))
    v_axis = np.array([10.0, 20.0, 30.0])
    
    m1, m2 = _compute_moments_12(mc, v_axis)
    
    assert np.all(np.isnan(m1))
    assert np.all(np.isnan(m2))

def test_compute_moments_12_all_nans():
    """Verify that a cube filled with NaNs (masked data) results in NaN moments."""
    mc = np.full((3, 4, 4), np.nan)
    v_axis = np.array([10.0, 20.0, 30.0])
    
    m1, m2 = _compute_moments_12(mc, v_axis)
    
    assert np.all(np.isnan(m1))
    assert np.all(np.isnan(m2))

def test_compute_moments_12_mixed_data():
    """Verify calculations when encountering a mix of NaNs, zeros, and valid data."""
    mc = np.array([
        [[np.nan, 0.0], [2.0, np.nan]],
        [[np.nan, 0.0], [2.0, 1.0]]
    ])
    v_axis = np.array([10.0, 20.0])
    
    m1, m2 = _compute_moments_12(mc, v_axis)
    
    # (0,0) is all NaNs -> m1 should be nan
    assert np.isnan(m1[0, 0])
    # (0,1) is all zeros -> m1 should be nan
    assert np.isnan(m1[0, 1])
    # (1,0) has [2.0, 2.0] -> m1 = (2*10 + 2*20)/4 = 60/4 = 15.0
    assert m1[1, 0] == 15.0
    # (1,1) has [nan, 1.0] -> acts like [0.0, 1.0] -> m1 = (0*10 + 1*20)/1 = 20.0
    assert m1[1, 1] == 20.0

def test_bilinear_interp_exact_pixel():
    """Scenario A: Sample exactly on a pixel coordinate."""
    cube = np.zeros((1, 3, 3))
    cube[0, 1, 1] = 5.0
    
    # Sample exactly at (1, 1)
    x0 = np.array([1]); y0 = np.array([1])
    x1 = np.array([1]); y1 = np.array([1])
    fx = np.array([0.0]); fy = np.array([0.0])
    
    out = np.zeros((1, 1))
    _bilinear_interp(cube, x0, y0, x1, y1, fx, fy, out)
    assert out[0, 0] == 5.0

def test_bilinear_interp_halfway():
    """Scenario B: Sample exactly halfway between pixels."""
    cube = np.zeros((1, 2, 2))
    cube[0, 0, 0] = 10.0
    cube[0, 1, 0] = 20.0
    cube[0, 0, 1] = 30.0
    cube[0, 1, 1] = 40.0
    
    # Sample exactly at (0.5, 0.5)
    x0 = np.array([0]); y0 = np.array([0])
    x1 = np.array([1]); y1 = np.array([1])
    fx = np.array([0.5]); fy = np.array([0.5])
    
    out = np.zeros((1, 1))
    _bilinear_interp(cube, x0, y0, x1, y1, fx, fy, out)
    
    # Math: 10*0.25 + 20*0.25 + 30*0.25 + 40*0.25 = 25.0
    assert out[0, 0] == 25.0

def test_bilinear_interp_out_of_bounds_handling():
    """Scenario C: Test edge cases for boundary sampling when passing out of bounds values."""
    # Note: _bilinear_interp expects pre-clipped integer coordinates.
    # If the calling code samples exactly at the edge, fx and fy will handle it.
    # We test sampling near a masked (NaN) pixel to ensure it propagates correctly.
    cube = np.zeros((1, 2, 2))
    cube[0, 0, 0] = 10.0
    cube[0, 1, 0] = np.nan # NaN right on the interpolation boundary
    cube[0, 0, 1] = 10.0
    cube[0, 1, 1] = 10.0
    
    x0 = np.array([0]); y0 = np.array([0])
    x1 = np.array([1]); y1 = np.array([1])
    fx = np.array([0.5]); fy = np.array([0.5])
    
    out = np.zeros((1, 1))
    _bilinear_interp(cube, x0, y0, x1, y1, fx, fy, out)
    
    # Mathematical average with a NaN propagates the NaN
    assert np.isnan(out[0, 0])

