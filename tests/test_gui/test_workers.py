import pytest
import numpy as np
from cubex.gui.controllers.workers import MomentWorker

def test_moment_worker_thresholding():
    """Verify that the background worker correctly masks sub-threshold voxels before moments."""
    # Synthetic cube (Nv=3, Nx=2, Ny=2)
    cube = np.zeros((3, 2, 2))
    cube[:, 0, 0] = 0.5  # Below threshold
    cube[:, 0, 1] = 5.0  # Above threshold
    cube[:, 1, 0] = -1.0 # Negative noise

    params = {
        'selected_cube': cube,
        'sub_v': np.array([10.0, 20.0, 30.0]),
        'minX': 10.0,
        'maxX': 30.0,
        'nx': 2,
        'ny': 2,
        'pix_scale_arcsec': 1.0,
        'display_unit': 'K',
        'panel_configs': [
            {'mtype': 'Moment 0', 'threshold': 2.0}
        ]
    }
    
    worker = MomentWorker(params, generation=1)
    
    results = []
    worker.result_ready.connect(results.append)
    
    # Execute the calculation payload synchronously for testing
    worker.run()
    
    assert len(results) == 1
    res = results[0]
    assert res['generation'] == 1
    
    m0_data = res['panel_results'][0]['data']
    assert m0_data.shape == (2, 2)
    
    # dv = abs(20.0 - 10.0) = 10.0
    # (0, 0): all values (0.5) < 2.0 -> fully masked -> NaN
    assert np.isnan(m0_data[0, 0])
    
    # (0, 1): all values (5.0) > 2.0 -> unmasked -> sum(5.0 * 3) * 10.0 = 150.0
    assert m0_data[0, 1] == 150.0
    
    # (1, 0): all values (-1.0) < 2.0 -> fully masked -> NaN
    assert np.isnan(m0_data[1, 0])

def test_moment_worker_pv_slice_averaging():
    """Verify that multi-pixel width PV extraction smoothly averages parallel slices."""
    cube = np.zeros((2, 5, 5))
    # Central row has high flux, surroundings have 0
    cube[:, 2, :] = 10.0
    
    p1 = [0.0, 0.0]
    p2 = [4.0, 0.0]
    
    # Width=1 (just the center line) -> should sample perfectly 10.0 if aligned, 
    # but the exact coordinate mapping depends on pix_scale. 
    # For this test, we just want to ensure width > 1 executes without matrix shape crashes
    offsets_1, pv_data_1 = MomentWorker._sample_pv_along_line(
        p1, p2, cube, nx=5, ny=5, pix_scale_arcsec=1.0, width=1
    )
    
    # Use pytest.warns to expect and suppress the "Mean of empty slice" RuntimeWarning
    # generated when parallel slices fall completely out of bounds at the edges.
    with pytest.warns(RuntimeWarning, match="Mean of empty slice"):
        offsets_3, pv_data_3 = MomentWorker._sample_pv_along_line(
            p1, p2, cube, nx=5, ny=5, pix_scale_arcsec=1.0, width=3
        )
    
    # Confirm shape consistency regardless of extraction width
    assert pv_data_1.shape == pv_data_3.shape
    assert len(offsets_1) == len(offsets_3)
    
    # Edges should properly evaluate to NaN due to out of bounds sampling
    assert np.any(np.isnan(pv_data_3))
    
    # The wider sample should theoretically dilute the high-flux central row (10.0)
    # by averaging it with the adjacent 0.0 rows.
    # We assert that the central in-bounds math executes correctly.
    valid_data = pv_data_3[np.isfinite(pv_data_3)]
    assert len(valid_data) > 0
    assert np.all(valid_data >= 0.0)
    assert np.all(valid_data < 10.0)
