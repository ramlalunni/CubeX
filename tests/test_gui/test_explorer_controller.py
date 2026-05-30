import numpy as np
from astropy.io import fits
from unittest.mock import MagicMock
from cubex.gui.controllers.explorer_controller import ExplorerController

def test_load_file_2d_fallback_logic(tmp_path):
    """
    Verify that loading a 2D FITS file cleanly triggers the custom fallback 
    logic without crashing, mocking the required 3D structures internally.
    """
    # Create synthetic 2D FITS (no spectral axis)
    data = np.ones((10, 10))
    header = fits.Header()
    header['BUNIT'] = 'Jy/beam'
    header['CDELT1'] = -0.001
    header['CDELT2'] = 0.001
    
    hdu = fits.PrimaryHDU(data=data, header=header)
    test_file = tmp_path / "test_2d.fits"
    hdu.writeto(test_file)
    
    # Mock view UI components
    mock_view = MagicMock()
    mock_view.parent_window.is_absolute_wcs = False
    
    controller = ExplorerController(mock_view)
    success = controller.load_file(str(test_file))
    
    assert success is True
    assert mock_view.is_2d_image is True
    
    # Verify spatial bounds extracted properly
    assert mock_view.nx == 10
    assert mock_view.ny == 10
    
    # Ensure data was safely promoted to a 3D structure for downstream pipelines
    assert mock_view.cube_clean.shape == (1, 10, 10)
    
    # Ensure spectral axis was instantiated as a dummy length-1 array
    assert len(mock_view.v_axis) == 1
    assert mock_view.v_axis[0] == 0.0

def test_spectral_stats_calculator():
    """
    Verify that the statistical metric calculator processes 1D flux arrays 
    correctly and generates the expected mathematical values based on dv steps.
    """
    mock_view = MagicMock()
    mock_view.spec_unit = 'K'
    
    # Synthetic 1D spectrum curve data representing flux
    mock_curve = MagicMock()
    # x_edges for stepMode: [-5, 5, 15, 25] -> resulting v_axis centers: [0, 10, 20]
    mock_curve.getData.return_value = (
        np.array([-5.0, 5.0, 15.0, 25.0]), 
        np.array([10.0, 20.0, 15.0])
    )
    
    mock_view.spectrum_curve = mock_curve
    mock_view.v_axis = np.array([0.0, 10.0, 20.0])
    # Dummy frequency logic to avoid crash if unit conversion is attempted
    mock_view.freq_array = np.array([1e9, 1e9, 1e9])
    
    # Mock selection bounds to encompass the entire curve
    mock_roi = MagicMock()
    mock_roi.getData.return_value = ([-10.0, 30.0], None)
    mock_view._get_popup_selected_boxes.return_value = [mock_roi]
    mock_view._get_popup_selected_apertures.return_value = []
    
    # Mock popup window UI selections
    mock_popup = MagicMock()
    cb_int = MagicMock(); cb_int.isChecked.return_value = True
    cb_peak = MagicMock(); cb_peak.isChecked.return_value = True
    mock_popup.stat_checkboxes = {
        "Integrated Intensity": cb_int, 
        "Peak (Max)": cb_peak
    }
    
    controller = ExplorerController(mock_view)
    controller._run_spectral_stats_calc(mock_popup)
    
    # Verify HTML generation
    assert mock_popup.lbl_result.setText.called
    html_output = mock_popup.lbl_result.setText.call_args[0][0]
    
    # Math: dv = 10.0. Flux sum = 10+20+15 = 45. Int = 450.0
    assert "450.0" in html_output
    # Peak flux is 20.0
    assert "20.0" in html_output

def test_get_velocity_subset():
    """
    Verify that the 1D velocity ROI slicer uses np.searchsorted correctly 
    to extract the exact bounds selected by the user.
    """
    mock_view = MagicMock()
    mock_view.v_axis = np.array([0.0, 10.0, 20.0, 30.0, 40.0])
    mock_view.cube_clean = np.ones((5, 2, 2))
    
    # Emulate the UI ROI bounding box placed between 5.0 and 25.0 km/s
    mock_view.region.getRegion.return_value = (5.0, 25.0)
    
    controller = ExplorerController(mock_view)
    sub_cube, sub_v, minX, maxX = controller.get_velocity_subset(use_full_range=False)
    
    # Should slice from index 1 (10.0) up to index 3 (exclusive) -> [10.0, 20.0]
    assert sub_cube.shape == (2, 2, 2)
    assert len(sub_v) == 2
    assert sub_v[0] == 10.0
    assert sub_v[1] == 20.0
    assert minX == 5.0
    assert maxX == 25.0
