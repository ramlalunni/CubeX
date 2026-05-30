import numpy as np
import pytest
import pyqtgraph as pg

from cubex.gui.controllers.main_controller import MainController
from cubex.core.exporters import export_spectrum_csv_core

def test_export_spectrum_csv_rectangular_roi(qtbot, tmp_path):
    """Test exporting spectral data to CSV format with a rectangular ROI to verify width/height labels."""
    
    filename = tmp_path / "test_spectrum_rect"
    parent_filename = "cube_rect_test.fits"
    regions_to_export = ["Region 2"]
    
    # Synthetic curves
    curves = {
        "Region 2": np.array([5.5, 6.2, 4.8])
    }
    v_sorted = np.array([-5.0, 0.0, 5.0])
    
    # We test the controller's formatted string which should now have width and height
    # By creating a dummy RectROI
    roi = pg.RectROI([0, 0], [10, 20], angle=0)
    
    class DummyView:
        pass
    
    controller = MainController(DummyView())
    roi_str = controller._format_roi_props(roi)
    assert "width=10.00" in roi_str
    assert "height=20.00" in roi_str
    
    roi_dict = {"Region 2": roi_str}
    spec_unit = "K"
    
    # Execute the export function
    export_spectrum_csv_core(
        filename=str(filename),
        parent_filename=parent_filename,
        regions_to_export=regions_to_export,
        curves=curves,
        v_sorted=v_sorted,
        roi_dict=roi_dict,
        spec_unit=spec_unit
    )
    
    # Check if file was created
    expected_file = tmp_path / "test_spectrum_rect.csv"
    assert expected_file.exists()
    
    # Read back and verify CSV formatting includes the correct label
    with open(expected_file, 'r') as f:
        lines = f.readlines()
        assert lines[0].strip() == "# Filename: cube_rect_test.fits"
        assert "width=10.00, height=20.00" in lines[1]
        assert lines[2].strip() == "Velocity (km/s),Flux (K)"
