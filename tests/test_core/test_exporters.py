import numpy as np
from astropy.io import fits
from cubex.core.exporters import export_fits_active_core, export_spectrum_csv_core

def test_export_fits_active_core(tmp_path):
    """Test exporting a synthetic 2D array to a FITS file and verify its headers and data."""
    export_data = np.array([
        [1.0, 2.0],
        [3.0, 4.0]
    ])
    
    filename = tmp_path / "test_moment.fits"
    panel_type = "Moment 0"
    bunit = "Jy/beam km/s"
    
    # Execute the export function
    export_fits_active_core(
        filename=str(filename),
        export_data=export_data,
        panel_type=panel_type,
        bunit=bunit,
        raw_header=None,
        new_wcs_header=None,
        bmaj=0.5, bmin=0.3, bpa=45.0
    )
    
    # Verify file was created in tmp_path
    assert filename.exists()
    
    # Read the FITS file back into memory
    with fits.open(str(filename)) as hdul:
        data = hdul[0].data
        header = hdul[0].header
        
        # Verify data shape matches synthetic input
        assert data.shape == (2, 2)
        np.testing.assert_array_equal(data, export_data)
        
        # Verify metadata
        assert header['BUNIT'] == "Jy/beam km/s"
        assert header['BMAJ'] == 0.5
        assert header['BMIN'] == 0.3
        assert header['BPA'] == 45.0
        assert "Moment 0" in header['HISTORY'][0]

def test_export_spectrum_csv_core(tmp_path):
    """Test exporting spectral data to CSV format."""
    filename = tmp_path / "test_spectrum"
    parent_filename = "cube_test.fits"
    regions_to_export = ["Region 1"]
    
    # Synthetic curves
    curves = {
        "Region 1": np.array([10.5, 11.2, 9.8])
    }
    v_sorted = np.array([-5.0, 0.0, 5.0])
    roi_dict = {"Region 1": "Circle - Center(0,0)"}
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
    
    # Check if file was created (if length is 1, it just appends .csv)
    expected_file = tmp_path / "test_spectrum.csv"
    assert expected_file.exists()
    
    # Read back and verify CSV formatting
    with open(expected_file, 'r') as f:
        lines = f.readlines()
        assert lines[0].strip() == "# Filename: cube_test.fits"
        assert lines[1].strip() == "# Region: Circle - Center(0,0)"
        assert lines[2].strip() == "Velocity (km/s),Flux (K)"
        assert lines[3].strip() == "-5.0,10.5"
        assert lines[4].strip() == "0.0,11.2"
        assert lines[5].strip() == "5.0,9.8"
