"""
Module for exporting spectra and image data to various formats (CSV, FITS, PDF).

This module contains core functionality for writing data arrays and Astropy WCS
information out to standardized scientific formats.
"""
import csv
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS

def export_spectrum_csv_core(filename, parent_filename, regions_to_export, curves, v_sorted, roi_dict, spec_unit):
    """
    Export spectral data to a CSV file.

    Parameters
    ----------
    filename : str
        The base output filename or path.
    parent_filename : str
        The name of the parent file from which the spectra were extracted.
    regions_to_export : list of str
        List of region names to be exported.
    curves : dict
        Dictionary mapping region names to their spectral flux data (`numpy.ndarray`).
    v_sorted : numpy.ndarray
        1D array of sorted velocity values corresponding to the spectral fluxes.
    roi_dict : dict
        Dictionary mapping region names to their formatted Region of Interest (ROI) strings.
    spec_unit : str
        The unit of the spectral flux (e.g., 'Jy/beam').

    Returns
    -------
    None

    Raises
    ------
    IOError
        If there is an error writing to the specified file path.

    Notes
    -----
    The output CSV contains a header with the parent filename and region string,
    followed by columns for Velocity and Flux. If multiple regions are exported,
    each is saved to a separate file with the region name appended.
    """
    for name in regions_to_export:
        curve_data = curves[name]
        fname = f"{filename}_{name.replace(' ', '_')}.csv" if len(regions_to_export) > 1 else f"{filename}.csv"
        roi_str = roi_dict.get(name, "Whole Map")
        
        with open(fname, mode='w', newline='') as file:
            file.write(f"# Filename: {parent_filename}\n")
            file.write(f"# Region: {roi_str}\n")
            writer = csv.writer(file)
            writer.writerow(["Velocity (km/s)", f"Flux ({spec_unit})"])
            for v, f in zip(v_sorted, curve_data): 
                writer.writerow([v, f])


def export_spectrum_fits_core(base_filename, regions_to_export, curves, v_sorted, spec_unit, restfrq=None):
    """
    Export spectral data to FITS format.

    Parameters
    ----------
    base_filename : str
        The base output filename or path for the FITS file.
    regions_to_export : list of str
        List of region names to be exported.
    curves : dict
        Dictionary mapping region names to spectral flux data arrays (`numpy.ndarray`).
    v_sorted : numpy.ndarray
        1D array of sorted velocity values (in km/s).
    spec_unit : str
        The physical unit for the spectral flux (e.g., 'K', 'Jy/beam').
    restfrq : float, optional
        The rest frequency of the spectral line in Hz, by default None.

    Returns
    -------
    None

    Raises
    ------
    OSError
        If the file cannot be written to the disk.

    Notes
    -----
    This function constructs a basic 1D WCS header assuming regular spacing in 
    velocity (`VRAD`). The channel width (`CDELT1`) is calculated from the first
    two elements of `v_sorted`. If `v_sorted` is non-linear, this linear 
    approximation will introduce small errors in the WCS coordinates.
    """
    dv = v_sorted[1] - v_sorted[0] if len(v_sorted) > 1 else 1.0
    for name in regions_to_export:
        spec_sorted = curves[name]
        hdu = fits.PrimaryHDU(spec_sorted)
        hdu.header['BUNIT'] = spec_unit
        hdu.header['CTYPE1'] = 'VRAD'
        hdu.header['CUNIT1'] = 'km/s'
        hdu.header['CRPIX1'] = 1
        hdu.header['CRVAL1'] = v_sorted[0]
        hdu.header['CDELT1'] = dv
        if restfrq is not None:
            hdu.header['RESTFRQ'] = restfrq
        fname = f"{base_filename}_{name.replace(' ', '_')}.fits" if len(regions_to_export) > 1 else f"{base_filename}.fits"
        hdu.writeto(fname, overwrite=True)

def export_spectrum_pdf_core(save_path, regions_to_export, curves, v_sorted, is_single_file, include_title, base_filename, spec_unit, color_map, catalog_items):
    """
    Export spectral data as PDF plots.

    Parameters
    ----------
    save_path : str
        The base output path and filename (without extension).
    regions_to_export : list of str
        List of region names to be plotted and exported.
    curves : dict
        Dictionary mapping region names to spectral flux arrays (`numpy.ndarray`).
    v_sorted : numpy.ndarray
        1D array of sorted velocity values (in km/s).
    is_single_file : bool
        If True, plots all regions into a single PDF file. Otherwise, creates one file per region.
    include_title : bool
        Whether to include a title on the generated plot(s).
    base_filename : str
        Base string used to generate the plot title(s).
    spec_unit : str
        String representing the flux units for the y-axis label.
    color_map : dict
        Dictionary mapping region names to matplotlib color strings or hex codes.
    catalog_items : list of dict
        List of spectral line markers and labels. Each dict contains 'type' 
        ('line' or 'text') and 'x' (velocity value).

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the catalog_items dicts are malformed.

    Notes
    -----
    The plotting assumes 'mid' step plotting for spectra, which is typical for 
    radio astronomy data to represent channels as discrete bins.
    """
    if is_single_file:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.set_xlabel('Radio Velocity (km/s)')
        ax.set_ylabel(f'Flux ({spec_unit})')
        if include_title:
            ax.set_title(f"{base_filename}_spectrum")
        
        for name in regions_to_export:
            spec_sorted = curves[name]
            c = color_map.get(name, '#3498db')
            ax.step(v_sorted, spec_sorted, color=c, where='mid', linewidth=1.5, label=name)
        if len(regions_to_export) > 1:
            ax.legend()
            
        for item in catalog_items:
            if item['type'] == 'line':
                ax.axvline(x=item['x'], color='#e74c3c', linestyle='--', linewidth=1)
            elif item['type'] == 'text':
                ymax = np.nanmax(curves[regions_to_export[0]]) if regions_to_export else 1.0
                ax.text(item['x'], ymax, item['text'], color='#e74c3c', rotation=90, verticalalignment='top', horizontalalignment='right')
        
        fname = f"{save_path}.pdf"
        plt.savefig(fname, format='pdf', bbox_inches='tight')
        plt.close(fig)
    else:
        for name in regions_to_export:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.set_xlabel('Radio Velocity (km/s)')
            ax.set_ylabel(f'Flux ({spec_unit})')
            if include_title:
                ax.set_title(f"{base_filename}_spectrum_{name.replace(' ', '_').lower()}")
            
            spec_sorted = curves[name]
            c = color_map.get(name, '#3498db')
            ax.step(v_sorted, spec_sorted, color=c, where='mid', linewidth=1.5, label=name)
            ax.legend()
            
            for item in catalog_items:
                if item['type'] == 'line':
                    ax.axvline(x=item['x'], color='#e74c3c', linestyle='--', linewidth=1)
                elif item['type'] == 'text':
                    ymax = np.nanmax(spec_sorted)
                    ax.text(item['x'], ymax, item['text'], color='#e74c3c', rotation=90, verticalalignment='top', horizontalalignment='right')
            fname = f"{save_path}_{name.replace(' ', '_')}.pdf"
            plt.savefig(fname, format='pdf', bbox_inches='tight')
            plt.close(fig)

def export_fits_active_core(filename, export_data, panel_type, bunit, raw_header, new_wcs_header, bmaj, bmin, bpa):
    """
    Export 2D or 3D active map data to a FITS file.

    Parameters
    ----------
    filename : str
        Output filename for the exported FITS file.
    export_data : numpy.ndarray
        The array of image or map data to export.
    panel_type : str
        Type of the panel (e.g., 'PV Diagram', 'Moment 0') used in the history.
    bunit : str
        The physical unit of the data array.
    raw_header : astropy.io.fits.Header or dict, optional
        Original header containing telescope and observation metadata.
    new_wcs_header : astropy.io.fits.Header or dict, optional
        New WCS header keywords to update the spatial/spectral coordinates.
    bmaj : float, optional
        Beam major axis.
    bmin : float, optional
        Beam minor axis.
    bpa : float, optional
        Beam position angle.

    Returns
    -------
    None

    Raises
    ------
    OSError
        If the file cannot be written to disk.

    Notes
    -----
    If the `panel_type` is not "PV Diagram", beam parameters are explicitly 
    written to the header if provided. The header retains observational 
    metadata like 'RESTFRQ' and 'OBSERVER' from the raw dataset.
    """
    hdu = fits.PrimaryHDU(export_data)
    
    if raw_header is not None:
        for key in ['OBJECT', 'TELESCOP', 'INSTRUME', 'OBSERVER', 'DATE-OBS', 'RESTFRQ', 'RADESYS', 'EQUINOX']:
            if key in raw_header:
                hdu.header[key] = raw_header[key]
    hdu.header['HISTORY'] = f"Exported from CubeX: {panel_type}"
    
    if new_wcs_header is not None:
        hdu.header.update(new_wcs_header)
        
    hdu.header['BUNIT'] = bunit
    if panel_type != "PV Diagram":
        if bmaj is not None: hdu.header['BMAJ'] = bmaj
        if bmin is not None: hdu.header['BMIN'] = bmin
        if bpa is not None: hdu.header['BPA'] = bpa
        
    hdu.writeto(filename, overwrite=True)

def export_pdf_active_core(filename, plot_data, task, cbar_label, cmap_name, levels, include_title, base_filename, wcs_2d, contour_params, is_absolute_wcs, extent):
    """
    Export active 2D maps (e.g., moments, PV diagrams) to PDF.

    Parameters
    ----------
    filename : str
        The output path and filename for the PDF.
    plot_data : numpy.ndarray
        2D array of data values to be plotted.
    task : str
        Task or panel name (e.g., 'Moment 0') used for the title.
    cbar_label : str
        Label for the colorbar.
    cmap_name : str
        Matplotlib colormap name.
    levels : tuple or list
        Minimum and maximum values for the colormap scaling, (vmin, vmax).
    include_title : bool
        Whether to include a title on the plot.
    base_filename : str
        Base filename used to construct the title.
    wcs_2d : astropy.wcs.WCS, optional
        A 2D WCS object used for proper coordinate projection if `is_absolute_wcs` is True.
    contour_params : dict or None
        Dictionary containing contour configurations. May contain keys like 'mode', 'n', and 'levels'.
    is_absolute_wcs : bool
        If True, plots using absolute sky coordinates (Right Ascension/Declination) via `wcs_2d`.
    extent : list or tuple
        The bounding box in data coordinates [xmin, xmax, ymin, ymax] used if `is_absolute_wcs` is False.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If `is_absolute_wcs` is True but `wcs_2d` is not provided.

    Notes
    -----
    When using relative coordinates (`is_absolute_wcs` = False), the image 
    is drawn using RA/DEC offsets in arcseconds. If contours are set to 'auto',
    they are generated linearly between the non-NaN minimum and maximum values
    of the image.
    """
    if is_absolute_wcs and wcs_2d is not None:
        fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={'projection': wcs_2d})
        if include_title:
            ax.set_title(f"{base_filename}_{task}")
        im = ax.imshow(plot_data, origin='lower', cmap=cmap_name, vmin=levels[0], vmax=levels[1])
        ax.set_xlabel('Right Ascension')
        ax.set_ylabel('Declination')
        
        if contour_params:
            if contour_params['mode'] == 'auto':
                valid = plot_data.T[~np.isnan(plot_data.T) & ~np.isinf(plot_data.T)]
                if len(valid) > 0:
                    min_v, max_v = np.nanmin(valid), np.nanmax(valid)
                    c_levels = np.linspace(min_v, max_v, contour_params['n'] + 2)[1:-1]
                    ax.contour(plot_data, levels=c_levels, colors='#2ecc71', linewidths=1.0)
            else:
                ax.contour(plot_data, levels=contour_params['levels'], colors='#2ecc71', linewidths=1.0)
    else:
        fig, ax = plt.subplots(figsize=(8, 6))
        if include_title:
            ax.set_title(f"{base_filename}_{task}")
        im = ax.imshow(plot_data, origin='lower', cmap=cmap_name, vmin=levels[0], vmax=levels[1], extent=extent)
        ax.set_xlabel('RA offset (arcsec)')
        ax.set_ylabel('DEC offset (arcsec)')
        
        if contour_params:
            if contour_params['mode'] == 'auto':
                valid = plot_data.T[~np.isnan(plot_data.T) & ~np.isinf(plot_data.T)]
                if len(valid) > 0:
                    min_v, max_v = np.nanmin(valid), np.nanmax(valid)
                    c_levels = np.linspace(min_v, max_v, contour_params['n'] + 2)[1:-1]
                    ax.contour(plot_data, levels=c_levels, colors='#2ecc71', linewidths=1.0, extent=extent)
            else:
                ax.contour(plot_data, levels=contour_params['levels'], colors='#2ecc71', linewidths=1.0, extent=extent)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)
    
    plt.savefig(filename, format='pdf', bbox_inches='tight')
    plt.close(fig)
