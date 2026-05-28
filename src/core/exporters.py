import csv
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS

def export_spectrum_csv_core(filename, parent_filename, regions_to_export, curves, v_sorted, roi_dict, spec_unit):
    """
    curves is a dict mapping name -> yData.
    roi_dict maps name -> formatted roi string.
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
    catalog_items is a list of dicts: [{'type': 'line', 'x': x_val}, {'type': 'text', 'x': x_val, 'text': text_val}]
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
