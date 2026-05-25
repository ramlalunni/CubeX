# Spectral Analysis

## Overview
CubeX’s Spectral Analysis module allows users to extract, manipulate, and analyze 1D spectral profiles from 3D data cubes. The application supports real-time extraction from complex, user-defined spatial regions and includes tools for statistical spatial collapsing, unit conversion, and spectral smoothing.

## Interface & Controls
To use the spectral analysis tools, ensure the top-right panel is set to **Spectrum** mode.

1. **Draw an Extraction Region (ROI):** 
   * Use the **ROI Type** dropdown to select a spatial shape (Whole Map, Point/Beam, Line, Rectangle, Ellipse, or Custom Polygon).
   * For Area ROIs (e.g., Rectangle, Ellipse), `Ctrl + Click and Drag` directly on the 2D Channel Map to draw your region.
2. **Multi-Region Overlay:** 
   * CubeX supports extracting and plotting multiple spectra simultaneously. Each drawn region is added to the "Regions" list in the spectrum panel.
   * Toggle the checkboxes next to each region's name to selectively hide or show their spectra on the plot.
3. **Statistical Collapsing:** 
   * For area ROIs encompassing multiple pixels, you must define how the spectrum is aggregated. Use the **Statistic** dropdown to choose between spatial *Mean*, *Median*, *Sum*, or *Max*.
4. **Spectral Smoothing:**
   * Click the **Smooth** button in the spectrum toolbar to open the Smoothing Dialog. 
   * Select a convolution kernel (e.g., Hanning, Boxcar) and apply. The smoothed spectrum will appear in a dedicated "Smoothed" tab alongside the raw data.

## Algorithmic Behavior
When a spectrum is extracted from an area ROI (like a Polygon or Ellipse), CubeX does not simply take a rectangular bounding box. Instead, it utilizes PyQtGraph's 2D rasterization engine to build a rigorous, sub-pixel boolean mask (`roi.getArrayRegion`). Pixels falling outside the geometric boundaries of the shape are securely converted to `NaN` before the chosen statistical collapse (e.g., `np.nanmean` or `np.nanmax`) is applied across the spatial axes. 

If unit conversions are requested (e.g., Jy/beam to Kelvin), the backend dynamically calculates the Rayleigh-Jeans equivalent brightness temperature using the dataset's native Rest Frequency and beam solid angle metadata extracted via Astropy.
