# Spatial Analysis

## Overview
While the default UI focuses on mapping the spectral domain, the **Spatial Analysis** mode flips this paradigm to strictly evaluate the spatial structure of the data at the currently viewed velocity channel. This mode provides instantaneous 1D cross-sectional profiles and computes high-precision spatial statistics for customized regions of interest.

## Interface & Controls
To activate this feature, locate the main dropdown in the top-right panel (defaulted to *Spectrum*) and switch it to **Spatial Analysis**. 

1. **Profile Slicing (Point & Line Tools):**
   * **Point:** Click anywhere on the Channel Map to drop a point. The right-hand panel will instantly split into two plots displaying the orthogonal **X Profile** (RA slice) and **Y Profile** (Dec slice) intersecting that exact point.
   * **Line:** Draw a line segment to extract a 1D spatial profile along an arbitrary angle. The panel updates to show the intensity as a function of physical distance (in arcseconds) along the cut.
2. **Area Statistics (Rectangle, Ellipse, Polygon):**
   * Draw a closed geometric shape over a feature in the Channel Map.
   * Instead of plotting a profile, the right-hand panel will display a comprehensive statistical readout for the pixels enclosed by the shape.

## Algorithmic Behavior
Spatial statistics are evaluated strictly on the 2D array of the currently active velocity channel. When an area ROI is drawn, the backend applies a stringent geometric mask—clipping all external background pixels to `NaN`. 

Once isolated, CubeX calculates the following spatial metrics for the valid data footprint:
* **Mean:** The average intensity (`np.nanmean`).
* **Sum:** The total integrated flux within the region (`np.nansum`).
* **Peak & Min:** The absolute maximum and minimum voxel values.
* **RMS:** The Root Mean Square of the region (`sqrt(mean(v^2))`).
* **Standard Deviation:** The spread of the intensity distribution (`np.nanstd`).

*(Note: If contour overlays are actively rendered on the channel map, CubeX will automatically extract and plot the spatial profiles of the overlay data alongside your primary FITS cube).*
