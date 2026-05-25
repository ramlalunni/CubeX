# Moment Map Analysis
## Overview
Moment maps collapse a 3D spectral data cube along its spectral (velocity/frequency) axis to create 2D representations of the gas distribution and kinematics. CubeX calculates moments using a dedicated, multi-threaded background worker, ensuring the main UI remains highly responsive even when processing large cubes.
Currently supported moments include:
* **Moment 0 (Integrated Intensity):** The sum of emission across the velocity axis, multiplied by the channel width ($dv$). 
* **Moment 1 (Velocity Field):** The intensity-weighted mean velocity.
* **Moment 2 (Velocity Dispersion):** The intensity-weighted velocity dispersion (line width).
* **Moment 8 (Peak Intensity):** The maximum intensity value along the spectral axis.
* **Moment 9 (Peak Velocity):** The velocity coordinate at which the peak intensity occurs.
## Interface & Controls
You can generate up to three independent moment maps simultaneously using the auxiliary panels at the bottom of the main window. 
1. **Velocity Range Selection:** In the top-right Spectrum panel, drag the edges of the **blue spectral region** to restrict the calculation to a specific velocity range. 
2. **Select Moment Type:** In any of the bottom panels, use the dropdown menu to select the desired moment type (e.g., *Moment 0*).
3. **Intensity Thresholding:** 
   * To prevent thermal noise from corrupting your moment maps, use the **Threshold** text box located in the panel's toolbar.
   * Alternatively, activate the **Dropper Tool** (eyedropper icon) and click a region of background noise on the main Channel Map. CubeX will automatically sample the pixel value and set it as the minimum intensity threshold.
## Algorithmic Behavior
When a moment map calculation is triggered (by moving the velocity region, changing a dropdown, or updating a threshold), CubeX initiates the following sequence:
1. **Spectral Subsetting:** The master data cube is sliced strictly to the channels bounded by the blue spectral region.
2. **Noise Masking:** The intensity threshold ($t$) is applied voxel-by-voxel to the sub-cube. Any voxel with an intensity $I_v \le t$ is converted to `NaN` to exclude it from calculations.
3. **Mathematical Integration:** 
   * **Moment 0** performs a discrete integration over the unmasked voxels, multiplying by the channel width $dv$.
   * **Moments 1 and 2** use optimized NumPy tensor dot-products to calculate the weighted velocity field and stable velocity dispersion.
4. **Rendering:** The calculated 2D arrays are routed back to the UI, mapped to the correct World Coordinate System (WCS), and rendered. Moment 1 and Moment 9 automatically default to velocity-optimized colormaps, while Moments 0, 2, and 8 inherit standard intensity colormaps.