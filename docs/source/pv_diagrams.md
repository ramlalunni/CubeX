# Position-Velocity (PV) Diagrams
## Overview
Position-Velocity (PV) diagrams are essential tools for diagnosing the kinematic structure of astronomical sources, such as rotating disks or outflows. A PV diagram takes a 1D spatial "cut" across the 2D sky plane and plots the spectral intensity along that cut against the velocity axis, effectively creating a 2D position-velocity image.
## Interface & Controls
To generate a PV diagram in CubeX, you must first define a spatial cut on the Channel Map, and then configure an auxiliary panel to display the results.
1. **Draw a PV Cut:**
   * In the top-left Channel Map panel, change the interaction mode to **PV Diagram**.
   * `Ctrl + Click and Drag` to draw a line segment across the region of interest. The segment will appear as a magenta line. 
   * You can adjust the endpoints of the cut dynamically at any time.
2. **Configure the Panel:**
   * In one of the three bottom auxiliary panels, select **PV Diagram** from the dropdown menu.
   * **Select Cut:** If multiple cuts are drawn, use the secondary dropdown to choose which cut (e.g., *Cut 1*) this panel should evaluate.
   * **Adjust Cut Width:** Use the width spin-box to average emission perpendicular to the cut. Increasing the width (in pixels) can drastically improve the Signal-to-Noise Ratio (SNR) of faint kinematic structures.
   * **Data Range Toggle:** Choose whether the PV diagram should be generated using the **Full Cube** (all velocity channels) or restricted to the **Current Range** defined by the blue spectral region.
## Algorithmic Behavior
When a PV diagram is requested, CubeX delegates the heavy lifting to the `MomentWorker` background thread:
1. **Coordinate Resolution:** The pixel endpoints of the drawn line segment are converted to rigorous World Coordinates (Arcseconds) using the active Astropy WCS header.
2. **Bilinear Interpolation:** The backend calculates the exact number of sub-pixel samples required along the physical length of the cut. It then performs rapid, pure-NumPy bilinear interpolation across the 3D cube to extract the intensities without altering the native grid.
3. **Spatial Averaging:** If the cut width is set $> 1$, CubeX generates parallel, offset interpolation tracks perpendicular to the primary line cut, averaging the extracted spectra before displaying them.
4. **Axis Mapping:** The final 2D array is plotted with the X-axis representing the physical offset along the cut (in arcseconds) and the Y-axis representing Radio Velocity (in km/s).