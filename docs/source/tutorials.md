# CubeX Tutorials

This section outlines practical, step-by-step workflows for common tasks in CubeX.

## 1. Extracting a Spectrum and Querying Lines

1. **Load Data:** Go to `File > Open FITS` and load a 3D data cube.
2. **Draw a Region:** In the top-right panel, ensure the mode is set to **Spectrum**. Use the ROI dropdown to select `Ellipse` or `Custom Polygon`.
3. **Select Area:** Left-click and drag on the top-left Channel Map to draw your region over a source of emission.
4. **View Spectrum:** The extracted 1D spectrum will immediately populate in the top-right panel.
5. **Identify Lines:** Click the `Splatalogue` button in the Spectrum panel. Enter a maximum energy limit (e.g., `500` K) and chemical species (e.g., `CO, CN`), then execute the search. 
6. Identified rest frequencies will appear as vertical dashed lines overlaid on your spectrum.

## 2. Generating a PV Diagram

1. **Change Panel Mode:** On any of the bottom three panels, click the dropdown menu (usually defaulting to a Moment map) and select **PV Diagram**.
2. **Enable Drawing:** In the Channel Map panel, change the interaction mode dropdown to `Draw PV Cut`.
3. **Draw the Cut:** Hold `Ctrl` (or `Cmd` on Mac) and click-and-drag across the channel map to define the spatial slice.
4. **Adjust Width:** In the bottom panel's controls, increase the `Width (px)` spinbox to average a wider spatial area.
5. **View:** The PV diagram will update in real-time as you drag the endpoints of the slice on the channel map.

<!-- HTML COMMENT PLACEHOLDER: Insert screenshot showing the PV cut line being drawn across a galactic disk in the channel map -->

## 3. Creating and Exporting a Moment 0 Map

1. **Define Velocity Range:** In the top-right Spectrum panel, click and drag the edges of the translucent blue region to tightly bracket the spectral line of interest. This defines $\Delta V$.
2. **Set Noise Floor:** In the top-left Channel Map, drag the horizontal yellow threshold line on the histogram to sit just above the background noise level.
3. **View Map:** The Moment 0 panel at the bottom will automatically calculate and display the integrated intensity.
4. **Export:** Click the `Save FITS` button located at the bottom of the Moment 0 panel to save the 2D map to your disk.
