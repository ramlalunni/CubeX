# UI Overview

The CubeX graphical interface is engineered to maximize screen real estate while keeping all critical analysis tools immediately accessible. The application utilizes a multi-tabbed interface, allowing you to load and explore multiple FITS data cubes simultaneously. 

## The Main Window Layout
The core workspace for any loaded cube is divided into three primary zones:

### 1. Channel Map (Top-Left)
This panel displays the 2D spatial slice of your data cube. 
* **Controls:** Includes a playback slider to step through velocity channels, and a sophisticated histogram/colormap widget to adjust image scaling dynamically.
* **Interactions:** This is your canvas for drawing spatial regions (ROIs), dropping the minimum-intensity threshold dropper, and drawing Position-Velocity (PV) cuts.

### 2. Spectral / Spatial Profiler (Top-Right)
This dynamically shifting panel adapts based on your selected mode:
* **Spectrum Mode:** Plots the 1D spectrum extracted from the ROIs drawn on the Channel Map. Contains tools for Splatalogue queries, spectral smoothing, and the blue "Velocity Range" selector.
* **Spatial Analysis Mode:** Plots the 1D spatial cross-sections (X/Y profiles) or spatial statistics for the active channel.

### 3. Auxiliary Analysis Panels (Bottom Row)
The bottom half of the screen contains three identical, independent auxiliary panels. By default, they display Moment 0, Moment 8, and Moment 1 maps. 
* Use the dropdown menus at the bottom of each panel to instantly switch them between any supported Moment Map or a Position-Velocity Diagram.

## Menus & Toolbars
* **File:** Load primary FITS cubes, load Contour Overlay FITS files, or export your current visualizer states to a PDF report.
* **View:** Toggle the visibility of the bottom auxiliary panels to maximize the primary 2D/1D viewers.
* **Info:** View the raw FITS header metadata for the active dataset.
