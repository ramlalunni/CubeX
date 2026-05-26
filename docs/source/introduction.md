# Introduction to CubeX

Welcome to **CubeX**, a high-performance, interactive visualization and analysis suite engineered for radio astronomy spectral data cubes. Built to handle modern, high-resolution datasets, CubeX provides astronomers with a powerful interface with all the necessary tools to view and analyze interferometric cubes and images.

## Core Capabilities

CubeX is designed with speed and scientific accuracy at its core. Its capabilities include:

- **Interactive 3D Data Exploration:** Rapidly navigate through massive FITS data cubes using real-time playback sliders and sophisticated histogram/colormap controls.
- **Dynamic Moment Mapping:** Generate 2D physical maps (Moment 0, 1, 2, 8, 9) instantly, with real-time updates as you adjust intensity thresholds and spectral bounding regions.
- **Position-Velocity (PV) Diagrams:** Draw custom line segments on the spatial map to extract and visualize velocity slices along complex astronomical structures.
- **Integrated Line Identification:** Query the NRAO Splatalogue database directly within the app to overlay molecular and atomic rest frequencies on extracted spectra.
- **Spatial Profiling:** Draw sub-pixel accurate regions of interest (ROIs) to calculate spatial statistics or cross-sections of the active velocity channel.
- **Publication-Ready Export:** Export visual states as high-quality PDF reports, and extract spectral data to CSV files for external analysis.

## Under the Hood
CubeX is built using a robust Python technology stack:
* **PyQtGraph / PyQt5:** For the high-performance, responsive GUI.
* **NumPy / Numba:** For mathematically rigorous and accelerated array computations.
* **Astropy / Spectral-Cube:** For strict WCS parsing, celestial coordinate transformations, and FITS I/O.

<!-- HTML COMMENT PLACEHOLDER: Insert a screenshot showcasing the full CubeX interface with a loaded FITS file, demonstrating the Channel Map, Spectrum Panel, and Moment Maps -->
