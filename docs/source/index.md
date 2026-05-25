# CubeX Documentation

Welcome to the official documentation for **CubeX**, a high-performance, interactive visualization and analysis suite for radio astronomy spectral data cubes. Built on PyQtGraph, NumPy, and Astropy, CubeX is designed to deliver seamless exploration of FITS data with mathematically rigorous spatial and spectral analytics.

```{warning}
This documentation is an experimental draft, and is currently under construction. <u>It is incomplete, and contains known errors</u>. **<span style="color: red;">DO NOT USE</span>** this documentation as an official reference until this warning is removed in a future update.
```

## Overviews

CubeX allows astronomers to seamlessly bridge the gap between 2D channel maps and 1D spectral profiles. By offering real-time moment map generation, interactive Position-Velocity (PV) slices, and integrated Splatalogue line querying, CubeX accelerates the data exploration workflow.

```{toctree}
:maxdepth: 2
:caption: Introduction & Installation

installation
```

```{toctree}
:maxdepth: 2
:caption: The User Interface

ui_overview
file_loading
```

```{toctree}
:maxdepth: 2
:caption: Core Analysis Tools

moment_maps
pv_diagrams
spectral_analysis
spatial_analysis
splatalogue_query
```

```{toctree}
:maxdepth: 2
:caption: Tutorials & Support

tutorial
troubleshooting
```