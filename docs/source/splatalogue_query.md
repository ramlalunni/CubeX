# Splatalogue Line Query

## Overview
Identifying the molecular or atomic origin of spectral emission is a fundamental workflow in radio astronomy. CubeX integrates directly with the NRAO Splatalogue database via the `astroquery` package, allowing you to fetch, filter, and overlay verified rest frequencies directly onto your extracted spectra.

## Interface & Controls
To initiate a query, click the **Line Catalog** button located in the Spectrum panel's toolbar. This opens the Splatalogue Interface dialog.

1. **Frequency Range:** By default, the query bounds are strictly synchronized to the physical frequency range of your currently loaded FITS cube. You can manually widen or narrow this window.
2. **Filters:** 
   * **Chemical Name:** Filter for specific molecules (e.g., `CO`, `CH3OH`).
   * **Energy Limits:** Restrict the query to specific Upper State Energy ($E_u$) thresholds (in Kelvin).
3. **Overlay & Rendering:**
   * After hitting "Query", the results populate a selectable table.
   * Select the transitions of interest and click **Overlay**. The lines will be injected into the Spectrum plot as vertical markers, labeled with the chemical formula and transition frequency.

## Algorithmic Behavior
Because network requests to the Splatalogue API can be heavily delayed by payload sizes or server latency, CubeX offloads the `astroquery.splatalogue.Splatalogue.query_lines()` execution to a dedicated background QThread (`SplatalogueWorker`). This ensures the main UI never freezes during a query.

When the JSON/Astropy Table payload is successfully returned, the data is parsed and handed back to the main thread. When a user chooses to overlay a line, CubeX calculates the expected observed velocity/frequency for that transition using the dataset's native spectral WCS axis and drops a dynamically labeled PyQtGraph `InfiniteLine` onto the spectrum plot canvas.
