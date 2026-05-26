# Working with CubeX

## Opening FITS files

CubeX leverages the `spectral-cube` and `astropy` libraries to parse Standard FITS files.

### 3D image cubes
When a 3D FITS cube (RA, Dec, Spectral) is loaded, CubeX attempts to parse the WCS (World Coordinate System) to establish celestial coordinates and velocity/frequency spacing. The cube is immediately rendered in the Channel Map, and the spectral axis is built.

### 2D images
CubeX also supports 2D FITS images (RA, Dec). In this mode, spectral tools (like the Spectrum panel, Splatalogue querying, and Moment Mapping) are automatically disabled.

### Supported Data & Header Requirements
To prevent crashes and ensure accurate physical units, CubeX relies on specific FITS header keywords. 
* **Spectral Axis:** The spectral axis must be defined. CubeX looks for `CRVAL3` (reference value) and `CDELT3` (increment). It attempts to parse `CTYPE3` to identify if the axis is velocity (`VRAD`, `VOPT`) or frequency (`FREQ`). If WCS parsing fails or keywords are missing, CubeX falls back to simple pixel indices for the Z-axis.
* **Beam Properties:** For physical scaling and Point ROI (Beam) sizing, the header should contain `BMAJ` (Beam Major Axis) and `BMIN` (Beam Minor Axis), typically in degrees. `BPA` (Beam Position Angle) is also read. If missing, CubeX defaults the beam size to 1 pixel.
* **Rest Frequency:** To convert between Frequency and Velocity, CubeX requires a rest frequency. It checks the header for `RESTFREQ` or `RESTFRQ`. If neither is found, velocity/frequency conversions may be unavailable.

<!-- HTML COMMENT PLACEHOLDER: Insert screenshot of the FITS Header Info pop-up dialog -->

## Analysis Tools

### Raster rendering
Raster images are rendered using `pyqtgraph.ImageView`. The histogram tool dynamically maps the underlying NumPy float array to an 8-bit colormap (e.g., Turbo, Viridis). 

### Contour overlays
You can overlay a secondary 2D FITS file as contours onto the primary channel map. CubeX automatically reprojects the overlay WCS onto the primary cube's WCS using `astropy.wcs` to ensure perfect celestial alignment, regardless of differing pixel scales.

### Channel map grids
The Channel Grid Visualizer extracts sequential velocity slices and renders them as a grid. 

### Spectral Analysis

**Scientific Context:** Spectral analysis is fundamental in radio astronomy. By examining the intensity of emission as a function of velocity (or frequency) along a line of sight, astronomers can determine the chemical composition, temperature, density, and kinematics of astronomical objects. Extracted 1D spectra reveal emission and absorption lines, which are the fingerprints of molecular clouds and galaxies.

**Features & Implementation:**
Extracts 1D spectral profiles from spatial regions of interest (ROIs). For a drawn polygonal or elliptical ROI, CubeX identifies all internal pixels and calculates the mean spectra, ignoring `NaN` values.

* **Spectral Smoothing:** High-resolution spectra are often dominated by instrumental or thermal noise. CubeX provides three smoothing algorithms to enhance the Signal-to-Noise Ratio (SNR):
  * **Boxcar (Uniform) Filter:** A simple moving average, implemented via `scipy.ndimage.uniform_filter1d`.
  * **Gaussian Filter:** Convolves the spectrum with a Gaussian kernel, controlled by a standard deviation ($\sigma$), implemented via `scipy.ndimage.gaussian_filter1d`.
  * **Savitzky-Golay Filter:** Fits a polynomial to a sliding window to smooth data without heavily degrading peak heights, implemented via `scipy.signal.savgol_filter`.
* **Spectral Statistics:** Calculated over a user-defined velocity range (the blue highlighted region $\Delta V$):
  * **Integrated Intensity (Area):** The total flux over the velocity range: $\int_{V_{min}}^{V_{max}} S_v dv$
  * **RMS Noise:** Calculated over the selected region: $\sqrt{\frac{1}{N}\sum_{i=1}^{N} (S_i - \bar{S})^2}$
  * **Peak (Max):** The maximum flux density within the region, and the velocity at which it occurs.
  * **SNR (Peak/RMS):** The ratio of the peak signal to the calculated RMS noise.
  * **Mean & Median:** Statistical averages of the flux densities in the region.
  * **Std. Deviation:** $\sigma = \sqrt{\frac{1}{N-1}\sum (S_i - \bar{S})^2}$
  * **Sum:** Simple numerical sum of the flux values.

### Splatalogue Queries

**Scientific Context:** Interstellar space is populated by a vast array of complex molecules, each with a unique quantum mechanical "fingerprint" of rotational transitions. The NRAO Splatalogue is a comprehensive database of these rest frequencies. By querying it, astronomers can identify unknown spectral lines and confirm the presence of specific molecules in their observations.

**Features & Implementation:**
* **Algorithm / Implementation:** CubeX uses an asynchronous background thread (`SplatalogueWorker`) to execute queries via the `astroquery.splatalogue` API. 
* It queries within the frequency range (`fmin`, `fmax`) corresponding to the current velocity limits, requesting `line_lists` (e.g., CDMS, JPL). 
* Responses are parsed into a Pandas DataFrame. The tool filters out lines exceeding a user-defined Upper State Energy (`e_max` in Kelvin), and matches against comma-separated chemical species strings.
* Identical lines within a 0.0001 GHz resolution tolerance are de-duplicated to prevent chart clutter.
* Chemical formulas are parsed to translate HTML tags into native Unicode sub/superscripts for clean GUI rendering.

### Spatial Analysis

**Scientific Context:** While spectra give us kinematic and chemical depth, spatial analysis allows us to measure the physical size, morphology, and total flux of a source on the sky. Extracting profiles across a galaxy or molecular core can reveal structural asymmetries, density gradients, or the presence of multiple unresolved sources.

**Features & Implementation:**
Allows drawing 1D or 2D regions (Lines, Ellipses, Polygons) on the channel map to extract spatial statistics.
* **Spatial Statistics:** Inside a drawn ROI, CubeX computes the following on the currently active velocity channel:
  * **Mean & Median Intensity**
  * **Max (Peak) Intensity**
  * **Flux Density:** The total flux integrated over the ROI, calculated as the sum of pixel intensities divided by the beam area (if `BMAJ` and `BMIN` are present): $S_{\nu} = \frac{\sum I_i}{\text{Pixels per Beam}}$
* **Sub-pixel Masking:** To eliminate statistical bias caused by zero-filling rectangular bounding boxes, CubeX employs strict sub-pixel masking for curved ROIs (Ellipse, Polygon). Only valid data inside the true geometry is included in statistical calculations. 

### Moment Maps

**Scientific Context:** 3D spectral data cubes are difficult to visualize in their entirety. Moment maps condense this 3D information into 2D physical maps based on the moments of the spectral line profile. 
* **Moment 0:** Shows the total column density (how much gas is there?).
* **Moment 1:** Shows the line-of-sight velocity field (is it rotating or flowing?).
* **Moment 2:** Shows the velocity dispersion (is it turbulent, or are there multiple velocity components along the line of sight?).

**Features & Implementation:**
Moment maps condense 3D spectral information into 2D physical maps. CubeX computes these on a background thread (`MomentWorker`) using a NumPy/BLAS implementation over a bounded velocity region ($\Delta V$). 
* **Intensity Masking:** A noise threshold ($T_{B, min}$) is strictly applied voxel-by-voxel using the yellow threshold slider in the Channel Map histogram. Voxels below the threshold are masked as `NaN` and ignored in calculations.
* **Moment 0 (Integrated Intensity):** Computed via a simple sum multiplied by the velocity channel width $dv$: 
  $$M_0 = \sum_{i} T_i \cdot dv$$
  Implementation: `np.nansum(mc, axis=0) * dv`
* **Moment 1 (Intensity-Weighted Velocity field):** Computed using tensor contraction to avoid large intermediate memory allocations:
  $$M_1 = \frac{\sum_{i} T_i \cdot v_i}{\sum_{i} T_i}$$
  Implementation: `np.tensordot(v_axis, mc_nz, axes=([0], [0])) / m0_safe`.
* **Moment 2 (Velocity Dispersion):** Calculated using the computational variance formula:
  $$M_2 = \sqrt{\frac{\sum_{i} T_i \cdot v_i^2}{\sum_{i} T_i} - M_1^2}$$
  Implementation: `np.sqrt(np.maximum(sum_wv2 / m0_safe - m1**2, 0.0))` via tensordot.
* **Moment 8 (Maximum Intensity):** 
  Implementation: `np.nanmax(mc, axis=0)`
* **Moment 9 (Velocity of Maximum Intensity):** 
  Implementation: Identifies the index of the max along the spectral axis and returns `v_axis[argmax]`.

<!-- HTML COMMENT PLACEHOLDER: Insert screenshot of the 3 bottom panels displaying different Moment Maps -->

### PV Diagrams

**Scientific Context:** A Position-Velocity (PV) Diagram is a 2D slice through a 3D data cube, where one axis represents spatial offset along a drawn line, and the other represents velocity. This is the primary tool for analyzing the kinematics of rotating disks (e.g., Keplerian rotation in protoplanetary disks or galactic rotation curves) and identifying outflows or expanding shells.

**Features & Implementation:**
Position-Velocity (PV) Diagrams plot velocity/frequency against a spatial offset along a user-drawn slice.
* **Interpolation:** CubeX utilizes a pure-NumPy bilinear interpolation kernel. For a line segment defined by $(x_0, y_0)$ and $(x_1, y_1)$, it calculates $N$ sample points spaced by the pixel scale. At each point, it gathers the 4 nearest spatial pixels and computes the weighted average:
  $$V_{interp} = (1-f_x)(1-f_y)P_{00} + f_x(1-f_y)P_{10} + (1-f_x)f_y P_{01} + f_x f_y P_{11}$$
* **Width Averaging:** If the cut width $>1$ (controlled via the Width spinbox), CubeX generates parallel offset lines, interpolates them individually, and computes the `nanmean` across the perpendicular axis to increase the signal-to-noise ratio.
*(Note: A Numba-accelerated kernel is available and seamlessly takes over the bilinear interpolation if `numba` is installed in the Python environment, significantly reducing execution time by compiling the loop in C).*

<!-- HTML COMMENT PLACEHOLDER: Insert screenshot showing a PV Diagram cut on the channel map and the resulting PV plot -->

## Exporting Data and Results

### FITS files
While CubeX does not modify the primary data cube, users can export generated Moment Maps or PV diagrams as new FITS files for external use.

### Spectra CSVs
1D spectrum plots can be exported to comma-separated values (CSV) files.

### PDF plots
Users can export the exact state of the GUI viewers to high-resolution vector PDF reports, which embed relevant plot titles, units, and colormaps.
