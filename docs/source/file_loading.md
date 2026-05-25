# File Loading & Data Handling

CubeX is designed specifically for radio astronomy data cubes and expects files adhering to the standard FITS format. 

## Loading a Data Cube
To load a dataset into CubeX:
1. Navigate to **File > Open FITS File** in the top menu bar.
2. Select your `.fits` or `.fits.gz` file. 
3. CubeX will spawn a new tab dedicated exclusively to this dataset.

## Header Requirements & Fallback Logic
Under the hood, CubeX relies primarily on the `spectral-cube` library to read data efficiently. It attempts to parse your FITS header to identify standard celestial axes (RA, Dec) and a spectral axis (Frequency or Velocity). During this process, it automatically attempts to align the spectral axis to the **Radio Velocity convention (km/s)**.

**Graceful Fallbacks for Malformed Headers:**
Radio astronomy FITS headers are notoriously inconsistent. If your header is missing critical WCS keywords (like `CUNIT3` or `RESTFRQ`) and `spectral-cube` throws an exception, CubeX will not crash. 

Instead, it triggers a robust fallback mechanism:
1. It bypasses `spectral-cube` and uses `astropy.io.fits` to forcefully load the raw NumPy data array.
2. If the loaded data is strictly a 2D image (e.g., an already-collapsed continuum map), CubeX "mocks" a 3D structure by injecting a dummy spectral axis of length 1. 
3. This ensures that you can still use CubeX’s spatial analysis tools, colormaps, and spatial statistics on 2D images without the program failing.

## Contour Overlays
In addition to the primary cube, you can load a secondary FITS file specifically to generate contour overlays across your data (e.g., overlaying high-resolution ALMA continuum over a VLA gas cube). 
* Use **File > Open Contour Overlay FITS**.
* The secondary cube is automatically reprojected on the fly to match the WCS footprint and pixel scale of your primary cube, ensuring perfectly aligned spatial contours.
