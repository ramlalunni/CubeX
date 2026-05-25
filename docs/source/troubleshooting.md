# Troubleshooting & FAQs

This page addresses common pitfalls, numerical issues, and environmental errors you may encounter while using or compiling CubeX.

## 1. Blank or "Garbage" Moment Maps
**Symptom:** You define a velocity region and select a moment map (particularly Moment 2), but the resulting image generates instantly and is entirely blank, speckled with NaN values, or contains extreme, ununphysical numbers.

**Solution:** This is a known numerical instability issue related to floating-point precision. 
* CubeX uses `numba` to severely accelerate mathematical integration. However, the one-pass variance algorithm used for Moment 2 can suffer from *catastrophic cancellation* if the velocity values are exceptionally large.
* **The Fix:** If you experience this, you can force CubeX to fall back to the mathematically stable NumPy routines. Simply uninstall Numba from your environment (`pip uninstall numba`). CubeX detects its absence at startup and automatically routes all math through stable, BLAS-accelerated NumPy tensors instead.

## 2. FITS File Crashes on Load
**Symptom:** You attempt to load a FITS file, but the viewer crashes, throws a terminal traceback, or refuses to render the spatial axes.

**Solution:** CubeX heavily relies on `astropy` and `spectral-cube` for rigorous WCS coordinate parsing. Your FITS header is likely malformed or missing critical standard keywords.
* Ensure your header contains a valid spectral axis definition (e.g., `CTYPE3 = 'FREQ'` or `'VRAD'`).
* Ensure the `RESTFRQ` or `RESTFREQ` keyword is present. If it is missing, CubeX cannot dynamically convert frequencies to the Radio Velocity (km/s) convention.
* *Note: If the spectral axis is entirely missing, CubeX's fallback logic will attempt to load the file as a flat 2D image, allowing you to still use basic spatial analysis tools.*

## 3. "Command Not Found" in WSL2 / Linux (Source Users)
**Symptom:** You have cloned the repository, set up a virtual environment, and installed `requirements.txt`. However, when you try to compile the application (`pyinstaller`) or build this documentation (`make html`), your terminal returns `command not found`.

**Solution:** This is a common pathing issue in Linux and WSL2 where the terminal cannot locate the executables installed inside your virtual environment's `bin/` directory.
* **The Fix:** Instead of calling the executables directly, invoke them through your active Python binary using the `-m` (module) flag.
* For Sphinx: run `python3 -m sphinx -b html docs/source docs/build/html` instead of `make html`.
* For PyInstaller: run `python3 -m PyInstaller` instead of `pyinstaller`.
