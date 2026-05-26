# FAQs and Troubleshooting

## Common Issues

### 1. My FITS file won't open or crashes the app.
**Cause:** The FITS header is likely missing strictly required structural keywords. CubeX requires at least a basic `NAXIS` structure. 
**Solution:** Open the FITS file using a script (e.g., `astropy.io.fits`) and verify that the header contains standard keywords. Try stripping extraneous non-standard History or Comment cards that might be poorly formatted.

### 2. The spectral axis is showing pixel numbers instead of Velocity or Frequency.
**Cause:** CubeX cannot parse the World Coordinate System (WCS) for the 3rd axis.
**Solution:** Ensure your FITS header contains `CRVAL3`, `CRPIX3`, and `CDELT3`. Additionally, `CTYPE3` must be defined (e.g., `VRAD`, `VOPT`, `FREQ`). If it is a frequency axis, a `RESTFREQ` keyword must also be present to convert it to velocity.

### 3. The Point (Beam) ROI tool is the size of a single pixel.
**Cause:** The FITS header is missing beam size information.
**Solution:** CubeX looks for the `BMAJ` and `BMIN` keywords in the primary header. If they are absent, CubeX cannot physically scale the beam ellipse and defaults to 1 pixel.

### 4. Splatalogue queries are returning zero results.
**Cause:** Several possibilities exist:
* Your specified $E_{max}$ (Upper State Energy) is too low.
* The molecule name string does not match the database exactly (e.g., searching "Carbon Monoxide" instead of "CO").
* The velocity range defined by the blue region in the spectrum panel corresponds to a rest frequency range that contains no known transitions for that species.

### 5. Moment maps are entirely blank (NaN).
**Cause:** The noise threshold is set too high.
**Solution:** Moment maps strictly mask out any voxels below the yellow threshold line located in the channel map's histogram. Drag the yellow line down to include more emission in the calculation.

<!-- HTML COMMENT PLACEHOLDER: Insert screenshot showing the yellow threshold line in the histogram being adjusted -->
