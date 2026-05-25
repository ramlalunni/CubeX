# Quickstart Tutorial: Exploring an AGB Star

This tutorial walks you through a standard, end-to-end data exploration workflow in CubeX. 

**Scenario:** You have just downloaded a high-resolution ALMA FITS data cube observing the expanding circumstellar envelope of an Asymptotic Giant Branch (AGB) star. Your goal is to isolate the expanding shell's emission, calculate an integrated intensity map, and extract a clean 1D spectrum of the outflow.

## Step 1: Loading & Inspecting the Data
1. Launch CubeX and navigate to **File > Open FITS File**.
2. Select your ALMA `.fits` file. CubeX will automatically parse the World Coordinate System (WCS) and load the 2D Channel Map on the left and the global 1D Spectrum on the right.
3. To verify the calibration and beam size of your ALMA data, open the **Info** or **File Header** menu to inspect the raw FITS metadata (look for `BMAJ`, `BMIN`, and `RESTFRQ`).

## Step 2: Masking Thermal Noise
The Moment maps will integrate background thermal noise if we don't set a noise floor.
1. Locate a region on the **Channel Map** that is clearly empty sky (no source emission).
2. Look at the bottom auxiliary panel where you intend to generate your moment map.
3. Click the **Dropper Tool** (eyedropper icon) in that panel's toolbar, then click on the empty sky region in the Channel Map. 
4. The **Threshold** text box will automatically populate with a safe minimum intensity value. All pixels below this value will now be masked (`NaN`) during calculations.

## Step 3: Generating a Moment 0 Map
Now we want to map the total integrated intensity of the expanding envelope.
1. In the top-right **Spectrum** panel, look for the vertical blue shaded region.
2. Drag the left and right edges of this blue region to bound only the velocity range where the AGB star's emission is visible (e.g., -50 km/s to +50 km/s).
3. In the bottom-left auxiliary panel, select **Moment 0 (Integrated Intensity)** from the dropdown menu.
4. CubeX will instantly calculate and display the 2D integrated intensity map of the envelope, properly applying the channel width ($dv$) and your noise threshold.

## Step 4: Extracting a Regional Spectrum
Finally, let's extract the kinematic profile of a specific clump within the envelope.
1. Ensure the top-right panel is set to **Spectrum** mode.
2. Above the Channel Map, select **Ellipse** from the Spatial Tool / ROI dropdown.
3. Hold `Ctrl`, then **Click and Drag** on the Channel Map to draw an ellipse directly over the bright clump in your Moment 0 or Channel Map.
4. The top-right plot will instantly update, extracting and plotting the 1D spectrum strictly from the pixels inside your drawn ellipse.
