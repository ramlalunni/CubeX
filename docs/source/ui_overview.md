# The CubeX UI

The graphical interface is engineered to maximize data visibility while keeping analytical tools readily accessible.

<!-- HTML COMMENT PLACEHOLDER: Insert screenshot of the full CubeX GUI layout -->

## Menu bar
Located at the very top of the window.
- **File:** Load primary FITS cubes, attach 2D FITS images as contour overlays, or export your current visualizer state to a PDF.
- **View:** Toggle the visibility of the bottom auxiliary panels, reset zoom states, or change color themes.
- **Info:** Inspect the raw FITS header metadata for the active dataset.

## Panels (Channel map, Spectrum, Spatial Analysis, Moment/PV)
The workspace is divided into three primary zones:

1. **Channel Map (Top-Left):** The primary 2D spatial canvas. Displays the active velocity channel or 2D image. Use this panel to adjust visual limits (via the histogram), playback velocity channels, and draw interaction ROIs.
2. **Spectral / Spatial Profiler (Top-Right):** A dynamically shifting panel that adapts based on the active mode (Spectrum or Spatial Analysis). 
3. **Auxiliary Analysis Panels (Bottom Row):** Three independent viewers at the bottom of the screen. Each can independently render any Moment Map (0, 1, 2, 8, 9) or a Position-Velocity (PV) Diagram.

## Tabs
CubeX utilizes a multi-tabbed interface. Every loaded FITS file opens in its own isolated tab ("Explorer Tab"). You can switch between independent datasets freely without losing your visual state, drawn ROIs, or loaded overlays.

## Pop-up windows
Additional analysis tools open as floating, non-blocking dialogs. 
- **Splatalogue Query:** For searching rest frequencies.
- **Channel Grid Visualizer:** A tiled viewer showing sequential velocity slices.
- **Contour Config:** Controls RMS multipliers, levels, and line styling for overlaid contours.
