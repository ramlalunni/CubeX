# CubeX 🌌

**A lightweight, real-time ALMA & VLA FITS cube visualization and kinematic exploration tool.**

CubeX is a desktop application built with Python, PyQt5, and PyQtGraph designed for radio astronomers. It provides a lightning-fast, interactive interface for exploring spectral data cubes, extracting local spectra, querying molecular line databases, and dynamically generating moment maps—all without the overhead of heavier legacy software.

---

## ✨ Features

* **Interactive Channel Maps:** Fast, slice-by-slice navigation through velocity/frequency axes with media playback controls and real-time WCS coordinate tracking.
* **Dynamic Spectrum Extraction:** Draw custom Regions of Interest (Points/Beams, Circles, Rectangles, Polygons) directly on the channel map to instantly extract the local spectrum.
* **Integrated Splatalogue Queries:** Select a velocity range and query the CDMS and JPL databases natively within the app. Instantly overlay identified molecular transitions onto your extracted spectrum.
* **On-the-Fly Moment Maps:** Dynamically calculate and render Integrated Intensity (M0), Velocity Field (M1), Velocity Dispersion (M2), Peak Intensity (M8), and Peak Velocity (M9) based on your chosen velocity window.
* **Interactive Threshold Masking:** Use the visual "Dropper" tool to click background noise in the M0 map and automatically apply it as a 3D cutoff mask for M1 and M2 maps.
* **Contouring & Theming:** Auto-generate or manually specify contour levels on any map. Built-in global Dark and Light themes.
* **Publication-Ready Exports:** Export any individual panel or spectrum directly to `.fits`, `.pdf`, or `.csv` (for 1D spectra).

---

## 🛠️ Installation

CubeX is built on a modern Python scientific stack. We recommend using a virtual environment (like `conda` or `venv`) to avoid conflicts with your system packages.

### Prerequisites
Make sure you have Python 3.8+ installed on your system.

### 1. Clone the Repository
```bash
git clone [https://github.com/YOUR_USERNAME/CubeX.git](https://github.com/YOUR_USERNAME/CubeX.git)
cd CubeX
```

### 2. Run the Install Script
CubeX provides an automated installation script that detects your OS, installs necessary UI fonts (like color emojis for Linux), creates a Python virtual environment, and installs all dependencies automatically.

```bash
./install.sh
```

*Core Dependencies: `numpy`, `pandas`, `astropy`, `astroquery`, `spectral-cube`, `matplotlib`, `pyqt5`, `pyqtgraph`*

---

## 🚀 Usage

If you installed using the `install.sh` script, an executable launcher was automatically created for you. You can launch the application by running:
```bash
./CubeX.sh
```

*(Alternatively, you can manually activate your environment and run `python3 main.py`)*

### Packaging for Linux (Standalone Executable)
If you want to create a single, double-clickable application file (so you don't need to run it via the terminal or install Python dependencies on other machines), you can build it using PyInstaller:

1. Install PyInstaller: `pip install pyinstaller`
2. Build the app:
```bash
pyinstaller --name CubeX --onefile --windowed main.py
```
3. Your standalone software will be generated in the `dist/` folder. You can move this file anywhere on your computer or share it with colleagues!

---

## 🕹️ Controls & Shortcuts

CubeX uses `pyqtgraph` under the hood, which relies on standard mouse interactions for incredibly fast rendering:

* **Pan:** `Left-Click` + Drag on any map or plot.
* **Zoom:** `Right-Click` + Drag (drag left/right to scale the X-axis, up/down to scale the Y-axis).
* **Reset Zoom:** `Middle-Click` (or press the `A` key) to auto-range the view to fit the data.
* **Jump to Channel:** `Left-Click` anywhere on the 1D spectrum plot to instantly snap the channel map to that velocity.
* **Clear ROI:** Press the `ESC` key to delete the currently active (yellow) spatial extraction shape.
* **Active Panel:** `Left-Click` a panel to make it active (highlighted with a blue border). Tools like *Export* and *Draw Contours* apply to the active panel.

---

## 📁 Project Structure

If you wish to contribute or modify the codebase, CubeX uses a modular "Separation of Concerns" architecture:
```text
CubeX/
├── main.py                 # Application entry point
├── src/                    
│   ├── core/               
│   │   └── splatalogue.py  # Background QThread for API querying and text formatting
│   └── gui/                
│       ├── main_window.py  # Shell, menus, shortcuts, and tab manager
│       ├── explorer_tab.py # Core data visualization, FITS I/O, and rendering engine
│       ├── dialogs.py      # Pop-up UI forms (Line query, Line selection, Contours)
│       └── custom.py       # Custom PyQtGraph monkey-patches and UI widgets
├── assets/                 
│   └── style.qss           # Global CSS stylesheet for the GUI
├── requirements.txt        
└── README.md               
```

---

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details. 

## 🙏 Acknowledgments
Built heavily upon the fantastic work of the [Astropy Project](https://www.astropy.org/), [Spectral-Cube](https://spectral-cube.readthedocs.io/), and [PyQtGraph](https://www.pyqtgraph.org/).