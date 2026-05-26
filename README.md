# CubeX

[![Documentation Status](https://readthedocs.org/projects/cubex/badge/?version=latest)](https://cubex.readthedocs.io)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![OS: Linux](https://img.shields.io/badge/OS-Linux-orange.svg?logo=linux)](https://github.com/ramlalunni/CubeX/releases)
[![OS: macOS](https://img.shields.io/badge/OS-macOS-lightgrey.svg?logo=apple)](https://github.com/ramlalunni/CubeX/releases)
[![GitHub issues](https://img.shields.io/github/issues/ramlalunni/CubeX.svg)](https://github.com/ramlalunni/CubeX/issues)

## Overview
CubeX is a lightweight, real-time interferometric FITS cube visualization and kinematic exploration tool designed for mm/sub-mm/radio astronomers. Built on a modern Python scientific stack with `PyQt5` and `PyQtGraph`, it offers a fast, interactive interface for navigating complex spectral data cubes without the heavy computational overhead of traditional software packages.

CubeX streamlines advanced interferometric data visualisation and analysis by providing dynamic, on-the-fly moment maps and position-velocity (PV) diagram generation, along with standard spectral extraction, spatial analysis, and statistics tools, all alongside a fast image cube renderer. Users can seamlessly interact with and process large datasets, extract local spectra from custom spatial regions, and query molecular transitions from the CDMS and JPL databases directly within the CubeX UI.

## Installation

CubeX is available as a packaged application for Linux and macOS, or can be installed as a standard Python package via `pip`.
> [!WARNING]
> **NOTE: The standalone macOS application will be provided soon. Currently only a Linux executable is available**.

### Option 1: Standalone Application (Recommended)
For users who prefer not to manage Python environments or dependencies, we provide pre-compiled, double-click executables for Linux and macOS. These bundle the entire Python ecosystem into a single app.

1. Navigate to the [Releases](https://github.com/ramlalunni/CubeX/releases) page.
2. Download the latest version for your operating system (e.g., `.dmg` or `.app.zip` for macOS, or the binary for Linux).
3. Extract and double-click to launch the CubeX GUI.

> **OS Notes:**
> - **macOS:** On your first launch, Apple's Gatekeeper may block the app. Simply right-click the app icon and select **"Open"** to bypass the warning.  
> - **Linux:** Depending on your distribution, you may need to make the file executable before double-clicking. You can do this by right-clicking the file -> Properties -> Permissions -> "Allow executing file as program", or via terminal: `chmod +x cubex-linux-binary`.

### Option 2: Install via Python (`pip`)
CubeX is designed for modern Python environments. It is highly recommended to use a virtual environment (`venv` or `conda`) for installation. If you prefer to run CubeX within your own `conda` or another `venv` environment:

```bash
git clone https://github.com/ramlalunni/CubeX.git
cd CubeX
pip install -r requirements.txt
```

Linux users may need to install system `Qt5` libraries (e.g., `sudo apt install python3-pyqt5` or `libxcb-*` packages) if UI rendering fails.  
*(Note: You can also use the included `./install.sh` script to install CubeX after automatically configuring a standalone Python environment and installing all dependencies, on Linux/MacOS systems).*

## Quickstart
After installation, you can launch the CubeX GUI directly from your terminal:

```bash
./CubeX.sh
```
*(Or manually run `python3 main.py` in your activated Python environment, from the CubeX parent directory).*

## Documentation
The comprehensive CubeX User Manual, including UI tutorials and details of the background methods CubeX uses, is hosted on ReadTheDocs, at
📖 **[CubeX Documentation](https://cubex.readthedocs.io)**.

> [!NOTE]
> The current documentation is an experimental draft, and is under construction. <u>It is incomplete, and contains known errors</u>. **<span style="color: red;">DO NOT USE</span>** this documentation as an official reference until this warning is removed in a future update.


## Contributing & Support
We welcome community contributions. If you encounter a bug or have a feature request, please [open an issue](https://github.com/ramlalunni/CubeX/issues) on GitHub. Pull requests for new analysis tools or optimizations are also encouraged.

## License
CubeX is free and open-source software distributed under the [GNU GPL 3.0 License](LICENSE).

**Disclaimer:** CubeX has been developed and tested for radio interferometric data analysis and visualisation, but is provided "as is" and without warranty of any kind, compliant with GPLv3.0.

## Acknowledgements
CubeX is built on the open-source scientific Python ecosystem, particularly [`astropy`](https://www.astropy.org/), [`pyqtgraph`](https://www.pyqtgraph.org/), and [`numpy`](https://numpy.org/). Manually supervised agentic AI assistants ([`Google Gemini 3`](https://gemini.google.com/) and [`DeepSeek v4`](https://www.deepseek.com/)) were utilized during development for code refactoring, boilerplate UI generation, and documentation drafting.

For any queries, contact the developer at ramlalunni@gmail.com.