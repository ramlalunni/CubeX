# CubeX

[![Documentation Status](https://readthedocs.org/projects/cubex/badge/?version=latest)](https://cubex.readthedocs.io)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![OS: Linux](https://img.shields.io/badge/OS-Linux-orange.svg?logo=linux)](https://github.com/ramlalunni/CubeX/releases)
[![OS: macOS](https://img.shields.io/badge/OS-macOS-lightgrey.svg?logo=apple)](https://github.com/ramlalunni/CubeX/releases)
[![GitHub issues](https://img.shields.io/github/issues/ramlalunni/CubeX.svg)](https://github.com/ramlalunni/CubeX/issues)

## Overview
CubeX is a lightweight, real-time interferometric FITS cube visualization and kinematic exploration tool designed for mm/sub-mm/radio astronomers. Built on a modern Python scientific stack with `PyQt5` and `PyQtGraph`, it offers a fast, interactive interface for navigating complex spectral data cubes without the heavy computational overhead of traditional software packages. CubeX streamlines advanced interferometric data visualisation and analysis by providing dynamic, on-the-fly moment map and position-velocity (PV) diagram generation, along with standard spectral extraction, spatial analysis, and statistics tools, all alongside a fast image cube renderer. Users can seamlessly interact with and process large datasets, extract local spectra from custom spatial regions, and query molecular transitions from the CDMS and JPL databases directly within the CubeX UI.

## Installation

Since CubeX is built completely in python, it can be easily installed with `pip` or `pipx` on any computer, independent of the operating system. CubeX is also available as a standalone packaged application for Linux and macOS. It can also be installed from source by cloning this github repository.
> [!NOTE]
> **The standalone macOS application will be provided soon. Currently only a Linux executable is available**.

### Option 1: Using `pip` or `pipx` (Recommended)
CubeX is fully packaged according to modern `pyproject.toml` standards. It is highly recommended to install CubeX inside an isolated virtual environment (`venv` or `conda`).

You can install the latest release directly from GitHub:

```bash
# Create and activate a virtual environment
python3 -m venv [your-venv-name]
source [your-venv-name]/bin/activate

# Install CubeX
pip install git+https://github.com/ramlalunni/CubeX.git
```

**Alternative, easier option:** If you want to install CubeX via Python, but without manually managing virtual environments, we highly recommend using [`pipx`](https://pipx.pypa.io/stable/). `pipx` automatically creates an isolated environment for CubeX and exposes the `cubex` command globally.

```bash
# Install CubeX with just this one command - the venv will be created and managed automatically
pipx install git+https://github.com/ramlalunni/CubeX.git
```

Both these commands will automatically install all dependencies CubeX needs. After installation, you can launch the CubeX GUI directly from your terminal, by typing `cubex`. This will work from any directory in your system (If you installed with `pip`, you have to manually ensure that the relevant `venv` is active).



### Option 2: Standalone Application
For users who prefer not to manage Python environments or dependencies, we also provide pre-compiled, double-click executables for Linux and macOS. These bundle the entire Python ecosystem into a single app.

1. Navigate to the [Releases](https://github.com/ramlalunni/CubeX/releases) page.
2. Download the latest version for your operating system (e.g., `.dmg` or `.app.zip` for macOS, or the binary for Linux).
3. Extract and double-click to launch the CubeX GUI.

> **OS Notes:**
> - **macOS:** On your first launch, Apple's Gatekeeper may block the app. Simply right-click the app icon and select **"Open"** to bypass the warning.  
> - **Linux:** Depending on your distribution, you may need to make the file executable before double-clicking. You can do this by right-clicking the file -> Properties -> Permissions -> "Allow executing file as program", or via terminal: `chmod +x CubeX-Linux`.

### Option 3: Build from source
CubeX is designed for modern Python environments. It is highly recommended to use a virtual environment (`venv` or `conda`) for installation. If you prefer to run CubeX within your own `conda` or another `venv` environment, on Linux/MacOS systems:

```bash
# Clone the github repo
git clone https://github.com/ramlalunni/CubeX.git
cd CubeX

# Install all dependencies and create simple CubeX executable
./install.sh # this will prompt for creating a venv
```

This will create a `cubex.sh` file, which you need to make executable:

```bash
chmod +x cubex.sh
```

Now, running `./cubex.sh` will launch the CubeX GUI.

Alternatively, you can also run the following commands yourself instead of using `./install.sh`:

```bash
pip install -r requirements.txt # installs dependencies into your active venv
python legacy_main.py # starts the CubeX GUI
```

## Documentation
The comprehensive CubeX User Manual, including UI tutorials and details of the background methods CubeX uses, is hosted on ReadTheDocs, at
📖 **[CubeX Documentation](https://cubex.readthedocs.io)**.

> [!CAUTION]
> The current documentation is an experimental draft, and is under construction. It is incomplete, and contains known errors. **DO NOT USE** this documentation as an official reference until this warning is removed in a future update.


## Contributing & Support
We welcome community contributions. If you encounter a bug or have a feature request, please [open an issue](https://github.com/ramlalunni/CubeX/issues) on GitHub. Pull requests for new analysis tools or optimizations are also encouraged.

## License
CubeX is free and open-source software distributed under the [GNU GPL 3.0 License](LICENSE).

**Disclaimer:** CubeX has been developed and tested for radio interferometric data analysis and visualisation, but is provided "as is" and without warranty of any kind, compliant with GPLv3.0.

## Acknowledgements
CubeX is built on the open-source scientific Python ecosystem, particularly [`astropy`](https://www.astropy.org/), [`pyqtgraph`](https://www.pyqtgraph.org/), and [`numpy`](https://numpy.org/). Manually supervised agentic AI assistants ([`Google Gemini 3`](https://gemini.google.com/) and [`DeepSeek v4`](https://www.deepseek.com/)) were utilized during development for code refactoring, boilerplate UI generation, and documentation drafting.

For any queries, please contact the developer [here](mailto:[ramlalunni@gmail.com]).