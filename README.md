# CubeX

[![Documentation Status](https://readthedocs.org/projects/cubex/badge/?version=latest)](https://cubex.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

## Overview
CubeX is a lightweight, real-time ALMA and VLA FITS cube visualization and kinematic exploration tool designed for radio astronomers. Built on a modern Python scientific stack with PyQt5 and PyQtGraph, it offers a lightning-fast, interactive interface for navigating complex spectral data cubes without the heavy computational overhead of traditional legacy software packages.

At its core, CubeX streamlines advanced interferometric data analysis by providing dynamic moment map generation (M0, M1, M2, M8, M9), integrated 2D Gaussian spatial fitting, and position-velocity (PV) diagram extraction. Researchers can seamlessly interact with large datasets, extract local spectra from custom spatial regions on-the-fly, and instantly query molecular transitions from the CDMS and JPL databases directly within the UI.

## Installation
CubeX is designed for modern Python environments. It is highly recommended to use a virtual environment (`venv` or `conda`) for installation.

```bash
git clone https://github.com/YOUR_USERNAME/CubeX.git
cd CubeX
pip install -r requirements.txt
```
*(Note: You can also use the included `./install.sh` script to automatically configure a standalone environment on Linux).*

## Quickstart
After installation, you can launch the CubeX GUI directly from your terminal:

```bash
./CubeX.sh
```
*(Or manually run `python3 main.py` in your activated Python environment).*

## Documentation
The comprehensive CubeX User Manual, including full theoretical breakdowns of the implemented mathematics, user interface tutorials, and developer API references, is hosted on ReadTheDocs.

📖 **[Read the Full Documentation Here](https://cubex.readthedocs.io)**

## Citation & License
CubeX is released under the [MIT License](LICENSE). 

If you utilize CubeX in your research, please cite the software in your publications:
```bibtex
@software{CubeX2026,
  author       = {Ramlal Unnikrishnan},
  title        = {CubeX: Real-Time Interferometric Kinematic Explorer},
  year         = {2026},
  publisher    = {GitHub},
  url          = {https://github.com/YOUR_USERNAME/CubeX}
}
```