# Installation

CubeX provides flexible deployment options. Depending on your needs, you can either download a pre-compiled standalone executable or install it from source.

## Option 1: Standalone Application (Recommended)
For most users, the standalone application is the simplest way to get started. It bundles Python and all required dependencies into a single executable file.

1. Navigate to the **Releases** page on the CubeX GitHub repository.
2. Download the appropriate executable for your operating system (e.g., `CubeX-Linux`).
3. Make the file executable and run it:
   ```bash
   chmod +x CubeX-Linux
   ./CubeX-Linux
   ```

## Option 2: Developer / Source Code
If you wish to modify the code, run on an unsupported OS architecture, or contribute to the project, you can install CubeX from source.

**Prerequisites:** Python 3.9+

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/CubeX.git
   cd CubeX
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install the dependencies:**
   CubeX requires `PyQt5`, `pyqtgraph`, `astropy`, `spectral-cube`, `astroquery`, `pandas`, and optionally `numba` for accelerated spatial interpolation.
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Application:**
   ```bash
   python main.py
   # Or use the provided wrapper script:
   ./CubeX.sh
   ```
