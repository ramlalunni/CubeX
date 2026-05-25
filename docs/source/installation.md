# Installation

CubeX is designed to be as accessible as possible. Depending on your needs, you can either download the pre-compiled, standalone application (recommended for most astronomers) or install it from source if you wish to modify the code.

## Option 1: Standalone Application (Recommended)
For users who want to avoid managing Python environments or dependencies, we provide pre-compiled executables that bundle the entire Python ecosystem into a single app. 

1. Navigate to the **Releases** page on the CubeX GitHub repository.
2. Download the appropriate executable for your operating system:
   * **Linux:** Download `CubeX-Linux`. Make the file executable (`chmod +x CubeX-Linux`) and run it.
   * **macOS:** *Note: The standalone macOS `.dmg` application will be provided soon. Currently, only a Linux executable is actively distributed.*
3. No further setup is required!

## Option 2: Developer / Source Code
If you want to contribute to the project, modify the codebase, or run CubeX on an unsupported OS architecture, you can run the application directly from the Python source.

**Prerequisites:** Python 3.9 or higher.

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/CubeX.git
   cd CubeX
   ```
2. **Create a virtual environment (Recommended):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. **Install the dependencies:**
   CubeX relies heavily on `PyQt5`, `pyqtgraph`, `astropy`, `spectral-cube`, and optionally `numba` for accelerated moment math.
   ```bash
   pip install -r requirements.txt
   ```
4. **Run the Application:**
   ```bash
   python main.py
   # Or use the provided bash wrapper:
   ./CubeX.sh
   ```
