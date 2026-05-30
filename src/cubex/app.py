"""
Main entry point for the CubeX application.

Initializes the PyQt5 application, loads the global stylesheet, 
sets up pyqtgraph configurations, and launches the main KinematicExplorerApp window.
"""
import sys
import os
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt, QTimer
import pyqtgraph as pg 

# Updated to use the new package namespace
from cubex.gui.components.main_window_view import KinematicExplorerApp

def get_resource_path(relative_path):
    """
    Get absolute path to resource, works for dev and for PyInstaller.

    PyInstaller creates a temp folder and stores the path in `sys._MEIPASS`. 
    This function ensures assets like stylesheets are found whether running 
    from source or as a compiled executable.

    Parameters
    ----------
    relative_path : str
        The relative path to the resource file.

    Returns
    -------
    str
        The absolute path to the resource file.
    """
    try:
        base_path = sys._MEIPASS
    except AttributeError:
        # __file__ now dynamically points to the inner `src/cubex/` directory
        base_path = os.path.abspath(os.path.dirname(__file__))
    
    return os.path.join(base_path, relative_path)

def load_stylesheet(filepath):
    """
    Load the contents of a QSS stylesheet file.

    Parameters
    ----------
    filepath : str
        The absolute path to the stylesheet file.

    Returns
    -------
    str
        The CSS/QSS string content of the file, or an empty string if not found.
    """
    if os.path.exists(filepath):
        with open(filepath, "r") as file:
            return file.read()
    else:
        print(f"Warning: Stylesheet '{filepath}' not found. Loading without custom styles.")
        return ""

def print_startup_banner():
    """Prints a styled ASCII banner to the terminal, leaving the cursor waiting."""
    
    BOLD = '\033[1m'
    DIM = '\033[2m'
    RESET = '\033[0m'
    
    banner = fr"""{BOLD}
  ____      _          __  __ 
 / ___|   _| |__   ___ \ \/ / 
| |  | | | | '_ \ / _ \ \  /  
| |__| |_| | |_) |  __/ /  \  
 \____\__,_|_.__/ \___|/_/\_\ 
{RESET}
  CubeX: Interferometric Image Cube Explorer
  Version 0.4-preview
  
  {DIM}Initializing GUI and mathematical kernels... {RESET}"""
    
    # Print the banner, do NOT print a newline, and force it to the screen
    print(banner, flush=True)


def on_gui_ready():
    """Callback triggered exactly when the Qt window finishes painting on screen."""
    DIM = '\033[2m'
    RESET = '\033[0m'
    # Complete the sentence that the banner left hanging
    print(f"  {DIM}Done.{RESET}\n")

def main():
    """
    Initialize and launch the main CubeX PyQt5 application.

    Sets up global Qt attributes, configures the pyqtgraph color scheme,
    loads the application stylesheet, and enters the main event loop.
    """

    # Print the terminal banner immediately so the user knows it's loading
    print_startup_banner()

    app = QApplication(sys.argv)
    app.setOverrideCursor(Qt.ArrowCursor)
    app.setStyle("Fusion")
    
    pg.setConfigOption('background', '#1a1a1a')
    pg.setConfigOption('foreground', '#e0e0e0')
    
    # Resolves to src/cubex/assets/style.qss during development
    style_path = get_resource_path(os.path.join("assets", "style.qss"))
    
    dark_stylesheet = load_stylesheet(style_path)
    if dark_stylesheet:
        app.setStyleSheet(dark_stylesheet)
    
    ex = KinematicExplorerApp()
    ex.show()

    # Tell PyQt5: "Once the event loop starts, wait 100 milliseconds 
    # to let the OS draw the window, then run on_gui_ready"
    QTimer.singleShot(100, on_gui_ready)

    sys.exit(app.exec_())

if __name__ == '__main__':
    main()