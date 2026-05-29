"""
Main entry point for the CubeX application.

Initializes the PyQt5 application, loads the global stylesheet, 
sets up pyqtgraph configurations, and launches the main KinematicExplorerApp window.
"""
import sys
import os
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
import pyqtgraph as pg

from src.gui.components.main_window_view import KinematicExplorerApp

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

def main():
    """
    Initialize and launch the main CubeX PyQt5 application.

    Sets up global Qt attributes, configures the pyqtgraph color scheme,
    loads the application stylesheet, and enters the main event loop.
    """
    app = QApplication(sys.argv)
    app.setOverrideCursor(Qt.ArrowCursor)
    app.setStyle("Fusion")
    
    pg.setConfigOption('background', '#1a1a1a')
    pg.setConfigOption('foreground', '#e0e0e0')
    
    # --- UPDATED PATH RESOLUTION ---
    # We use our new helper function to safely find the assets folder
    style_path = get_resource_path(os.path.join("assets", "style.qss"))
    
    dark_stylesheet = load_stylesheet(style_path)
    if dark_stylesheet:
        app.setStyleSheet(dark_stylesheet)
    
    ex = KinematicExplorerApp()
    ex.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()