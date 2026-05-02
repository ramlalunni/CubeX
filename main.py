import sys
import os
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
import pyqtgraph as pg

from src.gui.main_window import KinematicExplorerApp

def get_resource_path(relative_path):
    """
    Get absolute path to resource, works for dev and for PyInstaller.
    PyInstaller creates a temp folder and stores path in sys._MEIPASS.
    """
    try:
        base_path = sys._MEIPASS
    except AttributeError:
        base_path = os.path.abspath(os.path.dirname(__file__))
    
    return os.path.join(base_path, relative_path)

def load_stylesheet(filepath):
    if os.path.exists(filepath):
        with open(filepath, "r") as file:
            return file.read()
    else:
        print(f"Warning: Stylesheet '{filepath}' not found. Loading without custom styles.")
        return ""

def main():
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