"""
Module defining the main application window and high-level dialogs for CubeX.
"""
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import pyqtgraph as pg

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (QApplication, QMainWindow, QAction, QTabWidget, 
                             QFileDialog, QMessageBox, QMenu, QDialog, QDesktopWidget,
                             QVBoxLayout, QHBoxLayout, QTextEdit, QCheckBox, QPushButton, QLabel)

class ExportRegionsDialog(QDialog):
    """
    Dialog window for selecting which regions to export.

    Attributes
    ----------
    checkboxes : dict
        Mapping of region names to their corresponding QCheckBox widgets.
    single_file_cb : PyQt5.QtWidgets.QCheckBox or None
        Checkbox for selecting single-file overlay export (PDF only).
    include_title_cb : PyQt5.QtWidgets.QCheckBox or None
        Checkbox for including plot titles in the export (PDF only).
    """
    def __init__(self, parent, regions_dict, title, is_pdf=False):
        """
        Initialize the ExportRegionsDialog.

        Parameters
        ----------
        parent : PyQt5.QtWidgets.QWidget
            The parent widget.
        regions_dict : dict
            Dictionary mapping region names to their PlotDataItem curves.
        title : str
            The window title for the dialog.
        is_pdf : bool, optional
            Whether the export format is PDF, by default False.
        """
        super().__init__(parent)
        self.setWindowTitle(title)
        self.layout = QVBoxLayout(self)
        
        self.layout.addWidget(QLabel("Select regions to export:"))
        
        self.checkboxes = {}
        for name, curve in regions_dict.items():
            cb = QCheckBox(name)
            cb.setChecked(True)
            self.checkboxes[name] = cb
            self.layout.addWidget(cb)
            
        self.single_file_cb = None
        self.include_title_cb = None
        if is_pdf:
            self.single_file_cb = QCheckBox("Export to single file (overlaid)")
            self.single_file_cb.setChecked(False)
            self.layout.addWidget(self.single_file_cb)
            
            self.include_title_cb = QCheckBox("Include Plot Title in Export")
            self.include_title_cb.setChecked(False)
            self.layout.addWidget(self.include_title_cb)
            
        btn_layout = QHBoxLayout()
        btn_export = QPushButton("Export")
        btn_export.clicked.connect(self.accept)
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        
        btn_layout.addWidget(btn_export)
        btn_layout.addWidget(btn_cancel)
        self.layout.addLayout(btn_layout)
        
    def get_selected_regions(self):
        """
        Retrieve the names of all regions that the user has selected.

        Returns
        -------
        list of str
            List of selected region names.
        """
        return [name for name, cb in self.checkboxes.items() if cb.isChecked()]
        
    def is_single_file(self):
        """
        Check if the user requested a single overlaid file.

        Returns
        -------
        bool
            True if single file export is checked, False otherwise.
        """
        return self.single_file_cb.isChecked() if self.single_file_cb else False
        
    def is_include_title(self):
        """
        Check if the user requested to include plot titles in the export.

        Returns
        -------
        bool
            True if titles should be included, False otherwise.
        """
        return self.include_title_cb.isChecked() if self.include_title_cb else True

# Import the tab environment we built
from cubex.gui.components.explorer_view import ExplorerView
from cubex.core.math_kernels import _NUMBA_AVAILABLE
from cubex.gui.dialogs import ContourDialog
from cubex.gui.controllers.main_controller import MainController

# ==============================================================================
# MAIN WINDOW APP
# ==============================================================================
class KinematicExplorerApp(QMainWindow):
    """
    The main application window for CubeX.

    This class sets up the main Qt window, the tabbed interface for multiple
    open files, the global menu bar, and orchestrates the creation of new 
    `ExplorerView` tabs.

    Attributes
    ----------
    startup_width : int
        The initial width of the main window in pixels.
    startup_height : int
        The initial height of the main window in pixels.
    current_cmap : str
        The global default colormap string.
    is_absolute_wcs : bool
        Global flag indicating if coordinates should display as absolute WCS.
    controller : MainController
        The high-level controller handling global actions.
    tabs : PyQt5.QtWidgets.QTabWidget
        The tab widget holding individual `ExplorerView` instances.
    """
    def __init__(self):
        """
        Initialize the main application window and menu bar.
        """
        super().__init__()
        self.setWindowTitle("CubeX")
        screen = QDesktopWidget().availableGeometry()
        w = min(screen.width(), 1500)
        h = min(screen.height(), 600)
        self.startup_width = w
        self.startup_height = h
        x = max(0, (screen.width() - w) // 2)
        y = max(0, (screen.height() - h) // 2)
        self.setGeometry(x, y, w, h)
        self.current_cmap = 'turbo' 
        self.is_absolute_wcs = False 
        
        self.controller = MainController(self) 

        self.tabs = QTabWidget()
        self.tabs.setElideMode(Qt.ElideRight)
        self.tabs.setTabsClosable(True)
        self.tabs.tabCloseRequested.connect(self.close_tab)
        self.tabs.currentChanged.connect(self.update_menu_states)
        self.setCentralWidget(self.tabs)

        btn_new_tab = QPushButton("+")
        #btn_new_tab.setFixedSize(28, 24)
        btn_new_tab.setFixedWidth(40)
        btn_new_tab.setToolTip("New Tab")
        font = QFont("Segoe UI Symbol")  # Good option on Windows
        btn_new_tab.setFont(font)
        btn_new_tab.setStyleSheet("QPushButton { font-size: 16px; font-weight: bold; "
                                  "border: 1px solid #444; border-radius: 4px; "
                                  "background-color: #2a2a2a; color: #aaa; } "
                                  "QPushButton:hover { background-color: #3a3a3a; color: #fff; "
                                  "border-color: #666; }")
        btn_new_tab.clicked.connect(self.add_new_tab)
        self.tabs.setCornerWidget(btn_new_tab, Qt.TopRightCorner)
        
        self.statusBar().showMessage("Ready.")
        self.init_menu()
        self.add_new_tab() 

    def init_menu(self):
        """Initialize the main application menu bar and connect actions to the controller."""
        menubar = self.menuBar()
        
        # --- File Menu ---
        file_menu = menubar.addMenu('File')
        
        action_load = QAction('Open FITS File', self)
        action_load.triggered.connect(self.controller.load_file)
        file_menu.addAction(action_load)

        action_close = QAction('Close FITS File', self)
        action_close.triggered.connect(self.controller.close_cube)
        file_menu.addAction(action_close)

        file_menu.addSeparator()

        action_overlay = QAction('Overlay Image (Contours)...', self)
        action_overlay.triggered.connect(self.controller.load_overlay_file)
        file_menu.addAction(action_overlay)

        file_menu.addSeparator()

        action_new_tab = QAction('New Tab', self)
        action_new_tab.triggered.connect(self.add_new_tab)
        file_menu.addAction(action_new_tab)

        action_new_win = QAction('New Window', self)
        action_new_win.triggered.connect(self.spawn_new_window)
        file_menu.addAction(action_new_win)

        file_menu.addSeparator()

        action_exit = QAction('Exit', self)
        action_exit.triggered.connect(self.close)
        file_menu.addAction(action_exit)

        # --- View Menu ---
        view_menu = menubar.addMenu('View')
        
        action_header = QAction('File Header', self)
        action_header.triggered.connect(self.controller.show_header)
        view_menu.addAction(action_header)

        action_reset = QAction('Reset Zoom/Views', self)
        action_reset.triggered.connect(self.controller.reset_views)
        view_menu.addAction(action_reset)
        
        view_menu.addSeparator()
        
        self.action_wcs = QAction('Use Absolute WCS Coordinates', self, checkable=True)
        self.action_wcs.setChecked(False)
        self.action_wcs.triggered.connect(self.controller.toggle_wcs)
        view_menu.addAction(self.action_wcs)
        


        # --- Tools Menu ---
        tools_menu = menubar.addMenu('Tools')
        
        action_lines = QAction('Query Molecular Line Database...', self)
        action_lines.triggered.connect(self.controller.open_line_catalog)
        tools_menu.addAction(action_lines)

        action_contour = QAction('Draw Contours on Active Panel...', self)
        action_contour.triggered.connect(self.controller.show_contour_dialog)
        tools_menu.addAction(action_contour)
        tools_menu.addSeparator()

        action_clear_roi = QAction('Clear Spatial ROIs', self)
        action_clear_roi.triggered.connect(self.clear_roi)
        tools_menu.addAction(action_clear_roi)

        action_clear_pv = QAction('Clear PV Cuts', self)
        action_clear_pv.triggered.connect(self.clear_pv_cuts)
        tools_menu.addAction(action_clear_pv)

        action_clear_spec_regions = QAction('Clear Spectrum Regions', self)
        action_clear_spec_regions.triggered.connect(self.clear_spectrum_regions)
        tools_menu.addAction(action_clear_spec_regions)

        cmap_menu = QMenu('Colourmap', self)
        for c in ['turbo', 'inferno', 'viridis', 'plasma', 'magma', 'grey']:
            act = QAction(c.capitalize(), self)
            act.triggered.connect(lambda checked, cm=c: self.controller.set_colormap(cm))
            cmap_menu.addAction(act)
        tools_menu.addMenu(cmap_menu)

        # --- Export Menu ---
        export_menu = menubar.addMenu('Export')
        
        self.action_save_fits = QAction('Export active panel to FITS...', self)
        self.action_save_fits.triggered.connect(self.controller.export_fits_active)
        export_menu.addAction(self.action_save_fits)

        self.action_save_pdf = QAction('Save active panel as PDF...', self)
        self.action_save_pdf.triggered.connect(self.controller.export_pdf_active)
        export_menu.addAction(self.action_save_pdf)
        
        export_menu.addSeparator()

        self.action_export_spec_fits = QAction('Export spectrum as FITS...', self)
        self.action_export_spec_fits.triggered.connect(self.controller.export_spectrum_fits)
        export_menu.addAction(self.action_export_spec_fits)

        action_export_csv = QAction('Export spectrum as CSV...', self)
        action_export_csv.triggered.connect(self.controller.export_spectrum)
        export_menu.addAction(action_export_csv)
        
        self.action_export_spec_pdf = QAction('Save spectrum as PDF...', self)
        self.action_export_spec_pdf.triggered.connect(self.controller.export_spectrum_pdf)
        export_menu.addAction(self.action_export_spec_pdf)

        # --- Help Menu ---
        help_menu = menubar.addMenu('Help')
        
        self.action_docs = QAction('View Documentation', self)
        self.action_docs.triggered.connect(self.controller.open_documentation)
        help_menu.addAction(self.action_docs)

        self.action_source = QAction('View source code', self)
        self.action_source.triggered.connect(self.controller.open_source)
        help_menu.addAction(self.action_source)
        
        self.action_bug = QAction('Submit bug report / feature request', self)
        self.action_bug.triggered.connect(self.controller.open_bug_report)
        help_menu.addAction(self.action_bug)
        
        self.action_license = QAction('View License', self)
        self.action_license.triggered.connect(self.controller.open_license)
        help_menu.addAction(self.action_license)

        help_menu.addSeparator()
        self.action_about = QAction('About', self)
        self.action_about.triggered.connect(self.controller.show_about)
        help_menu.addAction(self.action_about)

    def keyPressEvent(self, event):
        """
        Handle global key presses for the main window (e.g., ESC to clear active regions).

        Parameters
        ----------
        event : QKeyEvent
            The key press event triggered by the user.
        """
        if event.key() == Qt.Key_Escape:
            tab = self.get_active_tab()
            if tab and getattr(tab, 'roi_selected', False):
                if getattr(tab, 'active_spatial_spectrum_roi', None) is not None:
                    tab.remove_spatial_spectrum_roi(tab.active_spatial_spectrum_roi)
                    tab.active_spatial_spectrum_roi = None
                elif getattr(tab, 'current_roi', None) is not None:
                    tab.clear_roi()
        super().keyPressEvent(event)

    def update_menu_states(self):
        """
        Enable or disable top-level menu items dynamically based on the current active tab's state.
        """
        tab = self.get_active_tab()
        if tab:
            is_image = tab.last_clicked_panel_id != 'spectrum'
            self.action_save_fits.setEnabled(is_image)
            self.action_save_pdf.setEnabled(is_image)
            
            is_3d = not getattr(tab, 'is_2d_image', False)
            for action in self.menuBar().findChildren(QAction):
                text = action.text()
                if text in ["Clear PV Cuts", "Clear Spectrum Regions"]:
                    action.setEnabled(is_3d)
                elif "spectrum" in text.lower() or "line database" in text.lower():
                    action.setEnabled(is_3d)



    def get_active_tab(self): 
        """
        Retrieve the currently active workspace tab.

        Returns
        -------
        ExplorerView or None
            The currently visible tab widget.
        """
        return self.tabs.currentWidget()

    def add_new_tab(self):
        """Instantiate and append a new, empty workspace tab to the tab bar."""
        from cubex.gui.components.explorer_view import ExplorerView
        tab = ExplorerView(self)
        idx = self.tabs.addTab(tab, "Untitled")
        self.tabs.setCurrentIndex(idx)
        self.update_menu_states()

    def close_tab(self, index):
        """
        Close a workspace tab, or reset it if it is the only remaining tab.

        Parameters
        ----------
        index : int
            The index of the tab to close.
        """
        if self.tabs.count() > 1:
            self.tabs.widget(index).deleteLater()
            self.tabs.removeTab(index)
        else:
            self.get_active_tab().close_file()
            self.tabs.setTabText(0, "Untitled")
            self.setWindowTitle("CubeX")
        self.update_menu_states()

    def spawn_new_window(self):
        """Open an entirely new, independent instance of the application window."""
        self.new_win = KinematicExplorerApp()
        self.new_win.show()

    def clear_roi(self):
        """Delegate clearing the primary spatial ROI to the active tab."""
        tab = self.get_active_tab()
        if tab: tab.clear_roi()

    def clear_spectrum_regions(self):
        """Delegate clearing all spectrum extraction boxes to the active tab."""
        tab = self.get_active_tab()
        if tab: tab.clear_spectrum_regions()

    def clear_pv_cuts(self):
        """Delegate clearing all drawn PV cuts to the active tab."""
        tab = self.get_active_tab()
        if tab: tab.clear_pv_cuts()


