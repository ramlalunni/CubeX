"""
Module defining the main ExplorerView UI component for CubeX.

This file contains the complex layout definitions for the channel maps,
spectral plotting areas, and moment/PV diagram panels.
"""
import csv
import warnings
import numpy as np
import pyqtgraph as pg
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
import astropy.constants as const
import astropy.units as u
from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPainterPath
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QMessageBox, QLineEdit, 
                             QComboBox, QFrame, QStackedWidget, QSizePolicy, QTabWidget,
                             QGroupBox, QCheckBox, QDialog, QScrollArea, QGridLayout)

try:
    from PyQt5.QtWidgets import FlowLayout
except ImportError:
    pass # we will use QHBoxLayout if FlowLayout is not easily available
from spectral_cube import SpectralCube

import qtawesome as qta

# Import our modularized components
from src.core.splatalogue import SplatalogueWorker
from src.gui.components.custom_widgets import JumpSlider, fix_axis_scaling, WCSAxisItem
from src.gui.dialogs import LineCatalogDialog, LineSelectionDialog, ContourDialog, ContourOptionsDialog

from src.core.math_kernels import _NUMBA_AVAILABLE, _bilinear_interp, _compute_moments_12


# ==============================================================================
# BACKGROUND WORKER — moment maps & PV diagrams
# ==============================================================================

from src.gui.controllers.workers import MomentWorker


# ==============================================================================
# INDIVIDUAL EXPLORER TAB
# ==============================================================================

from src.gui.components.graph_panels import make_roi_rotatable_with_ctrl, ChannelMapViewBox, SpectrumViewBox
from src.gui.controllers.explorer_controller import ExplorerController

class ExplorerView(QWidget):
    """
    Main user interface view for the CubeX data explorer.

    This class constructs the complex multi-panel layout containing the channel map,
    the spectrum extraction tool, and the moment/PV diagram panels. It delegates
    most logic and state management to the `ExplorerController`.

    Attributes
    ----------
    controller : ExplorerController
        The controller managing logic for this view.
    parent_window : PyQt5.QtWidgets.QMainWindow
        The main application window hosting this view.
    is_2d_image : bool
        Flag indicating whether the loaded data is a 2D image rather than a 3D cube.
    cube_clean : spectral_cube.SpectralCube or None
        The underlying spectral cube data.
    v_axis : numpy.ndarray or None
        The spectral axis values.
    display_unit : str
        The physical unit used for display.
    spec_unit : str
        The spectral unit.
    pix_scale_arcsec : float
        The pixel scale in arcseconds.
    pixels_per_beam : float
        Number of pixels per synthesized beam.
    """
    def __init__(self, parent_window):
        """
        Initialize the ExplorerView and build its layout hierarchy.

        Parameters
        ----------
        parent_window : PyQt5.QtWidgets.QMainWindow
            The parent application window.
        """
        super().__init__()
        self.controller = ExplorerController(self)
        self.parent_window = parent_window 
        self.is_2d_image = False
        self.cube_clean = None 
        self.v_axis = None
        self.display_unit = "Unknown"
        self.spec_unit = "Unknown"
        self.pix_scale_arcsec = 1.0
        self.pixels_per_beam = 1.0
        self.nx = 1
        self.ny = 1
        self.raw_header = None
        self.fits_header_text = ""
        self.ch_levels = (0, 1) 
        
        self.wcs_2d = None
        self.is_absolute_wcs = False
        
        self.rest_freq_hz = None

        self.is_drawing_polygon = False
        self.polygon_points = []
        self.roi_selected = False
        self.catalog_overlay_items = []
        self.spectrum_spatial_rois = [] # List of {"name": str, "roi": ROI, "checkbox": QCheckBox, "color": str}
        self.active_spatial_spectrum_roi = None
        self.roi_selected = False
        
        from PyQt5.QtWidgets import QShortcut
        from PyQt5.QtGui import QKeySequence
        self.esc_shortcut = QShortcut(QKeySequence(Qt.Key_Escape), self)
        self.esc_shortcut.setContext(Qt.WidgetWithChildrenShortcut)
        self.esc_shortcut.activated.connect(self.handle_escape)

        # Polygon drawing state
        self.is_drawing_polygon = False
        self.polygon_points = []
        self.polygon_preview_line = None
        
        self.region_colors = ['#8A2BE2', '#FF0000', '#FFA500', '#228B22', '#FF00FF', '#FFFF00', '#DDA0DD', '#E9967A', '#0000CD', '#D2691E']
        self.roi_selected = False
        self.current_m0_raw = None
        self.active_picker_panel = None
        self.pv_data = None
        self.pv_offset_axis = None
        self.pv_velocity_axis = None
        
        self.last_clicked_panel_id = 'channel' 
        self.contour_params = {'channel': None, 0: None, 1: None, 2: None}
        self.active_contours = {'channel': [], 0: [], 1: [], 2: []}

        self.contour_overlays = []
        self.overlay_color_palette = ['cyan', 'magenta', 'yellow', 'lime', 'orange', 'pink', 'aquamarine', 'gold', 'salmon', 'skyblue']

        self.contour_overlay_file = None
        self.contour_overlay_cube = None
        self.contour_overlay_wcs = None
        self.contour_overlay_v_axis = None
        self.contour_overlay_is_static = False
        self.contour_overlay_2d = None
        self.contour_overlay_nx = 0
        self.contour_overlay_ny = 0
        self.contour_overlay_pix_scale = 1.0
        self.contour_overlay_iso_items = []
        self._overlay_reproject_cache = None
        self.contour_options = {
            'mode': 'rms',
            'rms': 0.001,
            'multipliers_str': '3, 5, 10, 20, 40',
            'lin_min': 0.0,
            'lin_max': 10.0,
            'n_levels': 5,
            'log_min': 0.001,
            'log_max': 10.0,
            'log_base': 10.0,
            'percentages_str': '10, 30, 50, 70, 90',
            'color': 'white',
            'line_width': 1.5,
            'line_style': 'solid',
            'smooth': False,
            'smooth_kernel': 3,
        }

        self.overlay_spectrum_curves = {}
        self.overlay_spectrum_curves_smooth = {}
        
        self.playback_timer = QTimer()
        self.playback_timer.timeout.connect(self.step_channel)
        self.play_direction = 1

        # Background worker for moment / PV computation
        self._moment_worker = None
        self._moment_generation = 0
        self._pending_workers = []

        self.initUI()

    def initUI(self):
        """
        Initialize the layout, graphical components, and signals for the main explorer window.
        """
        main_layout = QVBoxLayout(self)
        self.frames = {} 

        # ==================== TOP HALF ====================
        top_half = QHBoxLayout()

        # --- Channel Map ---
        self.frame_channel = QFrame()
        self.frame_channel.setObjectName("PanelFrame")
        channel_layout = QVBoxLayout(self.frame_channel)
        
        lbl_ch_title = QLabel("Channel Map")
        lbl_ch_title.setStyleSheet("font-weight: bold; color: #3498db; font-size: 13px;")
        channel_layout.addWidget(lbl_ch_title)

        ch_bottom = WCSAxisItem(orientation='bottom')
        ch_left = WCSAxisItem(orientation='left')
        self.channel_viewbox = ChannelMapViewBox()
        self.channel_viewbox.parent_tab = self
        self.plot_channel = pg.PlotItem(viewBox=self.channel_viewbox, axisItems={'bottom': ch_bottom, 'left': ch_left})
        
        self.plot_channel.invertX(True)
        self.plot_channel.setLabel('bottom', 'RA offset (arcsec)')
        self.plot_channel.setLabel('left', 'Dec offset (arcsec)')
        
        self.view_channel = pg.ImageView(view=self.plot_channel)
        self.plot_channel.invertY(False) 
        
        self.view_channel.ui.roiBtn.hide()
        self.view_channel.ui.menuBtn.hide()
        self.view_channel.ui.histogram.gradient.loadPreset('turbo')
        self.view_channel.ui.histogram.setFixedWidth(160) 
        fix_axis_scaling(self.view_channel.ui.histogram.axis) 
        self.plot_channel.vb.sigRangeChanged.connect(self.update_beam_positions)
        channel_layout.addWidget(self.view_channel, stretch=1)

        self.lbl_hover_ch = QLabel("")
        self.lbl_hover_ch.setStyleSheet("color: #aaa; font-size: 9.5px;")
        channel_layout.addWidget(self.lbl_hover_ch)

        ctrl_layout = QHBoxLayout()
        ctrl_layout.setContentsMargins(0, 0, 0, 0)
        self.btn_prev = QPushButton("<")
        self.btn_play_rev = QPushButton("<<")
        self.btn_stop = QPushButton("■")
        self.btn_play_fwd = QPushButton(">>")
        self.btn_next = QPushButton(">")
        
        for btn in [self.btn_prev, self.btn_play_rev, self.btn_stop, self.btn_play_fwd, self.btn_next]:
            btn.setFixedWidth(40)
            btn.setFixedHeight(22)
            ctrl_layout.addWidget(btn)
        
        self.btn_prev.clicked.connect(lambda _=False: self.step_channel(-1))
        self.btn_next.clicked.connect(lambda _=False: self.step_channel(1))
        self.btn_play_rev.clicked.connect(lambda _=False: self.start_playback(-1))
        self.btn_play_fwd.clicked.connect(lambda _=False: self.start_playback(1))
        self.btn_stop.clicked.connect(self.stop_playback)
        
        self.slider_channel = JumpSlider(Qt.Horizontal)
        self.slider_channel.valueChanged.connect(self.update_channel_map)
        ctrl_layout.addWidget(self.slider_channel, stretch=1)
        
        ctrl_layout.addWidget(QLabel("Vel:"))
        self.input_channel_vel = QLineEdit("")
        self.input_channel_vel.setFixedWidth(55)
        self.input_channel_vel.setFixedHeight(22)
        self.input_channel_vel.editingFinished.connect(self.set_channel_from_text)
        ctrl_layout.addWidget(self.input_channel_vel)
        ctrl_layout.addWidget(QLabel("km/s"))
        
        channel_layout.addLayout(ctrl_layout)

        roi_layout = QHBoxLayout()
        roi_layout.setContentsMargins(0, 0, 0, 0)
        self.lbl_combo_roi = QLabel("Spectrum Region:")
        roi_layout.addWidget(self.lbl_combo_roi)
        self.combo_roi = QComboBox()
        self.combo_roi.setFixedHeight(22)
        self.combo_roi.addItems(["Whole Map", "Point (Beam)", "Ellipse", "Rectangle", "Custom Polygon"])
        self.combo_roi.activated[str].connect(self.change_roi)
        roi_layout.addWidget(self.combo_roi)
        
        self.lbl_spatial_tool = QLabel("Spatial Analysis Tool:")
        roi_layout.addWidget(self.lbl_spatial_tool)
        self.combo_spatial_tool = QComboBox()
        self.combo_spatial_tool.setFixedHeight(22)
        self.combo_spatial_tool.addItems(["None", "Point", "Line", "Rectangle", "Ellipse"])
        self.combo_spatial_tool.currentTextChanged.connect(self.change_spatial_tool)
        roi_layout.addWidget(self.combo_spatial_tool)
        
        self.btn_edit_region = QPushButton("Edit region")
        self.btn_edit_region.setFixedHeight(22)
        self.btn_edit_region.setStyleSheet("font-size: 11px; padding: 0px 4px;")
        self.btn_edit_region.hide()
        self.btn_edit_region.clicked.connect(self.open_edit_region_dialog)
        roi_layout.addWidget(self.btn_edit_region)
        
        self.lbl_spatial_tool.hide()
        self.combo_spatial_tool.hide()
        roi_layout.addStretch()
        
        # Draw a reliable square grid icon using QPainter to avoid any font or XPM string issues
        from PyQt5.QtGui import QIcon, QPixmap, QPainter, QPen, QColor
        pix = QPixmap(16, 16)
        pix.fill(Qt.transparent)
        painter = QPainter(pix)
        painter.setPen(QPen(QColor(255, 255, 255), 1))
        painter.drawRect(2, 2, 11, 11)
        painter.drawLine(6, 2, 6, 13)
        painter.drawLine(10, 2, 10, 13)
        painter.drawLine(2, 6, 13, 6)
        painter.drawLine(2, 10, 13, 10)
        painter.end()
        self.btn_contour_overlay = QPushButton("Contour Options")
        self.btn_contour_overlay.setToolTip("Configure contour overlay from external FITS file")
        self.btn_contour_overlay.setFixedHeight(22)
        self.btn_contour_overlay.setStyleSheet("font-size: 11px; padding: 0px 6px;")
        self.btn_contour_overlay.clicked.connect(self.open_contour_options)
        self.btn_contour_overlay.setEnabled(False)
        self.btn_contour_overlay.hide()
        roi_layout.addWidget(self.btn_contour_overlay)

        self.btn_grid = QPushButton()
        self.btn_grid.setIcon(QIcon(pix))
        self.btn_grid.setToolTip("View channel grid")
        self.btn_grid.setFixedWidth(30)
        self.btn_grid.setFixedHeight(22)
        self.btn_grid.clicked.connect(self.open_channel_grid_popup)
        roi_layout.addWidget(self.btn_grid)
        
        channel_layout.addLayout(roi_layout)

        top_half.addWidget(self.frame_channel, stretch=4)
        self.frames['channel'] = self.frame_channel

        # --- Spectrum / Spatial ---
        self.frame_spectrum = QFrame()
        self.frame_spectrum.setObjectName("PanelFrame")
        self.panel_layout = QVBoxLayout(self.frame_spectrum)
        
        top_bar = QHBoxLayout()
        self.combo_panel_mode = QComboBox()
        self.combo_panel_mode.addItems(["Spectrum", "Spatial Analysis"])
        self.combo_panel_mode.setStyleSheet("font-weight: bold; color: #3498db; font-size: 13px;")
        self.combo_panel_mode.currentTextChanged.connect(self.switch_panel_mode)
        top_bar.addWidget(self.combo_panel_mode)
        top_bar.addStretch()
        self.panel_layout.addLayout(top_bar)
        
        self.stacked_panel = QStackedWidget()
        self.panel_layout.addWidget(self.stacked_panel)
        
        self.spectrum_widget = QWidget()
        spectrum_layout = QVBoxLayout(self.spectrum_widget)
        spectrum_layout.setContentsMargins(0,0,0,0)
        self.stacked_panel.addWidget(self.spectrum_widget)
        
        self.spatial_widget = QWidget()
        spatial_layout = QVBoxLayout(self.spatial_widget)
        spatial_layout.setContentsMargins(0,0,0,0)
        
        spatial_controls_layout = QHBoxLayout()
        self.lbl_spatial_region_sel = QLabel("Select Region:")
        self.combo_spatial_regions = QComboBox()
        self.combo_spatial_regions.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.combo_spatial_regions.setMinimumContentsLength(14)
        self.combo_spatial_regions.addItem("None")
        self.combo_spatial_regions.currentTextChanged.connect(self.on_spatial_region_selected)
        
        self.btn_delete_spatial = QPushButton("Delete Selected")
        self.btn_delete_spatial.clicked.connect(self.delete_selected_spatial_via_button)
        
        spatial_controls_layout.addWidget(self.lbl_spatial_region_sel)
        spatial_controls_layout.addWidget(self.combo_spatial_regions)
        spatial_controls_layout.addWidget(self.btn_delete_spatial)
        spatial_controls_layout.addStretch()
        
        spatial_layout.addLayout(spatial_controls_layout)
        
        self.plot_spatial_1 = pg.PlotWidget(title="X Profile / Spatial Profile")
        self.plot_spatial_1.showGrid(x=True, y=True, alpha=0.3)
        self.plot_spatial_1.setLabel('bottom', 'Offset (arcsec)')
        self.plot_spatial_1.setLabel('left', 'Flux')
        self.plot_spatial_1.addLegend(offset=(1, 1))
        self.curve_spatial_1 = self.plot_spatial_1.plot([], [], pen=pg.mkPen('w', width=2), name="Base")
        
        self.plot_spatial_2 = pg.PlotWidget(title="Y Profile")
        self.plot_spatial_2.showGrid(x=True, y=True, alpha=0.3)
        self.plot_spatial_2.setLabel('bottom', 'Offset (arcsec)')
        self.plot_spatial_2.setLabel('left', 'Flux')
        self.plot_spatial_2.addLegend(offset=(1, 1))
        self.curve_spatial_2 = self.plot_spatial_2.plot([], [], pen=pg.mkPen('w', width=2), name="Base")
        
        self.lbl_spatial_stats = QLabel("Draw a region to see statistics.")
        self.lbl_spatial_stats.setAlignment(Qt.AlignCenter)
        self.lbl_spatial_stats.setStyleSheet("font-size: 13px; color: #aaa; background-color: #1a1a1a; border: 1px solid #333; border-radius: 4px; padding: 10px;")

        self.spatial_stats_scroll = QScrollArea()
        self.spatial_stats_scroll.setWidgetResizable(True)
        self.spatial_stats_scroll.setStyleSheet("QScrollArea { border: 1px solid #333; border-radius: 4px; background-color: #1a1a1a; }")
        self.spatial_stats_container = QWidget()
        self.spatial_stats_container.setStyleSheet("background-color: #1a1a1a;")
        self.spatial_stats_layout = QHBoxLayout(self.spatial_stats_container)
        self.spatial_stats_layout.setContentsMargins(5, 5, 5, 5)
        self.spatial_stats_scroll.setWidget(self.spatial_stats_container)
        self.spatial_stats_scroll.hide()

        self.stacked_spatial_info = QStackedWidget()
        self.stacked_spatial_info.addWidget(self.lbl_spatial_stats)
        self.stacked_spatial_info.addWidget(self.spatial_stats_scroll)

        spatial_layout.addWidget(self.plot_spatial_1, stretch=1)
        spatial_layout.addWidget(self.plot_spatial_2, stretch=1)
        spatial_layout.addWidget(self.stacked_spatial_info)
        self.stacked_panel.addWidget(self.spatial_widget)

        self.pv_widget = QWidget()
        pv_layout = QVBoxLayout(self.pv_widget)
        pv_layout.setContentsMargins(0, 0, 0, 0)

        pv_controls_layout = QHBoxLayout()
        self.lbl_pv_cut_sel = QLabel("Select Cut:")
        self.combo_pv_cuts = QComboBox()
        self.combo_pv_cuts.addItem("None")
        self.combo_pv_cuts.currentTextChanged.connect(self.on_pv_cut_selected)
        self.btn_edit_pv = QPushButton("Edit")
        self.btn_edit_pv.setFixedHeight(22)
        self.btn_edit_pv.setStyleSheet("font-size: 11px; padding: 0px 4px;")
        self.btn_edit_pv.clicked.connect(self.open_edit_pv_cut_dialog)
        self.btn_delete_pv = QPushButton("Delete Selected")
        self.btn_delete_pv.clicked.connect(self.delete_selected_pv_via_button)
        pv_controls_layout.addWidget(self.lbl_pv_cut_sel)
        pv_controls_layout.addWidget(self.combo_pv_cuts)
        pv_controls_layout.addWidget(self.btn_edit_pv)
        pv_controls_layout.addWidget(self.btn_delete_pv)
        pv_controls_layout.addStretch()
        pv_layout.addLayout(pv_controls_layout)

        self.lbl_pv_help = QLabel("Ctrl+drag on the channel map to draw a PV cut.")
        self.lbl_pv_help.setAlignment(Qt.AlignCenter)
        self.lbl_pv_help.setStyleSheet("font-size: 12px; color: #aaa;")
        pv_layout.addWidget(self.lbl_pv_help)

        self.pv_plot_item = pg.PlotItem(title="PV Diagram")
        self.pv_hover_line_main = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('m', width=1.5, style=Qt.DashLine))
        self.pv_hover_line_main.hide()
        self.pv_plot_item.addItem(self.pv_hover_line_main)
        self.pv_plot_item.showGrid(x=True, y=True, alpha=0.3)
        self.pv_plot_item.setLabel('bottom', 'Offset along cut (arcsec)')
        self.pv_plot_item.setLabel('left', 'Radio Velocity (km/s)')
        self.pv_view = pg.ImageView(view=self.pv_plot_item)
        self.pv_view.ui.roiBtn.hide()
        self.pv_view.ui.menuBtn.hide()
        self.pv_view.ui.histogram.gradient.loadPreset('turbo')
        self.pv_view.ui.histogram.setFixedWidth(160)
        fix_axis_scaling(self.pv_view.ui.histogram.axis)
        pv_layout.addWidget(self.pv_view, stretch=1)

        self.lbl_hover_pv = QLabel("")
        self.lbl_hover_pv.setStyleSheet("color: #aaa; font-size: 9.5px;")
        pv_layout.addWidget(self.lbl_hover_pv)
        self.stacked_panel.addWidget(self.pv_widget)
        
        self.spectrum_viewbox = SpectrumViewBox()
        self.spectrum_viewbox.parent_tab = self
        self.plot_widget = pg.PlotWidget(viewBox=self.spectrum_viewbox)
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        fix_axis_scaling(self.plot_widget.getAxis('left')) 
        self.plot_widget.setLabel('bottom', 'Radio Velocity (km/s)')
        self.plot_widget.setLabel('left', 'Flux') 
        self.plot_widget.addLegend(offset=(10, 10))
        
        self.spectrum_curves = {} # mapping from region name to PlotDataItem
        # Original curve (fallback for Whole Map)
        self.spectrum_curve = pg.PlotDataItem([], [], stepMode="center", pen=pg.mkPen('w', width=2))
        self.plot_widget.addItem(self.spectrum_curve)
        
        self.v_line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('g', width=2, style=Qt.DashLine))
        self.v_line.hide()
        self.plot_widget.addItem(self.v_line)
        
        self.region = pg.LinearRegionItem([0, 1], brush=pg.mkBrush(52, 152, 219, 40))
        self.region.setZValue(10)
        self.region.hide()
        for line in self.region.lines:
            line.setPen(pg.mkPen(color='#3498db', width=3))
            line.setHoverPen(pg.mkPen(color='#f1c40f', width=5))
        self.plot_widget.addItem(self.region)
        
        self.box_regions = QGroupBox("Regions")
        self.box_regions.setStyleSheet("QGroupBox { color: white; font-weight: bold; } QCheckBox { color: white; }")
        self.box_regions_layout = QHBoxLayout(self.box_regions)
        self.box_regions_layout.setContentsMargins(5, 10, 5, 5)
        self.box_regions.hide()
        spectrum_layout.addWidget(self.box_regions)
        
        self.spectrum_tabs = QTabWidget()
        self.spectrum_tabs.addTab(self.plot_widget, "Original")
        self.spectrum_tabs.setTabsClosable(True)
        self.spectrum_tabs.tabCloseRequested.connect(self._on_spectrum_tab_close_requested)
        from PyQt5.QtWidgets import QTabBar
        self.spectrum_tabs.tabBar().setTabButton(0, QTabBar.RightSide, None)
        self.spectrum_tabs.tabBar().setTabButton(0, QTabBar.LeftSide, None)
        self.spectrum_tabs.tabBar().hide()
        spectrum_layout.addWidget(self.spectrum_tabs)
        
        self.spectrum_viewbox_smooth = SpectrumViewBox()
        self.spectrum_viewbox_smooth.parent_tab = self
        self.plot_widget_smooth = pg.PlotWidget(viewBox=self.spectrum_viewbox_smooth)
        self.plot_widget_smooth.showGrid(x=True, y=True, alpha=0.3)
        fix_axis_scaling(self.plot_widget_smooth.getAxis('left'))
        self.plot_widget_smooth.setLabel('bottom', 'Radio Velocity (km/s)')
        self.plot_widget_smooth.setLabel('left', 'Flux')
        self.plot_widget_smooth.addLegend(offset=(10, 10))
        
        self.spectrum_curves_smooth = {}
        self.spectrum_curve_smooth = pg.PlotDataItem([], [], stepMode="center", pen=pg.mkPen('c', width=2))
        self.plot_widget_smooth.addItem(self.spectrum_curve_smooth)
        
        self.smooth_active_line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('g', width=2, style=Qt.DashLine))
        self.smooth_active_line.hide()
        self.plot_widget_smooth.addItem(self.smooth_active_line)
        
        self.smooth_velocity_region = pg.LinearRegionItem([0, 1], brush=pg.mkBrush(52, 152, 219, 40))
        self.smooth_velocity_region.setZValue(10)
        self.smooth_velocity_region.hide()
        for line in self.smooth_velocity_region.lines:
            line.setPen(pg.mkPen(color='#3498db', width=3))
            line.setHoverPen(pg.mkPen(color='#f1c40f', width=5))
        self.plot_widget_smooth.addItem(self.smooth_velocity_region)
        
        self.v_line.sigPositionChanged.connect(lambda *args: self.smooth_active_line.setValue(self.v_line.value()))
        self.region.sigRegionChanged.connect(self._sync_smooth_region_from_main)
        self.smooth_velocity_region.sigRegionChanged.connect(self._sync_main_region_from_smooth)

        self.smoothing_params = None
        self.spectrum_tabs.currentChanged.connect(self._on_spectrum_tab_changed)
        
        self.lbl_hover_spec = QLabel("")
        self.lbl_hover_spec.setStyleSheet("color: #aaa; font-size: 9.5px;")
        
        hover_layout = QHBoxLayout()
        hover_layout.addWidget(self.lbl_hover_spec)
        hover_layout.addStretch()
        
        self.lbl_rj_warning = QLabel("Warning: Brightness converted using the Rayleigh–Jeans approximation.")
        self.lbl_rj_warning.setStyleSheet("color: orange; font-weight: normal;")
        self.lbl_rj_warning.hide()
        hover_layout.addWidget(self.lbl_rj_warning)
        
        spectrum_layout.addLayout(hover_layout)

        input_layout = QHBoxLayout()
        input_layout.addWidget(QLabel("Statistic:"))
        self.combo_spec_stat = QComboBox()
        self.combo_spec_stat.addItems(["Flux Density", "Mean", "Median", "Max (Peak)"])
        self.combo_spec_stat.setCurrentText("Mean")
        self.combo_spec_stat.currentTextChanged.connect(self._update_spectrum_state_machine)
        self.combo_spec_stat.currentTextChanged.connect(self.update_spectrum)
        self.combo_spec_stat.currentTextChanged.connect(lambda: self.lbl_region_result.setText("---"))
        input_layout.addWidget(self.combo_spec_stat)
        
        input_layout.addWidget(QLabel("Unit:"))
        self.combo_spec_unit = QComboBox()
        self.combo_spec_unit.addItems(["Native", "Jy", "K", "Jy/beam"])
        self.combo_spec_unit.currentTextChanged.connect(self.update_spectrum)
        input_layout.addWidget(self.combo_spec_unit)
        
        #self.btn_smooth = QPushButton("Smooth")
        self.btn_smooth = QPushButton()
        smooth_icon = qta.icon('fa5s.bezier-curve')
        self.btn_smooth.setIcon(smooth_icon)
        self.btn_smooth.setToolTip("Spectral smoothing")
        self.btn_smooth.setFixedWidth(36)
        self.btn_smooth.setFixedHeight(26)
        self.btn_smooth.setStyleSheet("font-size: 10px; padding: 2px;")
        self.btn_smooth.clicked.connect(self.open_smoothing_dialog)
        input_layout.addWidget(self.btn_smooth)
        input_layout.addStretch()
        input_layout.addWidget(QLabel("Min:"))
        self.input_vmin = QLineEdit("0.00")
        self.input_vmin.setMinimumWidth(65)
        self.input_vmin.setMaximumWidth(80)
        input_layout.addWidget(self.input_vmin)
        
        input_layout.addWidget(QLabel("Max:"))
        self.input_vmax = QLineEdit("1.00")
        self.input_vmax.setMinimumWidth(65)
        self.input_vmax.setMaximumWidth(80)
        input_layout.addWidget(self.input_vmax)
        
        # --- NEW UI CODE START ---
        input_layout.addWidget(QLabel("Ref. Freq. (GHz):"))
        self.input_ref_freq = QLineEdit("")
        self.input_ref_freq.setMinimumWidth(115)
        self.input_ref_freq.returnPressed.connect(self.update_spectral_axis)
        self.input_ref_freq.editingFinished.connect(self.update_spectral_axis)
        input_layout.addWidget(self.input_ref_freq)
        
        self.combo_axis_type = QComboBox()
        self.combo_axis_type.addItems(["Radio Velocity", "Optical Velocity", "Frequency", "Wavelength", "Channel"])
        self.combo_axis_type.setCurrentText("Radio Velocity")
        self.combo_axis_type.setMaximumWidth(120)
        self.combo_axis_type.setStyleSheet("font-size: 11px;")
        self.combo_axis_type.currentIndexChanged.connect(self.update_spectral_axis)
        input_layout.addWidget(self.combo_axis_type)
        # --- NEW UI CODE END ---
        
        # Hidden backing widgets (used by update_spectrum_region_calc)
        self.combo_regions = QComboBox(); self.combo_regions.addItem("None")
        self.combo_regions_2 = QComboBox(); self.combo_regions_2.addItem("None")
        self.combo_regions_3 = QComboBox(); self.combo_regions_3.addItem("None")
        self.combo_region_calc = QComboBox()
        self.combo_region_calc.addItems(["Integrated intensity", "RMS"])
        self.lbl_region_result = QLabel("---")
        self.lbl_regions = QLabel("")
        self.lbl_plus1 = QLabel("")
        self.lbl_plus2 = QLabel("")
        self.lbl_calc = QLabel("")

        # Spectral Statistics popup button
        #self.btn_spectral_stats = QPushButton("Statistics")
        self.btn_spectral_stats = QPushButton()
        stats_icon = qta.icon('fa5s.chart-area')
        self.btn_spectral_stats.setIcon(stats_icon)
        self.btn_spectral_stats.setToolTip("Open spectral statistics panel for drawn velocity boxes")
        self.btn_spectral_stats.setFixedWidth(36)
        self.btn_spectral_stats.setFixedHeight(26)
        self.btn_spectral_stats.clicked.connect(self.open_spectral_stats_popup)
        self.btn_spectral_stats.hide()
        input_layout.addWidget(self.btn_spectral_stats)

        self.spectrum_rois = []
        self.rois_to_delete = []
        self._spectral_stats_popup = None
        self._channel_grid_popup = None
        
        spectrum_layout.addLayout(input_layout)

        self.input_vmin.editingFinished.connect(self.update_region_from_text)
        self.input_vmax.editingFinished.connect(self.update_region_from_text)
        self.region.sigRegionChanged.connect(self.update_text_from_region)
        self.region.sigRegionChanged.connect(self._on_region_drag_start)
        self.region.sigRegionChangeFinished.connect(self._on_region_drag_end)
        
        self.smooth_velocity_region.sigRegionChanged.connect(self.update_text_from_region)
        self.smooth_velocity_region.sigRegionChanged.connect(self._on_region_drag_start)
        self.smooth_velocity_region.sigRegionChangeFinished.connect(self._on_region_drag_end)
        
        self._region_dragging = False

        top_half.addWidget(self.frame_spectrum, stretch=7)
        self.spatial_rois = []
        self.spatial_rois_to_delete = []
        self.pv_cuts = []
        self.pv_cuts_to_delete = []
        self.frames['spectrum'] = self.frame_spectrum
        main_layout.addLayout(top_half, stretch=1)

        # ==================== BOTTOM HALF ====================
        self.toggle_bottom_btn = QPushButton("▼ Show Moment/PV Panels")
        self.toggle_bottom_btn.setStyleSheet("QPushButton { border: none; color: #3498db; text-align: left; padding: 5px; font-weight: bold; background: transparent; } QPushButton:hover { color: #5dade2; }")
        self.toggle_bottom_btn.clicked.connect(self.toggle_bottom_pane)
        main_layout.addWidget(self.toggle_bottom_btn)

        self.bottom_container = QWidget()
        self.bottom_half = QHBoxLayout(self.bottom_container)
        self.bottom_half.setContentsMargins(0, 0, 0, 0)
        self.bottom_half.setSpacing(6)
        self.panels = []

        moment_options = ["Moment -1 (Mean Intensity)", "Moment 0 (Integrated Intensity)",
                          "Moment 1 (Velocity Field)", "Moment 2 (Velocity Dispersion)",
                          "Moment 8 (Peak Intensity)", "Moment 9 (Peak Velocity)", "PV Diagram"]

        for i, default_option in enumerate([moment_options[1], moment_options[4], moment_options[2]]):
            panel = {}
            panel_frame = QFrame()
            panel_frame.setObjectName("PanelFrame")
            panel_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            panel_frame.setMinimumWidth(0)
            
            panel_layout = QVBoxLayout(panel_frame)
            
            top_ctrl_layout = QHBoxLayout()
            combo = QComboBox()
            combo.addItems(moment_options)
            combo.setCurrentText(default_option)
            combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
            combo.setMinimumContentsLength(10)
            top_ctrl_layout.addWidget(combo, stretch=1)
            
            aux_stack = QStackedWidget()
            aux_stack.setContentsMargins(0, 0, 0, 0)
            aux_stack.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

            thresh_widget = QWidget()
            thresh_layout = QHBoxLayout(thresh_widget)
            thresh_layout.setContentsMargins(5, 0, 0, 0)

            thresh_layout.addWidget(QLabel("Min. Intensity:"))
            input_thresh = QLineEdit("0.000")
            input_thresh.setMinimumWidth(80)
            thresh_layout.addWidget(input_thresh)
            
            btn_pick = QPushButton("💧")
            btn_pick.setToolTip("Pick min. intensity threshold from raw map")
            btn_pick.setFixedWidth(35)
            btn_pick.setCheckable(True)
            btn_pick.setStyleSheet("QPushButton:checked { background-color: #d35400; font-weight: bold; }")
            thresh_layout.addWidget(btn_pick)
            thresh_layout.addStretch()
            
            pv_controls_widget = QWidget()
            pv_controls_layout = QHBoxLayout(pv_controls_widget)
            pv_controls_layout.setContentsMargins(5, 0, 0, 0)
            pv_controls_layout.setSpacing(2)
            pv_controls_layout.addWidget(QLabel("Cut:"))
            combo_pv_cut = QComboBox()
            combo_pv_cut.addItem("None")
            combo_pv_cut.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            combo_pv_cut.setMinimumContentsLength(6)
            combo_pv_cut.setFixedWidth(80)
            pv_controls_layout.addWidget(combo_pv_cut, stretch=1)
            pv_controls_layout.addWidget(QLabel("Range:"))
            combo_pv_range = QComboBox()
            combo_pv_range.addItems(["Selected", "Full Cube"])
            combo_pv_range.setCurrentText("Selected")
            combo_pv_range.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            combo_pv_range.setMinimumContentsLength(9)
            combo_pv_range.setFixedWidth(98)
            pv_controls_layout.addWidget(combo_pv_range)
            btn_edit_pv = QPushButton()
            edit_icon = qta.icon('fa5s.pen')  # Creates a solid pencil icon
            btn_edit_pv.setIcon(edit_icon)
            btn_edit_pv.setToolTip("Edit PV cut")
            btn_edit_pv.setFixedHeight(22)
            btn_edit_pv.setStyleSheet("font-size: 11px; padding: 0px 4px;")
            pv_controls_layout.addWidget(btn_edit_pv)
            btn_delete_pv = QPushButton()
            trash_icon = qta.icon('mdi.trash-can-outline')
            btn_delete_pv.setIcon(trash_icon)
            btn_delete_pv.setToolTip("Remove PV Diagram Plot")
            btn_delete_pv.setFixedWidth(30)
            btn_delete_pv.setFixedHeight(22)
            pv_controls_layout.addWidget(btn_delete_pv)

            aux_stack.addWidget(thresh_widget)
            aux_stack.addWidget(pv_controls_widget)
            top_ctrl_layout.addWidget(aux_stack, stretch=1)
            panel_layout.addLayout(top_ctrl_layout)
            
            p_bottom = WCSAxisItem(orientation='bottom')
            p_left = WCSAxisItem(orientation='left')
            plot_item = pg.PlotItem(axisItems={'bottom': p_bottom, 'left': p_left})
            
            plot_item.invertX(True)
            plot_item.setLabel('bottom', 'RA offset (arcsec)')
            plot_item.setLabel('left', 'Dec offset (arcsec)')
            
            view = pg.ImageView(view=plot_item)
            plot_item.invertY(False)

            view.ui.roiBtn.hide()
            view.ui.menuBtn.hide()
            view.ui.histogram.setMaximumWidth(160)
            view.ui.histogram.setMinimumWidth(80)
            fix_axis_scaling(view.ui.histogram.axis) 
            plot_item.vb.sigRangeChanged.connect(self.update_beam_positions)
            panel_layout.addWidget(view, stretch=1)
            
            lbl_hover = QLabel("")
            lbl_hover.setStyleSheet("color: #aaa; font-size: 9.5px;")
            panel_layout.addWidget(lbl_hover)

            self.bottom_half.addWidget(panel_frame, stretch=1)
            self.frames[i] = panel_frame
            
            panel['combo'] = combo
            panel['view'] = view
            panel['plot_item'] = plot_item
            panel['aux_stack'] = aux_stack
            panel['thresh_widget'] = thresh_widget
            panel['input_thresh'] = input_thresh
            panel['btn_pick'] = btn_pick
            panel['pv_controls_widget'] = pv_controls_widget
            panel['combo_pv_cut'] = combo_pv_cut
            panel['combo_pv_range'] = combo_pv_range
            panel['btn_edit_pv'] = btn_edit_pv
            panel['btn_delete_pv'] = btn_delete_pv
            panel['lbl_hover'] = lbl_hover
            panel['current_data'] = None
            panel['pv_offset_axis'] = None
            panel['pv_velocity_axis'] = None
            panel['id'] = i
            panel['unit'] = ''
            panel['pv_hover_line'] = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('m', width=1.5, style=Qt.DashLine))
            panel['pv_hover_line'].hide()
            plot_item.addItem(panel['pv_hover_line'])
            self.panels.append(panel)

            combo.currentTextChanged.connect(self.update_moment_maps)
            input_thresh.editingFinished.connect(self.update_moment_maps)
            combo_pv_cut.currentTextChanged.connect(lambda _text, p_id=i: self.on_panel_pv_cut_selected(p_id))
            combo_pv_range.currentTextChanged.connect(self.update_moment_maps)
            btn_delete_pv.clicked.connect(lambda _checked=False, p_id=i: self.delete_panel_pv_cut(p_id))
            btn_edit_pv.clicked.connect(self.open_edit_pv_cut_dialog)
            plot_item.scene().sigMouseMoved.connect(lambda pos, p=panel: self.hover_panel(pos, p))
            btn_pick.clicked.connect(lambda checked, p_id=i: self.set_active_picker(checked, p_id))

        self.bottom_container.setVisible(False)
        main_layout.addWidget(self.bottom_container, stretch=1)

        self.set_active_panel('channel')

        self.plot_channel.scene().sigMouseMoved.connect(lambda pos: self.hover_event(pos, self.plot_channel, self.get_current_channel_data(), self.lbl_hover_ch, 'channel'))
        self.plot_widget.scene().sigMouseMoved.connect(lambda pos: self.hover_spectrum(pos, self.plot_widget))
        self.plot_widget_smooth.scene().sigMouseMoved.connect(lambda pos: self.hover_spectrum(pos, self.plot_widget_smooth))
        self.pv_plot_item.scene().sigMouseMoved.connect(self.hover_pv)
        
        self.plot_widget.scene().sigMouseClicked.connect(lambda event: self.universal_click_handler(event, self.plot_widget))
        self.plot_widget_smooth.scene().sigMouseClicked.connect(lambda event: self.universal_click_handler(event, self.plot_widget_smooth))
        self.plot_channel.scene().sigMouseClicked.connect(lambda event: self.universal_click_handler(event, self.plot_channel))
        self.pv_plot_item.scene().sigMouseClicked.connect(lambda _event: self.set_active_panel('spectrum'))
        for p in self.panels:
            p['plot_item'].scene().sigMouseClicked.connect(lambda event, view=p['plot_item']: self.universal_click_handler(event, view))

    def toggle_bottom_pane(self):
        """
        Toggle the visibility of the bottom Moment/PV panels container and resize the main window accordingly.
        """
        is_visible = self.bottom_container.isVisible()
        main_win = self.window()
        
        from PyQt5.QtWidgets import QApplication
        screen_height = QApplication.desktop().availableGeometry(main_win).height()
        
        if is_visible:
            # Hiding the bottom pane
            self.bottom_container.setVisible(False)
            self.toggle_bottom_btn.setText("▼ Show Moment/PV Panels")
            
            if main_win.isMaximized():
                main_win.showNormal()
                
            if hasattr(main_win, 'startup_width') and hasattr(main_win, 'startup_height'):
                main_win.resize(main_win.startup_width, main_win.startup_height)
            else:
                main_win.resize(main_win.width(), main_win.height() // 2)
        else:
            # Showing the bottom pane
            target_height = int(main_win.height() * 1.6)
            
            if screen_height > target_height:
                main_win.resize(main_win.width(), target_height)
            else:
                main_win.showMaximized()
                
            self.bottom_container.setVisible(True)
            self.toggle_bottom_btn.setText("▲ Hide Moment/PV Panels")

    def switch_panel_mode(self, mode):
        """
        Switch the active control panel mode between 'Spectrum' and 'Spatial Analysis'.

        Parameters
        ----------
        mode : str
            The target mode string.
        """
        if mode == "Spectrum":
            self.stacked_panel.setCurrentWidget(self.spectrum_widget)
            self.lbl_combo_roi.show()
            self.combo_roi.show()
            self.lbl_spatial_tool.hide()
            self.combo_spatial_tool.hide()
        elif mode == "Spatial Analysis":
            self.stacked_panel.setCurrentWidget(self.spatial_widget)
            self.lbl_combo_roi.hide()
            self.combo_roi.hide()
            self.lbl_spatial_tool.show()
            self.combo_spatial_tool.show()
            
            if not getattr(self, 'spatial_rois', []):
                self.combo_spatial_tool.blockSignals(True)
                self.combo_spatial_tool.setCurrentText("None")
                self.combo_spatial_tool.blockSignals(False)
            
            self.change_spatial_tool(self.combo_spatial_tool.currentText(), auto_draw=False)

    def delete_nr_roi(self):
        """
        Delete the Noise Region (NR) ROI and its associated label from the view.
        """
        if getattr(self, 'nr_roi', None) is not None:
            try:
                if self.nr_roi.scene() is not None:
                    self.view_channel.getView().removeItem(self.nr_roi)
            except Exception: pass
            self.nr_roi = None
        if getattr(self, 'nr_label', None) is not None:
            try:
                if self.nr_label.scene() is not None:
                    self.view_channel.getView().removeItem(self.nr_label)
            except Exception: pass
            self.nr_label = None
        self.nr_roi_selected = False

    def update_nr_rms(self):
        """
        Trigger an update of the calculated Root Mean Square (RMS) noise using the current NR ROI.
        """
        return self.controller.update_nr_rms()
    def handle_escape(self):
        """
        Handle 'Escape' key presses to gracefully exit active drawing/picking states or clear selections.
        """
        if getattr(self, 'active_picker_panel', None) is not None:
            pid = self.active_picker_panel
            self.set_active_picker(False, pid)
            if hasattr(self, 'panels') and pid < len(self.panels):
                self.panels[pid]['btn_pick'].setChecked(False)
            return
        if getattr(self, 'nr_roi_selected', False):
            self.delete_nr_roi()
            return
            
        if getattr(self, 'spatial_rois_to_delete', []):
            self.delete_selected_spatial_regions()
        if getattr(self, 'pv_cuts_to_delete', []):
            self.delete_selected_pv_cuts()
        if getattr(self, 'rois_to_delete', []):
            self.delete_selected_regions()
        if getattr(self, 'active_spatial_spectrum_roi', None):
            self.remove_spatial_spectrum_roi(self.active_spatial_spectrum_roi)
            self.active_spatial_spectrum_roi = None
    def open_smoothing_dialog(self):
        """
        Launch the spectral smoothing configuration dialog in non-modal mode.
        """
        # Prevent multiple instances
        if getattr(self, '_smooth_dialog_active', False) and hasattr(self, '_smooth_dialog') and self._smooth_dialog:
            self._smooth_dialog.raise_()
            self._smooth_dialog.activateWindow()
            return
            
        # Mutual exclusion: warn if Edit Region dialog is open
        if getattr(self, '_region_dialog', None) and self._region_dialog.isVisible():
            msg = QMessageBox(self.window())
            msg.setWindowTitle("Dialog Open")
            msg.setText("Please close the 'Edit Region' dialog before opening Smooth.")
            msg.setIcon(QMessageBox.Information)
            msg.setWindowFlags(msg.windowFlags() | Qt.WindowStaysOnTopHint)
            msg.exec_()
            self._region_dialog.raise_()
            self._region_dialog.activateWindow()
            return
            
        from src.gui.dialogs import SpectralSmoothingDialog
        self._smooth_dialog_active = True
        
        self._smooth_dialog = SpectralSmoothingDialog(self.window())
        self._smooth_dialog.setWindowModality(Qt.NonModal)
        
        def on_apply(params):
            """
            Callback to apply smoothing parameters from the dialog.

            Parameters
            ----------
            params : dict
                The smoothing configuration dictionary.
            """
            self.smoothing_params = params
            if self.spectrum_tabs.indexOf(self.plot_widget_smooth) == -1:
                self.spectrum_tabs.addTab(self.plot_widget_smooth, "Smoothed")
            self.spectrum_tabs.tabBar().show()
            self.spectrum_tabs.setCurrentWidget(self.plot_widget_smooth)
            self.update_spectrum()
            
        def on_close():
            """
            Callback triggered when the smoothing dialog finishes/closes.
            """
            self._smooth_dialog_active = False
            self._smooth_dialog = None
            
        self._smooth_dialog.apply_clicked.connect(on_apply)
        self._smooth_dialog.finished.connect(on_close)
        self._smooth_dialog.show()

    def _on_spectrum_tab_close_requested(self, index):
        """
        Handle requests to close a specific tab in the spectrum plot area.

        Parameters
        ----------
        index : int
            The tab index requested for closure.
        """
        if self.spectrum_tabs.widget(index) == self.plot_widget_smooth:
            self.remove_smoothed_spectrum()

    def remove_smoothed_spectrum(self):
        """
        Close the smoothed spectrum tab and remove its associated data.
        """
        self.smoothing_params = None
        idx = self.spectrum_tabs.indexOf(self.plot_widget_smooth)
        if idx != -1:
            self.spectrum_tabs.removeTab(idx)
        self.spectrum_tabs.tabBar().hide()
        self.spectrum_tabs.setCurrentWidget(self.plot_widget)
        self._on_spectrum_tab_changed()
        self.update_spectrum()

    def get_active_spectrum_plot(self):
        """
        Retrieve the currently active spectrum pyqtgraph PlotWidget.

        Returns
        -------
        pyqtgraph.PlotWidget or None
            The currently visible spectrum plot widget.
        """
        if getattr(self, 'spectrum_tabs', None) is not None and getattr(self, 'plot_widget_smooth', None) is not None:
            if self.spectrum_tabs.currentWidget() == self.plot_widget_smooth:
                return self.plot_widget_smooth
        return getattr(self, 'plot_widget', None)

    def get_active_spectrum_rois(self):
        """
        Get the list of active Regions of Interest (ROIs) for the current spectrum plot.

        Returns
        -------
        list
            A list of dictionary objects containing ROI metadata and pyqtgraph items.
        """
        if getattr(self, 'spectrum_tabs', None) is not None and getattr(self, 'plot_widget_smooth', None) is not None:
            if self.spectrum_tabs.currentWidget() == self.plot_widget_smooth:
                if not hasattr(self, 'spectrum_rois_smooth'):
                    self.spectrum_rois_smooth = []
                return self.spectrum_rois_smooth
        if not hasattr(self, 'spectrum_rois'):
            self.spectrum_rois = []
        return self.spectrum_rois

    def _on_spectrum_tab_changed(self):
        """
        Handle UI state synchronization when the user switches between spectrum tabs (e.g., raw vs smoothed).
        """
        current_widget = self.spectrum_tabs.currentWidget()
        if not current_widget: return
            
        self.update_region_ui_visibility()
        self.rename_regions()
        self.update_spectrum_region_calc()

    def any_pv_panels_active(self):
        """
        Check if any of the bottom auxiliary panels are currently set to 'PV Diagram' mode.

        Returns
        -------
        bool
            True if at least one PV panel is active, False otherwise.
        """
        return any(panel['combo'].currentText() == "PV Diagram" for panel in self.panels)

    def is_pv_drawing_mode(self):
        """
        Check if the application is currently in a state that allows drawing new PV cuts.

        Returns
        -------
        bool
            True if drawing is permitted, False otherwise.
        """
        return self.combo_panel_mode.currentText() != "Spatial Analysis" and self.any_pv_panels_active()

    def get_velocity_subset(self, use_full_range=False):
        """
        Retrieve a subset of the velocity axis based on the currently selected spectral region.

        Parameters
        ----------
        use_full_range : bool, optional
            If True, returns the full velocity axis regardless of the selected region.

        Returns
        -------
        numpy.ndarray
            The 1D array of velocity values.
        """
        return self.controller.get_velocity_subset(use_full_range)

    def configure_bottom_panel_axes(self, panel, is_pv):
        """
        Delegate the configuration of the PyqtGraph plot axes for a specific bottom panel.

        Parameters
        ----------
        panel : dict
            The panel configuration dictionary.
        is_pv : bool
            True if the panel is being configured for a Position-Velocity diagram.
        """
        return self.controller.configure_bottom_panel_axes(panel, is_pv)

    def configure_bottom_panel_controls(self, panel, mode):
        """
        Delegate the UI state configuration for the controls (buttons, combos) of a bottom panel.

        Parameters
        ----------
        panel : dict
            The panel configuration dictionary.
        mode : str
            The new mode string (e.g., 'Moment Map', 'PV Diagram').
        """
        return self.controller.configure_bottom_panel_controls(panel, mode)

    def get_pv_cut_by_name(self, name):
        """
        Delegate the retrieval of a specific PV cut object by its unique string name.

        Parameters
        ----------
        name : str
            The name of the PV cut (e.g., 'PV_1').

        Returns
        -------
        dict or None
            The dictionary containing the PV cut geometry and metadata.
        """
        return self.controller.get_pv_cut_by_name(name)

    def get_selected_pv_cut_name(self):
        """
        Delegate the retrieval of the currently active/selected PV cut name.

        Returns
        -------
        str or None
            The name of the currently selected PV cut, or None if no valid cut is selected.
        """
        return self.controller.get_selected_pv_cut_name()

    def set_selected_pv_cut(self, name):
        """
        Delegate the programmatic selection of a PV cut in the UI combo box.

        Parameters
        ----------
        name : str
            The name of the PV cut to select.
        """
        return self.controller.set_selected_pv_cut(name)

    def refresh_all_pv_cut_combos(self):
        """
        Delegate the population and synchronization of all PV cut dropdown menus across panels.
        """
        return self.controller.refresh_all_pv_cut_combos()

    def on_panel_pv_cut_selected(self, panel_id):
        """
        Delegate the handling of a user selecting a different PV cut from a panel's dropdown.

        Parameters
        ----------
        panel_id : int
            The integer index identifying the specific bottom panel.
        """
        return self.controller.on_panel_pv_cut_selected(panel_id)

    def delete_panel_pv_cut(self, panel_id):
        """
        Delegate the deletion of the currently selected PV cut associated with a specific panel.

        Parameters
        ----------
        panel_id : int
            The integer index identifying the specific bottom panel.
        """
        return self.controller.delete_panel_pv_cut(panel_id)

    def clear_panel_pv_diagram(self, panel):
        """
        Delegate the clearing of image data and axes from a specific PV panel view.

        Parameters
        ----------
        panel : dict
            The panel configuration dictionary.
        """
        return self.controller.clear_panel_pv_diagram(panel)

    def update_panel_pv_diagram(self, panel):
        """
        Delegate the extraction and drawing of the 2D PV slice onto a specific panel.

        Parameters
        ----------
        panel : dict
            The panel configuration dictionary.
        """
        return self.controller.update_panel_pv_diagram(panel)
    def hover_panel(self, pos, panel):
        """
        Route mouse hover events in auxiliary panels to their specific handlers.

        Parameters
        ----------
        pos : pyqtgraph.Point
            The mouse scene position.
        panel : dict
            The panel configuration dictionary.
        """
        if panel['combo'].currentText() == "PV Diagram":
            self.hover_panel_pv(pos, panel)
        else:
            self.hover_event(pos, panel['plot_item'], panel['current_data'], panel['lbl_hover'], panel['id'])

    def hover_panel_pv(self, pos, panel):
        """
        Handle mouse hover events specifically for Position-Velocity diagram panels.

        Parameters
        ----------
        pos : pyqtgraph.Point
            The mouse scene position.
        panel : dict
            The PV diagram panel configuration dictionary.
        """
        self.clear_all_hover_labels()
        if panel['current_data'] is None or panel['pv_offset_axis'] is None or panel['pv_velocity_axis'] is None:
            return
        if not panel['plot_item'].sceneBoundingRect().contains(pos):
            return

        mp = panel['plot_item'].vb.mapSceneToView(pos)
        x_idx = int(np.abs(panel['pv_offset_axis'] - mp.x()).argmin())
        y_idx = int(np.abs(panel['pv_velocity_axis'] - mp.y()).argmin())
        if 0 <= x_idx < panel['current_data'].shape[0] and 0 <= y_idx < panel['current_data'].shape[1]:
            val = panel['current_data'][x_idx, y_idx]
            val_str = f"{val:.3e}" if (np.isfinite(val) and abs(val) < 1e-3 and abs(val) > 0) else f"{val:.4g}" if np.isfinite(val) else "NaN"
            panel['lbl_hover'].setText(
                f"Offset: {panel['pv_offset_axis'][x_idx]:.2f} arcsec | Vel: {panel['pv_velocity_axis'][y_idx]:.2f} km/s | {val_str} {self.display_unit}"
            )
            panel['lbl_hover'].setStyleSheet("color: #3498db; font-weight: bold; font-size: 9.5px;")

    def change_spatial_tool(self, tool, auto_draw=True):
        """
        Switch the active spatial drawing tool (e.g., Circle, Rectangle, Line, Polygon).

        Parameters
        ----------
        tool : str
            The string identifier of the tool to activate.
        auto_draw : bool, optional
            Whether to immediately start drawing the shape at the center, by default True.
        """
        return self.controller.change_spatial_tool(tool, auto_draw)

    def add_spatial_region(self, roi, tool):
        """
        Register a new spatial Region of Interest (ROI) into the application state.

        Parameters
        ----------
        roi : pyqtgraph.ROI
            The geometry representing the spatial region.
        tool : str
            The name of the tool used to create it.
        """
        return self.controller.add_spatial_region(roi, tool)

    def line_roi_hit_test(self, roi, scene_pos, tolerance=10.0):
        """
        Perform a hit test to determine if a mouse click intersects a line-like ROI.

        Parameters
        ----------
        roi : pyqtgraph.LineSegmentROI or pyqtgraph.PolyLineROI
            The line region to test.
        scene_pos : pyqtgraph.Point
            The mouse scene position.
        tolerance : float, optional
            The click tolerance in pixels, by default 10.0.

        Returns
        -------
        bool
            True if the click was within the tolerance distance of the line, False otherwise.
        """
        return self.controller.line_roi_hit_test(roi, scene_pos, tolerance)

    def on_spatial_region_selected(self, name):
        """
        Highlight a spatial ROI in the main display when its name is selected.

        Parameters
        ----------
        name : str
            The name of the spatial region.
        """
        return self.controller.on_spatial_region_selected(name)

    def delete_selected_spatial_via_button(self):
        """
        Trigger the deletion of the active spatial region via a UI button press.
        """
        return self.controller.delete_selected_spatial_via_button()

    def select_spatial_region(self, roi):
        """
        Set a specific spatial region as the active target for editing or deletion.

        Parameters
        ----------
        roi : pyqtgraph.ROI
            The target spatial ROI.
        """
        return self.controller.select_spatial_region(roi)

    def delete_selected_spatial_regions(self):
        """
        Execute the deletion of all selected/highlighted spatial regions.
        """
        return self.controller.delete_selected_spatial_regions()
    def add_pv_cut(self, roi):
        """
        Register a new Position-Velocity (PV) cut ROI into the application state.

        Parameters
        ----------
        roi : pyqtgraph.LineSegmentROI or pyqtgraph.PolyLineROI
            The geometry representing the PV cut.
        """
        return self.controller.add_pv_cut(roi)

    def on_pv_cut_selected(self, name):
        """
        Highlight a PV cut in the main display when its name is selected from a dropdown.

        Parameters
        ----------
        name : str
            The name of the PV cut.
        """
        return self.controller.on_pv_cut_selected(name)

    def open_edit_pv_cut_dialog(self):
        """
        Open the properties dialog for the currently active PV cut to adjust its geometry.
        """
        return self.controller.open_edit_pv_cut_dialog()

    def select_pv_cut(self, roi):
        """
        Set a specific PV cut as the active target for editing or deletion.

        Parameters
        ----------
        roi : pyqtgraph.ROI
            The target PV cut region.
        """
        return self.controller.select_pv_cut(roi)

    def delete_selected_pv_via_button(self):
        """
        Trigger the deletion of the active PV cut via a UI button press.
        """
        return self.controller.delete_selected_pv_via_button()

    def delete_selected_pv_cuts(self):
        """
        Execute the deletion of all selected/highlighted PV cuts.
        """
        return self.controller.delete_selected_pv_cuts()

    def clear_pv_cuts(self):
        """
        Remove all PV cuts from the display and state completely.
        """
        return self.controller.clear_pv_cuts()

    def get_line_roi_points(self, roi):
        """
        Extract the ordered list of coordinates making up a PV cut's path.

        Parameters
        ----------
        roi : pyqtgraph.ROI
            The region of interest.

        Returns
        -------
        list of tuple
            A list of (x, y) coordinates defining the line geometry.
        """
        return self.controller.get_line_roi_points(roi)
    def world_to_pixel(self, x_world, y_world):
        """
        Convert physical/world offsets back to local image pixel coordinates.

        Parameters
        ----------
        x_world : float
            The horizontal offset in arcseconds from the center.
        y_world : float
            The vertical offset in arcseconds from the center.

        Returns
        -------
        tuple
            A 2-tuple of (x_pix, y_pix).
        """
        start_x = (self.nx / 2) * self.pix_scale_arcsec
        start_y = -(self.ny / 2) * self.pix_scale_arcsec
        x_pix = (start_x - x_world) / self.pix_scale_arcsec
        y_pix = (y_world - start_y) / self.pix_scale_arcsec
        return x_pix, y_pix

    def sample_cube_along_line(self, roi, cube_data=None, width=1):
        """
        Extract a 2D position-velocity slice from the 3D data cube along a specified path.

        Parameters
        ----------
        roi : pyqtgraph.ROI
            The line region defining the spatial cut.
        cube_data : numpy.ndarray, optional
            The 3D data cube to sample. Uses the clean cube if not provided.
        width : int, optional
            The width in pixels to average over perpendicular to the cut, by default 1.

        Returns
        -------
        tuple
            A 2-tuple containing the 1D spatial offset array and the 2D (velocity vs offset) slice array.
        """
        points = self.get_line_roi_points(roi)
        if points is None:
            return None, None
        if cube_data is None:
            cube_data = self.cube_clean
        if width < 1 or width % 2 == 0:
            width = 1

        p1, p2 = points
        dx_world = p2[0] - p1[0]
        dy_world = p2[1] - p1[1]
        length_arcsec = np.hypot(dx_world, dy_world)
        n_samples = max(int(np.ceil(length_arcsec / max(self.pix_scale_arcsec, 1e-6))) + 1, 2)

        if width == 1:
            xs = np.linspace(p1[0], p2[0], n_samples)
            ys = np.linspace(p1[1], p2[1], n_samples)
            offsets = np.linspace(0.0, length_arcsec, n_samples)

            x_pix, y_pix = self.world_to_pixel(xs, ys)
            valid = (
                (x_pix >= 0.0)
                & (x_pix <= self.nx - 1)
                & (y_pix >= 0.0)
                & (y_pix <= self.ny - 1)
            )

            nv = cube_data.shape[0]
            samples = np.full((nv, n_samples), np.nan, dtype=np.float64)
            if np.any(valid):
                x0 = np.floor(x_pix[valid]).astype(np.int64)
                y0 = np.floor(y_pix[valid]).astype(np.int64)
                x1 = np.clip(x0 + 1, 0, self.nx - 1).astype(np.int64)
                y1 = np.clip(y0 + 1, 0, self.ny - 1).astype(np.int64)
                fx = (x_pix[valid] - x0).astype(np.float64)
                fy = (y_pix[valid] - y0).astype(np.float64)
                buf = np.ascontiguousarray(cube_data, dtype=np.float64)
                out = np.empty((nv, int(valid.sum())), dtype=np.float64)
                _bilinear_interp(buf, x0, y0, x1, y1, fx, fy, out)
                samples[:, valid] = out

            return offsets, samples.T

        half_w = width // 2
        all_samples = []
        for k in range(-half_w, half_w + 1):
            off_x = k * self.pix_scale_arcsec * dy_world / max(length_arcsec, 1e-6)
            off_y = -k * self.pix_scale_arcsec * dx_world / max(length_arcsec, 1e-6)

            off_p1 = np.array([p1[0] + off_x, p1[1] + off_y], dtype=float)
            off_p2 = np.array([p2[0] + off_x, p2[1] + off_y], dtype=float)

            off_xs = np.linspace(off_p1[0], off_p2[0], n_samples)
            off_ys = np.linspace(off_p1[1], off_p2[1], n_samples)

            x_pix, y_pix = self.world_to_pixel(off_xs, off_ys)
            valid = (
                (x_pix >= 0.0)
                & (x_pix <= self.nx - 1)
                & (y_pix >= 0.0)
                & (y_pix <= self.ny - 1)
            )

            nv = cube_data.shape[0]
            row = np.full((nv, n_samples), np.nan, dtype=np.float64)
            if np.any(valid):
                x0 = np.floor(x_pix[valid]).astype(np.int64)
                y0 = np.floor(y_pix[valid]).astype(np.int64)
                x1 = np.clip(x0 + 1, 0, self.nx - 1).astype(np.int64)
                y1 = np.clip(y0 + 1, 0, self.ny - 1).astype(np.int64)
                fx = (x_pix[valid] - x0).astype(np.float64)
                fy = (y_pix[valid] - y0).astype(np.float64)
                buf = np.ascontiguousarray(cube_data, dtype=np.float64)
                out = np.empty((nv, int(valid.sum())), dtype=np.float64)
                _bilinear_interp(buf, x0, y0, x1, y1, fx, fy, out)
                row[:, valid] = out

            all_samples.append(row)

        if not all_samples:
            offsets = np.linspace(0.0, length_arcsec, n_samples)
            return offsets, np.full((n_samples, cube_data.shape[0]), np.nan, dtype=np.float64)

        offsets = np.linspace(0.0, length_arcsec, n_samples)
        stacked = np.dstack(all_samples)
        with np.errstate(all='ignore'):
            avg = np.nanmean(stacked, axis=2)
        return offsets, avg

    def update_pv_diagram(self, _=None):
        """
        Delegate the refresh of the primary active PV diagram.

        Parameters
        ----------
        _ : optional
            Ignored argument for signal compatibility.
        """
        return self.controller.update_pv_diagram(_)
    def update_spatial_analysis(self, _=None):
        """
        Recompute statistics and update info panels for the currently active spatial ROI.

        Parameters
        ----------
        _ : optional
            Ignored argument for signal compatibility.
        """
        if self.cube_clean is None: return
        data = self.get_current_channel_data()
        if data is None: return

        active_item = None
        if self.spatial_rois_to_delete:
            for item in self.spatial_rois:
                if item["roi"] == self.spatial_rois_to_delete[-1]:
                    active_item = item
                    break
        elif self.spatial_rois:
            active_item = self.spatial_rois[-1]

        if not active_item:
            self.curve_spatial_1.setData([], [])
            self.curve_spatial_2.setData([], [])
            self._clear_all_overlay_spatial_curves()
            self.stacked_spatial_info.setCurrentIndex(0)
            self.spatial_stats_scroll.hide()
            self.lbl_spatial_stats.setText("Draw a region to see statistics.")
            return

        roi = active_item["roi"]
        tool = active_item["tool"]

        if tool == "Point":
            self.plot_spatial_1.show()
            self.plot_spatial_1.setTitle("X Profile")
            self.plot_spatial_2.show()
        elif tool == "Line":
            self.plot_spatial_1.show()
            self.plot_spatial_1.setTitle("Spatial Profile")
            self.plot_spatial_2.hide()
        else:
            self.plot_spatial_1.hide()
            self.plot_spatial_2.hide()

        overlay_repr = self._get_overlay_reprojections_current()

        try:
            if tool == "Point":
                pos = roi.pos()
                size = roi.size()
                cx, cy = pos.x() + size.x()/2, pos.y() + size.y()/2

                start_x = (self.nx / 2) * self.pix_scale_arcsec
                start_y = -(self.ny / 2) * self.pix_scale_arcsec
                x_idx = int((cx - start_x) / (-self.pix_scale_arcsec))
                y_idx = int((cy - start_y) / self.pix_scale_arcsec)

                if 0 <= x_idx < self.nx and 0 <= y_idx < self.ny:
                    x_profile = data[:, y_idx]
                    y_profile = data[x_idx, :]

                    x_axis = (self.nx / 2 - np.arange(self.nx)) * self.pix_scale_arcsec
                    y_axis = (np.arange(self.ny) - self.ny / 2) * self.pix_scale_arcsec

                    self.curve_spatial_1.setData(x_axis, x_profile)
                    self.plot_spatial_1.setLabel('left', f'Flux ({self.display_unit})')
                    self.plot_spatial_1.setLabel('bottom', 'RA offset (arcsec)')

                    self.curve_spatial_2.setData(y_axis, y_profile)
                    self.plot_spatial_2.setLabel('left', f'Flux ({self.display_unit})')
                    self.plot_spatial_2.setLabel('bottom', 'Dec offset (arcsec)')

                    self.stacked_spatial_info.setCurrentIndex(0)
                    self.spatial_stats_scroll.hide()

                    for ov_name, ov_data in overlay_repr.items():
                        ov_x_profile = ov_data[:, y_idx]
                        ov_y_profile = ov_data[x_idx, :]
                        ov_color = self._overlay_color_by_name(ov_name)

                        self._update_overlay_spatial_curve(1, ov_name, x_axis, ov_x_profile, ov_color)
                        self._update_overlay_spatial_curve(2, ov_name, y_axis, ov_y_profile, ov_color)

                    self._cleanup_stale_overlay_spatial_curves(overlay_repr.keys())
                    self._refresh_spatial_legend(1)
                    self._refresh_spatial_legend(2)

            elif tool == "Line":
                offsets, profile_2d = self.sample_cube_along_line(roi, data[np.newaxis, :, :])
                if profile_2d is not None and profile_2d.size > 0:
                    profile = profile_2d[:, 0]
                    self.curve_spatial_1.setData(offsets, profile)
                    self.curve_spatial_2.setData([], [])
                    self.plot_spatial_1.setLabel('left', f'Flux ({self.display_unit})')
                    self.plot_spatial_1.setLabel('bottom', 'Distance (arcsec)')

                    points = self.get_line_roi_points(roi)
                    if points is not None:
                        p1, p2 = points
                        dx_w = p2[0] - p1[0]
                        dy_w = p2[1] - p1[1]
                        length_arcsec = np.hypot(dx_w, dy_w)
                        n_samples = max(int(np.ceil(length_arcsec / max(self.pix_scale_arcsec, 1e-6))) + 1, 2)

                        xs = np.linspace(p1[0], p2[0], n_samples)
                        ys = np.linspace(p1[1], p2[1], n_samples)

                        x_pix, y_pix = self.world_to_pixel(xs, ys)
                        valid = (x_pix >= 0) & (x_pix <= self.nx - 1) & (y_pix >= 0) & (y_pix <= self.ny - 1)

                        for ov_name, ov_data in overlay_repr.items():
                            ov_samples = np.full(n_samples, np.nan, dtype=np.float64)
                            if np.any(valid):
                                x0 = np.floor(x_pix[valid]).astype(np.int64)
                                y0 = np.floor(y_pix[valid]).astype(np.int64)
                                x1 = np.clip(x0 + 1, 0, self.nx - 1).astype(np.int64)
                                y1 = np.clip(y0 + 1, 0, self.ny - 1).astype(np.int64)
                                fx = (x_pix[valid] - x0).astype(np.float64)
                                fy = (y_pix[valid] - y0).astype(np.float64)
                                ov_buf = np.ascontiguousarray(ov_data[np.newaxis, :, :], dtype=np.float64)
                                ov_out = np.empty((1, int(valid.sum())), dtype=np.float64)
                                _bilinear_interp(ov_buf, x0, y0, x1, y1, fx, fy, ov_out)
                                ov_samples[valid] = ov_out[0]

                            ov_color = self._overlay_color_by_name(ov_name)
                            self._update_overlay_spatial_curve(1, ov_name, offsets, ov_samples, ov_color)

                    self.curve_spatial_2.setData([], [])
                    for curve_name in list(getattr(self, 'overlay_spatial_curves_2', {}).keys()):
                        getattr(self, 'overlay_spatial_curves_2')[curve_name].setData([], [])


                    self._cleanup_stale_overlay_spatial_curves_plot1(overlay_repr.keys())
                    self._refresh_spatial_legend(1)
                    self.stacked_spatial_info.setCurrentIndex(0)
                    self.spatial_stats_scroll.hide()
                    self.lbl_spatial_stats.setText("Line profile plotted.")

            elif tool in ["Rectangle", "Ellipse"]:
                if isinstance(roi, (pg.EllipseROI, pg.PolyLineROI)):
                    sub_data = roi.getArrayRegion(data, self.view_channel.getImageItem())
                    dummy_ones = np.ones_like(data)
                    roi_mask = roi.getArrayRegion(dummy_ones, self.view_channel.getImageItem())
                    if sub_data is not None and roi_mask is not None:
                        sub_data[roi_mask == 0] = np.nan
                elif isinstance(roi, pg.RectROI):
                    sub_data = roi.getArrayRegion(data, self.view_channel.getImageItem())
                    roi_mask = None
                elif isinstance(roi, (pg.LineSegmentROI, pg.LineROI, getattr(pg, 'PointROI', type('Dummy', (), {})))):
                    sub_data = roi.getArrayRegion(data, self.view_channel.getImageItem())
                    roi_mask = None
                else:
                    sub_data = roi.getArrayRegion(data, self.view_channel.getImageItem())
                    roi_mask = None

                if sub_data is not None and sub_data.size > 0:
                    valid = sub_data[~np.isnan(sub_data)]
                    if len(valid) > 0:
                        mean_val = np.mean(valid)
                        sum_val = np.sum(valid)
                        max_val = np.max(valid)
                        min_val = np.min(valid)
                        rms_val = np.sqrt(np.mean(valid**2))
                        std_val = np.std(valid)

                        self._clear_spatial_stats_panels()

                        base_panel = self._make_stats_panel("Base Cube", "white", [
                            ("Mean", f"{mean_val:.4g} {self.display_unit}"),
                            ("Sum", f"{sum_val:.4g} {self.display_unit}"),
                            ("Peak", f"{max_val:.4g} {self.display_unit}"),
                            ("Min", f"{min_val:.4g} {self.display_unit}"),
                            ("RMS", f"{rms_val:.4g} {self.display_unit}"),
                            ("Std Dev", f"{std_val:.4g} {self.display_unit}"),
                        ])
                        self.spatial_stats_layout.addWidget(base_panel)

                        for ov_name, ov_data in overlay_repr.items():
                            ov_sub = roi.getArrayRegion(ov_data, self.view_channel.getImageItem())
                            if ov_sub is not None and roi_mask is not None:
                                ov_sub[roi_mask == 0] = np.nan
                            if ov_sub is not None and ov_sub.size > 0:
                                ov_valid = ov_sub[~np.isnan(ov_sub)]
                                if len(ov_valid) > 0:
                                    ov_mean = np.mean(ov_valid)
                                    ov_sum = np.sum(ov_valid)
                                    ov_max = np.max(ov_valid)
                                    ov_min = np.min(ov_valid)
                                    ov_rms = np.sqrt(np.mean(ov_valid**2))
                                    ov_std = np.std(ov_valid)
                                    ov_color = self._overlay_color_by_name(ov_name)
                                    ov_panel = self._make_stats_panel(ov_name, ov_color, [
                                        ("Mean", f"{ov_mean:.4g}"),
                                        ("Sum", f"{ov_sum:.4g}"),
                                        ("Peak", f"{ov_max:.4g}"),
                                        ("Min", f"{ov_min:.4g}"),
                                        ("RMS", f"{ov_rms:.4g}"),
                                        ("Std Dev", f"{ov_std:.4g}"),
                                    ])
                                    self.spatial_stats_layout.addWidget(ov_panel)

                        self.spatial_stats_layout.addStretch()
                        self.stacked_spatial_info.setCurrentIndex(1)
                        self.spatial_stats_scroll.show()
                        self.curve_spatial_1.setData([], [])
                        self.curve_spatial_2.setData([], [])
                        self._clear_all_overlay_spatial_curves()
                    else:
                        self.stacked_spatial_info.setCurrentIndex(0)
                        self.spatial_stats_scroll.hide()
                        self.lbl_spatial_stats.setText("No valid data in region.")

        except Exception as e:
            with open("line_debug.txt", "a") as dbg:
                dbg.write(f"Exception: {e}\n")
            print(f"Error in update_spatial_analysis: {e}")

    def _get_overlay_reprojections_current(self):
        """
        Retrieve or compute the reprojections of all active contour overlays for the current channel.
        
        Returns
        -------
        dict
            A dictionary mapping overlay names to their reprojected 2D data arrays.
        """
        result = {}
        if not self.contour_overlays or self.cube_clean is None:
            return result
        current_ch = self.slider_channel.value()
        for ov in self.contour_overlays:
            if ov.get('_reproj_raw') is not None and ov.get('_reproj_channel') == current_ch:
                result[ov['name']] = ov['_reproj_raw']
                continue
            overlay_slice = self._get_overlay_slice_for_channel(ov)
            if overlay_slice is None:
                continue
            reprojected = self._reproject_overlay_slice(ov, overlay_slice)
            if reprojected is not None:
                result[ov['name']] = reprojected
                ov['_reproj_raw'] = reprojected
                ov['_reproj_channel'] = current_ch
        return result

    def _overlay_color_by_name(self, name):
        """
        Get the configured color for a specific contour overlay by its name.
        """
        for ov in self.contour_overlays:
            if ov['name'] == name:
                return ov['options']['color']
        return 'white'

    def _update_overlay_spatial_curve(self, plot_num, ov_name, x, y, color):
        """Delegate updating a specific spatial overlay contour line to the controller."""
        return self.controller._update_overlay_spatial_curve(plot_num, ov_name, x, y, color)
        
    def _cleanup_stale_overlay_spatial_curves(self, active_names):
        """Delegate removing contour items that are no longer active in the main plot."""
        return self.controller._cleanup_stale_overlay_spatial_curves(active_names)
        
    def _cleanup_stale_overlay_spatial_curves_plot1(self, active_names):
        """Delegate removing contour items that are no longer active in the smoothed plot."""
        return self.controller._cleanup_stale_overlay_spatial_curves_plot1(active_names)
        
    def _clear_all_overlay_spatial_curves(self):
        """Delegate wiping all spatial overlay contours across all plots."""
        return self.controller._clear_all_overlay_spatial_curves()
        
    def _refresh_spatial_legend(self, plot_num):
        """Delegate refreshing the overlay legend in the specified plot view."""
        return self.controller._refresh_spatial_legend(plot_num)
        
    def _make_stats_panel(self, title, color, rows):
        """
        Construct a customized UI widget for displaying statistical info for an ROI.
        
        Parameters
        ----------
        title : str
            The header title of the panel.
        color : str
            The hex color code for the title.
        rows : list of tuple
            A list of (label, value_str) pairs to display.
            
        Returns
        -------
        QWidget
            The constructed panel widget.
        """
        panel = QWidget()
        panel.setStyleSheet("background-color: #2a2a2a; border: 1px solid #555; border-radius: 6px; padding: 0px;")
        panel.setFixedWidth(220)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(3)
        title_lbl = QLabel(title)
        title_lbl.setStyleSheet(f"font-weight: bold; color: {color}; font-size: 13px; border: none; padding: 0px; background: transparent;")
        title_lbl.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_lbl)
        layout.addSpacing(5)
        for label, value in rows:
            row_layout = QHBoxLayout()
            row_layout.setSpacing(4)
            lbl_name = QLabel(label + ":")
            lbl_name.setStyleSheet("color: #999; font-size: 11px; border: none; padding: 0px; background: transparent;")
            lbl_val = QLabel(value)
            lbl_val.setStyleSheet("color: #eee; font-size: 11px; font-weight: bold; border: none; padding: 0px; background: transparent;")
            lbl_val.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            row_layout.addWidget(lbl_name)
            row_layout.addWidget(lbl_val, stretch=1)
            layout.addLayout(row_layout)
        return panel

    def _clear_spatial_stats_panels(self):
        """Delegate clearing all spatial statistics panels from the side bar."""
        return self.controller._clear_spatial_stats_panels()
    def update_wcs_mode(self, is_absolute):
        """
        Delegate the WCS coordinate mode update to the controller.

        Parameters
        ----------
        is_absolute : bool
            True if absolute World Coordinates are preferred, False for relative image/pixel coordinates.
        """
        return self.controller.update_wcs_mode(is_absolute)
    def set_active_panel(self, panel_id):
        """
        Visually highlight the currently active panel by updating its QFrame styling.

        Parameters
        ----------
        panel_id : str
            The identifier of the panel to mark as active (e.g., 'channel', 'spectrum', 'pv1').
        """
        self.last_clicked_panel_id = panel_id
        for pid, frame in self.frames.items():
            if pid == panel_id:
                frame.setStyleSheet("QFrame#PanelFrame { border: 2px solid #3498db; border-radius: 6px; background-color: #1a1a1a; }")
            else:
                frame.setStyleSheet("QFrame#PanelFrame { border: 1px solid #333; border-radius: 6px; background-color: #121212; }")
        
        self.parent_window.update_menu_states()

    def set_active_picker(self, checked, panel_id):
        """
        Toggle the crosshair picker tool for a specified moment/PV panel.

        Parameters
        ----------
        checked : bool
            True if the picker is enabled, False otherwise.
        panel_id : int
            The index of the moment panel interacting with the picker.
        """
        for i, p in enumerate(self.panels):
            if i != panel_id: p['btn_pick'].setChecked(False)
        self.active_picker_panel = panel_id if checked else None
        
        if checked:
            self.plot_channel.vb.setCursor(Qt.CrossCursor)
        else:
            self.plot_channel.vb.setCursor(Qt.ArrowCursor)

    def universal_click_handler(self, event, source_plot):
        """
        Handle mouse clicks across different pyqtgraph plots to synchronize global state 
        (e.g., spectral axis updates, WCS coordinate lookups, picker tools).

        Parameters
        ----------
        event : pyqtgraph.GraphicsScene.mouseEvents.MouseClickEvent
            The raw click event.
        source_plot : pyqtgraph.PlotWidget
            The specific plot widget where the click originated.
        """
        if self.cube_clean is None: return
        
        try:
            mp = source_plot.vb.mapSceneToView(event.scenePos())
        except Exception:
            if hasattr(source_plot, 'plotItem'):
                mp = source_plot.plotItem.vb.mapSceneToView(event.scenePos())
            else:
                return

        if source_plot == self.plot_channel:
            self.set_active_panel('channel')
        elif source_plot == self.plot_widget:
            self.set_active_panel('spectrum')
        else:
            for i, p in enumerate(self.panels):
                if source_plot == p['plot_item']:
                    self.set_active_panel(i)
                    break

        if source_plot == self.plot_channel and getattr(self, 'is_drawing_polygon', False):
            if event.button() == Qt.LeftButton:
                mp = source_plot.vb.mapSceneToView(event.scenePos())
                
                # Check for closing condition: double click OR clicking near first point
                is_closing = False
                if len(self.polygon_points) >= 3:
                    p0 = self.polygon_points[0]
                    dist = np.sqrt((mp.x() - p0[0])**2 + (mp.y() - p0[1])**2)
                    # Tolerance: 3% of view width
                    view_range = source_plot.vb.viewRect()
                    tol = view_range.width() * 0.03
                    if event.double() or dist < tol:
                        is_closing = True
                
                if is_closing:
                    self.finalize_polygon()
                else:
                    self.polygon_points.append([mp.x(), mp.y()])
                    # Update preview
                    pts = np.array(self.polygon_points)
                    self.polygon_preview_line.setData(pts[:,0], pts[:,1], symbol='o', symbolSize=5)
            elif event.button() == Qt.RightButton:
                self.cancel_polygon()
            return

        if source_plot in (self.plot_widget, getattr(self, 'plot_widget_smooth', None)):
            if event.button() == Qt.LeftButton:
                if event.modifiers() == Qt.NoModifier:
                    idx = (np.abs(self.v_axis - mp.x())).argmin()
                    self.slider_channel.setValue(idx)
                elif event.modifiers() == Qt.ControlModifier:
                    hit = False
                    active_rois = self.get_active_spectrum_rois()
                    if active_rois:
                        for item in active_rois:
                            roi = item["roi"]
                            if hasattr(roi, 'getData'):
                                x_data, _ = roi.getData()
                                if x_data is None or len(x_data) < 2:
                                    continue
                                min_x, max_x = min(x_data), max(x_data)
                            else:
                                r_pos = roi.pos()
                                r_size = roi.size()
                                min_x = min(r_pos.x(), r_pos.x() + r_size.x())
                                max_x = max(r_pos.x(), r_pos.x() + r_size.x())
                            if min_x <= mp.x() <= max_x:
                                self.select_region_for_deletion(roi)
                                hit = True
                                break
                    if hit:
                        event.accept()
            return

        if self.active_picker_panel is not None:
            if event.button() == Qt.LeftButton and source_plot == self.plot_channel:
                self.plot_channel.vb.setCursor(Qt.ArrowCursor)
                self.delete_nr_roi()
                
                w = 10.0 * abs(self.pix_scale_arcsec)
                h = 10.0 * abs(self.pix_scale_arcsec)
                
                self.nr_roi = pg.RectROI([mp.x() - w/2, mp.y() - h/2], [w, h], pen=pg.mkPen('#00FFFF', width=2))
                self.nr_roi.addScaleHandle([0, 0], [1, 1]); self.nr_roi.addScaleHandle([1, 1], [0, 0])
                self.nr_roi.addScaleHandle([0, 1], [1, 0]); self.nr_roi.addScaleHandle([1, 0], [0, 1])
                self.nr_roi.addScaleHandle([0.5, 0], [0.5, 1]); self.nr_roi.addScaleHandle([0.5, 1], [0.5, 0])
                self.nr_roi.addScaleHandle([0, 0.5], [1, 0.5]); self.nr_roi.addScaleHandle([1, 0.5], [0, 0.5])
                
                make_roi_rotatable_with_ctrl(self.nr_roi)
                    
                self.nr_label = pg.TextItem("NR", color='#00FFFF', anchor=(0, 1))
                self.nr_label.setParentItem(self.nr_roi)
                
                self.view_channel.getView().addItem(self.nr_roi)
                
                self.nr_roi.target_panel_id = self.active_picker_panel
                self.panels[self.active_picker_panel]['btn_pick'].setChecked(False)
                self.active_picker_panel = None
                
                self.nr_roi.sigRegionChangeFinished.connect(self.update_nr_rms)
                self.update_nr_rms()
            return 
            
        if source_plot == self.plot_channel:
            hit = False
            
            if getattr(self, 'nr_roi', None) is not None:
                r_pos = self.nr_roi.pos()
                r_size = self.nr_roi.size()
                min_x = min(r_pos.x(), r_pos.x() + r_size.x())
                max_x = max(r_pos.x(), r_pos.x() + r_size.x())
                min_y = min(r_pos.y(), r_pos.y() + r_size.y())
                max_y = max(r_pos.y(), r_pos.y() + r_size.y())
                
                if min_x <= mp.x() <= max_x and min_y <= mp.y() <= max_y:
                    self.nr_roi.setPen(pg.mkPen('y', width=3))
                    self.nr_roi_selected = True
                    hit = True
                    event.accept()
                    return
                    
            mode = self.combo_panel_mode.currentText()
            
            # Priority: check Spatial Analysis ROIs if in that mode
            if mode == "Spatial Analysis" and self.spatial_rois:
                for item in self.spatial_rois:
                    roi = item["roi"]
                    is_clicked = False
                    if hasattr(roi, 'shape'):
                         is_clicked = roi.shape().contains(roi.mapFromScene(event.scenePos()))
                    elif isinstance(roi, pg.LineSegmentROI):
                         is_clicked = self.line_roi_hit_test(roi, event.scenePos())
                    elif isinstance(roi, pg.ROI):
                         # For Points (small ROI) and others
                         r_pos = roi.pos()
                         r_size = roi.size()
                         # Buffer for points to make them clickable
                         buf = 0.5 * self.pix_scale_arcsec if r_size.x() < self.pix_scale_arcsec else 0
                         min_x = min(r_pos.x(), r_pos.x() + r_size.x()) - buf
                         max_x = max(r_pos.x(), r_pos.x() + r_size.x()) + buf
                         min_y = min(r_pos.y(), r_pos.y() + r_size.y()) - buf
                         max_y = max(r_pos.y(), r_pos.y() + r_size.y()) + buf
                         is_clicked = (min_x <= mp.x() <= max_x) and (min_y <= mp.y() <= max_y)

                    if is_clicked:
                        hit = True
                        self.select_spatial_region(roi)
                        break

            # Then check Spectrum ROIs
            if not hit and self.spectrum_spatial_rois:
                for r_dict in self.spectrum_spatial_rois:
                    roi = r_dict["roi"]
                    is_clicked = False
                    if hasattr(roi, 'shape'):
                        is_clicked = roi.shape().contains(roi.mapFromScene(event.scenePos()))
                    elif isinstance(roi, pg.LineSegmentROI):
                        is_clicked = self.line_roi_hit_test(roi, event.scenePos())
                    else:
                        r_pos = roi.pos()
                        r_size = roi.size()
                        min_x = min(r_pos.x(), r_pos.x() + r_size.x())
                        max_x = max(r_pos.x(), r_pos.x() + r_size.x())
                        min_y = min(r_pos.y(), r_pos.y() + r_size.y())
                        max_y = max(r_pos.y(), r_pos.y() + r_size.y())
                        is_clicked = (min_x <= mp.x() <= max_x) and (min_y <= mp.y() <= max_y)
                    
                    if is_clicked:
                        hit = True
                        roi.setPen(pg.mkPen('y', width=3))
                        self.active_spatial_spectrum_roi = roi
                        
                        # Sync UI tools with selected ROI type
                        roi_type = r_dict.get("type", "Ellipse")
                        self.combo_roi.blockSignals(True)
                        self.combo_roi.setCurrentText(roi_type)
                        self.combo_roi.blockSignals(False)
                        
                        if roi_type in ["Ellipse", "Rectangle", "Point (Beam)", "Custom Polygon"]:
                            self.btn_edit_region.show()
                        else:
                            self.btn_edit_region.hide()
                    else:
                        roi.setPen(pg.mkPen(r_dict["color"], width=3 if r_dict["checkbox"].isChecked() else 2))
                
                self.roi_selected = hit
                if not hit:
                    self.active_spatial_spectrum_roi = None
                    self.btn_edit_region.hide()


    # ==================== DATA LOADING ====================
    def load_file(self, file_name):
        """
        Delegate the FITS file loading process to the controller.

        Parameters
        ----------
        file_name : str
            The path to the FITS file to load.
        """
        return self.controller.load_file(file_name)
    def set_2d_ui_state(self, is_2d):
        """
        Update the UI state to enable or disable 3D/spectral features based on data dimensionality.

        Parameters
        ----------
        is_2d : bool
            True if the loaded data is a 2D image, False if it is a 3D cube.
        """
        enable = not is_2d
        
        for btn in [self.btn_prev, self.btn_play_rev, self.btn_stop, self.btn_play_fwd, self.btn_next]:
            btn.setEnabled(enable)
            
        self.slider_channel.setEnabled(enable)
        self.input_channel_vel.setEnabled(enable)
        self.btn_grid.setEnabled(enable)
        self.toggle_bottom_btn.setEnabled(enable)
        
        if is_2d and getattr(self, 'bottom_container', None) is not None and self.bottom_container.isVisible():
            if "Hide" in self.toggle_bottom_btn.text():
                self.toggle_bottom_pane()
            
        model = self.combo_panel_mode.model()
        spectrum_index = self.combo_panel_mode.findText("Spectrum")
        if spectrum_index >= 0:
            item = model.item(spectrum_index)
            if item:
                item.setEnabled(enable)
                
        if is_2d:
            self.combo_panel_mode.setCurrentText("Spatial Analysis")

    def close_file(self):
        """
        Reset the application state and clear all data from memory when a file is closed.
        """
        self.set_2d_ui_state(False)
        self.is_2d_image = False
        self.cube_clean = None
        self.fits_header_text = ""
        self.raw_header = None
        self.wcs_2d = None
        self.rest_freq_hz = None
        
        if hasattr(self, 'beam_visualizer_items'):
            for key, items_list in self.beam_visualizer_items.items():
                for item in items_list:
                    try:
                        if item.scene():
                            item.scene().removeItem(item)
                    except Exception: pass
            self.beam_visualizer_items.clear()

        self.is_drawing_polygon = False
        self.polygon_points = []
        if self.polygon_preview_line is not None:
            try:
                self.plot_channel.removeItem(self.polygon_preview_line)
            except Exception:
                pass
            self.polygon_preview_line = None
        self.roi_selected = False

        for item in getattr(self, 'spatial_rois', []):
            di = item.get("direction_item")
            if di is not None:
                try:
                    dscene = di.scene()
                    if dscene is not None: dscene.removeItem(di)
                    else: self.plot_channel.removeItem(di)
                except Exception: pass
                di.setData([], [])
            if "update_spatial_arrow" in item and item.get("roi") is not None:
                try: item["roi"].sigRegionChanged.disconnect(item["update_spatial_arrow"])
                except Exception: pass
            ti = item.get("text_item")
            if ti is not None:
                try:
                    tscene = ti.scene()
                    if tscene is not None: tscene.removeItem(ti)
                    else: self.plot_channel.removeItem(ti)
                except Exception: pass
            if "update_spatial_label" in item and item.get("roi") is not None:
                try: item["roi"].sigRegionChanged.disconnect(item["update_spatial_label"])
                except Exception: pass
            roi = item['roi']
            try: roi.sigRegionChanged.disconnect()
            except Exception: pass
            s = roi.scene()
            if s is not None:
                try: s.removeItem(roi)
                except Exception: pass
        self.spatial_rois = []
        self.spatial_rois_to_delete = []
        self.combo_spatial_regions.blockSignals(True)
        self.combo_spatial_regions.clear()
        self.combo_spatial_regions.addItem("None")
        self.combo_spatial_regions.blockSignals(False)

        for r_dict in list(getattr(self, 'spectrum_spatial_rois', [])):
            roi = r_dict["roi"]
            try:
                if roi.scene():
                    roi.scene().removeItem(roi)
                else:
                    self.view_channel.getView().removeItem(roi)
            except Exception: pass
            cb = r_dict.get("checkbox")
            if cb is not None:
                try: self.box_regions_layout.removeWidget(cb)
                except Exception: pass
                cb.deleteLater()
            name = r_dict.get("name", "")
            if name in getattr(self, 'spectrum_curves', {}):
                c = self.spectrum_curves.pop(name)
                try:
                    if c.scene(): c.scene().removeItem(c)
                    else: self.plot_widget.removeItem(c)
                except Exception: pass
            if name in getattr(self, 'spectrum_curves_smooth', {}):
                c = self.spectrum_curves_smooth.pop(name)
                try:
                    if c.scene(): c.scene().removeItem(c)
                    else: self.plot_widget_smooth.removeItem(c)
                except Exception: pass
        self.spectrum_spatial_rois = []

        active_rois = getattr(self, 'get_active_spectrum_rois', lambda: [])()
        for item in list(active_rois):
            roi = item["roi"]
            try:
                if roi.scene(): roi.scene().removeItem(roi)
                else: self.plot_widget.removeItem(roi)
            except Exception: pass
            ti = item.get("text_item")
            if ti is not None:
                try:
                    if ti.scene(): ti.scene().removeItem(ti)
                    else: self.plot_widget.removeItem(ti)
                except Exception: pass
        if hasattr(self, '_spectrum_regions'):
            self._spectrum_regions = []
        self.rois_to_delete.clear()

        self.combo_regions.blockSignals(True)
        self.combo_regions_2.blockSignals(True)
        self.combo_regions_3.blockSignals(True)
        self.combo_regions.clear()
        self.combo_regions_2.clear()
        self.combo_regions_3.clear()
        self.combo_regions.addItem("None")
        self.combo_regions_2.addItem("None")
        self.combo_regions_3.addItem("None")
        self.combo_regions.blockSignals(False)
        self.combo_regions_2.blockSignals(False)
        self.combo_regions_3.blockSignals(False)

        for name in list(getattr(self, 'spectrum_curves', {}).keys()):
            c = self.spectrum_curves.pop(name)
            try:
                if c.scene(): c.scene().removeItem(c)
                else: self.plot_widget.removeItem(c)
            except Exception: pass
        for name in list(getattr(self, 'spectrum_curves_smooth', {}).keys()):
            c = self.spectrum_curves_smooth.pop(name)
            try:
                if c.scene(): c.scene().removeItem(c)
                else: self.plot_widget_smooth.removeItem(c)
            except Exception: pass

        for cut_info in list(getattr(self, 'pv_cuts', [])):
            roi = cut_info['roi']
            try:
                roi.removeHandle(0)
            except Exception:
                pass
            try:
                roi.removeHandle(1)
            except Exception:
                pass
            try:
                roi.sigRegionChanged.disconnect()
            except Exception:
                pass
            s = roi.scene()
            if s is not None:
                try:
                    s.removeItem(roi)
                except Exception:
                    pass
            text_item = cut_info.get('text_item')
            if text_item is not None:
                try:
                    tscene = text_item.scene()
                    if tscene is not None:
                        tscene.removeItem(text_item)
                    else:
                        self.plot_channel.removeItem(text_item)
                except Exception:
                    pass
            direction_item = cut_info.get('direction_item')
            if direction_item is not None:
                try:
                    dscene = direction_item.scene()
                    if dscene is not None:
                        dscene.removeItem(direction_item)
                    else:
                        self.plot_channel.removeItem(direction_item)
                except Exception:
                    pass
            width_item = cut_info.get('width_item')
            if width_item is not None:
                try:
                    wscene = width_item.scene()
                    if wscene is not None:
                        wscene.removeItem(width_item)
                    else:
                        self.plot_channel.removeItem(width_item)
                except Exception:
                    pass
        self.pv_cuts = []
        self.pv_cuts_to_delete = []
        self.pv_data = None
        self.pv_offset_axis = None
        self.pv_velocity_axis = None
        if hasattr(self, 'pv_view'):
            self.pv_view.clear()
            self.lbl_hover_pv.setText("")
            self.combo_pv_cuts.blockSignals(True)
            self.combo_pv_cuts.clear()
            self.combo_pv_cuts.addItem("None")
            self.combo_pv_cuts.blockSignals(False)

        if hasattr(self, 'curve_spatial_1'):
            self.curve_spatial_1.setData([], [])
            self.curve_spatial_2.setData([], [])
            self.stacked_spatial_info.setCurrentIndex(0)
            self.spatial_stats_scroll.hide()
            self.lbl_spatial_stats.setText("Draw a region to see statistics.")

        self.lbl_region_result.setText("---")

        self.input_channel_vel.setText("")
        self.input_vmin.setText("0.00")
        self.input_vmin.setCursorPosition(0)
        self.input_vmax.setText("1.00")
        self.input_vmax.setCursorPosition(0)
        self.combo_spec_stat.blockSignals(True)
        self.combo_spec_stat.setCurrentText("Mean")
        self.combo_spec_stat.blockSignals(False)
        self.smoothing_params = None
        self.spectrum_curve_smooth.setData([], [])
        if hasattr(self, 'spectrum_tabs'):
            idx = self.spectrum_tabs.indexOf(self.plot_widget_smooth)
            if idx != -1:
                self.spectrum_tabs.removeTab(idx)
            self.spectrum_tabs.tabBar().hide()
            self.spectrum_tabs.setCurrentWidget(self.plot_widget)
        for p in self.panels:
            p['input_thresh'].setText("0.000")

        self.view_channel.getImageItem().clear()
        self.spectrum_curve.setData([], [])
        self.plot_widget.setLabel('left', 'Flux')
        if hasattr(self, 'plot_widget_smooth'):
            self.plot_widget_smooth.setLabel('left', 'Flux')

        self.v_line.hide()
        self.region.hide()
        if hasattr(self, 'smooth_active_line'):
            self.smooth_active_line.hide()
        if hasattr(self, 'smooth_velocity_region'):
            self.smooth_velocity_region.hide()

        for p in self.panels:
            p['combo_pv_cut'].blockSignals(True)
            p['combo_pv_cut'].clear()
            p['combo_pv_cut'].addItem("None")
            p['combo_pv_cut'].blockSignals(False)
            p['combo_pv_range'].setCurrentText("Selected Range")
            p['current_data'] = None
            p['pv_offset_axis'] = None
            p['pv_velocity_axis'] = None
            p['view'].clear()

        self.clear_all_hover_labels()
        self.contour_params = {'channel': None, 0: None, 1: None, 2: None}
        for k in self.active_contours:
            for iso in self.active_contours[k]: iso.setParentItem(None)
            self.active_contours[k] = []
        self.clear_catalog_lines()
        self._clear_overlay_contours()
        for name in list(self.overlay_spectrum_curves.keys()):
            c = self.overlay_spectrum_curves.pop(name)
            try:
                if c.scene(): c.scene().removeItem(c)
                else: self.plot_widget.removeItem(c)
            except Exception: pass
        for name in list(self.overlay_spectrum_curves_smooth.keys()):
            c = self.overlay_spectrum_curves_smooth.pop(name)
            try:
                if c.scene(): c.scene().removeItem(c)
                else: self.plot_widget_smooth.removeItem(c)
            except Exception: pass
        self.contour_overlays = []
        self._clear_all_overlay_spatial_curves()
        if hasattr(self.plot_widget, 'plotItem') and self.plot_widget.plotItem.legend is not None:
            self.plot_widget.plotItem.legend.clear()
        if hasattr(self, 'plot_widget_smooth') and hasattr(self.plot_widget_smooth, 'plotItem') and self.plot_widget_smooth.plotItem.legend is not None:
            self.plot_widget_smooth.plotItem.legend.clear()
        self.btn_contour_overlay.setEnabled(False)
        self.btn_contour_overlay.hide()
        self.parent_window.update_menu_states()

    def clear_roi(self):
        """
        Delegate the removal of the current generic Region of Interest to the controller.
        """
        return self.controller.clear_roi()
    def open_line_catalog(self):
        """
        Open the Splatalogue line catalog search dialog for the current spectral window.
        """
        if self.cube_clean is None or self.rest_freq_hz is None:
            QMessageBox.warning(self, "Missing Data", "Please load a cube containing RESTFRQ in its header first.")
            return

        min_v, max_v = self.region.getRegion()
        c_kms = const.c.to(u.km / u.s).value
        f_ref_ghz = self.rest_freq_hz / 1e9
        
        f1 = f_ref_ghz * (1.0 - min_v / c_kms)
        f2 = f_ref_ghz * (1.0 - max_v / c_kms)
        
        v_pad = 50.0 
        f_pad = (v_pad / c_kms) * f_ref_ghz
        
        obs_fmin = min(f1, f2) - f_pad
        obs_fmax = max(f1, f2) + f_pad

        dlg = LineCatalogDialog(self, obs_fmin, obs_fmax)
        if dlg.exec_():
            if dlg.action == 'clear':
                self.clear_catalog_lines()
            elif dlg.action == 'query':
                self.clear_catalog_lines()
                self.parent_window.statusBar().showMessage("Querying Splatalogue database... please wait.")
                
                v_sys = dlg.spin_vsys.value()
                e_max = dlg.spin_eup.value()
                species = dlg.edit_species.text()
                
                db_choice = dlg.combo_db.currentText()
                if db_choice == "CDMS": catalogs = ['CDMS']
                elif db_choice == "JPL": catalogs = ['JPL']
                else: catalogs = ['CDMS', 'JPL']

                rest_fmin = obs_fmin / (1.0 - (v_sys / c_kms))
                rest_fmax = obs_fmax / (1.0 - (v_sys / c_kms))

                self.worker = SplatalogueWorker(rest_fmin, rest_fmax, catalogs, v_sys, e_max, species)
                self.worker.finished.connect(lambda t: self.process_splatalogue_results(t, v_sys))
                self.worker.error.connect(self.show_splatalogue_error)
                self.worker.start()

    def process_splatalogue_results(self, parsed_data, v_sys):
        """
        Handle the incoming queried line list from the Splatalogue worker thread.

        Parameters
        ----------
        parsed_data : list of dict
            The molecular line data returned from astroquery.
        v_sys : float
            The systemic velocity offset applied during the query.
        """
        self.parent_window.statusBar().showMessage("Ready.")
        if parsed_data is None or len(parsed_data) == 0:
            QMessageBox.information(self, "No Results", "No molecular lines found matching those parameters.")
            return

        dlg = LineSelectionDialog(self, parsed_data, v_sys)
        if dlg.exec_():
            if dlg.selected_rows:
                self.draw_selected_lines(dlg.selected_rows, v_sys)

    def draw_selected_lines(self, selected_rows, v_sys):
        """
        Delegate drawing the vertical markers for catalog lines onto the spectrum.

        Parameters
        ----------
        selected_rows : list of dict
            The selected transitions to plot.
        v_sys : float
            The systemic velocity of the source for Doppler shifting.
        """
        return self.controller.draw_selected_lines(selected_rows, v_sys)

    def show_splatalogue_error(self, err_msg):
        """
        Display an error dialogue if the Splatalogue query fails or times out.

        Parameters
        ----------
        err_msg : str
            The failure message from the worker thread.
        """
        self.parent_window.statusBar().showMessage("Query failed.")
        QMessageBox.critical(self, "Splatalogue Error", err_msg)

    def clear_catalog_lines(self):
        """
        Remove all molecular line catalog markers from the 1D spectrum plot.
        """
        for item in self.catalog_overlay_items:
            self.plot_widget.removeItem(item)
        self.catalog_overlay_items = []

    def load_overlay_file(self, file_name):
        """
        Load a secondary FITS file to be overlaid as contours on the primary data.

        Parameters
        ----------
        file_name : str
            The path to the secondary FITS file.
        """
        try:
            overlay_cube = None
            overlay_header = None
            overlay_v = None
            is_static = False
            static_2d = None
            is_2d_image = False
            sc = None

            try:
                sc = SpectralCube.read(file_name).with_spectral_unit(u.km / u.s, velocity_convention='radio')
                overlay_cube = sc.filled_data[:].value
                overlay_cube = np.transpose(overlay_cube, (0, 2, 1))
                overlay_header = sc.header
            except (ValueError, Exception):
                hdul = fits.open(file_name)
                data_2d = hdul[0].data
                if data_2d is None or data_2d.ndim < 2:
                    hdul.close()
                    QMessageBox.critical(self, "Format Error",
                        "The selected FITS file does not contain 2D or 3D image data.")
                    return False
                if data_2d.ndim == 2:
                    is_2d_image = True
                    overlay_cube = np.transpose(data_2d, (1, 0))[np.newaxis, :, :]
                    overlay_header = hdul[0].header
                    is_static = True
                    static_2d = overlay_cube[0]
                elif data_2d.ndim == 3:
                    overlay_cube = np.transpose(data_2d, (0, 2, 1))
                    overlay_header = hdul[0].header
                    nv_o = overlay_cube.shape[0]
                    if nv_o == 1:
                        is_static = True
                        static_2d = overlay_cube[0]
                    else:
                        is_static = True
                        static_2d = overlay_cube[0]
                        QMessageBox.information(self, "Spectral Axis Missing",
                            "This FITS file has multiple channels but no spectral axis "
                            "information. Using the first channel as a static 2D overlay.")
                else:
                    hdul.close()
                    QMessageBox.critical(self, "Format Error",
                        "Unexpected data dimensions. Expected 2D or 3D.")
                    return False
                hdul.close()

            try:
                overlay_wcs = WCS(overlay_header).celestial
            except Exception:
                QMessageBox.critical(self, "WCS Error",
                    "The selected FITS file does not contain valid spatial WCS information.")
                return False

            primary_wcs = self.wcs_2d
            if primary_wcs is None:
                QMessageBox.critical(self, "WCS Error",
                    "The primary cube has no valid WCS. Please reload it.")
                return False

            nv_o = overlay_cube.shape[0]
            ny_o = overlay_cube.shape[1]
            nx_o = overlay_cube.shape[2]

            if nv_o == 1:
                is_static = True
                static_2d = overlay_cube[0]
            elif not is_2d_image and not is_static:
                try:
                    overlay_v = sc.spectral_axis.value
                except Exception:
                    pass
                if overlay_v is None and nv_o > 1:
                    is_static = True
                    static_2d = overlay_cube[0]

            cdelt2_o = overlay_header.get('CDELT2', None)
            cdelt1_o = overlay_header.get('CDELT1', None)
            pix_scale_o = abs(float(cdelt2_o)) * 3600.0 if cdelt2_o else 1.0
            
            bmaj_array_o = None
            bmin_array_o = None
            freq_array_o = None
            try:
                if overlay_header.get('CTYPE3', '').startswith('FREQ'):
                    freq_crval = overlay_header.get('CRVAL3')
                    freq_cdelt = overlay_header.get('CDELT3')
                    freq_crpix = overlay_header.get('CRPIX3')
                    freq_array_o = freq_crval + (np.arange(nv_o) - (freq_crpix - 1)) * freq_cdelt
                else:
                    rf = overlay_header.get('RESTFRQ', overlay_header.get('RESTFREQ'))
                    if rf and overlay_v is not None:
                        freq_array_o = rf * (1.0 - (overlay_v * 1000.0) / const.c.value)

                with fits.open(file_name) as hdul:
                    is_mb = overlay_header.get('CASAMBM', 'F') == 'T' or 'BEAMS' in hdul
                    if is_mb and 'BEAMS' in hdul:
                        b_data = hdul['BEAMS'].data
                        bmaj_raw = b_data['BMAJ']
                        bmin_raw = b_data['BMIN']
                        if len(bmaj_raw) == nv_o:
                            bmaj_unit = hdul['BEAMS'].columns['BMAJ'].unit
                            if bmaj_unit and 'deg' in str(bmaj_unit).lower():
                                bmaj_array_o = bmaj_raw
                                bmin_array_o = bmin_raw
                            else:
                                bmaj_array_o = bmaj_raw / 3600.0
                                bmin_array_o = bmin_raw / 3600.0
                        else:
                            bmaj = overlay_header.get('BMAJ')
                            bmin = overlay_header.get('BMIN')
                            if bmaj and bmin:
                                bmaj_array_o = np.full(nv_o, bmaj)
                                bmin_array_o = np.full(nv_o, bmin)
                    else:
                        bmaj = overlay_header.get('BMAJ')
                        bmin = overlay_header.get('BMIN')
                        if bmaj and bmin:
                            bmaj_array_o = np.full(nv_o, bmaj)
                            bmin_array_o = np.full(nv_o, bmin)
            except Exception:
                bmaj = overlay_header.get('BMAJ')
                bmin = overlay_header.get('BMIN')
                if bmaj and bmin:
                    bmaj_array_o = np.full(nv_o, bmaj)
                    bmin_array_o = np.full(nv_o, bmin)

            corners_px = np.array([[0, 0], [self.nx - 1, 0], [0, self.ny - 1], [self.nx - 1, self.ny - 1]], dtype=float)
            ra_primary, dec_primary = primary_wcs.wcs_pix2world(corners_px, 0).T
            ra_min_p, ra_max_p = ra_primary.min(), ra_primary.max()
            dec_min_p, dec_max_p = dec_primary.min(), dec_primary.max()

            corners_o_px = np.array([[0, 0], [nx_o - 1, 0], [0, ny_o - 1], [nx_o - 1, ny_o - 1]], dtype=float)
            ra_overlay, dec_overlay = overlay_wcs.wcs_pix2world(corners_o_px, 0).T
            ra_min_o, ra_max_o = ra_overlay.min(), ra_overlay.max()
            dec_min_o, dec_max_o = dec_overlay.min(), dec_overlay.max()

            ra_overlap = (ra_min_p <= ra_max_o and ra_max_p >= ra_min_o)
            dec_overlap = (dec_min_p <= dec_max_o and dec_max_p >= dec_min_o)
            if not ra_overlap or not dec_overlap:
                QMessageBox.critical(self, "No Overlap",
                    "No spatial overlap between the two cubes.\n"
                    "The contour file covers a different region of the sky.")
                return False

            if not is_static and overlay_v is not None:
                v_min_p, v_max_p = np.nanmin(self.v_axis), np.nanmax(self.v_axis)
                v_min_o, v_max_o = np.nanmin(overlay_v), np.nanmax(overlay_v)
                if v_max_p < v_min_o or v_max_o < v_min_p:
                    QMessageBox.critical(self, "No Overlap",
                        "No spectral overlap between the two cubes.\n"
                        "The contour cube covers a different velocity range.")
                    return False

            color_idx = len(self.contour_overlays) % len(self.overlay_color_palette)
            assigned_color = self.overlay_color_palette[color_idx]

            default_opts = {
                'mode': 'percent',
                'rms': 0.001,
                'multipliers_str': '3, 5, 10, 20, 40',
                'lin_min': 0.0,
                'lin_max': 10.0,
                'n_levels': 5,
                'log_min': 0.001,
                'log_max': 10.0,
                'log_base': 10.0,
                'percentages_str': '10, 30, 50, 70, 90',
                'color': assigned_color,
                'line_width': 1.5,
                'line_style': 'solid',
                'smooth': False,
                'smooth_kernel': 3,
            }
            overlay_name = file_name.split('/')[-1]
            if len(overlay_name) > 20:
                overlay_name = overlay_name[:17] + '...'

            overlay_dict = {
                'file': file_name,
                'name': overlay_name,
                'cube': overlay_cube,
                'wcs': overlay_wcs,
                'v_axis': overlay_v,
                'is_static': is_static,
                '2d': static_2d,
                'nx': nx_o,
                'ny': ny_o,
                'pix_scale': pix_scale_o,
                'cdelt1': cdelt1_o,
                'cdelt2': cdelt2_o,
                'bmaj_array': bmaj_array_o,
                'bmin_array': bmin_array_o,
                'freq_array': freq_array_o,
                'display_unit': overlay_header.get('BUNIT', 'Unknown').strip(),
                'iso_items': [],
                'reproject_cache': None,
                'options': default_opts.copy(),
                'color': assigned_color,
            }

            self._update_contour_options_rms_for(overlay_dict)
            self.contour_overlays.append(overlay_dict)

            self.btn_contour_overlay.setEnabled(True)
            self.btn_contour_overlay.show()
            self.update_channel_map()
            self.update_spectrum()

            v_info = "static" if is_static else f"{nv_o} channels"
            self.parent_window.statusBar().showMessage(
                f"Overlay {len(self.contour_overlays)} loaded: {overlay_name} ({v_info}, {assigned_color})")
            return True

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load overlay file:\n{str(e)}")
            return False

    def _make_default_overlay_options(self, color):
        """
        Generate the default styling options dictionary for a new contour overlay.
        """
        return {
            'mode': 'percent', 'rms': 0.001,
            'multipliers_str': '3, 5, 10, 20, 40',
            'lin_min': 0.0, 'lin_max': 10.0, 'n_levels': 5,
            'log_min': 0.001, 'log_max': 10.0, 'log_base': 10.0,
            'percentages_str': '10, 30, 50, 70, 90',
            'color': color, 'line_width': 1.5, 'line_style': 'solid',
            'smooth': False, 'smooth_kernel': 3,
        }

    def _update_contour_options_rms_for(self, overlay_dict):
        """
        Recalculate and update the base RMS noise value for a specific contour overlay.
        """
        return self.controller._update_contour_options_rms_for(overlay_dict)

    def _get_overlay_slice_for_channel(self, overlay_dict):
        """
        Extract the 2D image slice from a loaded secondary contour overlay cube 
        that most closely matches the velocity of the primary cube's current channel.

        Parameters
        ----------
        overlay_dict : dict
            The state dictionary for the specific overlay layer.

        Returns
        -------
        numpy.ndarray or None
            The matched 2D array, or None if out of bounds/invalid.
        """
        if overlay_dict is None:
            return None
        if overlay_dict['is_static']:
            return overlay_dict['2d']

        idx = self.slider_channel.value()
        if self.v_axis is None or idx >= len(self.v_axis):
            return None
        target_v = self.v_axis[idx]

        overlay_v = overlay_dict['v_axis']
        if overlay_v is None:
            return None

        if target_v < np.nanmin(overlay_v) or target_v > np.nanmax(overlay_v):
            return None

        nearest_idx = int(np.argmin(np.abs(overlay_v - target_v)))
        if nearest_idx < 0 or nearest_idx >= overlay_dict['cube'].shape[0]:
            return None
        return overlay_dict['cube'][nearest_idx]

    def _get_overlay_slice_for_current_channel(self):
        """
        Deprecated/legacy wrapper. Use `_get_overlay_slice_for_channel` directly.
        """
        if self.contour_overlay_cube is None:
            return None
        return self._get_overlay_slice_for_channel({
            'cube': self.contour_overlay_cube,
            'is_static': self.contour_overlay_is_static,
            '2d': self.contour_overlay_2d,
            'v_axis': self.contour_overlay_v_axis,
        })

    def _reproject_overlay_slice(self, overlay_dict, overlay_slice):
        """
        Fast-reproject a 2D secondary image slice onto the primary cube's WCS grid using
        a precomputed pixel coordinate lookup table.

        Parameters
        ----------
        overlay_dict : dict
            State containing the target WCS and caching arrays.
        overlay_slice : numpy.ndarray
            The original image slice to be interpolated.

        Returns
        -------
        numpy.ndarray or None
            The reprojected image array matching the primary spatial dimensions.
        """
        primary_wcs = self.wcs_2d
        overlay_wcs = overlay_dict['wcs']
        if primary_wcs is None or overlay_wcs is None or overlay_slice is None:
            return None

        cache = overlay_dict.get('reproject_cache')
        if cache is None:
            px = np.arange(self.nx, dtype=float)
            py = np.arange(self.ny, dtype=float)
            PX, PY = np.meshgrid(px, py, indexing='ij')
            pts_pix = np.column_stack([PX.ravel(), PY.ravel()])

            ra_arr, dec_arr = primary_wcs.wcs_pix2world(pts_pix, 0).T
            ox, oy = overlay_wcs.wcs_world2pix(np.column_stack([ra_arr, dec_arr]), 0).T

            ox = ox.reshape(self.nx, self.ny)
            oy = oy.reshape(self.nx, self.ny)

            valid = (ox >= 0) & (ox < overlay_dict['nx'] - 1) & (oy >= 0) & (oy < overlay_dict['ny'] - 1)
            if not np.any(valid):
                result = np.full((self.nx, self.ny), np.nan, dtype=np.float64)
                cache = {'valid': valid, 'x0': None, 'y0': None, 'x1': None, 'y1': None, 'fx': None, 'fy': None}
                overlay_dict['reproject_cache'] = cache
                return result

            vx = ox[valid]
            vy = oy[valid]
            x0 = np.floor(vx).astype(np.int64)
            y0 = np.floor(vy).astype(np.int64)
            x1 = np.clip(x0 + 1, 0, overlay_dict['nx'] - 1).astype(np.int64)
            y1 = np.clip(y0 + 1, 0, overlay_dict['ny'] - 1).astype(np.int64)
            fx = (vx - x0.astype(float)).astype(np.float64)
            fy = (vy - y0.astype(float)).astype(np.float64)

            cache = {'valid': valid, 'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1, 'fx': fx, 'fy': fy}
            overlay_dict['reproject_cache'] = cache
        elif cache.get('x0') is None:
            return np.full((self.nx, self.ny), np.nan, dtype=np.float64)

        result = np.full((self.nx, self.ny), np.nan, dtype=np.float64)
        valid = cache['valid']
        x0 = cache['x0']
        y0 = cache['y0']
        x1 = cache['x1']
        y1 = cache['y1']
        fx = cache['fx']
        fy = cache['fy']

        vals = ((1.0 - fx) * (1.0 - fy) * overlay_slice[x0, y0]
                + fx * (1.0 - fy) * overlay_slice[x1, y0]
                + (1.0 - fx) * fy * overlay_slice[x0, y1]
                + fx * fy * overlay_slice[x1, y1])
        result[valid] = vals
        return result

    def _compute_contour_levels(self, data, options, target_id=None):
        """
        Calculate specific contour threshold values based on the user's styling mode 
        (RMS multipliers, absolute values, percentages).

        Parameters
        ----------
        data : numpy.ndarray
            The 2D map used for normalization in certain modes.
        options : dict
            The contour parameter dictionary (mode, levels, rms).
        target_id : str or int, optional
            A context string for specialized behavior (unused locally).

        Returns
        -------
        list of float
            The sorted numerical values to draw contours at.
        """
        if data is None:
            return []
        valid = data[np.isfinite(data)]
        if len(valid) == 0:
            return []
            
        mode = options.get('mode', 'rms')
        lo = float(np.nanmin(valid))
        hi = float(np.nanmax(valid))
        
        is_vel = False
        if target_id is not None and target_id != 'channel':
            try:
                mtype = self.panels[target_id]['combo'].currentText()
                if "Moment 1" in mtype or "Moment 9" in mtype:
                    is_vel = True
            except:
                pass

        if mode == 'rms':
            rms = options.get('rms', 0.001)
            mean_val = float(np.nanmean(valid))
            try:
                mults = [float(x.strip()) for x in options.get('multipliers_str', '3,5,10,20,40').split(',') if x.strip()]
            except ValueError:
                mults = [3, 5, 10, 20]
            levels = [mean_val + m * rms for m in mults]
        elif mode == 'linear':
            u_lo = options.get('lin_min', lo)
            u_hi = options.get('lin_max', hi)
            n = max(int(options.get('n_levels', 5)), 1)
            levels = np.linspace(u_lo, u_hi, n).tolist()
        elif mode == 'log':
            pos_data = valid[valid > 0]
            if len(pos_data) == 0 or is_vel:
                n = max(int(options.get('n_levels', 5)), 1)
                levels = np.linspace(lo, hi, n).tolist()
            else:
                min_pos = np.nanmin(pos_data)
                n = max(int(options.get('n_levels', 5)), 1)
                base = options.get('log_base', 10.0)
                log_lo = np.log(min_pos) / np.log(base)
                log_hi = np.log(hi) / np.log(base)
                levels = np.logspace(log_lo, log_hi, n, base=base).tolist()
        elif mode == 'percent':
            try:
                pcts = [float(x.strip()) for x in options.get('percentages_str', '10,30,50,70,90').split(',') if x.strip()]
            except ValueError:
                pcts = [10, 30, 50, 70, 90]
                
            if is_vel:
                levels = [lo + (hi - lo) * (p / 100.0) for p in pcts]
            else:
                levels = [max(0, hi) * (p / 100.0) for p in pcts]
        else:
            levels = []

        return sorted([float(lv) for lv in levels if np.isfinite(lv)])

    def _clear_overlay_contours(self, overlay_dict=None):
        """
        Delegate removing contour graphics items for a given (or all) overlay.
        """
        return self.controller._clear_overlay_contours(overlay_dict)

    def draw_overlay_contours(self):
        """
        Delegate the rendering and updating of all active contour overlays on the current channel.
        """
        return self.controller.draw_overlay_contours()

    def close_overlay(self, index=None):
        """
        Close and discard a specific contour overlay, freeing memory and removing it from the view.

        Parameters
        ----------
        index : int, optional
            The list index of the overlay to remove, or None to clear all.
        """
        removed_name = None
        if index is not None:
            if 0 <= index < len(self.contour_overlays):
                ov = self.contour_overlays.pop(index)
                removed_name = ov['name']
                self._clear_overlay_contours(ov)
        else:
            self._clear_overlay_contours()
            self.contour_overlays = []
        if removed_name is not None:
            suffix = " (overlay)"
            to_remove = [n for n in self.overlay_spectrum_curves if n == f"{removed_name}{suffix}"]
            for name in to_remove:
                c = self.overlay_spectrum_curves.pop(name)
                try:
                    if c.scene():
                        c.scene().removeItem(c)
                    else:
                        self.plot_widget.removeItem(c)
                except Exception:
                    pass
            to_remove_s = [n for n in self.overlay_spectrum_curves_smooth if n == f"{removed_name}{suffix}"]
            for name in to_remove_s:
                c = self.overlay_spectrum_curves_smooth.pop(name)
                try:
                    if c.scene():
                        c.scene().removeItem(c)
                    else:
                        self.plot_widget_smooth.removeItem(c)
                except Exception:
                    pass
        else:
            for name in list(self.overlay_spectrum_curves.keys()):
                c = self.overlay_spectrum_curves.pop(name)
                try:
                    if c.scene():
                        c.scene().removeItem(c)
                    else: self.plot_widget.removeItem(c)
                except Exception: pass
            for name in list(self.overlay_spectrum_curves_smooth.keys()):
                c = self.overlay_spectrum_curves_smooth.pop(name)
                try:
                    if c.scene():
                        c.scene().removeItem(c)
                    else: self.plot_widget_smooth.removeItem(c)
                except Exception: pass
        if not self.contour_overlays:
            self.btn_contour_overlay.setEnabled(False)
            self.btn_contour_overlay.hide()
        self.parent_window.statusBar().showMessage("Contour overlay removed.")
        self.update_channel_map()
        self.update_spectrum()

    def open_contour_options(self):
        """
        Open the configuration dialog for adjusting contour properties (levels, styling).
        """
        if not self.contour_overlays:
            QMessageBox.warning(self, "No Overlay", "Please load a contour overlay file first.")
            return

        if hasattr(self, '_contour_options_dlg') and self._contour_options_dlg is not None and self._contour_options_dlg.isVisible():
            self._contour_options_dlg.raise_()
            self._contour_options_dlg.activateWindow()
            return

        dlg = ContourOptionsDialog(self, self.contour_overlays)
        self._contour_options_dlg = dlg
        dlg.setAttribute(Qt.WA_DeleteOnClose, True)
        dlg.destroyed.connect(lambda: setattr(self, '_contour_options_dlg', None))
        dlg.show()

    # ==================== MAP & ROI FUNCTIONS ====================
    def draw_contours(self, target_id, view, data):
        """
        Delegate the generation and drawing of pyqtgraph contour lines to the controller.

        Parameters
        ----------
        target_id : str or int
            Identifier for the panel receiving the contours (e.g., 'channel').
        view : pyqtgraph.ViewBox
            The view box to draw into.
        data : numpy.ndarray
            The 2D data array to contour.
        """
        return self.controller.draw_contours(target_id, view, data)
    def update_channel_map(self):
        """
        Delegate the refreshing of the main channel map view to the controller.
        """
        return self.controller.update_channel_map()
    def set_channel_from_text(self):
        """
        Update the current spectral channel slice based on a manually inputted velocity value.
        """
        if self.cube_clean is None: return
        try:
            target_v = float(self.input_channel_vel.text())
            idx = (np.abs(self.v_axis - target_v)).argmin()
            self.slider_channel.setValue(idx)
        except ValueError: self.update_channel_map()

    def get_current_channel_data(self):
        """
        Get the 2D image data for the currently selected spectral channel.

        Returns
        -------
        numpy.ndarray or None
            The 2D slice from the clean cube, or None if no cube is loaded.
        """
        return None if self.cube_clean is None else self.cube_clean[self.slider_channel.value()]
    
    def start_playback(self, d): 
        """
        Begin automatic animation through the spectral cube channels.
        
        Parameters
        ----------
        d : int
            The step direction (+1 for forward, -1 for reverse).
        """
        self.play_direction = d
        self.playback_timer.start(150)
        
    def stop_playback(self): 
        """Stop the channel playback animation."""
        self.playback_timer.stop()
        
    def step_channel(self, d=None):
        """
        Advance or retreat the current channel by a single step.
        """
        if self.cube_clean is None: return
        shift = d if type(d) is int else self.play_direction
        nv = self.slider_channel.value() + shift
        if 0 <= nv < len(self.v_axis): 
            self.slider_channel.setValue(nv)
        else: 
            self.stop_playback()

    def change_roi(self, roi_type, cx=None, cy=None):
        """
        Delegate the modification of a specific ROI type's geometry to the controller.

        Parameters
        ----------
        roi_type : str
            The type/identifier of the ROI.
        cx : float, optional
            The new X center coordinate.
        cy : float, optional
            The new Y center coordinate.
        """
        return self.controller.change_roi(roi_type, cx, cy)
    def finalize_polygon(self):
        """
        Complete the drawing of a custom polygon ROI, close the shape, and register it.
        """
        if len(self.polygon_points) < 3:
            self.cancel_polygon()
            return
            
        pts = self.polygon_points
        self.cancel_polygon()
        
        new_roi = pg.PolyLineROI(pts, closed=True, pen='#f1c40f')
        def custom_shape(roi=new_roi):
            """
            Generate a custom QPainterPath for hit testing the closed polygon.

            Parameters
            ----------
            roi : pyqtgraph.PolyLineROI, optional
                The polygon region object.

            Returns
            -------
            QPainterPath
                The painter path defining the interior bounds.
            """
            p = QPainterPath()
            if not roi.handles: return p
            p.moveTo(roi.handles[0]['item'].pos())
            for h in roi.handles[1:]:
                p.lineTo(h['item'].pos())
            p.closeSubpath()
            return p
        new_roi.shape = custom_shape
        
        self._finish_roi_addition(new_roi, "Custom Polygon")
        self.update_spectrum()

    def cancel_polygon(self):
        """
        Abort the current polygon drawing operation and clear temporary preview lines.
        """
        self.is_drawing_polygon = False
        self.polygon_points = []
        if self.polygon_preview_line:
            self.plot_channel.vb.removeItem(self.polygon_preview_line)
            self.polygon_preview_line = None

    def _finish_roi_addition(self, new_roi, roi_type):
        """Delegate the final registration and callback attachment for a new spatial ROI."""
        return self.controller._finish_roi_addition(new_roi, roi_type)

    def remove_spatial_spectrum_roi(self, roi):
        """Delegate the deletion and memory cleanup of a spatial spectrum ROI."""
        return self.controller.remove_spatial_spectrum_roi(roi)

    def open_edit_region_dialog(self):
        """
        Launch a dialog to precisely edit the numerical parameters (center, size, angle) 
        of the currently active spatial region.
        """
        roi = getattr(self, 'active_spatial_spectrum_roi', None)
        roi_dict = None
        
        if roi:
            roi_dict = next((r for r in getattr(self, 'spectrum_spatial_rois', []) if r["roi"] == roi), None)
        else:
            if getattr(self, 'spatial_rois_to_delete', []):
                roi = self.spatial_rois_to_delete[-1]
                roi_dict = next((r for r in getattr(self, 'spatial_rois', []) if r["roi"] == roi), None)
                
        if not roi: return
        
        # Mutual exclusion: warn if Smooth dialog is currently open
        if getattr(self, '_smooth_dialog_active', False):
            msg = QMessageBox(self.window())
            msg.setWindowTitle("Dialog Open")
            msg.setText("Please close the 'Smooth' dialog before opening Edit Region.")
            msg.setIcon(QMessageBox.Information)
            msg.setWindowFlags(msg.windowFlags() | Qt.WindowStaysOnTopHint)
            msg.exec_()
            return
        from src.gui.dialogs import RegionPropertiesDialog
        # If an edit dialog is already open for this ROI, just raise it
        if getattr(self, '_region_dialog', None) and self._region_dialog.isVisible():
            self._region_dialog.raise_()
            self._region_dialog.activateWindow()
            return
        dlg = RegionPropertiesDialog(roi, self, parent=self.window(), roi_dict=roi_dict)
        self._region_dialog = dlg
        dlg.show()

    def _update_spectrum_state_machine(self):
        """
        Delegate spectrum UI state transitions to the controller.
        """
        return self.controller._update_spectrum_state_machine()
        
    def update_spectral_axis(self):
        """
        Delegate the configuration of the spectral plot axis (frequency vs velocity) to the controller.
        """
        return self.controller.update_spectral_axis()
        
    def update_spectrum(self):
        """
        Delegate the extraction and plotting of 1D spectra to the controller.
        """
        return self.controller.update_spectrum()
    def _cleanup_removed_overlay_curves(self):
        """
        Delegate the cleanup of stale or deleted spectral overlay curves from the plot.
        """
        return self.controller._cleanup_removed_overlay_curves()
        
    def _clear_all_overlay_spectrum_curves(self):
        """
        Delegate the removal of all catalog/line overlay curves from the spectral plot.
        """
        return self.controller._clear_all_overlay_spectrum_curves()
        
    def update_text_from_region(self):
        """Delegate the synchronization of spatial coordinates in the UI text boxes after an ROI is dragged."""
        return self.controller.update_text_from_region()
        
    def update_region_from_text(self):
        """
        Delegate the manual updating of the spectral region bounds from the UI text fields.
        """
        return self.controller.update_region_from_text()
    def apply_cmap(self, view, is_velocity):
        """
        Delegate the application of a colormap to a specific view.

        Parameters
        ----------
        view : pyqtgraph.ImageView
            The image view to update.
        is_velocity : bool
            True if applying a diverging colormap suitable for velocity fields.
        """
        return self.controller.apply_cmap(view, is_velocity)
    def _sync_smooth_region_from_main(self):
        """
        Synchronize the bounds of the smoothed spectrum region to match the main raw spectrum region.
        """
        if not hasattr(self, 'smooth_velocity_region'): return
        self.smooth_velocity_region.blockSignals(True)
        self.smooth_velocity_region.setRegion(self.region.getRegion())
        self.smooth_velocity_region.blockSignals(False)

    def _sync_main_region_from_smooth(self):
        """
        Synchronize the bounds of the main raw spectrum region to match the smoothed spectrum region.
        """
        if not hasattr(self, 'region'): return
        self.region.blockSignals(True)
        self.region.setRegion(self.smooth_velocity_region.getRegion())
        self.region.blockSignals(False)

    def _on_region_drag_start(self):
        """
        Callback triggered when the user starts dragging the spectral region boundaries.
        Marks that a drag is in progress to prevent excessive recalculations.
        """
        self._region_dragging = True

    def _on_region_drag_end(self):
        """
        Callback triggered when the user finishes dragging the spectral region boundaries.
        Clears the drag flag and recomputes the moment maps.
        """
        self._region_dragging = False
        self.update_moment_maps()
        if self._channel_grid_popup and self._channel_grid_popup.isVisible():
            self._channel_grid_popup.update_grid()

    def update_beam_visualizers(self, panel_type, panel_id=None):
        """
        Delegate the drawing/updating of synthesized beam ellipses to the controller.

        Parameters
        ----------
        panel_type : str
            The type of panel receiving the beam (e.g., 'channel', 'moment').
        panel_id : int, optional
            The specific panel index if applicable.
        """
        return self.controller.update_beam_visualizers(panel_type, panel_id)
        
    def update_beam_positions(self, view_box, view_range=None):
        """
        Delegate repositioning the synthesized beam to stick to the bottom-left corner of the view.

        Parameters
        ----------
        view_box : pyqtgraph.ViewBox
            The view box containing the beam.
        view_range : list, optional
            The current visible range.
        """
        return self.controller.update_beam_positions(view_box, view_range)
        
    def update_moment_maps(self):
        """
        Delegate the asynchronous generation of moment maps based on the current spectral region.
        """
        self.controller.update_moment_maps()

    def _purge_finished_workers(self):
        """
        Delegate the cleanup of completed asynchronous worker threads to the controller.
        """
        self.controller._purge_finished_workers()

    def _on_moment_result(self, results: dict):
        """
        Delegate the handling of completed moment map computations to the controller.

        Parameters
        ----------
        results : dict
            The computed moment maps and metadata.
        """
        self.controller._on_moment_result(results)

    def clear_all_hover_labels(self):
        """
        Clear the text from all active hover labels across the UI.
        """
        for lbl in [self.lbl_hover_ch, self.lbl_hover_spec, self.lbl_hover_pv] + [p['lbl_hover'] for p in self.panels]:
            lbl.setText("")

    def hover_event(self, pos, plot_item, data_array, active_label, panel_id='channel'):
        """
        Handle mouse hover events to display coordinates and flux values dynamically.

        Parameters
        ----------
        pos : pyqtgraph.Point
            The scene position of the mouse.
        plot_item : pyqtgraph.PlotItem
            The plot being hovered over.
        data_array : numpy.ndarray
            The underlying data array for extracting flux.
        active_label : QLabel
            The UI label to update with coordinate text.
        panel_id : str, optional
            The string identifier of the panel being hovered.
        """
        self.clear_all_hover_labels()

        hovered_pv_cut_name = None
        hovered_offset = None
        
        if panel_id == 'channel' and plot_item.sceneBoundingRect().contains(pos):
            mp = plot_item.vb.mapSceneToView(pos)
            mouse_pt = np.array([mp.x(), mp.y()])
            
            from PyQt5.QtCore import QPointF
            for item in self.pv_cuts:
                hit = False
                w_item = item.get("width_item")
                if w_item is not None and w_item.isVisible() and w_item.contains(QPointF(mp.x(), mp.y())):
                    hit = True
                elif item.get("roi") is not None and self.line_roi_hit_test(item["roi"], pos, tolerance=10.0):
                    hit = True
                    
                if hit:
                    pts = self.get_line_roi_points(item["roi"])
                    if pts is not None:
                        p1, p2 = pts
                        vec = p2 - p1
                        length = np.hypot(vec[0], vec[1])
                        if length > 0:
                            unit = vec / length
                            offset = np.dot(mouse_pt - p1, unit)
                            offset = max(0.0, min(length, offset))
                            hovered_pv_cut_name = item["name"]
                            hovered_offset = offset
                            break

        if hasattr(self, 'pv_hover_line_main'):
            if self.combo_panel_mode.currentText() != "Spatial Analysis" and hovered_pv_cut_name is not None and getattr(self, 'combo_pv_cuts', None) and self.combo_pv_cuts.currentText() == hovered_pv_cut_name:
                self.pv_hover_line_main.setPos(hovered_offset)
                self.pv_hover_line_main.show()
                self.lbl_hover_pv.setText(f"Offset: {hovered_offset:.2f} arcsec")
                self.lbl_hover_pv.setStyleSheet("color: #3498db; font-weight: bold; font-size: 9.5px;")
            else:
                self.pv_hover_line_main.hide()

        for panel in self.panels:
            if 'pv_hover_line' in panel:
                if panel['combo'].currentText() == "PV Diagram" and hovered_pv_cut_name is not None and panel['combo_pv_cut'].currentText() == hovered_pv_cut_name:
                    panel['pv_hover_line'].setPos(hovered_offset)
                    panel['pv_hover_line'].show()
                    panel['lbl_hover'].setText(f"Offset: {hovered_offset:.2f} arcsec")
                    panel['lbl_hover'].setStyleSheet("color: #3498db; font-weight: bold; font-size: 9.5px;")
                else:
                    panel['pv_hover_line'].hide()

        if data_array is None or self.cube_clean is None: return

        if getattr(self, 'is_drawing_polygon', False) and panel_id == 'channel':
            if self.polygon_points and self.polygon_preview_line:
                mp = plot_item.vb.mapSceneToView(pos)
                pts = list(self.polygon_points)
                pts.append([mp.x(), mp.y()])
                arr = np.array(pts)
                self.polygon_preview_line.setData(arr[:,0], arr[:,1], symbol='o', symbolSize=5)
        if plot_item.sceneBoundingRect().contains(pos):
            mp = plot_item.vb.mapSceneToView(pos)
            
            start_x = (self.nx / 2) * self.pix_scale_arcsec
            start_y = -(self.ny / 2) * self.pix_scale_arcsec
            x_idx = int((mp.x() - start_x) / (-self.pix_scale_arcsec))
            y_idx = int((mp.y() - start_y) / self.pix_scale_arcsec)
            
            if 0 <= x_idx < self.nx and 0 <= y_idx < self.ny:
                val = data_array[x_idx, y_idx]
                unit_str = self.display_unit if panel_id == 'channel' else self.panels[panel_id].get('unit', '')
                
                # Format the flux value
                val_str = f"{val:.3e}" if (not np.isnan(val) and abs(val) < 1e-3 and abs(val)>0) else f"{val:.4g}" if not np.isnan(val) else "NaN"

                # Calculate Absolute RA and Dec
                if self.wcs_2d is not None:
                    coord = self.wcs_2d.pixel_to_world(x_idx, y_idx)
                    ra_str = coord.ra.to_string(unit=u.hourangle, sep=':', precision=2, pad=True)
                    dec_str = coord.dec.to_string(unit=u.degree, sep=':', precision=1, alwayssign=True, pad=True)
                    coord_text = f"RA: {ra_str}, DEC: {dec_str}"
                else:
                    # Fallback to pixels if WCS fails to load
                    coord_text = f"({x_idx}, {y_idx})"
                
                # Set the final label text (Pixels + Absolute RA/Dec + Value)
                active_label.setText(f"({x_idx}, {y_idx}) | {coord_text} | {val_str} {unit_str}")
                active_label.setStyleSheet("color: #3498db; font-weight: bold; font-size: 9.5px;")
                return
        active_label.setText("")

    def hover_spectrum(self, pos, widget=None):
        """
        Handle mouse hover events over spectrum plots to display flux/velocity coordinates.

        Parameters
        ----------
        pos : pyqtgraph.Point
            The mouse scene position.
        widget : pyqtgraph.PlotWidget, optional
            The specific plot widget being hovered. Inferred if not provided.
        """
        self.clear_all_hover_labels()
        if self.cube_clean is None or self.v_axis is None: return
        
        # If widget is not provided, try to find which one contains the mouse
        if widget is None:
            if self.plot_widget.sceneBoundingRect().contains(pos):
                widget = self.plot_widget
            elif self.plot_widget_smooth.sceneBoundingRect().contains(pos):
                widget = self.plot_widget_smooth
            else:
                return
        elif not widget.sceneBoundingRect().contains(pos):
            return

        is_smooth = (widget == self.plot_widget_smooth)
        mp = widget.plotItem.vb.mapSceneToView(pos)
        
        # Find closest velocity index
        idx = (np.abs(self.v_axis - mp.x())).argmin()
        if idx >= len(self.v_axis): return
        
        v_val = self.v_axis[idx]
        
        # Collect values from all active curves in this plot
        all_vals = [] # List of (sort_key, display_name, value_str)
        
        # Check Whole Map curve
        main_curve = self.spectrum_curve_smooth if is_smooth else self.spectrum_curve
        if main_curve.yData is not None and len(main_curve.yData) > idx:
            if len(main_curve.yData) > 0:
                y = main_curve.yData[idx]
                if np.isfinite(y):
                    val_str = f'{y:.3e}' if (abs(y) < 1e-3 and abs(y) > 0) else f'{y:.4g}'
                    # Sort key -1 to ensure it's first
                    all_vals.append((-1, "", val_str))
        
        # Check region curves
        region_curves = getattr(self, 'spectrum_curves_smooth' if is_smooth else 'spectrum_curves', {})
        for name, c in region_curves.items():
            if c.yData is not None and len(c.yData) > idx:
                y = c.yData[idx]
                if np.isfinite(y):
                    val_str = f'{y:.3e}' if (abs(y) < 1e-3 and abs(y) > 0) else f'{y:.4g}'
                    
                    if name.startswith("SR"):
                        try:
                            # Extract number, handling "SR 1" or "SR1"
                            num_str = name.replace("SR", "").strip()
                            num = int(num_str)
                            display = f"SR{num}"
                            sort_key = (1, num) # Group 1 for Regions
                        except:
                            display = name[:2].upper()
                            sort_key = (0, display) # Group 0 for Custom
                    else:
                        display = name[:2].upper()
                        sort_key = (0, display)
                        
                    all_vals.append((sort_key, display, val_str))
        
        # Sort values based on the key
        all_vals.sort(key=lambda x: x[0])
        
        if all_vals:
            # Format the strings
            display_parts = []
            for _, disp, v_str in all_vals:
                if disp == "":
                    display_parts.append(v_str)
                else:
                    display_parts.append(f"{disp}: {v_str}")
            
            unit = getattr(self, 'spec_unit', '')
            self.lbl_hover_spec.setText(f"Ch: {idx} | {v_val:.2f} km/s | " + " | ".join(display_parts) + f" {unit}")
            self.lbl_hover_spec.setStyleSheet("color: #3498db; font-weight: bold; font-size: 9.5px;")



    def hover_pv(self, pos):
        """
        Handle mouse hover events over the main PV diagram to display velocity and offset coordinates.

        Parameters
        ----------
        pos : pyqtgraph.Point
            The mouse scene position.
        """
        self.clear_all_hover_labels()
        if self.pv_data is None or self.pv_offset_axis is None or self.pv_velocity_axis is None:
            return
        if not self.pv_plot_item.sceneBoundingRect().contains(pos):
            return

        mp = self.pv_plot_item.vb.mapSceneToView(pos)
        x_idx = int(np.abs(self.pv_offset_axis - mp.x()).argmin())
        y_idx = int(np.abs(self.pv_velocity_axis - mp.y()).argmin())
        if 0 <= x_idx < self.pv_data.shape[0] and 0 <= y_idx < self.pv_data.shape[1]:
            val = self.pv_data[x_idx, y_idx]
            val_str = f"{val:.3e}" if (np.isfinite(val) and abs(val) < 1e-3 and abs(val) > 0) else f"{val:.4g}" if np.isfinite(val) else "NaN"
            self.lbl_hover_pv.setText(
                f"Offset: {self.pv_offset_axis[x_idx]:.2f} arcsec | Vel: {self.pv_velocity_axis[y_idx]:.2f} km/s | {val_str} {self.display_unit}"
            )
            self.lbl_hover_pv.setStyleSheet("color: #3498db; font-weight: bold; font-size: 9.5px;")

    def update_region_ui_visibility(self):
        """Delegate the toggling of visibility for region-related UI elements (dropdowns, labels)."""
        return self.controller.update_region_ui_visibility()
    def rename_regions(self):
        """
        Normalize and re-index the names of all active spectral extraction boxes (e.g., "Box 1", "Box 2")
        after one is added or removed, updating the associated UI dropdowns.
        """
        self.combo_regions.blockSignals(True)
        self.combo_regions_2.blockSignals(True)
        self.combo_regions_3.blockSignals(True)
        
        self.combo_regions.clear()
        self.combo_regions_2.clear()
        self.combo_regions_3.clear()
        
        self.combo_regions.addItem("None")
        self.combo_regions_2.addItem("None")
        self.combo_regions_3.addItem("None")
        
        active_rois = self.get_active_spectrum_rois()
        for i, item in enumerate(active_rois):
            new_name = f"Box {i + 1}"
            item["name"] = new_name
            item["text_item"].setText(new_name)
            
            self.combo_regions.addItem(new_name)
            self.combo_regions_2.addItem(new_name)
            self.combo_regions_3.addItem(new_name)
            
            item["roi"].setPen(pg.mkPen('c', width=2))
            
        self.combo_regions.blockSignals(False)
        self.combo_regions_2.blockSignals(False)
        self.combo_regions_3.blockSignals(False)
        
        self.on_region_selected()
        # Refresh popup box list
        if self._spectral_stats_popup and self._spectral_stats_popup.isVisible():
            self.refresh_spectral_stats_popup()

    def delete_region(self, roi):
        """
        Remove a specific spectral extraction ROI and its associated text label from the active plot.

        Parameters
        ----------
        roi : pyqtgraph.LinearRegionROI
            The region object to remove.
        """
        if roi.scene():
            roi.scene().removeItem(roi)
        else:
            self.get_active_spectrum_plot().removeItem(roi)
            
        active_rois = self.get_active_spectrum_rois()
        for i, item in enumerate(active_rois):
            if item["roi"] == roi:
                ti = item["text_item"]
                if ti.scene():
                    ti.scene().removeItem(ti)
                else:
                    self.get_active_spectrum_plot().removeItem(ti)
                active_rois.pop(i)
                break

    def delete_selected_regions(self):
        """
        Remove all spectral extraction regions currently marked for deletion by the user.
        """
        for roi in list(self.rois_to_delete):
            self.delete_region(roi)
        self.rois_to_delete.clear()
        self.update_region_ui_visibility()
        self.rename_regions()

    def clear_spectrum_regions(self):
        """
        Wipe all spectral extraction regions from the current plot entirely.
        """
        active_rois = self.get_active_spectrum_rois()
        for item in list(active_rois):
            self.delete_region(item["roi"])
        self.rois_to_delete.clear()
        self.update_region_ui_visibility()
        self.rename_regions()

    def select_region_for_deletion(self, roi):
        """
        Toggle the deletion queue state for a given spectral extraction region.

        Parameters
        ----------
        roi : pyqtgraph.LinearRegionROI
            The region to flag or unflag.
        """
        if roi in self.rois_to_delete:
            self.rois_to_delete.remove(roi)
        else:
            self.rois_to_delete.append(roi)
        self.on_region_selected()

    def add_spectrum_region(self, roi):
        """
        Delegate the registration of a newly drawn spectral extraction region to the controller.
        """
        return self.controller.add_spectrum_region(roi)
    def open_spectral_stats_popup(self):
        """
        Launch a floating window displaying statistical measurements (flux, peak, SNR) 
        for active spectral regions.
        """
        if self._spectral_stats_popup is None or not self._spectral_stats_popup.isVisible():

            main_win = self.window()

            popup = QDialog(main_win)
            popup.setWindowTitle("Spectral Statistics")
            popup.setMinimumWidth(480)
            popup.setWindowFlags(
                Qt.Window | Qt.WindowCloseButtonHint | Qt.WindowMinimizeButtonHint
            )
            popup.setAttribute(Qt.WA_DeleteOnClose, False)

            layout = QVBoxLayout(popup)
            layout.setSpacing(8)
            layout.setContentsMargins(10, 10, 10, 10)

            _BTN_STYLE = "font-size: 9px; padding: 1px 6px;"

            # ── Velocity Boxes ────────────────────────────────────────────
            popup.grp_boxes = QGroupBox("Velocity Boxes (drawn on spectrum)")
            boxes_outer = QVBoxLayout(popup.grp_boxes)
            boxes_outer.setSpacing(4)
            sel_row_b = QHBoxLayout()
            btn_b_all  = QPushButton("Select All");   btn_b_all.setStyleSheet(_BTN_STYLE);  btn_b_all.setFixedHeight(20)
            btn_b_none = QPushButton("Deselect All"); btn_b_none.setStyleSheet(_BTN_STYLE); btn_b_none.setFixedHeight(20)
            sel_row_b.addWidget(btn_b_all); sel_row_b.addWidget(btn_b_none); sel_row_b.addStretch()
            boxes_outer.addLayout(sel_row_b)
            popup.boxes_grid = QGridLayout(); popup.boxes_grid.setHorizontalSpacing(8); popup.boxes_grid.setVerticalSpacing(2)
            boxes_outer.addLayout(popup.boxes_grid)

            def _sel_all_boxes(_, p=popup):
                """Check all spectral box selection checkboxes."""
                for i in range(p.boxes_grid.count()):
                    w = p.boxes_grid.itemAt(i).widget()
                    if isinstance(w, QCheckBox): w.setChecked(True)
            def _sel_none_boxes(_, p=popup):
                """Uncheck all spectral box selection checkboxes."""
                for i in range(p.boxes_grid.count()):
                    w = p.boxes_grid.itemAt(i).widget()
                    if isinstance(w, QCheckBox): w.setChecked(False)
            btn_b_all.clicked.connect(_sel_all_boxes)
            btn_b_none.clicked.connect(_sel_none_boxes)
            layout.addWidget(popup.grp_boxes)

            # ── Spatial Aperture Regions ──────────────────────────────────
            popup.grp_apertures = QGroupBox("Spatial Aperture Regions")
            ap_outer = QVBoxLayout(popup.grp_apertures)
            ap_outer.setSpacing(4)
            sel_row_a = QHBoxLayout()
            btn_a_all  = QPushButton("Select All");   btn_a_all.setStyleSheet(_BTN_STYLE);  btn_a_all.setFixedHeight(20)
            btn_a_none = QPushButton("Deselect All"); btn_a_none.setStyleSheet(_BTN_STYLE); btn_a_none.setFixedHeight(20)
            ap_note = QLabel("(all unchecked = Whole Map)"); ap_note.setStyleSheet("font-style: italic; font-size: 10px;")
            sel_row_a.addWidget(btn_a_all); sel_row_a.addWidget(btn_a_none); sel_row_a.addWidget(ap_note); sel_row_a.addStretch()
            ap_outer.addLayout(sel_row_a)
            popup.apertures_grid = QGridLayout(); popup.apertures_grid.setHorizontalSpacing(8); popup.apertures_grid.setVerticalSpacing(2)
            ap_outer.addLayout(popup.apertures_grid)

            def _sel_all_ap(_, p=popup):
                """Check all spatial aperture selection checkboxes."""
                for i in range(p.apertures_grid.count()):
                    w = p.apertures_grid.itemAt(i).widget()
                    if isinstance(w, QCheckBox): w.setChecked(True)
            def _sel_none_ap(_, p=popup):
                """Uncheck all spatial aperture selection checkboxes."""
                for i in range(p.apertures_grid.count()):
                    w = p.apertures_grid.itemAt(i).widget()
                    if isinstance(w, QCheckBox): w.setChecked(False)
            btn_a_all.clicked.connect(_sel_all_ap)
            btn_a_none.clicked.connect(_sel_none_ap)
            layout.addWidget(popup.grp_apertures)

            # ── Statistics to compute (3-column grid) ────────────────────
            grp_stats = QGroupBox("Statistics to compute")
            stats_grid = QGridLayout(grp_stats)
            stats_grid.setHorizontalSpacing(12)
            _STAT_OPTIONS = [
                ("Integrated Intensity", False), ("RMS", False),        ("Peak (Max)", False),
                ("Min", False),                 ("Mean", False),       ("Median", False),
                ("Std. Deviation", False),      ("SNR (Peak/RMS)", False), ("Sum", False),
            ]
            popup.stat_checkboxes = {}
            for idx, (stat_name, default_on) in enumerate(_STAT_OPTIONS):
                cb = QCheckBox(stat_name)
                cb.setChecked(default_on)
                cb.toggled.connect(lambda checked, p=popup: self._run_spectral_stats_calc(p))
                stats_grid.addWidget(cb, *divmod(idx, 3))
                popup.stat_checkboxes[stat_name] = cb
            layout.addWidget(grp_stats)

            # ── Results (scrollable, dark bg + green text) ───────────────
            results_group = QGroupBox("Results")
            results_inner = QVBoxLayout(results_group)
            results_inner.setContentsMargins(4, 4, 4, 4)
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setMinimumHeight(90)
            scroll.setMaximumHeight(200)
            scroll.setFrameShape(QScrollArea.NoFrame)
            popup.lbl_result = QLabel("---")
            popup.lbl_result.setWordWrap(True)
            popup.lbl_result.setAlignment(Qt.AlignLeft | Qt.AlignTop)
            popup.lbl_result.setTextFormat(Qt.RichText)
            popup.lbl_result.setContentsMargins(6, 6, 6, 6)
            popup.lbl_result.setStyleSheet(
                "background-color: #0d0d0d; color: #2ecc71; font-weight: bold; font-size: 11px;"
            )
            scroll.setWidget(popup.lbl_result)
            results_inner.addWidget(scroll)
            layout.addWidget(results_group)
            # No separate Calculate button — all calculations run live on toggle

            self._spectral_stats_popup = popup
            self.refresh_spectral_stats_popup()
            btn_pos = self.btn_spectral_stats.mapToGlobal(self.btn_spectral_stats.rect().topRight())
            popup.move(btn_pos)
            popup.show()
        else:
            self._spectral_stats_popup.raise_()
            self._spectral_stats_popup.activateWindow()

    def open_channel_grid_popup(self):
        """
        Open a floating grid view window to display multiple contiguous spectral channel maps simultaneously.
        """
        if self._channel_grid_popup is None:
            from src.gui.components.channel_grid_view import ChannelGridView
            from src.gui.controllers.channel_grid_controller import ChannelGridController
            
            self._channel_grid_popup = ChannelGridView(self.window())
            self._channel_grid_controller = ChannelGridController(self._channel_grid_popup, self)
            
        self._channel_grid_controller.update_grid()
        self._channel_grid_popup.show()
        self._channel_grid_popup.raise_()
        self._channel_grid_popup.activateWindow()

    def refresh_spectral_stats_popup(self):
        """
        Delegate the updating of values inside the spectral stats popup window.
        """
        return self.controller.refresh_spectral_stats_popup()

    def refresh_spectral_stats_apertures(self):
        """
        Delegate the extraction of flux data from specific apertures for the stats popup.
        """
        return self.controller.refresh_spectral_stats_apertures()

    def _get_popup_selected_boxes(self, popup):
        """
        Extract the checked spectral extraction regions from the statistics popup window.
        
        Parameters
        ----------
        popup : QDialog
            The statistics popup window.
            
        Returns
        -------
        list of pyqtgraph.LinearRegionROI
            A list of the user-selected spectral region instances.
        """
        boxes = []
        active_rois = self.get_active_spectrum_rois()
        for i in range(popup.boxes_grid.count()):
            widget = popup.boxes_grid.itemAt(i).widget()
            if isinstance(widget, QCheckBox) and widget.isChecked():
                name = widget.text()
                for item in active_rois:
                    if item["name"] == name:
                        boxes.append(item["roi"])
                        break
        return boxes

    def _get_popup_selected_apertures(self, popup):
        """
        Returns list of r_dict for selected apertures; empty means Whole Map.
        
        Parameters
        ----------
        popup : QDialog
            The statistics popup window.
            
        Returns
        -------
        list of dict
            A list of dictionary configurations for selected spatial apertures.
        """
        selected = []
        spatial_rois = getattr(self, 'spectrum_spatial_rois', [])
        for i in range(popup.apertures_grid.count()):
            widget = popup.apertures_grid.itemAt(i).widget()
            if isinstance(widget, QCheckBox) and widget.isChecked():
                name = widget.text()
                for r_dict in spatial_rois:
                    if r_dict["name"] == name:
                        selected.append(r_dict)
                        break
        return selected

    def _extract_spectrum_for_stats(self, spatial_roi):
        """
        Extract mean spectrum from cube_clean for a given spatial ROI (or whole map if None).
        
        Parameters
        ----------
        spatial_roi : pyqtgraph.ROI or None
            The spatial region to extract over, or None for the whole map.
            
        Returns
        -------
        tuple
            (v_axis_sorted, flux_sorted) or (None, None) on failure.
        """
        if self.cube_clean is None: return None, None
        stat = self.combo_spec_stat.currentText()
        try:
            if spatial_roi is None:
                sub_data = self.cube_clean
            else:
                sub_data = spatial_roi.getArrayRegion(
                    self.cube_clean, self.view_channel.getImageItem(), axes=(1, 2))
            if "Max" in stat:
                spec = np.nanmax(sub_data, axis=(1, 2))
            elif "Sum" in stat or "Flux Density" in stat:
                spec = np.nansum(sub_data, axis=(1, 2))
                if self.pixels_per_beam > 1.0:
                    spec /= self.pixels_per_beam
            else:
                spec = np.nanmean(sub_data, axis=(1, 2))
            sort_idx = np.argsort(self.v_axis)
            return self.v_axis[sort_idx], spec[sort_idx]
        except Exception:
            return None, None

    def _run_spectral_stats_calc(self, popup):
        """
        Delegate execution of the live statistical calculations for the popup window.
        """
        self.controller._run_spectral_stats_calc(popup)

    def on_region_selected(self, _=None):
        """
        Callback fired when a spectral region's selection state changes (e.g., via dropdown or click),
        updating the pen colors (cyan=normal, yellow=selected, red=to delete) and triggering 
        any active calculation popups.
        """
        selected_names = [
            self.combo_regions.currentText(),
            self.combo_regions_2.currentText(),
            self.combo_regions_3.currentText()
        ]
        
        active_rois = self.get_active_spectrum_rois()
        for item in active_rois:
            roi = item["roi"]
            if roi in getattr(self, 'rois_to_delete', []):
                roi.setPen(pg.mkPen('r', width=3))
            elif item["name"] in selected_names and item["name"] != "None":
                roi.setPen(pg.mkPen('y', width=3))
            else:
                roi.setPen(pg.mkPen('c', width=2))
        # Trigger recalculation in popup if open
        if self._spectral_stats_popup and self._spectral_stats_popup.isVisible():
            self._run_spectral_stats_calc(self._spectral_stats_popup)

    def update_spectrum_region_calc(self, _=None):
        """
        Delegate the main panel's inline region statistic calculations (e.g., Integrated Flux).
        """
        return self.controller.update_spectrum_region_calc(_)
