import pyqtgraph as pg
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QScrollArea, 
    QLabel, QPushButton, QComboBox
)
from PyQt5.QtCore import pyqtSignal

class ChannelGridView(QDialog):
    """
    Pure UI View for the Channel Grid.
    Handles layout, theming, and user interaction signals.
    """
    # Expose user interactions as signals for the Controller
    cmap_changed = pyqtSignal(str)
    reset_zoom_clicked = pyqtSignal()
    export_pdf_clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Channel Grid")
        self.setMinimumSize(800, 600)
        
        # Set dark theme
        self.setStyleSheet("background-color: #1a1a1a; color: #e0e0e0;")
        
        layout = QHBoxLayout(self)
        
        # --- Left side: Scroll area + Hover label ---
        left_layout = QVBoxLayout()
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setStyleSheet("QScrollArea { border: none; }")
        
        self.grid_widget = pg.GraphicsLayoutWidget()
        self.grid_widget.setBackground('#1a1a1a')
        self.scroll.setWidget(self.grid_widget)
        left_layout.addWidget(self.scroll, stretch=1)
        
        self.lbl_hover = QLabel("Hover over a tile to see details")
        self.lbl_hover.setStyleSheet("font-family: monospace; font-size: 11.5px; color: #aaa; padding: 5px;")
        left_layout.addWidget(self.lbl_hover)
        
        layout.addLayout(left_layout, stretch=4)
        
        # --- Right side: Colorbar (Histogram) + Export Controls ---
        right_layout = QVBoxLayout()
        self.hist = pg.HistogramLUTWidget()
        right_layout.addWidget(self.hist, stretch=1)
        
        btn_layout = QHBoxLayout()
        
        self.combo_cmap = QComboBox()
        self.combo_cmap.addItems(['Turbo', 'Inferno', 'Viridis', 'Plasma', 'Magma', 'Cubehelix', 'Grey'])
        self.combo_cmap.setStyleSheet("background-color: #34495e; color: white; padding: 5px;")
        btn_layout.addWidget(self.combo_cmap)
        
        self.btn_reset_zoom = QPushButton("Reset zoom")
        self.btn_reset_zoom.setStyleSheet("background-color: #34495e; color: white; padding: 5px;")
        btn_layout.addWidget(self.btn_reset_zoom)
        
        self.btn_export = QPushButton("Export to PDF")
        self.btn_export.setStyleSheet("background-color: #2c3e50; color: white; padding: 5px;")
        btn_layout.addWidget(self.btn_export)
        
        right_layout.addLayout(btn_layout)
        layout.addLayout(right_layout, stretch=1)
        
        # --- Internal Signal Wiring ---
        self.combo_cmap.currentTextChanged.connect(self.cmap_changed.emit)
        self.btn_reset_zoom.clicked.connect(self.reset_zoom_clicked.emit)
        self.btn_export.clicked.connect(self.export_pdf_clicked.emit)
