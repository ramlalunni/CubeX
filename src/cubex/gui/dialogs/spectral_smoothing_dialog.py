"""
Module containing the dialog for configuring 1D spectral smoothing.

This dialog lets users select and parameterize different smoothing 
filters (Boxcar, Gaussian, Savitzky-Golay) to be applied to the 
active 1D spectrum plot.
"""
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QLineEdit, QComboBox, QDoubleSpinBox, 
                             QRadioButton, QSpinBox, QTableWidget, QTableWidgetItem, 
                             QHeaderView, QMessageBox, QStackedWidget, QWidget,
                             QScrollArea, QGridLayout, QFileDialog, QGroupBox, QCheckBox,
                             QTabWidget)
import numpy as np
import pyqtgraph as pg
import matplotlib.pyplot as plt

class SpectralSmoothingDialog(QDialog):
    """
    Dialog to select spectral smoothing method and parameters.

    Attributes
    ----------
    apply_clicked : PyQt5.QtCore.pyqtSignal
        Signal emitted when the user clicks 'Apply', passing the smoothing parameters dict.
    combo_method : PyQt5.QtWidgets.QComboBox
        Dropdown menu to select the smoothing method.
    param_stack : PyQt5.QtWidgets.QStackedWidget
        Container holding the specific parameter inputs for the selected method.
    """
    apply_clicked = pyqtSignal(dict)
    def __init__(self, parent=None):
        """
        Initialize the SpectralSmoothingDialog.

        Parameters
        ----------
        parent : PyQt5.QtWidgets.QWidget, optional
            The parent widget, by default None.
        """
        super().__init__(parent)
        self.setWindowTitle("Spectral Smoothing")
        self.setMinimumWidth(350)
        self.initUI()
        
    def initUI(self):
        """
        Build the UI elements for selecting smoothing methods and their parameters.
        """
        layout = QVBoxLayout(self)
        
        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel("Method:"))
        self.combo_method = QComboBox()
        self.combo_method.addItems(["Boxcar", "Hanning", "Gaussian", "Savitzky-Golay"])
        method_layout.addWidget(self.combo_method)
        layout.addLayout(method_layout)
        
        self.param_stack = QStackedWidget()
        
        self.widget_boxcar = QWidget()
        boxcar_layout = QHBoxLayout(self.widget_boxcar)
        boxcar_layout.setContentsMargins(0, 0, 0, 0)
        boxcar_layout.addWidget(QLabel("Window Size (channels):"))
        self.spin_boxcar_w = QSpinBox()
        self.spin_boxcar_w.setRange(2, 1000)
        self.spin_boxcar_w.setValue(3)
        boxcar_layout.addWidget(self.spin_boxcar_w)
        boxcar_layout.addStretch()
        self.param_stack.addWidget(self.widget_boxcar)
        
        self.widget_hanning = QWidget()
        hanning_layout = QHBoxLayout(self.widget_hanning)
        hanning_layout.setContentsMargins(0, 0, 0, 0)
        hanning_layout.addWidget(QLabel("Window Size (channels):"))
        self.spin_hanning_w = QSpinBox()
        self.spin_hanning_w.setRange(3, 1000)
        self.spin_hanning_w.setValue(5)
        hanning_layout.addWidget(self.spin_hanning_w)
        hanning_layout.addStretch()
        self.param_stack.addWidget(self.widget_hanning)

        self.widget_gauss = QWidget()
        gauss_layout = QHBoxLayout(self.widget_gauss)
        gauss_layout.setContentsMargins(0, 0, 0, 0)
        gauss_layout.addWidget(QLabel("Sigma (channels):"))
        self.spin_gauss_sigma = QDoubleSpinBox()
        self.spin_gauss_sigma.setDecimals(1)
        self.spin_gauss_sigma.setRange(0.1, 500.0)
        self.spin_gauss_sigma.setValue(1.0)
        self.spin_gauss_sigma.setSingleStep(0.5)
        gauss_layout.addWidget(self.spin_gauss_sigma)
        gauss_layout.addStretch()
        self.param_stack.addWidget(self.widget_gauss)
        
        self.widget_savgol = QWidget()
        savgol_layout = QHBoxLayout(self.widget_savgol)
        savgol_layout.setContentsMargins(0, 0, 0, 0)
        savgol_layout.addWidget(QLabel("Window Size (channels):"))
        self.spin_savgol_w = QSpinBox()
        self.spin_savgol_w.setRange(3, 999)
        self.spin_savgol_w.setSingleStep(2)
        self.spin_savgol_w.setValue(5)
        savgol_layout.addWidget(self.spin_savgol_w)
        
        savgol_layout.addWidget(QLabel("Poly Order:"))
        self.spin_savgol_p = QSpinBox()
        self.spin_savgol_p.setRange(1, 10)
        self.spin_savgol_p.setValue(2)
        savgol_layout.addWidget(self.spin_savgol_p)
        savgol_layout.addStretch()
        self.param_stack.addWidget(self.widget_savgol)
        
        layout.addWidget(self.param_stack)
        
        self.combo_method.currentIndexChanged.connect(self.param_stack.setCurrentIndex)
        self.spin_savgol_w.valueChanged.connect(self.validate_savgol)
        
        btn_layout = QHBoxLayout()
        btn_apply = QPushButton("Apply")
        btn_apply.setStyleSheet("background-color: #27ae60; font-weight: bold; color: white;")
        btn_cancel = QPushButton("Cancel")
        btn_apply.clicked.connect(self._on_apply)
        btn_cancel.clicked.connect(self.reject)
        btn_layout.addStretch()
        btn_layout.addWidget(btn_apply)
        btn_layout.addWidget(btn_cancel)
        layout.addLayout(btn_layout)
        
    def _on_apply(self):
        """
        Emit the `apply_clicked` signal with the current parameters when the Apply button is clicked.
        """
        params = self.get_params()
        if params:
            self.apply_clicked.emit(params)
        
    def validate_savgol(self, value):
        """
        Ensure the Savitzky-Golay window size remains an odd number.

        Parameters
        ----------
        value : int
            The current value of the window size spinbox.
        """
        if value % 2 == 0:
            self.spin_savgol_w.setValue(value + 1)
            
    def get_params(self):
        """
        Retrieve the currently selected smoothing method and its associated parameters.

        Returns
        -------
        dict or None
            A dictionary of parameters suitable for `spectral_smoothing` routines, 
            or None if the method is unrecognized.
        """
        method = self.combo_method.currentText()
        if method == "Boxcar":
            return {"method": "boxcar", "window": self.spin_boxcar_w.value()}
        elif method == "Gaussian":
            return {"method": "gaussian", "sigma": self.spin_gauss_sigma.value()}
        elif method == "Savitzky-Golay":
            w = self.spin_savgol_w.value()
            p = self.spin_savgol_p.value()
            if p >= w:
                p = w - 1
            return {"method": "savgol", "window": w, "polyorder": p}
        elif method == "Hanning":
            return {"method": "hanning", "window": self.spin_hanning_w.value()}
        return None
