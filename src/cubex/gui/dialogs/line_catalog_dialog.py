"""
Module containing the dialog for querying molecular line databases.

This dialog allows users to specify search parameters (v_sys, database, E_up, species)
to fetch spectral line data from the Splatalogue catalog over the current frequency range.
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

class LineCatalogDialog(QDialog):
    """
    Dialog to input parameters for querying the Splatalogue database.

    Attributes
    ----------
    fmin : float
        Minimum observed frequency in GHz.
    fmax : float
        Maximum observed frequency in GHz.
    action : str
        The action taken by the user ('query', 'clear', or 'cancel').
    """
    def __init__(self, parent, obs_fmin, obs_fmax):
        """
        Initialize the LineCatalogDialog.

        Parameters
        ----------
        parent : PyQt5.QtWidgets.QWidget
            The parent widget.
        obs_fmin : float
            Minimum observed frequency of the spectrum.
        obs_fmax : float
            Maximum observed frequency of the spectrum.
        """
        super().__init__(parent)
        self.setWindowTitle("Molecular Line Catalog Query")
        self.setMinimumWidth(420)
        self.fmin, self.fmax = obs_fmin, obs_fmax

        layout = QVBoxLayout(self)
        
        info = QLabel(f"<b>Selected Obs Range:</b> {self.fmin:.3f} to {self.fmax:.3f} GHz<br><i>(Rest frequency range will adjust based on V_sys)</i>")
        layout.addWidget(info)
        
        form = QVBoxLayout()
        
        h1 = QHBoxLayout()
        h1.addWidget(QLabel("Source Systemic Velocity ($v_{sys}$):"))
        self.spin_vsys = QDoubleSpinBox()
        self.spin_vsys.setRange(-1000, 1000)
        self.spin_vsys.setDecimals(2)
        self.spin_vsys.setValue(0.00)
        h1.addWidget(self.spin_vsys)
        h1.addWidget(QLabel("km/s"))
        form.addLayout(h1)
        
        h2 = QHBoxLayout()
        h2.addWidget(QLabel("Database:"))
        self.combo_db = QComboBox()
        self.combo_db.addItems(["CDMS", "JPL", "CDMS & JPL"])
        h2.addWidget(self.combo_db)
        form.addLayout(h2)
        
        h3 = QHBoxLayout()
        h3.addWidget(QLabel("Max Upper State Energy ($E_{up}$):"))
        self.spin_eup = QDoubleSpinBox()
        self.spin_eup.setRange(10, 5000)
        self.spin_eup.setValue(500.0)
        h3.addWidget(self.spin_eup)
        h3.addWidget(QLabel("K"))
        form.addLayout(h3)
        
        h4 = QHBoxLayout()
        h4.addWidget(QLabel("Species Filter:"))
        self.edit_species = QLineEdit()
        self.edit_species.setPlaceholderText("e.g., CS, CO, HCN (leave blank for all)")
        h4.addWidget(self.edit_species)
        form.addLayout(h4)

        layout.addLayout(form)
        layout.addSpacing(10)

        btn_layout = QHBoxLayout()
        self.btn_query = QPushButton("Query Splatalogue")
        self.btn_query.setStyleSheet("background-color: #27ae60; font-weight: bold; color: white;")
        self.btn_clear = QPushButton("Clear Existing Lines")
        self.btn_cancel = QPushButton("Close")

        self.btn_query.clicked.connect(self.accept)
        self.btn_clear.clicked.connect(self.clear_lines)
        self.btn_cancel.clicked.connect(self.reject)

        btn_layout.addWidget(self.btn_query)
        btn_layout.addWidget(self.btn_clear)
        btn_layout.addWidget(self.btn_cancel)
        layout.addLayout(btn_layout)

        self.action = 'cancel'

    def accept(self):
        """
        Set action to 'query' and accept the dialog.
        """
        self.action = 'query'
        super().accept()
        
    def clear_lines(self):
        """
        Set action to 'clear' and accept the dialog.
        """
        self.action = 'clear'
        super().accept()
