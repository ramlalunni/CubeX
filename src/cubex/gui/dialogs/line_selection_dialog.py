"""
Module containing the dialog for filtering and selecting queried catalog lines.

Presents the parsed Splatalogue data in a tabular format, allowing users
to selectively pick which transitions to plot over the 1D spectrum.
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

class LineSelectionDialog(QDialog):
    """
    Dialog to display fetched Splatalogue lines and allow the user to select 
    which ones to plot on the spectrum.

    Attributes
    ----------
    parsed_data : list of dict
        The full list of parsed molecular line dictionaries from the Astroquery call.
    v_sys : float
        The user-provided systemic velocity (in km/s) used during the query.
    selected_rows : list of dict
        The subset of `parsed_data` that the user has selected to plot.
    table : PyQt5.QtWidgets.QTableWidget
        The UI table widget displaying the line catalog.
    """
    def __init__(self, parent, parsed_data, v_sys):
        """
        Initialize the LineSelectionDialog.

        Parameters
        ----------
        parent : PyQt5.QtWidgets.QWidget
            The parent widget.
        parsed_data : list of dict
            List of transition metadata retrieved from Splatalogue.
        v_sys : float
            The systemic velocity of the source in km/s.
        """
        super().__init__(parent)
        self.setWindowTitle("Select Lines to Overlay")
        self.resize(900, 450)
        self.parsed_data = parsed_data
        self.v_sys = v_sys
        self.selected_rows = []

        layout = QVBoxLayout(self)
        info_lbl = QLabel(f"Found {len(parsed_data)} transitions. Select the ones you want to plot:")
        layout.addWidget(info_lbl)

        self.table = QTableWidget(len(parsed_data), 6)
        self.table.setHorizontalHeaderLabels(["Draw", "Molecule", "Name", "Transition QNs", "Rest Freq (GHz)", "E_up (K)"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents) 
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents) 
        self.table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeToContents)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)

        for i, row in enumerate(parsed_data):
            chk = QTableWidgetItem()
            chk.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            chk.setCheckState(Qt.Unchecked)
            self.table.setItem(i, 0, chk)

            form = str(row.get('formula', ''))
            sp = str(row.get('molecule_name', 'Unknown'))
            qn = str(row.get('QN', ''))
            fq = f"{row.get('restfreq', 0.0):.5f}"
            eu = f"{row.get('Eup(K)', 0.0):.1f}"

            self.table.setItem(i, 1, QTableWidgetItem(form))
            self.table.setItem(i, 2, QTableWidgetItem(sp))
            self.table.setItem(i, 3, QTableWidgetItem(qn))
            self.table.setItem(i, 4, QTableWidgetItem(fq))
            self.table.setItem(i, 5, QTableWidgetItem(eu))

        layout.addWidget(self.table)

        btn_layout = QHBoxLayout()
        btn_all = QPushButton("Select All")
        btn_species = QPushButton("Select all lines of chosen species")
        btn_none = QPushButton("Clear Selection")
        btn_draw = QPushButton("Plot Selected")
        btn_draw.setStyleSheet("background-color: #27ae60; font-weight: bold; color: white;")
        btn_cancel = QPushButton("Cancel")

        btn_all.clicked.connect(lambda: self.toggle_all(Qt.Checked))
        btn_none.clicked.connect(lambda: self.toggle_all(Qt.Unchecked))
        btn_species.clicked.connect(self.select_species)
        btn_draw.clicked.connect(self.accept)
        btn_cancel.clicked.connect(self.reject)

        btn_layout.addWidget(btn_all)
        btn_layout.addWidget(btn_species)
        btn_layout.addWidget(btn_none)
        btn_layout.addStretch()
        btn_layout.addWidget(btn_draw)
        btn_layout.addWidget(btn_cancel)
        layout.addLayout(btn_layout)

    def toggle_all(self, state):
        """
        Check or uncheck all checkboxes in the table.

        Parameters
        ----------
        state : Qt.CheckState
            The check state to apply to all rows.
        """
        for i in range(self.table.rowCount()):
            self.table.item(i, 0).setCheckState(state)

    def select_species(self):
        """
        Check all transitions belonging to the species of currently selected/checked rows.
        """
        target_molecules = set()
        for i in range(self.table.rowCount()):
            if self.table.item(i, 0).checkState() == Qt.Checked:
                target_molecules.add(self.table.item(i, 1).text())
                
        if not target_molecules:
            for item in self.table.selectedItems():
                row = item.row()
                target_molecules.add(self.table.item(row, 1).text())
                
        if target_molecules:
            for i in range(self.table.rowCount()):
                if self.table.item(i, 1).text() in target_molecules:
                    self.table.item(i, 0).setCheckState(Qt.Checked)

    def accept(self):
        """
        Store the selected rows into `self.selected_rows` and accept the dialog.
        """
        self.selected_rows = []
        for i in range(self.table.rowCount()):
            if self.table.item(i, 0).checkState() == Qt.Checked:
                self.selected_rows.append(self.parsed_data[i])
        super().accept()
