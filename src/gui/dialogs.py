from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QLineEdit, QComboBox, QDoubleSpinBox, 
                             QRadioButton, QSpinBox, QTableWidget, QTableWidgetItem, 
                             QHeaderView, QMessageBox)

# ==============================================================================
# LINE CATALOG DIALOG
# ==============================================================================
class LineCatalogDialog(QDialog):
    """
    Dialog to input parameters for querying the Splatalogue database.
    """
    def __init__(self, parent, obs_fmin, obs_fmax):
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
        self.action = 'query'
        super().accept()
        
    def clear_lines(self):
        self.action = 'clear'
        super().accept()


# ==============================================================================
# LINE SELECTION DIALOG
# ==============================================================================
class LineSelectionDialog(QDialog):
    """
    Dialog to display fetched Splatalogue lines and allow the user to select 
    which ones to plot on the spectrum.
    """
    def __init__(self, parent, parsed_data, v_sys):
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
        for i in range(self.table.rowCount()):
            self.table.item(i, 0).setCheckState(state)

    def select_species(self):
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
        self.selected_rows = []
        for i in range(self.table.rowCount()):
            if self.table.item(i, 0).checkState() == Qt.Checked:
                self.selected_rows.append(self.parsed_data[i])
        super().accept()


# ==============================================================================
# CONTOUR DIALOG
# ==============================================================================
class ContourDialog(QDialog):
    """
    Dialog to allow the user to specify auto-generated or manual contour levels 
    for the image panels.
    """
    def __init__(self, parent, current_params, title):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(350)
        self.params = current_params or {'mode': 'auto', 'n': 5, 'levels': []}

        layout = QVBoxLayout(self)

        self.rad_auto = QRadioButton("Auto-determine optimal levels")
        self.rad_auto.setChecked(self.params['mode'] == 'auto')
        self.spin_n = QSpinBox()
        self.spin_n.setRange(1, 50)
        self.spin_n.setValue(self.params['n'])

        h1 = QHBoxLayout()
        h1.addWidget(self.rad_auto)
        h1.addWidget(QLabel("Number of levels:"))
        h1.addWidget(self.spin_n)
        layout.addLayout(h1)

        self.rad_manual = QRadioButton("Manual contour levels")
        self.rad_manual.setChecked(self.params['mode'] == 'manual')
        self.edit_manual = QLineEdit()
        self.edit_manual.setPlaceholderText("e.g., 0.1, 0.5, 1.2, 5.0")
        if self.params['levels']:
            self.edit_manual.setText(", ".join(map(str, self.params['levels'])))

        h2 = QHBoxLayout()
        h2.addWidget(self.rad_manual)
        h2.addWidget(self.edit_manual)
        layout.addLayout(h2)

        btn_layout = QHBoxLayout()
        btn_apply = QPushButton("Apply Contours")
        btn_clear = QPushButton("Clear Contours")
        btn_cancel = QPushButton("Cancel")

        btn_apply.clicked.connect(self.apply)
        btn_clear.clicked.connect(self.clear)
        btn_cancel.clicked.connect(self.reject)

        btn_layout.addWidget(btn_apply)
        btn_layout.addWidget(btn_clear)
        btn_layout.addWidget(btn_cancel)
        layout.addLayout(btn_layout)

        self.result_params = None
        self.action = 'cancel'

    def apply(self):
        if self.rad_auto.isChecked():
            self.result_params = {'mode': 'auto', 'n': self.spin_n.value(), 'levels': []}
        else:
            try:
                lvls = [float(x.strip()) for x in self.edit_manual.text().split(',') if x.strip()]
                self.result_params = {'mode': 'manual', 'n': 0, 'levels': lvls}
            except ValueError:
                QMessageBox.warning(self, "Error", "Invalid manual levels. Please use comma-separated numbers.")
                return
        self.action = 'apply'
        self.accept()

    def clear(self):
        self.action = 'clear'
        self.accept()