from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QListWidget, 
                             QListWidgetItem, QPushButton, QCheckBox, QLabel, 
                             QAbstractItemView)
from PyQt5.QtCore import Qt

class ExportRegionsDialog(QDialog):
    """
    Dialog to allow the user to select which spectrum regions (curves) to export.
    If exporting to PDF, it optionally allows choosing between a single file or multiple files.
    """
    def __init__(self, parent=None, curves=None, title="Export Regions", is_pdf=False):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.curves = curves or {}
        self._is_pdf = is_pdf
        
        self.resize(300, 400)
        self._setup_ui()
        
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        
        layout.addWidget(QLabel("Select regions to export:"))
        
        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QAbstractItemView.ExtendedSelection)
        
        for name in self.curves.keys():
            item = QListWidgetItem(name)
            self.list_widget.addItem(item)
            item.setSelected(True)  # Select all regions by default
            
        layout.addWidget(self.list_widget)
        
        self.chk_single_file = None
        if self._is_pdf:
            self.chk_single_file = QCheckBox("Export to a single PDF file")
            self.chk_single_file.setChecked(True)
            layout.addWidget(self.chk_single_file)
            
        btn_layout = QHBoxLayout()
        self.btn_export = QPushButton("Export")
        self.btn_cancel = QPushButton("Cancel")
        
        self.btn_export.clicked.connect(self.accept)
        self.btn_cancel.clicked.connect(self.reject)
        
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_cancel)
        btn_layout.addWidget(self.btn_export)
        
        layout.addLayout(btn_layout)
        
    def get_selected_regions(self):
        """Return a list of the region names selected by the user."""
        return [item.text() for item in self.list_widget.selectedItems()]
        
    def is_single_file(self):
        """Return True if the user checked the 'single file' option for PDF export."""
        if self.chk_single_file:
            return self.chk_single_file.isChecked()
        return False
