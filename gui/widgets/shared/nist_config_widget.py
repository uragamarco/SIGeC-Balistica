"""
NISTConfigurationWidget - Shared NIST configuration component
Eliminates code duplication between analysis and comparison tabs
"""

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, 
                             QGroupBox, QCheckBox, QLineEdit, QComboBox, 
                             QLabel, QFrame, QPushButton, QDateEdit)
from PyQt5.QtCore import Qt, pyqtSignal, QDate
from PyQt5.QtGui import QFont
from typing import Dict, Any, Optional


class CollapsibleSection(QWidget):
    """Collapsible section for organizing NIST options"""
    
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.title = title
        self.is_expanded = False
        self.content_widget = None
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the collapsible section UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Header
        self.header = QPushButton(f"▶ {self.title}")
        self.header.setProperty("class", "collapsible-header")
        self.header.clicked.connect(self.toggle_expanded)
        layout.addWidget(self.header)
        
        # Content container
        self.content_container = QFrame()
        self.content_container.setProperty("class", "collapsible-content")
        self.content_container.hide()
        
        self.content_layout = QVBoxLayout(self.content_container)
        self.content_layout.setContentsMargins(20, 10, 10, 10)
        
        layout.addWidget(self.content_container)
        
    def add_content_widget(self, widget: QWidget):
        """Add content widget to the collapsible section"""
        self.content_widget = widget
        self.content_layout.addWidget(widget)
        
    def toggle_expanded(self):
        """Toggle expanded/collapsed state"""
        self.is_expanded = not self.is_expanded
        
        if self.is_expanded:
            self.header.setText(f"▼ {self.title}")
            self.content_container.show()
        else:
            self.header.setText(f"▶ {self.title}")
            self.content_container.hide()


class NISTConfigurationWidget(QWidget):
    """Shared NIST configuration widget"""
    
    configurationChanged = pyqtSignal(dict)  # Emitted when configuration changes
    
    def __init__(self, mode: str = "analysis", parent=None):
        """
        Initialize NIST configuration widget
        
        Args:
            mode: "analysis" or "comparison" to customize labels and options
        """
        super().__init__(parent)
        self.mode = mode
        self.config = {}
        
        self.setup_ui()
        self.connect_signals()
        
    def setup_ui(self):
        """Setup the user interface"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(15)
        
        # Main enable checkbox
        self.enable_nist_cb = QCheckBox(
            "Incluir metadatos en formato NIST" + 
            (" para comparación balística" if self.mode == "comparison" else " para análisis balístico")
        )
        self.enable_nist_cb.setProperty("class", "nist-enable-checkbox")
        layout.addWidget(self.enable_nist_cb)
        
        # Main configuration panel
        self.main_panel = QFrame()
        self.main_panel.setProperty("class", "nist-config-panel")
        self.main_panel.setEnabled(False)
        
        main_layout = QVBoxLayout(self.main_panel)
        main_layout.setContentsMargins(20, 15, 20, 15)
        main_layout.setSpacing(20)
        
        # Basic Information Section
        basic_section = self.create_basic_info_section()
        main_layout.addWidget(basic_section)
        
        # Equipment Information Section
        equipment_section = self.create_equipment_section()
        main_layout.addWidget(equipment_section)
        
        # Validation Options Section
        validation_section = self.create_validation_section()
        main_layout.addWidget(validation_section)
        
        # Advanced Options (Collapsible)
        advanced_section = self.create_advanced_section()
        main_layout.addWidget(advanced_section)
        
        layout.addWidget(self.main_panel)
        
    def create_basic_info_section(self) -> QWidget:
        """Create basic information section"""
        group = QGroupBox("Información del Laboratorio")
        group.setProperty("class", "nist-section")
        
        layout = QFormLayout(group)
        layout.setSpacing(10)
        
        # Laboratory name
        self.lab_name_edit = QLineEdit()
        self.lab_name_edit.setPlaceholderText("Nombre del laboratorio forense")
        self.lab_name_edit.setProperty("class", "nist-input")
        layout.addRow("Laboratorio:", self.lab_name_edit)
        
        # Accreditation
        self.lab_accreditation_edit = QLineEdit()
        self.lab_accreditation_edit.setPlaceholderText("Número de acreditación")
        self.lab_accreditation_edit.setProperty("class", "nist-input")
        layout.addRow("Acreditación:", self.lab_accreditation_edit)
        
        # Examiner
        self.examiner_edit = QLineEdit()
        self.examiner_edit.setPlaceholderText("Nombre del examinador")
        self.examiner_edit.setProperty("class", "nist-input")
        layout.addRow("Examinador:", self.examiner_edit)
        
        # Case number
        self.case_number_edit = QLineEdit()
        self.case_number_edit.setPlaceholderText("Número de caso")
        self.case_number_edit.setProperty("class", "nist-input")
        layout.addRow("Número de Caso:", self.case_number_edit)
        
        return group
        
    def create_equipment_section(self) -> QWidget:
        """Create equipment information section"""
        group = QGroupBox("Información del Equipo")
        group.setProperty("class", "nist-section")
        
        layout = QFormLayout(group)
        layout.setSpacing(10)
        
        # Capture device
        self.capture_device_edit = QLineEdit()
        self.capture_device_edit.setPlaceholderText("Ej: Microscopio de comparación Leica FSC")
        self.capture_device_edit.setProperty("class", "nist-input")
        layout.addRow("Dispositivo de Captura:", self.capture_device_edit)
        
        # Magnification
        self.magnification_edit = QLineEdit()
        self.magnification_edit.setPlaceholderText("Ej: 40x, 100x")
        self.magnification_edit.setProperty("class", "nist-input")
        layout.addRow("Magnificación:", self.magnification_edit)
        
        # Lighting type
        self.lighting_type_combo = QComboBox()
        self.lighting_type_combo.setProperty("class", "nist-combo")
        self.lighting_type_combo.addItems([
            "Seleccionar...",
            "Luz Blanca Coaxial",
            "Luz Oblicua", 
            "Luz Polarizada",
            "Luz LED Ring"
        ])
        layout.addRow("Tipo de Iluminación:", self.lighting_type_combo)
        
        # Calibration date
        self.calibration_date_edit = QDateEdit()
        self.calibration_date_edit.setProperty("class", "nist-date")
        self.calibration_date_edit.setDate(QDate.currentDate())
        self.calibration_date_edit.setCalendarPopup(True)
        layout.addRow("Fecha de Calibración:", self.calibration_date_edit)
        
        # Scale factor
        self.scale_factor_edit = QLineEdit()
        self.scale_factor_edit.setPlaceholderText("Ej: 0.5 μm/pixel")
        self.scale_factor_edit.setProperty("class", "nist-input")
        layout.addRow("Factor de Escala:", self.scale_factor_edit)
        
        return group
        
    def create_validation_section(self) -> QWidget:
        """Create validation options section"""
        group = QGroupBox("Opciones de Validación NIST")
        group.setProperty("class", "nist-section")
        
        layout = QVBoxLayout(group)
        layout.setSpacing(8)
        
        # Quality validation
        self.quality_validation_cb = QCheckBox("Validación de calidad de imagen NIST")
        self.quality_validation_cb.setProperty("class", "nist-option")
        self.quality_validation_cb.setChecked(True)
        layout.addWidget(self.quality_validation_cb)
        
        # Metadata validation
        self.metadata_validation_cb = QCheckBox("Validación de metadatos NIST")
        self.metadata_validation_cb.setProperty("class", "nist-option")
        layout.addWidget(self.metadata_validation_cb)
        
        # Authenticity validation
        self.authenticity_cb = QCheckBox("Validación de autenticidad")
        self.authenticity_cb.setProperty("class", "nist-option")
        layout.addWidget(self.authenticity_cb)
        
        # Compression analysis
        self.compression_cb = QCheckBox("Análisis de compresión")
        self.compression_cb.setProperty("class", "nist-option")
        layout.addWidget(self.compression_cb)
        
        # Chain of custody (for comparison mode)
        if self.mode == "comparison":
            self.chain_custody_cb = QCheckBox("Verificación de cadena de custodia")
            self.chain_custody_cb.setProperty("class", "nist-option")
            layout.addWidget(self.chain_custody_cb)
        
        return group
        
    def create_advanced_section(self) -> QWidget:
        """Create advanced options section (collapsible)"""
        advanced_section = CollapsibleSection("Opciones Avanzadas NIST")
        
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setSpacing(15)
        
        # Export options
        export_group = QGroupBox("Opciones de Exportación")
        export_group.setProperty("class", "nist-subsection")
        export_layout = QVBoxLayout(export_group)
        
        self.export_xml_cb = QCheckBox("Exportar metadatos en formato XML")
        self.export_xml_cb.setProperty("class", "nist-option")
        export_layout.addWidget(self.export_xml_cb)
        
        self.export_json_cb = QCheckBox("Exportar metadatos en formato JSON")
        self.export_json_cb.setProperty("class", "nist-option")
        export_layout.addWidget(self.export_json_cb)
        
        layout.addWidget(export_group)
        
        # Compliance options
        compliance_group = QGroupBox("Cumplimiento Normativo")
        compliance_group.setProperty("class", "nist-subsection")
        compliance_layout = QVBoxLayout(compliance_group)
        
        self.iso_compliance_cb = QCheckBox("Cumplimiento ISO/IEC 17025")
        self.iso_compliance_cb.setProperty("class", "nist-option")
        compliance_layout.addWidget(self.iso_compliance_cb)
        
        self.astm_compliance_cb = QCheckBox("Cumplimiento ASTM E2927")
        self.astm_compliance_cb.setProperty("class", "nist-option")
        compliance_layout.addWidget(self.astm_compliance_cb)
        
        layout.addWidget(compliance_group)
        
        advanced_section.add_content_widget(content)
        return advanced_section
        
    def connect_signals(self):
        """Connect widget signals"""
        self.enable_nist_cb.toggled.connect(self.main_panel.setEnabled)
        self.enable_nist_cb.toggled.connect(self.emit_configuration_changed)
        
        # Connect all input widgets
        widgets = [
            self.lab_name_edit, self.lab_accreditation_edit, self.examiner_edit,
            self.case_number_edit, self.capture_device_edit, self.magnification_edit,
            self.scale_factor_edit, self.quality_validation_cb, self.metadata_validation_cb,
            self.authenticity_cb, self.compression_cb
        ]
        
        if self.mode == "comparison":
            widgets.append(self.chain_custody_cb)
            
        for widget in widgets:
            if hasattr(widget, 'textChanged'):
                widget.textChanged.connect(self.emit_configuration_changed)
            elif hasattr(widget, 'toggled'):
                widget.toggled.connect(self.emit_configuration_changed)
            elif hasattr(widget, 'currentTextChanged'):
                widget.currentTextChanged.connect(self.emit_configuration_changed)
            elif hasattr(widget, 'dateChanged'):
                widget.dateChanged.connect(self.emit_configuration_changed)
                
    def emit_configuration_changed(self):
        """Emit configuration changed signal"""
        self.config = self.get_configuration()
        self.configurationChanged.emit(self.config)
        
    def get_configuration(self) -> Dict[str, Any]:
        """Get current NIST configuration"""
        config = {
            'enabled': self.enable_nist_cb.isChecked(),
            'lab_name': self.lab_name_edit.text(),
            'lab_accreditation': self.lab_accreditation_edit.text(),
            'examiner': self.examiner_edit.text(),
            'case_number': self.case_number_edit.text(),
            'capture_device': self.capture_device_edit.text(),
            'magnification': self.magnification_edit.text(),
            'lighting_type': self.lighting_type_combo.currentText(),
            'calibration_date': self.calibration_date_edit.date().toString('yyyy-MM-dd'),
            'scale_factor': self.scale_factor_edit.text(),
            'quality_validation': self.quality_validation_cb.isChecked(),
            'metadata_validation': self.metadata_validation_cb.isChecked(),
            'authenticity_validation': self.authenticity_cb.isChecked(),
            'compression_analysis': self.compression_cb.isChecked(),
        }
        
        if self.mode == "comparison":
            config['chain_custody'] = self.chain_custody_cb.isChecked()
            
        return config
        
    def set_configuration(self, config: Dict[str, Any]):
        """Set NIST configuration"""
        self.enable_nist_cb.setChecked(config.get('enabled', False))
        self.lab_name_edit.setText(config.get('lab_name', ''))
        self.lab_accreditation_edit.setText(config.get('lab_accreditation', ''))
        self.examiner_edit.setText(config.get('examiner', ''))
        self.case_number_edit.setText(config.get('case_number', ''))
        self.capture_device_edit.setText(config.get('capture_device', ''))
        self.magnification_edit.setText(config.get('magnification', ''))
        
        lighting_type = config.get('lighting_type', '')
        if lighting_type:
            index = self.lighting_type_combo.findText(lighting_type)
            if index >= 0:
                self.lighting_type_combo.setCurrentIndex(index)
                
        self.scale_factor_edit.setText(config.get('scale_factor', ''))
        self.quality_validation_cb.setChecked(config.get('quality_validation', True))
        self.metadata_validation_cb.setChecked(config.get('metadata_validation', False))
        self.authenticity_cb.setChecked(config.get('authenticity_validation', False))
        self.compression_cb.setChecked(config.get('compression_analysis', False))
        
        if self.mode == "comparison":
            self.chain_custody_cb.setChecked(config.get('chain_custody', False))
            
    def is_enabled(self) -> bool:
        """Check if NIST configuration is enabled"""
        return self.enable_nist_cb.isChecked()
        
    def validate_configuration(self) -> tuple[bool, list]:
        """Validate current configuration"""
        errors = []
        
        if not self.is_enabled():
            return True, []
            
        # Required fields validation
        if not self.lab_name_edit.text().strip():
            errors.append("Nombre del laboratorio es requerido")
            
        if not self.examiner_edit.text().strip():
            errors.append("Nombre del examinador es requerido")
            
        if not self.case_number_edit.text().strip():
            errors.append("Número de caso es requerido")
            
        if not self.capture_device_edit.text().strip():
            errors.append("Dispositivo de captura es requerido")
            
        return len(errors) == 0, errors
    
    def set_mode(self, mode: str):
        """Set the widget mode (for compatibility)"""
        self.mode = mode
        # Update UI elements based on mode if needed
        if hasattr(self, 'enable_nist_cb'):
            self.enable_nist_cb.setText(
                "Incluir metadatos en formato NIST" + 
                (" para comparación balística" if mode == "comparison" else " para análisis balístico")
            )