"""
AFTEConfigurationWidget - Shared AFTE configuration component
Eliminates code duplication between analysis and comparison tabs
"""

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
                             QGroupBox, QCheckBox, QRadioButton, QSlider,
                             QLabel, QFrame, QButtonGroup, QSpinBox,
                             QDoubleSpinBox, QComboBox, QTextEdit)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont
from typing import Dict, Any


class AFTEConfigurationWidget(QWidget):
    """Shared AFTE (Association of Firearm and Tool Mark Examiners) configuration widget"""
    
    configurationChanged = pyqtSignal(dict)  # Emitted when configuration changes
    
    def __init__(self, mode: str = "analysis", parent=None):
        """
        Initialize AFTE configuration widget
        
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
        self.enable_afte_cb = QCheckBox(
            "Generar conclusiones AFTE automáticas" + 
            (" para comparación" if self.mode == "comparison" else " para análisis")
        )
        self.enable_afte_cb.setProperty("class", "afte-enable-checkbox")
        layout.addWidget(self.enable_afte_cb)
        
        # Main configuration panel
        self.main_panel = QFrame()
        self.main_panel.setProperty("class", "afte-config-panel")
        self.main_panel.setEnabled(False)
        
        main_layout = QVBoxLayout(self.main_panel)
        main_layout.setContentsMargins(20, 15, 20, 15)
        main_layout.setSpacing(20)
        
        # CMC Configuration Section
        cmc_section = self.create_cmc_section()
        main_layout.addWidget(cmc_section)
        
        # AFTE Criteria Section
        criteria_section = self.create_criteria_section()
        main_layout.addWidget(criteria_section)
        
        # Confidence Level Section
        confidence_section = self.create_confidence_section()
        main_layout.addWidget(confidence_section)
        
        # Advanced Options Section
        advanced_section = self.create_advanced_section()
        main_layout.addWidget(advanced_section)
        
        layout.addWidget(self.main_panel)
        
    def create_cmc_section(self) -> QWidget:
        """Create CMC (Congruent Matching Cells) configuration section"""
        group = QGroupBox("Configuración CMC (Congruent Matching Cells)")
        group.setProperty("class", "afte-section")
        
        layout = QVBoxLayout(group)
        layout.setSpacing(15)
        
        # CMC Method selection
        method_layout = QFormLayout()
        self.cmc_method_combo = QComboBox()
        self.cmc_method_combo.setProperty("class", "afte-combo")
        self.cmc_method_combo.addItems([
            "CMC Estándar - Análisis de celdas congruentes",
            "CMC Avanzado - Con análisis de patrones",
            "CMC Híbrido - Combinado con características individuales",
            "CMC Automático - Selección inteligente de método"
        ])
        method_layout.addRow("Método CMC:", self.cmc_method_combo)
        layout.addLayout(method_layout)
        
        # CMC Threshold
        threshold_frame = QFrame()
        threshold_layout = QVBoxLayout(threshold_frame)
        
        threshold_label = QLabel("Umbral de Correlación CMC")
        threshold_label.setProperty("class", "afte-label")
        threshold_layout.addWidget(threshold_label)
        
        # Slider with value display
        slider_layout = QHBoxLayout()
        self.cmc_threshold_slider = QSlider(Qt.Horizontal)
        self.cmc_threshold_slider.setProperty("class", "afte-slider")
        self.cmc_threshold_slider.setRange(50, 95)
        self.cmc_threshold_slider.setValue(75)
        
        self.cmc_threshold_label = QLabel("0.75")
        self.cmc_threshold_label.setProperty("class", "afte-value-label")
        self.cmc_threshold_label.setMinimumWidth(40)
        
        slider_layout.addWidget(self.cmc_threshold_slider)
        slider_layout.addWidget(self.cmc_threshold_label)
        
        threshold_layout.addLayout(slider_layout)
        
        # Threshold description
        desc_label = QLabel("Valor recomendado: 0.70-0.80 para análisis estándar")
        desc_label.setProperty("class", "afte-description")
        threshold_layout.addWidget(desc_label)
        
        layout.addWidget(threshold_frame)
        
        return group
        
    def create_criteria_section(self) -> QWidget:
        """Create AFTE criteria section"""
        group = QGroupBox("Criterios de Conclusión AFTE")
        group.setProperty("class", "afte-section")
        
        layout = QVBoxLayout(group)
        layout.setSpacing(10)
        
        # Create button group for radio buttons
        self.afte_criteria_group = QButtonGroup()
        
        # Identification
        self.identification_rb = QRadioButton("Identification (≥85% CMC)")
        self.identification_rb.setProperty("class", "afte-radio identification")
        self.afte_criteria_group.addButton(self.identification_rb, 0)
        layout.addWidget(self.identification_rb)
        
        id_desc = QLabel("Suficiente acuerdo de características individuales para identificación positiva")
        id_desc.setProperty("class", "afte-radio-description")
        layout.addWidget(id_desc)
        
        # Inconclusive
        self.inconclusive_rb = QRadioButton("Inconclusive (70-84% CMC)")
        self.inconclusive_rb.setProperty("class", "afte-radio inconclusive")
        self.afte_criteria_group.addButton(self.inconclusive_rb, 1)
        layout.addWidget(self.inconclusive_rb)
        
        inc_desc = QLabel("Algunas características individuales presentes, pero insuficientes para conclusión")
        inc_desc.setProperty("class", "afte-radio-description")
        layout.addWidget(inc_desc)
        
        # Elimination
        self.elimination_rb = QRadioButton("Elimination (<70% CMC)")
        self.elimination_rb.setProperty("class", "afte-radio elimination")
        self.afte_criteria_group.addButton(self.elimination_rb, 2)
        layout.addWidget(self.elimination_rb)
        
        elim_desc = QLabel("Diferencias significativas que excluyen origen común")
        elim_desc.setProperty("class", "afte-radio-description")
        layout.addWidget(elim_desc)
        
        # Automatic determination
        self.auto_determination_rb = QRadioButton("Determinación Automática")
        self.auto_determination_rb.setProperty("class", "afte-radio auto")
        self.auto_determination_rb.setChecked(True)
        self.afte_criteria_group.addButton(self.auto_determination_rb, 3)
        layout.addWidget(self.auto_determination_rb)
        
        auto_desc = QLabel("El sistema determina automáticamente basado en el umbral CMC configurado")
        auto_desc.setProperty("class", "afte-radio-description")
        layout.addWidget(auto_desc)
        
        return group
        
    def create_confidence_section(self) -> QWidget:
        """Create confidence level section"""
        group = QGroupBox("Nivel de Confianza")
        group.setProperty("class", "afte-section")
        
        layout = QVBoxLayout(group)
        layout.setSpacing(15)
        
        # Enable confidence calculation
        self.calculate_confidence_cb = QCheckBox("Calcular nivel de confianza estadístico")
        self.calculate_confidence_cb.setProperty("class", "afte-option")
        self.calculate_confidence_cb.setChecked(True)
        layout.addWidget(self.calculate_confidence_cb)
        
        # Confidence parameters
        confidence_frame = QFrame()
        confidence_frame.setProperty("class", "afte-confidence-frame")
        confidence_layout = QFormLayout(confidence_frame)
        
        # Minimum confidence level
        self.min_confidence_spin = QDoubleSpinBox()
        self.min_confidence_spin.setProperty("class", "afte-spin")
        self.min_confidence_spin.setRange(0.50, 0.99)
        self.min_confidence_spin.setSingleStep(0.05)
        self.min_confidence_spin.setValue(0.85)
        self.min_confidence_spin.setSuffix("%")
        confidence_layout.addRow("Confianza Mínima:", self.min_confidence_spin)
        
        # Statistical method
        self.stat_method_combo = QComboBox()
        self.stat_method_combo.setProperty("class", "afte-combo")
        self.stat_method_combo.addItems([
            "Bayesiano - Análisis de probabilidad",
            "Frecuentista - Análisis estadístico clásico",
            "Bootstrap - Remuestreo estadístico",
            "Monte Carlo - Simulación estadística"
        ])
        confidence_layout.addRow("Método Estadístico:", self.stat_method_combo)
        
        layout.addWidget(confidence_frame)
        
        return group
        
    def create_advanced_section(self) -> QWidget:
        """Create advanced options section"""
        group = QGroupBox("Opciones Avanzadas AFTE")
        group.setProperty("class", "afte-section")
        
        layout = QVBoxLayout(group)
        layout.setSpacing(12)
        
        # Database comparison
        self.database_comparison_cb = QCheckBox("Comparación con base de datos de casos similares")
        self.database_comparison_cb.setProperty("class", "afte-option")
        layout.addWidget(self.database_comparison_cb)
        
        # Quality assessment
        self.quality_assessment_cb = QCheckBox("Evaluación de calidad de características")
        self.quality_assessment_cb.setProperty("class", "afte-option")
        self.quality_assessment_cb.setChecked(True)
        layout.addWidget(self.quality_assessment_cb)
        
        # Reproducibility test
        self.reproducibility_cb = QCheckBox("Prueba de reproducibilidad")
        self.reproducibility_cb.setProperty("class", "afte-option")
        layout.addWidget(self.reproducibility_cb)
        
        # Generate detailed report
        self.detailed_report_cb = QCheckBox("Generar reporte detallado AFTE")
        self.detailed_report_cb.setProperty("class", "afte-option")
        self.detailed_report_cb.setChecked(True)
        layout.addWidget(self.detailed_report_cb)
        
        # Custom notes
        notes_label = QLabel("Notas Adicionales:")
        notes_label.setProperty("class", "afte-label")
        layout.addWidget(notes_label)
        
        self.custom_notes_edit = QTextEdit()
        self.custom_notes_edit.setProperty("class", "afte-notes")
        self.custom_notes_edit.setMaximumHeight(80)
        self.custom_notes_edit.setPlaceholderText("Observaciones adicionales para el análisis AFTE...")
        layout.addWidget(self.custom_notes_edit)
        
        return group
        
    def connect_signals(self):
        """Connect widget signals"""
        self.enable_afte_cb.toggled.connect(self.main_panel.setEnabled)
        self.enable_afte_cb.toggled.connect(self.emit_configuration_changed)
        
        # CMC threshold slider
        self.cmc_threshold_slider.valueChanged.connect(self.update_threshold_label)
        self.cmc_threshold_slider.valueChanged.connect(self.emit_configuration_changed)
        
        # All other widgets
        widgets = [
            self.cmc_method_combo, self.calculate_confidence_cb, self.min_confidence_spin,
            self.stat_method_combo, self.database_comparison_cb, self.quality_assessment_cb,
            self.reproducibility_cb, self.detailed_report_cb, self.custom_notes_edit
        ]
        
        for widget in widgets:
            if hasattr(widget, 'currentTextChanged'):
                widget.currentTextChanged.connect(self.emit_configuration_changed)
            elif hasattr(widget, 'toggled'):
                widget.toggled.connect(self.emit_configuration_changed)
            elif hasattr(widget, 'valueChanged'):
                widget.valueChanged.connect(self.emit_configuration_changed)
            elif hasattr(widget, 'textChanged'):
                widget.textChanged.connect(self.emit_configuration_changed)
                
        # Radio button group
        self.afte_criteria_group.buttonClicked.connect(self.emit_configuration_changed)
        
    def update_threshold_label(self, value):
        """Update threshold label when slider changes"""
        threshold = value / 100.0
        self.cmc_threshold_label.setText(f"{threshold:.2f}")
        
    def emit_configuration_changed(self):
        """Emit configuration changed signal"""
        self.config = self.get_configuration()
        self.configurationChanged.emit(self.config)
        
    def get_configuration(self) -> Dict[str, Any]:
        """Get current AFTE configuration"""
        config = {
            'enabled': self.enable_afte_cb.isChecked(),
            'cmc_method': self.cmc_method_combo.currentText(),
            'cmc_threshold': self.cmc_threshold_slider.value() / 100.0,
            'criteria_mode': self.afte_criteria_group.checkedId(),
            'calculate_confidence': self.calculate_confidence_cb.isChecked(),
            'min_confidence': self.min_confidence_spin.value(),
            'statistical_method': self.stat_method_combo.currentText(),
            'database_comparison': self.database_comparison_cb.isChecked(),
            'quality_assessment': self.quality_assessment_cb.isChecked(),
            'reproducibility_test': self.reproducibility_cb.isChecked(),
            'detailed_report': self.detailed_report_cb.isChecked(),
            'custom_notes': self.custom_notes_edit.toPlainText()
        }
        
        return config
        
    def set_configuration(self, config: Dict[str, Any]):
        """Set AFTE configuration"""
        self.enable_afte_cb.setChecked(config.get('enabled', False))
        
        # CMC method
        cmc_method = config.get('cmc_method', '')
        if cmc_method:
            index = self.cmc_method_combo.findText(cmc_method)
            if index >= 0:
                self.cmc_method_combo.setCurrentIndex(index)
                
        # CMC threshold
        threshold = config.get('cmc_threshold', 0.75)
        self.cmc_threshold_slider.setValue(int(threshold * 100))
        
        # Criteria mode
        criteria_mode = config.get('criteria_mode', 3)  # Default to automatic
        if 0 <= criteria_mode < self.afte_criteria_group.buttons().__len__():
            self.afte_criteria_group.button(criteria_mode).setChecked(True)
            
        # Confidence settings
        self.calculate_confidence_cb.setChecked(config.get('calculate_confidence', True))
        self.min_confidence_spin.setValue(config.get('min_confidence', 0.85))
        
        # Statistical method
        stat_method = config.get('statistical_method', '')
        if stat_method:
            index = self.stat_method_combo.findText(stat_method)
            if index >= 0:
                self.stat_method_combo.setCurrentIndex(index)
                
        # Advanced options
        self.database_comparison_cb.setChecked(config.get('database_comparison', False))
        self.quality_assessment_cb.setChecked(config.get('quality_assessment', True))
        self.reproducibility_cb.setChecked(config.get('reproducibility_test', False))
        self.detailed_report_cb.setChecked(config.get('detailed_report', True))
        self.custom_notes_edit.setPlainText(config.get('custom_notes', ''))
        
    def is_enabled(self) -> bool:
        """Check if AFTE configuration is enabled"""
        return self.enable_afte_cb.isChecked()
        
    def get_cmc_threshold(self) -> float:
        """Get current CMC threshold value"""
        return self.cmc_threshold_slider.value() / 100.0
        
    def get_conclusion_criteria(self) -> str:
        """Get current conclusion criteria"""
        checked_id = self.afte_criteria_group.checkedId()
        criteria_map = {
            0: "identification",
            1: "inconclusive", 
            2: "elimination",
            3: "automatic"
        }
        return criteria_map.get(checked_id, "automatic")
        
    def validate_configuration(self) -> tuple[bool, list]:
        """Validate current configuration"""
        errors = []
        
        if not self.is_enabled():
            return True, []
            
        # Validate threshold range
        threshold = self.get_cmc_threshold()
        if threshold < 0.5 or threshold > 0.95:
            errors.append("Umbral CMC debe estar entre 0.50 y 0.95")
            
        # Validate confidence level
        if self.calculate_confidence_cb.isChecked():
            min_conf = self.min_confidence_spin.value()
            if min_conf < 0.5:
                errors.append("Confianza mínima debe ser al menos 50%")
                
        return len(errors) == 0, errors
    
    def set_mode(self, mode: str):
        """Set the widget mode (for compatibility)"""
        self.mode = mode
        # Update UI elements based on mode if needed
        if hasattr(self, 'enable_afte_cb'):
            self.enable_afte_cb.setText(
                "Generar conclusiones AFTE automáticas" + 
                (" para comparación" if mode == "comparison" else " para análisis")
            )