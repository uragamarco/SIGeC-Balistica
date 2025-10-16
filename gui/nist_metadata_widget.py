#!/usr/bin/env python3
"""
Widget de Metadatos NIST Integrado
Sistema SIGeC-Balistica - Análisis de Cartuchos y Balas Automático

Widget completo para la captura de metadatos según especificaciones NIST
integrado con el gestor de estado de la aplicación.
"""

import os
import json
from datetime import datetime
from typing import Dict, Any, Optional
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout,
    QPushButton, QLabel, QComboBox, QLineEdit, QTextEdit,
    QScrollArea, QGroupBox, QCheckBox, QSpinBox, QDoubleSpinBox,
    QFrame, QSizePolicy, QMessageBox, QFileDialog, QSplitter
)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QPalette, QPixmap, QIcon

try:
    from .app_state_manager import AppStateManager
    STATE_MANAGER_AVAILABLE = True
except ImportError:
    STATE_MANAGER_AVAILABLE = False
    print("Warning: AppStateManager not available")

# Cargar tema moderno si está disponible
try:
    from .styles import apply_modern_qss_to_widget
except Exception:
    def apply_modern_qss_to_widget(widget):
        return False

class CollapsibleGroupBox(QGroupBox):
    """GroupBox colapsable para organizar secciones"""
    
    def __init__(self, title="", parent=None):
        super().__init__(title, parent)
        self.setCheckable(True)
        self.setChecked(True)
        self.toggled.connect(self.on_toggled)
        
    def on_toggled(self, checked):
        """Maneja el colapso/expansión del grupo"""
        for child in self.findChildren(QWidget):
            if child != self:
                child.setVisible(checked)

class NISTMetadataWidget(QWidget):
    """Widget principal para metadatos balísticos según especificaciones NIST"""
    
    # Señales
    metadataChanged = pyqtSignal()
    validationChanged = pyqtSignal(bool)
    exportRequested = pyqtSignal(dict)
    fieldChanged = pyqtSignal(str, str)  # field_name, value
    
    def __init__(self, parent=None, state_manager=None):
        super().__init__(parent)
        self.fields = {}
        self.required_fields = set()
        self.validation_timer = QTimer()
        self.validation_timer.setSingleShot(True)
        self.validation_timer.timeout.connect(self._validate_fields)
        
        # Referencia al gestor de estado si está disponible
        self.state_manager = state_manager
        if self.state_manager is None and STATE_MANAGER_AVAILABLE:
            try:
                self.state_manager = AppStateManager()
            except Exception as e:
                print(f"Warning: Could not initialize AppStateManager: {e}")
        
        self.setup_ui()
        # Aplicar hoja de estilos moderna (si está disponible)
        try:
            apply_modern_qss_to_widget(self)
        except Exception:
            pass
        self.connect_signals()
        self.auto_fill_defaults()
        
    def setup_ui(self):
        """Configura la interfaz principal"""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(15)
        
        # Título principal
        title_label = QLabel("📋 Metadatos Balísticos NIST")
        title_label.setObjectName("main_title")
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)
        
        # Splitter para dividir en dos columnas
        splitter = QSplitter(Qt.Horizontal)
        
        # Columna izquierda - Información básica
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setSpacing(15)
        
        # Columna derecha - Información técnica
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setSpacing(15)
        
        # Crear secciones en columna izquierda
        self.create_study_section(left_layout)
        self.create_creator_section(left_layout)
        self.create_case_section(left_layout)
        
        # Crear secciones en columna derecha
        self.create_firearm_section(right_layout)
        self.create_bullet_section(right_layout)
        self.create_casing_section(right_layout)
        self.create_image_section(right_layout)
        
        # Añadir stretch a ambas columnas
        left_layout.addStretch()
        right_layout.addStretch()
        
        # Configurar splitter
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setSizes([400, 400])
        
        main_layout.addWidget(splitter)
        
        # Botones de acción
        self.create_action_buttons(main_layout)
        
    def create_study_section(self, parent_layout):
        """Crea la sección de Estudio"""
        group = CollapsibleGroupBox("📚 Información del Estudio")
        layout = QFormLayout(group)
        layout.setSpacing(8)
        
        # StudyID (auto-generado, editable por el usuario)
        self.fields['study_id'] = QLineEdit()
        self.fields['study_id'].setPlaceholderText("Generado automáticamente")
        # Permitir completar/editar manualmente el StudyID
        self.fields['study_id'].setEnabled(True)
        layout.addRow("StudyID:", self.fields['study_id'])
        
        # StudyName (requerido)
        self.fields['study_name'] = QLineEdit()
        self.fields['study_name'].setPlaceholderText("Nombre descriptivo del estudio")
        self.required_fields.add('study_name')
        layout.addRow("StudyName *:", self.fields['study_name'])
        
        # Description
        self.fields['description'] = QTextEdit()
        self.fields['description'].setMaximumHeight(60)
        self.fields['description'].setPlaceholderText("Descripción del estudio...")
        layout.addRow("Descripción:", self.fields['description'])
        
        parent_layout.addWidget(group)
        
    def create_creator_section(self, parent_layout):
        """Crea la sección de Creador/Perito"""
        group = CollapsibleGroupBox("👤 Información del Perito")
        layout = QFormLayout(group)
        layout.setSpacing(8)
        
        # FirstName (requerido)
        self.fields['first_name'] = QLineEdit()
        self.fields['first_name'].setPlaceholderText("Nombre del perito")
        self.required_fields.add('first_name')
        layout.addRow("Nombre *:", self.fields['first_name'])
        
        # LastName (requerido)
        self.fields['last_name'] = QLineEdit()
        self.fields['last_name'].setPlaceholderText("Apellido del perito")
        self.required_fields.add('last_name')
        layout.addRow("Apellido *:", self.fields['last_name'])
        
        # Organization (requerido)
        self.fields['organization'] = QLineEdit()
        self.fields['organization'].setPlaceholderText("Institución o laboratorio")
        self.required_fields.add('organization')
        layout.addRow("Organización *:", self.fields['organization'])
        
        parent_layout.addWidget(group)
        
    def create_case_section(self, parent_layout):
        """Crea la sección de Caso"""
        group = CollapsibleGroupBox("📁 Información del Caso")
        layout = QFormLayout(group)
        layout.setSpacing(8)
        
        # Case Number (requerido)
        self.fields['case_number'] = QLineEdit()
        self.fields['case_number'].setPlaceholderText("Número de caso")
        self.required_fields.add('case_number')
        layout.addRow("Número de Caso *:", self.fields['case_number'])
        
        # Evidence ID (requerido)
        self.fields['evidence_id'] = QLineEdit()
        self.fields['evidence_id'].setPlaceholderText("Identificador de evidencia")
        self.required_fields.add('evidence_id')
        layout.addRow("ID Evidencia *:", self.fields['evidence_id'])
        
        # Laboratory
        self.fields['laboratory'] = QLineEdit()
        self.fields['laboratory'].setPlaceholderText("Laboratorio de análisis")
        layout.addRow("Laboratorio:", self.fields['laboratory'])
        
        parent_layout.addWidget(group)
        
    def create_firearm_section(self, parent_layout):
        """Crea la sección de Arma de Fuego"""
        group = CollapsibleGroupBox("🔫 Información del Arma")
        layout = QFormLayout(group)
        layout.setSpacing(8)

        # Tipo de Arma (seleccionable)
        self.fields['firearm_type'] = QComboBox()
        self.fields['firearm_type'].setEditable(True)
        self.fields['firearm_type'].addItems([
            "", "Fusil", "Carabina", "Pistola", "Revólver", "Escopeta", "Otros"
        ])
        layout.addRow("Tipo de Arma:", self.fields['firearm_type'])

        # FirearmName (requerido)
        self.fields['firearm_name'] = QLineEdit()
        self.fields['firearm_name'].setPlaceholderText("Nombre o modelo del arma")
        self.required_fields.add('firearm_name')
        layout.addRow("Nombre del Arma *:", self.fields['firearm_name'])

        # Marca (requerido)
        self.fields['firearm_brand'] = QComboBox()
        self.fields['firearm_brand'].setEditable(True)
        firearm_brands = [
            "", "Beretta", "Bersa", "Browning", "Canik", "Colt", 
            "CZ-USA", "FN Herstal", "Glock", "Heckler & Koch", 
            "Kimber", "Remington", "Ruger", "Sig Sauer", 
            "Smith & Wesson", "Springfield", "Taurus", "Walther", "Otro"
        ]
        self.fields['firearm_brand'].addItems(firearm_brands)
        self.required_fields.add('firearm_brand')
        layout.addRow("Marca *:", self.fields['firearm_brand'])

        # Modelo (requerido)
        self.fields['firearm_model'] = QLineEdit()
        self.fields['firearm_model'].setPlaceholderText("Modelo específico")
        self.required_fields.add('firearm_model')
        layout.addRow("Modelo *:", self.fields['firearm_model'])

        # Calibre (requerido)
        self.fields['caliber'] = QComboBox()
        self.fields['caliber'].setEditable(True)
        calibers = [
            "", "22LR", "25 Auto", "32 Auto", "9 mm", "38/357", "357 Sig",
            "380 Auto", "40/10 mm", "44 Spl/Mag", "45 Auto", "Otro"
        ]
        self.fields['caliber'].addItems(calibers)
        self.required_fields.add('caliber')
        layout.addRow("Calibre *:", self.fields['caliber'])

        # Longitud de Cañón
        self.fields['barrel_length'] = QDoubleSpinBox()
        self.fields['barrel_length'].setRange(0.0, 2000.0)
        self.fields['barrel_length'].setSuffix(" mm")
        self.fields['barrel_length'].setDecimals(1)
        layout.addRow("Longitud de Cañón:", self.fields['barrel_length'])

        # Número de Serie
        self.fields['serial_number'] = QLineEdit()
        self.fields['serial_number'].setPlaceholderText("Número de serie (si disponible)")
        layout.addRow("Número de Serie:", self.fields['serial_number'])

        # Clase de Percutor
        self.fields['firing_pin_class'] = QComboBox()
        self.fields['firing_pin_class'].setEditable(True)
        firing_pin_classes = [
            "", "Hemispherical", "Glock Type", "Truncated Cone", 
            "Rectangular", "Other", "Not Specified"
        ]
        self.fields['firing_pin_class'].addItems(firing_pin_classes)
        layout.addRow("Clase de Percutor:", self.fields['firing_pin_class'])

        # Clase de Culote
        self.fields['breech_face_class'] = QComboBox()
        self.fields['breech_face_class'].setEditable(True)
        breech_face_classes = [
            "", "Arched", "Circular", "Cross Hatch", "Granular", 
            "Smooth", "Striated", "Other", "Not Specified"
        ]
        self.fields['breech_face_class'].addItems(breech_face_classes)
        layout.addRow("Clase de Culote:", self.fields['breech_face_class'])

        parent_layout.addWidget(group)
        
    def create_bullet_section(self, parent_layout):
        """Crea la sección de Proyectil"""
        group = CollapsibleGroupBox("🎯 Información del Proyectil")
        layout = QFormLayout(group)
        layout.setSpacing(8)
        
        # SpecimenName (requerido)
        self.fields['bullet_specimen_name'] = QLineEdit()
        self.fields['bullet_specimen_name'].setPlaceholderText("Identificador del espécimen")
        self.required_fields.add('bullet_specimen_name')
        layout.addRow("Nombre del Espécimen *:", self.fields['bullet_specimen_name'])
        
        # Brand
        self.fields['bullet_brand'] = QComboBox()
        self.fields['bullet_brand'].setEditable(True)
        bullet_brands = [
            "", "Federal", "Winchester", "Remington", "CCI", "Hornady",
            "Speer", "Blazer", "PMC", "Fiocchi", "Sellier & Bellot", "Otro"
        ]
        self.fields['bullet_brand'].addItems(bullet_brands)
        layout.addRow("Marca de Munición:", self.fields['bullet_brand'])
        
        # Weight
        self.fields['bullet_weight'] = QDoubleSpinBox()
        self.fields['bullet_weight'].setRange(0.0, 1000.0)
        self.fields['bullet_weight'].setSuffix(" gr")
        self.fields['bullet_weight'].setDecimals(1)
        layout.addRow("Peso:", self.fields['bullet_weight'])
        
        # Type
        self.fields['bullet_type'] = QComboBox()
        self.fields['bullet_type'].setEditable(True)
        bullet_types = [
            "", "FMJ", "HP", "SP", "SJHP", "JHP", "LRN", "SWC", "WC", "Otro"
        ]
        self.fields['bullet_type'].addItems(bullet_types)
        layout.addRow("Tipo de Proyectil:", self.fields['bullet_type'])

        parent_layout.addWidget(group)

    def create_casing_section(self, parent_layout):
        """Crea la sección de Vaina (casquillo)"""
        group = CollapsibleGroupBox("🧩 Información de la Vaina")
        layout = QFormLayout(group)
        layout.setSpacing(8)

        # Identificador del espécimen de vaina
        self.fields['casing_specimen_name'] = QLineEdit()
        self.fields['casing_specimen_name'].setPlaceholderText("Identificador del espécimen de vaina")
        layout.addRow("Nombre del Espécimen:", self.fields['casing_specimen_name'])

        # Tipo de cartucho
        self.fields['cartridge_type'] = QComboBox()
        self.fields['cartridge_type'].setEditable(True)
        self.fields['cartridge_type'].addItems([
            "", "Percusión en Borde (Rimfire)", "Percusión Central (Centerfire)", "Cartucho de Escopeta", "Otro"
        ])
        layout.addRow("Tipo de Cartucho:", self.fields['cartridge_type'])

        # Material de la vaina
        self.fields['casing_material'] = QComboBox()
        self.fields['casing_material'].setEditable(True)
        self.fields['casing_material'].addItems([
            "", "Latón", "Níquelado", "Acero", "Aluminio", "Plástico", "Otro"
        ])
        layout.addRow("Material de la Vaina:", self.fields['casing_material'])

        # Marcaje en culote (headstamp)
        self.fields['headstamp_text'] = QLineEdit()
        self.fields['headstamp_text'].setPlaceholderText("Texto del marcaje en culote (headstamp)")
        layout.addRow("Marcaje en Culote (Headstamp):", self.fields['headstamp_text'])

        # Estado de la vaina
        self.fields['casing_condition'] = QComboBox()
        self.fields['casing_condition'].setEditable(True)
        self.fields['casing_condition'].addItems([
            "", "Integra", "Deformada", "Marcada", "Quemada", "Otro"
        ])
        layout.addRow("Estado de la Vaina:", self.fields['casing_condition'])

        parent_layout.addWidget(group)
        
    def create_image_section(self, parent_layout):
        """Crea la sección de Imagen"""
        group = CollapsibleGroupBox("📷 Información de Imagen")
        layout = QFormLayout(group)
        layout.setSpacing(8)
        
        # Resolution
        self.fields['resolution'] = QLineEdit()
        self.fields['resolution'].setPlaceholderText("ej: 1920x1080")
        layout.addRow("Resolución:", self.fields['resolution'])
        
        # Magnification
        self.fields['magnification'] = QDoubleSpinBox()
        self.fields['magnification'].setRange(1.0, 1000.0)
        self.fields['magnification'].setSuffix("x")
        self.fields['magnification'].setDecimals(1)
        layout.addRow("Magnificación:", self.fields['magnification'])
        
        # Lighting
        self.fields['lighting'] = QComboBox()
        self.fields['lighting'].setEditable(True)
        lighting_types = [
            "", "Coaxial", "Ring Light", "Oblique", "Transmitted", "Otro"
        ]
        self.fields['lighting'].addItems(lighting_types)
        layout.addRow("Iluminación:", self.fields['lighting'])
        
        # Capture Date
        self.fields['capture_date'] = QLineEdit()
        self.fields['capture_date'].setPlaceholderText("YYYY-MM-DD HH:MM:SS")
        layout.addRow("Fecha de Captura:", self.fields['capture_date'])
        
        parent_layout.addWidget(group)
        
    def create_action_buttons(self, parent_layout):
        """Crea los botones de acción"""
        button_frame = QFrame()
        button_layout = QHBoxLayout(button_frame)
        button_layout.setSpacing(10)
        
        # Botón de validar
        self.validate_btn = QPushButton("✅ Validar Metadatos")
        self.validate_btn.setObjectName("primary_button")
        self.validate_btn.clicked.connect(self.validate_metadata)
        
        # Botón de limpiar
        self.clear_btn = QPushButton("🗑️ Limpiar")
        self.clear_btn.setObjectName("secondary_button")
        self.clear_btn.clicked.connect(self.clear_metadata)
        
        # Botón de exportar
        self.export_btn = QPushButton("💾 Exportar JSON")
        self.export_btn.setObjectName("secondary_button")
        self.export_btn.clicked.connect(self.export_metadata)
        
        # Botón de importar
        self.import_btn = QPushButton("📂 Importar JSON")
        self.import_btn.setObjectName("secondary_button")
        self.import_btn.clicked.connect(self.import_metadata)
        
        button_layout.addWidget(self.validate_btn)
        button_layout.addStretch()
        button_layout.addWidget(self.clear_btn)
        button_layout.addWidget(self.export_btn)
        button_layout.addWidget(self.import_btn)
        
        parent_layout.addWidget(button_frame)
        
    def setup_styles(self):
        """Configura los estilos del widget"""
        self.setStyleSheet("""
            QWidget {
                background-color: #2b2b2b;
                color: #ffffff;
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 11px;
            }
            
            #main_title {
                font-size: 16px;
                font-weight: bold;
                color: #4a90e2;
                padding: 10px;
                border-bottom: 2px solid #4a90e2;
                margin-bottom: 15px;
            }
            
            QGroupBox {
                font-weight: bold;
                border: 2px solid #555555;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
                background-color: #333333;
            }
            
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 8px 0 8px;
                color: #4a90e2;
                font-size: 12px;
            }
            
            QLineEdit, QTextEdit, QComboBox, QSpinBox, QDoubleSpinBox {
                background-color: #404040;
                border: 1px solid #666666;
                border-radius: 4px;
                padding: 6px;
                color: #ffffff;
                selection-background-color: #4a90e2;
            }
            
            QLineEdit:focus, QTextEdit:focus, QComboBox:focus {
                border-color: #4a90e2;
                background-color: #454545;
            }
            
            QComboBox::drop-down {
                border: none;
                width: 20px;
            }
            
            QComboBox::down-arrow {
                image: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAiIGhlaWdodD0iNiIgdmlld0JveD0iMCAwIDEwIDYiIGZpbGw9Im5vbmUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CjxwYXRoIGQ9Ik01IDZMMCAwSDEwTDUgNloiIGZpbGw9IiNmZmZmZmYiLz4KPC9zdmc+);
            }
            
            QPushButton {
                background-color: #555555;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                color: #ffffff;
                font-weight: bold;
                font-size: 11px;
            }
            
            QPushButton:hover {
                background-color: #666666;
            }
            
            QPushButton:pressed {
                background-color: #444444;
            }
            
            #primary_button {
                background-color: #4a90e2;
            }
            
            #primary_button:hover {
                background-color: #357abd;
            }
            
            #secondary_button {
                background-color: #666666;
            }
            
            QFormLayout QLabel {
                font-weight: bold;
                min-width: 120px;
                color: #cccccc;
            }
            
            QSplitter::handle {
                background-color: #555555;
                width: 2px;
            }
        """)
        
    def connect_signals(self):
        """Conecta las señales de los campos"""
        for field_name, field in self.fields.items():
            if isinstance(field, QLineEdit):
                field.textChanged.connect(lambda text, name=field_name: self._on_field_changed(name, text))
            elif isinstance(field, QComboBox):
                # Detectar cambios tanto por selección como por edición manual
                field.currentTextChanged.connect(lambda text, name=field_name: self._on_field_changed(name, text))
                try:
                    # Solo si es editable, capturar edición en vivo
                    if field.isEditable():
                        field.editTextChanged.connect(lambda text, name=field_name: self._on_field_changed(name, text))
                except Exception:
                    pass
            elif isinstance(field, QTextEdit):
                field.textChanged.connect(lambda name=field_name: self._on_field_changed(name, field.toPlainText()))
            elif isinstance(field, (QSpinBox, QDoubleSpinBox)):
                field.valueChanged.connect(lambda value, name=field_name: self._on_field_changed(name, str(value)))
            elif isinstance(field, QCheckBox):
                field.toggled.connect(lambda checked, name=field_name: self._on_field_changed(name, str(checked)))
                
    def auto_fill_defaults(self):
        """Llena automáticamente algunos campos con valores por defecto"""
        # Auto-generar IDs
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.fields['study_id'].setText(f"STUDY_{timestamp}")
        
        # Fecha de captura actual
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.fields['capture_date'].setText(current_datetime)
        
        # Valores por defecto comunes
        self.fields['magnification'].setValue(10.0)
        self.fields['lighting'].setCurrentText("Coaxial")
        
    def _on_field_changed(self, field_name: str, value: str):
        """Maneja cambios en los campos"""
        self.fieldChanged.emit(field_name, value)
        self.metadataChanged.emit()
        
        # Actualizar gestor de estado si está disponible
        if self.state_manager:
            metadata = self.get_metadata()
            self.state_manager.update_metadata(metadata)
        
        # Usar timer para evitar validaciones excesivas
        self.validation_timer.start(500)
        
    def _validate_fields(self):
        """Valida los campos requeridos"""
        all_valid = True
        
        for field_name in self.required_fields:
            if field_name in self.fields:
                field = self.fields[field_name]
                is_valid = False
                
                if isinstance(field, QLineEdit):
                    is_valid = bool(field.text().strip())
                elif isinstance(field, QComboBox):
                    is_valid = bool(field.currentText().strip())
                elif isinstance(field, QTextEdit):
                    is_valid = bool(field.toPlainText().strip())
                    
                # Aplicar estilo visual según validación
                current_style = field.styleSheet()
                if is_valid:
                    # Remover estilo de error si existe
                    field.setStyleSheet(current_style.replace("border-color: #ff6b6b;", ""))
                else:
                    # Añadir estilo de error
                    if "border-color: #ff6b6b;" not in current_style:
                        field.setStyleSheet(current_style + "border-color: #ff6b6b;")
                    all_valid = False
                    
        self.validationChanged.emit(all_valid)
        
    def get_metadata(self) -> Dict[str, Any]:
        """Obtiene todos los metadatos como diccionario"""
        metadata = {}
        
        for field_name, field in self.fields.items():
            if isinstance(field, QLineEdit):
                metadata[field_name] = field.text()
            elif isinstance(field, QComboBox):
                metadata[field_name] = field.currentText()
            elif isinstance(field, QTextEdit):
                metadata[field_name] = field.toPlainText()
            elif isinstance(field, (QSpinBox, QDoubleSpinBox)):
                metadata[field_name] = field.value()
            elif isinstance(field, QCheckBox):
                metadata[field_name] = field.isChecked()
                
        return metadata
        
    def set_metadata(self, metadata: Dict[str, Any]):
        """Establece los metadatos desde un diccionario"""
        for field_name, value in metadata.items():
            if field_name in self.fields:
                field = self.fields[field_name]
                
                if isinstance(field, QLineEdit):
                    field.setText(str(value))
                elif isinstance(field, QComboBox):
                    field.setCurrentText(str(value))
                elif isinstance(field, QTextEdit):
                    field.setPlainText(str(value))
                elif isinstance(field, (QSpinBox, QDoubleSpinBox)):
                    field.setValue(float(value) if value else 0.0)
                elif isinstance(field, QCheckBox):
                    field.setChecked(bool(value))

        # Emitir cambios y actualizar gestor de estado compartido
        try:
            self.metadataChanged.emit()
            if self.state_manager:
                self.state_manager.update_metadata(self.get_metadata())
        except Exception as e:
            print(f"Warning emitting metadataChanged after set_metadata: {e}")
                    
    def validate_metadata(self):
        """Valida manualmente los metadatos"""
        self._validate_fields()
        
        # Mostrar resultado de validación
        metadata = self.get_metadata()
        missing_fields = []
        
        for field_name in self.required_fields:
            if field_name not in metadata or not str(metadata[field_name]).strip():
                missing_fields.append(field_name)
                
        if missing_fields:
            QMessageBox.warning(
                self, 
                "Validación de Metadatos",
                f"Campos requeridos faltantes:\n• " + "\n• ".join(missing_fields)
            )
        else:
            QMessageBox.information(
                self,
                "Validación de Metadatos",
                "✅ Todos los campos requeridos están completos"
            )
            
    def clear_metadata(self):
        """Limpia todos los campos"""
        reply = QMessageBox.question(
            self,
            "Limpiar Metadatos",
            "¿Está seguro de que desea limpiar todos los campos?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            for field in self.fields.values():
                if isinstance(field, QLineEdit):
                    field.clear()
                elif isinstance(field, QComboBox):
                    field.setCurrentIndex(0)
                elif isinstance(field, QTextEdit):
                    field.clear()
                elif isinstance(field, (QSpinBox, QDoubleSpinBox)):
                    field.setValue(0)
                elif isinstance(field, QCheckBox):
                    field.setChecked(False)
                    
            # Restaurar valores por defecto
            self.auto_fill_defaults()

            # Notificar cambio y sincronizar estado
            try:
                self.metadataChanged.emit()
                if self.state_manager:
                    self.state_manager.update_metadata(self.get_metadata())
            except Exception as e:
                print(f"Warning updating state after clear: {e}")
            
    def export_metadata(self):
        """Exporta los metadatos a un archivo JSON"""
        metadata = self.get_metadata()
        
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Exportar Metadatos NIST",
            f"metadatos_nist_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            "JSON Files (*.json)"
        )
        
        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False)
                    
                QMessageBox.information(
                    self,
                    "Exportación Exitosa",
                    f"Metadatos exportados a:\n{filename}"
                )
                
                self.exportRequested.emit(metadata)
                
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Error de Exportación",
                    f"Error al exportar metadatos:\n{str(e)}"
                )
                
    def import_metadata(self):
        """Importa metadatos desde un archivo JSON"""
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Importar Metadatos NIST",
            "",
            "JSON Files (*.json)"
        )
        
        if filename:
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    
                self.set_metadata(metadata)
                
                QMessageBox.information(
                    self,
                    "Importación Exitosa",
                    f"Metadatos importados desde:\n{filename}"
                )
                
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Error de Importación",
                    f"Error al importar metadatos:\n{str(e)}"
                )