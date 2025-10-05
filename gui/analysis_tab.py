#!/usr/bin/env python3
"""
Pestaña de Análisis Balístico Individual
Flujo guiado paso a paso: Cargar evidencia → Datos del caso → Metadatos NIST → Configurar análisis → Procesar
"""

import os
import json
from typing import Optional, Dict, Any
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout,
    QLabel, QPushButton, QLineEdit, QTextEdit, QComboBox, QSpinBox,
    QCheckBox, QGroupBox, QScrollArea, QSplitter, QFrame, QSpacerItem,
    QSizePolicy, QFileDialog, QMessageBox, QProgressBar, QSlider, QDoubleSpinBox
)
from PyQt5.QtCore import Qt, pyqtSignal, QThread, QTimer
from PyQt5.QtGui import QFont, QPixmap

from .shared_widgets import (
    ImageDropZone, ResultCard, CollapsiblePanel, StepIndicator, 
    ProgressCard, ImageViewer
)
from .model_selector_dialog import ModelSelectorDialog

# Importaciones condicionales para Deep Learning
try:
    from deep_learning.models import BallisticCNN, SiameseNetwork
    from deep_learning.config.experiment_config import ModelConfig
    from deep_learning.ballistic_dl_models import ModelType
    DEEP_LEARNING_AVAILABLE = True
except ImportError:
    DEEP_LEARNING_AVAILABLE = False

class AnalysisWorker(QThread):
    """Worker thread para realizar análisis sin bloquear la UI"""
    
    progressUpdated = pyqtSignal(int, str)
    analysisCompleted = pyqtSignal(dict)
    analysisError = pyqtSignal(str)
    
    def __init__(self, analysis_params: dict):
        super().__init__()
        self.analysis_params = analysis_params
        
    def run(self):
        """Ejecuta el análisis balístico en segundo plano"""
        try:
            evidence_type = self.analysis_params.get('evidence_type', 'cartridge_case')
            
            # Pasos específicos según tipo de evidencia
            if evidence_type == 'cartridge_case':
                steps = [
                    (10, "Cargando imagen de cartucho..."),
                    (20, "Detectando marca de percutor..."),
                    (35, "Analizando cara de recámara..."),
                    (50, "Extrayendo marcas de extractor..."),
                    (65, "Calculando características balísticas..."),
                    (80, "Aplicando estándares NIST..."),
                    (95, "Generando conclusiones AFTE..."),
                    (100, "Análisis de cartucho completado")
                ]
            elif evidence_type == 'bullet':
                steps = [
                    (10, "Cargando imagen de proyectil..."),
                    (25, "Detectando estrías..."),
                    (40, "Analizando patrones de marcas..."),
                    (60, "Extrayendo características de superficie..."),
                    (75, "Calculando métricas balísticas..."),
                    (90, "Aplicando validación NIST..."),
                    (100, "Análisis de proyectil completado")
                ]
            else:  # projectile
                steps = [
                    (10, "Cargando evidencia balística..."),
                    (30, "Analizando morfología..."),
                    (50, "Extrayendo características..."),
                    (70, "Calculando métricas forenses..."),
                    (90, "Validando con estándares..."),
                    (100, "Análisis completado")
                ]
            
            results = {}
            
            for progress, message in steps:
                self.progressUpdated.emit(progress, message)
                self.msleep(500)  # Simular trabajo
                
                # Aquí iría la integración real con el backend balístico
                if progress == 100:
                    results = {
                        'image_path': self.analysis_params.get('image_path'),
                        'evidence_type': evidence_type,
                        'case_data': self.analysis_params.get('case_data', {}),
                        'nist_metadata': self.analysis_params.get('nist_metadata', {}),
                        'ballistic_config': self.analysis_params.get('ballistic_config', {}),
                        'ballistic_features': self._generate_ballistic_features(evidence_type),
                        'nist_compliance': {
                            'quality_score': 0.87,
                            'measurement_uncertainty': 0.05,
                            'traceability': 'NIST-compliant',
                            'validation_status': 'Passed'
                        },
                        'afte_conclusion': self._generate_afte_conclusion(),
                        'visualizations': {
                            'roi_detection': 'path/to/roi_overlay.png',
                            'feature_map': 'path/to/features.png',
                            'comparison_grid': 'path/to/comparison.png'
                        }
                    }
                    
            self.analysisCompleted.emit(results)
            
        except Exception as e:
            self.analysisError.emit(str(e))
    
    def _generate_ballistic_features(self, evidence_type: str) -> dict:
        """Genera características balísticas simuladas según el tipo de evidencia"""
        if evidence_type == 'cartridge_case':
            return {
                'firing_pin': {
                    'diameter': 1.2,
                    'depth': 0.08,
                    'eccentricity': 0.15,
                    'circularity': 0.92
                },
                'breech_face': {
                    'roughness': 2.3,
                    'orientation': 45.2,
                    'periodicity': 0.85,
                    'entropy': 7.2
                },
                'extractor_marks': {
                    'count': 2,
                    'length': 3.4,
                    'depth': 0.05,
                    'angle': 78.5
                }
            }
        elif evidence_type == 'bullet':
            return {
                'striation_patterns': {
                    'num_lines': 24,
                    'dominant_directions': [12.5, 167.8],
                    'parallelism_score': 0.89,
                    'density': 8.2
                },
                'land_groove': {
                    'width_ratio': 1.15,
                    'depth': 0.12,
                    'twist_rate': 1.0
                }
            }
        else:
            return {
                'general_characteristics': {
                    'surface_roughness': 1.8,
                    'morphology_score': 0.76
                }
            }
    
    def _generate_afte_conclusion(self) -> dict:
        """Genera conclusión AFTE simulada"""
        return {
            'conclusion_level': 'Inconclusive',
            'confidence': 0.65,
            'reasoning': 'Algunas características individuales en acuerdo, pero insuficientes para identificación',
            'examiner_notes': 'Se requiere análisis adicional con más muestras de comparación'
        }

class AnalysisTab(QWidget):
    """Pestaña de análisis individual con flujo guiado"""
    
    analysisCompleted = pyqtSignal(dict)
    
    def __init__(self):
        super().__init__()
        self.current_step = 0
        self.analysis_data = {}
        self.analysis_worker = None
        self.setup_ui()
        self.setup_connections()
        
    def setup_ui(self):
        """Configura la interfaz de la pestaña"""
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)
        
        # Panel izquierdo - Flujo de trabajo
        self.setup_workflow_panel()
        main_layout.addWidget(self.workflow_panel, 2)
        
        # Panel derecho - Visualización y resultados
        self.setup_results_panel()
        main_layout.addWidget(self.results_panel, 1)
        
    def setup_workflow_panel(self):
        """Configura el panel de flujo de trabajo"""
        self.workflow_panel = QFrame()
        self.workflow_panel.setProperty("class", "panel")
        
        layout = QVBoxLayout(self.workflow_panel)
        layout.setSpacing(20)
        
        # Título
        title_label = QLabel("Análisis Individual")
        title_label.setProperty("class", "title")
        layout.addWidget(title_label)
        
        # Indicador de pasos
        steps = ["Cargar Imagen", "Datos del Caso", "Metadatos NIST", "Configurar", "Procesar"]
        self.step_indicator = StepIndicator(steps)
        layout.addWidget(self.step_indicator)
        
        # Área de contenido con scroll
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setSpacing(20)
        
        # Paso 1: Cargar Imagen
        self.setup_step1_load_image()
        
        # Paso 2: Datos del Caso
        self.setup_step2_case_data()
        
        # Paso 3: Metadatos NIST (Opcional)
        self.setup_step3_nist_metadata()
        
        # Paso 4: Configuración de Procesamiento
        self.setup_step4_processing_config()
        
        # Paso 5: Procesar
        self.setup_step5_process()
        
        scroll_area.setWidget(self.content_widget)
        layout.addWidget(scroll_area)
        
        # Botones de navegación
        self.setup_navigation_buttons()
        layout.addWidget(self.navigation_frame)
        
    def setup_step1_load_image(self):
        """Paso 1: Cargar imagen"""
        self.step1_group = QGroupBox("Paso 1: Cargar Imagen")
        self.step1_group.setProperty("class", "step-group")
        
        layout = QVBoxLayout(self.step1_group)
        
        # Drop zone para imagen
        self.image_drop_zone = ImageDropZone(
            "Arrastrar imagen aquí",
            "Formatos soportados: PNG, JPG, JPEG, BMP, TIFF"
        )
        layout.addWidget(self.image_drop_zone)
        
        # Información de la imagen cargada
        self.image_info_frame = QFrame()
        self.image_info_frame.setProperty("class", "info-panel")
        self.image_info_frame.hide()
        
        info_layout = QFormLayout(self.image_info_frame)
        
        self.image_name_label = QLabel()
        self.image_size_label = QLabel()
        self.image_dimensions_label = QLabel()
        self.image_format_label = QLabel()
        
        info_layout.addRow("Archivo:", self.image_name_label)
        info_layout.addRow("Tamaño:", self.image_size_label)
        info_layout.addRow("Dimensiones:", self.image_dimensions_label)
        info_layout.addRow("Formato:", self.image_format_label)
        
        layout.addWidget(self.image_info_frame)
        
        self.content_layout.addWidget(self.step1_group)
        
    def setup_step2_case_data(self):
        """Paso 2: Datos del caso balístico"""
        self.step2_group = QGroupBox("Paso 2: Datos del Caso Balístico")
        self.step2_group.setProperty("class", "step-group")
        self.step2_group.setEnabled(False)
        
        layout = QVBoxLayout(self.step2_group)
        
        # Información básica del caso
        basic_info_group = QGroupBox("Información Básica")
        basic_layout = QGridLayout(basic_info_group)
        
        # Campos obligatorios
        self.case_number_edit = QLineEdit()
        self.case_number_edit.setPlaceholderText("Ej: BAL-2024-001")
        basic_layout.addWidget(QLabel("Número de Caso:*"), 0, 0)
        basic_layout.addWidget(self.case_number_edit, 0, 1)
        
        self.evidence_id_edit = QLineEdit()
        self.evidence_id_edit.setPlaceholderText("Ej: EV-001-CART")
        basic_layout.addWidget(QLabel("ID de Evidencia:*"), 1, 0)
        basic_layout.addWidget(self.evidence_id_edit, 1, 1)
        
        # Tipo de evidencia balística
        self.evidence_type_combo = QComboBox()
        self.evidence_type_combo.addItems([
            "Seleccionar tipo...",
            "Casquillo/Vaina (Cartridge Case)",
            "Bala/Proyectil (Bullet)",
            "Proyectil General"
        ])
        basic_layout.addWidget(QLabel("Tipo de Evidencia:*"), 2, 0)
        basic_layout.addWidget(self.evidence_type_combo, 2, 1)
        
        self.examiner_edit = QLineEdit()
        self.examiner_edit.setPlaceholderText("Nombre del perito balístico")
        basic_layout.addWidget(QLabel("Examinador:*"), 3, 0)
        basic_layout.addWidget(self.examiner_edit, 3, 1)
        
        layout.addWidget(basic_info_group)
        
        # Información del arma (opcional)
        weapon_info_group = QGroupBox("Información del Arma (Opcional)")
        weapon_layout = QGridLayout(weapon_info_group)
        
        self.weapon_make_edit = QLineEdit()
        self.weapon_make_edit.setPlaceholderText("Ej: Glock, Smith & Wesson, Beretta")
        weapon_layout.addWidget(QLabel("Marca del Arma:"), 0, 0)
        weapon_layout.addWidget(self.weapon_make_edit, 0, 1)
        
        self.weapon_model_edit = QLineEdit()
        self.weapon_model_edit.setPlaceholderText("Ej: 17, M&P Shield, 92FS")
        weapon_layout.addWidget(QLabel("Modelo:"), 1, 0)
        weapon_layout.addWidget(self.weapon_model_edit, 1, 1)
        
        self.caliber_edit = QLineEdit()
        self.caliber_edit.setPlaceholderText("Ej: 9mm Luger, .40 S&W, .45 ACP")
        weapon_layout.addWidget(QLabel("Calibre:"), 2, 0)
        weapon_layout.addWidget(self.caliber_edit, 2, 1)
        
        self.serial_number_edit = QLineEdit()
        self.serial_number_edit.setPlaceholderText("Número de serie del arma")
        weapon_layout.addWidget(QLabel("Número de Serie:"), 3, 0)
        weapon_layout.addWidget(self.serial_number_edit, 3, 1)
        
        layout.addWidget(weapon_info_group)
        
        # Información adicional
        additional_info_group = QGroupBox("Información Adicional")
        additional_layout = QVBoxLayout(additional_info_group)
        
        self.case_description_edit = QTextEdit()
        self.case_description_edit.setPlaceholderText("Descripción del caso, circunstancias del hallazgo, etc.")
        self.case_description_edit.setMaximumHeight(80)
        additional_layout.addWidget(QLabel("Descripción del Caso:"))
        additional_layout.addWidget(self.case_description_edit)
        
        layout.addWidget(additional_info_group)
        
        self.content_layout.addWidget(self.step2_group)
        
    def setup_step3_nist_metadata(self):
        """Paso 3: Metadatos NIST para Evidencia Balística (Opcional)"""
        self.step3_group = QGroupBox("Paso 3: Metadatos NIST Balísticos (Opcional)")
        self.step3_group.setProperty("class", "step-group")
        self.step3_group.setEnabled(False)
        
        layout = QVBoxLayout(self.step3_group)
        
        # Checkbox para habilitar metadatos NIST
        self.enable_nist_checkbox = QCheckBox("Incluir metadatos en formato NIST para evidencia balística")
        layout.addWidget(self.enable_nist_checkbox)
        
        # Panel de metadatos NIST (colapsable)
        self.nist_panel = CollapsiblePanel("Configuración de Metadatos NIST Balísticos")
        
        nist_form = QFormLayout()
        
        # Información del laboratorio
        self.lab_name_edit = QLineEdit()
        self.lab_name_edit.setPlaceholderText("Nombre del laboratorio forense")
        nist_form.addRow("Laboratorio:", self.lab_name_edit)
        
        self.lab_accreditation_edit = QLineEdit()
        self.lab_accreditation_edit.setPlaceholderText("Número de acreditación")
        nist_form.addRow("Acreditación:", self.lab_accreditation_edit)
        
        # Información del equipo de captura
        self.capture_device_edit = QLineEdit()
        self.capture_device_edit.setPlaceholderText("Ej: Microscopio de comparación Leica FSC")
        nist_form.addRow("Dispositivo de Captura:", self.capture_device_edit)
        
        self.magnification_edit = QLineEdit()
        self.magnification_edit.setPlaceholderText("Ej: 40x, 100x")
        nist_form.addRow("Magnificación:", self.magnification_edit)
        
        # Condiciones de iluminación
        self.lighting_type_combo = QComboBox()
        self.lighting_type_combo.addItems([
            "Seleccionar...",
            "Luz Blanca Coaxial",
            "Luz Oblicua",
            "Luz Polarizada",
            "Luz LED Ring"
        ])
        nist_form.addRow("Tipo de Iluminación:", self.lighting_type_combo)
        
        # Información de calibración
        self.calibration_date_edit = QLineEdit()
        self.calibration_date_edit.setPlaceholderText("YYYY-MM-DD")
        nist_form.addRow("Fecha de Calibración:", self.calibration_date_edit)
        
        self.scale_factor_edit = QLineEdit()
        self.scale_factor_edit.setPlaceholderText("Ej: 0.5 μm/pixel")
        nist_form.addRow("Factor de Escala:", self.scale_factor_edit)
        
        # Crear un widget contenedor para el layout
        nist_widget = QWidget()
        nist_widget.setLayout(nist_form)
        self.nist_panel.add_content_widget(nist_widget)
        layout.addWidget(self.nist_panel)
        
        self.content_layout.addWidget(self.step3_group)
        
    def setup_step4_processing_config(self):
        """Paso 4: Configuración de Análisis Balístico"""
        self.step4_group = QGroupBox("Paso 4: Configuración de Análisis Balístico")
        self.step4_group.setProperty("class", "step-group")
        self.step4_group.setEnabled(False)
        
        layout = QVBoxLayout(self.step4_group)
        
        # Configuración básica (siempre visible)
        basic_frame = QFrame()
        basic_frame.setProperty("class", "config-basic")
        basic_layout = QFormLayout(basic_frame)
        
        # Nivel de análisis balístico
        self.analysis_level_combo = QComboBox()
        self.analysis_level_combo.addItems([
            "Básico - Extracción de características principales",
            "Intermedio - Análisis detallado + métricas NIST",
            "Avanzado - Análisis completo + comparación automática",
            "Forense - Análisis exhaustivo + conclusiones AFTE"
        ])
        basic_layout.addRow("Nivel de Análisis:", self.analysis_level_combo)
        
        # Prioridad del procesamiento
        self.priority_combo = QComboBox()
        self.priority_combo.addItems([
            "Normal - Procesamiento estándar",
            "Alta - Procesamiento prioritario",
            "Crítica - Procesamiento inmediato"
        ])
        basic_layout.addRow("Prioridad:", self.priority_combo)
        
        layout.addWidget(basic_frame)
        
        # Opciones avanzadas (colapsables)
        self.advanced_panel = CollapsiblePanel("Opciones Avanzadas de Análisis Balístico")
        
        advanced_content = QWidget()
        advanced_layout = QVBoxLayout(advanced_content)
        
        # Características balísticas a extraer
        ballistic_features_group = QGroupBox("Características Balísticas")
        ballistic_features_layout = QVBoxLayout(ballistic_features_group)
        
        self.extract_firing_pin_cb = QCheckBox("Extracción de marcas de percutor (Firing Pin)")
        self.extract_breech_face_cb = QCheckBox("Análisis de cara de recámara (Breech Face)")
        self.extract_extractor_cb = QCheckBox("Marcas de extractor y eyector")
        self.extract_striations_cb = QCheckBox("Patrones de estriado (para balas)")
        self.extract_land_groove_cb = QCheckBox("Análisis de campos y estrías")
        
        ballistic_features_layout.addWidget(self.extract_firing_pin_cb)
        ballistic_features_layout.addWidget(self.extract_breech_face_cb)
        ballistic_features_layout.addWidget(self.extract_extractor_cb)
        ballistic_features_layout.addWidget(self.extract_striations_cb)
        ballistic_features_layout.addWidget(self.extract_land_groove_cb)
        
        advanced_layout.addWidget(ballistic_features_group)
        
        # Validación NIST
        nist_validation_group = QGroupBox("Validación NIST")
        nist_validation_layout = QVBoxLayout(nist_validation_group)
        
        self.nist_quality_check_cb = QCheckBox("Verificación de calidad de imagen")
        self.nist_authenticity_cb = QCheckBox("Validación de autenticidad")
        self.nist_compression_cb = QCheckBox("Análisis de compresión")
        self.nist_metadata_cb = QCheckBox("Validación de metadatos")
        
        nist_validation_layout.addWidget(self.nist_quality_check_cb)
        nist_validation_layout.addWidget(self.nist_authenticity_cb)
        nist_validation_layout.addWidget(self.nist_compression_cb)
        nist_validation_layout.addWidget(self.nist_metadata_cb)
        
        advanced_layout.addWidget(nist_validation_group)
        
        # Conclusiones AFTE
        afte_group = QGroupBox("Conclusiones AFTE")
        afte_layout = QVBoxLayout(afte_group)
        
        self.generate_afte_cb = QCheckBox("Generar conclusiones AFTE automáticas")
        self.afte_confidence_cb = QCheckBox("Calcular nivel de confianza")
        self.afte_comparison_cb = QCheckBox("Comparación con base de datos")
        
        afte_layout.addWidget(self.generate_afte_cb)
        afte_layout.addWidget(self.afte_confidence_cb)
        afte_layout.addWidget(self.afte_comparison_cb)
        
        advanced_layout.addWidget(afte_group)
        
        # Procesamiento de imagen balística
        image_processing_group = QGroupBox("Procesamiento de Imagen")
        image_processing_layout = QVBoxLayout(image_processing_group)
        
        self.noise_reduction_cb = QCheckBox("Reducción de ruido especializada")
        self.contrast_enhancement_cb = QCheckBox("Mejora de contraste para marcas")
        self.edge_detection_cb = QCheckBox("Detección de bordes de características")
        self.morphological_cb = QCheckBox("Operaciones morfológicas")
        
        image_processing_layout.addWidget(self.noise_reduction_cb)
        image_processing_layout.addWidget(self.contrast_enhancement_cb)
        image_processing_layout.addWidget(self.edge_detection_cb)
        image_processing_layout.addWidget(self.morphological_cb)
        
        advanced_layout.addWidget(image_processing_group)
        
        # Deep Learning (si está disponible)
        if DEEP_LEARNING_AVAILABLE:
            dl_group = QGroupBox("Modelos de Deep Learning")
            dl_group.setProperty("class", "dl-group")
            dl_layout = QVBoxLayout(dl_group)
            
            # Habilitar Deep Learning
            self.enable_dl_cb = QCheckBox("Usar modelos de Deep Learning")
            self.enable_dl_cb.setProperty("class", "dl-checkbox")
            self.enable_dl_cb.stateChanged.connect(self.toggle_dl_options)
            dl_layout.addWidget(self.enable_dl_cb)
            
            # Panel de opciones de DL (inicialmente deshabilitado)
            self.dl_options_frame = QFrame()
            dl_options_layout = QVBoxLayout(self.dl_options_frame)
            
            # Selector de modelo principal
            model_selection_layout = QFormLayout()
            
            self.dl_model_combo = QComboBox()
            self.dl_model_combo.setProperty("class", "dl-combo")
            self.dl_model_combo.addItems([
                "CNN - Extracción de características profundas",
                "Siamese - Comparación automática de pares",
                "U-Net - Segmentación de ROI (próximamente)",
                "Híbrido - Combinación de modelos"
            ])
            model_selection_layout.addRow("Modelo Principal:", self.dl_model_combo)
            
            # Configuración de CNN
            self.cnn_backbone_combo = QComboBox()
            self.cnn_backbone_combo.setProperty("class", "dl-combo")
            self.cnn_backbone_combo.addItems([
                "ResNet-18 (rápido)",
                "ResNet-50 (balanceado)",
                "EfficientNet-B0 (eficiente)",
                "Custom Ballistic CNN (especializado)"
            ])
            model_selection_layout.addRow("Arquitectura CNN:", self.cnn_backbone_combo)
            
            dl_options_layout.addLayout(model_selection_layout)
            
            # Configuración avanzada de DL
            dl_advanced_layout = QFormLayout()
            
            # Confianza mínima
            self.dl_confidence_slider = QSlider(Qt.Horizontal)
            self.dl_confidence_slider.setRange(50, 99)
            self.dl_confidence_slider.setValue(85)
            self.dl_confidence_label = QLabel("85%")
            self.dl_confidence_label.setProperty("class", "dl-label")
            self.dl_confidence_slider.valueChanged.connect(
                lambda v: self.dl_confidence_label.setText(f"{v}%")
            )
            
            confidence_layout = QHBoxLayout()
            confidence_layout.addWidget(self.dl_confidence_slider)
            confidence_layout.addWidget(self.dl_confidence_label)
            dl_advanced_layout.addRow("Confianza Mínima:", confidence_layout)
            
            # Batch size para procesamiento
            self.dl_batch_size_spin = QSpinBox()
            self.dl_batch_size_spin.setProperty("class", "dl-spin")
            self.dl_batch_size_spin.setRange(1, 32)
            self.dl_batch_size_spin.setValue(8)
            dl_advanced_layout.addRow("Batch Size:", self.dl_batch_size_spin)
            
            # Usar GPU si está disponible
            self.dl_use_gpu_cb = QCheckBox("Usar GPU (CUDA) si está disponible")
            self.dl_use_gpu_cb.setProperty("class", "dl-checkbox")
            self.dl_use_gpu_cb.setChecked(True)
            dl_advanced_layout.addRow("Aceleración:", self.dl_use_gpu_cb)
            
            dl_options_layout.addLayout(dl_advanced_layout)
            
            # Opciones específicas por modelo
            self.dl_model_specific_frame = QFrame()
            dl_model_specific_layout = QVBoxLayout(self.dl_model_specific_frame)
            
            # Opciones para Siamese Network
            self.siamese_options_frame = QFrame()
            siamese_layout = QFormLayout(self.siamese_options_frame)
            
            self.siamese_threshold_spin = QDoubleSpinBox()
            self.siamese_threshold_spin.setProperty("class", "dl-spin")
            self.siamese_threshold_spin.setRange(0.1, 1.0)
            self.siamese_threshold_spin.setSingleStep(0.05)
            self.siamese_threshold_spin.setValue(0.7)
            self.siamese_threshold_spin.setDecimals(2)
            siamese_layout.addRow("Umbral de Similitud:", self.siamese_threshold_spin)
            
            self.siamese_embedding_dim_spin = QSpinBox()
            self.siamese_embedding_dim_spin.setProperty("class", "dl-spin")
            self.siamese_embedding_dim_spin.setRange(128, 1024)
            self.siamese_embedding_dim_spin.setSingleStep(128)
            self.siamese_embedding_dim_spin.setValue(512)
            siamese_layout.addRow("Dimensión de Embedding:", self.siamese_embedding_dim_spin)
            
            dl_model_specific_layout.addWidget(self.siamese_options_frame)
            self.siamese_options_frame.hide()  # Ocultar inicialmente
            
            dl_options_layout.addWidget(self.dl_model_specific_frame)
            
            # Botón de configuración avanzada de DL
            self.dl_advanced_button = QPushButton("⚙️ Configuración Avanzada")
            self.dl_advanced_button.setProperty("class", "dl-advanced")
            self.dl_advanced_button.setEnabled(False)
            self.dl_advanced_button.clicked.connect(self.open_model_selector)
            dl_options_layout.addWidget(self.dl_advanced_button)
            
            # Conectar cambios de modelo
            self.dl_model_combo.currentTextChanged.connect(self.update_dl_model_options)
            
            self.dl_options_frame.setEnabled(False)
            dl_layout.addWidget(self.dl_options_frame)
            
            advanced_layout.addWidget(dl_group)
        
        # Crear un widget contenedor para el layout
        advanced_widget = QWidget()
        advanced_widget.setLayout(advanced_layout)
        self.advanced_panel.add_content_widget(advanced_widget)
        layout.addWidget(self.advanced_panel)
        
        # Resumen de configuración
        self.config_summary_frame = QFrame()
        self.config_summary_frame.setProperty("class", "config-summary")
        
        summary_layout = QVBoxLayout(self.config_summary_frame)
        summary_layout.addWidget(QLabel("Resumen de Configuración:"))
        
        self.config_summary_label = QLabel("Seleccione las opciones de análisis")
        self.config_summary_label.setWordWrap(True)
        summary_layout.addWidget(self.config_summary_label)
        
        layout.addWidget(self.config_summary_frame)
        
        self.content_layout.addWidget(self.step4_group)
        
    def setup_step5_process(self):
        """Paso 5: Procesar Análisis Balístico"""
        self.step5_group = QGroupBox("Paso 5: Procesar Análisis Balístico")
        self.step5_group.setProperty("class", "step-group")
        self.step5_group.setEnabled(False)
        
        layout = QVBoxLayout(self.step5_group)
        
        # Botones de acción
        buttons_layout = QHBoxLayout()
        
        self.process_button = QPushButton("🔍 Iniciar Análisis Balístico")
        self.process_button.setProperty("class", "primary-button")
        self.process_button.setMinimumHeight(50)
        buttons_layout.addWidget(self.process_button)
        
        self.save_config_button = QPushButton("💾 Guardar Configuración")
        self.save_config_button.setProperty("class", "secondary-button")
        buttons_layout.addWidget(self.save_config_button)
        
        layout.addLayout(buttons_layout)
        
        # Barra de progreso
        self.progress_frame = QFrame()
        self.progress_frame.setProperty("class", "progress-panel")
        self.progress_frame.hide()
        
        progress_layout = QVBoxLayout(self.progress_frame)
        
        self.progress_label = QLabel("Preparando análisis balístico...")
        progress_layout.addWidget(self.progress_label)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        progress_layout.addWidget(self.progress_bar)
        
        layout.addWidget(self.progress_frame)
        
        self.content_layout.addWidget(self.step5_group)
        
    def setup_step5_process(self):
        """Paso 5: Procesar"""
        self.step5_group = QGroupBox("Paso 5: Procesar y Guardar")
        self.step5_group.setProperty("class", "step-group")
        self.step5_group.setEnabled(False)
        
        layout = QVBoxLayout(self.step5_group)
        
        # Resumen de configuración
        self.config_summary_label = QLabel("Configuración lista para procesar")
        self.config_summary_label.setProperty("class", "body")
        layout.addWidget(self.config_summary_label)
        
        # Botones de acción
        buttons_layout = QHBoxLayout()
        
        self.process_button = QPushButton("🔄 Procesar Imagen")
        self.process_button.setProperty("class", "primary")
        self.process_button.setMinimumHeight(40)
        buttons_layout.addWidget(self.process_button)
        
        self.save_config_button = QPushButton("💾 Guardar Configuración")
        buttons_layout.addWidget(self.save_config_button)
        
        layout.addLayout(buttons_layout)
        
        # Progreso de análisis
        self.progress_card = ProgressCard("Análisis en Progreso")
        self.progress_card.hide()
        layout.addWidget(self.progress_card)
        
        self.content_layout.addWidget(self.step5_group)
        
    def setup_navigation_buttons(self):
        """Configura los botones de navegación"""
        self.navigation_frame = QFrame()
        layout = QHBoxLayout(self.navigation_frame)
        
        self.prev_button = QPushButton("← Anterior")
        self.prev_button.setEnabled(False)
        layout.addWidget(self.prev_button)
        
        layout.addStretch()
        
        self.next_button = QPushButton("Siguiente →")
        self.next_button.setEnabled(False)
        layout.addWidget(self.next_button)
        
        self.reset_button = QPushButton("🔄 Reiniciar")
        layout.addWidget(self.reset_button)
        
    def setup_results_panel(self):
        """Configura el panel de resultados y visualización"""
        self.results_panel = QFrame()
        self.results_panel.setProperty("class", "panel")
        
        layout = QVBoxLayout(self.results_panel)
        layout.setSpacing(15)
        
        # Título
        title_label = QLabel("Vista Previa y Resultados")
        title_label.setProperty("class", "subtitle")
        layout.addWidget(title_label)
        
        # Visor de imagen
        self.image_viewer = ImageViewer()
        layout.addWidget(self.image_viewer, 2)
        
        # Tarjetas de resultados
        self.results_scroll = QScrollArea()
        self.results_scroll.setWidgetResizable(True)
        self.results_scroll.setMaximumHeight(300)
        
        self.results_widget = QWidget()
        self.results_layout = QVBoxLayout(self.results_widget)
        
        # Placeholder para resultados
        placeholder_label = QLabel("Los resultados aparecerán aquí después del análisis")
        placeholder_label.setProperty("class", "caption")
        placeholder_label.setAlignment(Qt.AlignCenter)
        placeholder_label.setStyleSheet("color: #757575; padding: 40px;")
        self.results_layout.addWidget(placeholder_label)
        
        self.results_scroll.setWidget(self.results_widget)
        layout.addWidget(self.results_scroll, 1)
        
    def setup_connections(self):
        """Configura las conexiones de señales"""
        # Conexiones del flujo de trabajo
        self.image_drop_zone.imageLoaded.connect(self.on_image_loaded)
        self.case_number_edit.textChanged.connect(self.validate_step2)
        self.evidence_id_edit.textChanged.connect(self.validate_step2)
        self.evidence_type_combo.currentTextChanged.connect(self.validate_step2)
        self.examiner_edit.textChanged.connect(self.validate_step2)
        
        # Navegación
        self.next_button.clicked.connect(self.next_step)
        self.prev_button.clicked.connect(self.prev_step)
        self.reset_button.clicked.connect(self.reset_workflow)
        
        # Procesamiento
        self.process_button.clicked.connect(self.start_analysis)
        self.save_config_button.clicked.connect(self.save_configuration)
        
        # NIST metadata
        self.enable_nist_checkbox.toggled.connect(self.toggle_nist_panel)
        
        # Deep Learning connections (si está disponible)
        if DEEP_LEARNING_AVAILABLE:
            self.enable_dl_cb.stateChanged.connect(self.toggle_dl_options)
            self.dl_model_combo.currentTextChanged.connect(self.update_dl_model_options)
        
    def on_image_loaded(self, image_path: str):
        """Maneja la carga de imagen"""
        try:
            self.analysis_data['image_path'] = image_path
            
            # Mostrar información de la imagen
            file_info = os.path.basename(image_path)
            file_size = os.path.getsize(image_path) / (1024 * 1024)  # MB
            
            from PIL import Image
            with Image.open(image_path) as img:
                width, height = img.size
                format_name = img.format
                
            self.image_name_label.setText(file_info)
            self.image_size_label.setText(f"{file_size:.1f} MB")
            self.image_dimensions_label.setText(f"{width} x {height} px")
            self.image_format_label.setText(format_name)
            
            self.image_info_frame.show()
            
            # Cargar en el visor
            self.image_viewer.load_image(image_path)
            
            # Habilitar siguiente paso
            self.step2_group.setEnabled(True)
            self.next_button.setEnabled(True)
            self.update_step_indicator(0)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error cargando imagen: {str(e)}")
            
    def validate_step2(self):
        """Valida los datos del paso 2"""
        case_valid = bool(self.case_number_edit.text().strip())
        evidence_id_valid = bool(self.evidence_id_edit.text().strip())
        examiner_valid = bool(self.examiner_edit.text().strip())
        evidence_valid = self.evidence_type_combo.currentIndex() > 0
        
        if case_valid and evidence_id_valid and examiner_valid and evidence_valid:
            self.step3_group.setEnabled(True)
            if self.current_step == 1:
                self.next_button.setEnabled(True)
                
    def toggle_nist_panel(self, enabled: bool):
        """Alterna el panel de metadatos NIST"""
        self.nist_panel.set_expanded(enabled)
        
    def toggle_dl_options(self, state):
        """Alterna las opciones de Deep Learning con manejo robusto de errores"""
        try:
            if not DEEP_LEARNING_AVAILABLE:
                if state == 2:  # Si el usuario intenta habilitar DL
                    QMessageBox.warning(
                        self, "Deep Learning No Disponible",
                        "Los módulos de Deep Learning no están disponibles.\n"
                        "Verifique la instalación de las dependencias:\n\n"
                        "• torch\n"
                        "• torchvision\n"
                        "• tensorflow (opcional)\n\n"
                        "El análisis continuará con métodos tradicionales."
                    )
                    # Desmarcar el checkbox automáticamente
                    self.enable_dl_cb.setChecked(False)
                return
                
            enabled = state == 2  # Qt.Checked
            
            # Verificar que los widgets existen antes de usarlos
            if hasattr(self, 'dl_options_frame'):
                self.dl_options_frame.setEnabled(enabled)
            if hasattr(self, 'dl_advanced_button'):
                self.dl_advanced_button.setEnabled(enabled)
                
            # Actualizar indicador visual de estado
            if hasattr(self, 'dl_status_indicator'):
                if enabled:
                    self.dl_status_indicator.setProperty("class", "dl-status-active")
                    self.dl_status_indicator.setText("🟢 Deep Learning Activo")
                else:
                    self.dl_status_indicator.setProperty("class", "dl-status-inactive")
                    self.dl_status_indicator.setText("⚪ Deep Learning Inactivo")
                self.dl_status_indicator.style().unpolish(self.dl_status_indicator)
                self.dl_status_indicator.style().polish(self.dl_status_indicator)
                
        except Exception as e:
            QMessageBox.critical(
                self, "Error de Deep Learning",
                f"Error al configurar Deep Learning:\n{str(e)}\n\n"
                "El análisis continuará con métodos tradicionales."
            )
            # Asegurar que DL esté deshabilitado en caso de error
            if hasattr(self, 'enable_dl_cb'):
                self.enable_dl_cb.setChecked(False)
            
    def open_model_selector(self):
        """Abre el diálogo de selección de modelos de Deep Learning con manejo robusto de errores"""
        try:
            if not DEEP_LEARNING_AVAILABLE:
                QMessageBox.warning(
                    self, "Deep Learning No Disponible",
                    "Los módulos de Deep Learning no están disponibles.\n"
                    "Verifique la instalación de las dependencias:\n\n"
                    "• torch >= 1.9.0\n"
                    "• torchvision >= 0.10.0\n"
                    "• tensorflow >= 2.6.0 (opcional)\n"
                    "• scikit-learn >= 1.0.0\n\n"
                    "Para instalar las dependencias:\n"
                    "pip install torch torchvision tensorflow scikit-learn"
                )
                return
                
            # Verificar que el checkbox de DL esté habilitado
            if not hasattr(self, 'enable_dl_cb') or not self.enable_dl_cb.isChecked():
                QMessageBox.information(
                    self, "Deep Learning Deshabilitado",
                    "Primero debe habilitar el uso de Deep Learning\n"
                    "marcando la casilla correspondiente."
                )
                return
                
            # Obtener configuración actual
            current_config = self.get_current_dl_config()
            
            # Verificar que ModelSelectorDialog esté disponible
            try:
                from .model_selector_dialog import ModelSelectorDialog
            except ImportError as e:
                QMessageBox.critical(
                    self, "Error de Importación",
                    f"No se pudo cargar el diálogo de selección de modelos:\n{str(e)}\n\n"
                    "Verifique que todos los archivos estén presentes."
                )
                return
            
            # Crear y mostrar el diálogo
            dialog = ModelSelectorDialog(self, current_config)
            dialog.modelConfigured.connect(self.on_model_configured)
            
            # Manejar posibles errores al mostrar el diálogo
            try:
                dialog.exec_()
            except Exception as dialog_error:
                QMessageBox.critical(
                    self, "Error del Diálogo",
                    f"Error al mostrar el diálogo de configuración:\n{str(dialog_error)}"
                )
                
        except Exception as e:
            QMessageBox.critical(
                self, "Error Crítico",
                f"Error inesperado al abrir selector de modelos:\n{str(e)}\n\n"
                "Por favor, reporte este error al equipo de desarrollo."
            )
        
    def get_current_dl_config(self) -> dict:
        """Obtiene la configuración actual de Deep Learning con manejo robusto de errores"""
        try:
            # Verificar disponibilidad de Deep Learning
            if not DEEP_LEARNING_AVAILABLE:
                return {}
                
            # Verificar que el checkbox existe y está habilitado
            if not hasattr(self, 'enable_dl_cb') or not self.enable_dl_cb.isChecked():
                return {}
            
            config = {'enabled': True}
            
            # Obtener configuración básica con verificaciones
            try:
                if hasattr(self, 'dl_model_combo'):
                    model_text = self.dl_model_combo.currentText()
                    config['model_type'] = model_text.split(' - ')[0] if ' - ' in model_text else model_text
                else:
                    config['model_type'] = 'CNN'  # Valor por defecto
                    
                if hasattr(self, 'dl_confidence_slider'):
                    config['confidence_threshold'] = self.dl_confidence_slider.value() / 100.0
                else:
                    config['confidence_threshold'] = 0.85  # Valor por defecto
                    
                if hasattr(self, 'dl_batch_size_spin'):
                    config['batch_size'] = self.dl_batch_size_spin.value()
                else:
                    config['batch_size'] = 16  # Valor por defecto
                    
                if hasattr(self, 'dl_use_gpu_cb'):
                    config['use_gpu'] = self.dl_use_gpu_cb.isChecked()
                else:
                    config['use_gpu'] = False  # Valor por defecto
                    
            except Exception as e:
                print(f"Warning: Error obteniendo configuración básica de DL: {e}")
                # Usar valores por defecto en caso de error
                config.update({
                    'model_type': 'CNN',
                    'confidence_threshold': 0.85,
                    'batch_size': 16,
                    'use_gpu': False
                })
            
            # Agregar configuración específica del modelo con verificaciones
            try:
                if hasattr(self, 'cnn_backbone_combo'):
                    backbone_text = self.cnn_backbone_combo.currentText()
                    config['cnn_backbone'] = backbone_text.split(' (')[0] if ' (' in backbone_text else backbone_text
                    
                if hasattr(self, 'siamese_threshold_spin'):
                    config['siamese_threshold'] = self.siamese_threshold_spin.value()
                    
                if hasattr(self, 'siamese_embedding_dim_spin'):
                    config['embedding_dim'] = self.siamese_embedding_dim_spin.value()
                    
            except Exception as e:
                print(f"Warning: Error obteniendo configuración específica de modelo: {e}")
                # Los valores específicos del modelo son opcionales
                
            return config
            
        except Exception as e:
            print(f"Error crítico obteniendo configuración de DL: {e}")
            return {}  # Retornar configuración vacía en caso de error crítico
        
    def on_model_configured(self, config: dict):
        """Maneja la configuración del modelo desde el diálogo"""
        # Actualizar controles con la nueva configuración
        if 'model_type' in config:
            model_type = config['model_type']
            for i in range(self.dl_model_combo.count()):
                if self.dl_model_combo.itemText(i).startswith(model_type):
                    self.dl_model_combo.setCurrentIndex(i)
                    break
                    
        if 'confidence_threshold' in config:
            self.dl_confidence_slider.setValue(int(config['confidence_threshold'] * 100))
            
        if 'batch_size' in config:
            self.dl_batch_size_spin.setValue(config['batch_size'])
            
        if 'use_gpu' in config:
            self.dl_use_gpu_cb.setChecked(config['use_gpu'])
            
        # Guardar configuración avanzada para uso posterior
        self.advanced_dl_config = config
        
        # Actualizar resumen
        self.update_config_summary()
            
    def update_dl_model_options(self, model_text: str):
        """Actualiza las opciones específicas del modelo seleccionado"""
        if not DEEP_LEARNING_AVAILABLE:
            return
            
        # Ocultar todas las opciones específicas
        self.siamese_options_frame.hide()
        
        # Mostrar opciones según el modelo seleccionado
        if "Siamese" in model_text:
            self.siamese_options_frame.show()
        # Aquí se pueden agregar más opciones para otros modelos
        
    def next_step(self):
        """Avanza al siguiente paso"""
        if self.current_step < 4:
            self.current_step += 1
            self.update_step_indicator(self.current_step)
            self.update_navigation_buttons()
            
            # Habilitar pasos según progreso
            if self.current_step == 2:
                self.step3_group.setEnabled(True)
            elif self.current_step == 3:
                self.step4_group.setEnabled(True)
            elif self.current_step == 4:
                self.step5_group.setEnabled(True)
                self.update_config_summary()
                
    def prev_step(self):
        """Retrocede al paso anterior"""
        if self.current_step > 0:
            self.current_step -= 1
            self.update_step_indicator(self.current_step)
            self.update_navigation_buttons()
            
    def update_step_indicator(self, step: int):
        """Actualiza el indicador de pasos"""
        self.step_indicator.set_current_step(step)
        
    def update_navigation_buttons(self):
        """Actualiza el estado de los botones de navegación"""
        self.prev_button.setEnabled(self.current_step > 0)
        
        # El botón siguiente se habilita según validaciones específicas
        if self.current_step == 0:
            self.next_button.setEnabled(bool(self.analysis_data.get('image_path')))
        elif self.current_step == 1:
            self.validate_step2()
        else:
            self.next_button.setEnabled(self.current_step < 4)
            
    def update_config_summary(self):
        """Actualiza el resumen de configuración"""
        summary_parts = []
        
        if self.analysis_data.get('image_path'):
            summary_parts.append(f"Imagen: {os.path.basename(self.analysis_data['image_path'])}")
            
        case_number = self.case_number_edit.text().strip()
        if case_number:
            summary_parts.append(f"Caso: {case_number}")
            
        level = self.analysis_level_combo.currentText()
        summary_parts.append(f"Nivel: {level}")
        
        # Deep Learning (si está disponible y habilitado)
        if DEEP_LEARNING_AVAILABLE and hasattr(self, 'enable_dl_cb') and self.enable_dl_cb.isChecked():
            dl_model = self.dl_model_combo.currentText().split(' - ')[0]
            summary_parts.append(f"DL: {dl_model}")
        
        summary = " • ".join(summary_parts)
        self.config_summary_label.setText(summary)
        
    def collect_analysis_data(self) -> dict:
        """Recopila todos los datos para el análisis"""
        data = {
            'image_path': self.analysis_data.get('image_path'),
            'case_data': {
                'case_number': self.case_number_edit.text().strip(),
                'evidence_id': self.evidence_id_edit.text().strip(),
                'examiner': self.examiner_edit.text().strip(),
                'evidence_type': self.evidence_type_combo.currentText(),
                'description': self.case_description_edit.toPlainText().strip(),
                'weapon_make': self.weapon_make_edit.text().strip(),
                'weapon_model': self.weapon_model_edit.text().strip(),
                'caliber': self.caliber_edit.text().strip(),
                'serial_number': self.serial_number_edit.text().strip()
            },
            'processing_config': {
                'analysis_level': self.analysis_level_combo.currentIndex(),
                'priority': self.priority_combo.currentText(),
                'extract_firing_pin': self.extract_firing_pin_cb.isChecked(),
                'extract_breech_face': self.extract_breech_face_cb.isChecked(),
                'extract_extractor': self.extract_extractor_cb.isChecked(),
                'extract_striations': self.extract_striations_cb.isChecked(),
                'extract_land_groove': self.extract_land_groove_cb.isChecked(),
                'nist_quality_check': self.nist_quality_check_cb.isChecked(),
                'nist_authenticity': self.nist_authenticity_cb.isChecked(),
                'nist_compression': self.nist_compression_cb.isChecked(),
                'nist_metadata': self.nist_metadata_cb.isChecked(),
                'generate_afte': self.generate_afte_cb.isChecked(),
                'afte_confidence': self.afte_confidence_cb.isChecked(),
                'afte_comparison': self.afte_comparison_cb.isChecked(),
                'noise_reduction': self.noise_reduction_cb.isChecked(),
                'contrast_enhancement': self.contrast_enhancement_cb.isChecked(),
                'edge_detection': self.edge_detection_cb.isChecked(),
                'morphological': self.morphological_cb.isChecked()
            }
        }
        
        # Metadatos NIST si están habilitados
        if self.enable_nist_checkbox.isChecked():
            data['nist_metadata'] = {
                'lab_name': self.lab_name_edit.text().strip(),
                'lab_accreditation': self.lab_accreditation_edit.text().strip(),
                'capture_device': self.capture_device_edit.text().strip(),
                'magnification': self.magnification_edit.text().strip(),
                'lighting_type': self.lighting_type_combo.currentText(),
                'calibration_date': self.calibration_date_edit.text().strip(),
                'scale_factor': self.scale_factor_edit.text().strip()
            }
            
        # Configuración de Deep Learning si está habilitado
        if DEEP_LEARNING_AVAILABLE and hasattr(self, 'enable_dl_cb') and self.enable_dl_cb.isChecked():
            data['deep_learning_config'] = {
                'enabled': True,
                'model_type': self.dl_model_combo.currentText(),
                'cnn_backbone': self.cnn_backbone_combo.currentText(),
                'confidence_threshold': self.dl_confidence_slider.value() / 100.0,
                'batch_size': self.dl_batch_size_spin.value(),
                'use_gpu': self.dl_use_gpu_cb.isChecked(),
                'siamese_threshold': self.siamese_threshold_spin.value() if hasattr(self, 'siamese_threshold_spin') else 0.7,
                'embedding_dim': self.siamese_embedding_dim_spin.value() if hasattr(self, 'siamese_embedding_dim_spin') else 512
            }
        else:
            data['deep_learning_config'] = {'enabled': False}
            
        return data
        
    def start_analysis(self):
        """Inicia el análisis"""
        try:
            analysis_data = self.collect_analysis_data()
            
            # Validar datos mínimos
            if not analysis_data['image_path']:
                QMessageBox.warning(self, "Error", "No se ha cargado ninguna imagen")
                return
                
            if not analysis_data['case_data']['case_number']:
                QMessageBox.warning(self, "Error", "El número de caso es obligatorio")
                return
                
            if not analysis_data['case_data']['evidence_id']:
                QMessageBox.warning(self, "Error", "El ID de evidencia es obligatorio")
                return
                
            if not analysis_data['case_data']['examiner']:
                QMessageBox.warning(self, "Error", "El nombre del examinador es obligatorio")
                return
                
            # Mostrar progreso
            self.progress_card.show()
            self.process_button.setEnabled(False)
            
            # Iniciar worker thread
            self.analysis_worker = AnalysisWorker(analysis_data)
            self.analysis_worker.progressUpdated.connect(self.on_analysis_progress)
            self.analysis_worker.analysisCompleted.connect(self.on_analysis_completed)
            self.analysis_worker.analysisError.connect(self.on_analysis_error)
            self.analysis_worker.start()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error iniciando análisis: {str(e)}")
            
    def on_analysis_progress(self, progress: int, message: str):
        """Actualiza el progreso del análisis"""
        self.progress_card.set_progress(progress, message)
        
    def on_analysis_completed(self, results: dict):
        """Maneja la finalización del análisis"""
        self.progress_card.set_completed(True, "Análisis completado exitosamente")
        self.process_button.setEnabled(True)
        
        # Mostrar resultados
        self.display_results(results)
        
        # Emitir señal
        self.analysisCompleted.emit(results)
        
    def on_analysis_error(self, error_message: str):
        """Maneja errores en el análisis"""
        self.progress_card.set_completed(False, f"Error: {error_message}")
        self.process_button.setEnabled(True)
        QMessageBox.critical(self, "Error de Análisis", error_message)
        
    def display_results(self, results: dict):
        """Muestra los resultados del análisis con visualizaciones balísticas"""
        # Limpiar resultados anteriores
        for i in reversed(range(self.results_layout.count())):
            self.results_layout.itemAt(i).widget().setParent(None)
        
        # Título de resultados
        title = QLabel("📊 Resultados del Análisis Balístico")
        title.setProperty("class", "section-title")
        self.results_layout.addWidget(title)
        
        # Crear pestañas para diferentes visualizaciones
        from PyQt5.QtWidgets import QTabWidget
        results_tabs = QTabWidget()
        
        # Tab 1: Características Balísticas Extraídas
        features_tab = self.create_ballistic_features_tab(results)
        results_tabs.addTab(features_tab, "🎯 Características")
        
        # Tab 2: Visualizaciones de Marcas
        visualization_tab = self.create_ballistic_visualization_tab(results)
        results_tabs.addTab(visualization_tab, "🔍 Visualizaciones")
        
        # Tab 3: Métricas de Calidad NIST
        quality_tab = self.create_quality_metrics_tab(results)
        results_tabs.addTab(quality_tab, "📏 Métricas NIST")
        
        # Tab 4: Conclusiones AFTE
        conclusions_tab = self.create_afte_conclusions_tab(results)
        results_tabs.addTab(conclusions_tab, "⚖️ Conclusiones")
        
        self.results_layout.addWidget(results_tabs)
        
        # Botones de acción
        actions_layout = QHBoxLayout()
        
        save_btn = QPushButton("💾 Guardar Resultados")
        save_btn.clicked.connect(lambda: self.save_results(results))
        actions_layout.addWidget(save_btn)
        
        export_btn = QPushButton("📄 Generar Reporte")
        export_btn.clicked.connect(lambda: self.generate_report(results))
        actions_layout.addWidget(export_btn)
        
        compare_btn = QPushButton("🔄 Comparar con BD")
        compare_btn.clicked.connect(lambda: self.compare_with_database(results))
        actions_layout.addWidget(compare_btn)
        
        self.results_layout.addLayout(actions_layout)
    
    def create_ballistic_features_tab(self, results: dict) -> QWidget:
        """Crea la pestaña de características balísticas extraídas"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Scroll area para contenido
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        
        content = QWidget()
        content_layout = QVBoxLayout(content)
        
        ballistic_data = results.get('ballistic_features', {})
        
        # Firing Pin (Percutor)
        if 'firing_pin' in ballistic_data:
            fp_card = ResultCard("🎯 Marcas de Percutor", "", "success")
            fp_data = ballistic_data['firing_pin']
            
            fp_content = QWidget()
            fp_layout = QFormLayout(fp_content)
            
            fp_layout.addRow("Diámetro:", QLabel(f"{fp_data.get('diameter', 0):.2f} mm"))
            fp_layout.addRow("Profundidad:", QLabel(f"{fp_data.get('depth', 0):.3f} mm"))
            fp_layout.addRow("Excentricidad:", QLabel(f"{fp_data.get('eccentricity', 0):.2f}"))
            fp_layout.addRow("Circularidad:", QLabel(f"{fp_data.get('circularity', 0):.2f}"))
            
            content_layout.addWidget(fp_card)
        
        # Breech Face (Cara de Recámara)
        if 'breech_face' in ballistic_data:
            bf_card = ResultCard("🔧 Marcas de Cara de Recámara", "", "success")
            bf_data = ballistic_data['breech_face']
            
            bf_content = QWidget()
            bf_layout = QFormLayout(bf_content)
            
            bf_layout.addRow("Rugosidad:", QLabel(f"{bf_data.get('roughness', 0):.1f} Ra"))
            bf_layout.addRow("Orientación:", QLabel(f"{bf_data.get('orientation', 0):.1f}°"))
            bf_layout.addRow("Periodicidad:", QLabel(f"{bf_data.get('periodicity', 0):.2f}"))
            bf_layout.addRow("Entropía:", QLabel(f"{bf_data.get('entropy', 0):.1f}"))
            
            content_layout.addWidget(bf_card)
        
        # Extractor Marks
        if 'extractor_marks' in ballistic_data:
            em_card = ResultCard("🔩 Marcas de Extractor", "", "success")
            em_data = ballistic_data['extractor_marks']
            
            em_content = QWidget()
            em_layout = QFormLayout(em_content)
            
            em_layout.addRow("Cantidad:", QLabel(str(em_data.get('count', 0))))
            em_layout.addRow("Longitud:", QLabel(f"{em_data.get('length', 0):.1f} mm"))
            em_layout.addRow("Profundidad:", QLabel(f"{em_data.get('depth', 0):.3f} mm"))
            em_layout.addRow("Ángulo:", QLabel(f"{em_data.get('angle', 0):.1f}°"))
            
            content_layout.addWidget(em_card)
        
        # Striation Patterns (para balas)
        if 'striation_patterns' in ballistic_data:
            sp_card = ResultCard("📏 Patrones de Estriado", "", "success")
            sp_data = ballistic_data['striation_patterns']
            
            sp_content = QWidget()
            sp_layout = QFormLayout(sp_content)
            
            sp_layout.addRow("Número de líneas:", QLabel(str(sp_data.get('num_lines', 0))))
            sp_layout.addRow("Direcciones dominantes:", QLabel(str(sp_data.get('dominant_directions', []))))
            sp_layout.addRow("Puntuación paralelismo:", QLabel(f"{sp_data.get('parallelism_score', 0):.2f}"))
            sp_layout.addRow("Densidad:", QLabel(f"{sp_data.get('density', 0):.1f}"))
            
            content_layout.addWidget(sp_card)
        
        scroll.setWidget(content)
        layout.addWidget(scroll)
        
        return tab
    
    def create_ballistic_visualization_tab(self, results: dict) -> QWidget:
        """Crea la pestaña de visualizaciones balísticas"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Placeholder para visualizaciones
        placeholder = QLabel("🔧 Visualizaciones balísticas en desarrollo...")
        placeholder.setAlignment(Qt.AlignCenter)
        placeholder.setStyleSheet("color: #666; font-style: italic; padding: 40px;")
        layout.addWidget(placeholder)
        
        return tab
    
    def create_quality_metrics_tab(self, results: dict) -> QWidget:
        """Crea la pestaña de métricas de calidad NIST"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        nist_data = results.get('nist_compliance', {})
        
        # Métricas NIST
        nist_card = ResultCard("✅ Cumplimiento NIST", "", "success")
        nist_content = QWidget()
        nist_layout = QFormLayout(nist_content)
        
        nist_layout.addRow("Puntuación de calidad:", QLabel(f"{nist_data.get('quality_score', 0):.2%}"))
        nist_layout.addRow("Incertidumbre de medición:", QLabel(f"{nist_data.get('measurement_uncertainty', 0):.3f}"))
        nist_layout.addRow("Trazabilidad:", QLabel(nist_data.get('traceability', 'N/A')))
        nist_layout.addRow("Estado de validación:", QLabel(nist_data.get('validation_status', 'N/A')))
        
        layout.addWidget(nist_card)
        layout.addStretch()
        
        return tab
    
    def create_afte_conclusions_tab(self, results: dict) -> QWidget:
        """Crea la pestaña de conclusiones AFTE"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        afte_data = results.get('afte_conclusion', {})
        
        # Conclusión principal
        conclusion_card = ResultCard("⚖️ Conclusión AFTE", "", "warning")
        conclusion_content = QWidget()
        conclusion_layout = QFormLayout(conclusion_content)
        
        conclusion_layout.addRow("Nivel de conclusión:", QLabel(afte_data.get('conclusion_level', 'N/A')))
        conclusion_layout.addRow("Confianza:", QLabel(f"{afte_data.get('confidence', 0):.2%}"))
        conclusion_layout.addRow("Razonamiento:", QLabel(afte_data.get('reasoning', 'N/A')))
        
        layout.addWidget(conclusion_card)
        
        # Notas del examinador
        if afte_data.get('examiner_notes'):
            notes_card = ResultCard("📝 Notas del Examinador", "", "info")
            notes_label = QLabel(afte_data['examiner_notes'])
            notes_label.setWordWrap(True)
            layout.addWidget(notes_card)
        
        layout.addStretch()
        
        return tab
    
    def save_results(self, results: dict):
        """Guarda los resultados del análisis"""
        filename, _ = QFileDialog.getSaveFileName(
            self, "Guardar Resultados", "", "JSON Files (*.json)"
        )
        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                QMessageBox.information(self, "Éxito", "Resultados guardados correctamente")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Error al guardar: {str(e)}")
    
    def generate_report(self, results: dict):
        """Genera un reporte profesional"""
        # Aquí se integraría con la pestaña de reportes
        QMessageBox.information(self, "Reporte", "Funcionalidad de reporte será implementada")
    
    def compare_with_database(self, results: dict):
        """Compara con la base de datos"""
        # Aquí se integraría con la pestaña de comparación
        QMessageBox.information(self, "Comparación", "Funcionalidad de comparación será implementada")
        
    def save_configuration(self):
        """Guarda la configuración actual"""
        try:
            config_data = self.collect_analysis_data()
            
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Guardar Configuración",
                f"config_{config_data['case_data']['case_number']}.json",
                "Archivos JSON (*.json)"
            )
            
            if file_path:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(config_data, f, indent=2, ensure_ascii=False)
                    
                QMessageBox.information(self, "Éxito", "Configuración guardada correctamente")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error guardando configuración: {str(e)}")
            
    def reset_workflow(self):
        """Reinicia el flujo de trabajo"""
        reply = QMessageBox.question(
            self,
            "Confirmar Reinicio",
            "¿Está seguro de que desea reiniciar el flujo de trabajo?\n\nSe perderán todos los datos ingresados.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # Limpiar datos
            self.analysis_data.clear()
            self.current_step = 0
            
            # Limpiar campos
            self.image_drop_zone.clear()
            self.image_info_frame.hide()
            self.case_number_edit.clear()
            self.investigator_edit.clear()
            self.evidence_type_combo.setCurrentIndex(0)
            self.case_description_edit.clear()
            self.acquisition_date_edit.clear()
            self.chain_custody_edit.clear()
            
            # Resetear estados
            self.step2_group.setEnabled(False)
            self.step3_group.setEnabled(False)
            self.step4_group.setEnabled(False)
            self.step5_group.setEnabled(False)
            
            self.update_step_indicator(0)
            self.update_navigation_buttons()
            
            # Limpiar visor y resultados
            self.image_viewer.clear()
            self.progress_card.hide()
            
            # Limpiar resultados
            for i in reversed(range(self.results_layout.count())):
                self.results_layout.itemAt(i).widget().setParent(None)
                
            placeholder_label = QLabel("Los resultados aparecerán aquí después del análisis")
            placeholder_label.setProperty("class", "caption")
            placeholder_label.setAlignment(Qt.AlignCenter)
            placeholder_label.setStyleSheet("color: #757575; padding: 40px;")
            self.results_layout.addWidget(placeholder_label)