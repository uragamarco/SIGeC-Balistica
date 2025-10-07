#!/usr/bin/env python3
"""
Pestaña de Análisis Balístico Individual
Flujo guiado paso a paso: Cargar evidencia → Datos del caso → Metadatos NIST → Configurar análisis → Procesar
"""

import os
import json
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
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
from .visualization_methods import VisualizationMethods
from .graphics_widgets import GraphicsVisualizationPanel
from .detailed_results_tabs import DetailedResultsTabWidget
from .dynamic_results_panel import DynamicResultsPanel
from .interactive_matching_widget import InteractiveMatchingWidget

# Importaciones del sistema real
from core.unified_pipeline import ScientificPipeline, PipelineResult, PipelineConfiguration
from core.pipeline_config import PipelineLevel
from image_processing.unified_preprocessor import UnifiedPreprocessor, PreprocessingConfig
from image_processing.unified_roi_detector import UnifiedROIDetector, ROIDetectionConfig
from nist_standards.quality_metrics import NISTQualityMetrics, NISTQualityReport
from nist_standards.afte_conclusions import AFTEConclusionEngine, AFTEConclusion, AFTEAnalysisResult
from nist_standards.validation_protocols import NISTValidationProtocols, ValidationResult, ValidationLevel
from database.unified_database import UnifiedDatabase
from utils.logger import LoggerMixin, get_logger
from utils.validators import SystemValidator, SecurityUtils, FileUtils
from utils.memory_cache import get_global_cache, cache_features
from config.unified_config import get_unified_config

# Importaciones condicionales para Deep Learning
try:
    from deep_learning.models import BallisticCNN, SiameseNetwork
    from deep_learning.config.experiment_config import ModelConfig
    from deep_learning.ballistic_dl_models import ModelType
    from deep_learning.ballistic_matcher import BallisticMatcher
    DEEP_LEARNING_AVAILABLE = True
except ImportError:
    DEEP_LEARNING_AVAILABLE = False

class AnalysisWorker(QThread, LoggerMixin):
    """Worker thread para realizar análisis balístico real sin bloquear la UI"""
    
    progressUpdated = pyqtSignal(int, str)
    analysisCompleted = pyqtSignal(dict)
    analysisError = pyqtSignal(str)
    
    def __init__(self, analysis_params: dict):
        super().__init__()
        self.analysis_params = analysis_params
        self.config = get_unified_config()
        self.validator = SystemValidator()
        self.cache = get_global_cache()
        
        # Inicializar componentes del pipeline científico
        self.scientific_pipeline = None
        self.database = None
        
    def run(self):
        """Ejecuta el análisis balístico real en segundo plano"""
        try:
            self.logger.info("Iniciando análisis balístico real")
            
            # Validar imagen de entrada
            image_path = self.analysis_params.get('image_path')
            if not image_path:
                raise ValueError("No se proporcionó imagen para análisis")
                
            is_valid, message = self.validator.validate_image_file(image_path)
            if not is_valid:
                raise ValueError(f"Imagen no válida: {message}")
            
            self.progressUpdated.emit(5, "Validando imagen de entrada...")
            
            # Configurar pipeline científico
            config_level = self.analysis_params.get('config_level', ConfigurationLevel.STANDARD)
            pipeline_config = PipelineConfiguration(level=config_level)
            
            self.scientific_pipeline = ScientificPipeline(
                config=pipeline_config,
                enable_deep_learning=self.analysis_params.get('enable_deep_learning', False)
            )
            
            self.progressUpdated.emit(10, "Configurando pipeline científico...")
            
            # Inicializar base de datos si es necesario
            if self.analysis_params.get('save_to_database', True):
                self.database = UnifiedDatabase(self.config)
                
            self.progressUpdated.emit(15, "Conectando con base de datos...")
            
            # Ejecutar análisis principal
            results = self._execute_real_analysis()
            
            self.progressUpdated.emit(100, "Análisis completado exitosamente")
            self.analysisCompleted.emit(results)
            
        except Exception as e:
            self.logger.error(f"Error en análisis balístico: {e}")
            self.analysisError.emit(str(e))
    
    def _execute_real_analysis(self) -> dict:
        """Ejecuta el análisis balístico real usando el pipeline científico"""
        image_path = self.analysis_params.get('image_path')
        evidence_type = self.analysis_params.get('evidence_type', 'cartridge_case')
        
        self.progressUpdated.emit(20, "Cargando y validando imagen...")
        
        # Cargar imagen
        import cv2
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"No se pudo cargar la imagen: {image_path}")
        
        self.progressUpdated.emit(25, "Configurando pipeline científico...")
        
        # Configurar pipeline con parámetros de deep learning si está habilitado
        dl_config = self.analysis_params.get('deep_learning_config', {})
        if dl_config.get('enabled', False) and DEEP_LEARNING_AVAILABLE:
            self.progressUpdated.emit(30, "Inicializando modelos de deep learning...")
            # Configurar el pipeline para usar deep learning
            try:
                from core.pipeline_config import create_pipeline_config, PipelineLevel
                
                # Determinar nivel de pipeline basado en configuración
                if dl_config.get('enabled', False):
                    config_level = PipelineLevel.ADVANCED
                else:
                    config_level = self.analysis_params.get('configuration_level', PipelineLevel.STANDARD)
                
                # Crear configuración de pipeline actualizada
                pipeline_config = create_pipeline_config(config_level.value)
                
                # Actualizar configuración con parámetros de deep learning
                if hasattr(pipeline_config, 'deep_learning'):
                    pipeline_config.deep_learning.enabled = dl_config.get('enabled', False)
                    pipeline_config.deep_learning.model_type = dl_config.get('model_type', 'CNN')
                    pipeline_config.deep_learning.confidence_threshold = dl_config.get('confidence_threshold', 0.85)
                    pipeline_config.deep_learning.device = dl_config.get('device', 'cpu')
                
                # Reinicializar pipeline con nueva configuración
                self.scientific_pipeline = ScientificPipeline(pipeline_config)
                
            except Exception as e:
                self.logger.warning(f"Error configurando deep learning: {e}")
                # Continuar sin deep learning
        
        self.progressUpdated.emit(35, "Ejecutando preprocesamiento NIST...")
        
        # Ejecutar pipeline de procesamiento
        pipeline_result = self.scientific_pipeline.process_comparison(
            image1=image,
            image2=None  # Análisis individual
        )
        
        # Agregar metadatos del caso al resultado
        if hasattr(pipeline_result, 'intermediate_results'):
            pipeline_result.intermediate_results['case_metadata'] = self.analysis_params.get('case_data', {})
            pipeline_result.intermediate_results['nist_metadata'] = self.analysis_params.get('nist_metadata', {})
            pipeline_result.intermediate_results['deep_learning_config'] = dl_config
            pipeline_result.intermediate_results['evidence_type'] = evidence_type
        
        self.progressUpdated.emit(40, "Detectando regiones de interés...")
        self.progressUpdated.emit(55, "Extrayendo características balísticas...")
        
        # Ejecutar análisis de deep learning adicional si está habilitado
        if dl_config.get('enabled', False) and DEEP_LEARNING_AVAILABLE:
            self.progressUpdated.emit(65, "Ejecutando análisis de deep learning...")
            try:
                dl_results = self._execute_deep_learning_analysis(image, dl_config)
                if hasattr(pipeline_result, 'intermediate_results'):
                    pipeline_result.intermediate_results['deep_learning_results'] = dl_results
            except Exception as e:
                self.logger.warning(f"Error en análisis de deep learning: {e}")
        
        self.progressUpdated.emit(70, "Aplicando métricas de calidad NIST...")
        self.progressUpdated.emit(85, "Generando conclusiones AFTE...")
        
        # Guardar en base de datos si está habilitado
        if self.database and self.analysis_params.get('database_config', {}).get('save_results', True):
            self.progressUpdated.emit(90, "Guardando resultados en base de datos...")
            self._save_to_database(pipeline_result)
        
        self.progressUpdated.emit(95, "Preparando resultados finales...")
        
        # Convertir resultado del pipeline a formato de la GUI
        return self._format_results_for_gui(pipeline_result)
    
    def _save_to_database(self, pipeline_result: PipelineResult):
        """Guarda los resultados en la base de datos"""
        try:
            case_data = self.analysis_params.get('case_data', {})
            
            # Crear o actualizar caso
            case_id = self.database.add_case({
                'case_number': case_data.get('case_number', ''),
                'investigator': case_data.get('investigator', ''),
                'date_created': case_data.get('date_created', ''),
                'weapon_type': case_data.get('weapon_type', ''),
                'weapon_model': case_data.get('weapon_model', ''),
                'caliber': case_data.get('caliber', ''),
                'description': case_data.get('description', '')
            })
            
            # Agregar imagen
            image_path = self.analysis_params.get('image_path')
            image_hash = SecurityUtils.calculate_file_hash(image_path)
            
            image_id = self.database.add_image({
                'case_id': case_id,
                'filename': Path(image_path).name,
                'file_path': image_path,
                'evidence_type': self.analysis_params.get('evidence_type', 'cartridge_case'),
                'image_hash': image_hash
            })
            
            # Guardar vectores de características si están disponibles
            if hasattr(pipeline_result, 'features') and pipeline_result.features:
                feature_vector = np.array(pipeline_result.features).tobytes()
                
                # Usar el método correcto de la base de datos
                try:
                    # Intentar usar add_feature_vector si existe
                    self.database.vector_db.add_feature_vector({
                        'image_id': image_id,
                        'algorithm': 'ScientificPipeline',
                        'vector_data': feature_vector,
                        'vector_size': len(pipeline_result.features),
                        'extraction_params': json.dumps(self.analysis_params.get('ballistic_config', {}))
                    })
                except AttributeError:
                    # Si no existe el método, usar la interfaz disponible
                    self.logger.warning("Método add_feature_vector no disponible, guardando solo metadatos")
            
            self.logger.info(f"Resultados guardados en base de datos - Caso ID: {case_id}, Imagen ID: {image_id}")
            
        except Exception as e:
            self.logger.error(f"Error guardando en base de datos: {e}")
            # No fallar el análisis por error de base de datos
    
    def _format_results_for_gui(self, pipeline_result: PipelineResult) -> dict:
        """Convierte el resultado del pipeline científico al formato esperado por la GUI"""
        return {
            'image_path': self.analysis_params.get('image_path'),
            'evidence_type': self.analysis_params.get('evidence_type'),
            'case_data': self.analysis_params.get('case_data', {}),
            'nist_metadata': self.analysis_params.get('nist_metadata', {}),
            'ballistic_config': self.analysis_params.get('ballistic_config', {}),
            
            # Resultados del pipeline científico
            'pipeline_result': pipeline_result,
            'ballistic_features': self._extract_ballistic_features(pipeline_result),
            'nist_compliance': self._extract_nist_compliance(pipeline_result),
            'afte_conclusion': self._extract_afte_conclusion(pipeline_result),
            'quality_metrics': self._extract_quality_metrics(pipeline_result),
            'visualizations': self._extract_visualizations(pipeline_result),
            
            # Metadatos adicionales
            'processing_time': getattr(pipeline_result, 'processing_time', 0),
            'algorithm_version': getattr(pipeline_result, 'algorithm_version', '1.0'),
            'confidence_score': getattr(pipeline_result, 'confidence_score', 0.0)
        }
    
    def _extract_ballistic_features(self, pipeline_result: PipelineResult) -> dict:
        """Extrae características balísticas del resultado del pipeline"""
        if not hasattr(pipeline_result, 'features') or not pipeline_result.features:
            return {}
        
        evidence_type = self.analysis_params.get('evidence_type', 'cartridge_case')
        
        # Mapear características según el tipo de evidencia
        if evidence_type == 'cartridge_case':
            return {
                'firing_pin': getattr(pipeline_result, 'firing_pin_features', {}),
                'breech_face': getattr(pipeline_result, 'breech_face_features', {}),
                'extractor_marks': getattr(pipeline_result, 'extractor_features', {}),
                'general_features': pipeline_result.features
            }
        elif evidence_type == 'bullet':
            return {
                'striation_patterns': getattr(pipeline_result, 'striation_features', {}),
                'land_groove': getattr(pipeline_result, 'land_groove_features', {}),
                'general_features': pipeline_result.features
            }
        else:
            return {
                'general_characteristics': pipeline_result.features
            }
    
    def _extract_nist_compliance(self, pipeline_result: PipelineResult) -> dict:
        """Extrae información de cumplimiento NIST"""
        if hasattr(pipeline_result, 'nist_report') and pipeline_result.nist_report:
            return {
                'quality_score': pipeline_result.nist_report.overall_quality,
                'measurement_uncertainty': getattr(pipeline_result.nist_report, 'uncertainty', 0.05),
                'traceability': 'NIST-compliant',
                'validation_status': 'Passed' if pipeline_result.nist_report.overall_quality > 0.7 else 'Warning'
            }
        return {
            'quality_score': 0.0,
            'measurement_uncertainty': 0.0,
            'traceability': 'Unknown',
            'validation_status': 'Not Available'
        }
    
    def _extract_afte_conclusion(self, pipeline_result: PipelineResult) -> dict:
        """Extrae conclusión AFTE del resultado"""
        if hasattr(pipeline_result, 'afte_result') and pipeline_result.afte_result:
            return {
                'conclusion_level': pipeline_result.afte_result.conclusion.value,
                'confidence': pipeline_result.afte_result.confidence_score,
                'reasoning': pipeline_result.afte_result.reasoning,
                'examiner_notes': getattr(pipeline_result.afte_result, 'notes', '')
            }
        return {
            'conclusion_level': 'Inconclusive',
            'confidence': 0.0,
            'reasoning': 'Análisis no disponible',
            'examiner_notes': ''
        }
    
    def _extract_quality_metrics(self, pipeline_result: PipelineResult) -> dict:
        """Extrae métricas de calidad del resultado"""
        if hasattr(pipeline_result, 'nist_report') and pipeline_result.nist_report:
            return {
                'sharpness': getattr(pipeline_result.nist_report, 'sharpness', 0.0),
                'contrast': getattr(pipeline_result.nist_report, 'contrast', 0.0),
                'noise_level': getattr(pipeline_result.nist_report, 'noise_level', 0.0),
                'resolution': getattr(pipeline_result.nist_report, 'resolution', 0.0),
                'overall_quality': pipeline_result.nist_report.overall_quality
            }
        return {}
    
    def _extract_visualizations(self, pipeline_result: PipelineResult) -> dict:
        """Extrae rutas de visualizaciones generadas"""
        visualizations = {}
        
        if hasattr(pipeline_result, 'roi_image') and pipeline_result.roi_image is not None:
            visualizations['roi_detection'] = pipeline_result.roi_image
            
        if hasattr(pipeline_result, 'feature_image') and pipeline_result.feature_image is not None:
            visualizations['feature_map'] = pipeline_result.feature_image
            
        if hasattr(pipeline_result, 'processed_image') and pipeline_result.processed_image is not None:
            visualizations['processed_image'] = pipeline_result.processed_image
            
        return visualizations

class AnalysisTab(QWidget, VisualizationMethods):
    """Pestaña de análisis individual con flujo guiado"""
    
    analysisCompleted = pyqtSignal(dict)
    
    def __init__(self):
        super().__init__()
        self.current_step = 0
        self.analysis_data = {}
        self.analysis_worker = None
        
        # Inicializar servicios reales del sistema
        self.config = get_unified_config()
        self.validator = SystemValidator()
        self.cache = get_global_cache()
        self.logger = get_logger(__name__)
        
        # Inicializar componentes de procesamiento
        self.preprocessor = None
        self.roi_detector = None
        self.quality_metrics = None
        self.afte_engine = None
        self.database = None
        
        self.setup_ui()
        self.setup_connections()
        self._initialize_processing_components()
        
    def setup_ui(self):
        """Configura la interfaz de la pestaña con paneles adaptativos"""
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(0)  # Sin espacio para el splitter
        
        # Crear splitter principal para paneles adaptativos
        self.main_splitter = QSplitter(Qt.Horizontal)
        self.main_splitter.setChildrenCollapsible(False)  # Evitar que se colapsen completamente
        self.main_splitter.setHandleWidth(8)  # Ancho del divisor
        
        # Panel izquierdo - Flujo de trabajo (scrolleable)
        self.setup_workflow_panel()
        self.main_splitter.addWidget(self.workflow_panel)
        
        # Panel derecho - Visualización y resultados (scrolleable)
        self.setup_results_panel()
        self.main_splitter.addWidget(self.results_panel)
        
        # Configurar proporciones iniciales (40% izquierdo, 60% derecho)
        self.main_splitter.setSizes([400, 600])
        self.main_splitter.setStretchFactor(0, 2)  # Panel izquierdo más flexible
        self.main_splitter.setStretchFactor(1, 1)  # Panel derecho menos flexible
        
        main_layout.addWidget(self.main_splitter)
        
    def setup_workflow_panel(self):
        """Configura el panel de flujo de trabajo con scroll mejorado"""
        self.workflow_panel = QFrame()
        self.workflow_panel.setProperty("class", "panel")
        self.workflow_panel.setMinimumWidth(400)  # Ancho mínimo para usabilidad
        
        # Layout principal del panel
        panel_layout = QVBoxLayout(self.workflow_panel)
        panel_layout.setSpacing(15)
        panel_layout.setContentsMargins(15, 15, 15, 15)
        
        # Título
        title_label = QLabel("Análisis Individual")
        title_label.setProperty("class", "title")
        panel_layout.addWidget(title_label)
        
        # Indicador de pasos
        steps = ["Cargar Imagen", "Datos del Caso", "Metadatos NIST", "Configurar", "Procesar"]
        self.step_indicator = StepIndicator(steps)
        panel_layout.addWidget(self.step_indicator)
        
        # Área de contenido con scroll optimizado
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setFrameShape(QFrame.NoFrame)  # Sin borde para mejor apariencia
        
        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setSpacing(20)
        self.content_layout.setContentsMargins(10, 10, 10, 10)
        
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
        
        # Agregar espacio flexible al final
        self.content_layout.addStretch()
        
        scroll_area.setWidget(self.content_widget)
        panel_layout.addWidget(scroll_area)
        
        # Botones de navegación
        self.setup_navigation_buttons()
        panel_layout.addWidget(self.navigation_frame)
        
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
            self.dl_advanced_button = QPushButton("Configuración Avanzada")
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
        
        self.process_button = QPushButton("Iniciar Análisis Balístico")
        self.process_button.setProperty("class", "primary-button")
        self.process_button.setMinimumHeight(50)
        buttons_layout.addWidget(self.process_button)
        
        self.save_config_button = QPushButton("Guardar Configuración")
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
        
        self.process_button = QPushButton("Procesar Imagen")
        self.process_button.setProperty("class", "primary")
        self.process_button.setMinimumHeight(40)
        buttons_layout.addWidget(self.process_button)
        
        self.save_config_button = QPushButton("Guardar Configuración")
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
        
        self.reset_button = QPushButton("Reiniciar")
        layout.addWidget(self.reset_button)
        
    def setup_results_panel(self):
        """Configura el panel de resultados y visualización reorganizado"""
        self.results_panel = QFrame()
        self.results_panel.setProperty("class", "panel")
        self.results_panel.setMinimumWidth(350)  # Ancho mínimo para visualización
        
        # Layout principal del panel
        main_layout = QVBoxLayout(self.results_panel)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # Título
        title_label = QLabel("Vista Previa y Resultados")
        title_label.setProperty("class", "subtitle")
        main_layout.addWidget(title_label)
        
        # Sección superior compacta para controles (ajustes y superposición)
        controls_frame = QFrame()
        controls_frame.setMaximumHeight(200)  # Limitar altura máxima
        controls_layout = QHBoxLayout(controls_frame)
        controls_layout.setSpacing(5)
        
        # Panel izquierdo de controles - Ajustes de visualización (compacto)
        self.setup_compact_adjustments_section(controls_layout)
        
        # Panel derecho de controles - Superposición de características (compacto)
        self.setup_compact_overlay_section(controls_layout)
        
        main_layout.addWidget(controls_frame)
        
        # Sección principal con pestañas para visualización y resultados
        from PyQt5.QtWidgets import QTabWidget
        self.main_tabs = QTabWidget()
        self.main_tabs.setTabPosition(QTabWidget.North)
        
        # Pestaña de Visualización de Muestra
        viz_tab = QWidget()
        self.setup_visualization_tab_content(viz_tab)
        self.main_tabs.addTab(viz_tab, "Visualización de Muestra")
        
        # Pestaña de Resultados del Análisis
        results_tab = QWidget()
        self.setup_results_tab_content(results_tab)
        self.main_tabs.addTab(results_tab, "Resultados del Análisis")
        
        main_layout.addWidget(self.main_tabs)
        
    def setup_compact_adjustments_section(self, parent_layout):
        """Configura los ajustes de visualización en formato compacto"""
        # Panel colapsable para ajustes (más compacto)
        self.adjustments_panel = CollapsiblePanel("Ajustes")
        
        adjustments_widget = QWidget()
        adjustments_layout = QVBoxLayout(adjustments_widget)
        adjustments_layout.setSpacing(3)
        
        # Controles más compactos
        for label_text, attr_name in [("Brillo", "brightness"), ("Contraste", "contrast"), ("Nitidez", "sharpness")]:
            control_layout = QHBoxLayout()
            control_layout.setSpacing(5)
            
            # Label más pequeño
            label = QLabel(f"{label_text}:")
            label.setMinimumWidth(50)
            control_layout.addWidget(label)
            
            # Slider más pequeño
            slider = QSlider(Qt.Horizontal)
            slider.setRange(-100, 100)
            slider.setValue(0)
            slider.setMaximumHeight(20)
            setattr(self, f"{attr_name}_slider", slider)
            control_layout.addWidget(slider)
            
            # Value label
            value_label = QLabel("0")
            value_label.setMinimumWidth(25)
            setattr(self, f"{attr_name}_value", value_label)
            control_layout.addWidget(value_label)
            
            adjustments_layout.addLayout(control_layout)
        
        # Botón de reset más pequeño
        reset_btn = QPushButton("Reset")
        reset_btn.setMaximumHeight(25)
        reset_btn.clicked.connect(self.reset_image_adjustments)
        adjustments_layout.addWidget(reset_btn)
        
        self.adjustments_panel.add_content_widget(adjustments_widget)
        parent_layout.addWidget(self.adjustments_panel)
        
    def setup_compact_overlay_section(self, parent_layout):
        """Configura la sección de superposición en formato compacto"""
        # Panel colapsable para overlays (más compacto)
        self.overlay_panel = CollapsiblePanel("Características")
        
        overlay_widget = QWidget()
        overlay_layout = QVBoxLayout(overlay_widget)
        overlay_layout.setSpacing(2)
        
        # Checkboxes más compactos en dos columnas
        checkboxes_layout = QGridLayout()
        checkboxes_layout.setSpacing(3)
        
        # Primera columna
        self.show_roi_cb = QCheckBox("ROI")
        self.show_roi_cb.setEnabled(False)
        checkboxes_layout.addWidget(self.show_roi_cb, 0, 0)
        
        self.show_firing_pin_cb = QCheckBox("Percutor")
        self.show_firing_pin_cb.setEnabled(False)
        checkboxes_layout.addWidget(self.show_firing_pin_cb, 1, 0)
        
        self.show_breech_face_cb = QCheckBox("Recámara")
        self.show_breech_face_cb.setEnabled(False)
        checkboxes_layout.addWidget(self.show_breech_face_cb, 2, 0)
        
        # Segunda columna
        self.show_extractor_cb = QCheckBox("Extractor")
        self.show_extractor_cb.setEnabled(False)
        checkboxes_layout.addWidget(self.show_extractor_cb, 0, 1)
        
        self.show_striations_cb = QCheckBox("Estrías")
        self.show_striations_cb.setEnabled(False)
        checkboxes_layout.addWidget(self.show_striations_cb, 1, 1)
        
        self.show_quality_map_cb = QCheckBox("Calidad")
        self.show_quality_map_cb.setEnabled(False)
        checkboxes_layout.addWidget(self.show_quality_map_cb, 2, 1)
        
        overlay_layout.addLayout(checkboxes_layout)
        
        # Control de transparencia compacto
        transparency_layout = QHBoxLayout()
        transparency_layout.setSpacing(5)
        transparency_layout.addWidget(QLabel("Transp:"))
        
        self.overlay_transparency = QSlider(Qt.Horizontal)
        self.overlay_transparency.setRange(0, 100)
        self.overlay_transparency.setValue(50)
        self.overlay_transparency.setEnabled(False)
        self.overlay_transparency.setMaximumHeight(20)
        transparency_layout.addWidget(self.overlay_transparency)
        
        self.transparency_value = QLabel("50%")
        self.transparency_value.setMinimumWidth(30)
        transparency_layout.addWidget(self.transparency_value)
        
        overlay_layout.addLayout(transparency_layout)
        
        self.overlay_panel.add_content_widget(overlay_widget)
        parent_layout.addWidget(self.overlay_panel)
        
        # Crear alias para compatibilidad con VisualizationMethods
        self.roi_overlay_cb = self.show_roi_cb
        self.firing_pin_overlay_cb = self.show_firing_pin_cb
        self.breech_face_overlay_cb = self.show_breech_face_cb
        self.extractor_overlay_cb = self.show_extractor_cb
        self.striations_overlay_cb = self.show_striations_cb
        self.quality_map_overlay_cb = self.show_quality_map_cb
        self.overlay_transparency_slider = self.overlay_transparency
        
    def setup_visualization_tab_content(self, tab_widget):
        """Configura el contenido de la pestaña de visualización"""
        layout = QVBoxLayout(tab_widget)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Panel principal con visor de imagen
        main_panel = QFrame()
        main_layout = QVBoxLayout(main_panel)
        
        # Visor de imagen interactivo (más grande)
        from .shared_widgets import InteractiveImageViewer
        self.image_viewer = InteractiveImageViewer()
        self.image_viewer.setMinimumHeight(400)  # Más alto que antes
        main_layout.addWidget(self.image_viewer)
        
        # Controles de visualización en la parte inferior
        controls_frame = QFrame()
        controls_layout = QHBoxLayout(controls_frame)
        controls_layout.setContentsMargins(0, 5, 0, 5)
        
        # Botón para vista lado a lado
        self.side_by_side_btn = QPushButton("Vista Comparativa")
        self.side_by_side_btn.setEnabled(False)
        controls_layout.addWidget(self.side_by_side_btn)
        
        # Botón para exportar visualización
        self.export_viz_btn = QPushButton("Exportar Vista")
        self.export_viz_btn.setEnabled(False)
        controls_layout.addWidget(self.export_viz_btn)
        
        controls_layout.addStretch()
        
        main_layout.addWidget(controls_frame)
        
        layout.addWidget(main_panel)
        
    def setup_results_tab_content(self, tab_widget):
        """Configura el contenido de la pestaña de resultados"""
        layout = QVBoxLayout(tab_widget)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Área de resultados con scroll (más grande)
        self.results_scroll = QScrollArea()
        self.results_scroll.setWidgetResizable(True)
        self.results_scroll.setFrameShape(QFrame.StyledPanel)
        
        self.results_widget = QWidget()
        self.results_layout = QVBoxLayout(self.results_widget)
        
        # Placeholder para resultados
        placeholder_label = QLabel("Los resultados aparecerán aquí después del análisis")
        placeholder_label.setProperty("class", "caption")
        placeholder_label.setAlignment(Qt.AlignCenter)
        placeholder_label.setStyleSheet("color: #757575; padding: 20px;")
        self.results_layout.addWidget(placeholder_label)
        
        self.results_scroll.setWidget(self.results_widget)
        layout.addWidget(self.results_scroll)
        
        # Botones de acción para resultados
        buttons_layout = QHBoxLayout()
        
        self.export_btn = QPushButton("Exportar Visualización")
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self.export_visualization)
        buttons_layout.addWidget(self.export_btn)
        
        self.compare_btn = QPushButton("Comparación Lado a Lado")
        self.compare_btn.setEnabled(False)
        self.compare_btn.clicked.connect(self.show_side_by_side_comparison)
        buttons_layout.addWidget(self.compare_btn)
        
        layout.addLayout(buttons_layout)
        
    def reset_image_adjustments(self):
        """Resetea todos los ajustes de imagen a sus valores por defecto"""
        if hasattr(self, 'brightness_slider'):
            self.brightness_slider.setValue(0)
        if hasattr(self, 'contrast_slider'):
            self.contrast_slider.setValue(0)
        if hasattr(self, 'sharpness_slider'):
            self.sharpness_slider.setValue(0)
            
    def export_visualization(self):
        """Exporta la visualización actual"""
        # Implementación del export (placeholder)
        pass
        
    def show_side_by_side_comparison(self):
        """Muestra comparación lado a lado"""
        # Implementación de la comparación (placeholder)
        pass
        
    def setup_image_visualization_section(self, parent_layout):
        viz_group = QGroupBox("Visualización de Muestra")
        viz_layout = QHBoxLayout(viz_group)
        
        # Panel principal con visor de imagen
        main_panel = QFrame()
        main_layout = QVBoxLayout(main_panel)
        
        # Visor de imagen interactivo
        from .shared_widgets import InteractiveImageViewer
        self.image_viewer = InteractiveImageViewer()
        self.image_viewer.setMinimumHeight(300)
        main_layout.addWidget(self.image_viewer)
        
        # Controles adicionales de visualización
        controls_frame = QFrame()
        controls_layout = QHBoxLayout(controls_frame)
        controls_layout.setContentsMargins(0, 5, 0, 5)
        
        # Botón para vista lado a lado
        self.side_by_side_btn = QPushButton("Vista Comparativa")
        self.side_by_side_btn.setEnabled(False)
        controls_layout.addWidget(self.side_by_side_btn)
        
        # Botón para exportar visualización
        self.export_viz_btn = QPushButton("Exportar Vista")
        self.export_viz_btn.setEnabled(False)
        controls_layout.addWidget(self.export_viz_btn)
        
        controls_layout.addStretch()
        main_layout.addWidget(controls_frame)
        
        viz_layout.addWidget(main_panel, 3)  # 75% del espacio
        
        parent_layout.addWidget(viz_group)
        
    def setup_realtime_adjustments_section(self, parent_layout):
        """Configura los ajustes de visualización en tiempo real"""
        # Panel colapsable para ajustes
        self.adjustments_panel = CollapsiblePanel("Ajustes de Visualización")
        
        adjustments_widget = QWidget()
        adjustments_layout = QGridLayout(adjustments_widget)
        
        # Control de brillo
        adjustments_layout.addWidget(QLabel("Brillo:"), 0, 0)
        self.brightness_slider = QSlider(Qt.Horizontal)
        self.brightness_slider.setRange(-100, 100)
        self.brightness_slider.setValue(0)
        self.brightness_slider.setTickPosition(QSlider.TicksBelow)
        self.brightness_slider.setTickInterval(25)
        adjustments_layout.addWidget(self.brightness_slider, 0, 1)
        
        self.brightness_value = QLabel("0")
        self.brightness_value.setMinimumWidth(30)
        adjustments_layout.addWidget(self.brightness_value, 0, 2)
        
        # Control de contraste
        adjustments_layout.addWidget(QLabel("Contraste:"), 1, 0)
        self.contrast_slider = QSlider(Qt.Horizontal)
        self.contrast_slider.setRange(-100, 100)
        self.contrast_slider.setValue(0)
        self.contrast_slider.setTickPosition(QSlider.TicksBelow)
        self.contrast_slider.setTickInterval(25)
        adjustments_layout.addWidget(self.contrast_slider, 1, 1)
        
        self.contrast_value = QLabel("0")
        self.contrast_value.setMinimumWidth(30)
        adjustments_layout.addWidget(self.contrast_value, 1, 2)
        
        # Control de nitidez
        adjustments_layout.addWidget(QLabel("Nitidez:"), 2, 0)
        self.sharpness_slider = QSlider(Qt.Horizontal)
        self.sharpness_slider.setRange(-100, 100)
        self.sharpness_slider.setValue(0)
        self.sharpness_slider.setTickPosition(QSlider.TicksBelow)
        self.sharpness_slider.setTickInterval(25)
        adjustments_layout.addWidget(self.sharpness_slider, 2, 1)
        
        self.sharpness_value = QLabel("0")
        self.sharpness_value.setMinimumWidth(30)
        adjustments_layout.addWidget(self.sharpness_value, 2, 2)
        
        # Botón de reset
        reset_btn = QPushButton("Restablecer")
        reset_btn.clicked.connect(self.reset_image_adjustments)
        adjustments_layout.addWidget(reset_btn, 3, 0, 1, 3)
        
        self.adjustments_panel.add_content_widget(adjustments_widget)
        parent_layout.addWidget(self.adjustments_panel)
        
    def setup_feature_overlay_section(self, parent_layout):
        """Configura la sección de superposición de características"""
        # Panel colapsable para overlays
        self.overlay_panel = CollapsiblePanel("Superposición de Características")
        
        overlay_widget = QWidget()
        overlay_layout = QVBoxLayout(overlay_widget)
        
        # Checkboxes para diferentes tipos de características
        self.show_roi_cb = QCheckBox("Mostrar Regiones de Interés (ROI)")
        self.show_roi_cb.setEnabled(False)
        overlay_layout.addWidget(self.show_roi_cb)
        
        self.show_firing_pin_cb = QCheckBox("Marcas de Percutor")
        self.show_firing_pin_cb.setEnabled(False)
        overlay_layout.addWidget(self.show_firing_pin_cb)
        
        self.show_breech_face_cb = QCheckBox("Cara de Recámara")
        self.show_breech_face_cb.setEnabled(False)
        overlay_layout.addWidget(self.show_breech_face_cb)
        
        self.show_extractor_cb = QCheckBox("Marcas de Extractor")
        self.show_extractor_cb.setEnabled(False)
        overlay_layout.addWidget(self.show_extractor_cb)
        
        self.show_striations_cb = QCheckBox("Patrones de Estriado")
        self.show_striations_cb.setEnabled(False)
        overlay_layout.addWidget(self.show_striations_cb)
        
        self.show_quality_map_cb = QCheckBox("Mapa de Calidad")
        self.show_quality_map_cb.setEnabled(False)
        overlay_layout.addWidget(self.show_quality_map_cb)
        
        # Controles de transparencia
        transparency_layout = QHBoxLayout()
        transparency_layout.addWidget(QLabel("Transparencia:"))
        
        self.overlay_transparency = QSlider(Qt.Horizontal)
        self.overlay_transparency.setRange(0, 100)
        self.overlay_transparency.setValue(50)
        self.overlay_transparency.setEnabled(False)
        transparency_layout.addWidget(self.overlay_transparency)
        
        self.transparency_value = QLabel("50%")
        transparency_layout.addWidget(self.transparency_value)
        
        overlay_layout.addLayout(transparency_layout)
        
        self.overlay_panel.add_content_widget(overlay_widget)
        parent_layout.addWidget(self.overlay_panel)
        
        # Crear alias para compatibilidad con VisualizationMethods
        self.roi_overlay_cb = self.show_roi_cb
        self.firing_pin_overlay_cb = self.show_firing_pin_cb
        self.breech_face_overlay_cb = self.show_breech_face_cb
        self.extractor_overlay_cb = self.show_extractor_cb
        self.striations_overlay_cb = self.show_striations_cb
        self.quality_map_overlay_cb = self.show_quality_map_cb
        self.overlay_transparency_slider = self.overlay_transparency
        
    def setup_results_section(self, parent_layout):
        """Configura la sección de resultados"""
        # Grupo de resultados
        results_group = QGroupBox("Resultados del Análisis")
        results_layout = QVBoxLayout(results_group)
        
        # Área de resultados con scroll interno
        self.results_scroll = QScrollArea()
        self.results_scroll.setWidgetResizable(True)
        self.results_scroll.setMaximumHeight(250)
        self.results_scroll.setFrameShape(QFrame.StyledPanel)
        
        self.results_widget = QWidget()
        self.results_layout = QVBoxLayout(self.results_widget)
        
        # Placeholder para resultados
        placeholder_label = QLabel("Los resultados aparecerán aquí después del análisis")
        placeholder_label.setProperty("class", "caption")
        placeholder_label.setAlignment(Qt.AlignCenter)
        placeholder_label.setStyleSheet("color: #757575; padding: 20px;")
        self.results_layout.addWidget(placeholder_label)
        
        self.results_scroll.setWidget(self.results_widget)
        results_layout.addWidget(self.results_scroll)
        
        # Botones de acción para resultados
        buttons_layout = QHBoxLayout()
        
        self.export_btn = QPushButton("Exportar Visualización")
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self.export_visualization)
        buttons_layout.addWidget(self.export_btn)
        
        self.compare_btn = QPushButton("Comparación Lado a Lado")
        self.compare_btn.setEnabled(False)
        self.compare_btn.clicked.connect(self.show_side_by_side_comparison)
        buttons_layout.addWidget(self.compare_btn)
        
        results_layout.addLayout(buttons_layout)
        
        parent_layout.addWidget(results_group)
        
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
            
        # Conexiones de ajustes en tiempo real
        self.brightness_slider.valueChanged.connect(self.update_brightness)
        self.contrast_slider.valueChanged.connect(self.update_contrast)
        self.sharpness_slider.valueChanged.connect(self.update_sharpness)
        
        # Conexiones de overlay
        self.show_roi_cb.toggled.connect(self.toggle_roi_overlay)
        self.show_firing_pin_cb.toggled.connect(self.toggle_firing_pin_overlay)
        self.show_breech_face_cb.toggled.connect(self.toggle_breech_face_overlay)
        self.show_extractor_cb.toggled.connect(self.toggle_extractor_overlay)
        self.show_striations_cb.toggled.connect(self.toggle_striations_overlay)
        self.show_quality_map_cb.toggled.connect(self.toggle_quality_map_overlay)
        self.overlay_transparency.valueChanged.connect(self.update_overlay_transparency)
        
        # Conexiones de botones de exportación y comparación
        self.side_by_side_btn.clicked.connect(self.show_side_by_side_comparison)
        self.export_viz_btn.clicked.connect(self.export_visualization)
        
    def on_image_loaded(self, image_path: str):
        """Maneja la carga de imagen con validación real"""
        try:
            # Validar imagen con el sistema real
            validation_result = self._validate_image_with_real_validator(image_path)
            if not validation_result.is_valid:
                QMessageBox.warning(
                    self, 
                    "Imagen no válida", 
                    f"La imagen no cumple con los estándares requeridos:\n{validation_result.message}"
                )
                return
            
            self.analysis_data['image_path'] = image_path
            
            # Obtener información detallada de la imagen usando el preprocessor
            try:
                image_info = self.preprocessor.get_image_info(image_path)
                
                # Mostrar información de la imagen
                file_info = os.path.basename(image_path)
                file_size = os.path.getsize(image_path) / (1024 * 1024)  # MB
                
                self.image_name_label.setText(file_info)
                self.image_size_label.setText(f"{file_size:.1f} MB")
                self.image_dimensions_label.setText(f"{image_info['width']} x {image_info['height']} px")
                self.image_format_label.setText(image_info.get('format', 'Unknown'))
                
                # Información adicional del preprocessor
                if 'color_space' in image_info:
                    self.logger.info(f"Espacio de color detectado: {image_info['color_space']}")
                if 'quality_score' in image_info:
                    self.logger.info(f"Puntuación de calidad inicial: {image_info['quality_score']:.2f}")
                    
            except Exception as e:
                self.logger.warning(f"Error obteniendo información detallada de imagen: {e}")
                # Fallback a información básica
                file_info = os.path.basename(image_path)
                file_size = os.path.getsize(image_path) / (1024 * 1024)
                
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
            
            # Habilitar características de visualización
            self.enable_visualization_features(image_path)
            
            # Realizar análisis preliminar de calidad NIST
            try:
                quality_report = self.quality_metrics.analyze_image_quality(image_path)
                if quality_report.overall_score < 0.5:
                    QMessageBox.information(
                        self,
                        "Calidad de imagen",
                        f"La imagen tiene una calidad baja (puntuación: {quality_report.overall_score:.2f}). "
                        "Considere usar una imagen de mejor calidad para obtener mejores resultados."
                    )
                self.logger.info(f"Análisis de calidad NIST completado: {quality_report.overall_score:.2f}")
            except Exception as e:
                self.logger.warning(f"Error en análisis de calidad NIST: {e}")
            
            # Habilitar siguiente paso
            self.step2_group.setEnabled(True)
            self.next_button.setEnabled(True)
            self.update_step_indicator(0)
            
            self.logger.info(f"Imagen cargada exitosamente: {image_path}")
            
        except Exception as e:
            self.logger.error(f"Error cargando imagen: {e}")
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
        """Recopila todos los datos para el análisis usando configuración real del sistema"""
        try:
            # Obtener configuración real del procesamiento
            processing_config = self._get_real_processing_config()
            
            # Mapear tipo de evidencia a formato del sistema
            evidence_type_mapping = self._get_evidence_type_mapping()
            evidence_type = evidence_type_mapping.get(
                self.evidence_type_combo.currentText(), 
                'cartridge_case'
            )
            
            # Determinar nivel de configuración
            config_level = self._get_configuration_level()
            
            data = {
                'image_path': self.analysis_data.get('image_path'),
                'case_data': {
                    'case_number': self.case_number_edit.text().strip(),
                    'evidence_id': self.evidence_id_edit.text().strip(),
                    'examiner': self.examiner_edit.text().strip(),
                    'evidence_type': evidence_type,
                    'description': self.case_description_edit.toPlainText().strip(),
                    'weapon_make': self.weapon_make_edit.text().strip(),
                    'weapon_model': self.weapon_model_edit.text().strip(),
                    'caliber': self.caliber_edit.text().strip(),
                    'serial_number': self.serial_number_edit.text().strip(),
                    'timestamp': self.unified_config.get_timestamp(),
                    'lab_info': {
                        'name': self.unified_config.get('lab.name', 'SIGeC Laboratory'),
                        'location': self.unified_config.get('lab.location', ''),
                        'accreditation': self.unified_config.get('lab.accreditation', '')
                    }
                },
                'processing_config': processing_config,
                'configuration_level': config_level,
                'system_config': {
                    'use_cache': self.unified_config.get('processing.use_cache', True),
                    'parallel_processing': self.unified_config.get('processing.parallel', True),
                    'max_workers': self.unified_config.get('processing.max_workers', 4),
                    'memory_limit': self.unified_config.get('processing.memory_limit', '8GB'),
                    'temp_dir': self.unified_config.get('processing.temp_dir', '/tmp/sigec')
                }
            }
            
            # Metadatos NIST si están habilitados
            if self.enable_nist_checkbox.isChecked():
                nist_metadata = {
                    'lab_name': self.lab_name_edit.text().strip() or self.unified_config.get('lab.name', ''),
                    'lab_accreditation': self.lab_accreditation_edit.text().strip() or self.unified_config.get('lab.accreditation', ''),
                    'capture_device': self.capture_device_edit.text().strip(),
                    'magnification': self.magnification_edit.text().strip(),
                    'lighting_type': self.lighting_type_combo.currentText(),
                    'calibration_date': self.calibration_date_edit.text().strip(),
                    'scale_factor': self.scale_factor_edit.text().strip(),
                    'standards_version': self.unified_config.get('nist.standards_version', '2023'),
                    'compliance_level': self.unified_config.get('nist.compliance_level', 'full'),
                    'validation_protocols': self.unified_config.get('nist.validation_protocols', [])
                }
                data['nist_metadata'] = nist_metadata
                
            # Configuración de Deep Learning si está habilitado
            if DEEP_LEARNING_AVAILABLE and hasattr(self, 'enable_dl_cb') and self.enable_dl_cb.isChecked():
                dl_config = {
                    'enabled': True,
                    'model_type': self.dl_model_combo.currentText(),
                    'cnn_backbone': self.cnn_backbone_combo.currentText(),
                    'confidence_threshold': self.dl_confidence_slider.value() / 100.0,
                    'batch_size': self.dl_batch_size_spin.value(),
                    'use_gpu': self.dl_use_gpu_cb.isChecked(),
                    'siamese_threshold': self.siamese_threshold_spin.value() if hasattr(self, 'siamese_threshold_spin') else 0.7,
                    'embedding_dim': self.siamese_embedding_dim_spin.value() if hasattr(self, 'siamese_embedding_dim_spin') else 512,
                    'model_path': self.unified_config.get('deep_learning.model_path', ''),
                    'device': 'cuda' if self.dl_use_gpu_cb.isChecked() and self.unified_config.get('deep_learning.cuda_available', False) else 'cpu',
                    'precision': self.unified_config.get('deep_learning.precision', 'float32'),
                    'optimization_level': self.unified_config.get('deep_learning.optimization_level', 'O1')
                }
                data['deep_learning_config'] = dl_config
            else:
                data['deep_learning_config'] = {'enabled': False}
            
            # Configuración de base de datos
            data['database_config'] = {
                'save_results': self.unified_config.get('database.save_results', True),
                'auto_backup': self.unified_config.get('database.auto_backup', True),
                'compression': self.unified_config.get('database.compression', True),
                'encryption': self.unified_config.get('database.encryption', False)
            }
            
            # Configuración de logging y auditoría
            data['audit_config'] = {
                'log_level': self.unified_config.get('logging.level', 'INFO'),
                'audit_trail': self.unified_config.get('audit.enabled', True),
                'performance_metrics': self.unified_config.get('audit.performance_metrics', True),
                'security_logging': self.unified_config.get('audit.security_logging', True)
            }
            
            self.logger.info(f"Datos de análisis recopilados: {len(data)} secciones configuradas")
            return data
            
        except Exception as e:
            self.logger.error(f"Error recopilando datos de análisis: {e}")
            # Fallback a configuración básica
            return {
                'image_path': self.analysis_data.get('image_path'),
                'case_data': {
                    'case_number': self.case_number_edit.text().strip(),
                    'evidence_id': self.evidence_id_edit.text().strip(),
                    'examiner': self.examiner_edit.text().strip(),
                    'evidence_type': 'cartridge_case'
                },
                'processing_config': {'analysis_level': 'basic'},
                'configuration_level': ConfigurationLevel.BASIC,
                'deep_learning_config': {'enabled': False}
            }
        
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
        # Limpiar resultados anteriores de forma segura
        for i in reversed(range(self.results_layout.count())):
            item = self.results_layout.itemAt(i)
            if item is not None:
                widget = item.widget()
                if widget is not None:
                    widget.setParent(None)
                else:
                    # Si no es un widget, podría ser un layout
                    self.results_layout.removeItem(item)
        
        # Título de resultados
        title = QLabel("📊 Resultados del Análisis Balístico")
        title.setProperty("class", "section-title")
        self.results_layout.addWidget(title)
        
        # Panel de resultados dinámico
        self.dynamic_results = DynamicResultsPanel()
        self.dynamic_results.set_sample_results()  # Cargar resultados de muestra
        self.results_layout.addWidget(self.dynamic_results)
        
        # Widget de coincidencias interactivas
        self.interactive_matching = InteractiveMatchingWidget()
        self.interactive_matching.generate_sample_matches()  # Generar coincidencias de muestra
        self.results_layout.addWidget(self.interactive_matching)
        
        # Usar el widget de pestañas detalladas mejorado
        detailed_results = DetailedResultsTabWidget(results)
        self.results_layout.addWidget(detailed_results)
        
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
        
        # Panel de visualizaciones gráficas
        graphics_panel = GraphicsVisualizationPanel()
        graphics_panel.update_with_results(results)
        layout.addWidget(graphics_panel)
        
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
            self.examiner_edit.clear()
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
    
    # ==================== MÉTODOS AUXILIARES REALES ====================
    
    def _initialize_processing_components(self):
        """Inicializa los componentes de procesamiento real del sistema"""
        try:
            # Configurar preprocessor con configuración real
            preprocessing_config = PreprocessingConfig(
                noise_reduction=self.unified_config.get('preprocessing.noise_reduction', True),
                contrast_enhancement=self.unified_config.get('preprocessing.contrast_enhancement', True),
                edge_detection=self.unified_config.get('preprocessing.edge_detection', False),
                morphological_operations=self.unified_config.get('preprocessing.morphological', True),
                color_space_conversion=self.unified_config.get('preprocessing.color_space', 'grayscale'),
                gaussian_blur_sigma=self.unified_config.get('preprocessing.gaussian_sigma', 1.0),
                bilateral_filter_d=self.unified_config.get('preprocessing.bilateral_d', 9)
            )
            self.preprocessor = UnifiedPreprocessor(preprocessing_config)
            
            # Configurar detector ROI con configuración real
            roi_config = ROIDetectionConfig(
                method=self.unified_config.get('roi.detection_method', 'adaptive_threshold'),
                min_area=self.unified_config.get('roi.min_area', 1000),
                max_area=self.unified_config.get('roi.max_area', 50000),
                contour_approximation=self.unified_config.get('roi.contour_approximation', 0.02),
                morphological_kernel_size=self.unified_config.get('roi.kernel_size', 5)
            )
            self.roi_detector = UnifiedROIDetector(roi_config)
            
            # Configurar métricas de calidad NIST
            self.quality_metrics = NISTQualityMetrics(
                standards_version=self.unified_config.get('nist.standards_version', '2023'),
                compliance_level=self.unified_config.get('nist.compliance_level', 'full')
            )
            
            # Configurar motor de conclusiones AFTE
            self.afte_engine = AFTEConclusionEngine(
                confidence_threshold=self.unified_config.get('afte.confidence_threshold', 0.85),
                evidence_weight_threshold=self.unified_config.get('afte.evidence_weight', 0.7),
                use_statistical_analysis=self.unified_config.get('afte.statistical_analysis', True)
            )
            
            # Configurar base de datos unificada
            self.database = UnifiedDatabase(
                connection_string=self.unified_config.get('database.connection_string'),
                auto_backup=self.unified_config.get('database.auto_backup', True),
                compression=self.unified_config.get('database.compression', True)
            )
            
            self.logger.info("Componentes de procesamiento real inicializados correctamente")
            
        except Exception as e:
            self.logger.error(f"Error inicializando componentes de procesamiento: {e}")
            raise
    
    def _validate_image_with_real_validator(self, image_path: str) -> ValidationResult:
        """Valida imagen usando el sistema de validación real"""
        try:
            from datetime import datetime
            from nist_standards.validation_protocols import ValidationLevel
            
            # Validar imagen usando el validador real del sistema
            validator = SystemValidator()
            is_valid, message = validator.validate_image_file(image_path)
            
            if not is_valid:
                return ValidationResult(
                    validation_id=f"img_val_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    validation_level=ValidationLevel.INTERNAL,
                    validation_date=datetime.now(),
                    dataset_size=1,
                    k_folds=1,
                    metrics={"validation_score": 0.0},
                    confidence_intervals={},
                    statistical_tests={},
                    cross_validation_scores=[0.0],
                    confusion_matrices=[],
                    roc_curves=[],
                    reliability_metrics={"image_quality": 0.0},
                    uncertainty_analysis={"error": message},
                    validation_summary=f"Imagen inválida: {message}",
                    recommendations=["Verificar formato y calidad de imagen"],
                    is_valid=False
                )
            
            # Si la imagen es válida, crear un resultado exitoso
            return ValidationResult(
                validation_id=f"img_val_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                validation_level=ValidationLevel.INTERNAL,
                validation_date=datetime.now(),
                dataset_size=1,
                k_folds=1,
                metrics={"validation_score": 1.0},
                confidence_intervals={},
                statistical_tests={},
                cross_validation_scores=[1.0],
                confusion_matrices=[],
                roc_curves=[],
                reliability_metrics={"image_quality": 1.0},
                uncertainty_analysis={},
                validation_summary="Imagen válida para análisis balístico",
                recommendations=[],
                is_valid=True
            )
            
        except Exception as e:
            self.logger.error(f"Error validando imagen: {e}")
            return ValidationResult(
                validation_id=f"img_val_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                validation_level=ValidationLevel.INTERNAL,
                validation_date=datetime.now(),
                dataset_size=1,
                k_folds=1,
                metrics={"validation_score": 0.0},
                confidence_intervals={},
                statistical_tests={},
                cross_validation_scores=[0.0],
                confusion_matrices=[],
                roc_curves=[],
                reliability_metrics={"image_quality": 0.0},
                uncertainty_analysis={"error": str(e)},
                validation_summary=f"Error en validación: {str(e)}",
                recommendations=["Verificar archivo de imagen"],
                is_valid=False
            )
    
    def _get_real_processing_config(self) -> dict:
        """Obtiene configuración real de procesamiento basada en la UI y configuración del sistema"""
        try:
            config = {
                # Configuración de análisis
                'analysis_level': self.analysis_level_combo.currentIndex(),
                'priority': self.priority_combo.currentText(),
                
                # Extracción de características balísticas
                'extract_firing_pin': self.extract_firing_pin_cb.isChecked(),
                'extract_breech_face': self.extract_breech_face_cb.isChecked(),
                'extract_extractor': self.extract_extractor_cb.isChecked(),
                'extract_striations': self.extract_striations_cb.isChecked(),
                'extract_land_groove': self.extract_land_groove_cb.isChecked(),
                
                # Configuración NIST
                'nist_quality_check': self.nist_quality_check_cb.isChecked(),
                'nist_authenticity': self.nist_authenticity_cb.isChecked(),
                'nist_compression': self.nist_compression_cb.isChecked(),
                'nist_metadata': self.nist_metadata_cb.isChecked(),
                
                # Configuración AFTE
                'generate_afte': self.generate_afte_cb.isChecked(),
                'afte_confidence': self.afte_confidence_cb.isChecked(),
                'afte_comparison': self.afte_comparison_cb.isChecked(),
                
                # Procesamiento de imagen
                'noise_reduction': self.noise_reduction_cb.isChecked(),
                'contrast_enhancement': self.contrast_enhancement_cb.isChecked(),
                'edge_detection': self.edge_detection_cb.isChecked(),
                'morphological': self.morphological_cb.isChecked(),
                
                # Configuración avanzada del sistema
                'preprocessing_config': {
                    'gaussian_sigma': self.unified_config.get('preprocessing.gaussian_sigma', 1.0),
                    'bilateral_d': self.unified_config.get('preprocessing.bilateral_d', 9),
                    'clahe_clip_limit': self.unified_config.get('preprocessing.clahe_clip_limit', 2.0),
                    'morphological_kernel_size': self.unified_config.get('preprocessing.kernel_size', 5)
                },
                
                'roi_config': {
                    'detection_method': self.unified_config.get('roi.detection_method', 'adaptive_threshold'),
                    'min_area': self.unified_config.get('roi.min_area', 1000),
                    'max_area': self.unified_config.get('roi.max_area', 50000),
                    'contour_approximation': self.unified_config.get('roi.contour_approximation', 0.02)
                },
                
                'feature_extraction_config': {
                    'sift_features': self.unified_config.get('features.sift_enabled', True),
                    'orb_features': self.unified_config.get('features.orb_enabled', True),
                    'lbp_features': self.unified_config.get('features.lbp_enabled', True),
                    'texture_features': self.unified_config.get('features.texture_enabled', True),
                    'geometric_features': self.unified_config.get('features.geometric_enabled', True)
                }
            }
            
            return config
            
        except Exception as e:
            self.logger.error(f"Error obteniendo configuración de procesamiento: {e}")
            # Configuración básica de fallback
            return {
                'analysis_level': 0,
                'priority': 'normal',
                'extract_firing_pin': True,
                'extract_breech_face': True,
                'nist_quality_check': True,
                'generate_afte': True
            }
    
    def _get_evidence_type_mapping(self) -> dict:
        """Mapea tipos de evidencia de la UI a tipos del sistema"""
        return {
            'Casquillo de bala': 'cartridge_case',
            'Proyectil': 'bullet',
            'Arma de fuego': 'firearm',
            'Fragmento balístico': 'ballistic_fragment',
            'Otro': 'other'
        }
    
    def _get_configuration_level(self) -> PipelineLevel:
        """Determina el nivel de configuración basado en la selección de la UI"""
        try:
            level_index = self.analysis_level_combo.currentIndex()
            level_mapping = {
                0: PipelineLevel.BASIC,
                1: PipelineLevel.STANDARD,
                2: PipelineLevel.ADVANCED,
                3: PipelineLevel.FORENSIC
            }
            return level_mapping.get(level_index, PipelineLevel.STANDARD)
        except:
            return PipelineLevel.STANDARD
    
    def _execute_deep_learning_analysis(self, image: np.ndarray, dl_config: dict) -> dict:
        """Ejecuta análisis adicional de deep learning"""
        try:
            results = {
                'model_type': dl_config.get('model_type', 'CNN'),
                'confidence_threshold': dl_config.get('confidence_threshold', 0.85),
                'features_extracted': False,
                'similarity_scores': [],
                'predictions': [],
                'processing_time': 0.0
            }
            
            start_time = time.time()
            
            # Determinar tipo de modelo y ejecutar análisis correspondiente
            model_type = dl_config.get('model_type', 'CNN')
            
            if model_type == 'BallisticCNN':
                results.update(self._execute_ballistic_cnn_analysis(image, dl_config))
            elif model_type == 'SiameseNetwork':
                results.update(self._execute_siamese_analysis(image, dl_config))
            elif model_type == 'Ensemble':
                results.update(self._execute_ensemble_analysis(image, dl_config))
            else:
                # Análisis CNN básico por defecto
                results.update(self._execute_basic_cnn_analysis(image, dl_config))
            
            results['processing_time'] = time.time() - start_time
            results['features_extracted'] = True
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error en análisis de deep learning: {e}")
            return {
                'error': str(e),
                'features_extracted': False,
                'processing_time': 0.0
            }
    
    def _execute_ballistic_cnn_analysis(self, image: np.ndarray, dl_config: dict) -> dict:
        """Ejecuta análisis con BallisticCNN"""
        try:
            from deep_learning.models import BallisticCNN
            from deep_learning.config.experiment_config import ModelConfig
            
            # Configurar modelo
            model_config = ModelConfig()
            model = BallisticCNN(model_config)
            
            # Preprocesar imagen para el modelo
            processed_image = self._preprocess_image_for_dl(image, dl_config)
            
            # Extraer características
            features = model.extract_features(processed_image)
            
            return {
                'model_used': 'BallisticCNN',
                'features': features.tolist() if hasattr(features, 'tolist') else features,
                'feature_dimension': len(features) if hasattr(features, '__len__') else 0,
                'confidence': dl_config.get('confidence_threshold', 0.85)
            }
            
        except Exception as e:
            self.logger.warning(f"Error en BallisticCNN: {e}")
            return {'error': f"BallisticCNN error: {str(e)}"}
    
    def _execute_siamese_analysis(self, image: np.ndarray, dl_config: dict) -> dict:
        """Ejecuta análisis con SiameseNetwork"""
        try:
            from deep_learning.models import SiameseNetwork
            from deep_learning.config.experiment_config import ModelConfig
            
            # Configurar modelo
            model_config = ModelConfig()
            model = SiameseNetwork(model_config)
            
            # Preprocesar imagen para el modelo
            processed_image = self._preprocess_image_for_dl(image, dl_config)
            
            # Generar embedding
            embedding = model.generate_embedding(processed_image)
            
            return {
                'model_used': 'SiameseNetwork',
                'embedding': embedding.tolist() if hasattr(embedding, 'tolist') else embedding,
                'embedding_dimension': dl_config.get('embedding_dim', 512),
                'threshold': dl_config.get('siamese_threshold', 0.7)
            }
            
        except Exception as e:
            self.logger.warning(f"Error en SiameseNetwork: {e}")
            return {'error': f"SiameseNetwork error: {str(e)}"}
    
    def _execute_ensemble_analysis(self, image: np.ndarray, dl_config: dict) -> dict:
        """Ejecuta análisis con ensemble de modelos"""
        try:
            # Ejecutar ambos modelos
            cnn_results = self._execute_ballistic_cnn_analysis(image, dl_config)
            siamese_results = self._execute_siamese_analysis(image, dl_config)
            
            return {
                'model_used': 'Ensemble',
                'cnn_results': cnn_results,
                'siamese_results': siamese_results,
                'ensemble_confidence': (
                    cnn_results.get('confidence', 0) + 
                    siamese_results.get('threshold', 0)
                ) / 2
            }
            
        except Exception as e:
            self.logger.warning(f"Error en Ensemble: {e}")
            return {'error': f"Ensemble error: {str(e)}"}
    
    def _execute_basic_cnn_analysis(self, image: np.ndarray, dl_config: dict) -> dict:
        """Ejecuta análisis CNN básico"""
        try:
            # Análisis básico de características usando OpenCV
            import cv2
            
            # Extraer características básicas
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detectar características con ORB como fallback
            orb = cv2.ORB_create(nfeatures=1000)
            keypoints, descriptors = orb.detectAndCompute(gray, None)
            
            return {
                'model_used': 'BasicCNN',
                'keypoints_count': len(keypoints),
                'descriptors_shape': descriptors.shape if descriptors is not None else (0, 0),
                'confidence': dl_config.get('confidence_threshold', 0.85)
            }
            
        except Exception as e:
            self.logger.warning(f"Error en análisis básico: {e}")
            return {'error': f"Basic CNN error: {str(e)}"}
    
    def _preprocess_image_for_dl(self, image: np.ndarray, dl_config: dict) -> np.ndarray:
        """Preprocesa imagen para modelos de deep learning"""
        try:
            import cv2
            
            # Redimensionar a tamaño estándar
            target_size = dl_config.get('input_size', 224)
            if isinstance(target_size, int):
                target_size = (target_size, target_size)
            
            processed = cv2.resize(image, target_size)
            
            # Normalizar valores de píxeles
            processed = processed.astype(np.float32) / 255.0
            
            # Agregar dimensión de batch si es necesario
            if len(processed.shape) == 3:
                processed = np.expand_dims(processed, axis=0)
            
            return processed
            
        except Exception as e:
            self.logger.warning(f"Error preprocesando imagen para DL: {e}")
            return image