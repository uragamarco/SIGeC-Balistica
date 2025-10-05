#!/usr/bin/env python3
"""
Pestaña de Análisis Comparativo Balístico
Sistema SEACABAr - Análisis de Cartuchos y Balas Automático

Dos modos especializados:
1. Comparación Directa (Evidencia A vs Evidencia B) con análisis CMC
2. Búsqueda en Base de Datos con ranking y visualización de coincidencias
"""

import os
import json
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout,
    QLabel, QPushButton, QLineEdit, QTextEdit, QComboBox, QSpinBox,
    QCheckBox, QGroupBox, QScrollArea, QSplitter, QFrame, QSpacerItem,
    QSizePolicy, QFileDialog, QMessageBox, QProgressBar, QTabWidget,
    QListWidget, QListWidgetItem, QSlider, QDoubleSpinBox, QButtonGroup,
    QRadioButton, QTableWidget, QTableWidgetItem, QHeaderView
)
from PyQt5.QtCore import Qt, pyqtSignal, QThread, QTimer
from PyQt5.QtGui import QFont, QPixmap, QIcon, QPainter, QPen, QColor

from .shared_widgets import (
    ImageDropZone, ResultCard, CollapsiblePanel, StepIndicator, 
    ProgressCard, ImageViewer
)
from .model_selector_dialog import ModelSelectorDialog

# Importaciones para validación NIST
try:
    from image_processing.nist_compliance_validator import NISTComplianceValidator, NISTProcessingReport
    from nist_standards.quality_metrics import NISTQualityMetrics, NISTQualityReport
    from nist_standards.afte_conclusions import AFTEConclusionEngine, AFTEConclusion
    from nist_standards.validation_protocols import NISTValidationProtocols
    NIST_AVAILABLE = True
except ImportError:
    NIST_AVAILABLE = False

# Importaciones para Deep Learning
try:
    from deep_learning.ballistic_dl_models import BallisticCNN
    from deep_learning.models.siamese_models import SiameseNetwork
    from deep_learning.config.unified_config import ModelConfig, ModelType
    DEEP_LEARNING_AVAILABLE = True
except ImportError:
    DEEP_LEARNING_AVAILABLE = False

class BallisticComparisonWorker(QThread):
    """Worker thread especializado para comparaciones balísticas"""
    
    progressUpdated = pyqtSignal(int, str)
    comparisonCompleted = pyqtSignal(dict)
    comparisonError = pyqtSignal(str)
    
    def __init__(self, comparison_params: dict):
        super().__init__()
        self.comparison_params = comparison_params
        
    def run(self):
        """Ejecuta la comparación balística en segundo plano"""
        try:
            mode = self.comparison_params.get('mode', 'direct')
            
            if mode == 'direct':
                self.run_direct_ballistic_comparison()
            else:
                self.run_ballistic_database_search()
                
        except Exception as e:
            self.comparisonError.emit(str(e))
            
    def run_direct_ballistic_comparison(self):
        """Ejecuta comparación directa entre dos evidencias balísticas"""
        steps = [
            (5, "Inicializando análisis balístico..."),
            (15, "Extrayendo características de firing pin..."),
            (25, "Analizando breech face patterns..."),
            (35, "Detectando marcas de extractor/eyector..."),
            (45, "Calculando correlación CMC..."),
            (60, "Evaluando congruencia de células..."),
            (75, "Generando curvas CMC..."),
            (85, "Aplicando criterios AFTE..."),
            (95, "Generando visualizaciones..."),
            (100, "Comparación balística completada")
        ]
        
        for progress, message in steps:
            self.progressUpdated.emit(progress, message)
            self.msleep(400)
            
        # Simular resultados de comparación balística
        evidence_type = self.comparison_params.get('evidence_type', 'cartridge_case')
        cmc_score = np.random.uniform(0.65, 0.95)  # Simulado
        
        # Determinar conclusión AFTE basada en CMC score
        if cmc_score >= 0.85:
            afte_conclusion = "Identification"
            result_type = "success"
        elif cmc_score >= 0.70:
            afte_conclusion = "Inconclusive"
            result_type = "warning"
        else:
            afte_conclusion = "Elimination"
            result_type = "error"
        
        results = {
            'mode': 'direct',
            'evidence_type': evidence_type,
            'image_a': self.comparison_params.get('image_a'),
            'image_b': self.comparison_params.get('image_b'),
            'cmc_score': cmc_score,
            'afte_conclusion': afte_conclusion,
            'result_type': result_type,
            'ballistic_features': {
                'firing_pin_correlation': np.random.uniform(0.6, 0.9),
                'breech_face_correlation': np.random.uniform(0.7, 0.95),
                'extractor_marks_correlation': np.random.uniform(0.5, 0.85),
                'striation_correlation': np.random.uniform(0.6, 0.9) if evidence_type == 'bullet' else None
            },
            'cmc_analysis': {
                'total_cells': 64,
                'valid_cells': 58,
                'congruent_cells': int(58 * cmc_score),
                'convergence_score': np.random.uniform(0.7, 0.95),
                'cell_correlation_threshold': 0.6
            },
            'statistical_analysis': {
                'p_value': np.random.uniform(0.001, 0.05),
                'confidence_interval': [cmc_score - 0.05, cmc_score + 0.05],
                'false_positive_rate': np.random.uniform(0.001, 0.01)
            },
            'visualizations': {
                'cmc_curve': 'path/to/cmc_curve.png',
                'correlation_map': 'path/to/correlation_map.png',
                'feature_overlay': 'path/to/feature_overlay.png',
                'difference_map': 'path/to/difference_map.png'
            },
            'quality_metrics': {
                'image_a_quality': np.random.uniform(0.7, 0.95),
                'image_b_quality': np.random.uniform(0.7, 0.95),
                'alignment_quality': np.random.uniform(0.8, 0.98)
            }
        }
        
        self.comparisonCompleted.emit(results)
        
    def run_ballistic_database_search(self):
        """Ejecuta búsqueda en base de datos balística"""
        steps = [
            (5, "Preparando imagen de consulta..."),
            (15, "Extrayendo características balísticas..."),
            (25, "Indexando características en vector space..."),
            (40, "Buscando coincidencias en base de datos..."),
            (60, "Calculando scores CMC para candidatos..."),
            (75, "Aplicando filtros de calidad..."),
            (85, "Ordenando por relevancia balística..."),
            (95, "Generando reportes de coincidencias..."),
            (100, "Búsqueda balística completada")
        ]
        
        for progress, message in steps:
            self.progressUpdated.emit(progress, message)
            self.msleep(350)
            
        # Simular resultados de búsqueda balística
        evidence_type = self.comparison_params.get('evidence_type', 'cartridge_case')
        
        results = {
            'mode': 'database',
            'evidence_type': evidence_type,
            'query_image': self.comparison_params.get('query_image'),
            'total_searched': 2847,
            'candidates_found': 23,
            'high_confidence_matches': 8,
            'search_time': 3.7,
            'results': [
                {
                    'id': 'BAL_001_CC',
                    'path': 'db/ballistic/cartridge_cases/bal_001.jpg',
                    'cmc_score': 0.92,
                    'afte_conclusion': 'Identification',
                    'case_number': 'CASO-BAL-2024-001',
                    'weapon_type': 'Pistol 9mm',
                    'date_added': '2024-01-15',
                    'metadata': {
                        'caliber': '9mm Luger',
                        'manufacturer': 'Federal',
                        'firing_pin_type': 'Rectangular',
                        'location': 'Crime Scene Alpha'
                    }
                },
                {
                    'id': 'BAL_045_CC',
                    'path': 'db/ballistic/cartridge_cases/bal_045.jpg',
                    'cmc_score': 0.87,
                    'afte_conclusion': 'Inconclusive',
                    'case_number': 'CASO-BAL-2024-003',
                    'weapon_type': 'Pistol 9mm',
                    'date_added': '2024-01-20',
                    'metadata': {
                        'caliber': '9mm Luger',
                        'manufacturer': 'Winchester',
                        'firing_pin_type': 'Circular',
                        'location': 'Crime Scene Beta'
                    }
                },
                {
                    'id': 'BAL_123_CC',
                    'path': 'db/ballistic/cartridge_cases/bal_123.jpg',
                    'cmc_score': 0.74,
                    'afte_conclusion': 'Inconclusive',
                    'case_number': 'CASO-BAL-2024-007',
                    'weapon_type': 'Revolver .38',
                    'date_added': '2024-02-01',
                    'metadata': {
                        'caliber': '.38 Special',
                        'manufacturer': 'Remington',
                        'firing_pin_type': 'Circular',
                        'location': 'Crime Scene Gamma'
                    }
                }
            ]
        }
        
        self.comparisonCompleted.emit(results)

class CMCVisualizationWidget(QWidget):
    """Widget especializado para visualizar curvas CMC y análisis estadístico"""
    
    def __init__(self):
        super().__init__()
        self.cmc_data = None
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Título
        title = QLabel("Análisis CMC (Congruent Matching Cells)")
        title.setProperty("class", "subtitle")
        layout.addWidget(title)
        
        # Área de visualización
        self.visualization_area = QLabel("Cargar datos CMC para visualizar curva")
        self.visualization_area.setMinimumHeight(200)
        self.visualization_area.setStyleSheet("border: 1px solid #ccc; background: #f9f9f9;")
        self.visualization_area.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.visualization_area)
        
    def update_cmc_data(self, cmc_data: dict):
        """Actualiza la visualización con nuevos datos CMC"""
        self.cmc_data = cmc_data
        self.render_cmc_visualization()
        
    def render_cmc_visualization(self):
        """Renderiza la visualización CMC"""
        if not self.cmc_data:
            return
            
        # Crear pixmap para dibujar
        pixmap = QPixmap(400, 200)
        pixmap.fill(QColor(255, 255, 255))
        
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Dibujar curva CMC simulada
        pen = QPen(QColor(0, 120, 215), 2)
        painter.setPen(pen)
        
        # Simular curva CMC
        points = []
        for i in range(50):
            x = i * 8
            y = 180 - (self.cmc_data.get('cmc_score', 0.8) * 160 * (1 - np.exp(-i/10)))
            points.append((x, y))
            
        for i in range(len(points) - 1):
            painter.drawLine(int(points[i][0]), int(points[i][1]), 
                           int(points[i+1][0]), int(points[i+1][1]))
        
        painter.end()
        self.visualization_area.setPixmap(pixmap)

class ComparisonTab(QWidget):
    """Pestaña de análisis comparativo balístico especializada"""
    
    comparisonCompleted = pyqtSignal(dict)
    
    def __init__(self):
        super().__init__()
        self.current_mode = 'direct'
        self.comparison_data = {}
        self.comparison_worker = None
        self.selected_db_result = None
        self.setup_ui()
        self.setup_connections()
        
    def setup_ui(self):
        """Configura la interfaz especializada para análisis balístico"""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)
        
        # Header con información balística
        self.setup_ballistic_header()
        main_layout.addWidget(self.header_frame)
        
        # Contenido principal con modos especializados
        self.setup_ballistic_mode_tabs()
        main_layout.addWidget(self.mode_tabs)
        
    def setup_ballistic_header(self):
        """Configura el header especializado para análisis balístico"""
        self.header_frame = QFrame()
        self.header_frame.setProperty("class", "header-section")
        
        layout = QVBoxLayout(self.header_frame)
        
        # Título principal
        title_layout = QHBoxLayout()
        title_label = QLabel("🔬 Análisis Comparativo Balístico")
        title_label.setProperty("class", "title")
        title_layout.addWidget(title_label)
        
        title_layout.addStretch()
        
        # Selector de tipo de evidencia
        evidence_label = QLabel("Tipo de Evidencia:")
        evidence_label.setProperty("class", "body")
        title_layout.addWidget(evidence_label)
        
        self.evidence_type_combo = QComboBox()
        self.evidence_type_combo.addItems([
            "Casquillo (Cartridge Case)",
            "Bala (Bullet)",
            "Fragmento Balístico"
        ])
        self.evidence_type_combo.setMinimumWidth(200)
        title_layout.addWidget(self.evidence_type_combo)
        
        layout.addLayout(title_layout)
        
        # Selector de modo de comparación
        mode_layout = QHBoxLayout()
        mode_label = QLabel("Modo de Análisis:")
        mode_label.setProperty("class", "body")
        mode_layout.addWidget(mode_label)
        
        self.mode_combo = QComboBox()
        self.mode_combo.addItems([
            "🔄 Comparación Directa (A vs B)",
            "🔍 Búsqueda en Base de Datos Balística"
        ])
        self.mode_combo.setMinimumWidth(250)
        mode_layout.addWidget(self.mode_combo)
        
        mode_layout.addStretch()
        
        # Indicador de estándares
        standards_label = QLabel("📋 Cumple estándares NIST/AFTE")
        standards_label.setProperty("class", "caption")
        standards_label.setStyleSheet("color: #28a745; font-weight: bold;")
        mode_layout.addWidget(standards_label)
        
        layout.addLayout(mode_layout)
        
    def setup_ballistic_mode_tabs(self):
        """Configura las pestañas especializadas para análisis balístico"""
        self.mode_tabs = QTabWidget()
        self.mode_tabs.setProperty("class", "mode-tabs")
        
        # Modo 1: Comparación Directa Balística
        self.direct_tab = self.create_direct_ballistic_tab()
        self.mode_tabs.addTab(self.direct_tab, "🔄 Comparación Directa")
        
        # Modo 2: Búsqueda en Base de Datos Balística
        self.database_tab = self.create_database_ballistic_tab()
        self.mode_tabs.addTab(self.database_tab, "🔍 Búsqueda en BD")
        
    def create_direct_ballistic_tab(self) -> QWidget:
        """Crea la pestaña de comparación directa balística"""
        tab = QWidget()
        main_layout = QHBoxLayout(tab)
        main_layout.setSpacing(20)
        
        # Panel izquierdo - Configuración balística
        config_panel = self.create_direct_ballistic_config_panel()
        main_layout.addWidget(config_panel, 1)
        
        # Panel derecho - Visualización y resultados
        visual_panel = self.create_direct_ballistic_visual_panel()
        main_layout.addWidget(visual_panel, 2)
        
        return tab
        
    def create_direct_ballistic_config_panel(self) -> QWidget:
        """Crea el panel de configuración para comparación directa balística"""
        panel = QFrame()
        panel.setProperty("class", "panel")
        
        layout = QVBoxLayout(panel)
        layout.setSpacing(20)
        
        # Indicador de pasos balísticos
        steps = ["Cargar Evidencias", "Config. Balística", "Análisis CMC", "Conclusión AFTE"]
        self.direct_step_indicator = StepIndicator(steps)
        layout.addWidget(self.direct_step_indicator)
        
        # Paso 1: Cargar evidencias balísticas
        evidence_group = QGroupBox("Paso 1: Cargar Evidencias Balísticas")
        evidence_layout = QVBoxLayout(evidence_group)
        
        # Drop zones especializadas
        drop_layout = QHBoxLayout()
        
        self.evidence_a_zone = ImageDropZone("Evidencia A", "Arrastrar primera evidencia\n(casquillo, bala, etc.)")
        drop_layout.addWidget(self.evidence_a_zone)
        
        vs_label = QLabel("VS")
        vs_label.setProperty("class", "title")
        vs_label.setAlignment(Qt.AlignCenter)
        vs_label.setFixedWidth(40)
        drop_layout.addWidget(vs_label)
        
        self.evidence_b_zone = ImageDropZone("Evidencia B", "Arrastrar segunda evidencia\n(mismo tipo)")
        drop_layout.addWidget(self.evidence_b_zone)
        
        evidence_layout.addLayout(drop_layout)
        layout.addWidget(evidence_group)
        
        # Paso 2: Configuración de análisis balístico
        ballistic_config_group = QGroupBox("Paso 2: Configuración de Análisis Balístico")
        ballistic_config_group.setEnabled(False)
        ballistic_config_layout = QFormLayout(ballistic_config_group)
        
        # Método de análisis balístico
        self.ballistic_method_combo = QComboBox()
        self.ballistic_method_combo.addItems([
            "CMC (Congruent Matching Cells)",
            "Análisis de Características Individuales",
            "Correlación de Patrones de Estriado",
            "Análisis Multiespectal Combinado"
        ])
        ballistic_config_layout.addRow("Método de Análisis:", self.ballistic_method_combo)
        
        # Umbral de correlación CMC
        self.cmc_threshold_slider = QSlider(Qt.Horizontal)
        self.cmc_threshold_slider.setRange(50, 95)
        self.cmc_threshold_slider.setValue(75)
        self.cmc_threshold_label = QLabel("0.75")
        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(self.cmc_threshold_slider)
        threshold_layout.addWidget(self.cmc_threshold_label)
        ballistic_config_layout.addRow("Umbral CMC:", threshold_layout)
        
        # Criterios AFTE
        afte_group = QGroupBox("Criterios AFTE")
        afte_layout = QVBoxLayout(afte_group)
        
        self.afte_identification_rb = QRadioButton("Identification (≥85% CMC)")
        self.afte_inconclusive_rb = QRadioButton("Inconclusive (70-84% CMC)")
        self.afte_elimination_rb = QRadioButton("Elimination (<70% CMC)")
        self.afte_auto_rb = QRadioButton("Determinación Automática")
        self.afte_auto_rb.setChecked(True)
        
        afte_layout.addWidget(self.afte_identification_rb)
        afte_layout.addWidget(self.afte_inconclusive_rb)
        afte_layout.addWidget(self.afte_elimination_rb)
        afte_layout.addWidget(self.afte_auto_rb)
        
        ballistic_config_layout.addRow(afte_group)
        layout.addWidget(ballistic_config_group)
        
        # Opciones avanzadas balísticas
        self.advanced_ballistic_panel = CollapsiblePanel("Opciones Avanzadas de Análisis Balístico")
        
        advanced_content = QWidget()
        advanced_layout = QVBoxLayout(advanced_content)
        
        # Características balísticas específicas
        features_group = QGroupBox("Características a Analizar")
        features_layout = QVBoxLayout(features_group)
        
        self.analyze_firing_pin_cb = QCheckBox("Marcas de percutor (Firing Pin)")
        self.analyze_breech_face_cb = QCheckBox("Patrones de cara de recámara (Breech Face)")
        self.analyze_extractor_cb = QCheckBox("Marcas de extractor/eyector")
        self.analyze_striations_cb = QCheckBox("Patrones de estriado (para balas)")
        self.analyze_land_groove_cb = QCheckBox("Análisis de campos y estrías")
        
        self.analyze_firing_pin_cb.setChecked(True)
        self.analyze_breech_face_cb.setChecked(True)
        self.analyze_extractor_cb.setChecked(True)
        
        features_layout.addWidget(self.analyze_firing_pin_cb)
        features_layout.addWidget(self.analyze_breech_face_cb)
        features_layout.addWidget(self.analyze_extractor_cb)
        features_layout.addWidget(self.analyze_striations_cb)
        features_layout.addWidget(self.analyze_land_groove_cb)
        
        advanced_layout.addWidget(features_group)
        
        # Validación NIST
        nist_group = QGroupBox("Validación NIST")
        nist_layout = QVBoxLayout(nist_group)
        
        self.nist_quality_validation_cb = QCheckBox("Validación de calidad de imagen NIST")
        self.nist_metadata_validation_cb = QCheckBox("Validación de metadatos NIST")
        self.nist_chain_custody_cb = QCheckBox("Verificación de cadena de custodia")
        
        self.nist_quality_validation_cb.setChecked(True)
        
        nist_layout.addWidget(self.nist_quality_validation_cb)
        nist_layout.addWidget(self.nist_metadata_validation_cb)
        nist_layout.addWidget(self.nist_chain_custody_cb)
        
        advanced_layout.addWidget(nist_group)
        
        # Deep Learning (si está disponible)
        if DEEP_LEARNING_AVAILABLE:
            dl_group = QGroupBox("Análisis con Deep Learning")
            dl_group.setProperty("class", "dl-group")
            dl_layout = QVBoxLayout(dl_group)
            
            self.enable_dl_comparison_cb = QCheckBox("Habilitar análisis con Deep Learning")
            self.enable_dl_comparison_cb.setProperty("class", "dl-checkbox")
            dl_layout.addWidget(self.enable_dl_comparison_cb)
            
            # Selector de modelo
            model_layout = QFormLayout()
            self.dl_comparison_model_combo = QComboBox()
            self.dl_comparison_model_combo.setProperty("class", "dl-combo")
            self.dl_comparison_model_combo.addItems([
                "SiameseNetwork - Comparación de similitud",
                "BallisticCNN - Extracción de características",
                "Ensemble - Combinación de modelos"
            ])
            self.dl_comparison_model_combo.setEnabled(False)
            model_layout.addRow("Modelo DL:", self.dl_comparison_model_combo)
            
            # Umbral de confianza
            self.dl_confidence_spin = QDoubleSpinBox()
            self.dl_confidence_spin.setProperty("class", "dl-spin")
            self.dl_confidence_spin.setRange(0.1, 1.0)
            self.dl_confidence_spin.setSingleStep(0.05)
            self.dl_confidence_spin.setValue(0.85)
            self.dl_confidence_spin.setEnabled(False)
            model_layout.addRow("Confianza mínima:", self.dl_confidence_spin)
            
            dl_layout.addLayout(model_layout)
            
            # Botón de configuración avanzada
            self.dl_advanced_comparison_button = QPushButton("⚙️ Configuración Avanzada")
            self.dl_advanced_comparison_button.setProperty("class", "dl-advanced")
            self.dl_advanced_comparison_button.setEnabled(False)
            self.dl_advanced_comparison_button.clicked.connect(self.open_comparison_model_selector)
            dl_layout.addWidget(self.dl_advanced_comparison_button)
            
            # Conectar señales
            self.enable_dl_comparison_cb.toggled.connect(self.toggle_dl_comparison_options)
            advanced_layout.addWidget(dl_group)
            
            # Conectar señales
            self.enable_dl_comparison_cb.toggled.connect(self.dl_comparison_model_combo.setEnabled)
            self.enable_dl_comparison_cb.toggled.connect(self.dl_confidence_spin.setEnabled)
        
        self.advanced_ballistic_panel.add_content_widget(advanced_content)
        layout.addWidget(self.advanced_ballistic_panel)
        
        # Botón de análisis
        self.analyze_button = QPushButton("🔬 Iniciar Análisis Balístico")
        self.analyze_button.setProperty("class", "primary-button")
        self.analyze_button.setEnabled(False)
        layout.addWidget(self.analyze_button)
        
        # Progress card
        self.direct_progress_card = ProgressCard("Análisis en progreso...")
        self.direct_progress_card.hide()
        layout.addWidget(self.direct_progress_card)
        
        layout.addStretch()
        
        return panel
        
    def create_direct_ballistic_visual_panel(self) -> QWidget:
        """Crea el panel de visualización para comparación directa balística"""
        panel = QFrame()
        panel.setProperty("class", "panel")
        
        layout = QVBoxLayout(panel)
        layout.setSpacing(15)
        
        # Título
        title = QLabel("Resultados de Comparación Balística")
        title.setProperty("class", "subtitle")
        layout.addWidget(title)
        
        # Área de resultados con tabs
        self.results_tabs = QTabWidget()
        
        # Tab 1: Visualización CMC
        cmc_tab = QWidget()
        cmc_layout = QVBoxLayout(cmc_tab)
        
        self.cmc_visualization = CMCVisualizationWidget()
        cmc_layout.addWidget(self.cmc_visualization)
        
        # Métricas CMC
        cmc_metrics_group = QGroupBox("Métricas CMC")
        cmc_metrics_layout = QGridLayout(cmc_metrics_group)
        
        self.cmc_score_label = QLabel("Score CMC: --")
        self.total_cells_label = QLabel("Células Totales: --")
        self.valid_cells_label = QLabel("Células Válidas: --")
        self.congruent_cells_label = QLabel("Células Congruentes: --")
        
        cmc_metrics_layout.addWidget(self.cmc_score_label, 0, 0)
        cmc_metrics_layout.addWidget(self.total_cells_label, 0, 1)
        cmc_metrics_layout.addWidget(self.valid_cells_label, 1, 0)
        cmc_metrics_layout.addWidget(self.congruent_cells_label, 1, 1)
        
        cmc_layout.addWidget(cmc_metrics_group)
        
        self.results_tabs.addTab(cmc_tab, "📊 Análisis CMC")
        
        # Tab 2: Características balísticas
        features_tab = QWidget()
        features_layout = QVBoxLayout(features_tab)
        
        self.ballistic_features_text = QTextEdit()
        self.ballistic_features_text.setReadOnly(True)
        self.ballistic_features_text.setMaximumHeight(200)
        features_layout.addWidget(self.ballistic_features_text)
        
        self.results_tabs.addTab(features_tab, "🎯 Características")
        
        # Tab 3: Conclusión AFTE
        conclusion_tab = QWidget()
        conclusion_layout = QVBoxLayout(conclusion_tab)
        
        self.afte_conclusion_card = ResultCard("Conclusión AFTE", "Pendiente de análisis")
        conclusion_layout.addWidget(self.afte_conclusion_card)
        
        # Detalles estadísticos
        stats_group = QGroupBox("Análisis Estadístico")
        stats_layout = QFormLayout(stats_group)
        
        self.p_value_label = QLabel("--")
        self.confidence_interval_label = QLabel("--")
        self.false_positive_rate_label = QLabel("--")
        
        stats_layout.addRow("Valor p:", self.p_value_label)
        stats_layout.addRow("Intervalo de Confianza:", self.confidence_interval_label)
        stats_layout.addRow("Tasa de Falsos Positivos:", self.false_positive_rate_label)
        
        conclusion_layout.addWidget(stats_group)
        
        self.results_tabs.addTab(conclusion_tab, "⚖️ Conclusión")
        
        layout.addWidget(self.results_tabs)
        
        return panel
        
    def create_database_ballistic_tab(self) -> QWidget:
        """Crea la pestaña de búsqueda en base de datos balística"""
        tab = QWidget()
        main_layout = QHBoxLayout(tab)
        main_layout.setSpacing(20)
        
        # Panel izquierdo - Configuración de búsqueda
        search_panel = self.create_database_ballistic_config_panel()
        main_layout.addWidget(search_panel, 1)
        
        # Panel derecho - Resultados de búsqueda
        results_panel = self.create_database_ballistic_results_panel()
        main_layout.addWidget(results_panel, 2)
        
        return tab
        
    def create_database_ballistic_config_panel(self) -> QWidget:
        """Crea el panel de configuración para búsqueda en base de datos balística"""
        panel = QFrame()
        panel.setProperty("class", "panel")
        
        layout = QVBoxLayout(panel)
        layout.setSpacing(20)
        
        # Indicador de pasos
        steps = ["Cargar Consulta", "Config. Búsqueda", "Buscar", "Analizar Resultados"]
        self.db_step_indicator = StepIndicator(steps)
        layout.addWidget(self.db_step_indicator)
        
        # Paso 1: Imagen de consulta
        query_group = QGroupBox("Paso 1: Evidencia de Consulta")
        query_layout = QVBoxLayout(query_group)
        
        self.query_evidence_zone = ImageDropZone("Evidencia de Consulta", "Arrastrar evidencia balística\npara buscar coincidencias")
        query_layout.addWidget(self.query_evidence_zone)
        
        layout.addWidget(query_group)
        
        # Paso 2: Configuración de búsqueda balística
        search_config_group = QGroupBox("Paso 2: Configuración de Búsqueda Balística")
        search_config_group.setEnabled(False)
        search_config_layout = QFormLayout(search_config_group)
        
        # Filtros balísticos
        self.caliber_filter_combo = QComboBox()
        self.caliber_filter_combo.addItems([
            "Todos los calibres",
            "9mm Luger",
            ".40 S&W",
            ".45 ACP",
            ".38 Special",
            ".357 Magnum",
            "5.56mm NATO",
            "7.62mm NATO"
        ])
        search_config_layout.addRow("Filtro por Calibre:", self.caliber_filter_combo)
        
        self.weapon_type_filter_combo = QComboBox()
        self.weapon_type_filter_combo.addItems([
            "Todos los tipos",
            "Pistola",
            "Revólver",
            "Rifle",
            "Escopeta",
            "Subfusil"
        ])
        search_config_layout.addRow("Tipo de Arma:", self.weapon_type_filter_combo)
        
        # Umbral de similitud
        self.similarity_threshold_slider = QSlider(Qt.Horizontal)
        self.similarity_threshold_slider.setRange(50, 95)
        self.similarity_threshold_slider.setValue(70)
        self.similarity_threshold_label = QLabel("0.70")
        similarity_layout = QHBoxLayout()
        similarity_layout.addWidget(self.similarity_threshold_slider)
        similarity_layout.addWidget(self.similarity_threshold_label)
        search_config_layout.addRow("Umbral de Similitud:", similarity_layout)
        
        # Número máximo de resultados
        self.max_results_spinbox = QSpinBox()
        self.max_results_spinbox.setRange(5, 100)
        self.max_results_spinbox.setValue(20)
        search_config_layout.addRow("Máx. Resultados:", self.max_results_spinbox)
        
        layout.addWidget(search_config_group)
        
        # Opciones avanzadas de búsqueda
        self.advanced_search_panel = CollapsiblePanel("Opciones Avanzadas de Búsqueda")
        
        advanced_search_content = QWidget()
        advanced_search_layout = QVBoxLayout(advanced_search_content)
        
        # Filtros temporales
        temporal_group = QGroupBox("Filtros Temporales")
        temporal_layout = QFormLayout(temporal_group)
        
        self.date_from_edit = QLineEdit()
        self.date_from_edit.setPlaceholderText("YYYY-MM-DD")
        temporal_layout.addRow("Fecha Desde:", self.date_from_edit)
        
        self.date_to_edit = QLineEdit()
        self.date_to_edit.setPlaceholderText("YYYY-MM-DD")
        temporal_layout.addRow("Fecha Hasta:", self.date_to_edit)
        
        advanced_search_layout.addWidget(temporal_group)
        
        # Filtros de metadatos
        metadata_group = QGroupBox("Filtros de Metadatos")
        metadata_layout = QVBoxLayout(metadata_group)
        
        self.case_number_filter_edit = QLineEdit()
        self.case_number_filter_edit.setPlaceholderText("Número de caso...")
        metadata_layout.addWidget(QLabel("Número de Caso:"))
        metadata_layout.addWidget(self.case_number_filter_edit)
        
        self.location_filter_edit = QLineEdit()
        self.location_filter_edit.setPlaceholderText("Ubicación...")
        metadata_layout.addWidget(QLabel("Ubicación:"))
        metadata_layout.addWidget(self.location_filter_edit)
        
        advanced_search_layout.addWidget(metadata_group)
        
        # Deep Learning para búsqueda (si está disponible)
        if DEEP_LEARNING_AVAILABLE:
            dl_search_group = QGroupBox("Búsqueda con Deep Learning")
            dl_search_group.setProperty("class", "dl-group")
            dl_search_layout = QVBoxLayout(dl_search_group)
            
            self.enable_dl_search_cb = QCheckBox("Habilitar búsqueda con Deep Learning")
            self.enable_dl_search_cb.setProperty("class", "dl-checkbox")
            dl_search_layout.addWidget(self.enable_dl_search_cb)
            
            # Configuración de modelo para búsqueda
            search_model_layout = QFormLayout()
            self.dl_search_model_combo = QComboBox()
            self.dl_search_model_combo.setProperty("class", "dl-combo")
            self.dl_search_model_combo.addItems([
                "SiameseNetwork - Búsqueda por similitud",
                "BallisticCNN - Extracción de características",
                "Ensemble - Búsqueda híbrida"
            ])
            self.dl_search_model_combo.setEnabled(False)
            search_model_layout.addRow("Modelo DL:", self.dl_search_model_combo)
            
            # Configuración específica para búsqueda
            self.dl_search_confidence_spin = QDoubleSpinBox()
            self.dl_search_confidence_spin.setProperty("class", "dl-spin")
            self.dl_search_confidence_spin.setRange(0.1, 1.0)
            self.dl_search_confidence_spin.setSingleStep(0.05)
            self.dl_search_confidence_spin.setValue(0.75)
            self.dl_search_confidence_spin.setEnabled(False)
            search_model_layout.addRow("Confianza mínima:", self.dl_search_confidence_spin)
            
            self.dl_rerank_results_cb = QCheckBox("Re-ranking con DL")
            self.dl_rerank_results_cb.setProperty("class", "dl-checkbox")
            self.dl_rerank_results_cb.setEnabled(False)
            search_model_layout.addRow("", self.dl_rerank_results_cb)
            
            dl_search_layout.addLayout(search_model_layout)
            
            # Botón de configuración avanzada para búsqueda
            self.dl_advanced_search_button = QPushButton("⚙️ Configuración Avanzada")
            self.dl_advanced_search_button.setProperty("class", "dl-advanced")
            self.dl_advanced_search_button.setEnabled(False)
            self.dl_advanced_search_button.clicked.connect(self.open_search_model_selector)
            dl_search_layout.addWidget(self.dl_advanced_search_button)
            
            advanced_search_layout.addWidget(dl_search_group)
            
            # Conectar señales
            self.enable_dl_search_cb.toggled.connect(self.toggle_dl_search_options)
        
        self.advanced_search_panel.add_content_widget(advanced_search_content)
        layout.addWidget(self.advanced_search_panel)
        
        # Botón de búsqueda
        self.search_button = QPushButton("🔍 Buscar en Base de Datos")
        self.search_button.setProperty("class", "primary-button")
        self.search_button.setEnabled(False)
        layout.addWidget(self.search_button)
        
        # Progress card para búsqueda
        self.search_progress_card = ProgressCard("Búsqueda en progreso...")
        self.search_progress_card.hide()
        layout.addWidget(self.search_progress_card)
        
        layout.addStretch()
        
        return panel
        
    def create_database_ballistic_results_panel(self) -> QWidget:
        """Crea el panel de resultados para búsqueda en base de datos balística"""
        panel = QFrame()
        panel.setProperty("class", "panel")
        
        layout = QVBoxLayout(panel)
        layout.setSpacing(15)
        
        # Título y estadísticas
        header_layout = QHBoxLayout()
        
        title = QLabel("Resultados de Búsqueda Balística")
        title.setProperty("class", "subtitle")
        header_layout.addWidget(title)
        
        header_layout.addStretch()
        
        self.search_stats_label = QLabel("Estadísticas de búsqueda")
        self.search_stats_label.setProperty("class", "caption")
        header_layout.addWidget(self.search_stats_label)
        
        layout.addLayout(header_layout)
        
        # Lista de resultados
        self.results_list = QListWidget()
        self.results_list.setMinimumHeight(300)
        layout.addWidget(self.results_list)
        
        # Panel de comparación detallada
        comparison_group = QGroupBox("Comparación Detallada")
        comparison_layout = QVBoxLayout(comparison_group)
        
        # Visualización lado a lado
        comparison_visual_layout = QHBoxLayout()
        
        # Imagen de consulta
        query_frame = QFrame()
        query_frame.setProperty("class", "image-frame")
        query_layout = QVBoxLayout(query_frame)
        query_layout.addWidget(QLabel("Evidencia de Consulta"))
        self.query_image_viewer = ImageViewer()
        self.query_image_viewer.setMinimumSize(150, 150)
        query_layout.addWidget(self.query_image_viewer)
        comparison_visual_layout.addWidget(query_frame)
        
        # Indicador de comparación
        comparison_indicator = QLabel("⚖️")
        comparison_indicator.setAlignment(Qt.AlignCenter)
        comparison_indicator.setProperty("class", "title")
        comparison_visual_layout.addWidget(comparison_indicator)
        
        # Imagen seleccionada
        selected_frame = QFrame()
        selected_frame.setProperty("class", "image-frame")
        selected_layout = QVBoxLayout(selected_frame)
        selected_layout.addWidget(QLabel("Resultado Seleccionado"))
        self.selected_image_viewer = ImageViewer()
        self.selected_image_viewer.setMinimumSize(150, 150)
        selected_layout.addWidget(self.selected_image_viewer)
        comparison_visual_layout.addWidget(selected_frame)
        
        comparison_layout.addLayout(comparison_visual_layout)
        
        # Métricas de comparación
        metrics_layout = QGridLayout()
        
        self.selected_cmc_score_label = QLabel("Score CMC: --")
        self.selected_afte_conclusion_label = QLabel("Conclusión AFTE: --")
        self.selected_confidence_label = QLabel("Confianza: --")
        self.selected_case_info_label = QLabel("Info del Caso: --")
        
        metrics_layout.addWidget(self.selected_cmc_score_label, 0, 0)
        metrics_layout.addWidget(self.selected_afte_conclusion_label, 0, 1)
        metrics_layout.addWidget(self.selected_confidence_label, 1, 0)
        metrics_layout.addWidget(self.selected_case_info_label, 1, 1)
        
        comparison_layout.addLayout(metrics_layout)
        
        layout.addWidget(comparison_group)
        
        return panel
        
    def setup_connections(self):
        """Configura las conexiones de señales"""
        # Conexiones de modo
        self.mode_combo.currentIndexChanged.connect(self.on_mode_changed)
        self.mode_tabs.currentChanged.connect(self.sync_mode_selection)
        
        # Conexiones de tipo de evidencia
        self.evidence_type_combo.currentTextChanged.connect(self.on_evidence_type_changed)
        
        # Conexiones de carga de imágenes - Modo directo
        self.evidence_a_zone.imageLoaded.connect(self.on_evidence_a_loaded)
        self.evidence_b_zone.imageLoaded.connect(self.on_evidence_b_loaded)
        
        # Conexiones de carga de imágenes - Modo búsqueda
        self.query_evidence_zone.imageLoaded.connect(self.on_query_evidence_loaded)
        
        # Conexiones de sliders
        self.cmc_threshold_slider.valueChanged.connect(self.update_cmc_threshold_label)
        self.similarity_threshold_slider.valueChanged.connect(self.update_similarity_threshold_label)
        
        # Conexiones de botones
        self.analyze_button.clicked.connect(self.start_ballistic_comparison)
        self.search_button.clicked.connect(self.start_ballistic_search)
        
    def toggle_dl_comparison_options(self, enabled: bool):
        """Habilita/deshabilita las opciones de Deep Learning para comparación con manejo de errores"""
        try:
            if not DEEP_LEARNING_AVAILABLE:
                if enabled:
                    QMessageBox.warning(
                        self, 
                        "Deep Learning No Disponible",
                        "Los módulos de Deep Learning no están disponibles.\n"
                        "Instale las dependencias necesarias:\n"
                        "pip install torch torchvision tensorflow"
                    )
                    # Desmarcar el checkbox si existe
                    if hasattr(self, 'enable_dl_comparison_cb'):
                        self.enable_dl_comparison_cb.setChecked(False)
                return
                
            # Habilitar/deshabilitar controles con verificaciones
            if hasattr(self, 'dl_comparison_model_combo'):
                self.dl_comparison_model_combo.setEnabled(enabled)
            if hasattr(self, 'dl_confidence_spin'):
                self.dl_confidence_spin.setEnabled(enabled)
            if hasattr(self, 'dl_advanced_comparison_button'):
                self.dl_advanced_comparison_button.setEnabled(enabled)
                
        except Exception as e:
            print(f"Error en toggle_dl_comparison_options: {e}")
            QMessageBox.critical(
                self, 
                "Error de Configuración",
                f"Error al configurar opciones de Deep Learning:\n{str(e)}"
            )
            # Asegurar que el checkbox esté desmarcado en caso de error
            if hasattr(self, 'enable_dl_comparison_cb'):
                self.enable_dl_comparison_cb.setChecked(False)
            
    def toggle_dl_search_options(self, enabled: bool):
        """Habilita/deshabilita las opciones de Deep Learning para búsqueda con manejo de errores"""
        try:
            if not DEEP_LEARNING_AVAILABLE:
                if enabled:
                    QMessageBox.warning(
                        self, 
                        "Deep Learning No Disponible",
                        "Los módulos de Deep Learning no están disponibles.\n"
                        "Instale las dependencias necesarias:\n"
                        "pip install torch torchvision tensorflow"
                    )
                    # Desmarcar el checkbox si existe
                    if hasattr(self, 'enable_dl_search_cb'):
                        self.enable_dl_search_cb.setChecked(False)
                return
                
            # Habilitar/deshabilitar controles con verificaciones
            if hasattr(self, 'dl_search_model_combo'):
                self.dl_search_model_combo.setEnabled(enabled)
            if hasattr(self, 'dl_search_confidence_spin'):
                self.dl_search_confidence_spin.setEnabled(enabled)
            if hasattr(self, 'dl_rerank_results_cb'):
                self.dl_rerank_results_cb.setEnabled(enabled)
            if hasattr(self, 'dl_advanced_search_button'):
                self.dl_advanced_search_button.setEnabled(enabled)
                
        except Exception as e:
            print(f"Error en toggle_dl_search_options: {e}")
            QMessageBox.critical(
                self, 
                "Error de Configuración",
                f"Error al configurar opciones de Deep Learning:\n{str(e)}"
            )
            # Asegurar que el checkbox esté desmarcado en caso de error
            if hasattr(self, 'enable_dl_search_cb'):
                self.enable_dl_search_cb.setChecked(False)
            
    def open_comparison_model_selector(self):
        """Abre el diálogo de selección de modelos para comparación directa con manejo robusto de errores"""
        try:
            if not DEEP_LEARNING_AVAILABLE:
                QMessageBox.warning(
                    self, 
                    "Deep Learning No Disponible",
                    "Los módulos de Deep Learning no están disponibles.\n"
                    "Instale las dependencias necesarias:\n"
                    "pip install torch torchvision tensorflow\n\n"
                    "Verifique la instalación de las dependencias."
                )
                return
                
            # Verificar que el checkbox esté habilitado
            if hasattr(self, 'enable_dl_comparison_cb') and not self.enable_dl_comparison_cb.isChecked():
                QMessageBox.information(
                    self,
                    "Deep Learning Deshabilitado",
                    "Debe habilitar Deep Learning para comparación antes de configurar modelos."
                )
                return
                
            # Obtener configuración actual con manejo de errores
            current_config = self.get_current_comparison_dl_config()
            
            # Importar y crear diálogo con manejo de errores
            try:
                from .model_selector_dialog import ModelSelectorDialog
                dialog = ModelSelectorDialog(self, current_config)
                dialog.modelConfigured.connect(self.on_comparison_model_configured)
                dialog.exec_()
            except ImportError as e:
                QMessageBox.critical(
                    self,
                    "Error de Importación",
                    f"No se pudo cargar el diálogo de selección de modelos:\n{str(e)}\n\n"
                    "Verifique la instalación de los módulos de Deep Learning."
                )
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Error del Diálogo",
                    f"Error al mostrar el diálogo de configuración:\n{str(e)}"
                )
                
        except Exception as e:
            print(f"Error crítico en open_comparison_model_selector: {e}")
            QMessageBox.critical(
                self,
                "Error Crítico",
                f"Error inesperado al abrir configuración de modelos:\n{str(e)}"
            )
        
    def open_search_model_selector(self):
        """Abre el diálogo de selección de modelos para búsqueda en BD con manejo robusto de errores"""
        try:
            if not DEEP_LEARNING_AVAILABLE:
                QMessageBox.warning(
                    self, 
                    "Deep Learning No Disponible",
                    "Los módulos de Deep Learning no están disponibles.\n"
                    "Instale las dependencias necesarias:\n"
                    "pip install torch torchvision tensorflow\n\n"
                    "Verifique la instalación de las dependencias."
                )
                return
                
            # Verificar que el checkbox esté habilitado
            if hasattr(self, 'enable_dl_search_cb') and not self.enable_dl_search_cb.isChecked():
                QMessageBox.information(
                    self,
                    "Deep Learning Deshabilitado",
                    "Debe habilitar Deep Learning para búsqueda antes de configurar modelos."
                )
                return
                
            # Obtener configuración actual con manejo de errores
            current_config = self.get_current_search_dl_config()
            
            # Importar y crear diálogo con manejo de errores
            try:
                from .model_selector_dialog import ModelSelectorDialog
                dialog = ModelSelectorDialog(self, current_config)
                dialog.modelConfigured.connect(self.on_search_model_configured)
                dialog.exec_()
            except ImportError as e:
                QMessageBox.critical(
                    self,
                    "Error de Importación",
                    f"No se pudo cargar el diálogo de selección de modelos:\n{str(e)}\n\n"
                    "Verifique la instalación de los módulos de Deep Learning."
                )
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Error del Diálogo",
                    f"Error al mostrar el diálogo de configuración:\n{str(e)}"
                )
                
        except Exception as e:
            print(f"Error crítico en open_search_model_selector: {e}")
            QMessageBox.critical(
                self,
                "Error Crítico",
                f"Error inesperado al abrir configuración de modelos:\n{str(e)}"
            )
        
    def get_current_comparison_dl_config(self) -> dict:
        """Obtiene la configuración actual de DL para comparación con manejo robusto de errores"""
        try:
            # Verificar disponibilidad de Deep Learning
            if not DEEP_LEARNING_AVAILABLE:
                return {}
                
            # Verificar que el checkbox existe y está habilitado
            if not hasattr(self, 'enable_dl_comparison_cb') or not self.enable_dl_comparison_cb.isChecked():
                return {}
            
            config = {'enabled': True, 'task_type': 'comparison'}
            
            # Obtener configuración con verificaciones
            try:
                if hasattr(self, 'dl_comparison_model_combo'):
                    model_text = self.dl_comparison_model_combo.currentText()
                    config['model_type'] = model_text.split(' - ')[0] if ' - ' in model_text else model_text
                else:
                    config['model_type'] = 'CNN'  # Valor por defecto
                    
                if hasattr(self, 'dl_confidence_spin'):
                    config['confidence_threshold'] = self.dl_confidence_spin.value()
                else:
                    config['confidence_threshold'] = 0.85  # Valor por defecto
                    
            except Exception as e:
                print(f"Warning: Error obteniendo configuración de comparación DL: {e}")
                # Usar valores por defecto en caso de error
                config.update({
                    'model_type': 'CNN',
                    'confidence_threshold': 0.85
                })
                
            return config
            
        except Exception as e:
            print(f"Error crítico obteniendo configuración de comparación DL: {e}")
            return {}  # Retornar configuración vacía en caso de error crítico
        
    def get_current_search_dl_config(self) -> dict:
        """Obtiene la configuración actual de DL para búsqueda con manejo robusto de errores"""
        try:
            # Verificar disponibilidad de Deep Learning
            if not DEEP_LEARNING_AVAILABLE:
                return {}
                
            # Verificar que el checkbox existe y está habilitado
            if not hasattr(self, 'enable_dl_search_cb') or not self.enable_dl_search_cb.isChecked():
                return {}
            
            config = {'enabled': True, 'task_type': 'search'}
            
            # Obtener configuración con verificaciones
            try:
                if hasattr(self, 'dl_search_model_combo'):
                    model_text = self.dl_search_model_combo.currentText()
                    config['model_type'] = model_text.split(' - ')[0] if ' - ' in model_text else model_text
                else:
                    config['model_type'] = 'CNN'  # Valor por defecto
                    
                if hasattr(self, 'dl_search_confidence_spin'):
                    config['confidence_threshold'] = self.dl_search_confidence_spin.value()
                else:
                    config['confidence_threshold'] = 0.85  # Valor por defecto
                    
                if hasattr(self, 'dl_rerank_results_cb'):
                    config['rerank_results'] = self.dl_rerank_results_cb.isChecked()
                else:
                    config['rerank_results'] = False  # Valor por defecto
                    
            except Exception as e:
                print(f"Warning: Error obteniendo configuración de búsqueda DL: {e}")
                # Usar valores por defecto en caso de error
                config.update({
                    'model_type': 'CNN',
                    'confidence_threshold': 0.85,
                    'rerank_results': False
                })
                
            return config
            
        except Exception as e:
            print(f"Error crítico obteniendo configuración de búsqueda DL: {e}")
            return {}  # Retornar configuración vacía en caso de error crítico
        
    def on_comparison_model_configured(self, config: dict):
        """Maneja la configuración del modelo para comparación"""
        if 'model_type' in config:
            model_type = config['model_type']
            for i in range(self.dl_comparison_model_combo.count()):
                if self.dl_comparison_model_combo.itemText(i).startswith(model_type):
                    self.dl_comparison_model_combo.setCurrentIndex(i)
                    break
                    
        if 'confidence_threshold' in config:
            self.dl_confidence_spin.setValue(config['confidence_threshold'])
            
        # Guardar configuración avanzada
        self.advanced_comparison_dl_config = config
        
    def on_search_model_configured(self, config: dict):
        """Maneja la configuración del modelo para búsqueda"""
        if 'model_type' in config:
            model_type = config['model_type']
            for i in range(self.dl_search_model_combo.count()):
                if self.dl_search_model_combo.itemText(i).startswith(model_type):
                    self.dl_search_model_combo.setCurrentIndex(i)
                    break
                    
        if 'confidence_threshold' in config:
            self.dl_search_confidence_spin.setValue(config['confidence_threshold'])
            
        if 'rerank_results' in config:
            self.dl_rerank_results_cb.setChecked(config['rerank_results'])
            
        # Guardar configuración avanzada
        self.advanced_search_dl_config = config
        
        # Conexiones de resultados
        self.results_list.itemClicked.connect(self.on_result_selected)
        
    def on_mode_changed(self, index: int):
        """Maneja el cambio de modo"""
        self.current_mode = 'direct' if index == 0 else 'database'
        self.mode_tabs.setCurrentIndex(index)
        
    def sync_mode_selection(self, index: int):
        """Sincroniza la selección de modo entre combo y tabs"""
        self.mode_combo.setCurrentIndex(index)
        
    def on_evidence_type_changed(self, evidence_type: str):
        """Maneja el cambio de tipo de evidencia"""
        # Actualizar opciones según el tipo de evidencia
        if "Bala" in evidence_type:
            self.analyze_striations_cb.setEnabled(True)
            self.analyze_land_groove_cb.setEnabled(True)
            self.analyze_firing_pin_cb.setEnabled(False)
            self.analyze_breech_face_cb.setEnabled(False)
        else:  # Casquillo
            self.analyze_striations_cb.setEnabled(False)
            self.analyze_land_groove_cb.setEnabled(False)
            self.analyze_firing_pin_cb.setEnabled(True)
            self.analyze_breech_face_cb.setEnabled(True)
            
    def update_cmc_threshold_label(self, value: int):
        """Actualiza la etiqueta del umbral CMC"""
        threshold = value / 100.0
        self.cmc_threshold_label.setText(f"{threshold:.2f}")
        
    def update_similarity_threshold_label(self, value: int):
        """Actualiza la etiqueta del umbral de similitud"""
        threshold = value / 100.0
        self.similarity_threshold_label.setText(f"{threshold:.2f}")
        
    def on_evidence_a_loaded(self, image_path: str):
        """Maneja la carga de la evidencia A"""
        self.comparison_data['evidence_a'] = image_path
        self.check_direct_ready()
        
    def on_evidence_b_loaded(self, image_path: str):
        """Maneja la carga de la evidencia B"""
        self.comparison_data['evidence_b'] = image_path
        self.check_direct_ready()
        
    def on_query_evidence_loaded(self, image_path: str):
        """Maneja la carga de la evidencia de consulta"""
        self.comparison_data['query_evidence'] = image_path
        self.query_image_viewer.load_image(image_path)
        self.search_button.setEnabled(True)
        self.db_step_indicator.set_current_step(1)
        
    def check_direct_ready(self):
        """Verifica si la comparación directa está lista"""
        if ('evidence_a' in self.comparison_data and 
            'evidence_b' in self.comparison_data):
            self.analyze_button.setEnabled(True)
            self.direct_step_indicator.set_current_step(1)
            
    def start_ballistic_comparison(self):
        """Inicia la comparación balística directa"""
        if self.comparison_worker and self.comparison_worker.isRunning():
            return
            
        evidence_type_text = self.evidence_type_combo.currentText()
        evidence_type = 'cartridge_case' if 'Casquillo' in evidence_type_text else 'bullet'
        
        comparison_params = {
            'mode': 'direct',
            'evidence_type': evidence_type,
            'image_a': self.comparison_data.get('evidence_a'),
            'image_b': self.comparison_data.get('evidence_b'),
            'cmc_threshold': self.cmc_threshold_slider.value() / 100.0,
            'ballistic_features': {
                'firing_pin': self.analyze_firing_pin_cb.isChecked(),
                'breech_face': self.analyze_breech_face_cb.isChecked(),
                'extractor': self.analyze_extractor_cb.isChecked(),
                'striations': self.analyze_striations_cb.isChecked(),
                'land_groove': self.analyze_land_groove_cb.isChecked()
            },
            'nist_validation': {
                'quality': self.nist_quality_validation_cb.isChecked(),
                'metadata': self.nist_metadata_validation_cb.isChecked(),
                'chain_custody': self.nist_chain_custody_cb.isChecked()
            }
        }
        
        self.comparison_worker = BallisticComparisonWorker(comparison_params)
        self.comparison_worker.progressUpdated.connect(self.on_comparison_progress)
        self.comparison_worker.comparisonCompleted.connect(self.on_comparison_completed)
        self.comparison_worker.comparisonError.connect(self.on_comparison_error)
        
        self.direct_progress_card.show()
        self.direct_step_indicator.set_current_step(2)
        self.comparison_worker.start()
        
    def start_ballistic_search(self):
        """Inicia la búsqueda en base de datos balística"""
        if self.comparison_worker and self.comparison_worker.isRunning():
            return
            
        evidence_type_text = self.evidence_type_combo.currentText()
        evidence_type = 'cartridge_case' if 'Casquillo' in evidence_type_text else 'bullet'
        
        search_params = {
            'mode': 'database',
            'evidence_type': evidence_type,
            'query_image': self.comparison_data.get('query_evidence'),
            'filters': {
                'caliber': self.caliber_filter_combo.currentText(),
                'weapon_type': self.weapon_type_filter_combo.currentText(),
                'similarity_threshold': self.similarity_threshold_slider.value() / 100.0,
                'max_results': self.max_results_spinbox.value(),
                'date_from': self.date_from_edit.text(),
                'date_to': self.date_to_edit.text(),
                'case_number': self.case_number_filter_edit.text(),
                'location': self.location_filter_edit.text()
            }
        }
        
        self.comparison_worker = BallisticComparisonWorker(search_params)
        self.comparison_worker.progressUpdated.connect(self.on_search_progress)
        self.comparison_worker.comparisonCompleted.connect(self.on_search_completed)
        self.comparison_worker.comparisonError.connect(self.on_search_error)
        
        self.search_progress_card.show()
        self.db_step_indicator.set_current_step(2)
        self.comparison_worker.start()
        
    def on_comparison_progress(self, progress: int, message: str):
        """Actualiza el progreso de la comparación"""
        self.direct_progress_card.set_progress(progress, message)
        
    def on_search_progress(self, progress: int, message: str):
        """Actualiza el progreso de la búsqueda"""
        self.search_progress_card.set_progress(progress, message)
        
    def on_comparison_completed(self, results: dict):
        """Maneja la finalización de la comparación"""
        self.direct_progress_card.hide()
        self.direct_step_indicator.set_current_step(3)
        
        if results['mode'] == 'direct':
            self.display_ballistic_comparison_results(results)
        else:
            self.display_ballistic_search_results(results)
            
    def on_search_completed(self, results: dict):
        """Maneja la finalización de la búsqueda"""
        self.search_progress_card.hide()
        self.db_step_indicator.set_current_step(3)
        self.display_ballistic_search_results(results)
        
    def on_comparison_error(self, error_message: str):
        """Maneja errores en la comparación"""
        self.direct_progress_card.hide()
        QMessageBox.critical(self, "Error en Análisis Balístico", 
                           f"Error durante el análisis: {error_message}")
        
    def on_search_error(self, error_message: str):
        """Maneja errores en la búsqueda"""
        self.search_progress_card.hide()
        QMessageBox.critical(self, "Error en Búsqueda Balística", 
                           f"Error durante la búsqueda: {error_message}")
        
    def display_ballistic_comparison_results(self, results: dict):
        """Muestra los resultados de comparación balística con interfaz de pestañas"""
        # Crear widget de pestañas para resultados detallados
        results_tabs = QTabWidget()
        
        # Pestaña 1: Resumen
        summary_widget = QWidget()
        summary_layout = QVBoxLayout(summary_widget)
        
        # Actualizar visualización CMC
        self.cmc_visualization.update_cmc_data(results.get('cmc_analysis', {}))
        
        # Métricas principales
        metrics_group = QGroupBox("Métricas de Comparación")
        metrics_layout = QGridLayout(metrics_group)
        
        cmc_data = results.get('cmc_analysis', {})
        
        # Verificar que los labels existen antes de actualizarlos
        if hasattr(self, 'cmc_score_label') and self.cmc_score_label:
            self.cmc_score_label.setText(f"Score CMC: {results.get('cmc_score', 0):.3f}")
        if hasattr(self, 'total_cells_label') and self.total_cells_label:
            self.total_cells_label.setText(f"Células Totales: {cmc_data.get('total_cells', 0)}")
        if hasattr(self, 'valid_cells_label') and self.valid_cells_label:
            self.valid_cells_label.setText(f"Células Válidas: {cmc_data.get('valid_cells', 0)}")
        if hasattr(self, 'congruent_cells_label') and self.congruent_cells_label:
            self.congruent_cells_label.setText(f"Células Congruentes: {cmc_data.get('congruent_cells', 0)}")
        
        # Crear nuevos labels si no existen o fueron eliminados
        cmc_score_label = QLabel(f"Score CMC: {results.get('cmc_score', 0):.3f}")
        total_cells_label = QLabel(f"Células Totales: {cmc_data.get('total_cells', 0)}")
        valid_cells_label = QLabel(f"Células Válidas: {cmc_data.get('valid_cells', 0)}")
        congruent_cells_label = QLabel(f"Células Congruentes: {cmc_data.get('congruent_cells', 0)}")
        
        metrics_layout.addWidget(QLabel("Score CMC:"), 0, 0)
        metrics_layout.addWidget(cmc_score_label, 0, 1)
        metrics_layout.addWidget(QLabel("Células Totales:"), 1, 0)
        metrics_layout.addWidget(total_cells_label, 1, 1)
        metrics_layout.addWidget(QLabel("Células Válidas:"), 2, 0)
        metrics_layout.addWidget(valid_cells_label, 2, 1)
        metrics_layout.addWidget(QLabel("Células Congruentes:"), 3, 0)
        metrics_layout.addWidget(congruent_cells_label, 3, 1)
        
        summary_layout.addWidget(metrics_group)
        
        # Conclusión AFTE
        afte_conclusion = results.get('afte_conclusion', 'Inconclusive')
        result_type = results.get('result_type', 'warning')
        
        conclusion_text = f"Conclusión AFTE: {afte_conclusion}"
        if afte_conclusion == "Identification":
            conclusion_text += "\n✅ Las evidencias provienen del mismo arma de fuego"
        elif afte_conclusion == "Elimination":
            conclusion_text += "\n❌ Las evidencias NO provienen del mismo arma de fuego"
        else:
            conclusion_text += "\n⚠️ No se puede determinar con certeza el origen común"
            
        if hasattr(self, 'afte_conclusion_card') and self.afte_conclusion_card:
            self.afte_conclusion_card.set_value(conclusion_text, result_type)
            summary_layout.addWidget(self.afte_conclusion_card)
        
        results_tabs.addTab(summary_widget, "Resumen")
        
        # Pestaña 2: Análisis CMC Detallado
        cmc_widget = QWidget()
        cmc_layout = QVBoxLayout(cmc_widget)
        cmc_layout.addWidget(self.cmc_visualization)
        
        # Estadísticas detalladas
        stats_group = QGroupBox("Análisis Estadístico")
        stats_layout = QGridLayout(stats_group)
        
        stats = results.get('statistical_analysis', {})
        
        # Crear nuevos labels para estadísticas
        p_value_label = QLabel(f"{stats.get('p_value', 0):.4f}")
        ci = stats.get('confidence_interval', [0, 0])
        confidence_interval_label = QLabel(f"[{ci[0]:.3f}, {ci[1]:.3f}]")
        false_positive_rate_label = QLabel(f"{stats.get('false_positive_rate', 0):.4f}")
        
        stats_layout.addWidget(QLabel("Valor P:"), 0, 0)
        stats_layout.addWidget(p_value_label, 0, 1)
        stats_layout.addWidget(QLabel("Intervalo de Confianza:"), 1, 0)
        stats_layout.addWidget(confidence_interval_label, 1, 1)
        stats_layout.addWidget(QLabel("Tasa de Falsos Positivos:"), 2, 0)
        stats_layout.addWidget(false_positive_rate_label, 2, 1)
        
        cmc_layout.addWidget(stats_group)
        results_tabs.addTab(cmc_widget, "Análisis CMC")
        
        # Pestaña 3: Validación NIST
        if NIST_AVAILABLE:
            nist_widget = self.create_nist_validation_tab(results)
            results_tabs.addTab(nist_widget, "Validación NIST")
        
        # Pestaña 4: Características Balísticas
        features_widget = QWidget()
        features_layout = QVBoxLayout(features_widget)
        
        features_text = self.format_ballistic_features(results.get('ballistic_features', {}))
        self.ballistic_features_text.setText(features_text)
        features_layout.addWidget(self.ballistic_features_text)
        
        results_tabs.addTab(features_widget, "Características")
        
        # Reemplazar el contenido del panel de resultados
        if hasattr(self, 'direct_results_layout'):
            # Limpiar layout existente
            for i in reversed(range(self.direct_results_layout.count())):
                self.direct_results_layout.itemAt(i).widget().setParent(None)
            
            # Agregar pestañas de resultados
            self.direct_results_layout.addWidget(results_tabs)
            
            # Botones de acción
            actions_layout = QHBoxLayout()
            
            save_btn = QPushButton("💾 Guardar Resultados")
            save_btn.clicked.connect(lambda: self.save_comparison_results(results))
            
            report_btn = QPushButton("📄 Generar Reporte")
            report_btn.clicked.connect(lambda: self.generate_comparison_report(results))
            
            export_btn = QPushButton("📤 Exportar Datos")
            export_btn.clicked.connect(lambda: self.export_comparison_data(results))
            
            compare_db_btn = QPushButton("🔍 Comparar con BD")
            compare_db_btn.clicked.connect(lambda: self.compare_with_database(results))
            
            actions_layout.addWidget(save_btn)
            actions_layout.addWidget(report_btn)
            actions_layout.addWidget(export_btn)
            actions_layout.addWidget(compare_db_btn)
            actions_layout.addStretch()
            
            self.direct_results_layout.addLayout(actions_layout)
        
        # Emitir señal de finalización
        self.comparisonCompleted.emit(results)
        
    def display_ballistic_search_results(self, results: dict):
        """Muestra los resultados de búsqueda balística"""
        # Actualizar estadísticas de búsqueda
        stats_text = (f"Búsqueda completada: {results.get('total_searched', 0)} evidencias analizadas, "
                     f"{results.get('candidates_found', 0)} candidatos encontrados, "
                     f"{results.get('high_confidence_matches', 0)} coincidencias de alta confianza")
        self.search_stats_label.setText(stats_text)
        
        # Limpiar y llenar lista de resultados
        self.results_list.clear()
        
        for result in results.get('results', []):
            item_widget = self.create_ballistic_result_item_widget(result)
            item = QListWidgetItem()
            item.setSizeHint(item_widget.sizeHint())
            item.setData(Qt.UserRole, result)
            
            self.results_list.addItem(item)
            self.results_list.setItemWidget(item, item_widget)
            
    def create_ballistic_result_item_widget(self, result: dict) -> QWidget:
        """Crea un widget para mostrar un resultado de búsqueda balística"""
        widget = QFrame()
        widget.setProperty("class", "result-item")
        
        layout = QHBoxLayout(widget)
        layout.setSpacing(15)
        
        # Información principal
        info_layout = QVBoxLayout()
        
        # Línea 1: ID y Score CMC
        header_layout = QHBoxLayout()
        
        id_label = QLabel(f"ID: {result.get('id', 'N/A')}")
        id_label.setProperty("class", "body-bold")
        header_layout.addWidget(id_label)
        
        header_layout.addStretch()
        
        cmc_score = result.get('cmc_score', 0)
        cmc_label = QLabel(f"CMC: {cmc_score:.3f}")
        cmc_label.setProperty("class", "body-bold")
        
        # Color según score CMC
        if cmc_score >= 0.85:
            cmc_label.setStyleSheet("color: #28a745; font-weight: bold;")
        elif cmc_score >= 0.70:
            cmc_label.setStyleSheet("color: #ffc107; font-weight: bold;")
        else:
            cmc_label.setStyleSheet("color: #dc3545; font-weight: bold;")
            
        header_layout.addWidget(cmc_label)
        
        info_layout.addLayout(header_layout)
        
        # Línea 2: Conclusión AFTE
        afte_label = QLabel(f"AFTE: {result.get('afte_conclusion', 'N/A')}")
        afte_label.setProperty("class", "body")
        info_layout.addWidget(afte_label)
        
        # Línea 3: Información del caso
        case_info = f"Caso: {result.get('case_number', 'N/A')} | {result.get('weapon_type', 'N/A')}"
        case_label = QLabel(case_info)
        case_label.setProperty("class", "caption")
        info_layout.addWidget(case_label)
        
        # Línea 4: Metadatos balísticos
        metadata = result.get('metadata', {})
        metadata_info = f"Calibre: {metadata.get('caliber', 'N/A')} | Fabricante: {metadata.get('manufacturer', 'N/A')}"
        metadata_label = QLabel(metadata_info)
        metadata_label.setProperty("class", "caption")
        info_layout.addWidget(metadata_label)
        
        layout.addLayout(info_layout)
        
        # Indicador visual de confianza
        confidence_indicator = QLabel("●")
        confidence_indicator.setProperty("class", "title")
        
        afte_conclusion = result.get('afte_conclusion', '')
        if afte_conclusion == 'Identification':
            confidence_indicator.setStyleSheet("color: #28a745; font-size: 20px;")
        elif afte_conclusion == 'Inconclusive':
            confidence_indicator.setStyleSheet("color: #ffc107; font-size: 20px;")
        else:
            confidence_indicator.setStyleSheet("color: #dc3545; font-size: 20px;")
            
        layout.addWidget(confidence_indicator)
        
        return widget
        
    def on_result_selected(self, item: QListWidgetItem):
        """Maneja la selección de un resultado de búsqueda"""
        result_data = item.data(Qt.UserRole)
        if result_data:
            self.selected_db_result = result_data
            
            # Cargar imagen seleccionada
            image_path = result_data.get('path', '')
            if image_path and os.path.exists(image_path):
                self.selected_image_viewer.load_image(image_path)
            
            # Actualizar métricas de comparación
            self.selected_cmc_score_label.setText(f"Score CMC: {result_data.get('cmc_score', 0):.3f}")
            self.selected_afte_conclusion_label.setText(f"Conclusión AFTE: {result_data.get('afte_conclusion', 'N/A')}")
            
            # Calcular confianza basada en CMC score
            cmc_score = result_data.get('cmc_score', 0)
            confidence = "Alta" if cmc_score >= 0.85 else "Media" if cmc_score >= 0.70 else "Baja"
            self.selected_confidence_label.setText(f"Confianza: {confidence}")
            
            case_info = f"Caso: {result_data.get('case_number', 'N/A')} | {result_data.get('weapon_type', 'N/A')}"
            self.selected_case_info_label.setText(case_info)
            
            self.db_step_indicator.set_current_step(3)
            
    def format_ballistic_features(self, features: dict) -> str:
        """Formatea las características balísticas para mostrar"""
        text_parts = []
        
        if 'firing_pin_correlation' in features:
            text_parts.append(f"🎯 Correlación Firing Pin: {features['firing_pin_correlation']:.3f}")
            
        if 'breech_face_correlation' in features:
            text_parts.append(f"🔍 Correlación Breech Face: {features['breech_face_correlation']:.3f}")
            
        if 'extractor_marks_correlation' in features:
            text_parts.append(f"⚙️ Correlación Marcas Extractor: {features['extractor_marks_correlation']:.3f}")
            
        if features.get('striation_correlation'):
            text_parts.append(f"📏 Correlación Estriado: {features['striation_correlation']:.3f}")
            
        return "\n".join(text_parts) if text_parts else "No hay características disponibles"
    
    def create_nist_validation_tab(self, results: dict) -> QWidget:
        """Crea la pestaña de validación NIST"""
        nist_widget = QWidget()
        nist_layout = QVBoxLayout(nist_widget)
        
        # Validación de calidad de imagen
        quality_group = QGroupBox("Validación de Calidad de Imagen")
        quality_layout = QVBoxLayout(quality_group)
        
        nist_data = results.get('nist_validation', {})
        quality_metrics = nist_data.get('quality_metrics', {})
        
        # Tabla de métricas de calidad
        quality_table = QTableWidget()
        quality_table.setColumnCount(3)
        quality_table.setHorizontalHeaderLabels(["Métrica", "Valor", "Estado"])
        quality_table.horizontalHeader().setStretchLastSection(True)
        
        metrics = [
            ("Resolución (DPI)", quality_metrics.get('resolution_dpi', 'N/A'), 
             "✅ Cumple" if quality_metrics.get('resolution_compliant', False) else "❌ No cumple"),
            ("Contraste", f"{quality_metrics.get('contrast', 0):.3f}", 
             "✅ Cumple" if quality_metrics.get('contrast_compliant', False) else "❌ No cumple"),
            ("Nitidez", f"{quality_metrics.get('sharpness', 0):.3f}", 
             "✅ Cumple" if quality_metrics.get('sharpness_compliant', False) else "❌ No cumple"),
            ("Ruido", f"{quality_metrics.get('noise_level', 0):.3f}", 
             "✅ Cumple" if quality_metrics.get('noise_compliant', False) else "❌ No cumple"),
            ("Iluminación", f"{quality_metrics.get('illumination_uniformity', 0):.3f}", 
             "✅ Cumple" if quality_metrics.get('illumination_compliant', False) else "❌ No cumple")
        ]
        
        quality_table.setRowCount(len(metrics))
        for i, (metric, value, status) in enumerate(metrics):
            quality_table.setItem(i, 0, QTableWidgetItem(metric))
            quality_table.setItem(i, 1, QTableWidgetItem(str(value)))
            quality_table.setItem(i, 2, QTableWidgetItem(status))
        
        quality_layout.addWidget(quality_table)
        nist_layout.addWidget(quality_group)
        
        # Validación de metadatos
        metadata_group = QGroupBox("Validación de Metadatos")
        metadata_layout = QVBoxLayout(metadata_group)
        
        metadata_table = QTableWidget()
        metadata_table.setColumnCount(2)
        metadata_table.setHorizontalHeaderLabels(["Campo", "Estado"])
        metadata_table.horizontalHeader().setStretchLastSection(True)
        
        metadata_validation = nist_data.get('metadata_validation', {})
        metadata_items = [
            ("Fecha de Captura", "✅ Presente" if metadata_validation.get('capture_date', False) else "❌ Faltante"),
            ("Información del Dispositivo", "✅ Presente" if metadata_validation.get('device_info', False) else "❌ Faltante"),
            ("Configuración de Cámara", "✅ Presente" if metadata_validation.get('camera_settings', False) else "❌ Faltante"),
            ("Cadena de Custodia", "✅ Válida" if metadata_validation.get('chain_of_custody', False) else "❌ Inválida"),
            ("Hash de Integridad", "✅ Verificado" if metadata_validation.get('integrity_hash', False) else "❌ No verificado")
        ]
        
        metadata_table.setRowCount(len(metadata_items))
        for i, (field, status) in enumerate(metadata_items):
            metadata_table.setItem(i, 0, QTableWidgetItem(field))
            metadata_table.setItem(i, 1, QTableWidgetItem(status))
        
        metadata_layout.addWidget(metadata_table)
        nist_layout.addWidget(metadata_group)
        
        # Reporte de cumplimiento general
        compliance_group = QGroupBox("Cumplimiento General NIST")
        compliance_layout = QVBoxLayout(compliance_group)
        
        overall_compliance = nist_data.get('overall_compliance', False)
        compliance_score = nist_data.get('compliance_score', 0)
        
        compliance_label = QLabel()
        if overall_compliance:
            compliance_label.setText(f"✅ CUMPLE con estándares NIST (Puntuación: {compliance_score:.1f}/100)")
            compliance_label.setStyleSheet("color: green; font-weight: bold; font-size: 14px;")
        else:
            compliance_label.setText(f"❌ NO CUMPLE con estándares NIST (Puntuación: {compliance_score:.1f}/100)")
            compliance_label.setStyleSheet("color: red; font-weight: bold; font-size: 14px;")
        
        compliance_layout.addWidget(compliance_label)
        
        # Recomendaciones
        recommendations = nist_data.get('recommendations', [])
        if recommendations:
            rec_label = QLabel("Recomendaciones:")
            rec_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
            compliance_layout.addWidget(rec_label)
            
            for rec in recommendations:
                rec_item = QLabel(f"• {rec}")
                rec_item.setWordWrap(True)
                compliance_layout.addWidget(rec_item)
        
        nist_layout.addWidget(compliance_group)
        
        return nist_widget
    
    def save_comparison_results(self, results: dict):
        """Guarda los resultados de comparación"""
        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Guardar Resultados de Comparación", 
                f"comparison_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                "JSON Files (*.json);;All Files (*)"
            )
            
            if file_path:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False, default=str)
                
                QMessageBox.information(self, "Éxito", 
                                      f"Resultados guardados exitosamente en:\n{file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error al guardar resultados:\n{str(e)}")
    
    def generate_comparison_report(self, results: dict):
        """Genera un reporte de comparación"""
        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Generar Reporte de Comparación", 
                f"comparison_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                "PDF Files (*.pdf);;All Files (*)"
            )
            
            if file_path:
                # Aquí se integraría con el módulo de reportes
                QMessageBox.information(self, "Reporte", 
                                      f"Funcionalidad de reporte será implementada.\nArchivo: {file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error al generar reporte:\n{str(e)}")
    
    def export_comparison_data(self, results: dict):
        """Exporta los datos de comparación"""
        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Exportar Datos de Comparación", 
                f"comparison_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "CSV Files (*.csv);;Excel Files (*.xlsx);;All Files (*)"
            )
            
            if file_path:
                # Aquí se implementaría la exportación a CSV/Excel
                QMessageBox.information(self, "Exportación", 
                                      f"Funcionalidad de exportación será implementada.\nArchivo: {file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error al exportar datos:\n{str(e)}")
    
    def compare_with_database(self, results: dict):
        """Compara los resultados con la base de datos"""
        try:
            # Cambiar a la pestaña de búsqueda en base de datos
            self.mode_tabs.setCurrentIndex(1)  # Asumiendo que es la segunda pestaña
            
            QMessageBox.information(self, "Comparación con BD", 
                                  "Cambiando a modo de búsqueda en base de datos.\n"
                                  "Configure los parámetros y ejecute la búsqueda.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error al cambiar a búsqueda en BD:\n{str(e)}")