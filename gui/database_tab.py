#!/usr/bin/env python3
"""
Pestaña de Base de Datos Balística
Sistema SEACABAr - Análisis de Cartuchos y Balas Automático

Gestión completa de base de datos balística con:
- Búsqueda por características balísticas específicas
- Filtros por metadatos NIST/AFTE
- Visualización de evidencias y comparaciones
- Estadísticas de la base de datos
- Gestión de colecciones y casos
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout,
    QLabel, QPushButton, QLineEdit, QTextEdit, QComboBox, QSpinBox,
    QCheckBox, QGroupBox, QScrollArea, QSplitter, QFrame, QSpacerItem,
    QSizePolicy, QFileDialog, QMessageBox, QProgressBar, QTabWidget,
    QListWidget, QListWidgetItem, QSlider, QDoubleSpinBox, QDateEdit,
    QTableWidget, QTableWidgetItem, QHeaderView, QTreeWidget, QTreeWidgetItem,
    QButtonGroup, QRadioButton, QCalendarWidget
)
from PyQt5.QtCore import Qt, pyqtSignal, QThread, QTimer, QDate, QSize
from PyQt5.QtGui import QFont, QPixmap, QIcon, QPainter, QPen, QColor, QBrush

from .shared_widgets import (
    ImageDropZone, ResultCard, CollapsiblePanel, StepIndicator, 
    ProgressCard, ImageViewer
)
from .backend_integration import get_backend_integration

logger = logging.getLogger(__name__)

class BallisticDatabaseWorker(QThread):
    """Worker thread para operaciones de base de datos balística"""
    
    searchCompleted = pyqtSignal(list)
    statsUpdated = pyqtSignal(dict)
    progressUpdated = pyqtSignal(int, str)
    searchError = pyqtSignal(str)
    
    def __init__(self, operation_type: str, params: dict = None):
        super().__init__()
        self.operation_type = operation_type
        self.params = params or {}
        
    def run(self):
        """Ejecuta la operación de base de datos"""
        try:
            if self.operation_type == "search":
                self.perform_ballistic_search()
            elif self.operation_type == "stats":
                self.get_database_statistics()
            elif self.operation_type == "advanced_search":
                self.perform_advanced_search()
                
        except Exception as e:
            self.searchError.emit(str(e))
            
    def perform_ballistic_search(self):
        """Realiza búsqueda balística en la base de datos"""
        steps = [
            (10, "Conectando a base de datos balística..."),
            (25, "Aplicando filtros de características..."),
            (40, "Buscando coincidencias de firing pin..."),
            (55, "Analizando breech face impressions..."),
            (70, "Evaluando striations patterns..."),
            (85, "Aplicando criterios NIST/AFTE..."),
            (95, "Ordenando resultados por relevancia..."),
            (100, "Búsqueda completada")
        ]
        
        for progress, message in steps:
            self.progressUpdated.emit(progress, message)
            self.msleep(200)
            
        # Simular resultados de búsqueda balística
        search_results = self.generate_ballistic_search_results()
        self.searchCompleted.emit(search_results)
        
    def perform_advanced_search(self):
        """Realiza búsqueda avanzada con múltiples criterios"""
        steps = [
            (10, "Preparando búsqueda avanzada..."),
            (20, "Aplicando filtros temporales..."),
            (35, "Filtrando por tipo de evidencia..."),
            (50, "Aplicando criterios de calidad..."),
            (65, "Buscando por metadatos NIST..."),
            (80, "Correlacionando características..."),
            (95, "Generando resultados finales..."),
            (100, "Búsqueda avanzada completada")
        ]
        
        for progress, message in steps:
            self.progressUpdated.emit(progress, message)
            self.msleep(250)
            
        # Simular resultados avanzados
        advanced_results = self.generate_advanced_search_results()
        self.searchCompleted.emit(advanced_results)
        
    def get_database_statistics(self):
        """Obtiene estadísticas de la base de datos"""
        self.progressUpdated.emit(50, "Calculando estadísticas...")
        self.msleep(300)
        
        # Simular estadísticas de BD balística
        stats = {
            'total_cases': 1247,
            'total_cartridges': 3891,
            'total_bullets': 2156,
            'total_fragments': 892,
            'evidence_types': {
                'Cartridge Case': 3891,
                'Bullet': 2156,
                'Fragment': 892,
                'Tool Mark': 234
            },
            'calibers': {
                '9mm': 1456,
                '.40 S&W': 987,
                '.45 ACP': 743,
                '.38 Special': 621,
                '7.62mm': 445,
                'Other': 1039
            },
            'weapon_types': {
                'Pistol': 2847,
                'Revolver': 1234,
                'Rifle': 567,
                'Shotgun': 234,
                'Unknown': 409
            },
            'quality_distribution': {
                'Excellent': 1567,
                'Good': 2234,
                'Fair': 1456,
                'Poor': 1034
            },
            'recent_additions': {
                'last_week': 23,
                'last_month': 89,
                'last_year': 456
            },
            'nist_compliance': {
                'compliant': 4234,
                'partial': 567,
                'non_compliant': 234
            }
        }
        
        self.progressUpdated.emit(100, "Estadísticas actualizadas")
        self.statsUpdated.emit(stats)
        
    def generate_ballistic_search_results(self) -> List[Dict]:
        """Genera resultados simulados de búsqueda balística"""
        results = []
        
        for i in range(15):
            result = {
                'id': f'BAL-{2024000 + i}',
                'case_number': f'CASE-{2024}-{1000 + i}',
                'evidence_type': ['Cartridge Case', 'Bullet', 'Fragment'][i % 3],
                'caliber': ['9mm', '.40 S&W', '.45 ACP', '.38 Special'][i % 4],
                'weapon_type': ['Pistol', 'Revolver', 'Rifle'][i % 3],
                'firing_pin_shape': ['Circular', 'Rectangular', 'Elliptical'][i % 3],
                'breech_face_pattern': ['Granular', 'Smooth', 'Striated'][i % 3],
                'striations_count': 12 + (i % 8),
                'quality_score': 0.75 + (i % 4) * 0.05,
                'confidence_level': 0.85 + (i % 3) * 0.05,
                'date_added': (datetime.now() - timedelta(days=i*7)).strftime('%Y-%m-%d'),
                'location': f'Evidence Room {chr(65 + i % 5)}',
                'investigator': f'Detective {chr(65 + i % 3)}. Smith',
                'nist_compliant': i % 4 != 0,
                'afte_conclusion': ['Identification', 'Probable', 'Possible', 'Inconclusive'][i % 4],
                'image_path': f'images/ballistic_{i+1}.jpg',
                'thumbnail_path': f'thumbnails/ballistic_{i+1}_thumb.jpg',
                'metadata': {
                    'acquisition_date': (datetime.now() - timedelta(days=i*7)).isoformat(),
                    'equipment': 'Leica DM6000M',
                    'magnification': '50x',
                    'lighting': 'Coaxial',
                    'resolution': '2048x2048'
                },
                'ballistic_features': {
                    'firing_pin_impression': {
                        'diameter': 1.2 + (i % 5) * 0.1,
                        'depth': 0.05 + (i % 3) * 0.01,
                        'shape_quality': 0.8 + (i % 4) * 0.05
                    },
                    'breech_face_marks': {
                        'pattern_type': ['Granular', 'Smooth', 'Striated'][i % 3],
                        'coverage_area': 0.7 + (i % 4) * 0.05,
                        'clarity': 0.75 + (i % 5) * 0.04
                    },
                    'extractor_marks': {
                        'present': i % 3 != 0,
                        'width': 0.3 + (i % 4) * 0.05,
                        'depth': 0.02 + (i % 3) * 0.005
                    }
                }
            }
            results.append(result)
            
        return results
        
    def generate_advanced_search_results(self) -> List[Dict]:
        """Genera resultados simulados de búsqueda avanzada"""
        # Similar a generate_ballistic_search_results pero con filtros aplicados
        all_results = self.generate_ballistic_search_results()
        
        # Aplicar filtros según parámetros
        filtered_results = []
        for result in all_results:
            if self.matches_search_criteria(result):
                filtered_results.append(result)
                
        return filtered_results[:10]  # Limitar resultados
        
    def matches_search_criteria(self, result: Dict) -> bool:
        """Verifica si un resultado coincide con los criterios de búsqueda"""
        # Implementar lógica de filtrado basada en self.params
        return True  # Simplificado para demo

class BallisticVisualizationWidget(QWidget):
    """Widget para visualización de características balísticas"""
    
    def __init__(self):
        super().__init__()
        self.setup_ui()
        
    def setup_ui(self):
        """Configura la interfaz de visualización"""
        layout = QVBoxLayout(self)
        
        # Título
        title = QLabel("Visualización de Características Balísticas")
        title.setStyleSheet("font-size: 14px; font-weight: bold; color: #2c3e50; margin-bottom: 10px;")
        layout.addWidget(title)
        
        # Tabs para diferentes visualizaciones
        self.viz_tabs = QTabWidget()
        
        # Tab 1: Características generales
        self.features_tab = QWidget()
        self.setup_features_tab()
        self.viz_tabs.addTab(self.features_tab, "Características")
        
        # Tab 2: Comparación visual
        self.comparison_tab = QWidget()
        self.setup_comparison_tab()
        self.viz_tabs.addTab(self.comparison_tab, "Comparación")
        
        # Tab 3: Metadatos NIST
        self.metadata_tab = QWidget()
        self.setup_metadata_tab()
        self.viz_tabs.addTab(self.metadata_tab, "Metadatos NIST")
        
        layout.addWidget(self.viz_tabs)
        
    def setup_features_tab(self):
        """Configura el tab de características"""
        layout = QVBoxLayout(self.features_tab)
        
        # Área de scroll para características
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        
        self.features_content = QWidget()
        self.features_layout = QVBoxLayout(self.features_content)
        
        # Placeholder inicial
        placeholder = QLabel("Seleccione un elemento para ver sus características balísticas")
        placeholder.setAlignment(Qt.AlignCenter)
        placeholder.setStyleSheet("color: #7f8c8d; font-style: italic; padding: 20px;")
        self.features_layout.addWidget(placeholder)
        
        scroll.setWidget(self.features_content)
        layout.addWidget(scroll)
        
    def setup_comparison_tab(self):
        """Configura el tab de comparación"""
        layout = QVBoxLayout(self.comparison_tab)
        
        # Área para mostrar comparaciones lado a lado
        comparison_frame = QFrame()
        comparison_frame.setFrameStyle(QFrame.StyledPanel)
        comparison_layout = QHBoxLayout(comparison_frame)
        
        # Imagen de referencia
        ref_group = QGroupBox("Imagen de Referencia")
        ref_layout = QVBoxLayout(ref_group)
        self.ref_image_label = QLabel("No hay imagen de referencia")
        self.ref_image_label.setAlignment(Qt.AlignCenter)
        self.ref_image_label.setMinimumSize(200, 200)
        self.ref_image_label.setStyleSheet("border: 2px dashed #bdc3c7; background: #ecf0f1;")
        ref_layout.addWidget(self.ref_image_label)
        
        # Imagen seleccionada
        sel_group = QGroupBox("Imagen Seleccionada")
        sel_layout = QVBoxLayout(sel_group)
        self.sel_image_label = QLabel("Seleccione un elemento")
        self.sel_image_label.setAlignment(Qt.AlignCenter)
        self.sel_image_label.setMinimumSize(200, 200)
        self.sel_image_label.setStyleSheet("border: 2px dashed #bdc3c7; background: #ecf0f1;")
        sel_layout.addWidget(self.sel_image_label)
        
        comparison_layout.addWidget(ref_group)
        comparison_layout.addWidget(sel_group)
        
        layout.addWidget(comparison_frame)
        
        # Métricas de comparación
        metrics_group = QGroupBox("Métricas de Comparación")
        metrics_layout = QGridLayout(metrics_group)
        
        self.similarity_label = QLabel("Similitud: --")
        self.confidence_label = QLabel("Confianza: --")
        self.afte_label = QLabel("Conclusión AFTE: --")
        
        metrics_layout.addWidget(QLabel("Similitud:"), 0, 0)
        metrics_layout.addWidget(self.similarity_label, 0, 1)
        metrics_layout.addWidget(QLabel("Confianza:"), 1, 0)
        metrics_layout.addWidget(self.confidence_label, 1, 1)
        metrics_layout.addWidget(QLabel("AFTE:"), 2, 0)
        metrics_layout.addWidget(self.afte_label, 2, 1)
        
        layout.addWidget(metrics_group)
        
    def setup_metadata_tab(self):
        """Configura el tab de metadatos NIST"""
        layout = QVBoxLayout(self.metadata_tab)
        
        # Área de scroll para metadatos
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        
        self.metadata_content = QWidget()
        self.metadata_layout = QVBoxLayout(self.metadata_content)
        
        # Placeholder inicial
        placeholder = QLabel("Seleccione un elemento para ver sus metadatos NIST")
        placeholder.setAlignment(Qt.AlignCenter)
        placeholder.setStyleSheet("color: #7f8c8d; font-style: italic; padding: 20px;")
        self.metadata_layout.addWidget(placeholder)
        
        scroll.setWidget(self.metadata_content)
        layout.addWidget(scroll)
        
    def display_ballistic_features(self, item_data: Dict):
        """Muestra las características balísticas de un elemento"""
        # Limpiar contenido anterior
        for i in reversed(range(self.features_layout.count())):
            self.features_layout.itemAt(i).widget().setParent(None)
            
        # Información general
        general_group = QGroupBox("Información General")
        general_layout = QGridLayout(general_group)
        
        general_layout.addWidget(QLabel("ID:"), 0, 0)
        general_layout.addWidget(QLabel(item_data.get('id', 'N/A')), 0, 1)
        general_layout.addWidget(QLabel("Tipo:"), 1, 0)
        general_layout.addWidget(QLabel(item_data.get('evidence_type', 'N/A')), 1, 1)
        general_layout.addWidget(QLabel("Calibre:"), 2, 0)
        general_layout.addWidget(QLabel(item_data.get('caliber', 'N/A')), 2, 1)
        general_layout.addWidget(QLabel("Calidad:"), 3, 0)
        general_layout.addWidget(QLabel(f"{item_data.get('quality_score', 0):.2f}"), 3, 1)
        
        self.features_layout.addWidget(general_group)
        
        # Características balísticas específicas
        if 'ballistic_features' in item_data:
            features = item_data['ballistic_features']
            
            # Firing Pin
            if 'firing_pin_impression' in features:
                fp_group = QGroupBox("Impresión de Firing Pin")
                fp_layout = QGridLayout(fp_group)
                fp_data = features['firing_pin_impression']
                
                fp_layout.addWidget(QLabel("Diámetro:"), 0, 0)
                fp_layout.addWidget(QLabel(f"{fp_data.get('diameter', 0):.2f} mm"), 0, 1)
                fp_layout.addWidget(QLabel("Profundidad:"), 1, 0)
                fp_layout.addWidget(QLabel(f"{fp_data.get('depth', 0):.3f} mm"), 1, 1)
                fp_layout.addWidget(QLabel("Calidad:"), 2, 0)
                fp_layout.addWidget(QLabel(f"{fp_data.get('shape_quality', 0):.2f}"), 2, 1)
                
                self.features_layout.addWidget(fp_group)
            
            # Breech Face
            if 'breech_face_marks' in features:
                bf_group = QGroupBox("Marcas de Breech Face")
                bf_layout = QGridLayout(bf_group)
                bf_data = features['breech_face_marks']
                
                bf_layout.addWidget(QLabel("Patrón:"), 0, 0)
                bf_layout.addWidget(QLabel(bf_data.get('pattern_type', 'N/A')), 0, 1)
                bf_layout.addWidget(QLabel("Cobertura:"), 1, 0)
                bf_layout.addWidget(QLabel(f"{bf_data.get('coverage_area', 0):.2f}"), 1, 1)
                bf_layout.addWidget(QLabel("Claridad:"), 2, 0)
                bf_layout.addWidget(QLabel(f"{bf_data.get('clarity', 0):.2f}"), 2, 1)
                
                self.features_layout.addWidget(bf_group)
                
        self.features_layout.addStretch()
        
    def display_metadata(self, item_data: Dict):
        """Muestra los metadatos NIST de un elemento"""
        # Limpiar contenido anterior
        for i in reversed(range(self.metadata_layout.count())):
            self.metadata_layout.itemAt(i).widget().setParent(None)
            
        # Metadatos de adquisición
        if 'metadata' in item_data:
            metadata = item_data['metadata']
            
            acq_group = QGroupBox("Metadatos de Adquisición")
            acq_layout = QGridLayout(acq_group)
            
            acq_layout.addWidget(QLabel("Fecha:"), 0, 0)
            acq_layout.addWidget(QLabel(metadata.get('acquisition_date', 'N/A')), 0, 1)
            acq_layout.addWidget(QLabel("Equipo:"), 1, 0)
            acq_layout.addWidget(QLabel(metadata.get('equipment', 'N/A')), 1, 1)
            acq_layout.addWidget(QLabel("Magnificación:"), 2, 0)
            acq_layout.addWidget(QLabel(metadata.get('magnification', 'N/A')), 2, 1)
            acq_layout.addWidget(QLabel("Iluminación:"), 3, 0)
            acq_layout.addWidget(QLabel(metadata.get('lighting', 'N/A')), 3, 1)
            acq_layout.addWidget(QLabel("Resolución:"), 4, 0)
            acq_layout.addWidget(QLabel(metadata.get('resolution', 'N/A')), 4, 1)
            
            self.metadata_layout.addWidget(acq_group)
            
        # Cumplimiento NIST
        nist_group = QGroupBox("Cumplimiento NIST/AFTE")
        nist_layout = QGridLayout(nist_group)
        
        nist_layout.addWidget(QLabel("NIST Compliant:"), 0, 0)
        nist_compliant = "Sí" if item_data.get('nist_compliant', False) else "No"
        nist_layout.addWidget(QLabel(nist_compliant), 0, 1)
        nist_layout.addWidget(QLabel("Conclusión AFTE:"), 1, 0)
        nist_layout.addWidget(QLabel(item_data.get('afte_conclusion', 'N/A')), 1, 1)
        
        self.metadata_layout.addWidget(nist_group)
        self.metadata_layout.addStretch()


class DatabaseTab(QWidget):
    """
    Pestaña de base de datos balística con búsqueda especializada y visualización de características
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Estado de la pestaña
        self.current_results = []
        self.selected_item = None
        self.view_mode = "grid"  # "grid" o "list"
        
        # Workers
        self.search_worker = None
        self.stats_worker = None
        
        # Configurar UI
        self._setup_ui()
        self._connect_signals()
        
        # Cargar estadísticas iniciales
        self._load_database_stats()
        
        logger.info("DatabaseTab balística inicializada")
    
    def _setup_ui(self):
        """Configura la interfaz de usuario balística"""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)
        
        # Splitter principal
        main_splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(main_splitter)
        
        # Panel izquierdo - Búsqueda y filtros balísticos
        left_panel = self._create_ballistic_search_panel()
        main_splitter.addWidget(left_panel)
        
        # Panel derecho - Resultados y visualización balística
        right_panel = self._create_ballistic_results_panel()
        main_splitter.addWidget(right_panel)
        
        # Configurar proporciones del splitter
        main_splitter.setSizes([350, 850])
        main_splitter.setStretchFactor(0, 0)
        main_splitter.setStretchFactor(1, 1)
    
    def _create_ballistic_search_panel(self) -> QWidget:
        """Crea el panel de búsqueda balística especializada"""
        panel = QFrame()
        panel.setObjectName("ballisticSearchPanel")
        panel.setMaximumWidth(400)
        
        layout = QVBoxLayout(panel)
        layout.setSpacing(15)
        
        # Título balístico
        title = QLabel("Base de Datos Balística")
        title.setObjectName("sectionTitle")
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: #2c3e50; margin-bottom: 10px;")
        layout.addWidget(title)
        
        # Estadísticas balísticas
        self.ballistic_stats_card = self._create_ballistic_stats_card()
        layout.addWidget(self.ballistic_stats_card)
        
        # Búsqueda rápida balística
        quick_search_group = QGroupBox("Búsqueda Rápida")
        quick_layout = QVBoxLayout(quick_search_group)
        
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Buscar por ID, calibre, tipo de evidencia...")
        quick_layout.addWidget(self.search_input)
        
        # Filtros rápidos por tipo de evidencia
        evidence_layout = QHBoxLayout()
        self.evidence_filter = QComboBox()
        self.evidence_filter.addItems([
            "Todos los tipos",
            "Cartridge Case",
            "Bullet", 
            "Fragment",
            "Tool Mark"
        ])
        evidence_layout.addWidget(QLabel("Tipo:"))
        evidence_layout.addWidget(self.evidence_filter)
        quick_layout.addLayout(evidence_layout)
        
        search_btn = QPushButton("🔍 Buscar")
        search_btn.setObjectName("primaryButton")
        search_btn.clicked.connect(self._perform_quick_search)
        quick_layout.addWidget(search_btn)
        
        layout.addWidget(quick_search_group)
        
        # Filtros balísticos avanzados
        ballistic_filters_panel = CollapsiblePanel("Filtros Balísticos Avanzados")
        ballistic_filters_content = self._create_ballistic_filters_content()
        ballistic_filters_panel.add_content_widget(ballistic_filters_content)
        layout.addWidget(ballistic_filters_panel)
        
        # Filtros NIST/AFTE
        nist_filters_panel = CollapsiblePanel("Filtros NIST/AFTE")
        nist_filters_content = self._create_nist_filters_content()
        nist_filters_panel.add_content_widget(nist_filters_content)
        layout.addWidget(nist_filters_panel)
        
        # Botones de acción
        actions_layout = QVBoxLayout()
        
        self.advanced_search_btn = QPushButton("🎯 Búsqueda Avanzada")
        self.advanced_search_btn.clicked.connect(self._perform_advanced_search)
        actions_layout.addWidget(self.advanced_search_btn)
        
        self.clear_filters_btn = QPushButton("🗑️ Limpiar Filtros")
        self.clear_filters_btn.clicked.connect(self._clear_filters)
        actions_layout.addWidget(self.clear_filters_btn)
        
        self.export_btn = QPushButton("📊 Exportar Resultados")
        self.export_btn.clicked.connect(self._export_results)
        self.export_btn.setEnabled(False)
        actions_layout.addWidget(self.export_btn)
        
        layout.addLayout(actions_layout)
        layout.addStretch()
        
        return panel
        
    def _create_ballistic_stats_card(self) -> ResultCard:
        """Crea la tarjeta de estadísticas balísticas"""
        stats_card = ResultCard("Estadísticas de la Base de Datos")
        
        # Layout para estadísticas
        stats_layout = QGridLayout()
        
        # Placeholders iniciales
        self.total_images_label = QLabel("--")
        self.total_cases_label = QLabel("--")
        self.total_cartridges_label = QLabel("--")
        self.total_bullets_label = QLabel("--")
        self.nist_compliant_label = QLabel("--")
        self.last_update_label = QLabel("--")
        
        stats_layout.addWidget(QLabel("Total Imágenes:"), 0, 0)
        stats_layout.addWidget(self.total_images_label, 0, 1)
        stats_layout.addWidget(QLabel("Total Casos:"), 1, 0)
        stats_layout.addWidget(self.total_cases_label, 1, 1)
        stats_layout.addWidget(QLabel("Cartridge Cases:"), 2, 0)
        stats_layout.addWidget(self.total_cartridges_label, 2, 1)
        stats_layout.addWidget(QLabel("Bullets:"), 3, 0)
        stats_layout.addWidget(self.total_bullets_label, 3, 1)
        stats_layout.addWidget(QLabel("NIST Compliant:"), 4, 0)
        stats_layout.addWidget(self.nist_compliant_label, 4, 1)
        stats_layout.addWidget(QLabel("Última Actualización:"), 5, 0)
        stats_layout.addWidget(self.last_update_label, 5, 1)
        
        # Barra de progreso para carga
        self.stats_progress = QProgressBar()
        self.stats_progress.setVisible(False)
        
        # Crear un widget contenedor para el layout
        stats_widget = QWidget()
        stats_widget.setLayout(stats_layout)
        
        # Crear un layout principal para la tarjeta
        card_layout = QVBoxLayout()
        card_layout.addWidget(stats_widget)
        card_layout.addWidget(self.stats_progress)
        
        # Crear un widget contenedor final
        content_widget = QWidget()
        content_widget.setLayout(card_layout)
        
        # Agregar el contenido a la tarjeta usando su layout interno
        stats_card.layout().addWidget(content_widget)
        
        return stats_card
        
    def _create_ballistic_filters_content(self) -> QWidget:
        """Crea el contenido de filtros balísticos"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Filtro por calibre
        caliber_group = QGroupBox("Calibre")
        caliber_layout = QVBoxLayout(caliber_group)
        
        self.caliber_filter = QComboBox()
        self.caliber_filter.addItems([
            "Todos los calibres",
            "9mm",
            ".40 S&W", 
            ".45 ACP",
            ".38 Special",
            "7.62mm",
            ".22 LR",
            ".357 Magnum",
            "Otro"
        ])
        caliber_layout.addWidget(self.caliber_filter)
        layout.addWidget(caliber_group)
        
        # Filtro por tipo de arma
        weapon_group = QGroupBox("Tipo de Arma")
        weapon_layout = QVBoxLayout(weapon_group)
        
        self.weapon_filter = QComboBox()
        self.weapon_filter.addItems([
            "Todos los tipos",
            "Pistol",
            "Revolver",
            "Rifle", 
            "Shotgun",
            "Unknown"
        ])
        weapon_layout.addWidget(self.weapon_filter)
        layout.addWidget(weapon_group)
        
        # Filtros de características balísticas
        features_group = QGroupBox("Características Balísticas")
        features_layout = QVBoxLayout(features_group)
        
        # Firing Pin Shape
        fp_layout = QHBoxLayout()
        fp_layout.addWidget(QLabel("Firing Pin:"))
        self.firing_pin_filter = QComboBox()
        self.firing_pin_filter.addItems([
            "Cualquier forma",
            "Circular",
            "Rectangular", 
            "Elliptical",
            "Irregular"
        ])
        fp_layout.addWidget(self.firing_pin_filter)
        features_layout.addLayout(fp_layout)
        
        # Breech Face Pattern
        bf_layout = QHBoxLayout()
        bf_layout.addWidget(QLabel("Breech Face:"))
        self.breech_face_filter = QComboBox()
        self.breech_face_filter.addItems([
            "Cualquier patrón",
            "Granular",
            "Smooth",
            "Striated"
        ])
        bf_layout.addWidget(self.breech_face_filter)
        features_layout.addLayout(bf_layout)
        
        layout.addWidget(features_group)
        
        # Filtro por calidad
        quality_group = QGroupBox("Calidad de Evidencia")
        quality_layout = QVBoxLayout(quality_group)
        
        quality_range_layout = QHBoxLayout()
        quality_range_layout.addWidget(QLabel("Mínima:"))
        self.min_quality_spin = QDoubleSpinBox()
        self.min_quality_spin.setRange(0.0, 1.0)
        self.min_quality_spin.setSingleStep(0.1)
        self.min_quality_spin.setValue(0.0)
        quality_range_layout.addWidget(self.min_quality_spin)
        
        quality_range_layout.addWidget(QLabel("Máxima:"))
        self.max_quality_spin = QDoubleSpinBox()
        self.max_quality_spin.setRange(0.0, 1.0)
        self.max_quality_spin.setSingleStep(0.1)
        self.max_quality_spin.setValue(1.0)
        quality_range_layout.addWidget(self.max_quality_spin)
        
        quality_layout.addLayout(quality_range_layout)
        layout.addWidget(quality_group)
        
        return widget
        
    def _create_nist_filters_content(self) -> QWidget:
        """Crea el contenido de filtros NIST/AFTE"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Cumplimiento NIST
        nist_group = QGroupBox("Cumplimiento NIST")
        nist_layout = QVBoxLayout(nist_group)
        
        self.nist_compliant_check = QCheckBox("Solo elementos NIST compliant")
        nist_layout.addWidget(self.nist_compliant_check)
        layout.addWidget(nist_group)
        
        # Conclusiones AFTE
        afte_group = QGroupBox("Conclusiones AFTE")
        afte_layout = QVBoxLayout(afte_group)
        
        self.afte_filter = QComboBox()
        self.afte_filter.addItems([
            "Todas las conclusiones",
            "Identification",
            "Probable",
            "Possible", 
            "Inconclusive",
            "Elimination"
        ])
        afte_layout.addWidget(self.afte_filter)
        layout.addWidget(afte_group)
        
        # Filtro temporal
        date_group = QGroupBox("Rango de Fechas")
        date_layout = QGridLayout(date_group)
        
        date_layout.addWidget(QLabel("Desde:"), 0, 0)
        self.start_date = QDateEdit()
        self.start_date.setDate(QDate.currentDate().addYears(-1))
        self.start_date.setCalendarPopup(True)
        date_layout.addWidget(self.start_date, 0, 1)
        
        date_layout.addWidget(QLabel("Hasta:"), 1, 0)
        self.end_date = QDateEdit()
        self.end_date.setDate(QDate.currentDate())
        self.end_date.setCalendarPopup(True)
        date_layout.addWidget(self.end_date, 1, 1)
        
        layout.addWidget(date_group)
        
        return widget
        
    def _create_ballistic_results_panel(self) -> QWidget:
        """Crea el panel de resultados balísticos"""
        panel = QFrame()
        panel.setObjectName("ballisticResultsPanel")
        
        layout = QVBoxLayout(panel)
        layout.setSpacing(15)
        
        # Header con controles de vista
        header_layout = QHBoxLayout()
        
        # Título de resultados
        self.results_title = QLabel("Resultados de Búsqueda Balística")
        self.results_title.setObjectName("sectionTitle")
        self.results_title.setStyleSheet("font-size: 16px; font-weight: bold; color: #2c3e50;")
        header_layout.addWidget(self.results_title)
        
        header_layout.addStretch()
        
        # Controles de vista
        view_controls = QHBoxLayout()
        
        # Selector de modo de vista
        view_controls.addWidget(QLabel("Vista:"))
        self.view_mode_combo = QComboBox()
        self.view_mode_combo.addItems(["Grilla", "Lista"])
        self.view_mode_combo.currentTextChanged.connect(self._set_view_mode)
        view_controls.addWidget(self.view_mode_combo)
        
        # Ordenamiento
        view_controls.addWidget(QLabel("Ordenar por:"))
        self.sort_combo = QComboBox()
        self.sort_combo.addItems([
            "Relevancia",
            "Fecha (Reciente)",
            "Fecha (Antigua)", 
            "Calidad (Alta)",
            "Calidad (Baja)",
            "Calibre",
            "Tipo de Evidencia"
        ])
        view_controls.addWidget(self.sort_combo)
        
        header_layout.addLayout(view_controls)
        layout.addLayout(header_layout)
        
        # Barra de progreso para búsquedas
        self.search_progress = QProgressBar()
        self.search_progress.setVisible(False)
        layout.addWidget(self.search_progress)
        
        # Splitter para resultados y visualización
        results_splitter = QSplitter(Qt.Horizontal)
        
        # Panel de resultados
        results_container = QFrame()
        results_container.setObjectName("resultsContainer")
        results_layout = QVBoxLayout(results_container)
        
        # Área de scroll para resultados
        self.results_scroll = QScrollArea()
        self.results_scroll.setWidgetResizable(True)
        self.results_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.results_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        # Widget contenedor de resultados
        self.results_widget = QWidget()
        self.results_layout = QVBoxLayout(self.results_widget)
        
        # Mensaje inicial
        self.no_results_label = QLabel("Realice una búsqueda para ver resultados")
        self.no_results_label.setAlignment(Qt.AlignCenter)
        self.no_results_label.setStyleSheet("color: #7f8c8d; font-style: italic; padding: 40px;")
        self.results_layout.addWidget(self.no_results_label)
        
        self.results_scroll.setWidget(self.results_widget)
        results_layout.addWidget(self.results_scroll)
        
        results_splitter.addWidget(results_container)
        
        # Panel de visualización balística
        self.ballistic_viz_widget = BallisticVisualizationWidget()
        results_splitter.addWidget(self.ballistic_viz_widget)
        
        # Configurar proporciones
        results_splitter.setSizes([500, 400])
        results_splitter.setStretchFactor(0, 1)
        results_splitter.setStretchFactor(1, 0)
        
        layout.addWidget(results_splitter)
        
        return panel
    
    def _create_stats_card(self) -> ResultCard:
        """Crea la tarjeta de estadísticas de la BD"""
        stats_card = ResultCard("Estadísticas de Base de Datos", "info")
        
        # Contenido inicial
        stats_content = QWidget()
        stats_layout = QVBoxLayout(stats_content)
        
        self.total_images_label = QLabel("Total de imágenes: Cargando...")
        self.total_cases_label = QLabel("Total de casos: Cargando...")
        self.last_update_label = QLabel("Última actualización: Cargando...")
        
        stats_layout.addWidget(self.total_images_label)
        stats_layout.addWidget(self.total_cases_label)
        stats_layout.addWidget(self.last_update_label)
        
        stats_card.setContent(stats_content)
        
        return stats_card
    
    def _create_filters_content(self) -> QWidget:
        """Crea el contenido de filtros avanzados"""
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setSpacing(10)
        
        # Filtro por tipo de evidencia
        evidence_layout = QHBoxLayout()
        evidence_layout.addWidget(QLabel("Tipo de Evidencia:"))
        self.evidence_type_combo = QComboBox()
        self.evidence_type_combo.addItems([
            "Todos", "Huella Dactilar", "Huella Palmar", 
            "Huella Plantar", "Otro"
        ])
        evidence_layout.addWidget(self.evidence_type_combo)
        layout.addLayout(evidence_layout)
        
        # Filtro por calidad
        quality_layout = QHBoxLayout()
        quality_layout.addWidget(QLabel("Calidad Mínima:"))
        self.quality_spin = QSpinBox()
        self.quality_spin.setRange(0, 100)
        self.quality_spin.setValue(0)
        self.quality_spin.setSuffix("%")
        quality_layout.addWidget(self.quality_spin)
        layout.addLayout(quality_layout)
        
        # Filtro por fecha
        date_layout = QVBoxLayout()
        date_layout.addWidget(QLabel("Rango de Fechas:"))
        
        date_range_layout = QHBoxLayout()
        self.date_from = QDateEdit()
        self.date_from.setDate(QDate.currentDate().addYears(-1))
        self.date_from.setCalendarPopup(True)
        date_range_layout.addWidget(QLabel("Desde:"))
        date_range_layout.addWidget(self.date_from)
        
        self.date_to = QDateEdit()
        self.date_to.setDate(QDate.currentDate())
        self.date_to.setCalendarPopup(True)
        date_range_layout.addWidget(QLabel("Hasta:"))
        date_range_layout.addWidget(self.date_to)
        
        date_layout.addLayout(date_range_layout)
        layout.addLayout(date_layout)
        
        # Filtro por tags
        tags_layout = QVBoxLayout()
        tags_layout.addWidget(QLabel("Tags:"))
        self.tags_input = QLineEdit()
        self.tags_input.setPlaceholderText("Separar tags con comas")
        tags_layout.addWidget(self.tags_input)
        layout.addLayout(tags_layout)
        
        # Opciones adicionales
        options_layout = QVBoxLayout()
        self.include_processed_check = QCheckBox("Incluir imágenes procesadas")
        self.include_processed_check.setChecked(True)
        options_layout.addWidget(self.include_processed_check)
        
        self.include_matches_check = QCheckBox("Solo con matches encontrados")
        options_layout.addWidget(self.include_matches_check)
        
        layout.addLayout(options_layout)
        
        return content
    
    def _create_results_panel(self) -> QWidget:
        """Crea el panel de resultados"""
        panel = QFrame()
        panel.setObjectName("resultsPanel")
        
        layout = QVBoxLayout(panel)
        layout.setSpacing(15)
        
        # Header con controles de vista
        header_layout = QHBoxLayout()
        
        self.results_title = QLabel("Resultados de Búsqueda")
        self.results_title.setObjectName("sectionTitle")
        header_layout.addWidget(self.results_title)
        
        header_layout.addStretch()
        
        # Controles de vista
        view_controls = QHBoxLayout()
        
        self.view_grid_btn = QPushButton("Vista Grid")
        self.view_grid_btn.setCheckable(True)
        self.view_grid_btn.setChecked(True)
        self.view_grid_btn.clicked.connect(lambda: self._set_view_mode("grid"))
        view_controls.addWidget(self.view_grid_btn)
        
        self.view_list_btn = QPushButton("Vista Lista")
        self.view_list_btn.setCheckable(True)
        self.view_list_btn.clicked.connect(lambda: self._set_view_mode("list"))
        view_controls.addWidget(self.view_list_btn)
        
        header_layout.addLayout(view_controls)
        layout.addLayout(header_layout)
        
        # Splitter para resultados y vista previa
        results_splitter = QSplitter(Qt.Vertical)
        layout.addWidget(results_splitter)
        
        # Panel de resultados
        self.results_container = QTabWidget()
        self._setup_results_views()
        results_splitter.addWidget(self.results_container)
        
        # Panel de vista previa
        preview_panel = self._create_preview_panel()
        results_splitter.addWidget(preview_panel)
        
        # Configurar proporciones
        results_splitter.setSizes([500, 300])
        results_splitter.setStretchFactor(0, 1)
        results_splitter.setStretchFactor(1, 0)
        
        return panel
    
    def _setup_results_views(self):
        """Configura las vistas de resultados"""
        # Vista Grid
        self.grid_scroll = QScrollArea()
        self.grid_scroll.setWidgetResizable(True)
        self.grid_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.grid_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        self.grid_widget = QWidget()
        self.grid_layout = QGridLayout(self.grid_widget)
        self.grid_layout.setSpacing(10)
        self.grid_scroll.setWidget(self.grid_widget)
        
        self.results_container.addTab(self.grid_scroll, "Vista Grid")
        
        # Vista Lista
        self.list_table = QTableWidget()
        self.list_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.list_table.setAlternatingRowColors(True)
        self.list_table.horizontalHeader().setStretchLastSection(True)
        self.list_table.itemSelectionChanged.connect(self._on_list_selection_changed)
        
        self.results_container.addTab(self.list_table, "Vista Lista")
    
    def _create_preview_panel(self) -> QWidget:
        """Crea el panel de vista previa"""
        panel = QFrame()
        panel.setObjectName("previewPanel")
        panel.setMaximumHeight(350)
        
        layout = QHBoxLayout(panel)
        layout.setSpacing(15)
        
        # Vista previa de imagen
        image_container = QFrame()
        image_container.setObjectName("imagePreviewContainer")
        image_container.setMaximumWidth(300)
        
        image_layout = QVBoxLayout(image_container)
        
        image_title = QLabel("Vista Previa")
        image_title.setObjectName("subsectionTitle")
        image_layout.addWidget(image_title)
        
        self.preview_image = ImageViewer()
        self.preview_image.setMinimumSize(250, 200)
        image_layout.addWidget(self.preview_image)
        
        layout.addWidget(image_container)
        
        # Metadatos y detalles
        details_container = QFrame()
        details_layout = QVBoxLayout(details_container)
        
        details_title = QLabel("Detalles del Elemento")
        details_title.setObjectName("subsectionTitle")
        details_layout.addWidget(details_title)
        
        self.details_text = QTextEdit()
        self.details_text.setReadOnly(True)
        self.details_text.setMaximumHeight(200)
        details_layout.addWidget(self.details_text)
        
        # Botones de acción para el elemento seleccionado
        actions_layout = QHBoxLayout()
        
        self.view_full_btn = QPushButton("Ver Completo")
        self.view_full_btn.clicked.connect(self._view_full_item)
        self.view_full_btn.setEnabled(False)
        actions_layout.addWidget(self.view_full_btn)
        
        self.compare_btn = QPushButton("Comparar")
        self.compare_btn.clicked.connect(self._compare_item)
        self.compare_btn.setEnabled(False)
        actions_layout.addWidget(self.compare_btn)
        
        self.export_item_btn = QPushButton("Exportar")
        self.export_item_btn.clicked.connect(self._export_item)
        self.export_item_btn.setEnabled(False)
        actions_layout.addWidget(self.export_item_btn)
        
        details_layout.addLayout(actions_layout)
        
        layout.addWidget(details_container)
        
        return panel
    
    def _connect_signals(self):
        """Conecta las señales de la interfaz"""
        # Búsqueda en tiempo real
        self.search_input.textChanged.connect(self._on_search_text_changed)
        
        # Timer para búsqueda con delay
        self.search_timer = QTimer()
        self.search_timer.setSingleShot(True)
        self.search_timer.timeout.connect(self._perform_delayed_search)
    
    def _load_database_stats(self):
        """Carga las estadísticas de la base de datos"""
        if self.stats_worker and self.stats_worker.isRunning():
            return
        
        self.stats_worker = BallisticDatabaseWorker("stats")
        self.stats_worker.statsUpdated.connect(self._update_stats_display)
        self.stats_worker.searchError.connect(self._handle_stats_error)
        self.stats_worker.start()
    
    def _update_stats_display(self, stats: Dict[str, Any]):
        """Actualiza la visualización de estadísticas"""
        try:
            total_images = stats.get('total_images', 0)
            total_cases = stats.get('total_cases', 0)
            last_update = stats.get('last_update', 'Desconocido')
            
            self.total_images_label.setText(f"Total de imágenes: {total_images:,}")
            self.total_cases_label.setText(f"Total de casos: {total_cases:,}")
            self.last_update_label.setText(f"Última actualización: {last_update}")
            
        except Exception as e:
            logger.error(f"Error actualizando estadísticas: {e}")
    
    def _handle_stats_error(self, error_msg: str):
        """Maneja errores al cargar estadísticas"""
        self.total_images_label.setText("Error cargando estadísticas")
        self.total_cases_label.setText("")
        self.last_update_label.setText("")
        logger.error(f"Error en estadísticas de BD: {error_msg}")
    
    def _on_search_text_changed(self, text: str):
        """Maneja cambios en el texto de búsqueda"""
        self.search_timer.stop()
        if text.strip():
            self.search_timer.start(500)  # Delay de 500ms
    
    def _perform_delayed_search(self):
        """Realiza búsqueda con delay"""
        search_text = self.search_input.text().strip()
        if search_text:
            self._perform_quick_search()
    
    def _perform_quick_search(self):
        """Realiza búsqueda rápida"""
        search_text = self.search_input.text().strip()
        if not search_text:
            return
        
        search_params = {
            'quick_search': search_text,
            'limit': 100
        }
        
        self._execute_search(search_params)
    
    def _perform_advanced_search(self):
        """Realiza búsqueda avanzada con filtros"""
        search_params = self._build_search_params()
        self._execute_search(search_params)
    
    def _build_search_params(self) -> Dict[str, Any]:
        """Construye parámetros de búsqueda desde los filtros"""
        params = {}
        
        # Texto de búsqueda
        search_text = self.search_input.text().strip()
        if search_text:
            params['search_text'] = search_text
        
        # Tipo de evidencia
        evidence_type = self.evidence_filter.currentText()
        if evidence_type != "Todos los tipos":
            params['evidence_type'] = evidence_type
        
        # Calidad mínima (usando el widget que existe)
        if hasattr(self, 'min_quality_spin'):
            min_quality = self.min_quality_spin.value()
            if min_quality > 0:
                params['min_quality'] = min_quality
        
        # Calibre
        if hasattr(self, 'caliber_filter'):
            caliber = self.caliber_filter.currentText()
            if caliber != "Todos los calibres":
                params['caliber'] = caliber
        
        # Tipo de arma
        if hasattr(self, 'weapon_filter'):
            weapon_type = self.weapon_filter.currentText()
            if weapon_type != "Todos los tipos":
                params['weapon_type'] = weapon_type
        
        # Límite de resultados
        params['limit'] = 1000
        
        return params
    
    def _execute_search(self, search_params: Dict[str, Any]):
        """Ejecuta búsqueda en base de datos"""
        if self.search_worker and self.search_worker.isRunning():
            return
        
        self.search_progress.setVisible(True)
        self.search_progress.setValue(0)
        
        self.search_worker = BallisticDatabaseWorker("search", search_params)
        self.search_worker.searchCompleted.connect(self._handle_search_results)
        self.search_worker.progressUpdated.connect(self._update_search_progress)
        self.search_worker.searchError.connect(self._handle_search_error)
        self.search_worker.start()
    
    def _update_search_progress(self, percentage: int, message: str):
        """Actualiza progreso de búsqueda"""
        self.search_progress.setValue(percentage)
        # Opcional: mostrar mensaje en status bar
    
    def _handle_search_results(self, results: List[Dict[str, Any]]):
        """Maneja resultados de búsqueda"""
        self.search_progress.setVisible(False)
        self.current_results = results
        
        # Actualizar título
        self.results_title.setText(f"Resultados de Búsqueda ({len(results)} elementos)")
        
        # Habilitar exportación si hay resultados
        self.export_results_btn.setEnabled(len(results) > 0)
        
        # Mostrar resultados según el modo de vista
        if self.view_mode == "grid":
            self._display_grid_results(results)
        else:
            self._display_list_results(results)
    
    def _handle_search_error(self, error_msg: str):
        """Maneja errores de búsqueda"""
        self.search_progress.setVisible(False)
        QMessageBox.warning(self, "Error de Búsqueda", f"Error al buscar en la base de datos:\n{error_msg}")
    
    def _display_grid_results(self, results: List[Dict[str, Any]]):
        """Muestra resultados en vista grid"""
        # Limpiar grid anterior
        for i in reversed(range(self.grid_layout.count())):
            self.grid_layout.itemAt(i).widget().setParent(None)
        
        # Agregar nuevos resultados
        cols = 4
        for i, result in enumerate(results):
            row = i // cols
            col = i % cols
            
            item_card = self._create_result_card(result)
            self.grid_layout.addWidget(item_card, row, col)
        
        # Cambiar a tab de grid
        self.results_container.setCurrentIndex(0)
    
    def _display_list_results(self, results: List[Dict[str, Any]]):
        """Muestra resultados en vista lista"""
        # Configurar columnas
        headers = ["ID", "Nombre", "Tipo", "Calidad", "Fecha", "Tags"]
        self.list_table.setColumnCount(len(headers))
        self.list_table.setHorizontalHeaderLabels(headers)
        
        # Configurar filas
        self.list_table.setRowCount(len(results))
        
        # Llenar datos
        for i, result in enumerate(results):
            self.list_table.setItem(i, 0, QTableWidgetItem(str(result.get('id', ''))))
            self.list_table.setItem(i, 1, QTableWidgetItem(result.get('name', '')))
            self.list_table.setItem(i, 2, QTableWidgetItem(result.get('evidence_type', '')))
            self.list_table.setItem(i, 3, QTableWidgetItem(f"{result.get('quality', 0)}%"))
            self.list_table.setItem(i, 4, QTableWidgetItem(str(result.get('date', ''))))
            self.list_table.setItem(i, 5, QTableWidgetItem(', '.join(result.get('tags', []))))
        
        # Ajustar columnas
        self.list_table.resizeColumnsToContents()
        
        # Cambiar a tab de lista
        self.results_container.setCurrentIndex(1)
    
    def _create_result_card(self, result: Dict[str, Any]) -> QWidget:
        """Crea una tarjeta para un resultado individual"""
        card = QFrame()
        card.setObjectName("resultCard")
        card.setMaximumSize(200, 250)
        card.setCursor(Qt.PointingHandCursor)
        
        layout = QVBoxLayout(card)
        layout.setSpacing(5)
        
        # Imagen miniatura
        image_label = QLabel()
        image_label.setFixedSize(150, 120)
        image_label.setAlignment(Qt.AlignCenter)
        image_label.setStyleSheet("border: 1px solid #ddd; background: #f5f5f5;")
        
        # Cargar imagen si está disponible
        image_path = result.get('image_path') or result.get('thumbnail_path')
        if image_path and os.path.exists(image_path):
            pixmap = QPixmap(image_path)
            if not pixmap.isNull():
                scaled_pixmap = pixmap.scaled(150, 120, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                image_label.setPixmap(scaled_pixmap)
            else:
                image_label.setText("Sin imagen")
        else:
            image_label.setText("Sin imagen")
        
        layout.addWidget(image_label)
        
        # Información
        name_label = QLabel(result.get('name', 'Sin nombre'))
        name_label.setWordWrap(True)
        name_label.setMaximumHeight(40)
        layout.addWidget(name_label)
        
        info_layout = QHBoxLayout()
        type_label = QLabel(result.get('evidence_type', 'N/A'))
        type_label.setObjectName("infoLabel")
        quality_label = QLabel(f"{result.get('quality', 0)}%")
        quality_label.setObjectName("infoLabel")
        
        info_layout.addWidget(type_label)
        info_layout.addStretch()
        info_layout.addWidget(quality_label)
        
        layout.addLayout(info_layout)
        
        # Conectar click
        card.mousePressEvent = lambda event: self._select_result_item(result)
        
        return card
    
    def _select_result_item(self, result: Dict[str, Any]):
        """Selecciona un elemento de resultado"""
        self.selected_item = result
        self._update_preview(result)
        
        # Habilitar botones de acción
        self.view_full_btn.setEnabled(True)
        self.compare_btn.setEnabled(True)
        self.export_item_btn.setEnabled(True)
    
    def _on_list_selection_changed(self):
        """Maneja cambio de selección en vista lista"""
        current_row = self.list_table.currentRow()
        if current_row >= 0 and current_row < len(self.current_results):
            result = self.current_results[current_row]
            self._select_result_item(result)
    
    def _update_preview(self, result: Dict[str, Any]):
        """Actualiza la vista previa con el elemento seleccionado"""
        # Actualizar imagen
        image_path = result.get('image_path')
        if image_path and os.path.exists(image_path):
            self.preview_image.load_image(image_path)
        else:
            self.preview_image.clear()
        
        # Actualizar detalles
        details = []
        details.append(f"ID: {result.get('id', 'N/A')}")
        details.append(f"Nombre: {result.get('name', 'N/A')}")
        details.append(f"Tipo: {result.get('evidence_type', 'N/A')}")
        details.append(f"Calidad: {result.get('quality', 0)}%")
        details.append(f"Fecha: {result.get('date', 'N/A')}")
        details.append(f"Tamaño: {result.get('file_size', 'N/A')}")
        
        if result.get('tags'):
            details.append(f"Tags: {', '.join(result['tags'])}")
        
        if result.get('description'):
            details.append(f"Descripción: {result['description']}")
        
        self.details_text.setPlainText('\n'.join(details))
    
    def _set_view_mode(self, mode: str):
        """Cambia el modo de vista"""
        self.view_mode = mode
        
        # Actualizar combo box
        if mode == "grid":
            self.view_mode_combo.setCurrentText("Grilla")
        else:
            self.view_mode_combo.setCurrentText("Lista")
        
        # Mostrar resultados en el nuevo modo
        if hasattr(self, 'current_results') and self.current_results:
            if mode == "grid":
                self._display_grid_results(self.current_results)
            else:
                self._display_list_results(self.current_results)
    
    def _clear_filters(self):
        """Limpia todos los filtros"""
        self.search_input.clear()
        self.evidence_type_combo.setCurrentIndex(0)
        self.quality_spin.setValue(0)
        self.date_from.setDate(QDate.currentDate().addYears(-1))
        self.date_to.setDate(QDate.currentDate())
        self.tags_input.clear()
        self.include_processed_check.setChecked(True)
        self.include_matches_check.setChecked(False)
    
    def _view_full_item(self):
        """Abre vista completa del elemento seleccionado"""
        if not self.selected_item:
            return
        
        # Aquí se podría abrir una ventana de detalles completos
        # o cambiar a la pestaña de análisis con este elemento
        QMessageBox.information(self, "Vista Completa", 
                               f"Abriendo vista completa para: {self.selected_item.get('name', 'Elemento')}")
    
    def _compare_item(self):
        """Inicia comparación con el elemento seleccionado"""
        if not self.selected_item:
            return
        
        # Cambiar a pestaña de comparación con este elemento como base
        QMessageBox.information(self, "Comparar", 
                               f"Iniciando comparación con: {self.selected_item.get('name', 'Elemento')}")
    
    def _export_item(self):
        """Exporta el elemento seleccionado"""
        if not self.selected_item:
            return
        
        filename, _ = QFileDialog.getSaveFileName(
            self, "Exportar Elemento", 
            f"{self.selected_item.get('name', 'elemento')}.json",
            "JSON Files (*.json);;All Files (*)"
        )
        
        if filename:
            try:
                import json
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(self.selected_item, f, indent=2, ensure_ascii=False)
                
                QMessageBox.information(self, "Exportación Exitosa", 
                                       f"Elemento exportado a: {filename}")
            except Exception as e:
                QMessageBox.warning(self, "Error de Exportación", 
                                   f"Error al exportar elemento:\n{str(e)}")
    
    def _export_results(self):
        """Exporta todos los resultados de búsqueda"""
        if not self.current_results:
            return
        
        filename, _ = QFileDialog.getSaveFileName(
            self, "Exportar Resultados", 
            "resultados_busqueda.json",
            "JSON Files (*.json);;CSV Files (*.csv);;All Files (*)"
        )
        
        if filename:
            try:
                if filename.endswith('.csv'):
                    self._export_results_csv(filename)
                else:
                    self._export_results_json(filename)
                
                QMessageBox.information(self, "Exportación Exitosa", 
                                       f"Resultados exportados a: {filename}")
            except Exception as e:
                QMessageBox.warning(self, "Error de Exportación", 
                                   f"Error al exportar resultados:\n{str(e)}")
    
    def _export_results_json(self, filename: str):
        """Exporta resultados en formato JSON"""
        import json
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.current_results, f, indent=2, ensure_ascii=False)
    
    def _export_results_csv(self, filename: str):
        """Exporta resultados en formato CSV"""
        import csv
        
        if not self.current_results:
            return
        
        # Obtener todas las claves posibles
        all_keys = set()
        for result in self.current_results:
            all_keys.update(result.keys())
        
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=sorted(all_keys))
            writer.writeheader()
            writer.writerows(self.current_results)

if __name__ == "__main__":
    # Prueba básica de la pestaña
    import sys
    from PyQt5.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    
    # Aplicar tema
    from .styles import apply_seacaba_theme
    apply_seacaba_theme(app)
    
    # Crear y mostrar pestaña
    tab = DatabaseTab()
    tab.show()
    
    sys.exit(app.exec_())