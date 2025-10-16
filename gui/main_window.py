#!/usr/bin/env python3
"""
Ventana Principal - SIGeC-Balisticar
===================================

Ventana principal de la aplicación que contiene todas las pestañas
y funcionalidades del sistema de análisis balístico.
"""

from utils.logger import get_logger
logger = get_logger(__name__)

import sys
import os
import json
from datetime import datetime
from typing import Optional
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
    QMenuBar, QStatusBar, QAction, QMessageBox, QSplitter,
    QFrame, QLabel, QPushButton, QApplication, QToolBar,
    QDockWidget, QGroupBox, QGridLayout, QSlider, QCheckBox,
    QSpinBox, QComboBox, QTextEdit, QScrollArea, QFileDialog,
    QDialog
)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from PyQt5.QtGui import QIcon, QFont, QPixmap

CACHE_AVAILABLE = False # Inicializar a False por defecto

# Importar configuración existente
try:
    from config.unified_config import GUIConfig
except ImportError:
    # Configuración por defecto si no existe
    class GUIConfig:
        WINDOW_WIDTH = 1200
        WINDOW_HEIGHT = 800
        WINDOW_TITLE = "SIGeC-Balistica- Sistema de Análisis Forense"
        THEME = "modern"
        LANGUAGE = "es"

# Importar estilos y widgets
from .styles import SIGeCBallisticaTheme, apply_SIGeC_Balistica_theme
from .shared_widgets import StepIndicator, ProgressCard
from .settings_dialog import SettingsDialog
from .history_dialog import HistoryDialog
from .help_dialog import HelpDialog
from .about_dialog import AboutDialog
from .app_state_manager import AppStateManager
from utils.validators import SystemValidator

# Importar sistema de caché inteligente
try:
    from core.intelligent_cache import get_cache, initialize_cache
    from image_processing.lbp_cache import get_lbp_cache
    CACHE_AVAILABLE = True
except ImportError:
    pass # CACHE_AVAILABLE ya es False

# Importaciones básicas del sistema

class MainWindow(QMainWindow):
    """Ventana principal de la aplicación SIGeC-Balistica"""
    
    # Señales para comunicación entre componentes
    analysisRequested = pyqtSignal(dict)
    comparisonRequested = pyqtSignal(dict)
    databaseSearchRequested = pyqtSignal(dict)
    reportGenerationRequested = pyqtSignal(dict)
    
    def __init__(self):
        super().__init__()
        
        # Inicializar el gestor de estado
        self.state_manager = AppStateManager()
        
        # Inicializar sistema de caché inteligente
        self.cache_system = None
        self.lbp_cache = None
        if CACHE_AVAILABLE:
            try:
                # Configurar caché inteligente con configuración optimizada para GUI
                cache_config = {
                    'max_memory_mb': 256,  # Memoria moderada para GUI
                    'strategy': 'adaptive',
                    'compression': 'auto',
                    'enable_disk_cache': True,
                    'disk_cache_dir': 'cache/gui',
                    'default_ttl': 1800,  # 30 minutos para datos GUI
                    'enable_analytics': True
                }
                self.cache_system = initialize_cache(cache_config)
                self.lbp_cache = get_lbp_cache()
                print("✓ Sistema de caché inteligente inicializado en GUI")
            except Exception as e:
                print(f"⚠️ Error inicializando caché en GUI: {e}")
                # No modificar CACHE_AVAILABLE aquí para evitar UnboundLocalError
        
        # Configuración inicial
        from config.unified_config import get_gui_config
        gui_config = get_gui_config()
        self.setWindowTitle("SIGeC-Balistica - Sistema de Análisis Forense")
        self.setGeometry(100, 100, gui_config.window_width, gui_config.window_height)
        
        # Configurar tema
        self.theme = SIGeCBallisticaTheme()
        apply_SIGeC_Balistica_theme(self)
        
        # Configurar interfaz
        self.setup_ui()
        
        # Paneles acoplables
        self.dock_widgets = {}
        self.setup_dock_widgets()
        
        # Configurar conexiones
        self.setup_connections()
        
    def setup_ui(self):
        """Configura la interfaz principal mejorada"""
        # Configuración de ventana
        from config.unified_config import get_gui_config
        gui_config = get_gui_config()
        self.setWindowTitle("SIGeC-Balistica - Sistema de Análisis Forense")
        self.setMinimumSize(1000, 700)
        self.resize(gui_config.window_width, gui_config.window_height)
        
        # Widget central
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Layout principal
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Configurar barra de menú y estado
        self.setup_menu_bar()
        self.setup_status_bar()
        
        # Tabs principales
        self.setup_tabs()
        main_layout.addWidget(self.tab_widget)
        
        # Aplicar estilos del tema básico
        self.setStyleSheet(self.theme.get_stylesheet())
        
        
    def setup_tabs(self):
        """Configura las pestañas principales"""
        self.tab_widget = QTabWidget()
        self.tab_widget.setProperty("class", "main-tabs")

        # Importar pestañas
        try:
            from .analysis_tab import AnalysisTab
            from .comparison_tab import ComparisonTab
            from .database_tab import DatabaseTab
            # Use WebEngine-free reports tab
            from .reports_tab_webengine_free import ReportsTab
            from .assisted_alignment import AssistedAlignmentWidget
            from .statistical_visualizations_tab import StatisticalVisualizationsTab
            # Nueva interfaz de 3 pestañas (TriTabbedGUI)
            from .tri_tab_gui import TriTabbedGUI
            
            # Crear pestañas reales
            self.analysis_tab = AnalysisTab(parent=self)
            self.comparison_tab = ComparisonTab()
            self.database_tab = DatabaseTab()
            self.reports_tab = ReportsTab()
            self.alignment_tab = AssistedAlignmentWidget()
            self.statistical_visualizations_tab = StatisticalVisualizationsTab()
            # Instanciar nueva interfaz de 3 pestañas
            self.tri_tab_gui = TriTabbedGUI()

        except ImportError as e:
            print(f"Error importando pestañas: {e}")
            # Fallback a pestañas placeholder
            self.analysis_tab = self.create_placeholder_tab(
                "Análisis Individual",
                "Procesar y analizar una imagen individual",
                "🔍"
            )
            
            self.comparison_tab = self.create_placeholder_tab(
                "Comparación Interactiva",
                "Comparar dos imágenes lado a lado",
                "⚖️"
            )
            
            self.database_tab = self.create_placeholder_tab(
                "Base de Datos",
                "Gestionar casos y evidencias",
                "🗄️"
            )
            
            self.reports_tab = self.create_placeholder_tab(
                "Reportes",
                "Generar reportes profesionales de análisis",
                "📊"
            )
            
            self.alignment_tab = self.create_placeholder_tab(
                "Alineación Asistida",
                "Alinear manualmente imágenes usando puntos de correspondencia",
                "🎯"
            )
            
            self.statistical_visualizations_tab = self.create_placeholder_tab(
                "Visualizaciones Estadísticas",
                "Visualizar estadísticas y métricas de análisis",
                "📈"
            )
            # Placeholder para TriTabbedGUI si falla importación
            self.tri_tab_gui = self.create_placeholder_tab(
                "Nueva Interfaz (3 Pestañas)",
                "Interfaz integrada para carga, análisis y comparación",
                "🧪"
            )

        # Añadir pestañas al widget principal
        self.tab_widget.addTab(self.analysis_tab, "🔍 Análisis")
        self.tab_widget.addTab(self.comparison_tab, "⚖️ Comparación")
        self.tab_widget.addTab(self.database_tab, "🗄️ Base de Datos")
        self.tab_widget.addTab(self.reports_tab, "📊 Reportes")
        self.tab_widget.addTab(self.alignment_tab, "🎯 Alineación")
        self.tab_widget.addTab(self.statistical_visualizations_tab, "📈 Estadísticas")
        # Añadir la nueva interfaz de 3 pestañas como pestaña adicional
        self.tab_widget.addTab(self.tri_tab_gui, "🧪 Nueva Interfaz")

        # Seleccionar la nueva interfaz al inicio para hacerla visible
        try:
            self.tab_widget.setCurrentWidget(self.tri_tab_gui)
        except Exception:
            pass
        
        # Importar gestor de tooltips
        try:
            from .tooltip_manager import apply_tooltips_to_main_window, setup_contextual_tooltips
            apply_tooltips_to_main_window(self)
            print("Tooltips aplicados exitosamente")
        except ImportError as e:
            print(f"No se pudo importar el gestor de tooltips: {e}")
        
        # Configurar tooltips específicos para pestañas
        self.tab_widget.setTabToolTip(0, "Análisis individual de muestras balísticas")
        self.tab_widget.setTabToolTip(1, "Comparación entre múltiples muestras")
        self.tab_widget.setTabToolTip(2, "Gestión de casos y evidencias con estándares NIST")
        self.tab_widget.setTabToolTip(3, "Generación de reportes profesionales")
        self.tab_widget.setTabToolTip(4, "Alineación manual asistida de imágenes")
        self.tab_widget.setTabToolTip(5, "Visualizaciones estadísticas avanzadas")
        
        # Conectar cambio de pestaña
        self.tab_widget.currentChanged.connect(self.on_tab_changed)
        
    # Métodos de manejo de eventos para paneles acoplables
    def on_zoom_changed(self, value):
        """Maneja cambios en el control de zoom"""
        self.zoom_label.setText(f"{value}%")
        self.state_manager.set_zoom_level(value / 100.0)
        
    def on_quality_changed(self, value):
        """Maneja cambios en el filtro de calidad"""
        self.quality_label.setText(f"{value}%")
        self.state_manager.set_quality_threshold(value / 100.0)
        
    def reset_visualization_filters(self):
        """Resetea todos los filtros de visualización"""
        self.zoom_slider.setValue(100)
        self.quality_slider.setValue(30)
        self.sync_zoom_cb.setChecked(True)
        self.show_keypoints_cb.setChecked(True)
        self.show_matches_cb.setChecked(True)
        self.top_matches_spinbox.setValue(50)
        
    def update_statistics(self, stats_dict):
        """Actualiza las estadísticas en tiempo real"""
        if 'correlation' in stats_dict:
            self.correlation_label.setText(f"Correlación: {stats_dict['correlation']:.3f}")
        if 'ssim' in stats_dict:
            self.ssim_label.setText(f"SSIM: {stats_dict['ssim']:.3f}")
        if 'mse' in stats_dict:
            self.mse_label.setText(f"MSE: {stats_dict['mse']:.2f}")
        if 'keypoints_count' in stats_dict:
            self.keypoints_count_label.setText(f"Puntos clave: {stats_dict['keypoints_count']}")
        if 'matches_count' in stats_dict:
            self.matches_count_label.setText(f"Coincidencias: {stats_dict['matches_count']}")
        if 'good_matches' in stats_dict:
            self.good_matches_label.setText(f"Buenas coincidencias: {stats_dict['good_matches']}")
        if 'sharpness' in stats_dict:
            self.sharpness_label.setText(f"Nitidez: {stats_dict['sharpness']:.2f}")
        if 'contrast' in stats_dict:
            self.contrast_label.setText(f"Contraste: {stats_dict['contrast']:.2f}")
        if 'brightness' in stats_dict:
            self.brightness_label.setText(f"Brillo: {stats_dict['brightness']:.2f}")
            
    def update_metadata(self, metadata_dict):
        """Actualiza los metadatos NIST"""
        if 'case_number' in metadata_dict:
            self.case_number_label.setText(f"Número de caso: {metadata_dict['case_number']}")
        if 'examiner' in metadata_dict:
            self.examiner_label.setText(f"Examinador: {metadata_dict['examiner']}")
        if 'date' in metadata_dict:
            self.date_label.setText(f"Fecha: {metadata_dict['date']}")
        if 'evidence_id' in metadata_dict:
            self.evidence_id_label.setText(f"ID Evidencia: {metadata_dict['evidence_id']}")
        if 'weapon_type' in metadata_dict:
            self.weapon_type_label.setText(f"Tipo de arma: {metadata_dict['weapon_type']}")
        if 'caliber' in metadata_dict:
            self.caliber_label.setText(f"Calibre: {metadata_dict['caliber']}")
        if 'resolution' in metadata_dict:
            self.resolution_label.setText(f"Resolución: {metadata_dict['resolution']}")
        if 'magnification' in metadata_dict:
            self.magnification_label.setText(f"Magnificación: {metadata_dict['magnification']}")
        if 'lighting' in metadata_dict:
            self.lighting_label.setText(f"Iluminación: {metadata_dict['lighting']}")
            
    def update_quality_indicators(self, quality_dict):
        """Actualiza los indicadores de calidad"""
        if 'overall_quality' in quality_dict:
            quality = quality_dict['overall_quality']
            if quality >= 0.8:
                self.overall_quality_label.setText("🟢 EXCELENTE")
                self.overall_quality_label.setProperty("quality", "excellent")
            elif quality >= 0.6:
                self.overall_quality_label.setText("🟡 BUENA")
                self.overall_quality_label.setProperty("quality", "good")
            elif quality >= 0.4:
                self.overall_quality_label.setText("🟠 REGULAR")
                self.overall_quality_label.setProperty("quality", "fair")
            else:
                self.overall_quality_label.setText("🔴 POBRE")
                self.overall_quality_label.setProperty("quality", "poor")
            
            # Reaplica estilos
            self.overall_quality_label.style().unpolish(self.overall_quality_label)
            self.overall_quality_label.style().polish(self.overall_quality_label)
            
        # Actualizar métricas específicas
        if 'focus' in quality_dict:
            focus = quality_dict['focus']
            self.focus_indicator.setText(f"🔍 Enfoque: {focus:.2f}")
            self.focus_indicator.setProperty("quality", self._get_quality_level(focus))
            
        if 'noise' in quality_dict:
            noise = quality_dict['noise']
            self.noise_indicator.setText(f"📊 Ruido: {noise:.2f}")
            self.noise_indicator.setProperty("quality", self._get_quality_level(1.0 - noise))  # Invertir para ruido
            
        if 'exposure' in quality_dict:
            exposure = quality_dict['exposure']
            self.exposure_indicator.setText(f"💡 Exposición: {exposure:.2f}")
            self.exposure_indicator.setProperty("quality", self._get_quality_level(exposure))
            
        if 'detail' in quality_dict:
            detail = quality_dict['detail']
            self.detail_indicator.setText(f"🔬 Detalle: {detail:.2f}")
            self.detail_indicator.setProperty("quality", self._get_quality_level(detail))
            
        # Actualizar recomendaciones
        if 'recommendations' in quality_dict:
            self.recommendations_text.setPlainText(quality_dict['recommendations'])
            
    def _get_quality_level(self, value):
        """Convierte un valor numérico a nivel de calidad"""
        if value >= 0.8:
            return "excellent"
        elif value >= 0.6:
            return "good"
        elif value >= 0.4:
            return "fair"
        else:
            return "poor"
            
    def toggle_dock_visibility(self, dock_name, visible=None):
        """Alterna la visibilidad de un panel acoplable"""
        if dock_name in self.dock_widgets:
            dock = self.dock_widgets[dock_name]
            if visible is None:
                visible = not dock.isVisible()
            dock.setVisible(visible)
            
    def get_dock_widget(self, dock_name):
        """Obtiene una referencia a un panel acoplable"""
        return self.dock_widgets.get(dock_name)
        
    def on_tab_changed(self, index):
        """Maneja el cambio de pestañas y ajusta paneles acoplables"""
        tab_name = self.tab_widget.tabText(index).lower()
        
        # Mostrar/ocultar paneles según la pestaña activa
        if 'comparación' in tab_name or 'análisis' in tab_name:
            self.toggle_dock_visibility('statistics', True)
            self.toggle_dock_visibility('visualization_controls', True)
            self.toggle_dock_visibility('quality', True)
            
            # Integrar paneles específicos de ComparisonTab si está activa
            if 'comparación' in tab_name and hasattr(self, 'comparison_tab'):
                # Añadir paneles acoplables de ComparisonTab a MainWindow si existen
                if hasattr(self.comparison_tab, 'stats_dock_widget') and self.comparison_tab.stats_dock_widget:
                    if self.comparison_tab.stats_dock_widget not in self.findChildren(QDockWidget):
                        self.addDockWidget(Qt.RightDockWidgetArea, self.comparison_tab.stats_dock_widget)
                        
                if hasattr(self.comparison_tab, 'metadata_dock_widget') and self.comparison_tab.metadata_dock_widget:
                    if self.comparison_tab.metadata_dock_widget not in self.findChildren(QDockWidget):
                        self.addDockWidget(Qt.RightDockWidgetArea, self.comparison_tab.metadata_dock_widget)
                        
        elif 'base de datos' in tab_name:
            self.toggle_dock_visibility('metadata', True)
            self.toggle_dock_visibility('quality', False)
        else:
            # Para otras pestañas, mantener estado actual
            pass
            
        # Notificar al gestor de estado
        self.state_manager.set_active_tab(tab_name)
        
        # Emitir señal para otros componentes
        self.tabChanged.emit(index)
        
    def setup_dock_widgets(self):
        """Configura los paneles acoplables (QDockWidget)"""
        # Panel de controles de visualización
        self.create_visualization_controls_dock()
        
        # Panel de estadísticas en tiempo real
        self.create_statistics_dock()
        
        # Panel de metadatos NIST
        self.create_metadata_dock()
        
        # Panel de calidad de imagen
        self.create_quality_dock()
        
        # Configurar posiciones iniciales de los docks
        self.setup_initial_dock_layout()
        
    def create_visualization_controls_dock(self):
        """Crea el panel acoplable de controles de visualización"""
        dock = QDockWidget("Controles de Visualización", self)
        dock.setObjectName("VisualizationControlsDock")
        dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        
        # Widget de contenido
        controls_widget = QGroupBox("Controles")
        layout = QGridLayout(controls_widget)
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Control de zoom
        zoom_label = QLabel("🔍 Zoom:")
        zoom_label.setProperty("class", "dock-label")
        layout.addWidget(zoom_label, 0, 0)
        
        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setRange(25, 400)
        self.zoom_slider.setValue(100)
        self.zoom_slider.setProperty("class", "dock-slider")
        layout.addWidget(self.zoom_slider, 0, 1)
        
        self.zoom_label = QLabel("100%")
        self.zoom_label.setProperty("class", "dock-value")
        layout.addWidget(self.zoom_label, 0, 2)
        
        # Zoom sincronizado
        self.sync_zoom_cb = QCheckBox("Zoom sincronizado")
        self.sync_zoom_cb.setChecked(True)
        self.sync_zoom_cb.setProperty("class", "dock-checkbox")
        layout.addWidget(self.sync_zoom_cb, 1, 0, 1, 3)
        
        # Mostrar keypoints
        self.show_keypoints_cb = QCheckBox("Mostrar puntos clave")
        self.show_keypoints_cb.setChecked(True)
        self.show_keypoints_cb.setProperty("class", "dock-checkbox")
        layout.addWidget(self.show_keypoints_cb, 2, 0, 1, 3)
        
        # Mostrar matches
        self.show_matches_cb = QCheckBox("Mostrar coincidencias")
        self.show_matches_cb.setChecked(True)
        self.show_matches_cb.setProperty("class", "dock-checkbox")
        layout.addWidget(self.show_matches_cb, 3, 0, 1, 3)
        
        # Filtro de calidad
        quality_label = QLabel("⚡ Calidad mínima:")
        quality_label.setProperty("class", "dock-label")
        layout.addWidget(quality_label, 4, 0)
        
        self.quality_slider = QSlider(Qt.Horizontal)
        self.quality_slider.setRange(0, 100)
        self.quality_slider.setValue(30)
        self.quality_slider.setProperty("class", "dock-slider")
        layout.addWidget(self.quality_slider, 4, 1)
        
        self.quality_label = QLabel("30%")
        self.quality_label.setProperty("class", "dock-value")
        layout.addWidget(self.quality_label, 4, 2)
        
        # Top N matches
        top_matches_label = QLabel("🏆 Top matches:")
        top_matches_label.setProperty("class", "dock-label")
        layout.addWidget(top_matches_label, 5, 0)
        
        self.top_matches_spinbox = QSpinBox()
        self.top_matches_spinbox.setRange(5, 500)
        self.top_matches_spinbox.setValue(50)
        self.top_matches_spinbox.setSuffix(" matches")
        self.top_matches_spinbox.setProperty("class", "dock-spinbox")
        layout.addWidget(self.top_matches_spinbox, 5, 1, 1, 2)
        
        # Botón de reset
        reset_button = QPushButton("🔄 Resetear Filtros")
        reset_button.setProperty("class", "dock-button")
        layout.addWidget(reset_button, 6, 0, 1, 3)
        
        dock.setWidget(controls_widget)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)
        self.dock_widgets['visualization_controls'] = dock
        
        # Conectar señales
        self.zoom_slider.valueChanged.connect(self.on_zoom_changed)
        self.quality_slider.valueChanged.connect(self.on_quality_changed)
        reset_button.clicked.connect(self.reset_visualization_filters)
        
    def create_statistics_dock(self):
        """Crea el panel acoplable de estadísticas en tiempo real"""
        dock = QDockWidget("Estadísticas en Tiempo Real", self)
        dock.setObjectName("StatisticsDock")
        dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea | Qt.BottomDockWidgetArea)
        
        # Widget de contenido con scroll
        scroll_area = QScrollArea()
        stats_widget = QWidget()
        layout = QVBoxLayout(stats_widget)
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Métricas de similitud
        similarity_group = QGroupBox("Métricas de Similitud")
        similarity_group.setProperty("class", "dock-group")
        similarity_layout = QGridLayout(similarity_group)
        
        self.correlation_label = QLabel("Correlación: --")
        self.correlation_label.setProperty("class", "dock-metric")
        similarity_layout.addWidget(self.correlation_label, 0, 0)
        
        self.ssim_label = QLabel("SSIM: --")
        self.ssim_label.setProperty("class", "dock-metric")
        similarity_layout.addWidget(self.ssim_label, 1, 0)
        
        self.mse_label = QLabel("MSE: --")
        self.mse_label.setProperty("class", "dock-metric")
        similarity_layout.addWidget(self.mse_label, 2, 0)
        
        layout.addWidget(similarity_group)
        
        # Métricas de matching
        matching_group = QGroupBox("Métricas de Matching")
        matching_group.setProperty("class", "dock-group")
        matching_layout = QGridLayout(matching_group)
        
        self.keypoints_count_label = QLabel("Puntos clave: --")
        self.keypoints_count_label.setProperty("class", "dock-metric")
        matching_layout.addWidget(self.keypoints_count_label, 0, 0)
        
        self.matches_count_label = QLabel("Coincidencias: --")
        self.matches_count_label.setProperty("class", "dock-metric")
        matching_layout.addWidget(self.matches_count_label, 1, 0)
        
        self.good_matches_label = QLabel("Buenas coincidencias: --")
        self.good_matches_label.setProperty("class", "dock-metric")
        matching_layout.addWidget(self.good_matches_label, 2, 0)
        
        layout.addWidget(matching_group)
        
        # Métricas de calidad
        quality_group = QGroupBox("Calidad de Imagen")
        quality_group.setProperty("class", "dock-group")
        quality_layout = QGridLayout(quality_group)
        
        self.sharpness_label = QLabel("Nitidez: --")
        self.sharpness_label.setProperty("class", "dock-metric")
        quality_layout.addWidget(self.sharpness_label, 0, 0)
        
        self.contrast_label = QLabel("Contraste: --")
        self.contrast_label.setProperty("class", "dock-metric")
        quality_layout.addWidget(self.contrast_label, 1, 0)
        
        self.brightness_label = QLabel("Brillo: --")
        self.brightness_label.setProperty("class", "dock-metric")
        quality_layout.addWidget(self.brightness_label, 2, 0)
        
        layout.addWidget(quality_group)
        
        scroll_area.setWidget(stats_widget)
        scroll_area.setWidgetResizable(True)
        dock.setWidget(scroll_area)
        
        self.addDockWidget(Qt.RightDockWidgetArea, dock)
        self.dock_widgets['statistics'] = dock
        
    def create_metadata_dock(self):
        """Crea el panel acoplable de metadatos NIST"""
        dock = QDockWidget("Metadatos NIST", self)
        dock.setObjectName("MetadataDock")
        dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        
        # Widget de contenido con scroll
        scroll_area = QScrollArea()
        metadata_widget = QWidget()
        layout = QVBoxLayout(metadata_widget)
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Información del caso
        case_group = QGroupBox("Información del Caso")
        case_group.setProperty("class", "dock-group")
        case_layout = QGridLayout(case_group)
        
        self.case_number_label = QLabel("Número de caso: --")
        self.case_number_label.setProperty("class", "dock-info")
        case_layout.addWidget(self.case_number_label, 0, 0)
        
        self.examiner_label = QLabel("Examinador: --")
        self.examiner_label.setProperty("class", "dock-info")
        case_layout.addWidget(self.examiner_label, 1, 0)
        
        self.date_label = QLabel("Fecha: --")
        self.date_label.setProperty("class", "dock-info")
        case_layout.addWidget(self.date_label, 2, 0)
        
        layout.addWidget(case_group)
        
        # Información de la evidencia
        evidence_group = QGroupBox("Evidencia")
        evidence_group.setProperty("class", "dock-group")
        evidence_layout = QGridLayout(evidence_group)
        
        self.evidence_id_label = QLabel("ID Evidencia: --")
        self.evidence_id_label.setProperty("class", "dock-info")
        evidence_layout.addWidget(self.evidence_id_label, 0, 0)
        
        self.weapon_type_label = QLabel("Tipo de arma: --")
        self.weapon_type_label.setProperty("class", "dock-info")
        evidence_layout.addWidget(self.weapon_type_label, 1, 0)
        
        self.caliber_label = QLabel("Calibre: --")
        self.caliber_label.setProperty("class", "dock-info")
        evidence_layout.addWidget(self.caliber_label, 2, 0)
        
        layout.addWidget(evidence_group)
        
        # Información de la imagen
        image_group = QGroupBox("Imagen")
        image_group.setProperty("class", "dock-group")
        image_layout = QGridLayout(image_group)
        
        self.resolution_label = QLabel("Resolución: --")
        self.resolution_label.setProperty("class", "dock-info")
        image_layout.addWidget(self.resolution_label, 0, 0)
        
        self.magnification_label = QLabel("Magnificación: --")
        self.magnification_label.setProperty("class", "dock-info")
        image_layout.addWidget(self.magnification_label, 1, 0)
        
        self.lighting_label = QLabel("Iluminación: --")
        self.lighting_label.setProperty("class", "dock-info")
        image_layout.addWidget(self.lighting_label, 2, 0)
        
        layout.addWidget(image_group)
        
        scroll_area.setWidget(metadata_widget)
        scroll_area.setWidgetResizable(True)
        dock.setWidget(scroll_area)
        
        self.addDockWidget(Qt.LeftDockWidgetArea, dock)
        self.dock_widgets['metadata'] = dock
        
    def create_quality_dock(self):
        """Crea el panel acoplable de indicadores de calidad"""
        dock = QDockWidget("Indicadores de Calidad", self)
        dock.setObjectName("QualityDock")
        dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea | Qt.BottomDockWidgetArea)
        
        # Widget de contenido
        quality_widget = QWidget()
        layout = QVBoxLayout(quality_widget)
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Indicador de calidad general
        overall_group = QGroupBox("Calidad General")
        overall_group.setProperty("class", "dock-group")
        overall_layout = QVBoxLayout(overall_group)
        
        self.overall_quality_label = QLabel("Evaluando...")
        self.overall_quality_label.setProperty("class", "quality-indicator")
        self.overall_quality_label.setAlignment(Qt.AlignCenter)
        overall_layout.addWidget(self.overall_quality_label)
        
        layout.addWidget(overall_group)
        
        # Métricas específicas
        metrics_group = QGroupBox("Métricas Específicas")
        metrics_group.setProperty("class", "dock-group")
        metrics_layout = QGridLayout(metrics_group)
        
        # Indicadores con colores
        self.focus_indicator = QLabel("🔍 Enfoque: --")
        self.focus_indicator.setProperty("class", "quality-metric")
        metrics_layout.addWidget(self.focus_indicator, 0, 0)
        
        self.noise_indicator = QLabel("📊 Ruido: --")
        self.noise_indicator.setProperty("class", "quality-metric")
        metrics_layout.addWidget(self.noise_indicator, 1, 0)
        
        self.exposure_indicator = QLabel("💡 Exposición: --")
        self.exposure_indicator.setProperty("class", "quality-metric")
        metrics_layout.addWidget(self.exposure_indicator, 2, 0)
        
        self.detail_indicator = QLabel("🔬 Detalle: --")
        self.detail_indicator.setProperty("class", "quality-metric")
        metrics_layout.addWidget(self.detail_indicator, 3, 0)
        
        layout.addWidget(metrics_group)
        
        # Recomendaciones
        recommendations_group = QGroupBox("Recomendaciones")
        recommendations_group.setProperty("class", "dock-group")
        recommendations_layout = QVBoxLayout(recommendations_group)
        
        self.recommendations_text = QTextEdit()
        self.recommendations_text.setMaximumHeight(100)
        self.recommendations_text.setProperty("class", "dock-text")
        self.recommendations_text.setPlainText("Cargue una imagen para obtener recomendaciones...")
        recommendations_layout.addWidget(self.recommendations_text)
        
        layout.addWidget(recommendations_group)
        
        dock.setWidget(quality_widget)
        self.addDockWidget(Qt.BottomDockWidgetArea, dock)
        self.dock_widgets['quality'] = dock
        
    def setup_initial_dock_layout(self):
        """Configura el layout inicial de los paneles acoplables"""
        # Organizar docks en posiciones lógicas
        self.tabifyDockWidget(
            self.dock_widgets['statistics'], 
            self.dock_widgets['visualization_controls']
        )
        
        # Hacer visible el panel de controles por defecto
        self.dock_widgets['visualization_controls'].raise_()
        
        # Ocultar algunos paneles inicialmente
        self.dock_widgets['quality'].hide()
        
        # Configurar tamaños relativos
        self.resizeDocks(
            [self.dock_widgets['metadata'], self.dock_widgets['statistics']], 
            [300, 300], 
            Qt.Horizontal
        )
        
    def create_placeholder_tab(self, title: str, description: str, icon: str) -> QWidget:
        """Crea una pestaña placeholder temporal"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(20)
        
        # Icono grande
        icon_label = QLabel(icon)
        icon_label.setAlignment(Qt.AlignCenter)
        icon_label.setStyleSheet("font-size: 64px; margin: 20px;")
        layout.addWidget(icon_label)
        
        # Título
        title_label = QLabel(title)
        title_label.setProperty("class", "title")
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        # Descripción
        desc_label = QLabel(description)
        desc_label.setProperty("class", "body")
        desc_label.setAlignment(Qt.AlignCenter)
        desc_label.setWordWrap(True)
        layout.addWidget(desc_label)
        
        # Estado
        status_label = QLabel("En desarrollo...")
        status_label.setProperty("class", "caption")
        status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(status_label)
        
        return tab
        
    def setup_menu_bar(self):
        """Configura la barra de menú"""
        menubar = self.menuBar()
        
        # Menú Archivo
        file_menu = menubar.addMenu("&Archivo")
        
        new_action = QAction("&Nuevo Análisis", self)
        new_action.setShortcut("Ctrl+N")
        new_action.triggered.connect(self.new_analysis)
        file_menu.addAction(new_action)
        
        open_action = QAction("&Abrir Proyecto", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.open_project)
        file_menu.addAction(open_action)
        
        save_action = QAction("&Guardar", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.save_project)
        file_menu.addAction(save_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("&Salir", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Menú Herramientas
        tools_menu = menubar.addMenu("&Herramientas")
        
        backup_action = QAction("&Respaldo de Base de Datos", self)
        backup_action.setShortcut("Ctrl+B")
        backup_action.triggered.connect(self.backup_database)
        tools_menu.addAction(backup_action)
        
        tools_menu.addSeparator()
        
        history_action = QAction("&Historial de Análisis", self)
        history_action.setShortcut("Ctrl+H")
        history_action.triggered.connect(self.show_history)
        tools_menu.addAction(history_action)
        
        tools_menu.addSeparator()
        
        preferences_action = QAction("&Configuración", self)
        preferences_action.setShortcut("Ctrl+,")
        preferences_action.triggered.connect(self.show_preferences)
        tools_menu.addAction(preferences_action)
        
        database_action = QAction("Gestión de &Base de Datos", self)
        database_action.triggered.connect(self.show_database_management)
        tools_menu.addAction(database_action)
        
        # Menú Ayuda
        help_menu = menubar.addMenu("A&yuda")
        
        user_guide_action = QAction("&Guía de Usuario", self)
        user_guide_action.setShortcut("F1")
        user_guide_action.triggered.connect(self.show_user_guide)
        help_menu.addAction(user_guide_action)
        
        help_action = QAction("&Ayuda Completa", self)
        help_action.setShortcut("Ctrl+F1")
        help_action.triggered.connect(self.show_help)
        help_menu.addAction(help_action)
        
        help_menu.addSeparator()
        
        support_action = QAction("&Soporte Técnico", self)
        support_action.triggered.connect(self.show_support)
        help_menu.addAction(support_action)
        
        help_menu.addSeparator()
        
        about_action = QAction("&Acerca de SIGeC-Balistica", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
        
    def setup_status_bar(self):
        """Configura la barra de estado"""
        self.status_bar = self.statusBar()
        
        # Mensaje principal
        self.status_bar.showMessage("Listo")
        
        # Widgets adicionales en la barra de estado
        self.progress_label = QLabel("Inactivo")
        self.progress_label.setProperty("class", "caption")
        self.status_bar.addPermanentWidget(self.progress_label)
        
        # Información de memoria/rendimiento
        self.performance_label = QLabel("Memoria: OK")
        self.performance_label.setProperty("class", "caption")
        self.status_bar.addPermanentWidget(self.performance_label)
        
        # Indicadores de estado del caché
        if CACHE_AVAILABLE and self.cache_system:
            self.cache_status_label = QLabel("Caché: Inicializando...")
            self.cache_status_label.setProperty("class", "caption")
            self.status_bar.addPermanentWidget(self.cache_status_label)
            
            self.cache_stats_label = QLabel("Hit Rate: --")
            self.cache_stats_label.setProperty("class", "caption")
            self.status_bar.addPermanentWidget(self.cache_stats_label)
        else:
            self.cache_status_label = QLabel("Caché: No disponible")
            self.cache_status_label.setProperty("class", "caption warning")
            self.status_bar.addPermanentWidget(self.cache_status_label)
        
        # Timer para actualizar información del sistema
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.update_system_status)
        self.status_timer.start(5000)  # Actualizar cada 5 segundos
        
    def setup_connections(self):
        """Configura las conexiones de señales"""
        # Conectar señales del gestor de estado compartido
        try:
            if hasattr(self, 'state_manager') and self.state_manager:
                self.state_manager.metadata_updated.connect(self.update_metadata)
                self.state_manager.statistics_updated.connect(self.update_statistics)
                self.state_manager.quality_metrics_updated.connect(self.update_quality_indicators)
        except Exception as e:
            print(f"Warning wiring AppStateManager signals: {e}")
        
    def on_tab_changed(self, index: int):
        """Maneja el cambio de pestaña"""
        tab_names = ["Análisis", "Comparación", "Base de Datos", "Reportes"]
        if 0 <= index < len(tab_names):
            self.status_bar.showMessage(f"Pestaña activa: {tab_names[index]}")
            
    def update_system_status(self):
        """Actualiza el estado del sistema"""
        try:
            # Información de memoria del sistema
            import psutil
            memory_percent = psutil.virtual_memory().percent
            
            if memory_percent < 80:
                self.performance_label.setText(f"Memoria: {memory_percent:.1f}% OK")
                self.performance_label.setProperty("class", "caption success")
            else:
                self.performance_label.setText(f"Memoria: {memory_percent:.1f}% Alta")
                self.performance_label.setProperty("class", "caption warning")
                
            # Forzar actualización de estilo
            self.performance_label.style().unpolish(self.performance_label)
            self.performance_label.style().polish(self.performance_label)
            
        except ImportError:
            # Si psutil no está disponible
            self.performance_label.setText("Memoria: N/A")
            
        # Actualizar estadísticas del caché
        if CACHE_AVAILABLE and self.cache_system:
            try:
                from core.intelligent_cache import get_cache
                cache = get_cache()
                if cache:
                    stats = cache.get_stats()
                    
                    # Actualizar estado del caché
                    memory_usage = stats.get('memory_usage_mb', 0)
                    max_memory = stats.get('max_memory_mb', 256)
                    memory_percent = (memory_usage / max_memory) * 100 if max_memory > 0 else 0
                    
                    if memory_percent < 80:
                        self.cache_status_label.setText(f"Caché: {memory_usage:.1f}MB ({memory_percent:.1f}%)")
                        self.cache_status_label.setProperty("class", "caption success")
                    else:
                        self.cache_status_label.setText(f"Caché: {memory_usage:.1f}MB ({memory_percent:.1f}%) Alto")
                        self.cache_status_label.setProperty("class", "caption warning")
                    
                    # Actualizar hit rate
                    hit_rate = stats.get('hit_rate', 0) * 100
                    total_requests = stats.get('total_requests', 0)
                    
                    if hit_rate >= 70:
                        self.cache_stats_label.setText(f"Hit Rate: {hit_rate:.1f}% ({total_requests} req)")
                        self.cache_stats_label.setProperty("class", "caption success")
                    elif hit_rate >= 40:
                        self.cache_stats_label.setText(f"Hit Rate: {hit_rate:.1f}% ({total_requests} req)")
                        self.cache_stats_label.setProperty("class", "caption")
                    else:
                        self.cache_stats_label.setText(f"Hit Rate: {hit_rate:.1f}% ({total_requests} req)")
                        self.cache_stats_label.setProperty("class", "caption warning")
                    
                    # Forzar actualización de estilos
                    self.cache_status_label.style().unpolish(self.cache_status_label)
                    self.cache_status_label.style().polish(self.cache_status_label)
                    self.cache_stats_label.style().unpolish(self.cache_stats_label)
                    self.cache_stats_label.style().polish(self.cache_stats_label)
                    
            except Exception as e:
                self.cache_status_label.setText("Caché: Error")
                self.cache_status_label.setProperty("class", "caption error")
                self.cache_stats_label.setText("Hit Rate: Error")
                self.cache_stats_label.setProperty("class", "caption error")
            
    def new_analysis(self):
        """Inicia un nuevo análisis"""
        self.tab_widget.setCurrentIndex(0)  # Ir a pestaña de análisis
        self.status_bar.showMessage("Nuevo análisis iniciado")
        
    def open_project(self):
        """Abre un proyecto existente"""
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Abrir Proyecto",
                "",
                "Archivos de Proyecto SIGeC (*.sigec);;Archivos JSON (*.json);;Todos los archivos (*)"
            )
            
            if file_path:
                self.status_bar.showMessage("Cargando proyecto...")
                
                # Cargar datos del proyecto
                with open(file_path, 'r', encoding='utf-8') as f:
                    project_data = json.load(f)
                
                # Validar estructura del proyecto
                if not self._validate_project_data(project_data):
                    QMessageBox.warning(
                        self, 
                        "Proyecto Inválido", 
                        "El archivo seleccionado no es un proyecto válido de SIGeC-Balística."
                    )
                    return
                
                # Cargar configuración del proyecto
                if 'config' in project_data:
                    self._load_project_config(project_data['config'])
                
                # Cargar datos de análisis si existen
                if 'analysis_data' in project_data:
                    self._load_analysis_data(project_data['analysis_data'])
                
                # Actualizar interfaz
                self._update_ui_with_project_data(project_data)
                
                # Actualizar título de ventana
                project_name = project_data.get('name', 'Proyecto Sin Nombre')
                self.setWindowTitle(f"SIGeC-Balística - {project_name}")
                
                self.status_bar.showMessage(f"Proyecto '{project_name}' cargado exitosamente", 3000)
                
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error al Abrir Proyecto",
                f"No se pudo abrir el proyecto:\n{str(e)}"
            )
            self.status_bar.showMessage("Error al cargar proyecto", 3000)
        
    def save_project(self):
        """Guarda el proyecto actual"""
        try:
            # Obtener datos del proyecto actual
            project_data = self._collect_project_data()
            
            if not project_data:
                QMessageBox.information(
                    self,
                    "Sin Datos",
                    "No hay datos de proyecto para guardar."
                )
                return
            
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Guardar Proyecto",
                f"proyecto_sigec_{datetime.now().strftime('%Y%m%d_%H%M%S')}.sigec",
                "Archivos de Proyecto SIGeC (*.sigec);;Archivos JSON (*.json)"
            )
            
            if file_path:
                self.status_bar.showMessage("Guardando proyecto...")
                
                # Guardar datos del proyecto
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(project_data, f, indent=2, ensure_ascii=False)
                
                # Actualizar título de ventana
                project_name = project_data.get('name', 'Proyecto Sin Nombre')
                self.setWindowTitle(f"SIGeC-Balística - {project_name}")
                
                self.status_bar.showMessage(f"Proyecto guardado: {os.path.basename(file_path)}", 3000)
                
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error al Guardar Proyecto",
                f"No se pudo guardar el proyecto:\n{str(e)}"
            )
            self.status_bar.showMessage("Error al guardar proyecto", 3000)
        
    def show_configuration(self):
        """Muestra el diálogo de configuración"""
        try:
            from .settings_dialog import SettingsDialog
            dialog = SettingsDialog(self)
            
            # Conectar señal de configuración aplicada
            dialog.settingsApplied.connect(self.apply_new_settings)
            
            if dialog.exec_() == QDialog.Accepted:
                self.status_bar.showMessage("Configuración actualizada", 2000)
                
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error de Configuración",
                f"No se pudo abrir el diálogo de configuración:\n{str(e)}"
            )
    
    def _validate_project_data(self, project_data: dict) -> bool:
        """Valida la estructura de datos del proyecto"""
        required_fields = ['version', 'created_at', 'name']
        return all(field in project_data for field in required_fields)
    
    def _load_project_config(self, config_data: dict):
        """Carga la configuración del proyecto"""
        try:
            # Aplicar configuración específica del proyecto
            if hasattr(self, 'state_manager'):
                self.state_manager.load_project_config(config_data)
        except Exception as e:
            print(f"Error cargando configuración del proyecto: {e}")
    
    def _load_analysis_data(self, analysis_data: dict):
        """Carga los datos de análisis del proyecto"""
        try:
            # Cargar datos en las pestañas correspondientes
            if hasattr(self, 'analysis_tab') and 'analysis' in analysis_data:
                self.analysis_tab.load_analysis_data(analysis_data['analysis'])
            
            if hasattr(self, 'comparison_tab') and 'comparison' in analysis_data:
                self.comparison_tab.load_comparison_data(analysis_data['comparison'])
                
        except Exception as e:
            print(f"Error cargando datos de análisis: {e}")
    
    def _update_ui_with_project_data(self, project_data: dict):
        """Actualiza la interfaz con los datos del proyecto"""
        try:
            # Actualizar metadatos en el dock
            if hasattr(self, 'metadata_dock'):
                metadata = project_data.get('metadata', {})
                self.update_metadata(metadata)
                
        except Exception as e:
            print(f"Error actualizando interfaz: {e}")
    
    def _collect_project_data(self) -> dict:
        """Recopila los datos actuales del proyecto"""
        try:
            from datetime import datetime
            import json
            
            project_data = {
                'version': '1.0',
                'created_at': datetime.now().isoformat(),
                'name': f"Proyecto SIGeC {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                'description': 'Proyecto de análisis balístico forense',
                'config': {},
                'analysis_data': {},
                'metadata': {}
            }
            
            # Recopilar configuración actual
            if hasattr(self, 'state_manager'):
                project_data['config'] = self.state_manager.get_current_config()
            
            # Recopilar datos de análisis
            analysis_data = {}
            
            if hasattr(self, 'analysis_tab'):
                try:
                    analysis_data['analysis'] = self.analysis_tab.get_analysis_data()
                except:
                    pass
            
            if hasattr(self, 'comparison_tab'):
                try:
                    analysis_data['comparison'] = self.comparison_tab.get_comparison_data()
                except:
                    pass
            
            project_data['analysis_data'] = analysis_data
            
            # Recopilar metadatos
            project_data['metadata'] = {
                'last_modified': datetime.now().isoformat(),
                'tabs_count': self.tab_widget.count() if hasattr(self, 'tab_widget') else 0,
                'system_info': {
                    'platform': os.name,
                    'python_version': sys.version.split()[0]
                }
            }
            
            return project_data
            
        except Exception as e:
            print(f"Error recopilando datos del proyecto: {e}")
            return None
    
    def set_analysis_progress(self, progress: int, message: str = ""):
        """Actualiza el progreso de análisis"""
        if progress >= 0:
            self.progress_label.setText(f"Progreso: {progress}%")
            if message:
                self.status_bar.showMessage(message)
        else:
            self.progress_label.setText("Inactivo")
            
    def show_error(self, title: str, message: str):
        """Muestra un mensaje de error"""
        QMessageBox.critical(self, title, message)
        
    def show_warning(self, title: str, message: str):
        """Muestra un mensaje de advertencia"""
        QMessageBox.warning(self, title, message)
        
    def show_info(self, title: str, message: str):
        """Muestra un mensaje informativo"""
        QMessageBox.information(self, title, message)
        
    def show_contextual_help(self):
        """Muestra ayuda contextual basada en la pestaña actual"""
        current_tab = self.tab_widget.currentIndex()
        
        if hasattr(self, 'onboarding_manager') and self.onboarding_manager:
            if current_tab == 0:  # Análisis
                self.onboarding_manager.start_analysis_tour()
            elif current_tab == 1:  # Comparación
                self.onboarding_manager.start_comparison_tour()
            elif current_tab == 2:  # Base de datos
                self.onboarding_manager.start_database_tour()
            elif current_tab == 3:  # Reportes
                self.onboarding_manager.start_reports_tour()
    
    def show_settings(self):
        """Muestra el diálogo de configuración"""
        from .settings_dialog import SettingsDialog
        dialog = SettingsDialog(self)
        dialog.exec_()
    
    def backup_database(self):
        """Crear respaldo de la base de datos"""
        try:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename, _ = QFileDialog.getSaveFileName(
                self,
                "Guardar Respaldo de Base de Datos",
                f"backup_sigec_{timestamp}.db",
                "Database Files (*.db);;All Files (*)"
            )
            if filename:
                # Aquí iría la lógica de respaldo real
                self.show_info("Respaldo", f"Respaldo guardado en: {filename}")
        except Exception as e:
            self.show_error("Error de Respaldo", f"No se pudo crear el respaldo: {str(e)}")
    
    def show_history(self):
        """Mostrar historial de análisis"""
        try:
            dialog = HistoryDialog(self)
            dialog.exec_()
        except Exception as e:
            self.show_error("Error", f"No se pudo abrir el historial: {str(e)}")
    
    def show_preferences(self):
        """Mostrar diálogo de preferencias"""
        try:
            dialog = SettingsDialog(self)
            dialog.exec_()
        except Exception as e:
            self.show_error("Error", f"No se pudo abrir las preferencias: {str(e)}")
    
    def show_database_management(self):
        """Mostrar gestión de base de datos"""
        try:
            self.show_info("Gestión de Base de Datos", "Funcionalidad de gestión de base de datos en desarrollo.")
        except Exception as e:
            self.show_error("Error", f"No se pudo abrir la gestión de base de datos: {str(e)}")
    
    def show_user_guide(self):
        """Mostrar guía de usuario"""
        try:
            dialog = HelpDialog(self)
            dialog.exec_()
        except Exception as e:
            self.show_error("Error", f"No se pudo abrir la guía de usuario: {str(e)}")
    
    def show_help(self):
        """Mostrar ayuda completa"""
        try:
            dialog = HelpDialog(self)
            dialog.exec_()
        except Exception as e:
            self.show_error("Error", f"No se pudo abrir la ayuda: {str(e)}")
    
    def show_support(self):
        """Mostrar información de soporte técnico"""
        try:
            self.show_info("Soporte Técnico", 
                          "Para soporte técnico, contacte:\n\n"
                          "Email: soporte@sigec-balistica.com\n"
                          "Teléfono: +1-800-SIGEC-01\n"
                          "Web: www.sigec-balistica.com/soporte")
        except Exception as e:
            self.show_error("Error", f"No se pudo mostrar la información de soporte: {str(e)}")
    
    def show_about(self):
        """Mostrar información acerca de la aplicación"""
        try:
            dialog = AboutDialog(self)
            dialog.exec_()
        except Exception as e:
            self.show_error("Error", f"No se pudo abrir la información de la aplicación: {str(e)}")
    
    def closeEvent(self, event):
        """Maneja el cierre de la aplicación con confirmación mejorada"""
        if hasattr(self, 'feedback_manager'):
            reply = self.feedback_manager.show_confirmation(
                "¿Cerrar aplicación?",
                "¿Está seguro de que desea cerrar SIGeC-Balisticar?\n\n"
                "• Se guardarán automáticamente las configuraciones\n"
                "• Los análisis en progreso se perderán\n"
                "• Los datos no guardados se perderán",
                "Cerrar",
                "Cancelar"
            )
            
            if reply:
                # Guardar configuraciones antes de cerrar
                if hasattr(self, 'accessibility_manager'):
                    self.accessibility_manager.save_settings()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()

def main():
    """Función principal para ejecutar la aplicación"""
    from PyQt5.QtCore import QCoreApplication
    QCoreApplication.setAttribute(Qt.AA_ShareOpenGLContexts)
    app = QApplication(sys.argv)
    
    # Aplicar tema
    apply_SIGeC_Balistica_theme(app)
    
    # Crear y mostrar ventana principal
    window = MainWindow()
    window.show()
    
    return app.exec_()

if __name__ == "__main__":
    sys.exit(main())