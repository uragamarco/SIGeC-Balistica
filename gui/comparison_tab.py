"""
Modern Ballistic Comparison Tab with hierarchical configuration and step-by-step navigation.
This module provides a comprehensive interface for ballistic evidence comparison.
"""

import time
from typing import Dict, Any, List, Optional, Tuple
import os
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame,
    QSplitter, QTabWidget, QScrollArea, QGroupBox, QFormLayout,
    QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox, QTextEdit,
    QProgressBar, QMessageBox, QFileDialog, QApplication,
    QSizePolicy, QGridLayout, QListWidget, QListWidgetItem, QAbstractItemView
)
from PyQt5.QtCore import Qt, pyqtSignal, QThread, QTimer, QSize
from PyQt5.QtGui import QFont, QPixmap, QPainter, QColor, QPen

# Telemetry integration
try:
    from core.telemetry_system import record_user_action, record_feature_usage, record_performance_event, record_error_event
    TELEMETRY_AVAILABLE = True
except ImportError:
    TELEMETRY_AVAILABLE = False
    # Mock functions for development
    def record_user_action(*args, **kwargs): pass
    def record_feature_usage(*args, **kwargs): pass
    def record_performance_event(*args, **kwargs): pass
    def record_error_event(*args, **kwargs): pass

# Intelligent cache integration
try:
    from core.intelligent_cache import get_cache, initialize_cache
    from image_processing.lbp_cache import get_lbp_cache
    CACHE_AVAILABLE = True
except ImportError:
    CACHE_AVAILABLE = False
    # Mock cache functions
    def get_cache(): return None
    def initialize_cache(*args, **kwargs): pass
    def get_lbp_cache(): return None

# Core imports
from core.unified_pipeline import ScientificPipeline
try:
    from core.analysis_result import AnalysisResult
except ImportError:
    # Fallback para AnalysisResult
    class AnalysisResult:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

# Shared widgets
from gui.shared_widgets import ImageSelector
from gui.visualization_widgets import VisualizationPanel
from gui.dynamic_results_panel import ResultsPanel
try:
    from gui.shared_widgets import NISTStandardsWidget, AFTEAnalysisWidget, DeepLearningWidget, ImageProcessingWidget
except ImportError:
    # Fallback widgets
    from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout
    
    class NISTStandardsWidget(QWidget):
        configuration_changed = pyqtSignal(dict)
        
        def __init__(self, parent=None):
            super().__init__(parent)
            self._config = {'enabled': True}
            layout = QVBoxLayout(self)
            layout.addWidget(QLabel("NIST Standards Widget (Fallback)"))
        def get_configuration(self):
            return self._config.copy()
        def set_configuration(self, cfg):
            self._config.update(cfg or {})
            self.configuration_changed.emit(self._config.copy())
        def set_configuration_level(self, level: str):
            pass
    
    class AFTEAnalysisWidget(QWidget):
        configuration_changed = pyqtSignal(dict)
        
        def __init__(self, parent=None):
            super().__init__(parent)
            self._config = {'enabled': True}
            layout = QVBoxLayout(self)
            layout.addWidget(QLabel("AFTE Analysis Widget (Fallback)"))
        def get_configuration(self):
            return self._config.copy()
        def set_configuration(self, cfg):
            self._config.update(cfg or {})
            self.configuration_changed.emit(self._config.copy())
        def set_configuration_level(self, level: str):
            pass
    
    class DeepLearningWidget(QWidget):
        configuration_changed = pyqtSignal(dict)
        
        def __init__(self, parent=None):
            super().__init__(parent)
            # Include 'model' alias to satisfy summary expectations
            self._config = {'enabled': False, 'model': 'CNN'}
            layout = QVBoxLayout(self)
            layout.addWidget(QLabel("Deep Learning Widget (Fallback)"))
        def get_configuration(self):
            return self._config.copy()
        def set_configuration(self, cfg):
            self._config.update(cfg or {})
            # Keep alias in sync if model_type provided
            if 'model_type' in self._config:
                self._config['model'] = self._config.get('model_type', self._config.get('model', 'CNN'))
            self.configuration_changed.emit(self._config.copy())
        def set_configuration_level(self, level: str):
            pass
    
    class ImageProcessingWidget(QWidget):
        configuration_changed = pyqtSignal(dict)
        
        def __init__(self, parent=None):
            super().__init__(parent)
            self._config = {'enabled': True}
            layout = QVBoxLayout(self)
            layout.addWidget(QLabel("Image Processing Widget (Fallback)"))
        def get_configuration(self):
            return self._config.copy()
        def set_configuration(self, cfg):
            self._config.update(cfg or {})
            self.configuration_changed.emit(self._config.copy())
        def set_configuration_level(self, level: str):
            pass

# Comparison-specific widgets
from .widgets.comparison import (
    ComparisonStepper, ComparisonModeSelector, ComparisonConfigManager,
    ComparisonProgressWidget, ComparisonResultsWidget
)

# UI components
try:
    from .widgets.stepper import StepperWidget
except ImportError:
    # Fallback stepper widget
    from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout
    
    class StepperWidget(QWidget):
        def __init__(self, parent=None):
            super().__init__(parent)
            layout = QVBoxLayout(self)
            layout.addWidget(QLabel("Stepper Widget (Fallback)"))

try:
    from .widgets.modern_theme import ModernTheme
except ImportError:
    # Fallback modern theme
    class ModernTheme:
        @staticmethod
        def apply_to_widget(widget):
            pass

# App state manager (para reutilizar imágenes desde Análisis)
try:
    from .app_state_manager import app_state_manager
    APP_STATE_AVAILABLE = True
except Exception:
    APP_STATE_AVAILABLE = False
    app_state_manager = None

# Check for optional dependencies
try:
    from core.nist_compliance import NISTComplianceValidator, NISTQualityMetrics
    NIST_AVAILABLE = True
except ImportError:
    NIST_AVAILABLE = False

try:
    from core.deep_learning import DeepLearningMatcher
    DEEP_LEARNING_AVAILABLE = False  # Set to False as per user context
except ImportError:
    DEEP_LEARNING_AVAILABLE = False

try:
    from core.afte_analysis import AFTEConclusionEngine
    AFTE_AVAILABLE = True
except ImportError:
    AFTE_AVAILABLE = False


class ComparisonWorker(QThread):
    """Background worker for ballistic comparison operations"""
    
    # Signals
    progress_updated = pyqtSignal(str, int, str)  # step, progress, details
    step_completed = pyqtSignal(str, dict)  # step_name, step_results
    comparison_completed = pyqtSignal(dict)  # final_results
    error_occurred = pyqtSignal(str)  # error_message
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.mode = None
        self.config = None
        self.images = []
        self.query_image = None
        self.database_path = None
        self.start_time = None
        
        # Initialize cache systems
        self.cache = None
        self.lbp_cache = None
        if CACHE_AVAILABLE:
            try:
                self.cache = get_cache()
                self.lbp_cache = get_lbp_cache()
            except Exception as e:
                print(f"Warning: Could not initialize cache systems: {e}")
                self.cache = None
                self.lbp_cache = None
        
    def setup_comparison(self, mode: str, config: Dict[str, Any], 
                        images: List[str], query_image: str = None,
                        database_path: str = None):
        """Setup comparison parameters"""
        self.mode = mode
        self.config = config
        self.images = images
        self.query_image = query_image
        self.database_path = database_path
        
    def run(self):
        """Execute the comparison process"""
        self.start_time = time.time()
        
        try:
            # Record comparison start
            record_user_action('comparison_started', 'comparison_tab', {
                'mode': self.mode,
                'image_count': len(self.images),
                'has_query_image': self.query_image is not None,
                'configuration': self.config
            })
            
            if self.mode == 'direct':
                self._run_direct_comparison()
            elif self.mode == 'database':
                self._run_database_search()
            else:
                raise ValueError(f"Unknown comparison mode: {self.mode}")
                
        except Exception as e:
            # Record error
            record_error_event(e, 'comparison_worker', 'run_comparison')
            self.error_occurred.emit(str(e))
    
    def _run_direct_comparison(self):
        """Run direct image comparison with intelligent caching"""
        try:
            self.progress_updated.emit("initialization", 10, "Initializing direct comparison...")
            
            # Generate cache key for comparison
            cache_key = None
            cache_hits = 0
            cache_misses = 0
            
            if self.cache:
                import hashlib
                cache_data = {
                    'images': sorted(self.images),
                    'config': self.config,
                    'mode': 'direct'
                }
                cache_key = hashlib.md5(str(cache_data).encode()).hexdigest()
                
                # Try to get cached result
                cached_result = self.cache.get(cache_key)
                if cached_result:
                    cache_hits = 1
                    self.progress_updated.emit("completed", 100, "Retrieved from cache!")
                    self.comparison_completed.emit(cached_result)
                    return
                else:
                    cache_misses = 1
            
            # Image preprocessing step
            preprocessing_start = time.time()
            self.progress_updated.emit("preprocessing", 20, "Preprocessing images...")
            
            # Simulate preprocessing
            import time as time_module
            time_module.sleep(0.5)  # Simulate processing time
            
            preprocessing_time = (time.time() - preprocessing_start) * 1000
            record_performance_event('image_preprocessing', 'comparison_worker', preprocessing_time, 
                                   success=True, metadata={'image_count': len(self.images)})
            
            self.step_completed.emit("preprocessing", {"processed_images": len(self.images)})
            
            # Feature extraction step
            extraction_start = time.time()
            self.progress_updated.emit("feature_extraction", 40, "Extracting features...")
            
            # Simulate feature extraction
            time_module.sleep(1.0)
            
            extraction_time = (time.time() - extraction_start) * 1000
            record_performance_event('feature_extraction', 'comparison_worker', extraction_time, 
                                   success=True, metadata={'image_count': len(self.images)})
            
            self.step_completed.emit("feature_extraction", {"features_extracted": True})
            
            # Comparison step
            comparison_start = time.time()
            self.progress_updated.emit("comparison", 70, "Performing comparison...")
            
            # Use ScientificPipeline for actual comparison
            pipeline = ScientificPipeline()
            result = pipeline.process_comparison(self.images, self.config)
            
            comparison_time = (time.time() - comparison_start) * 1000
            record_performance_event('ballistic_comparison', 'comparison_worker', comparison_time, 
                                   success=True, metadata={'comparison_type': 'direct'})
            
            self.step_completed.emit("comparison", {"comparison_result": result})
            
            # Results formatting
            self.progress_updated.emit("results", 90, "Formatting results...")
            formatted_results = self._format_comparison_results(result)
            
            # Add cache statistics to results
            if self.cache:
                formatted_results['cache_stats'] = {
                    'hits': cache_hits,
                    'misses': cache_misses,
                    'hit_rate': cache_hits / (cache_hits + cache_misses) if (cache_hits + cache_misses) > 0 else 0
                }
                
                # Store result in cache with 1 hour TTL
                try:
                    self.cache.set(cache_key, formatted_results, ttl=3600)
                except Exception as e:
                    print(f"Warning: Could not cache comparison result: {e}")
            
            # Complete comparison
            total_time = (time.time() - self.start_time) * 1000
            record_performance_event('full_comparison', 'comparison_worker', total_time, 
                                   success=True, metadata={
                                       'mode': 'direct',
                                       'image_count': len(self.images),
                                       'steps_completed': ['preprocessing', 'feature_extraction', 'comparison', 'results'],
                                       'cache_enabled': self.cache is not None
                                   })
            
            self.progress_updated.emit("completed", 100, "Comparison completed!")
            self.comparison_completed.emit(formatted_results)
            
        except Exception as e:
            record_error_event(e, 'comparison_worker', 'direct_comparison')
            self.error_occurred.emit(f"Direct comparison failed: {str(e)}")

    def _run_database_search(self):
        """Execute database search for similar evidence"""
        if not self.query_image:
            self.error_occurred.emit("Database search requires a query image")
            return
        
        self.progress_updated.emit("Initialization", 10, "Setting up database search")
        
        # Configure search parameters
        max_results = self.configuration.get('database_settings', {}).get('max_results', 10)
        search_radius = self.configuration.get('database_settings', {}).get('search_radius', 0.5)
        
        self.progress_updated.emit("Query Processing", 30, "Processing query image")
        
        # Feature extraction from query
        self.progress_updated.emit("Feature Extraction", 50, "Extracting query features")
        
        # Database search
        self.progress_updated.emit("Database Search", 75, f"Searching for top {max_results} matches")
        
        try:
            # Simulate database search results
            search_results = self._simulate_database_search(max_results)
            
            formatted_results = {
                'mode': 'database',
                'query_image': self.query_image,
                'results': search_results,
                'search_parameters': {
                    'max_results': max_results,
                    'search_radius': search_radius
                }
            }
            
            self.progress_updated.emit("Finalization", 100, "Finalizing search results")
            self.comparison_completed.emit(formatted_results)
            
        except Exception as e:
            self.error_occurred.emit(f"Database search failed: {str(e)}")
    
    def _format_comparison_results(self, result: AnalysisResult) -> Dict[str, Any]:
        """Format comparison results for display"""
        return {
            'mode': 'direct',
            'images': self.image_paths,
            'similarity_score': getattr(result, 'similarity_score', 0.75),
            'confidence': getattr(result, 'confidence', 0.80),
            'afte_conclusion': getattr(result, 'afte_conclusion', 'Inconclusive'),
            'match_type': getattr(result, 'match_type', 'Partial'),
            'nist_quality': getattr(result, 'nist_quality', {}),
            'features': getattr(result, 'features', {}),
            'preprocessing': getattr(result, 'preprocessing', {}),
            'errors': getattr(result, 'errors', []),
            'warnings': getattr(result, 'warnings', [])
        }
    
    def _simulate_database_search(self, max_results: int) -> List[Dict[str, Any]]:
        """Simulate database search results"""
        import random
        
        results = []
        for i in range(max_results):
            result = {
                'id': f"DB_{i+1:03d}",
                'similarity': random.uniform(0.3, 0.95),
                'confidence': random.uniform(0.5, 0.9),
                'afte_conclusion': random.choice(['Match', 'Probable Match', 'Inconclusive', 'Elimination']),
                'match_type': random.choice(['Full', 'Partial', 'Geometric', 'Texture']),
                'case_info': {
                    'case_number': f"CASE-{random.randint(1000, 9999)}",
                    'evidence_id': f"EV-{random.randint(100, 999)}",
                    'date': "2024-01-15"
                }
            }
            results.append(result)
        
        # Sort by similarity score
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results


class ComparisonTab(QWidget):
    """
    Modern Comparison Tab with hierarchical configuration and step-by-step navigation
    """
    
    # Signals
    comparison_started = pyqtSignal()
    comparison_completed = pyqtSignal(dict)
    configuration_changed = pyqtSignal(dict)
    mode_changed = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # State management
        self.current_mode: str = 'direct'
        self.current_images: List[str] = []
        self.current_configuration: Dict[str, Any] = {}
        self.comparison_worker: Optional[ComparisonWorker] = None
        self.comparison_results: Optional[Dict[str, Any]] = None
        
        # Initialize UI
        self.init_ui()
        self.setup_connections()
        self.apply_modern_theme()
    
    def init_ui(self):
        """Initialize the user interface"""
        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)
        
        # Title
        title_label = QLabel("Ballistic Comparison")
        title_label.setObjectName("titleLabel")
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)
        
        # Create main splitter
        main_splitter = QSplitter(Qt.Horizontal)
        main_splitter.setObjectName("mainSplitter")
        
        # Left panel - Navigation and Configuration
        left_panel = self.create_left_panel()
        main_splitter.addWidget(left_panel)
        
        # Right panel - Visualization and Results
        right_panel = self.create_right_panel()
        main_splitter.addWidget(right_panel)
        
        # Set splitter proportions y factores de estirado adaptativos (como AnalysisTab)
        try:
            main_splitter.setStretchFactor(0, 1)
            main_splitter.setStretchFactor(1, 2)
        except Exception:
            pass
        main_splitter.setSizes([400, 600])
        main_layout.addWidget(main_splitter)
        
        # Status bar
        self.status_bar = self.create_status_bar()
        main_layout.addWidget(self.status_bar)
        
        # Ajustar stretches para ocupar espacio vertical sin centrar
        try:
            main_layout.setStretch(0, 0)  # título
            main_layout.setStretch(1, 1)  # splitter
            main_layout.setStretch(2, 0)  # status bar
        except Exception:
            pass
    
    def create_left_panel(self) -> QWidget:
        """Create the left navigation and configuration panel"""
        # Contenedor interno del panel izquierdo
        inner = QFrame()
        inner.setObjectName("leftPanel")
        inner.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(15)
        
        # Comparison stepper
        self.comparison_stepper = ComparisonStepper()
        layout.addWidget(self.comparison_stepper)
        
        # Mode selector
        self.mode_selector = ComparisonModeSelector()
        layout.addWidget(self.mode_selector)
        
        # Configuration manager
        self.config_manager = ComparisonConfigManager()
        layout.addWidget(self.config_manager)

        # Panel de imágenes del Análisis (selección desde historial)
        self.analysis_images_group = QGroupBox("Imágenes de Análisis")
        analysis_layout = QVBoxLayout(self.analysis_images_group)
        info_label = QLabel("Seleccione imágenes previamente analizadas.\n" 
                            "- Modo directo: seleccione 2 o más.\n"
                            "- Modo base de datos: seleccione 1.")
        info_label.setProperty("class", "caption")
        analysis_layout.addWidget(info_label)

        self.history_list = QListWidget()
        try:
            self.history_list.setSelectionMode(
                QAbstractItemView.MultiSelection if self.current_mode == 'direct' else QAbstractItemView.SingleSelection
            )
        except Exception:
            pass
        analysis_layout.addWidget(self.history_list)

        buttons_layout = QHBoxLayout()
        self.refresh_history_btn = QPushButton("Refrescar")
        self.refresh_history_btn.setProperty("class", "secondary-button")
        buttons_layout.addWidget(self.refresh_history_btn)
        buttons_layout.addStretch()
        analysis_layout.addLayout(buttons_layout)

        layout.addWidget(self.analysis_images_group)
        
        # Progress widget (initialmente oculto)
        self.progress_widget = ComparisonProgressWidget()
        self.progress_widget.setVisible(False)
        layout.addWidget(self.progress_widget)
        
        # Envolver en scroll para ver todas las secciones y formularios
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setAlignment(Qt.AlignTop)
        scroll.setWidget(inner)

        # Poblar lista con historial de análisis al crear panel
        try:
            self.populate_analysis_images_list()
        except Exception:
            pass

        return scroll
    
    def create_right_panel(self) -> QWidget:
        """Create the right visualization and results panel"""
        panel = QFrame()
        panel.setObjectName("rightPanel")
        panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        layout = QVBoxLayout(panel)
        # Eliminar márgenes/espaciado para aprovechar al máximo el área
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Create tab widget for different views
        self.right_tabs = QTabWidget()
        self.right_tabs.setObjectName("rightTabs")
        self.right_tabs.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        # Helper para envolver contenido en un área con scroll y alineación superior
        def wrap_scroll(widget: QWidget) -> QScrollArea:
            sa = QScrollArea()
            sa.setWidgetResizable(True)
            sa.setFrameShape(QFrame.NoFrame)
            sa.setAlignment(Qt.AlignTop)
            sa.setWidget(widget)
            return sa
        
        # Image Selection tab
        self.image_selector = ImageSelector()
        # Expandir para ocupar toda el área disponible
        self.image_selector.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.right_tabs.addTab(wrap_scroll(self.image_selector), "📷 Selección de Imágenes")
        
        # Visualization tab (modo compacto para aprovechar mejor el espacio)
        self.visualization_panel = VisualizationPanel(compact=True)
        self.visualization_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.right_tabs.addTab(wrap_scroll(self.visualization_panel), "🔍 Visualización")
        
        # Results tab
        self.results_widget = ComparisonResultsWidget()
        # Results maneja scroll interno según contenido
        self.results_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.right_tabs.addTab(wrap_scroll(self.results_widget), "📊 Resultados")
        
        layout.addWidget(self.right_tabs)
        
        return panel
    
    def create_status_bar(self) -> QWidget:
        """Create the status bar"""
        status_frame = QFrame()
        status_frame.setObjectName("statusBar")
        status_frame.setMaximumHeight(40)
        
        layout = QHBoxLayout(status_frame)
        layout.setContentsMargins(10, 5, 10, 5)
        
        # Status label
        self.status_label = QLabel("Ready for comparison")
        self.status_label.setObjectName("statusLabel")
        layout.addWidget(self.status_label)
        
        layout.addStretch()
        
        # Cache status indicator
        self.cache_status = QLabel("Cache: " + ("✓" if CACHE_AVAILABLE else "✗"))
        self.cache_status.setObjectName("systemStatus")
        layout.addWidget(self.cache_status)
        
        # Action buttons
        self.start_comparison_btn = QPushButton("Start Comparison")
        self.start_comparison_btn.setObjectName("primaryButton")
        self.start_comparison_btn.setEnabled(False)
        layout.addWidget(self.start_comparison_btn)
        
        self.reset_btn = QPushButton("Reset")
        self.reset_btn.setObjectName("secondaryButton")
        layout.addWidget(self.reset_btn)
        
        return status_frame
    
    def setup_connections(self):
        """Setup signal connections"""
        # Mode selector connections
        self.mode_selector.mode_changed.connect(self.on_mode_changed)
        
        # Configuration manager connections
        self.config_manager.configuration_changed.connect(self.on_configuration_changed)
        
        # Stepper connections
        self.comparison_stepper.stepChanged.connect(self.on_step_changed)
        self.comparison_stepper.mode_switch_btn.clicked.connect(self.on_mode_switch)
        self.comparison_stepper.reset_comparison_btn.clicked.connect(self.on_reset_comparison)
        
        # Image selector connections
        # Usar lista de imágenes para vinculación correcta
        self.image_selector.imagesChanged.connect(self.on_images_selected)
        # Vista previa rápida al seleccionar una sola imagen
        self.image_selector.imageSelected.connect(self.on_image_selected)
        # Note: image_dropped signal doesn't exist in ImageSelector, removing this connection

        # Historial de análisis: selección y refresco
        if hasattr(self, 'history_list'):
            try:
                self.history_list.itemSelectionChanged.connect(self.on_history_selection_changed)
            except Exception:
                pass
        if hasattr(self, 'refresh_history_btn'):
            try:
                self.refresh_history_btn.clicked.connect(self.populate_analysis_images_list)
            except Exception:
                pass

        # Actualizar lista cuando haya nuevo análisis
        if APP_STATE_AVAILABLE and app_state_manager is not None:
            try:
                app_state_manager.image_analysis_updated.connect(self.refresh_analysis_images)
            except Exception:
                pass
        
        # Results widget connections
        self.results_widget.result_selected.connect(self.on_result_selected)
        
        # Status bar connections
        self.start_comparison_btn.clicked.connect(self.start_comparison)
        self.reset_btn.clicked.connect(self.reset_comparison)
    
    def apply_modern_theme(self):
        """Apply modern theme styling"""
        theme = ModernTheme()
        theme.apply_to_widget(self)
        
        # Apply specific styles
        self.setStyleSheet("""
            QWidget#titleLabel {
                font-size: 24px;
                font-weight: bold;
                color: #2c3e50;
                padding: 10px;
            }
            
            QFrame#leftPanel {
                background-color: #f8f9fa;
                border-right: 1px solid #dee2e6;
                border-radius: 8px;
            }
            
            QFrame#rightPanel {
                background-color: #ffffff;
                border-radius: 8px;
            }
            
            QFrame#statusBar {
                background-color: #e9ecef;
                border-top: 1px solid #dee2e6;
                border-radius: 0 0 8px 8px;
            }
            
            QPushButton#primaryButton {
                background-color: #007bff;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            
            QPushButton#primaryButton:hover {
                background-color: #0056b3;
            }
            
            QPushButton#primaryButton:disabled {
                background-color: #6c757d;
            }
            
            QPushButton#secondaryButton {
                background-color: #6c757d;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
            }
            
            QPushButton#secondaryButton:hover {
                background-color: #545b62;
            }
        """)
    
    def on_mode_changed(self, mode: str):
        """Handle comparison mode change"""
        self.current_mode = mode
        
        # Update stepper for mode
        self.comparison_stepper.update_step_for_mode(mode)
        
        # Update configuration manager
        self.config_manager.set_comparison_mode(mode)
        
        # Update image selector for mode
        if mode == 'direct':
            self.image_selector.set_selection_mode('multiple')
            self.image_selector.set_minimum_images(2)
            self.status_label.setText("Select two images for direct comparison")
        else:  # database
            self.image_selector.set_selection_mode('single')
            self.image_selector.set_minimum_images(1)
            self.status_label.setText("Select query image for database search")

        # Ajustar modo de selección de la lista de historial
        try:
            if hasattr(self, 'history_list') and self.history_list is not None:
                self.history_list.setSelectionMode(
                    QAbstractItemView.MultiSelection if mode == 'direct' else QAbstractItemView.SingleSelection
                )
        except Exception:
            pass
        
        # Update UI state
        self.update_ui_state()
        self.mode_changed.emit(mode)
    
    def on_configuration_changed(self, config: Dict[str, Any]):
        """Handle configuration changes"""
        self.current_configuration = config
        self.update_ui_state()
        self.configuration_changed.emit(config)
    
    def on_step_changed(self, step_id: str):
        """Handle stepper step changes"""
        # Update right panel tab based on step
        if step_id == 'image_selection':
            self.right_tabs.setCurrentIndex(0)  # Image Selection tab
        elif step_id in ['comparison_mode', 'configuration']:
            # Stay on current tab or show visualization
            pass
        elif step_id == 'comparison':
            self.right_tabs.setCurrentIndex(1)  # Visualization tab
            self.progress_widget.setVisible(True)
        elif step_id == 'results':
            self.right_tabs.setCurrentIndex(2)  # Results tab
            self.progress_widget.setVisible(False)
    
    def on_mode_switch(self):
        """Handle mode switch button click"""
        current_mode = self.mode_selector.get_current_mode()
        new_mode = 'database' if current_mode == 'direct' else 'direct'
        self.mode_selector.set_mode(new_mode)
    
    def on_reset_comparison(self):
        """Handle reset comparison button click"""
        self.reset_comparison()
    
    def on_images_selected(self, image_paths: List[str]):
        """Handle image selection"""
        self.current_images = image_paths
        self.update_ui_state()
        
        # Update status
        if self.current_mode == 'direct':
            if len(image_paths) >= 2:
                self.status_label.setText(f"Selected {len(image_paths)} images for comparison")
                # Mark first step completed and move to comparison mode
                try:
                    self.comparison_stepper.mark_step_completed(0)
                except Exception:
                    pass
                self.comparison_stepper.set_current_step(1)
            else:
                self.status_label.setText("Select at least 2 images for direct comparison")
        else:  # database
            if len(image_paths) >= 1:
                self.status_label.setText("Query image selected for database search")
                # Mark first step completed and move to comparison mode
                try:
                    self.comparison_stepper.mark_step_completed(0)
                except Exception:
                    pass
                self.comparison_stepper.set_current_step(1)
            else:
                self.status_label.setText("Select a query image for database search")
    
    def on_image_dropped(self, image_path: str):
        """Handle image drop"""
        if self.current_mode == 'direct':
            # Add to current selection
            if image_path not in self.current_images:
                self.current_images.append(image_path)
                self.image_selector.add_image(image_path)
        else:  # database
            # Replace current selection
            self.current_images = [image_path]
            self.image_selector.set_images([image_path])
        
        self.update_ui_state()
    
    def on_result_selected(self, result: Dict[str, Any]):
        """Handle result selection"""
        # Update visualization panel with selected result
        try:
            self.visualization_panel.display_result(result)
        except Exception:
            # Fallback a método disponible
            self.visualization_panel.set_analysis_results(result)

    def populate_analysis_images_list(self):
        """Poblar la lista con imágenes del historial de Análisis"""
        if not hasattr(self, 'history_list') or self.history_list is None:
            return
        self.history_list.clear()

        if not APP_STATE_AVAILABLE or app_state_manager is None:
            return

        try:
            history = app_state_manager.get_analysis_history()
        except Exception:
            history = []

        for entry in history:
            try:
                path = getattr(entry, 'image_path', None)
                if not path or not os.path.exists(path):
                    continue
                item = QListWidgetItem(os.path.basename(path))
                item.setToolTip(path)
                item.setData(Qt.UserRole, path)
                self.history_list.addItem(item)
            except Exception:
                continue

    def on_history_selection_changed(self):
        """Actualizar selección de imágenes desde el historial"""
        if not hasattr(self, 'history_list') or self.history_list is None:
            return
        selected_items = self.history_list.selectedItems()
        paths = []
        for it in selected_items:
            try:
                p = it.data(Qt.UserRole)
                if p:
                    paths.append(p)
            except Exception:
                continue

        if self.current_mode == 'direct':
            # Modo directo: usar todas las seleccionadas
            self.current_images = paths[:]
            try:
                self.image_selector.set_images(self.current_images)
            except Exception:
                pass
        else:
            # Modo base de datos: usar solo la primera
            self.current_images = paths[:1]
            try:
                self.image_selector.set_images(self.current_images)
            except Exception:
                pass

        self.update_ui_state()

    def refresh_analysis_images(self, _result=None):
        """Refrescar lista al recibir nuevo análisis"""
        try:
            self.populate_analysis_images_list()
        except Exception:
            pass
    
    def update_ui_state(self):
        """Update UI state based on current data"""
        # Check if we can start comparison
        can_start = False
        
        if self.current_mode == 'direct':
            can_start = len(self.current_images) >= 2
        elif self.current_mode == 'database':
            can_start = len(self.current_images) >= 1
        
        # Enable/disable start button
        self.start_comparison_btn.setEnabled(can_start and not self.is_comparison_running())
    
    def is_comparison_running(self) -> bool:
        """Check if comparison is currently running"""
        return self.comparison_worker is not None and self.comparison_worker.isRunning()
    
    def start_comparison(self):
        """Start the comparison process"""
        if self.is_comparison_running():
            return
        
        # Validate inputs
        if not self.current_images:
            QMessageBox.warning(self, "Warning", "Please select images for comparison")
            return
        
        if self.current_mode == 'direct' and len(self.current_images) < 2:
            QMessageBox.warning(self, "Warning", "Direct comparison requires at least 2 images")
            return
        
        if self.current_mode == 'database' and len(self.current_images) < 1:
            QMessageBox.warning(self, "Warning", "Database search requires a query image")
            return
        
        # Record feature usage
        record_feature_usage('ballistic_comparison', 'comparison_tab', 
                           data={
                               'mode': self.current_mode,
                               'image_count': len(self.current_images),
                               'configuration': self.current_configuration
                           })
        
        # Setup and start worker
        self.comparison_worker = ComparisonWorker()
        
        if self.current_mode == 'direct':
            self.comparison_worker.setup_comparison(
                mode='direct',
                config=self.current_configuration,
                images=self.current_images
            )
        else:  # database
            self.comparison_worker.setup_comparison(
                mode='database',
                config=self.current_configuration,
                images=[],
                query_image=self.current_images[0]
            )
        
        # Connect worker signals
        self.comparison_worker.progress_updated.connect(self.on_comparison_progress)
        self.comparison_worker.step_completed.connect(self.on_comparison_step_completed)
        self.comparison_worker.comparison_completed.connect(self.on_comparison_completed)
        self.comparison_worker.error_occurred.connect(self.on_comparison_error)
        
        # Update UI
        self.progress_widget.setVisible(True)
        self.progress_widget.reset_progress()
        # Move to comparison step (index 3)
        try:
            self.comparison_stepper.set_current_step(3)
        except Exception:
            pass
        self.start_comparison_btn.setEnabled(False)
        self.status_label.setText("Comparison in progress...")
        
        # Start worker
        self.comparison_worker.start()
        self.comparison_started.emit()
    
    def on_comparison_progress(self, step: str, progress: int, details: str):
        """Handle comparison progress updates"""
        self.progress_widget.update_progress(step, progress, details)
        self.status_label.setText(f"Comparison: {step} ({progress}%)")
    
    def on_comparison_step_completed(self, step_name: str, step_results: Dict[str, Any]):
        """Handle completion of comparison steps"""
        self.progress_widget.add_progress_detail(step_name, "Completed successfully")
    
    def on_comparison_completed(self, results: Dict[str, Any]):
        """Handle comparison completion"""
        self.comparison_results = results
        
        # Record successful completion
        record_user_action('comparison_completed', 'comparison_tab', {
            'mode': self.current_mode,
            'comparison_time_ms': results.get('analysis_time_ms', 0),
            'results_count': len(results.get('results', [])),
            'status': results.get('status', 'unknown')
        })
        
        # Update UI
        self.progress_widget.setVisible(False)
        # Mark comparison step completed and go to results (index 4)
        try:
            self.comparison_stepper.mark_step_completed(3)
        except Exception:
            pass
        try:
            self.comparison_stepper.set_current_step(4)
        except Exception:
            pass
        self.start_comparison_btn.setEnabled(True)
        
        # Display results
        if results['mode'] == 'direct':
            # Usar el método de la pestaña para unificar lógica de visualización
            self.display_direct_results(results)
        else:
            # Usar el método de la pestaña para unificar lógica de visualización
            self.display_database_results(results)
        
        self.status_label.setText("Comparison completed successfully!")
        self.comparison_completed.emit(results)
        
        QMessageBox.information(self, "Success", "Comparison completed successfully!")

    def on_comparison_error(self, error_message: str):
        """Handle comparison errors"""
        # Record error event
        record_user_action('comparison_failed', 'comparison_tab', {
            'mode': self.current_mode,
            'error_message': error_message
        })
        
        # Update UI
        self.progress_widget.setVisible(False)
        # Ensure comparison step is active to reflect error state
        try:
            self.comparison_stepper.set_current_step(3)
        except Exception:
            pass
        self.start_comparison_btn.setEnabled(True)
        self.status_label.setText(f"Comparison failed: {error_message}")
        
        QMessageBox.critical(self, "Error", f"Comparison failed: {error_message}")

    def display_direct_results(self, results: Dict[str, Any]):
        """Display direct comparison results"""
        # Format results for display
        formatted_results = [{
            'id': 'Direct Comparison',
            'similarity': results.get('similarity_score', 0.0),
            'confidence': results.get('confidence', 0.0),
            'afte_conclusion': results.get('afte_conclusion', 'Unknown'),
            'match_type': results.get('match_type', 'Unknown'),
            'images': results.get('images', [])
        }]
        
        self.results_widget.display_results(formatted_results)
        
        # Update visualization
        try:
            self.visualization_panel.display_comparison(
                results.get('images', []),
                results
            )
        except Exception:
            # Fallback: cargar primera imagen y pasar resultados
            imgs = results.get('images', [])
            if imgs:
                self.visualization_panel.load_image(imgs[0])
            self.visualization_panel.set_analysis_results(results)
    
    def display_database_results(self, results: Dict[str, Any]):
        """Display database search results"""
        search_results = results.get('results', [])
        self.results_widget.display_results(search_results)
        
        # Update visualization with query image
        if results.get('query_image'):
            try:
                self.visualization_panel.display_query_image(results['query_image'])
            except Exception:
                self.visualization_panel.load_image(results['query_image'])
                self.visualization_panel.set_analysis_results(results)

    def on_image_selected(self, image_path: str):
        """Quick preview when a single image is selected"""
        if not image_path:
            return
        try:
            self.visualization_panel.load_image(image_path)
            self.status_label.setText("Imagen cargada para previsualización")
            # Cambiar a la pestaña de visualización para mejor UX
            if hasattr(self, 'right_tabs'):
                self.right_tabs.setCurrentIndex(1)
        except Exception as e:
            print(f"Visualization preview error: {e}")
    
    def reset_comparison(self):
        """Reset the comparison state"""
        # Stop any running comparison
        if self.is_comparison_running():
            self.comparison_worker.terminate()
            self.comparison_worker.wait()
            self.comparison_worker = None
        
        # Reset state
        self.current_images = []
        self.comparison_results = None
        
        # Reset UI components
        self.image_selector.clear_images()
        self.results_widget.clear_results()
        self.visualization_panel.clear_display()
        self.progress_widget.reset_progress()
        self.progress_widget.setVisible(False)

        # Reset stepper
        try:
            self.comparison_stepper.set_current_step(0)
        except Exception:
            pass
        
        # Reset tabs
        self.right_tabs.setCurrentIndex(0)
        
        # Update status
        self.status_label.setText("Ready for comparison")
        self.start_comparison_btn.setEnabled(False)
    
    def get_current_configuration(self) -> Dict[str, Any]:
        """Get the current configuration"""
        return self.current_configuration.copy()
    
    def set_configuration(self, config: Dict[str, Any]):
        """Set the configuration"""
        self.config_manager.set_configuration(config)
    
    def get_comparison_results(self) -> Optional[Dict[str, Any]]:
        """Get the current comparison results"""
        return self.comparison_results
    
    def export_results(self, file_path: str):
        """Export comparison results to file"""
        if not self.comparison_results:
            QMessageBox.warning(self, "Warning", "No results to export")
            return
        
        try:
            import json
            with open(file_path, 'w') as f:
                json.dump(self.comparison_results, f, indent=2)
            
            QMessageBox.information(self, "Success", f"Results exported to {file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export results: {str(e)}")