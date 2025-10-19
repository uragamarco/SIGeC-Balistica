"""
Refactored Analysis Tab for SIGeC-Balisticar
Modern, hierarchical architecture with step-by-step navigation and shared widgets
"""

import os
import sys
import time
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from gui.styles import apply_modern_qss_to_widget

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QSplitter,
    QLabel, QPushButton, QProgressBar, QTextEdit, QGroupBox,
    QScrollArea, QFrame, QMessageBox, QFileDialog, QApplication,
    QSizePolicy
)
from PyQt5.QtCore import QThread, pyqtSignal, QTimer, Qt, QSize
from PyQt5.QtGui import QFont, QPixmap, QIcon

# Import shared widgets
from gui.widgets.shared import (
    StepperWidget, NISTConfigurationWidget, AFTEConfigurationWidget,
    DeepLearningConfigWidget, ImageProcessingWidget
)

# Import analysis-specific widgets
from gui.widgets.analysis import AnalysisStepper, ConfigurationLevelsManager

# Import existing components (keeping essential ones)
from gui.visualization_widgets import VisualizationPanel
from gui.dynamic_results_panel import ResultsPanel
from gui.shared_widgets import ImageSelector
from gui.nist_metadata_widget import NISTMetadataWidget

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

# Core pipeline imports
try:
    from core.unified_pipeline import ScientificPipeline, PipelineLevel, PipelineResult, create_pipeline_config
    from core.image_processing import ImageProcessor
    from core.nist_standards import NISTValidator
    from core.afte_conclusions import AFTEAnalyzer
    from core.database import DatabaseManager
    CORE_AVAILABLE = True
except ImportError:
    CORE_AVAILABLE = False
    # Mock classes for development - solo las mínimas necesarias
    class PipelineLevel:
        BASIC = "basic"
        INTERMEDIATE = "intermediate"
        ADVANCED = "advanced"
    
    class PipelineResult:
        def __init__(self, data): self.data = data

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

# Deep Learning imports
try:
    from core.deep_learning import DeepLearningAnalyzer
    DEEP_LEARNING_AVAILABLE = True
except ImportError:
    DEEP_LEARNING_AVAILABLE = False


class AnalysisWorker(QThread):
    """Background worker for ballistic analysis"""
    
    # Signals
    progress_updated = pyqtSignal(int, str)  # progress, message
    analysis_completed = pyqtSignal(dict)    # results
    error_occurred = pyqtSignal(str)         # error message
    step_completed = pyqtSignal(str, dict)   # step_name, step_results
    
    def __init__(self, images: List[str], configuration: Dict[str, Any]):
        super().__init__()
        self.images = images
        self.configuration = configuration
        self.is_cancelled = False
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
        
    def run(self):
        """Execute the analysis pipeline"""
        self.start_time = time.time()
        
        try:
            # Record analysis start
            record_user_action('analysis_started', 'analysis_tab', {
                'image_count': len(self.images),
                'level': self.configuration.get('level', 'unknown'),
                'configuration': self.configuration
            })
            
            self.progress_updated.emit(10, "Initializing analysis pipeline...")
            
            # Initialize pipeline based on configuration level
            level = self.configuration.get('level', PipelineLevel.BASIC)
            if CORE_AVAILABLE:
                pipeline_config = create_pipeline_config(level)
                pipeline = ScientificPipeline(pipeline_config)
            else:
                pipeline = ScientificPipeline(level)
            
            self.progress_updated.emit(20, "Validating images...")
            # Image validation step
            validation_start = time.time()
            valid_images = self._validate_images()
            validation_time = (time.time() - validation_start) * 1000
            
            record_performance_event('image_validation', 'analysis_worker', validation_time, 
                                   success=len(valid_images) > 0,
                                   metadata={'total_images': len(self.images), 'valid_images': len(valid_images)})
            
            if not valid_images:
                self.error_occurred.emit("No valid images found for analysis")
                return
            
            self.step_completed.emit("validation", {"valid_images": len(valid_images)})
            
            self.progress_updated.emit(40, "Processing images...")
            # Image processing step
            processing_start = time.time()
            processed_results = self._process_images(valid_images)
            processing_time = (time.time() - processing_start) * 1000
            
            record_performance_event('image_processing', 'analysis_worker', processing_time, 
                                   success=True, metadata={'processed_images': len(valid_images)})
            
            self.step_completed.emit("processing", processed_results)
            
            self.progress_updated.emit(60, "Running ballistic analysis...")
            # Main analysis step
            analysis_start = time.time()
            if CORE_AVAILABLE:
                # Use process_comparison for two images or handle multiple images
                if len(valid_images) >= 2:
                    analysis_results = pipeline.process_comparison(valid_images[0], valid_images[1])
                else:
                    analysis_results = {"status": "error", "message": "Need at least 2 images for comparison"}
            else:
                analysis_results = {"status": "mock", "results": {}}
            analysis_time = (time.time() - analysis_start) * 1000
            
            record_performance_event('ballistic_analysis', 'analysis_worker', analysis_time, 
                                   success=True, metadata={'level': level})
            
            self.step_completed.emit("analysis", analysis_results)
            
            self.progress_updated.emit(80, "Generating reports...")
            # Report generation step
            report_start = time.time()
            reports = self._generate_reports(analysis_results)
            report_time = (time.time() - report_start) * 1000
            
            record_performance_event('report_generation', 'analysis_worker', report_time, 
                                   success=True)
            
            self.step_completed.emit("reports", reports)
            
            # Complete analysis
            total_time = (time.time() - self.start_time) * 1000
            record_performance_event('full_analysis', 'analysis_worker', total_time, 
                                   success=True, metadata={
                                       'image_count': len(self.images),
                                       'level': level,
                                       'steps_completed': ['validation', 'processing', 'analysis', 'reports']
                                   })
            
            # Emit results
            results = {
                "status": "completed",
                "processed_images": len(valid_images),
                "analysis_time_ms": total_time,
                "analysis_results": analysis_results,
                "reports": reports
            }
            
            self.progress_updated.emit(100, "Analysis completed!")
            self.analysis_completed.emit(results)
            
        except Exception as e:
            # Record error
            record_error_event(e, 'analysis_worker', 'run_analysis')
            self.error_occurred.emit(str(e))

    def _validate_images(self) -> List[str]:
        """Validate input images"""
        valid_images = []
        for image_path in self.images:
            if os.path.exists(image_path) and self._is_valid_image(image_path):
                valid_images.append(image_path)
        return valid_images
    
    def _is_valid_image(self, image_path: str) -> bool:
        """Check if image is valid for ballistic analysis"""
        try:
            # Basic validation - can be extended
            valid_extensions = ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']
            return any(image_path.lower().endswith(ext) for ext in valid_extensions)
        except:
            return False
    
    def _process_images(self, images: List[str]) -> Dict[str, Any]:
        """Process images according to configuration with intelligent caching"""
        processing_config = self.configuration.get('image_processing', {})
        processed_results = []
        cache_hits = 0
        cache_misses = 0
        
        for image_path in images:
            try:
                # Generate cache key based on image path and processing config
                cache_key = f"image_processing_{hash(image_path)}_{hash(str(processing_config))}"
                
                # Try to get from cache first
                cached_result = None
                if self.cache:
                    try:
                        cached_result = self.cache.get(cache_key)
                        if cached_result:
                            cache_hits += 1
                            processed_results.append(cached_result)
                            continue
                    except Exception as e:
                        print(f"Cache retrieval error: {e}")
                
                cache_misses += 1
                
                # Process image if not in cache
                # This would use the actual image processing pipeline
                result = {
                    "image_path": image_path,
                    "processing_config": processing_config,
                    "processed": True,
                    "timestamp": time.time()
                }
                
                # Store in cache for future use
                if self.cache:
                    try:
                        # Cache with 1 hour TTL for processed images
                        self.cache.set(cache_key, result, ttl=3600)
                    except Exception as e:
                        print(f"Cache storage error: {e}")
                
                processed_results.append(result)
                
            except Exception as e:
                print(f"Error processing image {image_path}: {e}")
                continue
        
        return {
            "processed_count": len(processed_results),
            "processing_config": processing_config,
            "cache_stats": {
                "hits": cache_hits,
                "misses": cache_misses,
                "hit_rate": cache_hits / (cache_hits + cache_misses) if (cache_hits + cache_misses) > 0 else 0
            },
            "results": processed_results
        }
    
    def _generate_reports(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate analysis reports"""
        return {
            "nist_report": self.configuration.get('nist', {}).get('generate_report', False),
            "afte_report": self.configuration.get('afte', {}).get('generate_report', False),
            "technical_report": True
        }
    
    def cancel(self):
        """Cancel the analysis"""
        self.is_cancelled = True
        self.quit()


class AnalysisTab(QWidget):
    """
    Modern Analysis Tab with hierarchical configuration and step-by-step navigation
    """
    
    # Signals
    analysis_started = pyqtSignal()
    analysis_completed = pyqtSignal(dict)
    configuration_changed = pyqtSignal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # State management
        self.current_images: List[str] = []
        self.current_configuration: Dict[str, Any] = {}
        self.analysis_worker: Optional[AnalysisWorker] = None
        self.analysis_results: Optional[Dict[str, Any]] = None
        
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
        title_label = QLabel("Análisis Balístico")
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

        # Improve stretch behavior to maximize vertical/overall space usage
        main_splitter.setStretchFactor(0, 1)
        main_splitter.setStretchFactor(1, 2)
        
        # Set splitter proportions
        main_splitter.setSizes([400, 600])
        main_layout.addWidget(main_splitter)
        
        # Status bar
        self.status_bar = self.create_status_bar()
        main_layout.addWidget(self.status_bar)
        
        # Asegurar que el splitter ocupe la mayor parte del espacio vertical
        # y evitar espacios vacíos arriba/abajo
        try:
            main_layout.setStretch(0, 0)  # título
            main_layout.setStretch(1, 1)  # splitter
            main_layout.setStretch(2, 0)  # status bar
        except Exception:
            pass
        
    def create_left_panel(self) -> QWidget:
        """Create the left navigation and configuration panel"""
        panel = QWidget()
        panel.setObjectName("leftPanel")
        panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(15)
        
        # Analysis Stepper (mantenido para lógica interna, oculto para ahorrar espacio)
        self.analysis_stepper = AnalysisStepper()
        self.analysis_stepper.setObjectName("analysisStepper")
        self.analysis_stepper.setVisible(False)
        
        # Tabbed interface for analysis steps on the left panel
        from PyQt5.QtWidgets import QTabWidget
        self.step_tabs = QTabWidget()
        self.step_tabs.setObjectName("stepTabs")
        self.step_tabs.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # --- Step 1: Image Selection + Case Metadata ---
        step1 = QWidget()
        step1_layout = QVBoxLayout(step1)
        step1_layout.setContentsMargins(5, 5, 5, 5)
        step1_layout.setSpacing(10)

        intro_label = QLabel("Selección de imágenes y metadatos del caso")
        intro_label.setObjectName("panelSubtitle")
        step1_layout.addWidget(intro_label)

        # Botón para seleccionar imágenes desde el panel izquierdo (multi-selección)
        self.select_images_btn = QPushButton("Seleccionar imágenes…")
        self.select_images_btn.setObjectName("primaryButton")
        step1_layout.addWidget(self.select_images_btn)

        # Formulario de metadatos NIST en el panel izquierdo
        try:
            shared_state = None
            try:
                if hasattr(self.parent(), 'state_manager'):
                    shared_state = getattr(self.parent(), 'state_manager')
            except Exception:
                shared_state = None

            self.nist_metadata_widget = NISTMetadataWidget(state_manager=shared_state)
            step1_layout.addWidget(self.nist_metadata_widget)
        except Exception as e:
            print(f"Warning: NISTMetadataWidget not available in step tabs: {e}")
            self.nist_metadata_widget = None

        # Hacer scrollable el contenido del paso 1
        step1_scroll = QScrollArea()
        step1_scroll.setWidget(step1)
        step1_scroll.setWidgetResizable(True)
        step1_scroll.setFrameShape(QFrame.NoFrame)
        step1_scroll.setAlignment(Qt.AlignTop)
        self.step_tabs.addTab(step1_scroll, "1) Imágenes y Caso")

        # --- Step 2: Configuration ---
        step2 = QWidget()
        step2_layout = QVBoxLayout(step2)
        step2_layout.setContentsMargins(5, 5, 5, 5)
        step2_layout.setSpacing(10)

        config_label = QLabel("Configuración del análisis")
        config_label.setObjectName("panelSubtitle")
        step2_layout.addWidget(config_label)

        # Configuration Levels Manager dentro de la pestaña de configuración
        self.config_manager = ConfigurationLevelsManager()
        self.config_manager.setObjectName("configManager")
        step2_layout.addWidget(self.config_manager)

        # Hacer scrollable el contenido del paso 2
        step2_scroll = QScrollArea()
        step2_scroll.setWidget(step2)
        step2_scroll.setWidgetResizable(True)
        step2_scroll.setFrameShape(QFrame.NoFrame)
        step2_scroll.setAlignment(Qt.AlignTop)
        self.step_tabs.addTab(step2_scroll, "2) Configuración")

        # --- Step 3: Execute Analysis with review sections (collapsible) ---
        step3 = QWidget()
        step3_layout = QVBoxLayout(step3)
        step3_layout.setContentsMargins(5, 5, 5, 5)
        step3_layout.setSpacing(10)

        review_label = QLabel("Revisión de datos y configuración")
        review_label.setObjectName("panelSubtitle")
        step3_layout.addWidget(review_label)

        # Secciones de revisión
        self.review_case_group = QGroupBox("Datos del caso y metadatos")
        self.review_case_group.setCheckable(True)
        self.review_case_group.setChecked(True)
        review_case_layout = QVBoxLayout(self.review_case_group)
        self.review_case_text = QTextEdit()
        self.review_case_text.setReadOnly(True)
        review_case_layout.addWidget(self.review_case_text)
        step3_layout.addWidget(self.review_case_group)

        self.review_config_group = QGroupBox("Configuración del análisis")
        self.review_config_group.setCheckable(True)
        self.review_config_group.setChecked(True)
        review_config_layout = QVBoxLayout(self.review_config_group)
        self.review_config_text = QTextEdit()
        self.review_config_text.setReadOnly(True)
        review_config_layout.addWidget(self.review_config_text)
        step3_layout.addWidget(self.review_config_group)

        # Progreso del análisis (movido desde panel derecho)
        progress_group = QGroupBox("Estado del proceso de análisis")
        progress_group.setObjectName("progressGroup")
        progress_layout = QVBoxLayout(progress_group)
        self.progress_bar = QProgressBar()
        self.progress_bar.setObjectName("analysisProgressBar")
        self.progress_bar.setVisible(False)
        progress_layout.addWidget(self.progress_bar)
        self.progress_label = QLabel("Listo para iniciar el análisis")
        self.progress_label.setObjectName("progressLabel")
        progress_layout.addWidget(self.progress_label)
        step3_layout.addWidget(progress_group)

        # Botones de acción para ejecutar análisis
        run_layout = QHBoxLayout()
        self.start_analysis_btn = QPushButton("Iniciar análisis")
        self.start_analysis_btn.setObjectName("primaryButton")
        self.start_analysis_btn.setEnabled(False)
        run_layout.addWidget(self.start_analysis_btn)
        run_layout.addStretch()
        step3_layout.addLayout(run_layout)

        # Hacer scrollable el contenido del paso 3
        step3_scroll = QScrollArea()
        step3_scroll.setWidget(step3)
        step3_scroll.setWidgetResizable(True)
        step3_scroll.setFrameShape(QFrame.NoFrame)
        step3_scroll.setAlignment(Qt.AlignTop)
        self.step_tabs.addTab(step3_scroll, "3) Ejecutar")

        # --- Step 4: Results summary ---
        step4 = QWidget()
        step4_layout = QVBoxLayout(step4)
        step4_layout.setContentsMargins(5, 5, 5, 5)
        step4_layout.setSpacing(10)

        results_label = QLabel("Resultados del análisis")
        results_label.setObjectName("panelSubtitle")
        step4_layout.addWidget(results_label)

        self.left_results_text = QTextEdit()
        self.left_results_text.setReadOnly(True)
        step4_layout.addWidget(self.left_results_text)

        # Botón para enfocar la pestaña de resultados del panel derecho
        focus_results_btn = QPushButton("Ver resultados en panel derecho")
        focus_results_btn.setObjectName("secondaryButton")
        focus_results_btn.clicked.connect(lambda: self.tab_widget.setCurrentIndex(2))
        step4_layout.addWidget(focus_results_btn)

        # Hacer scrollable el contenido del paso 4
        step4_scroll = QScrollArea()
        step4_scroll.setWidget(step4)
        step4_scroll.setWidgetResizable(True)
        step4_scroll.setFrameShape(QFrame.NoFrame)
        step4_scroll.setAlignment(Qt.AlignTop)
        self.step_tabs.addTab(step4_scroll, "4) Resultados")

        # Añadir tabs de pasos al panel izquierdo (con stretch para ocupar altura)
        layout.addWidget(self.step_tabs, 1)

        # Botón de reset siempre visible
        self.reset_btn = QPushButton("Reiniciar")
        self.reset_btn.setObjectName("secondaryButton")
        layout.addWidget(self.reset_btn)

        # Nota: Eliminamos el stretch inferior para que las pestañas ocupen toda la altura disponible
        
        return panel
    
    def create_right_panel(self) -> QWidget:
        """Create the right visualization and results panel"""
        panel = QWidget()
        panel.setObjectName("rightPanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(15)
        
        # Create tabbed interface for different views
        from PyQt5.QtWidgets import QTabWidget, QScrollArea, QFrame
        
        self.tab_widget = QTabWidget()
        self.tab_widget.setObjectName("analysisTabWidget")

        # Helper para envolver contenido en un área con scroll
        def wrap_scroll(widget: QWidget) -> QScrollArea:
            sa = QScrollArea()
            sa.setWidgetResizable(True)
            sa.setFrameShape(QFrame.NoFrame)
            sa.setAlignment(Qt.AlignTop)
            sa.setWidget(widget)
            return sa
        
        # Pestaña Imágenes
        self.image_selector = ImageSelector()
        self.tab_widget.addTab(wrap_scroll(self.image_selector), "Imágenes")
        
        # Pestaña Visualización
        self.visualization_panel = VisualizationPanel()
        self.tab_widget.addTab(wrap_scroll(self.visualization_panel), "Visualización")
        
        # Pestaña Resultados
        self.results_panel = ResultsPanel()
        # Aunque ResultsPanel ya incluye scroll interno para tarjetas,
        # lo envolvemos para permitir scroll de nivel de pestaña.
        self.tab_widget.addTab(wrap_scroll(self.results_panel), "Resultados")
        
        layout.addWidget(self.tab_widget)
        # Nota: el progreso se gestiona ahora en el paso 3 del panel izquierdo
        
        return panel
    
    def create_status_bar(self) -> QWidget:
        """Create the status bar"""
        status_frame = QFrame()
        status_frame.setObjectName("statusFrame")
        status_frame.setFrameStyle(QFrame.StyledPanel)
        
        layout = QHBoxLayout(status_frame)
        layout.setContentsMargins(10, 5, 10, 5)
        
        self.status_label = QLabel("Listo")
        self.status_label.setObjectName("statusLabel")
        layout.addWidget(self.status_label)
        
        layout.addStretch()
        
        # System status indicators
        self.core_status = QLabel("Núcleo: " + ("✓" if CORE_AVAILABLE else "✗"))
        self.core_status.setObjectName("systemStatus")
        layout.addWidget(self.core_status)
        
        self.dl_status = QLabel("DL: " + ("✓" if DEEP_LEARNING_AVAILABLE else "✗"))
        self.dl_status.setObjectName("systemStatus")
        layout.addWidget(self.dl_status)
        
        # Cache status indicator
        self.cache_status = QLabel("Caché: " + ("✓" if CACHE_AVAILABLE else "✗"))
        self.cache_status.setObjectName("systemStatus")
        layout.addWidget(self.cache_status)
        
        return status_frame
    
    def setup_connections(self):
        """Setup signal-slot connections"""
        # Stepper connections
        self.analysis_stepper.stepChanged.connect(self.on_step_changed)
        self.analysis_stepper.nextRequested.connect(self.on_next_step)
        self.analysis_stepper.previousRequested.connect(self.on_previous_step)
        
        # Configuration manager connections
        self.config_manager.configurationChanged.connect(self.on_configuration_changed)
        self.config_manager.levelChanged.connect(self.on_level_changed)
        
        # Image selector connections
        self.image_selector.imagesChanged.connect(self.on_images_changed)

        # Left-panel image selection button -> open file dialog and set in right panel
        self.select_images_btn.clicked.connect(self.on_select_images_left)
        
        # Button connections
        self.start_analysis_btn.clicked.connect(self.start_analysis)
        self.reset_btn.clicked.connect(self.reset_analysis)

        # Metadata widget signals affect analysis readiness
        if getattr(self, 'nist_metadata_widget', None):
            try:
                self.nist_metadata_widget.validationChanged.connect(self.update_analysis_button_state)
                self.nist_metadata_widget.metadataChanged.connect(self.update_analysis_button_state)
            except Exception as e:
                print(f"Warning wiring metadata signals: {e}")
        
    def apply_modern_theme(self):
        """Apply the modern theme to the widget"""
        try:
            # Apply sanitized modern theme stylesheet using safe loader
            applied = apply_modern_qss_to_widget(self)
            if not applied:
                # Fallback: no-op if stylesheet not available
                pass
        except Exception as e:
            print(f"Warning: Could not load modern theme safely: {e}")
    
    def on_step_changed(self, step_index: int):
        """Handle step change in the stepper"""
        step_names = ["Image Selection", "Configuration", "Analysis", "Results"]
        if 0 <= step_index < len(step_names):
            self.status_label.setText(f"Current step: {step_names[step_index]}")
            
            # Cambiar pestañas en el panel izquierdo (pasos)
            try:
                self.step_tabs.setCurrentIndex(step_index)
            except Exception:
                pass

            # Switch to appropriate tab on the right panel
            if step_index == 0:  # Image Selection
                self.tab_widget.setCurrentIndex(0)  # Images
            elif step_index == 1:  # Configuration
                # Mantener la pestaña de imágenes activa
                self.tab_widget.setCurrentIndex(0)
                # Actualizar revisión con la última configuración
                self.update_review_sections()
            elif step_index == 2:  # Analysis
                self.tab_widget.setCurrentIndex(1)  # Visualization
                self.update_review_sections()
            elif step_index == 3:  # Results
                self.tab_widget.setCurrentIndex(2)  # Results
    
    def on_next_step(self):
        """Handle next step request"""
        current_step = self.analysis_stepper.current_step
        
        if current_step == 0:  # Image Selection -> Configuration
            if not self.current_images:
                QMessageBox.warning(self, "Warning", "Please select images before proceeding.")
                return
            self.analysis_stepper.set_step_completed(0, True)
            self.analysis_stepper.set_current_step(1)
            
        elif current_step == 1:  # Configuration -> Analysis
            if not self.validate_configuration():
                QMessageBox.warning(self, "Warning", "Please complete the configuration.")
                return
            self.analysis_stepper.set_step_completed(1, True)
            self.analysis_stepper.set_current_step(2)
            
        elif current_step == 2:  # Analysis -> Results
            if not self.analysis_results:
                QMessageBox.warning(self, "Warning", "Please run the analysis first.")
                return
            self.analysis_stepper.set_step_completed(2, True)
            self.analysis_stepper.set_current_step(3)
    
    def on_previous_step(self):
        """Handle previous step request"""
        current_step = self.analysis_stepper.current_step
        if current_step > 0:
            self.analysis_stepper.set_current_step(current_step - 1)
    
    def on_configuration_changed(self, config: Dict[str, Any]):
        """Handle configuration changes"""
        self.current_configuration = config
        self.configuration_changed.emit(config)
        
        # Update analysis button state
        self.update_analysis_button_state()
    
    def on_level_changed(self, level: str):
        """Handle analysis level change"""
        # Prevent re-entrant recursion from levelChanged -> set_level -> levelChanged
        if getattr(self, '_handling_level_change', False):
            return
        self._handling_level_change = True
        try:
            # Record level change (telemetry guarded)
            try:
                record_user_action('level_changed', 'analysis_tab', data={
                    'new_level': level,
                    'previous_level': self.current_configuration.get('level', 'unknown')
                })
            except Exception as e:
                print(f"Telemetry record_user_action failed: {e}")
            
            # Update local configuration state
            self.current_configuration['level'] = level
            self.configuration_changed.emit(self.current_configuration)
            self.update_analysis_button_state()

            # Update stepper UI if available
            if hasattr(self, 'analysis_stepper') and hasattr(self.analysis_stepper, 'set_configuration_level'):
                try:
                    self.analysis_stepper.set_configuration_level(level)
                except Exception as e:
                    print(f"Stepper level update failed: {e}")
            
            # Update configuration panel based on level (ensure correct attribute, avoid signal loops)
            if hasattr(self, 'config_manager'):
                try:
                    current = None
                    if hasattr(self.config_manager, 'get_current_level'):
                        current = self.config_manager.get_current_level()
                    if current != level:
                        try:
                            self.config_manager.blockSignals(True)
                        except Exception:
                            pass
                        self.config_manager.set_level(level)
                except Exception as e:
                    print(f"Config manager level update failed: {e}")
                finally:
                    try:
                        self.config_manager.blockSignals(False)
                    except Exception:
                        pass
        finally:
            self._handling_level_change = False

    def on_images_changed(self, images: List[str]):
        """Handle image selection change"""
        # Record image selection
        record_user_action('images_selected', 'analysis_tab', 
                         data={
                             'image_count': len(images),
                             'previous_count': len(self.current_images) if self.current_images else 0
                         })
        
        self.current_images = images
        self.update_analysis_button_state()

        # Prefill NIST metadata with basic case info from image name
        try:
            if getattr(self, 'nist_metadata_widget', None) and images:
                first_image = images[0]
                img_name = os.path.basename(first_image)
                case_number_guess = os.path.splitext(img_name)[0]
                prefill = {
                    'study_name': 'Análisis Balístico',
                    'first_name': os.getenv('USER', '') or 'Perito',
                    'last_name': '',
                    'organization': 'Laboratorio Forense',
                    'case_number': case_number_guess,
                    'evidence_id': img_name,
                    'laboratory': 'SIGeC',
                    'firearm_type': '',
                }
                self.nist_metadata_widget.set_metadata(prefill)
                # Sincronizar con el gestor de estado si existe
                if getattr(self.nist_metadata_widget, 'state_manager', None):
                    try:
                        self.nist_metadata_widget.state_manager.update_metadata(self.nist_metadata_widget.get_metadata())
                    except Exception as se:
                        print(f"State sync after prefill failed: {se}")
        except Exception as e:
            print(f"Metadata prefill error: {e}")
        
        # Update image display using optional set_images, with safe checks
        if hasattr(self, 'image_selector'):
            try:
                if hasattr(self.image_selector, 'set_images'):
                    self.image_selector.set_images(images)
                else:
                    # Fallback: if set_image exists, show first image only
                    if images:
                        if hasattr(self.image_selector, 'set_image'):
                            self.image_selector.set_image(images[0])
            except Exception as e:
                # Avoid crashing the UI due to optional API differences
                print(f"Image update error: {e}")

        # Actualizar secciones de revisión si estamos en paso 3
        try:
            self.update_review_sections()
        except Exception:
            pass

    def validate_configuration(self) -> bool:
        """Validate current configuration"""
        if not self.current_configuration:
            return False
        
        # Basic validation - can be extended
        required_sections = ['level']
        cfg_ok = all(section in self.current_configuration for section in required_sections)

        # Minimal NIST metadata required if widget is present
        meta_ok = True
        if getattr(self, 'nist_metadata_widget', None):
            try:
                meta = self.nist_metadata_widget.get_metadata()
                required_meta = ['study_name', 'first_name', 'last_name', 'organization', 'case_number', 'evidence_id']
                meta_ok = all(str(meta.get(k, '')).strip() for k in required_meta)
            except Exception as e:
                print(f"Metadata validation error: {e}")
                meta_ok = False

        return cfg_ok and meta_ok
    
    def update_analysis_button_state(self):
        """Update the state of the analysis button"""
        can_analyze = (
            len(self.current_images) > 0 and
            self.validate_configuration() and
            self.analysis_worker is None
        )
        self.start_analysis_btn.setEnabled(can_analyze)

    def on_select_images_left(self):
        """Open multi-file dialog and set images in the right panel selector"""
        try:
            files, _ = QFileDialog.getOpenFileNames(
                self,
                "Seleccionar imágenes balísticas",
                "",
                "Imágenes (*.png *.jpg *.jpeg *.bmp *.tiff *.tif);;Todos los archivos (*)"
            )
            if files:
                # Usar el selector de la derecha para gestionar y visualizar
                if hasattr(self, 'image_selector') and hasattr(self.image_selector, 'set_images'):
                    self.image_selector.set_images(files)
        except Exception as e:
            print(f"Error al seleccionar imágenes: {e}")

    def update_review_sections(self):
        """Update the review text sections with current metadata and configuration"""
        # Case/metadata summary
        case_lines = []
        try:
            if getattr(self, 'nist_metadata_widget', None):
                meta = self.nist_metadata_widget.get_metadata()
                # Mostrar un subconjunto clave
                keys = ['study_name', 'first_name', 'last_name', 'organization', 'case_number', 'evidence_id']
                for k in keys:
                    case_lines.append(f"{k}: {meta.get(k, '')}")
            else:
                case_lines.append("Metadatos NIST no disponibles")
        except Exception as e:
            case_lines.append(f"Error leyendo metadatos: {e}")
        case_lines.append(f"Imágenes seleccionadas: {len(self.current_images)}")
        self.review_case_text.setPlainText("\n".join(case_lines))

        # Configuration summary
        cfg_lines = []
        try:
            level = self.current_configuration.get('level', 'no establecido')
            cfg_lines.append(f"Nivel: {level}")
            # Mostrar claves principales si existen
            for k, v in self.current_configuration.items():
                if k != 'nist_metadata':
                    cfg_lines.append(f"{k}: {v}")
        except Exception as e:
            cfg_lines.append(f"Error leyendo configuración: {e}")
        self.review_config_text.setPlainText("\n".join(cfg_lines))
    
    def start_analysis(self):
        """Start the ballistic analysis"""
        if not self.current_images or not self.validate_configuration():
            QMessageBox.warning(self, "Advertencia", "Por favor seleccione imágenes y complete la configuración.")
            return
        
        # Record feature usage
        record_feature_usage('ballistic_analysis', 'analysis_tab', {
            'level': self.current_configuration.get('level', 'unknown'),
            'image_count': len(self.current_images)
        })
        
        # Create and start analysis worker
        self.analysis_worker = AnalysisWorker(self.current_images, self.current_configuration)
        
        # Connect worker signals
        self.analysis_worker.progress_updated.connect(self.on_analysis_progress)
        self.analysis_worker.analysis_completed.connect(self.on_analysis_completed)
        self.analysis_worker.error_occurred.connect(self.on_analysis_error)
        self.analysis_worker.step_completed.connect(self.on_analysis_step_completed)
        
        # Update UI
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.start_analysis_btn.setEnabled(False)
        self.start_analysis_btn.setText("Analizando...")
        
        # Start analysis
        self.analysis_worker.start()
        self.analysis_started.emit()
        
        # Move to analysis step
        self.analysis_stepper.set_current_step(2)
    
    def on_analysis_progress(self, progress: int, message: str):
        """Handle analysis progress updates"""
        self.progress_bar.setValue(progress)
        self.progress_label.setText(message)
        self.status_label.setText(message)
    
    def on_analysis_completed(self, results: Dict[str, Any]):
        """Handle analysis completion"""
        self.analysis_results = results
        self.analysis_worker = None
        
        # Record successful completion
        record_user_action('analysis_completed', 'analysis_tab', {
            'analysis_time_ms': results.get('analysis_time_ms', 0),
            'processed_images': results.get('processed_images', 0),
            'status': results.get('status', 'unknown')
        })
        
        # Update UI
        self.progress_bar.setVisible(False)
        self.start_analysis_btn.setEnabled(True)
        self.start_analysis_btn.setText("Start Analysis")
        self.progress_label.setText("¡Análisis completado con éxito!")
        
        # Update results panel with safe hasattr check
        try:
            if hasattr(self.results_panel, 'display_results'):
                self.results_panel.display_results(results)
            elif hasattr(self.results_panel, 'set_sample_results'):
                # Fallback preview if display_results not available
                self.results_panel.set_sample_results()
        except Exception as e:
            print(f"Results display error: {e}")
        
        # Move to results step
        self.analysis_stepper.set_step_completed(2, True)
        self.analysis_stepper.set_current_step(3)
        
        # Emit completion signal
        self.analysis_completed.emit(results)
        
        QMessageBox.information(self, "Éxito", "¡Análisis completado con éxito!")

        # Update left results summary
        try:
            summary_lines = []
            summary_lines.append(f"Estado: {results.get('status', 'desconocido')}")
            summary_lines.append(f"Tiempo (ms): {results.get('analysis_time_ms', 0)}")
            summary_lines.append(f"Imágenes procesadas: {results.get('processed_images', 0)}")
            summaries = results.get('summaries', {})
            for k, v in summaries.items():
                summary_lines.append(f"{k}: {v}")
            self.left_results_text.setPlainText("\n".join(summary_lines))
        except Exception as e:
            print(f"Left results update error: {e}")
    
    def on_analysis_error(self, error_message: str):
        """Handle analysis errors"""
        self.analysis_worker = None
        
        # Update UI
        self.progress_bar.setVisible(False)
        self.start_analysis_btn.setEnabled(True)
        self.start_analysis_btn.setText("Iniciar análisis")
        self.progress_label.setText("Análisis fallido")
        self.status_label.setText("Ocurrió un error")
        
        QMessageBox.critical(self, "Error de Análisis", f"El análisis falló:\n{error_message}")
    
    def on_analysis_step_completed(self, step_name: str, step_results: Dict[str, Any]):
        """Handle individual analysis step completion"""
        print(f"Step completed: {step_name} - {step_results}")
        # This can be used for more detailed progress tracking
    
    def reset_analysis(self):
        """Reset the analysis to initial state"""
        # Cancel running analysis if any
        if self.analysis_worker:
            self.analysis_worker.cancel()
            self.analysis_worker = None
        
        # Reset state
        self.analysis_results = None
        self.current_configuration = {}
        # Keep selected images; user can reselect as needed
        
        # Reset UI
        self.progress_bar.setVisible(False)
        self.progress_bar.setValue(0)
        self.progress_label.setText("Listo para iniciar el análisis")
        self.status_label.setText("Ready")
        self.start_analysis_btn.setEnabled(False)
        self.start_analysis_btn.setText("Iniciar análisis")
        
        # Reset stepper
        self.analysis_stepper.reset()
        try:
            self.step_tabs.setCurrentIndex(0)
        except Exception:
            pass
        
        # Reset configuration manager
        self.config_manager.reset_to_defaults()
        
        # Clear results
        self.results_panel.clear_results()
        try:
            self.left_results_text.clear()
        except Exception:
            pass

        # Clear NIST metadata if available
        try:
            if getattr(self, 'nist_metadata_widget', None):
                self.nist_metadata_widget.clear_metadata()
        except Exception as e:
            print(f"Metadata reset error: {e}")
        
        QMessageBox.information(self, "Reset", "Analysis has been reset to initial state.")
    
    def get_current_configuration(self) -> Dict[str, Any]:
        """Get the current configuration and include NIST metadata snapshot if available"""
        config = self.current_configuration.copy()
        try:
            if getattr(self, 'nist_metadata_widget', None):
                config['nist_metadata'] = self.nist_metadata_widget.get_metadata()
        except Exception as e:
            print(f"NIST metadata attach error: {e}")
        return config
    
    def get_analysis_results(self) -> Optional[Dict[str, Any]]:
        """Get the current analysis results"""
        return self.analysis_results.copy() if self.analysis_results else None
    
    def export_results(self, file_path: str) -> bool:
        """Export analysis results to file"""
        if not self.analysis_results:
            return False
        
        export_start = time.time()
        
        try:
            # Record export attempt
            record_user_action('results_export_started', 'analysis_tab', {
                'file_path': file_path,
                'has_results': bool(self.analysis_results)
            })
            
            # Perform export (mock implementation)
            import json
            with open(file_path, 'w') as f:
                json.dump(self.analysis_results, f, indent=2, default=str)
            
            export_time = (time.time() - export_start) * 1000
            
            # Record successful export
            record_performance_event('results_export', 'analysis_tab', export_time, 
                                   success=True, metadata={'file_path': file_path})
            
            return True
            
        except Exception as e:
            export_time = (time.time() - export_start) * 1000
            
            # Record failed export
            record_performance_event('results_export', 'analysis_tab', export_time, 
                                   success=False)
            record_error_event(e, 'analysis_tab', 'export_results')
            
            return False