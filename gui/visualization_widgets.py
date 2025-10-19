"""
Widgets Base para Visualizaciones
Sistema de widgets reutilizables para integrar visualizadores de procesamiento de imágenes
"""

import sys
import os
from typing import Optional, Dict, Any, List
from pathlib import Path
import tempfile
import logging
import cv2

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QFrame, QSplitter, QScrollArea, QGroupBox, QTabWidget,
    QProgressBar, QTextEdit, QComboBox, QCheckBox, QSpinBox,
    QSlider, QSizePolicy, QApplication
)
from PyQt5.QtCore import Qt, pyqtSignal, QThread, pyqtSlot, QTimer, QUrl
from PyQt5.QtGui import QPixmap, QFont, QPalette

# Importaciones condicionales para diferentes backends de visualización
try:
    from PyQt5.QtWebEngineWidgets import QWebEngineView
    WEBENGINE_AVAILABLE = True
except ImportError:
    WEBENGINE_AVAILABLE = False
    print("QWebEngineView no disponible - visualizaciones interactivas limitadas")

# Permitir desactivar WebEngine por entorno
if os.environ.get('SIGEC_DISABLE_WEBENGINE') == '1':
    WEBENGINE_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use('Qt5Agg')
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Matplotlib no disponible - visualizaciones estáticas limitadas")

# Importar visualizadores
try:
    from image_processing.statistical_visualizer import StatisticalVisualizer
    from image_processing.preprocessing_visualizer import PreprocessingVisualizer
    from image_processing.feature_visualizer import FeatureVisualizer
    from image_processing.roi_visualizer import ROIVisualizer
    VISUALIZERS_AVAILABLE = True
except ImportError as e:
    VISUALIZERS_AVAILABLE = False
    print(f"Error importando visualizadores: {e}")

logger = logging.getLogger(__name__)

# Visor sincronizado para comparación
try:
    from gui.synchronized_viewer import SynchronizedViewer
    SYNC_VIEWER_AVAILABLE = True
except ImportError:
    SYNC_VIEWER_AVAILABLE = False
    logger.warning("SynchronizedViewer no disponible - la pestaña de comparación será limitada")


class VisualizationWidget(QWidget):
    """Widget base para todas las visualizaciones"""
    
    # Señales
    visualizationReady = pyqtSignal(str)  # Ruta del archivo generado
    visualizationError = pyqtSignal(str)  # Mensaje de error
    progressUpdated = pyqtSignal(int)     # Progreso (0-100)
    
    def __init__(self, title: str = "Visualización", parent=None):
        super().__init__(parent)
        self.title = title
        self.temp_dir = Path(tempfile.mkdtemp(prefix="sigec_viz_"))
        self.setup_ui()
        
    def setup_ui(self):
        """Configura la interfaz base"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # Header con título y controles
        header_frame = QFrame()
        header_frame.setFrameStyle(QFrame.StyledPanel)
        header_layout = QHBoxLayout(header_frame)
        
        # Título
        title_label = QLabel(self.title)
        title_label.setFont(QFont("Arial", 12, QFont.Bold))
        header_layout.addWidget(title_label)
        
        header_layout.addStretch()
        
        # Botones de control
        self.refresh_btn = QPushButton("🔄 Actualizar")
        self.export_btn = QPushButton("💾 Exportar")
        self.settings_btn = QPushButton("⚙️ Configurar")
        
        header_layout.addWidget(self.refresh_btn)
        header_layout.addWidget(self.export_btn)
        header_layout.addWidget(self.settings_btn)
        
        layout.addWidget(header_frame)
        
        # Barra de progreso
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Área de contenido (será sobrescrita por subclases)
        self.content_area = QWidget()
        layout.addWidget(self.content_area, 1)
        
        # Conectar señales
        self.refresh_btn.clicked.connect(self.refresh_visualization)
        self.export_btn.clicked.connect(self.export_visualization)
        self.settings_btn.clicked.connect(self.show_settings)
        self.progressUpdated.connect(self.progress_bar.setValue)
        
    def show_progress(self, show: bool = True):
        """Muestra u oculta la barra de progreso"""
        self.progress_bar.setVisible(show)
        
    def set_progress(self, value: int):
        """Actualiza el progreso"""
        self.progressUpdated.emit(value)
        
    def refresh_visualization(self):
        """Actualiza la visualización - implementar en subclases"""
        pass
        
    def export_visualization(self):
        """Exporta la visualización - implementar en subclases"""
        pass
        
    def show_settings(self):
        """Muestra configuración - implementar en subclases"""
        pass
        
    def cleanup(self):
        """Limpia archivos temporales"""
        try:
            import shutil
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
        except Exception as e:
            logger.warning(f"Error limpiando archivos temporales: {e}")


class MatplotlibWidget(VisualizationWidget):
    """Widget para visualizaciones con Matplotlib"""
    
    def __init__(self, title: str = "Gráfico Matplotlib", parent=None):
        self.figure = None
        self.canvas = None
        super().__init__(title, parent)
        
    def setup_ui(self):
        """Configura la interfaz con canvas de matplotlib"""
        super().setup_ui()
        
        if not MATPLOTLIB_AVAILABLE:
            error_label = QLabel("Matplotlib no está disponible")
            error_label.setAlignment(Qt.AlignCenter)
            error_label.setStyleSheet("color: red; font-size: 14px;")
            
            content_layout = QVBoxLayout(self.content_area)
            content_layout.addWidget(error_label)
            return
            
        # Crear figura y canvas
        self.figure = Figure(figsize=(12, 8), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        # Layout del área de contenido
        content_layout = QVBoxLayout(self.content_area)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.addWidget(self.canvas)
        
    def clear_figure(self):
        """Limpia la figura"""
        if self.figure:
            self.figure.clear()
            self.canvas.draw()
            
    def save_figure(self, filepath: str, **kwargs):
        """Guarda la figura"""
        if self.figure:
            self.figure.savefig(filepath, **kwargs)
            
    def refresh_canvas(self):
        """Actualiza el canvas"""
        if self.canvas:
            self.canvas.draw()


class PlotlyWidget(VisualizationWidget):
    """Widget para visualizaciones interactivas con Plotly"""
    
    def __init__(self, title: str = "Gráfico Interactivo", parent=None):
        self.webview = None
        super().__init__(title, parent)
        
    def setup_ui(self):
        """Configura la interfaz con WebEngine para Plotly"""
        global WEBENGINE_AVAILABLE
        super().setup_ui()
        
        if not WEBENGINE_AVAILABLE:
            error_label = QLabel("QWebEngineView no está disponible")
            error_label.setAlignment(Qt.AlignCenter)
            error_label.setStyleSheet("color: red; font-size: 14px;")
            
            content_layout = QVBoxLayout(self.content_area)
            content_layout.addWidget(error_label)
            return
            
        # Crear WebView para Plotly
        try:
            self.webview = QWebEngineView()
        except Exception as e:
            # Fallback si falla la instanciación por entorno
            logger.warning(f"Fallo al crear QWebEngineView, desactivando WebEngine: {e}")
            self.webview = None
            WEBENGINE_AVAILABLE = False
            error_label = QLabel("Visualización interactiva desactivada por entorno")
            error_label.setAlignment(Qt.AlignCenter)
            error_label.setStyleSheet("color: orange; font-size: 14px;")
            content_layout = QVBoxLayout(self.content_area)
            content_layout.addWidget(error_label)
            return
        
        # Layout del área de contenido
        content_layout = QVBoxLayout(self.content_area)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.addWidget(self.webview)
        
    def load_html_file(self, filepath: str):
        """Carga un archivo HTML en el WebView"""
        if self.webview and os.path.exists(filepath):
            self.webview.load(QUrl.fromLocalFile(os.path.abspath(filepath)))
            
    def load_html_content(self, html_content: str):
        """Carga contenido HTML directamente"""
        if self.webview:
            self.webview.setHtml(html_content)


class PreprocessingVisualizationWidget(MatplotlibWidget):
    """Widget especializado para visualización de preprocesamiento"""
    
    def __init__(self, parent=None):
        super().__init__("Visualización de Preprocesamiento", parent)
        self.preprocessor_visualizer = None
        self.current_result = None
        
        if VISUALIZERS_AVAILABLE:
            self.preprocessor_visualizer = PreprocessingVisualizer()
            
    def visualize_preprocessing(self, image_path: str, evidence_type: str = "unknown"):
        """Visualiza el preprocesamiento de una imagen"""
        if not self.preprocessor_visualizer:
            self.visualizationError.emit("PreprocessingVisualizer no disponible")
            return
            
        try:
            self.show_progress(True)
            self.set_progress(10)
            
            # Ejecutar preprocesamiento con visualización
            result = self.preprocessor_visualizer.preprocess_with_visualization(
                image_path=image_path,
                evidence_type=evidence_type,
                output_dir=str(self.temp_dir),
                save_steps=True
            )
            
            self.set_progress(80)
            
            if result.success:
                self.current_result = result
                self._display_preprocessing_steps()
                self.visualizationReady.emit(result.visualization_path or "")
            else:
                self.visualizationError.emit(result.error_message)
                
        except Exception as e:
            self.visualizationError.emit(f"Error en visualización: {str(e)}")
        finally:
            self.show_progress(False)
            
    def _display_preprocessing_steps(self):
        """Muestra los pasos de preprocesamiento en el canvas"""
        if not self.current_result or not self.figure:
            return
            
        self.clear_figure()
        
        steps = self.current_result.steps
        if not steps:
            return
            
        # Crear subplots para mostrar pasos
        n_steps = len(steps)
        cols = min(3, n_steps)
        rows = (n_steps + cols - 1) // cols
        
        for i, step in enumerate(steps):
            ax = self.figure.add_subplot(rows, cols, i + 1)
            
            # Mostrar imagen después del paso
            if len(step.image_after.shape) == 3:
                ax.imshow(step.image_after)
            else:
                ax.imshow(step.image_after, cmap='gray')
                
            ax.set_title(f"{i+1}. {step.name}", fontsize=10)
            ax.axis('off')
            
        self.figure.tight_layout()
        self.refresh_canvas()


class FeatureVisualizationWidget(MatplotlibWidget):
    """Widget especializado para visualización de características"""
    
    def __init__(self, parent=None):
        super().__init__("Visualización de Características", parent)
        self.feature_visualizer = None
        
        if VISUALIZERS_AVAILABLE:
            self.feature_visualizer = FeatureVisualizer(str(self.temp_dir))
    
    def extract_features(self, image_path: str, feature_config: dict):
        """Extrae y visualiza características de una imagen"""
        if not self.feature_visualizer:
            self.visualizationError.emit("FeatureVisualizer no disponible")
            return
            
        try:
            self.show_progress(True)
            self.set_progress(10)
            
            # Extraer características con visualización
            result = self.feature_visualizer.generate_comprehensive_report(
                image=cv2.imread(image_path),
                keypoints=None,  # Se detectarán internamente
                descriptors=None,
                output_prefix=f"features_{os.path.basename(image_path).split('.')[0]}"
            )
            
            self.set_progress(80)
            
            if result and isinstance(result, dict) and result:
                self._display_feature_results(result)
                # Emitir la primera ruta de archivo generada como señal
                first_file = next(iter(result.values()), "")
                self.visualizationReady.emit(first_file)
            else:
                error_msg = 'Error desconocido en extracción de características'
                self.visualizationError.emit(error_msg)
                
        except Exception as e:
            self.visualizationError.emit(f"Error en extracción de características: {str(e)}")
        finally:
            self.show_progress(False)
    
    def _display_feature_results(self, result):
        """Muestra los resultados de extracción de características en el canvas"""
        if not result or not self.figure:
            return
            
        self.clear_figure()
        
        # Mostrar imagen con características detectadas
        if hasattr(result, 'visualization_image') and result.visualization_image is not None:
            ax = self.figure.add_subplot(111)
            ax.imshow(result.visualization_image)
            
            # Título con información de características
            num_features = getattr(result, 'num_keypoints', 0)
            algorithm = getattr(result, 'algorithm', 'Unknown')
            ax.set_title(f"Características {algorithm.upper()}: {num_features} puntos detectados", fontsize=12)
            ax.axis('off')
            
        self.figure.tight_layout()
        self.refresh_canvas()
            
    def visualize_keypoints(self, image, keypoints, algorithm="SIFT"):
        """Visualiza puntos clave en la imagen"""
        if not self.feature_visualizer:
            self.visualizationError.emit("FeatureVisualizer no disponible")
            return
            
        try:
            self.show_progress(True)
            self.set_progress(50)
            
            # Generar visualización
            result_image = self.feature_visualizer.visualize_keypoints(
                image, keypoints, algorithm
            )
            
            # Mostrar en canvas
            self.clear_figure()
            ax = self.figure.add_subplot(111)
            ax.imshow(result_image)
            ax.set_title(f"Puntos Clave {algorithm} ({len(keypoints)} puntos)")
            ax.axis('off')
            
            self.refresh_canvas()
            self.visualizationReady.emit("keypoints_visualization")
            
        except Exception as e:
            self.visualizationError.emit(f"Error visualizando características: {str(e)}")
        finally:
            self.show_progress(False)


class StatisticalVisualizationWidget(PlotlyWidget):
    """Widget especializado para visualizaciones estadísticas interactivas"""
    
    def __init__(self, parent=None):
        super().__init__("Análisis Estadístico Interactivo", parent)
        self.statistical_visualizer = None
        
        if VISUALIZERS_AVAILABLE:
            self.statistical_visualizer = StatisticalVisualizer(
                str(self.temp_dir), interactive_mode=True
            )
            
    def create_interactive_dashboard(self, analysis_results: Dict[str, Any]):
        """Crea un dashboard interactivo con los resultados de análisis"""
        if not self.statistical_visualizer:
            self.visualizationError.emit("StatisticalVisualizer no disponible")
            return
            
        try:
            self.show_progress(True)
            self.set_progress(30)
            
            # Generar dashboard interactivo
            dashboard_path = self.statistical_visualizer.create_interactive_dashboard(
                analysis_results, save_path=str(self.temp_dir / "dashboard.html")
            )
            
            self.set_progress(80)
            
            # Cargar en WebView
            self.load_html_file(dashboard_path)
            self.visualizationReady.emit(dashboard_path)
            
        except Exception as e:
            self.visualizationError.emit(f"Error creando dashboard: {str(e)}")
        finally:
            self.show_progress(False)


class ROIVisualizationWidget(MatplotlibWidget):
    """Widget especializado para visualización de ROI"""
    
    def __init__(self, parent=None):
        super().__init__("Visualización de Regiones de Interés", parent)
        self.roi_visualizer = None
        
        if VISUALIZERS_AVAILABLE:
            self.roi_visualizer = ROIVisualizer(str(self.temp_dir))
            
    def visualize_roi_regions(self, image_path: str, roi_regions: List[Dict[str, Any]], 
                            evidence_type: str = "unknown"):
        """Visualiza regiones de interés detectadas"""
        if not self.roi_visualizer:
            self.visualizationError.emit("ROIVisualizer no disponible")
            return
            
        try:
            self.show_progress(True)
            self.set_progress(40)
            
            # Generar visualizaciones de ROI
            visualizations = self.roi_visualizer.generate_comprehensive_report(
                image_path, roi_regions, evidence_type, "roi_analysis"
            )
            
            self.set_progress(80)
            
            # Cargar visualización principal
            if 'overview' in visualizations:
                # Cargar imagen en matplotlib
                import matplotlib.image as mpimg
                img = mpimg.imread(visualizations['overview'])
                
                self.clear_figure()
                ax = self.figure.add_subplot(111)
                ax.imshow(img)
                ax.set_title(f"Regiones de Interés ({len(roi_regions)} detectadas)")
                ax.axis('off')
                
                self.refresh_canvas()
                self.visualizationReady.emit(visualizations['overview'])
            
        except Exception as e:
            self.visualizationError.emit(f"Error visualizando ROI: {str(e)}")
        finally:
            self.show_progress(False)


# Widget de prueba para desarrollo
class VisualizationTestWidget(QWidget):
    """Widget de prueba para los visualizadores"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Pestañas para diferentes visualizadores
        tab_widget = QTabWidget()
        
        # Pestaña de preprocesamiento
        preprocessing_widget = PreprocessingVisualizationWidget()
        tab_widget.addTab(preprocessing_widget, "Preprocesamiento")
        
        # Pestaña de características
        feature_widget = FeatureVisualizationWidget()
        tab_widget.addTab(feature_widget, "Características")
        
        # Pestaña estadística
        statistical_widget = StatisticalVisualizationWidget()
        tab_widget.addTab(statistical_widget, "Estadísticas")
        
        # Pestaña ROI
        roi_widget = ROIVisualizationWidget()
        tab_widget.addTab(roi_widget, "ROI")
        
        layout.addWidget(tab_widget)


class VisualizationPanel(QWidget):
    """Panel principal de visualización para análisis balístico"""
    
    # Señales
    visualization_changed = pyqtSignal(str)
    export_requested = pyqtSignal(str)
    
    def __init__(self, parent=None, compact: bool = False):
        super().__init__(parent)
        self.current_visualization = None
        self.visualization_widgets = {}
        self.compact_mode = compact
        self.setup_ui()
        self.setup_connections()
        
    def setup_ui(self):
        """Configura la interfaz del panel de visualización"""
        layout = QVBoxLayout(self)
        # Ajuste de modo compacto
        if self.compact_mode:
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(0)
        else:
            layout.setContentsMargins(10, 10, 10, 10)
            layout.setSpacing(10)
        
        # Header con controles
        header_frame = QFrame()
        header_frame.setFrameStyle(QFrame.StyledPanel)
        header_layout = QHBoxLayout(header_frame)
        
        # Título
        title_label = QLabel("🔍 Panel de Visualización")
        title_label.setFont(QFont("Arial", 12, QFont.Bold))
        header_layout.addWidget(title_label)
        
        header_layout.addStretch()
        
        # Selector de tipo de visualización
        self.viz_type_combo = QComboBox()
        self.viz_type_combo.addItems([
            "Preprocesamiento",
            "Características",
            "Estadísticas",
            "ROI",
            "Comparación"
        ])
        header_layout.addWidget(QLabel("Tipo:"))
        header_layout.addWidget(self.viz_type_combo)
        
        # Botones de control
        self.refresh_btn = QPushButton("🔄 Actualizar")
        self.export_btn = QPushButton("💾 Exportar")
        
        header_layout.addWidget(self.refresh_btn)
        header_layout.addWidget(self.export_btn)
        
        # En modo compacto, reducir la altura del header y tipografía
        if self.compact_mode:
            header_frame.setMaximumHeight(40)
            title_label.setFont(QFont("Arial", 11, QFont.Bold))
        layout.addWidget(header_frame)
        
        # Área de visualización con tabs
        self.viz_tabs = QTabWidget()
        
        # Tab de preprocesamiento
        self.preprocessing_widget = PreprocessingVisualizationWidget()
        self.viz_tabs.addTab(self.preprocessing_widget, "🔧 Preprocesamiento")
        self.visualization_widgets['preprocessing'] = self.preprocessing_widget
        
        # Tab de características
        self.feature_widget = FeatureVisualizationWidget()
        self.viz_tabs.addTab(self.feature_widget, "🎯 Características")
        self.visualization_widgets['features'] = self.feature_widget
        
        # Tab de estadísticas
        self.statistical_widget = StatisticalVisualizationWidget()
        self.viz_tabs.addTab(self.statistical_widget, "📊 Estadísticas")
        self.visualization_widgets['statistics'] = self.statistical_widget
        
        # Tab de ROI
        self.roi_widget = ROIVisualizationWidget()
        self.viz_tabs.addTab(self.roi_widget, "🔍 ROI")
        self.visualization_widgets['roi'] = self.roi_widget

        # Tab de comparación (SynchronizedViewer)
        self.comparison_widget = None
        if SYNC_VIEWER_AVAILABLE:
            try:
                # Usar un contenedor interno para mantener consistencia con VisualizationWidget
                comparison_container = QWidget()
                comparison_layout = QVBoxLayout(comparison_container)
                # Márgenes mínimos en modo compacto
                if self.compact_mode:
                    comparison_layout.setContentsMargins(0, 0, 0, 0)
                    comparison_layout.setSpacing(0)
                else:
                    comparison_layout.setContentsMargins(5, 5, 5, 5)
                    comparison_layout.setSpacing(5)
                self.comparison_widget = SynchronizedViewer()
                comparison_layout.addWidget(self.comparison_widget)
                self.viz_tabs.addTab(comparison_container, "⚖️ Comparación")
                self.visualization_widgets['comparison'] = self.comparison_widget
            except Exception as e:
                logger.warning(f"Error creando SynchronizedViewer: {e}")
        else:
            # Fallback simple si no está disponible
            fallback = QLabel("Comparación no disponible")
            fallback.setAlignment(Qt.AlignCenter)
            self.viz_tabs.addTab(fallback, "⚖️ Comparación")
            self.visualization_widgets['comparison'] = fallback
        
        layout.addWidget(self.viz_tabs)
        
        # Barra de estado
        self.status_label = QLabel("Listo para visualizar")
        self.status_label.setStyleSheet("color: #666; font-size: 10px;")
        layout.addWidget(self.status_label)
        
    def setup_connections(self):
        """Configura las conexiones de señales"""
        self.viz_type_combo.currentTextChanged.connect(self.on_viz_type_changed)
        self.refresh_btn.clicked.connect(self.refresh_current_visualization)
        self.export_btn.clicked.connect(self.export_current_visualization)
        
        # Conectar señales de widgets de visualización
        for widget in self.visualization_widgets.values():
            if hasattr(widget, 'visualizationReady'):
                widget.visualizationReady.connect(self.on_visualization_ready)
            if hasattr(widget, 'visualizationError'):
                widget.visualizationError.connect(self.on_visualization_error)
                
    def on_viz_type_changed(self, viz_type):
        """Maneja el cambio de tipo de visualización"""
        type_mapping = {
            "Preprocesamiento": 0,
            "Características": 1,
            "Estadísticas": 2,
            "ROI": 3,
            "Comparación": 4
        }
        
        if viz_type in type_mapping:
            self.viz_tabs.setCurrentIndex(type_mapping[viz_type])
            self.status_label.setText(f"Visualización: {viz_type}")
            self.visualization_changed.emit(viz_type.lower())
            
    def refresh_current_visualization(self):
        """Actualiza la visualización actual"""
        current_widget = self.viz_tabs.currentWidget()
        if hasattr(current_widget, 'refresh_visualization'):
            current_widget.refresh_visualization()
            self.status_label.setText("Visualización actualizada")
            
    def export_current_visualization(self):
        """Exporta la visualización actual"""
        current_widget = self.viz_tabs.currentWidget()
        if hasattr(current_widget, 'export_visualization'):
            current_widget.export_visualization()
            self.export_requested.emit("current")
            self.status_label.setText("Visualización exportada")
            
    def on_visualization_ready(self, filepath):
        """Maneja cuando una visualización está lista"""
        self.status_label.setText(f"Visualización lista: {Path(filepath).name}")
        
    def on_visualization_error(self, error_msg):
        """Maneja errores de visualización"""
        self.status_label.setText(f"Error: {error_msg}")
        
    def load_image(self, image_path):
        """Carga una imagen en todas las visualizaciones"""
        for widget in self.visualization_widgets.values():
            if hasattr(widget, 'load_image'):
                widget.load_image(image_path)
        self.status_label.setText(f"Imagen cargada: {Path(image_path).name}")
        
    def set_analysis_results(self, results):
        """Establece los resultados de análisis para visualización"""
        if 'statistics' in self.visualization_widgets:
            stats_widget = self.visualization_widgets['statistics']
            if hasattr(stats_widget, 'create_interactive_dashboard'):
                stats_widget.create_interactive_dashboard(results)
        self.status_label.setText("Resultados de análisis cargados")
        
    def clear_visualizations(self):
        """Limpia todas las visualizaciones"""
        for widget in self.visualization_widgets.values():
            if hasattr(widget, 'clear_figure'):
                widget.clear_figure()
        self.status_label.setText("Visualizaciones limpiadas")

    # Métodos de conveniencia para ComparisonTab
    def clear_display(self):
        """Alias conveniente para limpiar visualizaciones desde otros paneles"""
        self.clear_visualizations()

    def display_query_image(self, image_path: str):
        """Muestra la imagen de consulta en los visualizadores relevantes"""
        try:
            self.load_image(image_path)
            self.viz_type_combo.setCurrentText("Preprocesamiento")
            self.status_label.setText(f"Imagen de consulta: {Path(image_path).name}")
        except Exception as e:
            self.on_visualization_error(f"Error cargando imagen de consulta: {e}")

    def display_result(self, result: Dict[str, Any]):
        """Visualiza un resultado seleccionado en pestañas adecuadas"""
        try:
            # Si el resultado tiene imágenes, cargar la primera
            images = result.get('images') or result.get('image_paths') or []
            if isinstance(images, list) and images:
                self.load_image(images[0])
            # Si hay estadísticas/resultados, mostrarlos
            self.set_analysis_results(result)
            # Cambiar a pestaña de estadísticas para resaltar datos
            self.viz_type_combo.setCurrentText("Estadísticas")
            self.status_label.setText("Resultado seleccionado visualizado")
        except Exception as e:
            self.on_visualization_error(f"Error mostrando resultado: {e}")

    def display_comparison(self, images: List[str], results: Dict[str, Any]):
        """Visualiza una comparación directa entre imágenes usando SynchronizedViewer"""
        try:
            # Preparar pixmaps
            pixmaps = []
            if isinstance(images, list):
                for img_path in images[:2]:
                    pm = QPixmap(str(img_path))
                    if not pm.isNull():
                        pixmaps.append(pm)
            # Si hay al menos una imagen, configurar visor
            if 'comparison' in self.visualization_widgets:
                comp_widget = self.visualization_widgets['comparison']
                if isinstance(comp_widget, SynchronizedViewer) and pixmaps:
                    if len(pixmaps) == 1:
                        # Duplicar pixmap para comparación mínima
                        comp_widget.set_images(pixmaps[0], pixmaps[0])
                    else:
                        comp_widget.set_images(pixmaps[0], pixmaps[1])
            # Mostrar métricas/estadísticas en panel correspondiente si aplica
            self.set_analysis_results(results)
            # Cambiar a pestaña de comparación
            self.viz_type_combo.setCurrentText("Comparación")
            self.viz_tabs.setCurrentIndex(4)
            self.status_label.setText("Comparación directa visualizada")
        except Exception as e:
            self.on_visualization_error(f"Error mostrando comparación: {e}")


# Widget de prueba para desarrollo
class VisualizationTestWidget(QWidget):
    """Widget de prueba para los visualizadores"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Pestañas para diferentes visualizadores
        tab_widget = QTabWidget()
        
        # Pestaña de preprocesamiento
        preprocessing_widget = PreprocessingVisualizationWidget()
        tab_widget.addTab(preprocessing_widget, "Preprocesamiento")
        
        # Pestaña de características
        feature_widget = FeatureVisualizationWidget()
        tab_widget.addTab(feature_widget, "Características")
        
        # Pestaña estadística
        statistical_widget = StatisticalVisualizationWidget()
        tab_widget.addTab(statistical_widget, "Estadísticas")
        
        # Pestaña ROI
        roi_widget = ROIVisualizationWidget()
        tab_widget.addTab(roi_widget, "ROI")
        
        layout.addWidget(tab_widget)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Crear ventana de prueba
    test_widget = VisualizationTestWidget()
    test_widget.setWindowTitle("Prueba de Widgets de Visualización")
    test_widget.resize(1200, 800)
    test_widget.show()
    
    sys.exit(app.exec_())