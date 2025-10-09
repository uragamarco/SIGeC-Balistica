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
        super().setup_ui()
        
        if not WEBENGINE_AVAILABLE:
            error_label = QLabel("QWebEngineView no está disponible")
            error_label.setAlignment(Qt.AlignCenter)
            error_label.setStyleSheet("color: red; font-size: 14px;")
            
            content_layout = QVBoxLayout(self.content_area)
            content_layout.addWidget(error_label)
            return
            
        # Crear WebView para Plotly
        self.webview = QWebEngineView()
        
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


if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Crear ventana de prueba
    test_widget = VisualizationTestWidget()
    test_widget.setWindowTitle("Prueba de Widgets de Visualización")
    test_widget.resize(1200, 800)
    test_widget.show()
    
    sys.exit(app.exec_())