#!/usr/bin/env python3
"""
Nueva GUI con tres pestañas:
- Carga: imágenes y metadatos NIST (visualización al cargar)
- Análisis: seleccionar imágenes cargadas, ejecutar análisis y mostrar resultados/visualizaciones
- Comparación: elegir entre imágenes cargadas o base de datos, comparar/analizar y mostrar resultados/visualizaciones
"""

from typing import List, Dict, Any
import os
from pathlib import Path
import numpy as np
import cv2

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QTabWidget, QPushButton,
    QLabel, QFileDialog, QComboBox, QFrame, QSizePolicy, QScrollArea
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QPixmap

# Reutilizar widgets existentes del proyecto
from gui.shared_widgets import ImageSelector
from gui.visualization_widgets import VisualizationPanel
from gui.dynamic_results_panel import ResultsPanel
from gui.nist_metadata_widget import NISTMetadataWidget
from gui.synchronized_viewer import SynchronizedViewer
from gui.graphics_widgets import HeatmapWidget, GraphicsVisualizationPanel
from gui.interactive_cmc_widget import InteractiveCMCWidget, CMCCurveData
from gui.styles import apply_modern_qss_to_widget
from gui.analysis_worker import OptimizedAnalysisWorker
from gui.comparison_worker import OptimizedComparisonWorker
try:
    import seaborn as sns
    sns.set_style("whitegrid")
except Exception:
    pass

# Matcher unificado para obtener datos reales de CMC/heatmaps
MATCHER_AVAILABLE = False
try:
    from matching.unified_matcher import UnifiedMatcher, AlgorithmType
    MATCHER_AVAILABLE = True
except Exception:
    UnifiedMatcher = None
    AlgorithmType = None


class TriTabbedGUI(QWidget):
    """
    Interfaz principal con tres pestañas orientadas al flujo de trabajo.
    """

    imagesChanged = pyqtSignal(list)  # rutas de imágenes cargadas

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("SIGeC - Nueva Interfaz (3 Pestañas)")
        self.selected_images: List[str] = []
        self.metadata_snapshot: Dict[str, Any] = {}

        self.init_ui()
        self.setup_connections()

        # Inicializar matcher si está disponible
        try:
            if MATCHER_AVAILABLE:
                self.matcher = UnifiedMatcher()
            else:
                self.matcher = None
        except Exception:
            self.matcher = None

        # Workers asíncronos
        self.analysis_worker = None
        self.comparison_worker = None

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # Título
        title = QLabel("Sistema de Comparación y Análisis Balístico")
        title.setAlignment(Qt.AlignCenter)
        title.setObjectName("titleLabel")
        layout.addWidget(title)

        # Pestañas principales
        self.tabs = QTabWidget()
        self.tabs.setObjectName("mainTabs")

        self.load_tab = self.create_load_tab()
        self.analysis_tab = self.create_analysis_tab()
        self.comparison_tab = self.create_comparison_tab()

        self.tabs.addTab(self.load_tab, "Carga")
        self.tabs.addTab(self.analysis_tab, "Análisis")
        self.tabs.addTab(self.comparison_tab, "Comparación")

        layout.addWidget(self.tabs)

        # Barra de estado simple
        self.status_label = QLabel("Listo")
        self.status_label.setObjectName("statusLabel")
        layout.addWidget(self.status_label)

        # Aplicar tema profesional minimalista (QSS saneado)
        try:
            apply_modern_qss_to_widget(self)
        except Exception:
            pass

    # --- Pestaña 1: Carga ---
    def create_load_tab(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("loadPanel")
        panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        v = QVBoxLayout(panel)
        v.setSpacing(10)

        controls = QHBoxLayout()
        self.btn_add_images = QPushButton("Agregar Imágenes…")
        controls.addWidget(self.btn_add_images)

        self.btn_clear_images = QPushButton("Limpiar")
        controls.addWidget(self.btn_clear_images)

        controls.addStretch()
        v.addLayout(controls)

        # Visor de imágenes + Metadatos en splitter
        splitter = QSplitter(Qt.Horizontal)

        # Selector/visor de imágenes
        self.image_selector = ImageSelector()
        splitter.addWidget(self.image_selector)

        # Metadatos NIST
        self.nist_widget = NISTMetadataWidget()
        splitter.addWidget(self.nist_widget)

        splitter.setSizes([600, 400])
        v.addWidget(splitter)

        return panel

    # --- Pestaña 2: Análisis ---
    def create_analysis_tab(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("analysisPanel")
        v = QVBoxLayout(panel)
        v.setSpacing(10)

        # Controles (convertidos a subpestaña con scroll)
        controls_widget = QFrame()
        controls_layout = QHBoxLayout(controls_widget)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        self.lbl_selected_count = QLabel("Imágenes seleccionadas: 0")
        controls_layout.addWidget(self.lbl_selected_count)
        controls_layout.addStretch()
        self.btn_run_analysis = QPushButton("Ejecutar Análisis")
        controls_layout.addWidget(self.btn_run_analysis)

        # Widgets de visualización individuales (cada uno será subpestaña)
        self.analysis_visualization = VisualizationPanel()
        self.analysis_heatmap = HeatmapWidget()
        self.analysis_cmc = InteractiveCMCWidget()
        self.analysis_graphics = GraphicsVisualizationPanel()
        # Resultados
        self.analysis_results = ResultsPanel()

        # Helper para envolver en scroll
        def wrap_scroll(widget: QWidget) -> QScrollArea:
            sa = QScrollArea()
            sa.setWidgetResizable(True)
            sa.setFrameShape(QFrame.NoFrame)
            sa.setWidget(widget)
            return sa

        # Subpestañas de Análisis
        self.analysis_sections = QTabWidget()
        self.analysis_sections.addTab(wrap_scroll(controls_widget), "Controles")
        self.analysis_sections.addTab(wrap_scroll(self.analysis_visualization), "Visualización")
        self.analysis_sections.addTab(wrap_scroll(self.analysis_heatmap), "Heatmap")
        self.analysis_sections.addTab(wrap_scroll(self.analysis_cmc), "CMC")
        self.analysis_sections.addTab(wrap_scroll(self.analysis_graphics), "Gráficos")
        self.analysis_sections.addTab(wrap_scroll(self.analysis_results), "Resultados")

        # Índices para navegación programática
        self.analysis_tab_indices = {
            'Controles': 0,
            'Visualización': 1,
            'Heatmap': 2,
            'CMC': 3,
            'Gráficos': 4,
            'Resultados': 5,
        }

        v.addWidget(self.analysis_sections)

        return panel

    # --- Pestaña 3: Comparación ---
    def create_comparison_tab(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("comparisonPanel")
        v = QVBoxLayout(panel)
        v.setSpacing(10)

        # Controles de origen/operación (subpestaña con scroll)
        controls_widget = QFrame()
        controls = QHBoxLayout(controls_widget)
        controls.setContentsMargins(0, 0, 0, 0)
        controls.addWidget(QLabel("Origen:"))
        self.origin_combo = QComboBox()
        self.origin_combo.addItems(["Imágenes cargadas", "Base de datos (simulada)"])
        controls.addWidget(self.origin_combo)
        controls.addStretch()
        self.btn_compare = QPushButton("Comparar")
        controls.addWidget(self.btn_compare)

        # Visor sincronizado
        self.sync_viewer = SynchronizedViewer()

        # Widgets de visualización individuales
        self.comparison_visualization = VisualizationPanel()
        self.comparison_heatmap = HeatmapWidget()
        self.comparison_cmc = InteractiveCMCWidget()
        self.comparison_graphics = GraphicsVisualizationPanel()
        # Resultados
        self.comparison_results = ResultsPanel()

        # Helper para envolver en scroll
        def wrap_scroll(widget: QWidget) -> QScrollArea:
            sa = QScrollArea()
            sa.setWidgetResizable(True)
            sa.setFrameShape(QFrame.NoFrame)
            sa.setWidget(widget)
            return sa

        # Subpestañas de Comparación
        self.comparison_sections = QTabWidget()
        self.comparison_sections.addTab(wrap_scroll(controls_widget), "Controles")
        self.comparison_sections.addTab(wrap_scroll(self.sync_viewer), "Visor sincronizado")
        self.comparison_sections.addTab(wrap_scroll(self.comparison_visualization), "Visualización")
        self.comparison_sections.addTab(wrap_scroll(self.comparison_heatmap), "Heatmap")
        self.comparison_sections.addTab(wrap_scroll(self.comparison_cmc), "CMC")
        self.comparison_sections.addTab(wrap_scroll(self.comparison_graphics), "Gráficos")
        self.comparison_sections.addTab(wrap_scroll(self.comparison_results), "Resultados")

        # Índices para navegación programática
        self.comparison_tab_indices = {
            'Controles': 0,
            'Visor': 1,
            'Visualización': 2,
            'Heatmap': 3,
            'CMC': 4,
            'Gráficos': 5,
            'Resultados': 6,
        }

        v.addWidget(self.comparison_sections)
        return panel

    def setup_connections(self):
        # Botones de carga
        self.btn_add_images.clicked.connect(self.on_add_images)
        self.btn_clear_images.clicked.connect(self.on_clear_images)

        # Imagen seleccionada desde ImageSelector
        # Compatibilidad: si ImageSelector emite señales, intentar conectarlas
        if hasattr(self.image_selector, 'images_selected'):
            try:
                self.image_selector.images_selected.connect(self.on_images_updated)
            except Exception:
                pass

        # Ejecutar análisis
        self.btn_run_analysis.clicked.connect(self.on_run_analysis)

        # Comparación
        self.btn_compare.clicked.connect(self.on_compare)

        # Propagación de cambios
        self.imagesChanged.connect(self.on_images_changed)

    # --- Lógica de carga ---
    def on_add_images(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Seleccionar imágenes", str(Path.cwd()),
            "Imágenes (*.png *.jpg *.jpeg *.bmp)"
        )
        if files:
            self.selected_images = files
            # Mostrar en ImageSelector
            try:
                if hasattr(self.image_selector, 'set_images'):
                    self.image_selector.set_images(files)
                elif hasattr(self.image_selector, 'set_image'):
                    self.image_selector.set_image(files[0])
            except Exception:
                pass
            # Actualizar metadatos mínimos
            try:
                if hasattr(self.nist_widget, 'get_metadata'):
                    self.metadata_snapshot = self.nist_widget.get_metadata()
            except Exception:
                pass

            self.imagesChanged.emit(files)
            self.status_label.setText(f"{len(files)} imágenes cargadas")

    def on_clear_images(self):
        self.selected_images = []
        try:
            if hasattr(self.image_selector, 'clear_selection'):
                self.image_selector.clear_selection()
        except Exception:
            pass
        self.imagesChanged.emit([])
        self.status_label.setText("Selección de imágenes limpiada")

    def on_images_updated(self, images: List[str]):
        self.selected_images = images or []
        self.imagesChanged.emit(self.selected_images)

    def on_images_changed(self, images: List[str]):
        # Actualizar contador en análisis
        self.lbl_selected_count.setText(f"Imágenes seleccionadas: {len(images)}")

    # --- Lógica de análisis ---
    def on_run_analysis(self):
        # Iniciar análisis en segundo plano usando OptimizedAnalysisWorker
        if not self.selected_images:
            self.status_label.setText("Seleccione imágenes en la pestaña Carga")
            return
        
        # Visualizar primera imagen como referencia
        try:
            self.analysis_visualization.load_image(self.selected_images[0])
            if hasattr(self, 'analysis_sections'):
                self.analysis_sections.setCurrentIndex(self.analysis_tab_indices.get('Visualización', 0))
        except Exception:
            pass
        
        # Evitar múltiples ejecuciones simultáneas
        if self.analysis_worker is not None:
            self.status_label.setText("Análisis en ejecución…")
            return
        
        # Configurar y arrancar worker
        self.analysis_worker = OptimizedAnalysisWorker(self)
        try:
            # Conectar señales
            self.analysis_worker.progress_updated.connect(self._on_analysis_progress)
            self.analysis_worker.status_changed.connect(self._on_analysis_status)
            self.analysis_worker.analysis_completed.connect(self._on_analysis_completed)
            self.analysis_worker.error_occurred.connect(self._on_analysis_error)
            self.analysis_worker.visualization_ready.connect(self._on_analysis_visualization)
            self.analysis_worker.memory_usage_updated.connect(self._on_analysis_memory)
            self.analysis_worker.quality_metrics_ready.connect(self._on_analysis_quality)
        except Exception:
            pass
        
        # Preparar parámetros
        case_data = {
            'source': 'TriTabbedGUI',
            'selected_count': len(self.selected_images)
        }
        nist_md = self.metadata_snapshot if isinstance(self.metadata_snapshot, dict) else None
        processing_config = {
            'enable_parallel_analysis': True,
            'cache_intermediate_results': True,
            'memory_optimization': True,
            'async_visualizations': True
        }
        
        try:
            self.analysis_worker.setup_analysis(
                image_path=self.selected_images[0],
                case_data=case_data,
                nist_metadata=nist_md,
                processing_config=processing_config
            )
        except Exception:
            # Si setup falla, informar y no iniciar
            self.status_label.setText("No se pudo configurar el análisis")
            self.analysis_worker = None
            return
        
        # Deshabilitar botón mientras corre
        try:
            self.btn_run_analysis.setEnabled(False)
        except Exception:
            pass
        
        self.status_label.setText("Análisis en ejecución…")
        self.analysis_worker.start()

        # Nota: Implementación asíncrona, la lógica síncrona anterior se ha eliminado.
        # Los resultados y visualizaciones se actualizan vía señales del worker.
        return

    # --- Lógica de comparación ---
    def on_compare(self):
        # Comparación asincrónica usando OptimizedComparisonWorker
        origin = self.origin_combo.currentText()
        
        # Elegir imágenes fuente
        if origin.startswith("Imágenes"):
            images = self.selected_images
        else:
            assets_dir = Path("assets")
            images = [str(p) for p in assets_dir.glob("*.png")][:2]
        
        if len(images) < 2:
            self.status_label.setText("Necesita 2 imágenes para comparar")
            return
        
        # Cargar en visor sincronizado
        try:
            pix1 = QPixmap(images[0])
            pix2 = QPixmap(images[1])
            self.sync_viewer.set_images(pix1, pix2)
        except Exception:
            pass
        
        # Evitar múltiples ejecuciones simultáneas
        if self.comparison_worker is not None:
            self.status_label.setText("Comparación en ejecución…")
            return
        
        # Configurar y arrancar worker
        self.comparison_worker = OptimizedComparisonWorker(self)
        try:
            # Conectar señales
            self.comparison_worker.progress_updated.connect(self._on_comparison_progress)
            self.comparison_worker.status_changed.connect(self._on_comparison_status)
            self.comparison_worker.analysis_completed.connect(self._on_comparison_completed)
            self.comparison_worker.error_occurred.connect(self._on_comparison_error)
            self.comparison_worker.visualization_ready.connect(self._on_comparison_visualization)
            self.comparison_worker.memory_usage_updated.connect(self._on_comparison_memory)
            self.comparison_worker.match_found.connect(self._on_comparison_match_found)
        except Exception:
            pass
        
        comparison_config = {
            'enable_parallel_processing': True,
            'memory_optimization': True,
            'cache_intermediate_results': True
        }
        
        try:
            self.comparison_worker.setup_direct_comparison(
                image_a_path=images[0],
                image_b_path=images[1],
                comparison_config=comparison_config
            )
        except Exception:
            self.status_label.setText("No se pudo configurar la comparación")
            self.comparison_worker = None
            return
        
        try:
            self.btn_compare.setEnabled(False)
        except Exception:
            pass
        
        self.status_label.setText("Comparando…")
        self.comparison_worker.start()
    def _on_analysis_progress(self, progress: int, message: str):
        try:
            self.status_label.setText(f"Análisis {progress}% - {message}")
        except Exception:
            pass

    def _on_analysis_status(self, text: str):
        try:
            self.status_label.setText(text)
        except Exception:
            pass

    def _on_analysis_memory(self, memory_mb: float):
        try:
            # Mostrar uso de memoria de forma compacta
            self.status_label.setText(f"Memoria: {memory_mb:.0f} MB")
        except Exception:
            pass

    def _on_analysis_visualization(self, viz_name: str, viz_data: object):
        # Integraciones ligeras según tipo de visualización
        try:
            if viz_name == 'histogram' and isinstance(viz_data, dict):
                # Actualizar panel de gráficos si soporta
                summary = {'has_histogram': True}
                if hasattr(self.analysis_graphics, 'update_with_results'):
                    self.analysis_graphics.update_with_results(summary)
            elif viz_name == 'feature_map' and isinstance(viz_data, dict):
                data = viz_data.get('data')
                if isinstance(data, dict) and 'matrix' in data:
                    import numpy as np
                    matrix = np.array(data['matrix'])
                    self.analysis_heatmap.set_heatmap_data(matrix, title='Mapa de Características', colormap='viridis')
                    if hasattr(self, 'analysis_sections'):
                        self.analysis_sections.setCurrentIndex(self.analysis_tab_indices.get('Heatmap', 2))
        except Exception:
            pass

    def _on_analysis_quality(self, metrics: object):
        try:
            if isinstance(metrics, dict):
                oq = metrics.get('overall_quality')
                if oq is not None:
                    self.status_label.setText(f"Calidad global: {oq:.2f}")
        except Exception:
            pass

    def _on_analysis_completed(self, result: object):
        # Adaptar AnalysisResult a ResultsPanel
        try:
            res_list = []
            # Similarity puede no aplicar en análisis individual; usar métrica de calidad si disponible
            quality = 0.0
            try:
                q = getattr(result, 'quality_metrics', None)
                if isinstance(q, dict):
                    quality = float(q.get('overall_quality', 0.0) or 0.0)
            except Exception:
                pass
            res_list.append({
                'id': 'Análisis Optimizado',
                'similarity': quality,
                'confidence': float(getattr(result, 'confidence', 0.0) or 0.0),
                'match_type': 'Individual',
                'image_path': getattr(result, 'image_path', self.selected_images[0] if self.selected_images else None)
            })
            
            if hasattr(self.analysis_results, 'display_results'):
                self.analysis_results.display_results(res_list)
            elif hasattr(self.analysis_results, 'set_sample_results'):
                self.analysis_results.set_sample_results()
            
            if hasattr(self, 'analysis_sections'):
                self.analysis_sections.setCurrentIndex(self.analysis_tab_indices.get('Resultados', 5))
            
            # Rehabilitar botón y limpiar worker
            try:
                self.btn_run_analysis.setEnabled(True)
            except Exception:
                pass
            self.analysis_worker = None
            self.status_label.setText("Análisis completado")
        except Exception:
            try:
                self.btn_run_analysis.setEnabled(True)
            except Exception:
                pass
            self.analysis_worker = None

    def _on_analysis_error(self, message: str, details: str = ""):
        try:
            self.status_label.setText(f"Error en análisis: {message}")
        except Exception:
            pass
        try:
            self.btn_run_analysis.setEnabled(True)
        except Exception:
            pass
        self.analysis_worker = None

    def _on_comparison_progress(self, progress: int, message: str):
        try:
            self.status_label.setText(f"Comparación {progress}% - {message}")
        except Exception:
            pass

    def _on_comparison_status(self, text: str):
        try:
            self.status_label.setText(text)
        except Exception:
            pass

    def _on_comparison_memory(self, memory_mb: float):
        try:
            self.status_label.setText(f"Memoria: {memory_mb:.0f} MB")
        except Exception:
            pass

    def _on_comparison_visualization(self, viz_name: str, viz_data: object):
        try:
            # Si se entrega un futuro, esperar de forma no bloqueante
            from concurrent.futures import Future
            if isinstance(viz_data, Future):
                def _done(fut: Future):
                    data = None
                    try:
                        data = fut.result()
                    except Exception:
                        data = None
                    # Programar actualización en hilo de GUI
                    from PyQt5.QtCore import QTimer
                    QTimer.singleShot(0, lambda: self._apply_comparison_visualization(viz_name, data))
                viz_data.add_done_callback(_done)
            else:
                self._apply_comparison_visualization(viz_name, viz_data)
        except Exception:
            pass

    def _apply_comparison_visualization(self, viz_name: str, data: object):
        try:
            if viz_name == 'match_visualization' and isinstance(data, dict):
                # Mostrar visualización básica y llevar a pestaña correspondiente
                if hasattr(self.comparison_visualization, 'load_image') and self.selected_images:
                    self.comparison_visualization.load_image(self.selected_images[0])
                if hasattr(self, 'comparison_sections'):
                    self.comparison_sections.setCurrentIndex(self.comparison_tab_indices.get('Visualización', 2))
        except Exception:
            pass

    def _on_comparison_match_found(self, match: object):
        # Podríamos actualizar resultados incrementales si se requiere
        pass

    def _on_comparison_completed(self, result: object):
        try:
            comp_results = [{
                'id': 'Comparación Optimizada',
                'similarity': float(getattr(result, 'similarity_score', 0.0) or 0.0),
                'confidence': float((getattr(result, 'comparison_results', {}) or {}).get('similarity_analysis', {}).get('confidence', 0.0)),
                'match_type': 'CMC',
                'images': self.selected_images[:2]
            }]
            
            if hasattr(self.comparison_results, 'display_results'):
                self.comparison_results.display_results(comp_results)
            elif hasattr(self.comparison_results, 'set_sample_results'):
                self.comparison_results.set_sample_results()
            
            if hasattr(self, 'comparison_sections'):
                self.comparison_sections.setCurrentIndex(self.comparison_tab_indices.get('Resultados', 6))
            
            # Panel de gráficos con resumen
            try:
                summary = {
                    'count': len(comp_results),
                    'avg_similarity': float(np.mean([r.get('similarity', 0) for r in comp_results])),
                    'avg_confidence': float(np.mean([r.get('confidence', 0) for r in comp_results]))
                }
                if hasattr(self.comparison_graphics, 'update_with_results'):
                    self.comparison_graphics.update_with_results(summary)
            except Exception:
                pass
            
            try:
                self.btn_compare.setEnabled(True)
            except Exception:
                pass
            self.comparison_worker = None
            self.status_label.setText("Comparación completada")
        except Exception:
            try:
                self.btn_compare.setEnabled(True)
            except Exception:
                pass
            self.comparison_worker = None

    def _on_comparison_error(self, message: str, details: str = ""):
        try:
            self.status_label.setText(f"Error en comparación: {message}")
        except Exception:
            pass
        try:
            self.btn_compare.setEnabled(True)
        except Exception:
            pass
        self.comparison_worker = None