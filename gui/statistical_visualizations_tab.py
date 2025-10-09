#!/usr/bin/env python3
"""
Pestaña de Visualizaciones Estadísticas
Interfaz completa para análisis estadístico y visualizaciones interactivas de datos balísticos
"""

import os
import json
import numpy as np
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout,
    QLabel, QPushButton, QLineEdit, QTextEdit, QComboBox, QSpinBox,
    QCheckBox, QGroupBox, QScrollArea, QSplitter, QFrame, QSpacerItem,
    QSizePolicy, QFileDialog, QMessageBox, QProgressBar, QSlider, 
    QDoubleSpinBox, QTabWidget, QTableWidget, QTableWidgetItem,
    QHeaderView, QListWidget, QListWidgetItem
)
from PyQt5.QtCore import Qt, pyqtSignal, QThread, QTimer
from PyQt5.QtGui import QFont, QPixmap, QIcon

from .shared_widgets import (
    ImageDropZone, ResultCard, CollapsiblePanel, StepIndicator, 
    ProgressCard, ImageViewer
)
from .visualization_widgets import StatisticalVisualizationWidget

# Importaciones del sistema de análisis estadístico
try:
    from image_processing.statistical_visualizer import StatisticalVisualizer
    from image_processing.statistical_analyzer import StatisticalAnalyzer
    from database.unified_database import UnifiedDatabase
    STATISTICAL_MODULES_AVAILABLE = True
except ImportError as e:
    STATISTICAL_MODULES_AVAILABLE = False
    print(f"Módulos estadísticos no disponibles: {e}")

from utils.logger import LoggerMixin, get_logger


class StatisticalAnalysisWorker(QThread, LoggerMixin):
    """Worker thread para realizar análisis estadístico en segundo plano"""
    
    progressUpdated = pyqtSignal(int, str)
    analysisCompleted = pyqtSignal(dict)
    analysisError = pyqtSignal(str)
    
    def __init__(self, analysis_params: dict):
        super().__init__()
        self.analysis_params = analysis_params
        
    def run(self):
        """Ejecuta el análisis estadístico"""
        try:
            self.logger.info("Iniciando análisis estadístico")
            
            # Simular análisis con datos de ejemplo
            self.progressUpdated.emit(20, "Cargando datos...")
            
            # Generar datos de ejemplo para demostración
            sample_data = self._generate_sample_data()
            
            self.progressUpdated.emit(50, "Ejecutando análisis PCA...")
            
            # Simular análisis PCA
            pca_results = self._perform_pca_analysis(sample_data)
            
            self.progressUpdated.emit(70, "Ejecutando análisis de clustering...")
            
            # Simular análisis de clustering
            clustering_results = self._perform_clustering_analysis(sample_data)
            
            self.progressUpdated.emit(90, "Generando correlaciones...")
            
            # Simular análisis de correlación
            correlation_results = self._perform_correlation_analysis(sample_data)
            
            self.progressUpdated.emit(100, "Análisis completado")
            
            # Compilar resultados
            results = {
                'pca_analysis': pca_results,
                'clustering_analysis': clustering_results,
                'correlation_analysis': correlation_results,
                'sample_data': sample_data,
                'timestamp': datetime.now().isoformat()
            }
            
            self.analysisCompleted.emit(results)
            
        except Exception as e:
            self.logger.error(f"Error en análisis estadístico: {e}")
            self.analysisError.emit(str(e))
    
    def _generate_sample_data(self) -> Dict[str, Any]:
        """Genera datos de ejemplo para demostración"""
        np.random.seed(42)
        n_samples = 100
        
        # Generar características balísticas simuladas
        features = {
            'diameter': np.random.normal(9.0, 0.1, n_samples),
            'weight': np.random.normal(124.0, 2.0, n_samples),
            'velocity': np.random.normal(350.0, 15.0, n_samples),
            'rifling_depth': np.random.normal(0.15, 0.02, n_samples),
            'land_width': np.random.normal(2.5, 0.3, n_samples),
            'groove_width': np.random.normal(3.2, 0.4, n_samples)
        }
        
        return {
            'features': features,
            'n_samples': n_samples,
            'feature_names': list(features.keys())
        }
    
    def _perform_pca_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Simula análisis PCA"""
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        
        # Preparar datos
        features_matrix = np.column_stack([data['features'][name] for name in data['feature_names']])
        
        # Normalizar
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features_matrix)
        
        # PCA
        pca = PCA(n_components=min(len(data['feature_names']), 6))
        pca_result = pca.fit_transform(features_scaled)
        
        return {
            'components': pca_result,
            'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
            'feature_names': data['feature_names'],
            'n_components': pca.n_components_
        }
    
    def _perform_clustering_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Simula análisis de clustering"""
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        
        # Preparar datos
        features_matrix = np.column_stack([data['features'][name] for name in data['feature_names']])
        
        # Normalizar
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features_matrix)
        
        # K-means
        kmeans = KMeans(n_clusters=3, random_state=42)
        cluster_labels = kmeans.fit_predict(features_scaled)
        
        return {
            'cluster_labels': cluster_labels.tolist(),
            'cluster_centers': kmeans.cluster_centers_.tolist(),
            'n_clusters': 3,
            'feature_names': data['feature_names']
        }
    
    def _perform_correlation_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Simula análisis de correlación"""
        import pandas as pd
        
        # Crear DataFrame
        df = pd.DataFrame(data['features'])
        
        # Calcular matriz de correlación
        correlation_matrix = df.corr()
        
        return {
            'correlation_matrix': correlation_matrix.values.tolist(),
            'feature_names': data['feature_names']
        }


class StatisticalVisualizationsTab(QWidget, LoggerMixin):
    """Pestaña principal para visualizaciones estadísticas"""
    
    # Señales
    analysisStarted = pyqtSignal()
    analysisCompleted = pyqtSignal(dict)
    
    def __init__(self):
        super().__init__()
        self.current_results = None
        self.statistical_visualizer = None
        self.database = None
        
        # Inicializar componentes si están disponibles
        if STATISTICAL_MODULES_AVAILABLE:
            try:
                self.statistical_visualizer = StatisticalVisualizer(
                    output_dir="temp/statistical_viz",
                    interactive_mode=True
                )
                self.database = UnifiedDatabase()
            except Exception as e:
                self.logger.warning(f"Error inicializando componentes estadísticos: {e}")
        
        self.setup_ui()
        self.setup_connections()
        
    def setup_ui(self):
        """Configura la interfaz de usuario"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # Header con título y controles principales
        self.setup_header(layout)
        
        # Splitter principal
        main_splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(main_splitter)
        
        # Panel de control izquierdo
        self.setup_control_panel(main_splitter)
        
        # Panel de visualización derecho
        self.setup_visualization_panel(main_splitter)
        
        # Configurar proporciones del splitter
        main_splitter.setSizes([300, 900])
        
    def setup_header(self, parent_layout):
        """Configura el header con título y controles"""
        header_frame = QFrame()
        header_frame.setFrameStyle(QFrame.StyledPanel)
        header_frame.setMaximumHeight(80)
        header_layout = QHBoxLayout(header_frame)
        
        # Título y descripción
        title_layout = QVBoxLayout()
        
        title_label = QLabel("📊 Visualizaciones Estadísticas")
        title_label.setFont(QFont("Arial", 16, QFont.Bold))
        title_layout.addWidget(title_label)
        
        desc_label = QLabel("Análisis estadístico avanzado y visualizaciones interactivas de datos balísticos")
        desc_label.setStyleSheet("color: #666; font-size: 11px;")
        title_layout.addWidget(desc_label)
        
        header_layout.addLayout(title_layout)
        header_layout.addStretch()
        
        # Botones principales
        self.new_analysis_btn = QPushButton("🔬 Nuevo Análisis")
        self.new_analysis_btn.setMinimumHeight(35)
        self.new_analysis_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        
        self.load_data_btn = QPushButton("📂 Cargar Datos")
        self.load_data_btn.setMinimumHeight(35)
        
        self.export_btn = QPushButton("💾 Exportar")
        self.export_btn.setMinimumHeight(35)
        self.export_btn.setEnabled(False)
        
        header_layout.addWidget(self.new_analysis_btn)
        header_layout.addWidget(self.load_data_btn)
        header_layout.addWidget(self.export_btn)
        
        parent_layout.addWidget(header_frame)
        
    def setup_control_panel(self, parent_splitter):
        """Configura el panel de control izquierdo"""
        control_widget = QWidget()
        control_layout = QVBoxLayout(control_widget)
        
        # Panel de configuración de análisis
        config_group = QGroupBox("⚙️ Configuración de Análisis")
        config_layout = QVBoxLayout(config_group)
        
        # Tipo de análisis
        analysis_layout = QFormLayout()
        
        self.analysis_type_combo = QComboBox()
        self.analysis_type_combo.addItems([
            "Análisis Completo",
            "Solo PCA",
            "Solo Clustering", 
            "Solo Correlaciones",
            "Análisis Personalizado"
        ])
        analysis_layout.addRow("Tipo de Análisis:", self.analysis_type_combo)
        
        # Número de componentes PCA
        self.pca_components_spin = QSpinBox()
        self.pca_components_spin.setRange(2, 10)
        self.pca_components_spin.setValue(3)
        analysis_layout.addRow("Componentes PCA:", self.pca_components_spin)
        
        # Número de clusters
        self.n_clusters_spin = QSpinBox()
        self.n_clusters_spin.setRange(2, 10)
        self.n_clusters_spin.setValue(3)
        analysis_layout.addRow("Número de Clusters:", self.n_clusters_spin)
        
        config_layout.addLayout(analysis_layout)
        control_layout.addWidget(config_group)
        
        # Panel de fuente de datos
        data_group = QGroupBox("📊 Fuente de Datos")
        data_layout = QVBoxLayout(data_group)
        
        self.data_source_combo = QComboBox()
        self.data_source_combo.addItems([
            "Datos de Ejemplo",
            "Base de Datos",
            "Archivo CSV",
            "Análisis Previo"
        ])
        data_layout.addWidget(self.data_source_combo)
        
        self.data_info_label = QLabel("Seleccione una fuente de datos")
        self.data_info_label.setStyleSheet("color: #666; font-size: 10px;")
        data_layout.addWidget(self.data_info_label)
        
        control_layout.addWidget(data_group)
        
        # Panel de progreso
        progress_group = QGroupBox("📈 Progreso")
        progress_layout = QVBoxLayout(progress_group)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        progress_layout.addWidget(self.progress_bar)
        
        self.progress_label = QLabel("Listo para análisis")
        self.progress_label.setStyleSheet("font-size: 10px;")
        progress_layout.addWidget(self.progress_label)
        
        control_layout.addWidget(progress_group)
        
        # Panel de resultados disponibles
        results_group = QGroupBox("📋 Resultados Disponibles")
        results_layout = QVBoxLayout(results_group)
        
        self.results_list = QListWidget()
        self.results_list.setMaximumHeight(150)
        results_layout.addWidget(self.results_list)
        
        control_layout.addWidget(results_group)
        
        control_layout.addStretch()
        parent_splitter.addWidget(control_widget)
        
    def setup_visualization_panel(self, parent_splitter):
        """Configura el panel de visualización derecho"""
        viz_widget = QWidget()
        viz_layout = QVBoxLayout(viz_widget)
        
        # Tabs para diferentes tipos de visualización
        self.viz_tabs = QTabWidget()
        
        # Tab de Dashboard Interactivo
        self.dashboard_tab = StatisticalVisualizationWidget()
        self.viz_tabs.addTab(self.dashboard_tab, "🎯 Dashboard Interactivo")
        
        # Tab de Análisis PCA
        self.pca_tab = StatisticalVisualizationWidget()
        self.viz_tabs.addTab(self.pca_tab, "📊 Análisis PCA")
        
        # Tab de Clustering
        self.clustering_tab = StatisticalVisualizationWidget()
        self.viz_tabs.addTab(self.clustering_tab, "🎯 Clustering")
        
        # Tab de Correlaciones
        self.correlation_tab = StatisticalVisualizationWidget()
        self.viz_tabs.addTab(self.correlation_tab, "🔗 Correlaciones")
        
        # Tab de Reportes
        self.reports_tab = self.setup_reports_tab()
        self.viz_tabs.addTab(self.reports_tab, "📄 Reportes")
        
        viz_layout.addWidget(self.viz_tabs)
        parent_splitter.addWidget(viz_widget)
        
    def setup_reports_tab(self) -> QWidget:
        """Configura el tab de reportes"""
        reports_widget = QWidget()
        reports_layout = QVBoxLayout(reports_widget)
        
        # Header del reporte
        header_layout = QHBoxLayout()
        
        report_title = QLabel("📄 Reportes Estadísticos")
        report_title.setFont(QFont("Arial", 14, QFont.Bold))
        header_layout.addWidget(report_title)
        
        header_layout.addStretch()
        
        generate_report_btn = QPushButton("📋 Generar Reporte")
        save_report_btn = QPushButton("💾 Guardar Reporte")
        
        header_layout.addWidget(generate_report_btn)
        header_layout.addWidget(save_report_btn)
        
        reports_layout.addLayout(header_layout)
        
        # Área de texto para el reporte
        self.report_text = QTextEdit()
        self.report_text.setPlainText("No hay análisis disponible. Ejecute un análisis para generar reportes.")
        reports_layout.addWidget(self.report_text)
        
        # Conectar botones
        generate_report_btn.clicked.connect(self.generate_report)
        save_report_btn.clicked.connect(self.save_report)
        
        return reports_widget
        
    def setup_connections(self):
        """Configura las conexiones de señales"""
        self.new_analysis_btn.clicked.connect(self.start_new_analysis)
        self.load_data_btn.clicked.connect(self.load_external_data)
        self.export_btn.clicked.connect(self.export_results)
        
        # Conectar cambios en configuración
        self.analysis_type_combo.currentTextChanged.connect(self.on_analysis_type_changed)
        self.data_source_combo.currentTextChanged.connect(self.on_data_source_changed)
        
    def start_new_analysis(self):
        """Inicia un nuevo análisis estadístico"""
        try:
            # Recopilar parámetros de análisis
            analysis_params = {
                'analysis_type': self.analysis_type_combo.currentText(),
                'pca_components': self.pca_components_spin.value(),
                'n_clusters': self.n_clusters_spin.value(),
                'data_source': self.data_source_combo.currentText()
            }
            
            # Mostrar progreso
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            self.progress_label.setText("Iniciando análisis...")
            
            # Deshabilitar botón
            self.new_analysis_btn.setEnabled(False)
            
            # Crear y ejecutar worker
            self.analysis_worker = StatisticalAnalysisWorker(analysis_params)
            self.analysis_worker.progressUpdated.connect(self.on_analysis_progress)
            self.analysis_worker.analysisCompleted.connect(self.on_analysis_completed)
            self.analysis_worker.analysisError.connect(self.on_analysis_error)
            self.analysis_worker.start()
            
            self.analysisStarted.emit()
            
        except Exception as e:
            self.logger.error(f"Error iniciando análisis: {e}")
            QMessageBox.critical(self, "Error", f"Error iniciando análisis: {str(e)}")
            
    def on_analysis_progress(self, progress: int, message: str):
        """Actualiza el progreso del análisis"""
        self.progress_bar.setValue(progress)
        self.progress_label.setText(message)
        
    def on_analysis_completed(self, results: dict):
        """Maneja la finalización del análisis"""
        try:
            self.current_results = results
            
            # Ocultar progreso
            self.progress_bar.setVisible(False)
            self.progress_label.setText("Análisis completado")
            
            # Habilitar botones
            self.new_analysis_btn.setEnabled(True)
            self.export_btn.setEnabled(True)
            
            # Actualizar visualizaciones
            self.update_visualizations(results)
            
            # Agregar a lista de resultados
            timestamp = datetime.now().strftime("%H:%M:%S")
            item_text = f"Análisis {timestamp} - {self.analysis_type_combo.currentText()}"
            self.results_list.addItem(item_text)
            
            # Generar reporte automático
            self.generate_report()
            
            self.analysisCompleted.emit(results)
            
        except Exception as e:
            self.logger.error(f"Error procesando resultados: {e}")
            QMessageBox.critical(self, "Error", f"Error procesando resultados: {str(e)}")
            
    def on_analysis_error(self, error_message: str):
        """Maneja errores en el análisis"""
        self.progress_bar.setVisible(False)
        self.progress_label.setText("Error en análisis")
        self.new_analysis_btn.setEnabled(True)
        
        QMessageBox.critical(self, "Error de Análisis", f"Error durante el análisis:\n{error_message}")
        
    def update_visualizations(self, results: dict):
        """Actualiza todas las visualizaciones con los nuevos resultados"""
        try:
            if not self.statistical_visualizer:
                self.logger.warning("StatisticalVisualizer no disponible")
                return
                
            # Actualizar dashboard interactivo
            self.dashboard_tab.create_interactive_dashboard(results)
            
            # Las otras pestañas se pueden actualizar con visualizaciones específicas
            # Por ahora, mostrar mensaje de éxito
            self.logger.info("Visualizaciones actualizadas correctamente")
            
        except Exception as e:
            self.logger.error(f"Error actualizando visualizaciones: {e}")
            
    def generate_report(self):
        """Genera un reporte textual de los resultados"""
        if not self.current_results:
            self.report_text.setPlainText("No hay resultados disponibles para generar reporte.")
            return
            
        try:
            report_lines = []
            report_lines.append("=" * 60)
            report_lines.append("REPORTE DE ANÁLISIS ESTADÍSTICO")
            report_lines.append("=" * 60)
            report_lines.append(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report_lines.append(f"Tipo de Análisis: {self.analysis_type_combo.currentText()}")
            report_lines.append("")
            
            # Resumen de datos
            if 'sample_data' in self.current_results:
                data = self.current_results['sample_data']
                report_lines.append("RESUMEN DE DATOS:")
                report_lines.append(f"- Número de muestras: {data.get('n_samples', 'N/A')}")
                report_lines.append(f"- Características analizadas: {len(data.get('feature_names', []))}")
                report_lines.append(f"- Variables: {', '.join(data.get('feature_names', []))}")
                report_lines.append("")
            
            # Resultados PCA
            if 'pca_analysis' in self.current_results:
                pca = self.current_results['pca_analysis']
                report_lines.append("ANÁLISIS DE COMPONENTES PRINCIPALES (PCA):")
                report_lines.append(f"- Componentes extraídos: {pca.get('n_components', 'N/A')}")
                
                if 'explained_variance_ratio' in pca:
                    variance_ratios = pca['explained_variance_ratio']
                    total_variance = sum(variance_ratios) * 100
                    report_lines.append(f"- Varianza explicada total: {total_variance:.2f}%")
                    
                    for i, ratio in enumerate(variance_ratios[:3]):
                        report_lines.append(f"  - PC{i+1}: {ratio*100:.2f}%")
                report_lines.append("")
            
            # Resultados de Clustering
            if 'clustering_analysis' in self.current_results:
                clustering = self.current_results['clustering_analysis']
                report_lines.append("ANÁLISIS DE CLUSTERING:")
                report_lines.append(f"- Número de clusters: {clustering.get('n_clusters', 'N/A')}")
                
                if 'cluster_labels' in clustering:
                    labels = clustering['cluster_labels']
                    unique_labels = set(labels)
                    for label in unique_labels:
                        count = labels.count(label)
                        percentage = (count / len(labels)) * 100
                        report_lines.append(f"  - Cluster {label}: {count} muestras ({percentage:.1f}%)")
                report_lines.append("")
            
            # Resultados de Correlación
            if 'correlation_analysis' in self.current_results:
                report_lines.append("ANÁLISIS DE CORRELACIÓN:")
                report_lines.append("- Matriz de correlación generada exitosamente")
                report_lines.append("- Ver visualización para detalles específicos")
                report_lines.append("")
            
            report_lines.append("=" * 60)
            report_lines.append("Fin del reporte")
            
            self.report_text.setPlainText("\n".join(report_lines))
            
        except Exception as e:
            self.logger.error(f"Error generando reporte: {e}")
            self.report_text.setPlainText(f"Error generando reporte: {str(e)}")
            
    def save_report(self):
        """Guarda el reporte actual"""
        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Guardar Reporte", 
                f"reporte_estadistico_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                "Archivos de texto (*.txt);;Todos los archivos (*)"
            )
            
            if file_path:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(self.report_text.toPlainText())
                
                QMessageBox.information(self, "Éxito", f"Reporte guardado en:\n{file_path}")
                
        except Exception as e:
            self.logger.error(f"Error guardando reporte: {e}")
            QMessageBox.critical(self, "Error", f"Error guardando reporte: {str(e)}")
            
    def load_external_data(self):
        """Carga datos externos para análisis"""
        QMessageBox.information(self, "Función en Desarrollo", 
                              "La carga de datos externos estará disponible en una próxima versión.")
        
    def export_results(self):
        """Exporta los resultados actuales"""
        if not self.current_results:
            QMessageBox.warning(self, "Sin Resultados", "No hay resultados para exportar.")
            return
            
        QMessageBox.information(self, "Función en Desarrollo", 
                              "La exportación de resultados estará disponible en una próxima versión.")
        
    def on_analysis_type_changed(self, analysis_type: str):
        """Maneja cambios en el tipo de análisis"""
        # Actualizar configuraciones según el tipo
        if analysis_type == "Solo PCA":
            self.n_clusters_spin.setEnabled(False)
        elif analysis_type == "Solo Clustering":
            self.pca_components_spin.setEnabled(False)
        else:
            self.pca_components_spin.setEnabled(True)
            self.n_clusters_spin.setEnabled(True)
            
    def on_data_source_changed(self, data_source: str):
        """Maneja cambios en la fuente de datos"""
        if data_source == "Datos de Ejemplo":
            self.data_info_label.setText("Usando datos sintéticos para demostración")
        elif data_source == "Base de Datos":
            self.data_info_label.setText("Conectar a base de datos del sistema")
        elif data_source == "Archivo CSV":
            self.data_info_label.setText("Cargar archivo CSV personalizado")
        else:
            self.data_info_label.setText("Usar resultados de análisis previo")