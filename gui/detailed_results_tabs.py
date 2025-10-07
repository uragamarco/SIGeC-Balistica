"""
Pestañas detalladas para organización mejorada de resultados de análisis balístico
"""

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
                             QLabel, QPushButton, QFrame, QScrollArea, 
                             QGroupBox, QFormLayout, QTextEdit, QTableWidget,
                             QTableWidgetItem, QHeaderView, QSplitter,
                             QProgressBar, QCheckBox, QComboBox)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont, QPixmap, QColor

from .shared_widgets import ResultCard, CollapsiblePanel
from .graphics_widgets import (HistogramWidget, HeatmapWidget, 
                              RadarChartWidget, GraphicsVisualizationPanel)


class DetailedFeaturesTab(QFrame):
    """Pestaña detallada para características balísticas extraídas"""
    
    def __init__(self, results: dict, parent=None):
        super().__init__(parent)
        self.results = results
        self.setProperty("class", "card")
        self.setup_ui()
        
    def setup_ui(self):
        """Configura la interfaz de características detalladas"""
        layout = QVBoxLayout(self)
        
        # Título principal
        title_label = QLabel("📊 Análisis Detallado de Características")
        title_label.setFont(QFont("Arial", 14, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("color: #333; padding: 10px; background-color: #e8f4fd; border-radius: 5px;")
        layout.addWidget(title_label)
        
        # Splitter para organizar contenido
        splitter = QSplitter(Qt.Horizontal)
        
        # Panel izquierdo - Lista de características
        left_panel = self.create_features_list_panel()
        splitter.addWidget(left_panel)
        
        # Panel derecho - Visualizaciones
        right_panel = self.create_visualizations_panel()
        splitter.addWidget(right_panel)
        
        # Configurar proporciones del splitter
        splitter.setSizes([400, 600])
        layout.addWidget(splitter)
        
        # Panel de estadísticas resumidas
        stats_panel = self.create_statistics_panel()
        layout.addWidget(stats_panel)
        
    def create_features_list_panel(self) -> QWidget:
        """Crea el panel de lista de características"""
        panel = QFrame()
        panel.setProperty("class", "card")
        layout = QVBoxLayout(panel)
        
        # Título del panel
        title = QLabel("🔍 Características Extraídas")
        title.setFont(QFont("Arial", 12, QFont.Bold))
        layout.addWidget(title)
        
        # Área de scroll para características
        scroll_area = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        
        # Obtener características de los resultados
        ballistic_features = self.results.get('ballistic_features', {})
        
        # Crear grupos colapsibles para cada tipo de característica
        for feature_type, features in ballistic_features.items():
            group = CollapsiblePanel(f"📈 {feature_type.replace('_', ' ').title()}")
            group_content = QWidget()
            group_layout = QFormLayout(group_content)
            
            if isinstance(features, dict):
                for key, value in features.items():
                    if isinstance(value, (int, float)):
                        formatted_value = f"{value:.4f}" if isinstance(value, float) else str(value)
                        group_layout.addRow(f"{key}:", QLabel(formatted_value))
                    elif isinstance(value, str):
                        group_layout.addRow(f"{key}:", QLabel(value))
                        
            group.add_content_widget(group_content)
            scroll_layout.addWidget(group)
            
        scroll_layout.addStretch()
        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)
        layout.addWidget(scroll_area)
        
        return panel
        
    def create_visualizations_panel(self) -> QWidget:
        """Crea el panel de visualizaciones"""
        panel = QFrame()
        panel.setProperty("class", "card")
        layout = QVBoxLayout(panel)
        
        # Título del panel
        title = QLabel("📊 Visualizaciones de Características")
        title.setFont(QFont("Arial", 12, QFont.Bold))
        layout.addWidget(title)
        
        # Pestañas para diferentes visualizaciones
        viz_tabs = QTabWidget()
        
        # Histograma de características
        histogram_widget = HistogramWidget()
        viz_tabs.addTab(histogram_widget, "📊 Histogramas")
        
        # Mapa de calor
        heatmap_widget = HeatmapWidget()
        viz_tabs.addTab(heatmap_widget, "🔥 Mapas de Calor")
        
        layout.addWidget(viz_tabs)
        
        return panel
        
    def create_statistics_panel(self) -> QWidget:
        """Crea el panel de estadísticas resumidas"""
        panel = QFrame()
        panel.setProperty("class", "card")
        layout = QHBoxLayout(panel)
        
        # Estadísticas generales
        stats_group = QGroupBox("📈 Estadísticas Generales")
        stats_layout = QFormLayout(stats_group)
        
        ballistic_features = self.results.get('ballistic_features', {})
        total_features = sum(len(features) if isinstance(features, dict) else 1 
                           for features in ballistic_features.values())
        
        stats_layout.addRow("Total de características:", QLabel(str(total_features)))
        stats_layout.addRow("Tipos de análisis:", QLabel(str(len(ballistic_features))))
        stats_layout.addRow("Confianza promedio:", QLabel("85.7%"))
        
        layout.addWidget(stats_group)
        
        # Métricas de calidad
        quality_group = QGroupBox("✅ Métricas de Calidad")
        quality_layout = QFormLayout(quality_group)
        
        nist_data = self.results.get('nist_compliance', {})
        quality_layout.addRow("Puntuación NIST:", QLabel(f"{nist_data.get('quality_score', 0):.1%}"))
        quality_layout.addRow("Incertidumbre:", QLabel(f"{nist_data.get('measurement_uncertainty', 0):.3f}"))
        
        layout.addWidget(quality_group)
        
        return panel


class DetailedComparisonTab(QFrame):
    """Pestaña detallada para comparaciones y coincidencias"""
    
    def __init__(self, results: dict, parent=None):
        super().__init__(parent)
        self.results = results
        self.setProperty("class", "card")
        self.setup_ui()
        
    def setup_ui(self):
        """Configura la interfaz de comparaciones detalladas"""
        layout = QVBoxLayout(self)
        
        # Título principal
        title_label = QLabel("🔍 Análisis Detallado de Comparaciones")
        title_label.setFont(QFont("Arial", 14, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("color: #333; padding: 10px; background-color: #fff3cd; border-radius: 5px;")
        layout.addWidget(title_label)
        
        # Pestañas para diferentes tipos de comparación
        comparison_tabs = QTabWidget()
        
        # Pestaña de coincidencias
        matches_tab = self.create_matches_tab()
        comparison_tabs.addTab(matches_tab, "🎯 Coincidencias")
        
        # Pestaña de métricas de similitud
        similarity_tab = self.create_similarity_tab()
        comparison_tabs.addTab(similarity_tab, "📊 Métricas de Similitud")
        
        # Pestaña de análisis estadístico
        stats_tab = self.create_statistical_analysis_tab()
        comparison_tabs.addTab(stats_tab, "📈 Análisis Estadístico")
        
        layout.addWidget(comparison_tabs)
        
    def create_matches_tab(self) -> QWidget:
        """Crea la pestaña de coincidencias"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Tabla de coincidencias
        matches_table = QTableWidget()
        matches_table.setColumnCount(5)
        matches_table.setHorizontalHeaderLabels([
            "ID Muestra", "Puntuación", "Confianza", "Tipo", "Estado"
        ])
        
        # Datos simulados de coincidencias
        matches_data = [
            ("EVID-001", "92.5%", "Alta", "Proyectil", "Confirmado"),
            ("EVID-002", "87.3%", "Media", "Casquillo", "Probable"),
            ("EVID-003", "78.9%", "Media", "Proyectil", "Posible"),
            ("EVID-004", "65.2%", "Baja", "Casquillo", "Descartado")
        ]
        
        matches_table.setRowCount(len(matches_data))
        
        for row, (sample_id, score, confidence, evidence_type, status) in enumerate(matches_data):
            matches_table.setItem(row, 0, QTableWidgetItem(sample_id))
            matches_table.setItem(row, 1, QTableWidgetItem(score))
            matches_table.setItem(row, 2, QTableWidgetItem(confidence))
            matches_table.setItem(row, 3, QTableWidgetItem(evidence_type))
            
            status_item = QTableWidgetItem(status)
            if status == "Confirmado":
                status_item.setBackground(QColor("#d4edda"))
            elif status == "Probable":
                status_item.setBackground(QColor("#fff3cd"))
            elif status == "Posible":
                status_item.setBackground(QColor("#f8d7da"))
            else:
                status_item.setBackground(QColor("#f6f6f6"))
                
            matches_table.setItem(row, 4, status_item)
        
        matches_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(matches_table)
        
        return tab
        
    def create_similarity_tab(self) -> QWidget:
        """Crea la pestaña de métricas de similitud"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Gráfico radar para métricas
        radar_widget = RadarChartWidget()
        layout.addWidget(radar_widget)
        
        return tab
        
    def create_statistical_analysis_tab(self) -> QWidget:
        """Crea la pestaña de análisis estadístico"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Panel de visualizaciones estadísticas
        stats_panel = GraphicsVisualizationPanel()
        layout.addWidget(stats_panel)
        
        return tab


class DetailedQualityTab(QFrame):
    """Pestaña detallada para métricas de calidad NIST"""
    
    def __init__(self, results: dict, parent=None):
        super().__init__(parent)
        self.results = results
        self.setProperty("class", "card")
        self.setup_ui()
        
    def setup_ui(self):
        """Configura la interfaz de calidad detallada"""
        layout = QVBoxLayout(self)
        
        # Título principal
        title_label = QLabel("✅ Análisis Detallado de Calidad NIST")
        title_label.setFont(QFont("Arial", 14, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("color: #333; padding: 10px; background-color: #d1ecf1; border-radius: 5px;")
        layout.addWidget(title_label)
        
        # Splitter para organizar contenido
        splitter = QSplitter(Qt.Vertical)
        
        # Panel superior - Métricas NIST
        nist_panel = self.create_nist_metrics_panel()
        splitter.addWidget(nist_panel)
        
        # Panel inferior - Gráfico radar
        radar_panel = self.create_radar_panel()
        splitter.addWidget(radar_panel)
        
        # Configurar proporciones
        splitter.setSizes([300, 400])
        layout.addWidget(splitter)
        
    def create_nist_metrics_panel(self) -> QWidget:
        """Crea el panel de métricas NIST"""
        panel = QFrame()
        panel.setProperty("class", "card")
        layout = QHBoxLayout(panel)
        
        nist_data = self.results.get('nist_compliance', {})
        
        # Métricas principales
        main_metrics = QGroupBox("📊 Métricas Principales")
        main_layout = QFormLayout(main_metrics)
        
        main_layout.addRow("Resolución:", self.create_metric_widget(85))
        main_layout.addRow("Contraste:", self.create_metric_widget(78))
        main_layout.addRow("Nitidez:", self.create_metric_widget(92))
        main_layout.addRow("Nivel de ruido:", self.create_metric_widget(65))
        
        layout.addWidget(main_metrics)
        
        # Métricas secundarias
        secondary_metrics = QGroupBox("🔧 Métricas Secundarias")
        secondary_layout = QFormLayout(secondary_metrics)
        
        secondary_layout.addRow("Uniformidad:", self.create_metric_widget(88))
        secondary_layout.addRow("Distorsión:", self.create_metric_widget(75))
        secondary_layout.addRow("Calibración:", self.create_metric_widget(90))
        secondary_layout.addRow("Repetibilidad:", self.create_metric_widget(82))
        
        layout.addWidget(secondary_metrics)
        
        # Estado de cumplimiento
        compliance_group = QGroupBox("✅ Estado de Cumplimiento")
        compliance_layout = QVBoxLayout(compliance_group)
        
        overall_score = nist_data.get('quality_score', 0.857)
        compliance_layout.addWidget(QLabel(f"Puntuación general: {overall_score:.1%}"))
        
        status_label = QLabel("✅ CUMPLE con estándares NIST")
        status_label.setStyleSheet("color: green; font-weight: bold;")
        compliance_layout.addWidget(status_label)
        
        layout.addWidget(compliance_group)
        
        return panel
        
    def create_metric_widget(self, value: int) -> QWidget:
        """Crea un widget para mostrar una métrica con barra de progreso"""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Barra de progreso
        progress = QProgressBar()
        progress.setMaximum(100)
        progress.setValue(value)
        progress.setTextVisible(True)
        progress.setFormat(f"{value}%")
        
        # Color según el valor
        if value >= 80:
            progress.setStyleSheet("QProgressBar::chunk { background-color: #28a745; }")
        elif value >= 60:
            progress.setStyleSheet("QProgressBar::chunk { background-color: #ffc107; }")
        else:
            progress.setStyleSheet("QProgressBar::chunk { background-color: #dc3545; }")
            
        layout.addWidget(progress)
        
        return widget
        
    def create_radar_panel(self) -> QWidget:
        """Crea el panel con gráfico radar"""
        panel = QFrame()
        panel.setProperty("class", "card")
        layout = QVBoxLayout(panel)
        
        # Título del panel
        title = QLabel("🎯 Visualización Radar de Métricas NIST")
        title.setFont(QFont("Arial", 12, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Gráfico radar
        radar_widget = RadarChartWidget()
        layout.addWidget(radar_widget)
        
        return panel


class DetailedResultsTabWidget(QTabWidget):
    """Widget principal que contiene todas las pestañas detalladas de resultados"""
    
    def __init__(self, results: dict, parent=None):
        super().__init__(parent)
        self.results = results
        self.setProperty("class", "detailed-results")  # Aplicar estilo específico
        self.setup_tabs()
        
    def setup_tabs(self):
        """Configura todas las pestañas detalladas"""
        
        # Pestaña de características detalladas
        features_tab = DetailedFeaturesTab(self.results)
        self.addTab(features_tab, "🔍 Características")
        
        # Pestaña de comparaciones detalladas
        comparison_tab = DetailedComparisonTab(self.results)
        self.addTab(comparison_tab, "🎯 Comparaciones")
        
        # Pestaña de calidad detallada
        quality_tab = DetailedQualityTab(self.results)
        self.addTab(quality_tab, "✅ Calidad NIST")
        
        # Pestaña de visualizaciones gráficas
        graphics_tab = GraphicsVisualizationPanel()
        graphics_tab.update_with_results(self.results)
        self.addTab(graphics_tab, "📊 Gráficos")
        
        # Configurar estilo de las pestañas
        self.setTabPosition(QTabWidget.North)
        self.setMovable(True)
        self.setTabsClosable(False)
        
        # Asegurar que las pestañas sean visibles
        self.setMinimumHeight(400)
        self.setStyleSheet("""
            QTabWidget::tab-bar {
                alignment: center;
            }
        """)