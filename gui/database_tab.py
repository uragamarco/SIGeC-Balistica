#!/usr/bin/env python3
"""
Pestaña de Base de Datos Balística
Sistema SIGeC-Balistica - Análisis de Cartuchos y Balas Automático

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
from typing import Dict, List, Optional, Any, Tuple, Set
from datetime import datetime, timedelta
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout,
    QLabel, QPushButton, QLineEdit, QTextEdit, QComboBox, QSpinBox,
    QCheckBox, QGroupBox, QScrollArea, QSplitter, QFrame, QSpacerItem,
    QSizePolicy, QFileDialog, QMessageBox, QProgressBar, QTabWidget,
    QListWidget, QListWidgetItem, QSlider, QDoubleSpinBox, QDateEdit,
    QTableWidget, QTableWidgetItem, QHeaderView, QTreeWidget, QTreeWidgetItem,
    QButtonGroup, QRadioButton, QCalendarWidget, QApplication, QMenu, QAction
)
from PyQt5.QtCore import Qt, pyqtSignal, QThread, QTimer, QDate, QSize
from PyQt5.QtGui import QFont, QPixmap, QIcon, QPainter, QPen, QColor, QBrush

# Importaciones para gráficos y visualización
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    import seaborn as sns
    import numpy as np
    import pandas as pd
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib/seaborn not available. Charts will be disabled.")

# Importaciones para mapas geoespaciales
try:
    import folium
    from folium.plugins import HeatMap, MarkerCluster
    import webbrowser
    import tempfile
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False
    print("Warning: folium not available. Geospatial analysis will be disabled.")

from .shared_widgets import (
    ImageDropZone, ResultCard, CollapsiblePanel, StepIndicator, 
    ProgressCard, ImageViewer
)
from .backend_integration import get_backend_integration
from .database_tab_handlers import DatabaseTabHandlers
from .database_tab_styles import apply_database_tab_styles

# Importaciones para visualización de ROI
try:
    from .visualization_widgets import ROIVisualizationWidget
    from image_processing.roi_visualizer import ROIVisualizer
    ROI_VISUALIZATION_AVAILABLE = True
except ImportError:
    ROI_VISUALIZATION_AVAILABLE = False
    print("Warning: ROI visualization not available. ROI features will be disabled.")

logger = logging.getLogger(__name__)

class InteractiveDashboardWidget(QWidget):
    """Widget de dashboard interactivo con gráficos y estadísticas avanzadas"""
    
    def __init__(self):
        super().__init__()
        self.data = []
        self.setup_ui()
        
    def setup_ui(self):
        """Configura la interfaz del dashboard"""
        layout = QVBoxLayout(self)
        
        # Título del dashboard
        title = QLabel("Dashboard de Base de Datos Balística")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #2c3e50; margin-bottom: 15px;")
        layout.addWidget(title)
        
        # Tabs para diferentes visualizaciones
        self.dashboard_tabs = QTabWidget()
        
        # Tab 1: Estadísticas generales
        self.stats_tab = QWidget()
        self.setup_stats_tab()
        self.dashboard_tabs.addTab(self.stats_tab, "📊 Estadísticas")
        
        # Tab 2: Análisis de calibres
        self.caliber_tab = QWidget()
        self.setup_caliber_tab()
        self.dashboard_tabs.addTab(self.caliber_tab, "🎯 Calibres")
        
        # Tab 3: Análisis temporal
        self.temporal_tab = QWidget()
        self.setup_temporal_tab()
        self.dashboard_tabs.addTab(self.temporal_tab, "📅 Temporal")
        
        # Tab 4: Análisis geoespacial
        if FOLIUM_AVAILABLE:
            self.geo_tab = QWidget()
            self.setup_geo_tab()
            self.dashboard_tabs.addTab(self.geo_tab, "🗺️ Geoespacial")
        
        layout.addWidget(self.dashboard_tabs)
        
    def setup_stats_tab(self):
        """Configura el tab de estadísticas generales"""
        layout = QVBoxLayout(self.stats_tab)
        
        # Controles superiores
        controls_layout = QHBoxLayout()
        
        refresh_btn = QPushButton("🔄 Actualizar")
        refresh_btn.clicked.connect(self.refresh_dashboard)
        controls_layout.addWidget(refresh_btn)
        
        export_btn = QPushButton("📤 Exportar Gráficos")
        export_btn.clicked.connect(self.export_charts)
        controls_layout.addWidget(export_btn)
        
        controls_layout.addStretch()
        layout.addLayout(controls_layout)
        
        # Área de scroll para gráficos
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_widget = QWidget()
        self.stats_layout = QVBoxLayout(scroll_widget)
        
        # Placeholder para gráficos
        if MATPLOTLIB_AVAILABLE:
            self.create_stats_charts()
        else:
            no_charts_label = QLabel("Gráficos no disponibles - matplotlib no instalado")
            no_charts_label.setAlignment(Qt.AlignCenter)
            no_charts_label.setStyleSheet("color: #e74c3c; font-style: italic; padding: 40px;")
            self.stats_layout.addWidget(no_charts_label)
        
        scroll_area.setWidget(scroll_widget)
        layout.addWidget(scroll_area)
        
    def setup_caliber_tab(self):
        """Configura el tab de análisis de calibres"""
        layout = QVBoxLayout(self.caliber_tab)
        
        # Controles de filtro
        filter_layout = QHBoxLayout()
        
        filter_layout.addWidget(QLabel("Filtrar por:"))
        
        self.caliber_filter = QComboBox()
        self.caliber_filter.addItems(["Todos los calibres", "9mm", ".40", ".45", ".380", "7.62mm"])
        self.caliber_filter.currentTextChanged.connect(self.update_caliber_charts)
        filter_layout.addWidget(self.caliber_filter)
        
        self.weapon_filter = QComboBox()
        self.weapon_filter.addItems(["Todas las armas", "Pistola", "Revólver", "Rifle", "Escopeta"])
        self.weapon_filter.currentTextChanged.connect(self.update_caliber_charts)
        filter_layout.addWidget(self.weapon_filter)
        
        filter_layout.addStretch()
        layout.addLayout(filter_layout)
        
        # Área de gráficos de calibres
        if MATPLOTLIB_AVAILABLE:
            self.caliber_canvas = self.create_caliber_charts()
            layout.addWidget(self.caliber_canvas)
        else:
            no_charts_label = QLabel("Análisis de calibres no disponible - matplotlib no instalado")
            no_charts_label.setAlignment(Qt.AlignCenter)
            no_charts_label.setStyleSheet("color: #e74c3c; font-style: italic; padding: 40px;")
            layout.addWidget(no_charts_label)
            
    def setup_temporal_tab(self):
        """Configura el tab de análisis temporal"""
        layout = QVBoxLayout(self.temporal_tab)
        
        # Controles de período
        period_layout = QHBoxLayout()
        
        period_layout.addWidget(QLabel("Período:"))
        
        self.period_combo = QComboBox()
        self.period_combo.addItems(["Último mes", "Últimos 3 meses", "Último año", "Todo el período"])
        self.period_combo.currentTextChanged.connect(self.update_temporal_charts)
        period_layout.addWidget(self.period_combo)
        
        self.grouping_combo = QComboBox()
        self.grouping_combo.addItems(["Por día", "Por semana", "Por mes"])
        self.grouping_combo.currentTextChanged.connect(self.update_temporal_charts)
        period_layout.addWidget(self.grouping_combo)
        
        period_layout.addStretch()
        layout.addLayout(period_layout)
        
        # Área de gráficos temporales
        if MATPLOTLIB_AVAILABLE:
            self.temporal_canvas = self.create_temporal_charts()
            layout.addWidget(self.temporal_canvas)
        else:
            no_charts_label = QLabel("Análisis temporal no disponible - matplotlib no instalado")
            no_charts_label.setAlignment(Qt.AlignCenter)
            no_charts_label.setStyleSheet("color: #e74c3c; font-style: italic; padding: 40px;")
            layout.addWidget(no_charts_label)
            
    def setup_geo_tab(self):
        """Configura el tab de análisis geoespacial"""
        layout = QVBoxLayout(self.geo_tab)
        
        # Controles del mapa
        map_controls = QHBoxLayout()
        
        self.map_type_combo = QComboBox()
        self.map_type_combo.addItems(["Mapa de puntos", "Mapa de calor", "Clusters"])
        self.map_type_combo.currentTextChanged.connect(self.update_geo_map)
        map_controls.addWidget(self.map_type_combo)
        
        generate_map_btn = QPushButton("🗺️ Generar Mapa")
        generate_map_btn.clicked.connect(self.generate_geo_map)
        map_controls.addWidget(generate_map_btn)
        
        export_map_btn = QPushButton("📤 Exportar Mapa")
        export_map_btn.clicked.connect(self.export_geo_map)
        map_controls.addWidget(export_map_btn)
        
        map_controls.addStretch()
        layout.addLayout(map_controls)
        
        # Información del mapa
        self.map_info_label = QLabel("Haga clic en 'Generar Mapa' para visualizar la distribución geográfica de evidencias")
        self.map_info_label.setAlignment(Qt.AlignCenter)
        self.map_info_label.setStyleSheet("color: #7f8c8d; font-style: italic; padding: 20px;")
        layout.addWidget(self.map_info_label)
        
    def create_stats_charts(self):
        """Crea los gráficos de estadísticas generales"""
        # Configurar estilo de seaborn
        sns.set_style("whitegrid")
        plt.style.use('seaborn-v0_8')
        
        # Crear figura con subplots
        fig = Figure(figsize=(12, 8))
        canvas = FigureCanvas(fig)
        
        # Datos de ejemplo para gráficos
        evidence_types = ['Casquillos', 'Balas', 'Fragmentos', 'Otros']
        evidence_counts = [45, 32, 18, 5]
        
        quality_ranges = ['Excelente\n(90-100%)', 'Buena\n(70-89%)', 'Regular\n(50-69%)', 'Baja\n(<50%)']
        quality_counts = [25, 35, 28, 12]
        
        # Gráfico de torta - Tipos de evidencia
        ax1 = fig.add_subplot(2, 2, 1)
        colors = ['#3498db', '#e74c3c', '#f39c12', '#95a5a6']
        wedges, texts, autotexts = ax1.pie(evidence_counts, labels=evidence_types, autopct='%1.1f%%', 
                                          colors=colors, startangle=90)
        ax1.set_title('Distribución por Tipo de Evidencia', fontsize=12, fontweight='bold')
        
        # Gráfico de barras - Calidad de evidencias
        ax2 = fig.add_subplot(2, 2, 2)
        bars = ax2.bar(quality_ranges, quality_counts, color=['#27ae60', '#f39c12', '#e67e22', '#e74c3c'])
        ax2.set_title('Distribución por Calidad', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Cantidad de Evidencias')
        
        # Agregar valores en las barras
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{int(height)}', ha='center', va='bottom')
        
        # Gráfico de línea - Tendencia temporal (ejemplo)
        ax3 = fig.add_subplot(2, 2, 3)
        months = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun']
        cases = [12, 15, 18, 14, 22, 19]
        ax3.plot(months, cases, marker='o', linewidth=2, markersize=6, color='#3498db')
        ax3.set_title('Casos por Mes', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Número de Casos')
        ax3.grid(True, alpha=0.3)
        
        # Gráfico de barras horizontales - Top calibres
        ax4 = fig.add_subplot(2, 2, 4)
        calibers = ['9mm', '.40', '.45', '.380', '7.62mm']
        caliber_counts = [28, 22, 15, 12, 8]
        bars = ax4.barh(calibers, caliber_counts, color='#9b59b6')
        ax4.set_title('Top 5 Calibres', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Cantidad de Evidencias')
        
        # Agregar valores en las barras
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax4.text(width + 0.5, bar.get_y() + bar.get_height()/2.,
                    f'{int(width)}', ha='left', va='center')
        
        fig.tight_layout(pad=3.0)
        self.stats_layout.addWidget(canvas)
        
    def create_caliber_charts(self):
        """Crea los gráficos de análisis de calibres"""
        fig = Figure(figsize=(12, 6))
        canvas = FigureCanvas(fig)
        
        # Datos de ejemplo para calibres
        calibers = ['9mm', '.40 S&W', '.45 ACP', '.380 ACP', '7.62mm', '.22 LR', '.357 Mag']
        counts = [35, 28, 22, 18, 15, 12, 8]
        
        # Gráfico de barras principal
        ax1 = fig.add_subplot(1, 2, 1)
        bars = ax1.bar(calibers, counts, color='#3498db', alpha=0.8)
        ax1.set_title('Distribución de Calibres', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Cantidad de Evidencias')
        ax1.tick_params(axis='x', rotation=45)
        
        # Agregar valores en las barras
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{int(height)}', ha='center', va='bottom')
        
        # Gráfico de dona para porcentajes
        ax2 = fig.add_subplot(1, 2, 2)
        colors = plt.cm.Set3(np.linspace(0, 1, len(calibers)))
        wedges, texts, autotexts = ax2.pie(counts, labels=calibers, autopct='%1.1f%%',
                                          colors=colors, pctdistance=0.85)
        
        # Crear el agujero central para hacer una dona
        centre_circle = plt.Circle((0,0), 0.70, fc='white')
        ax2.add_artist(centre_circle)
        ax2.set_title('Porcentaje por Calibre', fontsize=14, fontweight='bold')
        
        fig.tight_layout()
        return canvas
        
    def create_temporal_charts(self):
        """Crea los gráficos de análisis temporal"""
        fig = Figure(figsize=(12, 8))
        canvas = FigureCanvas(fig)
        
        # Datos de ejemplo temporales
        dates = pd.date_range('2024-01-01', periods=30, freq='D')
        daily_cases = np.random.poisson(3, 30)  # Simulación de casos diarios
        
        # Gráfico de línea temporal
        ax1 = fig.add_subplot(2, 1, 1)
        ax1.plot(dates, daily_cases, marker='o', linewidth=1.5, markersize=4, color='#e74c3c')
        ax1.set_title('Casos Diarios - Último Mes', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Número de Casos')
        ax1.grid(True, alpha=0.3)
        
        # Formatear fechas en el eje x
        ax1.tick_params(axis='x', rotation=45)
        
        # Gráfico de barras por día de la semana
        ax2 = fig.add_subplot(2, 1, 2)
        weekdays = ['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Dom']
        weekday_counts = [15, 18, 22, 19, 25, 8, 5]  # Ejemplo de distribución semanal
        
        bars = ax2.bar(weekdays, weekday_counts, color='#f39c12', alpha=0.8)
        ax2.set_title('Distribución por Día de la Semana', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Promedio de Casos')
        
        # Agregar valores en las barras
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{int(height)}', ha='center', va='bottom')
        
        fig.tight_layout()
        return canvas
        
    def refresh_dashboard(self):
        """Actualiza todos los gráficos del dashboard"""
        if MATPLOTLIB_AVAILABLE:
            # Limpiar layouts existentes
            self.clear_layout(self.stats_layout)
            self.create_stats_charts()
            
    def update_caliber_charts(self):
        """Actualiza los gráficos de calibres según filtros"""
        # Implementar lógica de filtrado
        pass
        
    def update_temporal_charts(self):
        """Actualiza los gráficos temporales según período seleccionado"""
        # Implementar lógica de actualización temporal
        pass
        
    def update_geo_map(self):
        """Actualiza el mapa geoespacial según tipo seleccionado"""
        # Implementar lógica de actualización del mapa
        pass
        
    def generate_geo_map(self):
        """Genera el mapa geoespacial con las evidencias"""
        if not FOLIUM_AVAILABLE:
            QMessageBox.warning(self, "Error", "Folium no está disponible para análisis geoespacial")
            return
            
        # Datos de ejemplo con coordenadas
        evidence_locations = [
            {"lat": -34.6037, "lon": -58.3816, "case": "Caso 001", "type": "Casquillo", "caliber": "9mm"},
            {"lat": -34.6118, "lon": -58.3960, "case": "Caso 002", "type": "Bala", "caliber": ".40"},
            {"lat": -34.5989, "lon": -58.3731, "case": "Caso 003", "type": "Casquillo", "caliber": "9mm"},
            {"lat": -34.6092, "lon": -58.3842, "case": "Caso 004", "type": "Fragmento", "caliber": ".45"},
            {"lat": -34.6156, "lon": -58.3919, "case": "Caso 005", "type": "Casquillo", "caliber": "9mm"},
        ]
        
        # Crear mapa centrado en Buenos Aires (ejemplo)
        center_lat = sum([loc["lat"] for loc in evidence_locations]) / len(evidence_locations)
        center_lon = sum([loc["lon"] for loc in evidence_locations]) / len(evidence_locations)
        
        m = folium.Map(location=[center_lat, center_lon], zoom_start=13)
        
        map_type = self.map_type_combo.currentText()
        
        if map_type == "Mapa de puntos":
            # Agregar marcadores individuales
            for loc in evidence_locations:
                popup_text = f"Caso: {loc['case']}<br>Tipo: {loc['type']}<br>Calibre: {loc['caliber']}"
                folium.Marker(
                    [loc["lat"], loc["lon"]],
                    popup=popup_text,
                    icon=folium.Icon(color='red' if loc['type'] == 'Casquillo' else 'blue')
                ).add_to(m)
                
        elif map_type == "Clusters":
            # Usar MarkerCluster
            marker_cluster = MarkerCluster().add_to(m)
            for loc in evidence_locations:
                popup_text = f"Caso: {loc['case']}<br>Tipo: {loc['type']}<br>Calibre: {loc['caliber']}"
                folium.Marker(
                    [loc["lat"], loc["lon"]],
                    popup=popup_text
                ).add_to(marker_cluster)
                
        elif map_type == "Mapa de calor":
            # Crear mapa de calor
            heat_data = [[loc["lat"], loc["lon"]] for loc in evidence_locations]
            HeatMap(heat_data).add_to(m)
        
        # Guardar mapa temporal y abrirlo
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.html')
        m.save(temp_file.name)
        
        # Abrir en navegador
        webbrowser.open('file://' + temp_file.name)
        
        self.map_info_label.setText(f"Mapa generado con {len(evidence_locations)} evidencias. Abierto en navegador.")
        
    def export_geo_map(self):
        """Exporta el mapa geoespacial"""
        filename, _ = QFileDialog.getSaveFileName(self, "Exportar Mapa", "mapa_evidencias.html", "HTML Files (*.html)")
        if filename:
            self.generate_geo_map()  # Regenerar y guardar en ubicación específica
            
    def export_charts(self):
        """Exporta los gráficos como imágenes"""
        filename, _ = QFileDialog.getSaveFileName(self, "Exportar Gráficos", "dashboard_charts.png", "PNG Files (*.png)")
        if filename and MATPLOTLIB_AVAILABLE:
            # Implementar exportación de gráficos
            QMessageBox.information(self, "Exportar", f"Gráficos exportados a {filename}")
            
    def clear_layout(self, layout):
        """Limpia un layout de todos sus widgets"""
        while layout.count():
            child = layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
                
    def update_data(self, data: List[Dict]):
        """Actualiza los datos del dashboard"""
        self.data = data
        self.refresh_dashboard()

class CaseManagementWidget(QWidget):
    """Widget para gestión avanzada de casos y agrupación de evidencias"""
    
    caseSelected = pyqtSignal(dict)
    evidenceGrouped = pyqtSignal(list, str)
    
    def __init__(self):
        super().__init__()
        self.cases = {}
        self.selected_evidences = set()
        self.setup_ui()
        
    def setup_ui(self):
        """Configura la interfaz de gestión de casos"""
        layout = QVBoxLayout(self)
        
        # Título
        title = QLabel("Gestión de Casos y Evidencias")
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: #2c3e50; margin-bottom: 10px;")
        layout.addWidget(title)
        
        # Controles superiores
        controls_layout = QHBoxLayout()
        
        # Crear nuevo caso
        self.new_case_btn = QPushButton("➕ Nuevo Caso")
        self.new_case_btn.clicked.connect(self.create_new_case)
        controls_layout.addWidget(self.new_case_btn)
        
        # Buscar casos
        self.case_search = QLineEdit()
        self.case_search.setPlaceholderText("Buscar casos...")
        self.case_search.textChanged.connect(self.filter_cases)
        controls_layout.addWidget(self.case_search)
        
        # Filtro por estado
        self.status_filter = QComboBox()
        self.status_filter.addItems(["Todos", "Activo", "Cerrado", "En revisión", "Archivado"])
        self.status_filter.currentTextChanged.connect(self.filter_cases)
        controls_layout.addWidget(self.status_filter)
        
        controls_layout.addStretch()
        layout.addLayout(controls_layout)
        
        # Splitter principal
        main_splitter = QSplitter(Qt.Horizontal)
        
        # Panel izquierdo - Lista de casos
        cases_panel = QFrame()
        cases_panel.setObjectName("casesPanel")
        cases_layout = QVBoxLayout(cases_panel)
        
        cases_label = QLabel("Casos")
        cases_label.setStyleSheet("font-weight: bold; margin-bottom: 5px;")
        cases_layout.addWidget(cases_label)
        
        # Tree widget para casos jerárquicos
        self.cases_tree = QTreeWidget()
        self.cases_tree.setHeaderLabels(["Caso", "Estado", "Evidencias", "Fecha"])
        self.cases_tree.itemClicked.connect(self.on_case_selected)
        self.cases_tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.cases_tree.customContextMenuRequested.connect(self.show_case_context_menu)
        cases_layout.addWidget(self.cases_tree)
        
        main_splitter.addWidget(cases_panel)
        
        # Panel derecho - Detalles del caso y evidencias
        details_panel = QFrame()
        details_panel.setObjectName("detailsPanel")
        details_layout = QVBoxLayout(details_panel)
        
        # Información del caso seleccionado
        case_info_group = QGroupBox("Información del Caso")
        case_info_layout = QFormLayout(case_info_group)
        
        self.case_id_label = QLabel("No seleccionado")
        self.case_name_label = QLabel("No seleccionado")
        self.case_status_label = QLabel("No seleccionado")
        self.case_date_label = QLabel("No seleccionado")
        self.case_description_text = QTextEdit()
        self.case_description_text.setMaximumHeight(80)
        
        case_info_layout.addRow("ID:", self.case_id_label)
        case_info_layout.addRow("Nombre:", self.case_name_label)
        case_info_layout.addRow("Estado:", self.case_status_label)
        case_info_layout.addRow("Fecha:", self.case_date_label)
        case_info_layout.addRow("Descripción:", self.case_description_text)
        
        details_layout.addWidget(case_info_group)
        
        # Evidencias del caso
        evidences_group = QGroupBox("Evidencias del Caso")
        evidences_layout = QVBoxLayout(evidences_group)
        
        # Controles de evidencias
        evidence_controls = QHBoxLayout()
        
        self.add_evidence_btn = QPushButton("➕ Agregar Evidencia")
        self.add_evidence_btn.clicked.connect(self.add_evidence_to_case)
        self.add_evidence_btn.setEnabled(False)
        evidence_controls.addWidget(self.add_evidence_btn)
        
        self.remove_evidence_btn = QPushButton("➖ Quitar Evidencia")
        self.remove_evidence_btn.clicked.connect(self.remove_evidence_from_case)
        self.remove_evidence_btn.setEnabled(False)
        evidence_controls.addWidget(self.remove_evidence_btn)
        
        evidence_controls.addStretch()
        evidences_layout.addLayout(evidence_controls)
        
        # Lista de evidencias
        self.evidences_list = QListWidget()
        self.evidences_list.setSelectionMode(QListWidget.MultiSelection)
        self.evidences_list.itemSelectionChanged.connect(self.on_evidence_selection_changed)
        evidences_layout.addWidget(self.evidences_list)
        
        details_layout.addWidget(evidences_group)
        
        main_splitter.addWidget(details_panel)
        
        # Configurar proporciones
        main_splitter.setSizes([300, 500])
        layout.addWidget(main_splitter)
        
        # Cargar casos de ejemplo
        self.load_sample_cases()
        
    def create_new_case(self):
        """Crea un nuevo caso"""
        from PyQt5.QtWidgets import QInputDialog
        
        case_name, ok = QInputDialog.getText(self, "Nuevo Caso", "Nombre del caso:")
        if ok and case_name:
            case_id = f"CASE_{len(self.cases) + 1:04d}"
            new_case = {
                "id": case_id,
                "name": case_name,
                "status": "Activo",
                "date": datetime.now().strftime("%Y-%m-%d"),
                "description": "",
                "evidences": []
            }
            
            self.cases[case_id] = new_case
            self.refresh_cases_tree()
            
    def load_sample_cases(self):
        """Carga casos de ejemplo"""
        sample_cases = [
            {
                "id": "CASE_0001",
                "name": "Homicidio Av. Corrientes",
                "status": "Activo",
                "date": "2024-01-15",
                "description": "Caso de homicidio con arma de fuego en Av. Corrientes",
                "evidences": ["EVD_001", "EVD_002", "EVD_003"]
            },
            {
                "id": "CASE_0002", 
                "name": "Robo a mano armada - Palermo",
                "status": "En revisión",
                "date": "2024-01-20",
                "description": "Robo con arma de fuego en zona de Palermo",
                "evidences": ["EVD_004", "EVD_005"]
            },
            {
                "id": "CASE_0003",
                "name": "Disparo accidental",
                "status": "Cerrado",
                "date": "2024-01-10",
                "description": "Investigación de disparo accidental",
                "evidences": ["EVD_006"]
            }
        ]
        
        for case in sample_cases:
            self.cases[case["id"]] = case
            
        self.refresh_cases_tree()
        
    def refresh_cases_tree(self):
        """Actualiza el árbol de casos"""
        self.cases_tree.clear()
        
        for case_id, case_data in self.cases.items():
            case_item = QTreeWidgetItem([
                case_data["name"],
                case_data["status"],
                str(len(case_data["evidences"])),
                case_data["date"]
            ])
            case_item.setData(0, Qt.UserRole, case_id)
            
            # Agregar evidencias como hijos
            for evidence_id in case_data["evidences"]:
                evidence_item = QTreeWidgetItem([f"📄 {evidence_id}", "", "", ""])
                case_item.addChild(evidence_item)
                
            self.cases_tree.addTopLevelItem(case_item)
            
        self.cases_tree.expandAll()
        
    def filter_cases(self):
        """Filtra los casos según criterios de búsqueda"""
        search_text = self.case_search.text().lower()
        status_filter = self.status_filter.currentText()
        
        for i in range(self.cases_tree.topLevelItemCount()):
            item = self.cases_tree.topLevelItem(i)
            case_id = item.data(0, Qt.UserRole)
            case_data = self.cases[case_id]
            
            # Aplicar filtros
            text_match = (search_text in case_data["name"].lower() or 
                         search_text in case_data["id"].lower())
            status_match = (status_filter == "Todos" or 
                           case_data["status"] == status_filter)
            
            item.setHidden(not (text_match and status_match))
            
    def on_case_selected(self, item, column):
        """Maneja la selección de un caso"""
        case_id = item.data(0, Qt.UserRole)
        if case_id and case_id in self.cases:
            case_data = self.cases[case_id]
            
            # Actualizar información del caso
            self.case_id_label.setText(case_data["id"])
            self.case_name_label.setText(case_data["name"])
            self.case_status_label.setText(case_data["status"])
            self.case_date_label.setText(case_data["date"])
            self.case_description_text.setPlainText(case_data["description"])
            
            # Actualizar lista de evidencias
            self.evidences_list.clear()
            for evidence_id in case_data["evidences"]:
                self.evidences_list.addItem(evidence_id)
                
            # Habilitar controles
            self.add_evidence_btn.setEnabled(True)
            
            # Emitir señal
            self.caseSelected.emit(case_data)
            
    def show_case_context_menu(self, position):
        """Muestra menú contextual para casos"""
        item = self.cases_tree.itemAt(position)
        if item:
            menu = QMenu(self)
            
            edit_action = QAction("✏️ Editar caso", self)
            edit_action.triggered.connect(lambda: self.edit_case(item))
            menu.addAction(edit_action)
            
            delete_action = QAction("🗑️ Eliminar caso", self)
            delete_action.triggered.connect(lambda: self.delete_case(item))
            menu.addAction(delete_action)
            
            menu.addSeparator()
            
            export_action = QAction("📤 Exportar caso", self)
            export_action.triggered.connect(lambda: self.export_case(item))
            menu.addAction(export_action)
            
            menu.exec_(self.cases_tree.mapToGlobal(position))
            
    def edit_case(self, item):
        """Edita un caso"""
        # Implementar diálogo de edición
        pass
        
    def delete_case(self, item):
        """Elimina un caso"""
        case_id = item.data(0, Qt.UserRole)
        if case_id:
            reply = QMessageBox.question(self, "Confirmar", 
                                       f"¿Eliminar el caso {case_id}?",
                                       QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.Yes:
                del self.cases[case_id]
                self.refresh_cases_tree()
                
    def export_case(self, item):
        """Exporta un caso"""
        # Implementar exportación de caso
        pass
        
    def add_evidence_to_case(self):
        """Agrega evidencia al caso seleccionado"""
        # Implementar diálogo para seleccionar evidencias
        pass
        
    def remove_evidence_from_case(self):
        """Quita evidencia del caso seleccionado"""
        selected_items = self.evidences_list.selectedItems()
        if selected_items:
            for item in selected_items:
                self.evidences_list.takeItem(self.evidences_list.row(item))
                
    def on_evidence_selection_changed(self):
        """Maneja cambios en la selección de evidencias"""
        has_selection = len(self.evidences_list.selectedItems()) > 0
        self.remove_evidence_btn.setEnabled(has_selection)

class BatchActionsWidget(QWidget):
    """Widget para acciones por lotes en múltiples evidencias"""
    
    batchActionExecuted = pyqtSignal(str, list)
    
    def __init__(self):
        super().__init__()
        self.selected_items = []
        self.setup_ui()
        
    def setup_ui(self):
        """Configura la interfaz de acciones por lotes"""
        layout = QVBoxLayout(self)
        
        # Título
        title = QLabel("Acciones por Lotes")
        title.setStyleSheet("font-size: 14px; font-weight: bold; color: #2c3e50; margin-bottom: 10px;")
        layout.addWidget(title)
        
        # Información de selección
        self.selection_info = QLabel("No hay elementos seleccionados")
        self.selection_info.setStyleSheet("color: #7f8c8d; font-style: italic;")
        layout.addWidget(self.selection_info)
        
        # Acciones disponibles
        actions_group = QGroupBox("Acciones Disponibles")
        actions_layout = QVBoxLayout(actions_group)
        
        # Exportar seleccionados
        export_layout = QHBoxLayout()
        self.export_btn = QPushButton("📤 Exportar Seleccionados")
        self.export_btn.clicked.connect(self.export_selected)
        self.export_btn.setEnabled(False)
        export_layout.addWidget(self.export_btn)
        
        self.export_format = QComboBox()
        self.export_format.addItems(["JSON", "CSV", "PDF", "Excel"])
        export_layout.addWidget(self.export_format)
        
        actions_layout.addLayout(export_layout)
        
        # Agregar a caso
        case_layout = QHBoxLayout()
        self.add_to_case_btn = QPushButton("📁 Agregar a Caso")
        self.add_to_case_btn.clicked.connect(self.add_to_case)
        self.add_to_case_btn.setEnabled(False)
        case_layout.addWidget(self.add_to_case_btn)
        
        self.case_combo = QComboBox()
        self.case_combo.addItems(["Seleccionar caso...", "CASE_0001", "CASE_0002", "Nuevo caso..."])
        case_layout.addWidget(self.case_combo)
        
        actions_layout.addLayout(case_layout)
        
        # Cambiar estado
        status_layout = QHBoxLayout()
        self.change_status_btn = QPushButton("🔄 Cambiar Estado")
        self.change_status_btn.clicked.connect(self.change_status)
        self.change_status_btn.setEnabled(False)
        status_layout.addWidget(self.change_status_btn)
        
        self.status_combo = QComboBox()
        self.status_combo.addItems(["Pendiente", "En análisis", "Analizado", "Archivado"])
        status_layout.addWidget(self.status_combo)
        
        actions_layout.addLayout(status_layout)
        
        # Agregar etiquetas
        tags_layout = QHBoxLayout()
        self.add_tags_btn = QPushButton("🏷️ Agregar Etiquetas")
        self.add_tags_btn.clicked.connect(self.add_tags)
        self.add_tags_btn.setEnabled(False)
        tags_layout.addWidget(self.add_tags_btn)
        
        self.tags_input = QLineEdit()
        self.tags_input.setPlaceholderText("Etiquetas separadas por comas...")
        tags_layout.addWidget(self.tags_input)
        
        actions_layout.addLayout(tags_layout)
        
        # Eliminar seleccionados
        self.delete_btn = QPushButton("🗑️ Eliminar Seleccionados")
        self.delete_btn.clicked.connect(self.delete_selected)
        self.delete_btn.setEnabled(False)
        self.delete_btn.setStyleSheet("QPushButton { background-color: #e74c3c; color: white; }")
        actions_layout.addWidget(self.delete_btn)
        
        layout.addWidget(actions_group)
        
        # Progreso de acciones
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        layout.addStretch()
        
    def update_selection(self, selected_items: List[Dict]):
        """Actualiza la selección de elementos"""
        self.selected_items = selected_items
        count = len(selected_items)
        
        if count == 0:
            self.selection_info.setText("No hay elementos seleccionados")
            self._enable_actions(False)
        else:
            self.selection_info.setText(f"{count} elemento(s) seleccionado(s)")
            self._enable_actions(True)
            
    def _enable_actions(self, enabled: bool):
        """Habilita o deshabilita las acciones"""
        self.export_btn.setEnabled(enabled)
        self.add_to_case_btn.setEnabled(enabled)
        self.change_status_btn.setEnabled(enabled)
        self.add_tags_btn.setEnabled(enabled)
        self.delete_btn.setEnabled(enabled)
        
    def export_selected(self):
        """Exporta los elementos seleccionados"""
        if not self.selected_items:
            return
            
        format_type = self.export_format.currentText()
        filename, _ = QFileDialog.getSaveFileName(
            self, 
            f"Exportar {len(self.selected_items)} elementos",
            f"evidencias_seleccionadas.{format_type.lower()}",
            f"{format_type} Files (*.{format_type.lower()})"
        )
        
        if filename:
            self._execute_batch_action("export", {"filename": filename, "format": format_type})
            
    def add_to_case(self):
        """Agrega elementos seleccionados a un caso"""
        if not self.selected_items:
            return
            
        case_id = self.case_combo.currentText()
        if case_id == "Seleccionar caso...":
            QMessageBox.warning(self, "Error", "Debe seleccionar un caso")
            return
            
        if case_id == "Nuevo caso...":
            from PyQt5.QtWidgets import QInputDialog
            case_name, ok = QInputDialog.getText(self, "Nuevo Caso", "Nombre del caso:")
            if ok and case_name:
                case_id = f"CASE_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                # Crear nuevo caso
                
        self._execute_batch_action("add_to_case", {"case_id": case_id})
        
    def change_status(self):
        """Cambia el estado de elementos seleccionados"""
        if not self.selected_items:
            return
            
        new_status = self.status_combo.currentText()
        self._execute_batch_action("change_status", {"status": new_status})
        
    def add_tags(self):
        """Agrega etiquetas a elementos seleccionados"""
        if not self.selected_items:
            return
            
        tags_text = self.tags_input.text().strip()
        if not tags_text:
            QMessageBox.warning(self, "Error", "Debe ingresar al menos una etiqueta")
            return
            
        tags = [tag.strip() for tag in tags_text.split(",") if tag.strip()]
        self._execute_batch_action("add_tags", {"tags": tags})
        
    def delete_selected(self):
        """Elimina elementos seleccionados"""
        if not self.selected_items:
            return
            
        reply = QMessageBox.question(
            self, 
            "Confirmar Eliminación",
            f"¿Está seguro de eliminar {len(self.selected_items)} elemento(s)?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self._execute_batch_action("delete", {})
            
    def _execute_batch_action(self, action: str, params: Dict):
        """Ejecuta una acción por lotes"""
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        # Simular progreso
        for i in range(101):
            self.progress_bar.setValue(i)
            QApplication.processEvents()
            
        self.progress_bar.setVisible(False)
        
        # Emitir señal
        self.batchActionExecuted.emit(action, self.selected_items)
        
        # Mostrar mensaje de éxito
        QMessageBox.information(self, "Éxito", f"Acción '{action}' ejecutada en {len(self.selected_items)} elemento(s)")

class AdvancedSearchWidget(QWidget):
    """Widget para búsqueda avanzada con filtros por facetas y etiquetado"""
    
    searchExecuted = pyqtSignal(dict)
    
    def __init__(self):
        super().__init__()
        self.setup_ui()
        
    def setup_ui(self):
        """Configura la interfaz de búsqueda avanzada"""
        layout = QVBoxLayout(self)
        
        # Título
        title = QLabel("Búsqueda Avanzada y Filtros por Facetas")
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: #2c3e50; margin-bottom: 10px;")
        layout.addWidget(title)
        
        # Búsqueda por texto libre
        text_search_group = QGroupBox("Búsqueda por Texto")
        text_search_layout = QVBoxLayout(text_search_group)
        
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Buscar en todos los campos...")
        text_search_layout.addWidget(self.search_input)
        
        # Opciones de búsqueda de texto
        text_options_layout = QHBoxLayout()
        
        self.case_sensitive_cb = QCheckBox("Sensible a mayúsculas")
        text_options_layout.addWidget(self.case_sensitive_cb)
        
        self.whole_words_cb = QCheckBox("Palabras completas")
        text_options_layout.addWidget(self.whole_words_cb)
        
        self.regex_cb = QCheckBox("Expresión regular")
        text_options_layout.addWidget(self.regex_cb)
        
        text_options_layout.addStretch()
        text_search_layout.addLayout(text_options_layout)
        
        layout.addWidget(text_search_group)
        
        # Filtros por facetas
        facets_group = QGroupBox("Filtros por Facetas")
        facets_layout = QGridLayout(facets_group)
        
        # Tipo de evidencia
        facets_layout.addWidget(QLabel("Tipo:"), 0, 0)
        self.evidence_type_combo = QComboBox()
        self.evidence_type_combo.addItems(["Todos", "Casquillo", "Bala", "Fragmento", "Otro"])
        facets_layout.addWidget(self.evidence_type_combo, 0, 1)
        
        # Calibre
        facets_layout.addWidget(QLabel("Calibre:"), 0, 2)
        self.caliber_combo = QComboBox()
        self.caliber_combo.addItems(["Todos", "9mm", ".40", ".45", ".380", "7.62mm", ".22"])
        facets_layout.addWidget(self.caliber_combo, 0, 3)
        
        # Estado
        facets_layout.addWidget(QLabel("Estado:"), 1, 0)
        self.status_combo = QComboBox()
        self.status_combo.addItems(["Todos", "Pendiente", "En análisis", "Analizado", "Archivado"])
        facets_layout.addWidget(self.status_combo, 1, 1)
        
        # Calidad
        facets_layout.addWidget(QLabel("Calidad:"), 1, 2)
        quality_layout = QHBoxLayout()
        self.min_quality_spin = QSpinBox()
        self.min_quality_spin.setRange(0, 100)
        self.min_quality_spin.setSuffix("%")
        quality_layout.addWidget(self.min_quality_spin)
        quality_layout.addWidget(QLabel("a"))
        self.max_quality_spin = QSpinBox()
        self.max_quality_spin.setRange(0, 100)
        self.max_quality_spin.setValue(100)
        self.max_quality_spin.setSuffix("%")
        quality_layout.addWidget(self.max_quality_spin)
        facets_layout.addLayout(quality_layout, 1, 3)
        
        layout.addWidget(facets_group)
        
        # Filtros temporales
        temporal_group = QGroupBox("Filtros Temporales")
        temporal_layout = QGridLayout(temporal_group)
        
        temporal_layout.addWidget(QLabel("Desde:"), 0, 0)
        self.date_from = QDateEdit()
        self.date_from.setDate(QDate.currentDate().addDays(-30))
        self.date_from.setCalendarPopup(True)
        temporal_layout.addWidget(self.date_from, 0, 1)
        
        temporal_layout.addWidget(QLabel("Hasta:"), 0, 2)
        self.date_to = QDateEdit()
        self.date_to.setDate(QDate.currentDate())
        self.date_to.setCalendarPopup(True)
        temporal_layout.addWidget(self.date_to, 0, 3)
        
        # Presets temporales
        presets_layout = QHBoxLayout()
        
        last_week_btn = QPushButton("Última semana")
        last_week_btn.clicked.connect(lambda: self.set_date_preset(7))
        presets_layout.addWidget(last_week_btn)
        
        last_month_btn = QPushButton("Último mes")
        last_month_btn.clicked.connect(lambda: self.set_date_preset(30))
        presets_layout.addWidget(last_month_btn)
        
        last_year_btn = QPushButton("Último año")
        last_year_btn.clicked.connect(lambda: self.set_date_preset(365))
        presets_layout.addWidget(last_year_btn)
        
        presets_layout.addStretch()
        temporal_layout.addLayout(presets_layout, 1, 0, 1, 4)
        
        layout.addWidget(temporal_group)
        
        # Etiquetas
        tags_group = QGroupBox("Filtros por Etiquetas")
        tags_layout = QVBoxLayout(tags_group)
        
        # Entrada de etiquetas
        tags_input_layout = QHBoxLayout()
        
        self.tags_input = QLineEdit()
        self.tags_input.setPlaceholderText("Etiquetas separadas por comas...")
        tags_input_layout.addWidget(self.tags_input)
        
        self.tags_mode_combo = QComboBox()
        self.tags_mode_combo.addItems(["Cualquier etiqueta (OR)", "Todas las etiquetas (AND)", "Excluir etiquetas (NOT)"])
        tags_input_layout.addWidget(self.tags_mode_combo)
        
        tags_layout.addLayout(tags_input_layout)
        
        # Etiquetas populares
        popular_tags_label = QLabel("Etiquetas populares:")
        popular_tags_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        tags_layout.addWidget(popular_tags_label)
        
        popular_tags_layout = QHBoxLayout()
        popular_tags = ["homicidio", "robo", "accidental", "suicidio", "evidencia-clave", "alta-calidad"]
        
        for tag in popular_tags:
            tag_btn = QPushButton(f"#{tag}")
            tag_btn.setCheckable(True)
            tag_btn.clicked.connect(lambda checked, t=tag: self.toggle_tag(t, checked))
            tag_btn.setStyleSheet("""
                QPushButton {
                    border: 1px solid #3498db;
                    border-radius: 15px;
                    padding: 5px 10px;
                    background-color: white;
                    color: #3498db;
                }
                QPushButton:checked {
                    background-color: #3498db;
                    color: white;
                }
            """)
            popular_tags_layout.addWidget(tag_btn)
            
        popular_tags_layout.addStretch()
        tags_layout.addLayout(popular_tags_layout)
        
        layout.addWidget(tags_group)
        
        # Botones de acción
        actions_layout = QHBoxLayout()
        
        self.search_btn = QPushButton("🔍 Buscar")
        self.search_btn.clicked.connect(self.execute_search)
        self.search_btn.setObjectName("primaryButton")
        actions_layout.addWidget(self.search_btn)
        
        self.clear_btn = QPushButton("🗑️ Limpiar Filtros")
        self.clear_btn.clicked.connect(self.clear_filters)
        actions_layout.addWidget(self.clear_btn)
        
        self.save_search_btn = QPushButton("💾 Guardar Búsqueda")
        self.save_search_btn.clicked.connect(self.save_search)
        actions_layout.addWidget(self.save_search_btn)
        
        actions_layout.addStretch()
        layout.addLayout(actions_layout)
        
        # Búsquedas guardadas
        saved_searches_group = QGroupBox("Búsquedas Guardadas")
        saved_searches_layout = QHBoxLayout(saved_searches_group)
        
        self.saved_searches_combo = QComboBox()
        self.saved_searches_combo.addItems(["Seleccionar búsqueda guardada...", "Casos activos alta calidad", "Evidencias último mes"])
        saved_searches_layout.addWidget(self.saved_searches_combo)
        
        load_search_btn = QPushButton("📂 Cargar")
        load_search_btn.clicked.connect(self.load_saved_search)
        saved_searches_layout.addWidget(load_search_btn)
        
        delete_search_btn = QPushButton("🗑️")
        delete_search_btn.clicked.connect(self.delete_saved_search)
        saved_searches_layout.addWidget(delete_search_btn)
        
        saved_searches_layout.addStretch()
        layout.addWidget(saved_searches_group)
        
    def set_date_preset(self, days: int):
        """Establece un preset de fechas"""
        self.date_from.setDate(QDate.currentDate().addDays(-days))
        self.date_to.setDate(QDate.currentDate())
        
    def toggle_tag(self, tag: str, checked: bool):
        """Alterna una etiqueta en la búsqueda"""
        current_tags = self.tags_input.text()
        tags_list = [t.strip() for t in current_tags.split(",") if t.strip()]
        
        if checked and tag not in tags_list:
            tags_list.append(tag)
        elif not checked and tag in tags_list:
            tags_list.remove(tag)
            
        self.tags_input.setText(", ".join(tags_list))
        
    def execute_search(self):
        """Ejecuta la búsqueda con los filtros actuales"""
        search_params = {
            "text": self.search_input.text(),
            "case_sensitive": self.case_sensitive_cb.isChecked(),
            "whole_words": self.whole_words_cb.isChecked(),
            "regex": self.regex_cb.isChecked(),
            "evidence_type": self.evidence_type_combo.currentText(),
            "caliber": self.caliber_combo.currentText(),
            "status": self.status_combo.currentText(),
            "min_quality": self.min_quality_spin.value(),
            "max_quality": self.max_quality_spin.value(),
            "date_from": self.date_from.date().toString("yyyy-MM-dd"),
            "date_to": self.date_to.date().toString("yyyy-MM-dd"),
            "tags": [t.strip() for t in self.tags_input.text().split(",") if t.strip()],
            "tags_mode": self.tags_mode_combo.currentText()
        }
        
        self.searchExecuted.emit(search_params)
        
    def clear_filters(self):
        """Limpia todos los filtros"""
        self.search_input.clear()
        self.case_sensitive_cb.setChecked(False)
        self.whole_words_cb.setChecked(False)
        self.regex_cb.setChecked(False)
        self.evidence_type_combo.setCurrentIndex(0)
        self.caliber_combo.setCurrentIndex(0)
        self.status_combo.setCurrentIndex(0)
        self.min_quality_spin.setValue(0)
        self.max_quality_spin.setValue(100)
        self.date_from.setDate(QDate.currentDate().addDays(-30))
        self.date_to.setDate(QDate.currentDate())
        self.tags_input.clear()
        self.tags_mode_combo.setCurrentIndex(0)
        
        # Desmarcar botones de etiquetas
        for child in self.findChildren(QPushButton):
            if child.text().startswith("#"):
                child.setChecked(False)
                
    def save_search(self):
        """Guarda la búsqueda actual"""
        from PyQt5.QtWidgets import QInputDialog
        
        name, ok = QInputDialog.getText(self, "Guardar Búsqueda", "Nombre de la búsqueda:")
        if ok and name:
            # Implementar guardado de búsqueda
            QMessageBox.information(self, "Guardado", f"Búsqueda '{name}' guardada exitosamente")
            
    def load_saved_search(self):
        """Carga una búsqueda guardada"""
        search_name = self.saved_searches_combo.currentText()
        if search_name != "Seleccionar búsqueda guardada...":
            # Implementar carga de búsqueda guardada
            QMessageBox.information(self, "Cargado", f"Búsqueda '{search_name}' cargada")
            
    def delete_saved_search(self):
        """Elimina una búsqueda guardada"""
        search_name = self.saved_searches_combo.currentText()
        if search_name != "Seleccionar búsqueda guardada...":
            reply = QMessageBox.question(self, "Confirmar", f"¿Eliminar la búsqueda '{search_name}'?")
            if reply == QMessageBox.Yes:
                # Implementar eliminación
                QMessageBox.information(self, "Eliminado", f"Búsqueda '{search_name}' eliminada")

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
        
        # Tab 4: Regiones de Interés (ROI)
        if ROI_VISUALIZATION_AVAILABLE:
            self.roi_tab = QWidget()
            self.setup_roi_tab()
            self.viz_tabs.addTab(self.roi_tab, "🎯 ROI")
        
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
        
    def setup_roi_tab(self):
        """Configura el tab de visualización de ROI"""
        layout = QVBoxLayout(self.roi_tab)
        
        # Widget de visualización de ROI
        self.roi_visualization_widget = ROIVisualizationWidget()
        layout.addWidget(self.roi_visualization_widget)
        
        # Panel de controles para ROI
        controls_group = QGroupBox("Controles de ROI")
        controls_layout = QHBoxLayout(controls_group)
        
        # Botón para generar visualización de ROI
        self.generate_roi_btn = QPushButton("🎯 Generar Visualización ROI")
        self.generate_roi_btn.setEnabled(False)
        self.generate_roi_btn.clicked.connect(self.generate_roi_visualization)
        controls_layout.addWidget(self.generate_roi_btn)
        
        # Botón para exportar ROI
        self.export_roi_btn = QPushButton("💾 Exportar ROI")
        self.export_roi_btn.setEnabled(False)
        self.export_roi_btn.clicked.connect(self.export_roi_visualization)
        controls_layout.addWidget(self.export_roi_btn)
        
        controls_layout.addStretch()
        layout.addWidget(controls_group)
        
        # Información de ROI detectadas
        self.roi_info_label = QLabel("Seleccione un elemento para visualizar sus regiones de interés")
        self.roi_info_label.setAlignment(Qt.AlignCenter)
        self.roi_info_label.setStyleSheet("color: #7f8c8d; font-style: italic; padding: 10px;")
        layout.addWidget(self.roi_info_label)
        
    def generate_roi_visualization(self):
        """Genera visualización de ROI para el elemento seleccionado"""
        if not hasattr(self, 'current_item_data') or not self.current_item_data:
            return
            
        try:
            # Obtener datos de ROI del elemento
            roi_regions = self.current_item_data.get('roi_regions', [])
            image_path = self.current_item_data.get('image_path', '')
            evidence_type = self.current_item_data.get('evidence_type', 'unknown')
            
            if not roi_regions:
                # Si no hay ROI, intentar generar datos sintéticos para demostración
                roi_regions = self._generate_sample_roi_data()
                
            # Actualizar información
            self.roi_info_label.setText(f"Visualizando {len(roi_regions)} regiones de interés detectadas")
            
            # Generar visualización
            self.roi_visualization_widget.visualize_roi_regions(
                image_path, roi_regions, evidence_type
            )
            
            # Habilitar exportación
            self.export_roi_btn.setEnabled(True)
            
        except Exception as e:
            logger.error(f"Error generando visualización ROI: {str(e)}")
            self.roi_info_label.setText(f"Error: {str(e)}")
            
    def export_roi_visualization(self):
        """Exporta la visualización de ROI actual"""
        try:
            # Obtener directorio de exportación
            export_dir = QFileDialog.getExistingDirectory(
                self, "Seleccionar directorio de exportación"
            )
            
            if export_dir:
                # Exportar visualizaciones
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                export_path = os.path.join(export_dir, f"roi_export_{timestamp}")
                
                # Aquí se implementaría la lógica de exportación
                QMessageBox.information(
                    self, "Exportación Completada",
                    f"Visualizaciones ROI exportadas a:\n{export_path}"
                )
                
        except Exception as e:
            logger.error(f"Error exportando ROI: {str(e)}")
            QMessageBox.warning(self, "Error", f"Error exportando ROI: {str(e)}")
            
    def _generate_sample_roi_data(self):
        """Genera datos de ROI de ejemplo para demostración"""
        return [
            {
                'id': 1,
                'bbox': [100, 100, 200, 200],
                'confidence': 0.85,
                'detection_method': 'enhanced_watershed',
                'area': 10000,
                'centroid': [150, 150]
            },
            {
                'id': 2,
                'bbox': [300, 150, 400, 250],
                'confidence': 0.92,
                'detection_method': 'circle_detection',
                'area': 10000,
                'centroid': [350, 200]
            },
            {
                'id': 3,
                'bbox': [200, 300, 350, 400],
                'confidence': 0.78,
                'detection_method': 'contour_detection',
                'area': 15000,
                'centroid': [275, 350]
            }
        ]
        
    def display_roi_information(self, item_data: Dict):
        """Muestra información de ROI para un elemento seleccionado"""
        self.current_item_data = item_data
        
        # Habilitar controles
        self.generate_roi_btn.setEnabled(True)
        
        # Actualizar información
        roi_count = len(item_data.get('roi_regions', []))
        if roi_count > 0:
            self.roi_info_label.setText(f"Elemento seleccionado tiene {roi_count} regiones de interés detectadas")
        else:
            self.roi_info_label.setText("Elemento seleccionado - Generar visualización para detectar ROI")


class DatabaseTab(QWidget, DatabaseTabHandlers):
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
        
        # Aplicar estilos CSS mejorados
        apply_database_tab_styles(self)
        
        # Cargar estadísticas iniciales
        self._load_database_stats()
        
        logger.info("DatabaseTab balística inicializada")
    
    def _setup_ui(self):
        """Configura la interfaz de usuario balística"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # Crear pestañas principales
        self.main_tabs = QTabWidget()
        
        # Pestaña de Dashboard
        dashboard_tab = self._create_dashboard_tab()
        self.main_tabs.addTab(dashboard_tab, "📊 Dashboard")
        
        # Pestaña de Búsqueda y Resultados
        search_tab = self._create_search_tab()
        self.main_tabs.addTab(search_tab, "🔍 Búsqueda")
        
        # Pestaña de Gestión de Casos
        cases_tab = self._create_cases_tab()
        self.main_tabs.addTab(cases_tab, "📁 Casos")
        
        # Pestaña de Acciones por Lotes
        batch_tab = self._create_batch_tab()
        self.main_tabs.addTab(batch_tab, "⚡ Acciones por Lotes")
        
        layout.addWidget(self.main_tabs)
    
    def _create_dashboard_tab(self):
        """Crea la pestaña del dashboard"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Crear dashboard interactivo
        self.dashboard_widget = InteractiveDashboardWidget()
        layout.addWidget(self.dashboard_widget)
        
        return widget
    
    def _create_search_tab(self):
        """Crea la pestaña de búsqueda"""
        widget = QWidget()
        layout = QHBoxLayout(widget)
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
        
        return widget
    
    def _create_cases_tab(self):
        """Crea la pestaña de gestión de casos"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Crear widget de gestión de casos
        self.case_management_widget = CaseManagementWidget()
        layout.addWidget(self.case_management_widget)
        
        return widget
    
    def _create_batch_tab(self):
        """Crea la pestaña de acciones por lotes"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Crear widget de acciones por lotes
        self.batch_actions_widget = BatchActionsWidget()
        layout.addWidget(self.batch_actions_widget)
        
        return widget
    
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
        
        # Integrar búsqueda avanzada en el panel de búsqueda
        advanced_search_panel = CollapsiblePanel("Búsqueda Avanzada")
        self.advanced_search_widget = AdvancedSearchWidget()
        advanced_search_panel.add_content_widget(self.advanced_search_widget)
        layout.addWidget(advanced_search_panel)
        
        # Botones de acción
        actions_layout = QVBoxLayout()
        
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
        
        # Conectar señales de los nuevos widgets
        if hasattr(self, 'advanced_search_widget'):
            self.advanced_search_widget.searchExecuted.connect(self._handle_advanced_search)
        
        if hasattr(self, 'case_management_widget'):
            self.case_management_widget.caseSelected.connect(self._handle_case_selection)
            self.case_management_widget.evidenceGrouped.connect(self._handle_evidence_grouping)
        
        if hasattr(self, 'batch_actions_widget'):
            self.batch_actions_widget.batchActionExecuted.connect(self._handle_batch_action)
    
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
        
        # Actualizar visualización de ROI si está disponible
        if hasattr(self, 'ballistic_viz') and self.ballistic_viz:
            self.ballistic_viz.display_roi_information(result)
        
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
    from .styles import apply_SIGeC_Balistica_theme
    apply_SIGeC_Balistica_theme(app)
    
    # Crear y mostrar pestaña
    tab = DatabaseTab()
    tab.show()
    
    sys.exit(app.exec_())