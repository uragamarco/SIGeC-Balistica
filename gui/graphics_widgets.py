"""
Widgets especializados para visualizaciones gráficas de análisis balístico
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
# Importaciones de visualización - Comentadas temporalmente para pruebas
try:
    import seaborn as sns
except Exception:
    # Mock temporal para seaborn cuando no está disponible
    class MockSeaborn:
        def set_style(self, *args, **kwargs):
            pass
        def heatmap(self, *args, **kwargs):
            pass
        def scatterplot(self, *args, **kwargs):
            pass

    sns = MockSeaborn()
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QFrame, QComboBox, QCheckBox,
                             QGroupBox, QTabWidget, QScrollArea, QSizePolicy)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont


class HistogramWidget(QFrame):
    """Widget para mostrar histogramas de características balísticas"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setProperty("class", "card")
        self.setup_ui()
        
    def setup_ui(self):
        """Configura la interfaz del histograma"""
        layout = QVBoxLayout(self)
        
        # Título y controles
        header_layout = QHBoxLayout()
        
        title_label = QLabel("Histograma de Características")
        title_label.setFont(QFont("Arial", 12, QFont.Bold))
        header_layout.addWidget(title_label)
        
        header_layout.addStretch()
        
        # Selector de característica
        self.feature_combo = QComboBox()
        self.feature_combo.addItems([
            "Distribución de Intensidad",
            "Gradientes de Bordes", 
            "Texturas Locales",
            "Patrones de Estrías",
            "Calidad de Imagen"
        ])
        self.feature_combo.currentTextChanged.connect(self.update_histogram)
        header_layout.addWidget(self.feature_combo)
        
        # Botón de exportar
        self.export_btn = QPushButton("Exportar")
        self.export_btn.clicked.connect(self.export_histogram)
        header_layout.addWidget(self.export_btn)
        
        layout.addLayout(header_layout)
        
        # Canvas de matplotlib
        self.figure = Figure(figsize=(8, 4), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.canvas)
        
        # Información estadística
        self.stats_label = QLabel("Estadísticas: -")
        self.stats_label.setStyleSheet("color: #666; font-size: 11px; padding: 5px;")
        layout.addWidget(self.stats_label)
        
    def update_histogram(self, feature_name=None):
        """Actualiza el histograma con datos simulados"""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        # Generar datos simulados según la característica
        if not feature_name:
            feature_name = self.feature_combo.currentText()
            
        if "Intensidad" in feature_name:
            data = np.random.normal(128, 40, 1000).clip(0, 255)
            ax.set_xlabel("Intensidad de Pixel")
            ax.set_ylabel("Frecuencia")
            color = '#2E86AB'
        elif "Gradientes" in feature_name:
            data = np.random.exponential(2, 1000)
            ax.set_xlabel("Magnitud del Gradiente")
            ax.set_ylabel("Frecuencia")
            color = '#A23B72'
        elif "Texturas" in feature_name:
            data = np.random.gamma(2, 2, 1000)
            ax.set_xlabel("Índice de Textura")
            ax.set_ylabel("Frecuencia")
            color = '#F18F01'
        elif "Estrías" in feature_name:
            data = np.random.beta(2, 5, 1000) * 100
            ax.set_xlabel("Intensidad de Estrías")
            ax.set_ylabel("Frecuencia")
            color = '#C73E1D'
        else:  # Calidad
            data = np.random.normal(0.75, 0.15, 1000).clip(0, 1)
            ax.set_xlabel("Índice de Calidad")
            ax.set_ylabel("Frecuencia")
            color = '#7209B7'
            
        # Crear histograma
        n, bins, patches = ax.hist(data, bins=30, alpha=0.7, color=color, edgecolor='black', linewidth=0.5)
        
        # Agregar línea de media
        mean_val = np.mean(data)
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Media: {mean_val:.2f}')
        
        ax.set_title(f"Histograma - {feature_name}", fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Actualizar estadísticas
        std_val = np.std(data)
        min_val = np.min(data)
        max_val = np.max(data)
        
        stats_text = f"Media: {mean_val:.2f} | Desv. Est.: {std_val:.2f} | Min: {min_val:.2f} | Max: {max_val:.2f}"
        self.stats_label.setText(f"Estadísticas: {stats_text}")
        
        self.figure.tight_layout()
        self.canvas.draw()
        
    def export_histogram(self):
        """Exporta el histograma actual"""
        from PyQt5.QtWidgets import QFileDialog
        filename, _ = QFileDialog.getSaveFileName(
            self, "Exportar Histograma", 
            f"histograma_{self.feature_combo.currentText().lower().replace(' ', '_')}.png",
            "PNG Files (*.png);;PDF Files (*.pdf)"
        )
        if filename:
            self.figure.savefig(filename, dpi=300, bbox_inches='tight')


class HeatmapWidget(QFrame):
    """Widget para mostrar mapas de calor de análisis balístico"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setProperty("class", "card")
        # Datos externos opcionales para renderizar matrices reales
        self.external_matrix = None
        self.external_title = None
        self.external_colormap = None
        # Indicador de estado (fuente de datos)
        self.status_label = None
        self.setup_ui()
        
    def setup_ui(self):
        """Configura la interfaz del mapa de calor"""
        layout = QVBoxLayout(self)
        
        # Título y controles
        header_layout = QHBoxLayout()
        
        title_label = QLabel("Mapa de Calor de Análisis")
        title_label.setFont(QFont("Arial", 12, QFont.Bold))
        header_layout.addWidget(title_label)
        
        header_layout.addStretch()
        
        # Selector de tipo de mapa
        self.map_type_combo = QComboBox()
        self.map_type_combo.addItems([
            "Calidad de Imagen",
            "Densidad de Características",
            "Correlación Espacial",
            "Mapa de Confianza",
            "Distribución de Estrías"
        ])
        self.map_type_combo.currentTextChanged.connect(self.update_heatmap)
        header_layout.addWidget(self.map_type_combo)
        
        # Selector de paleta de colores
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems(['viridis', 'plasma', 'inferno', 'magma', 'hot', 'cool', 'jet'])
        self.colormap_combo.currentTextChanged.connect(self.update_heatmap)
        header_layout.addWidget(self.colormap_combo)
        
        # Botón de exportar
        self.export_btn = QPushButton("Exportar")
        self.export_btn.clicked.connect(self.export_heatmap)
        header_layout.addWidget(self.export_btn)
        
        layout.addLayout(header_layout)
        
        # Canvas de matplotlib
        self.figure = Figure(figsize=(8, 6), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.canvas)

        # Indicador en el borde inferior para fuente de datos
        self.status_label = QLabel("Fuente: Simulada")
        self.status_label.setStyleSheet(
            "color: #666; font-size: 11px; padding: 4px 8px;"
            "border-top: 1px solid #ddd;"
        )
        layout.addWidget(self.status_label)
        
    def set_heatmap_data(self, matrix, title: str = None, colormap: str = None):
        """Establece datos externos para el mapa de calor.
        Args:
            matrix: Matriz 2D (lista de listas o np.ndarray).
            title: Título opcional.
            colormap: Nombre del colormap opcional.
        """
        try:
            import numpy as _np
            self.external_matrix = _np.array(matrix) if matrix is not None else None
        except Exception:
            self.external_matrix = None
        self.external_title = title
        self.external_colormap = colormap
        self.update_heatmap()

    def clear_external_data(self):
        """Limpia los datos externos inyectados."""
        self.external_matrix = None
        self.external_title = None
        self.external_colormap = None
        self.update_heatmap()

    def update_heatmap(self):
        """Actualiza el mapa de calor con datos externos o simulados"""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        if self.external_matrix is not None:
            # Usar matriz externa
            data = self.external_matrix
            if getattr(data, 'ndim', 2) != 2:
                data = np.atleast_2d(data)
            colormap = self.external_colormap or self.colormap_combo.currentText()
            title = self.external_title or "Mapa de Calor"
            # Actualizar indicador de estado
            if self.status_label is not None:
                self.status_label.setText("Fuente: Externa (matriz real)")
                self.status_label.setStyleSheet(
                    "color: #2E7D32; font-size: 11px; padding: 4px 8px;"
                    "border-top: 1px solid #ddd;"
                )
        else:
            # Generar datos simulados según el tipo de mapa
            map_type = self.map_type_combo.currentText()
            colormap = self.colormap_combo.currentText()
            
            if "Calidad" in map_type:
                # Simular mapa de calidad con zonas de alta y baja calidad
                data = np.random.random((50, 50))
                # Agregar algunas zonas de alta calidad
                data[20:30, 20:30] += 0.5
                data[10:15, 35:40] += 0.3
                title = "Mapa de Calidad de Imagen"
                
            elif "Densidad" in map_type:
                # Simular densidad de características
                x, y = np.meshgrid(np.linspace(0, 10, 50), np.linspace(0, 10, 50))
                data = np.sin(x) * np.cos(y) + np.random.normal(0, 0.1, (50, 50))
                title = "Densidad de Características"
                
            elif "Correlación" in map_type:
                # Simular correlación espacial
                data = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], (50, 50))[:, :, 0]
                title = "Correlación Espacial"
                
            elif "Confianza" in map_type:
                # Simular mapa de confianza
                center_x, center_y = 25, 25
                x, y = np.meshgrid(np.arange(50), np.arange(50))
                data = np.exp(-((x - center_x)**2 + (y - center_y)**2) / 200)
                title = "Mapa de Confianza"
                
            else:  # Estrías
                # Simular distribución de estrías
                data = np.random.exponential(1, (50, 50))
                # Agregar patrones lineales para simular estrías
                for i in range(0, 50, 5):
                    data[i:i+2, :] *= 1.5
                title = "Distribución de Estrías"
            # Actualizar indicador de estado
            if self.status_label is not None:
                self.status_label.setText("Fuente: Simulada")
                self.status_label.setStyleSheet(
                    "color: #666; font-size: 11px; padding: 4px 8px;"
                    "border-top: 1px solid #ddd;"
                )
        
        # Crear mapa de calor
        im = ax.imshow(data, cmap=colormap, aspect='auto', interpolation='bilinear')
        
        # Agregar barra de colores
        cbar = self.figure.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Intensidad', rotation=270, labelpad=15)
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Coordenada X')
        ax.set_ylabel('Coordenada Y')
        
        self.figure.tight_layout()
        self.canvas.draw()
        
    def export_heatmap(self):
        """Exporta el mapa de calor actual"""
        from PyQt5.QtWidgets import QFileDialog
        filename, _ = QFileDialog.getSaveFileName(
            self, "Exportar Mapa de Calor", 
            f"heatmap_{self.map_type_combo.currentText().lower().replace(' ', '_')}.png",
            "PNG Files (*.png);;PDF Files (*.pdf)"
        )
        if filename:
            self.figure.savefig(filename, dpi=300, bbox_inches='tight')


class RadarChartWidget(QFrame):
    """Widget para mostrar gráficos radar de métricas NIST"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setProperty("class", "card")
        self.setup_ui()
        
    def setup_ui(self):
        """Configura la interfaz del gráfico radar"""
        layout = QVBoxLayout(self)
        
        # Título y controles
        header_layout = QHBoxLayout()
        
        title_label = QLabel("Gráfico Radar - Métricas NIST")
        title_label.setFont(QFont("Arial", 12, QFont.Bold))
        header_layout.addWidget(title_label)
        
        header_layout.addStretch()
        
        # Checkbox para mostrar comparación
        self.comparison_cb = QCheckBox("Mostrar Comparación")
        self.comparison_cb.stateChanged.connect(self.update_radar)
        header_layout.addWidget(self.comparison_cb)
        
        # Botón de exportar
        self.export_btn = QPushButton("Exportar")
        self.export_btn.clicked.connect(self.export_radar)
        header_layout.addWidget(self.export_btn)
        
        layout.addLayout(header_layout)
        
        # Canvas de matplotlib
        self.figure = Figure(figsize=(8, 8), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.canvas)
        
        # Leyenda de métricas
        legend_label = QLabel(
            "Métricas NIST: Resolución, Contraste, Nitidez, Ruido, "
            "Uniformidad, Distorsión, Calibración, Repetibilidad"
        )
        legend_label.setStyleSheet("color: #666; font-size: 10px; padding: 5px;")
        legend_label.setWordWrap(True)
        layout.addWidget(legend_label)
        
    def update_radar(self):
        """Actualiza el gráfico radar con métricas NIST simuladas"""
        self.figure.clear()
        
        # Métricas NIST simuladas
        categories = [
            'Resolución', 'Contraste', 'Nitidez', 'Ruido',
            'Uniformidad', 'Distorsión', 'Calibración', 'Repetibilidad'
        ]
        
        # Valores simulados (0-100)
        values1 = [85, 78, 92, 65, 88, 75, 90, 82]  # Muestra actual
        values2 = [80, 85, 88, 70, 85, 80, 85, 85]  # Referencia/comparación
        
        # Número de variables
        N = len(categories)
        
        # Ángulos para cada eje
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Completar el círculo
        
        # Crear subplot polar
        ax = self.figure.add_subplot(111, projection='polar')
        
        # Agregar valores para completar el círculo
        values1 += values1[:1]
        values2 += values2[:1]
        
        # Dibujar el gráfico principal
        ax.plot(angles, values1, 'o-', linewidth=2, label='Muestra Actual', color='#2E86AB')
        ax.fill(angles, values1, alpha=0.25, color='#2E86AB')
        
        # Dibujar comparación si está habilitada
        if self.comparison_cb.isChecked():
            ax.plot(angles, values2, 'o-', linewidth=2, label='Referencia', color='#A23B72')
            ax.fill(angles, values2, alpha=0.25, color='#A23B72')
        
        # Agregar etiquetas
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=10)
        
        # Configurar escala radial
        ax.set_ylim(0, 100)
        ax.set_yticks([20, 40, 60, 80, 100])
        ax.set_yticklabels(['20', '40', '60', '80', '100'], fontsize=8)
        ax.grid(True)
        
        # Título y leyenda
        ax.set_title('Métricas de Calidad NIST', size=14, fontweight='bold', pad=20)
        
        if self.comparison_cb.isChecked():
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        self.figure.tight_layout()
        self.canvas.draw()
        
    def export_radar(self):
        """Exporta el gráfico radar actual"""
        from PyQt5.QtWidgets import QFileDialog
        filename, _ = QFileDialog.getSaveFileName(
            self, "Exportar Gráfico Radar", 
            "radar_metricas_nist.png",
            "PNG Files (*.png);;PDF Files (*.pdf)"
        )
        if filename:
            self.figure.savefig(filename, dpi=300, bbox_inches='tight')


class GraphicsVisualizationPanel(QFrame):
    """Panel principal que contiene todas las visualizaciones gráficas"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setProperty("class", "card")
        self.setup_ui()
        
    def setup_ui(self):
        """Configura el panel de visualizaciones"""
        layout = QVBoxLayout(self)
        
        # Título principal
        title_label = QLabel("Visualizaciones Gráficas de Análisis")
        title_label.setFont(QFont("Arial", 14, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("color: #333; padding: 10px; background-color: #f0f0f0; border-radius: 5px;")
        layout.addWidget(title_label)
        
        # Pestañas para diferentes tipos de gráficos
        self.tab_widget = QTabWidget()
        
        # Pestaña de histogramas
        self.histogram_widget = HistogramWidget()
        self.tab_widget.addTab(self.histogram_widget, "📊 Histogramas")
        
        # Pestaña de mapas de calor
        self.heatmap_widget = HeatmapWidget()
        self.tab_widget.addTab(self.heatmap_widget, "🔥 Mapas de Calor")
        
        # Pestaña de gráficos radar
        self.radar_widget = RadarChartWidget()
        self.tab_widget.addTab(self.radar_widget, "🎯 Métricas NIST")
        
        layout.addWidget(self.tab_widget)
        
        # Botones de acción globales
        actions_layout = QHBoxLayout()
        
        self.refresh_all_btn = QPushButton("🔄 Actualizar Todo")
        self.refresh_all_btn.clicked.connect(self.refresh_all_graphics)
        actions_layout.addWidget(self.refresh_all_btn)
        
        actions_layout.addStretch()
        
        self.export_all_btn = QPushButton("📁 Exportar Todo")
        self.export_all_btn.clicked.connect(self.export_all_graphics)
        actions_layout.addWidget(self.export_all_btn)
        
        layout.addLayout(actions_layout)
        
        # Inicializar gráficos
        self.refresh_all_graphics()
        
    def refresh_all_graphics(self):
        """Actualiza todos los gráficos"""
        self.histogram_widget.update_histogram()
        self.heatmap_widget.update_heatmap()
        self.radar_widget.update_radar()
        
    def export_all_graphics(self):
        """Exporta todos los gráficos"""
        from PyQt5.QtWidgets import QFileDialog, QMessageBox
        import os
        
        folder = QFileDialog.getExistingDirectory(self, "Seleccionar Carpeta para Exportar")
        if folder:
            try:
                # Exportar histograma
                hist_path = os.path.join(folder, "histograma_caracteristicas.png")
                self.histogram_widget.figure.savefig(hist_path, dpi=300, bbox_inches='tight')
                
                # Exportar mapa de calor
                heat_path = os.path.join(folder, "mapa_calor_analisis.png")
                self.heatmap_widget.figure.savefig(heat_path, dpi=300, bbox_inches='tight')
                
                # Exportar gráfico radar
                radar_path = os.path.join(folder, "radar_metricas_nist.png")
                self.radar_widget.figure.savefig(radar_path, dpi=300, bbox_inches='tight')
                
                QMessageBox.information(self, "Exportación Completa", 
                                      f"Todos los gráficos han sido exportados a:\n{folder}")
                
            except Exception as e:
                QMessageBox.warning(self, "Error de Exportación", 
                                  f"Error al exportar gráficos:\n{str(e)}")
    
    def update_with_results(self, results: dict):
        """Actualiza las visualizaciones con resultados reales"""
        # Aquí se implementaría la lógica para usar datos reales
        # Por ahora mantiene los datos simulados
        self.refresh_all_graphics()