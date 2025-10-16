"""
Mapa de Correlación 2D para Comparación de Cara de Recámara
Genera mapas de calor que representan visualmente las zonas de alta y baja
correlación entre dos muestras balísticas.
"""

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QFrame, QSlider, QCheckBox,
                             QGroupBox, QComboBox, QSpinBox, QProgressBar)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer, QThread, pyqtSlot
from PyQt5.QtGui import (QPixmap, QPainter, QPen, QColor, QFont, QBrush, 
                         QLinearGradient, QRadialGradient, QPolygonF)
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
# Importaciones de visualización - Comentadas temporalmente para pruebas
import seaborn as sns

# Mock temporal para seaborn
class MockSeaborn:
    def set_style(self, *args, **kwargs):
        pass
    def heatmap(self, *args, **kwargs):
        pass
    def scatterplot(self, *args, **kwargs):
        pass

sns = MockSeaborn()


class CorrelationWorker(QThread):
    """Worker thread para calcular correlaciones sin bloquear la UI"""
    
    correlationComputed = pyqtSignal(np.ndarray, dict)  # correlation_map, stats
    progressUpdated = pyqtSignal(int)  # progress percentage
    
    def __init__(self, image1, image2, method='ncc', window_size=32, step_size=8):
        super().__init__()
        self.image1 = image1
        self.image2 = image2
        self.method = method
        self.window_size = window_size
        self.step_size = step_size
        self.is_cancelled = False
        
    def run(self):
        """Ejecuta el cálculo de correlación"""
        try:
            if self.method == 'ncc':
                correlation_map, stats = self.compute_ncc_correlation()
            elif self.method == 'ssim':
                correlation_map, stats = self.compute_ssim_correlation()
            elif self.method == 'mutual_info':
                correlation_map, stats = self.compute_mutual_info_correlation()
            else:
                correlation_map, stats = self.compute_template_matching()
                
            if not self.is_cancelled:
                self.correlationComputed.emit(correlation_map, stats)
                
        except Exception as e:
            print(f"Error en cálculo de correlación: {e}")
            
    def compute_ncc_correlation(self):
        """Calcula correlación cruzada normalizada"""
        h1, w1 = self.image1.shape
        h2, w2 = self.image2.shape
        
        # Calcular dimensiones del mapa de correlación
        map_h = (h1 - self.window_size) // self.step_size + 1
        map_w = (w1 - self.window_size) // self.step_size + 1
        
        correlation_map = np.zeros((map_h, map_w))
        total_operations = map_h * map_w
        completed = 0
        
        for i in range(map_h):
            if self.is_cancelled:
                break
                
            for j in range(map_w):
                # Extraer ventana de la imagen 1
                y1 = i * self.step_size
                x1 = j * self.step_size
                window1 = self.image1[y1:y1+self.window_size, x1:x1+self.window_size]
                
                # Encontrar mejor coincidencia en imagen 2
                best_corr = -1
                
                # Buscar en área limitada para eficiencia
                search_range = min(50, min(h2-self.window_size, w2-self.window_size))
                
                for dy in range(-search_range//2, search_range//2, self.step_size):
                    for dx in range(-search_range//2, search_range//2, self.step_size):
                        y2 = max(0, min(h2-self.window_size, y1 + dy))
                        x2 = max(0, min(w2-self.window_size, x1 + dx))
                        
                        window2 = self.image2[y2:y2+self.window_size, x2:x2+self.window_size]
                        
                        # Calcular correlación normalizada
                        corr = cv2.matchTemplate(window1, window2, cv2.TM_CCOEFF_NORMED)[0, 0]
                        best_corr = max(best_corr, corr)
                        
                correlation_map[i, j] = best_corr
                
                completed += 1
                if completed % 10 == 0:
                    progress = int((completed / total_operations) * 100)
                    self.progressUpdated.emit(progress)
                    
        # Calcular estadísticas
        stats = {
            'mean_correlation': np.mean(correlation_map),
            'max_correlation': np.max(correlation_map),
            'min_correlation': np.min(correlation_map),
            'std_correlation': np.std(correlation_map),
            'high_corr_percentage': np.sum(correlation_map > 0.7) / correlation_map.size * 100
        }
        
        return correlation_map, stats
        
    def compute_ssim_correlation(self):
        """Calcula correlación usando SSIM (Structural Similarity Index)"""
        from skimage.metrics import structural_similarity as ssim
        
        h1, w1 = self.image1.shape
        map_h = (h1 - self.window_size) // self.step_size + 1
        map_w = (w1 - self.window_size) // self.step_size + 1
        
        correlation_map = np.zeros((map_h, map_w))
        total_operations = map_h * map_w
        completed = 0
        
        for i in range(map_h):
            if self.is_cancelled:
                break
                
            for j in range(map_w):
                y1 = i * self.step_size
                x1 = j * self.step_size
                window1 = self.image1[y1:y1+self.window_size, x1:x1+self.window_size]
                
                # Extraer ventana correspondiente de imagen 2
                y2 = min(y1, self.image2.shape[0] - self.window_size)
                x2 = min(x1, self.image2.shape[1] - self.window_size)
                window2 = self.image2[y2:y2+self.window_size, x2:x2+self.window_size]
                
                # Calcular SSIM
                ssim_value = ssim(window1, window2, data_range=255)
                correlation_map[i, j] = ssim_value
                
                completed += 1
                if completed % 10 == 0:
                    progress = int((completed / total_operations) * 100)
                    self.progressUpdated.emit(progress)
                    
        stats = {
            'mean_correlation': np.mean(correlation_map),
            'max_correlation': np.max(correlation_map),
            'min_correlation': np.min(correlation_map),
            'std_correlation': np.std(correlation_map),
            'high_corr_percentage': np.sum(correlation_map > 0.8) / correlation_map.size * 100
        }
        
        return correlation_map, stats
        
    def compute_template_matching(self):
        """Calcula correlación usando template matching de OpenCV"""
        # Usar toda la imagen como template
        result = cv2.matchTemplate(self.image2, self.image1, cv2.TM_CCOEFF_NORMED)
        
        # Redimensionar para que coincida con el formato esperado
        target_size = (50, 50)  # Tamaño estándar para visualización
        correlation_map = cv2.resize(result, target_size)
        
        stats = {
            'mean_correlation': np.mean(correlation_map),
            'max_correlation': np.max(correlation_map),
            'min_correlation': np.min(correlation_map),
            'std_correlation': np.std(correlation_map),
            'high_corr_percentage': np.sum(correlation_map > 0.7) / correlation_map.size * 100
        }
        
        return correlation_map, stats
        
    def cancel(self):
        """Cancela el cálculo"""
        self.is_cancelled = True


class InteractiveHeatmapCanvas(FigureCanvas):
    """Canvas interactivo para mostrar el mapa de calor"""
    
    pointClicked = pyqtSignal(int, int, float)  # x, y, correlation_value
    
    def __init__(self, parent=None):
        self.figure = Figure(figsize=(8, 6))
        super().__init__(self.figure)
        self.setParent(parent)
        
        self.correlation_map = None
        self.colormap = 'viridis'
        self.show_values = False
        self.show_contours = False
        
        # Configurar interactividad
        self.mpl_connect('button_press_event', self.on_click)
        self.mpl_connect('motion_notify_event', self.on_hover)
        
    def set_correlation_map(self, correlation_map, title="Mapa de Correlación"):
        """Establece el mapa de correlación a mostrar"""
        self.correlation_map = correlation_map
        self.plot_heatmap(title)
        
    def plot_heatmap(self, title="Mapa de Correlación"):
        """Dibuja el mapa de calor"""
        if self.correlation_map is None:
            return
            
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        # Crear el mapa de calor
        im = ax.imshow(self.correlation_map, cmap=self.colormap, 
                      interpolation='bilinear', aspect='auto')
        
        # Añadir barra de color
        cbar = self.figure.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Correlación', rotation=270, labelpad=20)
        
        # Añadir contornos si está habilitado
        if self.show_contours:
            contours = ax.contour(self.correlation_map, levels=10, colors='white', alpha=0.5)
            ax.clabel(contours, inline=True, fontsize=8)
            
        # Mostrar valores si está habilitado
        if self.show_values:
            h, w = self.correlation_map.shape
            for i in range(0, h, max(1, h//10)):
                for j in range(0, w, max(1, w//10)):
                    value = self.correlation_map[i, j]
                    ax.text(j, i, f'{value:.2f}', ha='center', va='center',
                           color='white' if value < 0.5 else 'black', fontsize=8)
                           
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Posición X')
        ax.set_ylabel('Posición Y')
        
        self.draw()
        
    def on_click(self, event):
        """Maneja clics en el mapa de calor"""
        if event.inaxes and self.correlation_map is not None:
            x, y = int(event.xdata), int(event.ydata)
            if 0 <= y < self.correlation_map.shape[0] and 0 <= x < self.correlation_map.shape[1]:
                correlation_value = self.correlation_map[y, x]
                self.pointClicked.emit(x, y, correlation_value)
                
    def on_hover(self, event):
        """Maneja el hover sobre el mapa"""
        if event.inaxes and self.correlation_map is not None:
            x, y = int(event.xdata), int(event.ydata)
            if 0 <= y < self.correlation_map.shape[0] and 0 <= x < self.correlation_map.shape[1]:
                correlation_value = self.correlation_map[y, x]
                self.setToolTip(f"Posición: ({x}, {y})\nCorrelación: {correlation_value:.3f}")
                
    def set_colormap(self, colormap):
        """Cambia el mapa de colores"""
        self.colormap = colormap
        if self.correlation_map is not None:
            self.plot_heatmap()
            
    def toggle_values(self, show_values):
        """Alterna la visualización de valores"""
        self.show_values = show_values
        if self.correlation_map is not None:
            self.plot_heatmap()
            
    def toggle_contours(self, show_contours):
        """Alterna la visualización de contornos"""
        self.show_contours = show_contours
        if self.correlation_map is not None:
            self.plot_heatmap()


class CorrelationHeatmapWidget(QWidget):
    """Widget principal para visualización de mapas de correlación"""
    
    correlationAnalyzed = pyqtSignal(dict)  # statistics
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.image1 = None
        self.image2 = None
        self.correlation_worker = None
        
        self.setup_ui()
        self.connect_signals()
        
    def setup_ui(self):
        """Configura la interfaz principal"""
        layout = QVBoxLayout(self)
        
        # Panel de controles
        controls_group = QGroupBox("Configuración de Análisis")
        controls_layout = QHBoxLayout(controls_group)
        
        # Método de correlación
        method_label = QLabel("Método:")
        self.method_combo = QComboBox()
        self.method_combo.addItems([
            "NCC (Correlación Cruzada Normalizada)",
            "SSIM (Índice de Similitud Estructural)",
            "Template Matching",
            "Información Mutua"
        ])
        
        # Tamaño de ventana
        window_label = QLabel("Tamaño de Ventana:")
        self.window_spin = QSpinBox()
        self.window_spin.setRange(16, 128)
        self.window_spin.setValue(32)
        self.window_spin.setSingleStep(16)
        
        # Paso de análisis
        step_label = QLabel("Paso:")
        self.step_spin = QSpinBox()
        self.step_spin.setRange(4, 32)
        self.step_spin.setValue(8)
        self.step_spin.setSingleStep(4)
        
        # Botón de análisis
        self.analyze_btn = QPushButton("Analizar Correlación")
        self.analyze_btn.clicked.connect(self.start_correlation_analysis)
        
        # Barra de progreso
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        
        controls_layout.addWidget(method_label)
        controls_layout.addWidget(self.method_combo)
        controls_layout.addWidget(window_label)
        controls_layout.addWidget(self.window_spin)
        controls_layout.addWidget(step_label)
        controls_layout.addWidget(self.step_spin)
        controls_layout.addStretch()
        controls_layout.addWidget(self.analyze_btn)
        
        # Panel de visualización
        viz_group = QGroupBox("Opciones de Visualización")
        viz_layout = QHBoxLayout(viz_group)
        
        # Mapa de colores
        colormap_label = QLabel("Mapa de Colores:")
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems([
            "viridis", "plasma", "inferno", "magma", "hot", "cool", 
            "spring", "summer", "autumn", "winter", "jet", "rainbow"
        ])
        self.colormap_combo.currentTextChanged.connect(self.change_colormap)
        
        # Opciones de visualización
        self.show_values_cb = QCheckBox("Mostrar Valores")
        self.show_values_cb.toggled.connect(self.toggle_values)
        
        self.show_contours_cb = QCheckBox("Mostrar Contornos")
        self.show_contours_cb.toggled.connect(self.toggle_contours)
        
        # Botón de exportar
        self.export_btn = QPushButton("Exportar Mapa")
        self.export_btn.clicked.connect(self.export_heatmap)
        self.export_btn.setEnabled(False)
        
        viz_layout.addWidget(colormap_label)
        viz_layout.addWidget(self.colormap_combo)
        viz_layout.addWidget(self.show_values_cb)
        viz_layout.addWidget(self.show_contours_cb)
        viz_layout.addStretch()
        viz_layout.addWidget(self.export_btn)
        
        # Canvas del mapa de calor
        self.heatmap_canvas = InteractiveHeatmapCanvas()
        
        # Panel de estadísticas
        stats_group = QGroupBox("Estadísticas de Correlación")
        stats_layout = QVBoxLayout(stats_group)
        
        self.stats_labels = {}
        stats_items = [
            ("Correlación Promedio", "N/A"),
            ("Correlación Máxima", "N/A"),
            ("Correlación Mínima", "N/A"),
            ("Desviación Estándar", "N/A"),
            ("% Alta Correlación", "N/A")
        ]
        
        for label, value in stats_items:
            row_layout = QHBoxLayout()
            label_widget = QLabel(f"{label}:")
            value_widget = QLabel(value)
            value_widget.setStyleSheet("font-weight: bold; color: #2196f3;")
            
            row_layout.addWidget(label_widget)
            row_layout.addStretch()
            row_layout.addWidget(value_widget)
            
            stats_layout.addLayout(row_layout)
            self.stats_labels[label] = value_widget
            
        # Layout principal
        layout.addWidget(controls_group)
        layout.addWidget(self.progress_bar)
        layout.addWidget(viz_group)
        layout.addWidget(self.heatmap_canvas)
        layout.addWidget(stats_group)
        
    def connect_signals(self):
        """Conecta las señales"""
        self.heatmap_canvas.pointClicked.connect(self.on_point_clicked)
        
    def set_images(self, image1, image2):
        """Establece las imágenes para análisis de correlación"""
        # Convertir a escala de grises si es necesario
        if len(image1.shape) == 3:
            self.image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        else:
            self.image1 = image1.copy()
            
        if len(image2.shape) == 3:
            self.image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
        else:
            self.image2 = image2.copy()
            
        self.analyze_btn.setEnabled(True)
        
    def start_correlation_analysis(self):
        """Inicia el análisis de correlación"""
        if self.image1 is None or self.image2 is None:
            return
            
        # Configurar parámetros
        method_map = {
            0: 'ncc',
            1: 'ssim', 
            2: 'template_matching',
            3: 'mutual_info'
        }
        
        method = method_map[self.method_combo.currentIndex()]
        window_size = self.window_spin.value()
        step_size = self.step_spin.value()
        
        # Crear y configurar worker
        self.correlation_worker = CorrelationWorker(
            self.image1, self.image2, method, window_size, step_size
        )
        
        self.correlation_worker.correlationComputed.connect(self.on_correlation_computed)
        self.correlation_worker.progressUpdated.connect(self.update_progress)
        
        # Mostrar progreso y deshabilitar controles
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.analyze_btn.setEnabled(False)
        
        # Iniciar análisis
        self.correlation_worker.start()
        
    @pyqtSlot(np.ndarray, dict)
    def on_correlation_computed(self, correlation_map, stats):
        """Maneja la finalización del análisis de correlación"""
        # Mostrar mapa de calor
        self.heatmap_canvas.set_correlation_map(correlation_map)
        
        # Actualizar estadísticas
        self.update_statistics(stats)
        
        # Ocultar progreso y habilitar controles
        self.progress_bar.setVisible(False)
        self.analyze_btn.setEnabled(True)
        self.export_btn.setEnabled(True)
        
        # Emitir señal con estadísticas
        self.correlationAnalyzed.emit(stats)
        
    @pyqtSlot(int)
    def update_progress(self, progress):
        """Actualiza la barra de progreso"""
        self.progress_bar.setValue(progress)
        
    def update_statistics(self, stats):
        """Actualiza las estadísticas mostradas"""
        stat_mapping = {
            "Correlación Promedio": f"{stats['mean_correlation']:.3f}",
            "Correlación Máxima": f"{stats['max_correlation']:.3f}",
            "Correlación Mínima": f"{stats['min_correlation']:.3f}",
            "Desviación Estándar": f"{stats['std_correlation']:.3f}",
            "% Alta Correlación": f"{stats['high_corr_percentage']:.1f}%"
        }
        
        for label, value in stat_mapping.items():
            if label in self.stats_labels:
                self.stats_labels[label].setText(value)
                
    def on_point_clicked(self, x, y, correlation_value):
        """Maneja clics en puntos del mapa de calor"""
        print(f"Punto clickeado: ({x}, {y}) - Correlación: {correlation_value:.3f}")
        
    def change_colormap(self, colormap):
        """Cambia el mapa de colores"""
        self.heatmap_canvas.set_colormap(colormap)
        
    def toggle_values(self, show_values):
        """Alterna la visualización de valores"""
        self.heatmap_canvas.toggle_values(show_values)
        
    def toggle_contours(self, show_contours):
        """Alterna la visualización de contornos"""
        self.heatmap_canvas.toggle_contours(show_contours)
        
    def export_heatmap(self):
        """Exporta el mapa de calor"""
        if self.heatmap_canvas.correlation_map is not None:
            # Aquí se implementaría la lógica de exportación
            print("Exportando mapa de calor...")
            
    def generate_sample_correlation(self):
        """Genera un mapa de correlación de ejemplo"""
        # Crear datos de ejemplo
        x = np.linspace(0, 4*np.pi, 50)
        y = np.linspace(0, 4*np.pi, 50)
        X, Y = np.meshgrid(x, y)
        
        # Generar patrón de correlación simulado
        correlation_map = 0.5 + 0.3 * np.sin(X) * np.cos(Y) + 0.2 * np.random.random((50, 50))
        correlation_map = np.clip(correlation_map, 0, 1)
        
        # Estadísticas simuladas
        stats = {
            'mean_correlation': np.mean(correlation_map),
            'max_correlation': np.max(correlation_map),
            'min_correlation': np.min(correlation_map),
            'std_correlation': np.std(correlation_map),
            'high_corr_percentage': np.sum(correlation_map > 0.7) / correlation_map.size * 100
        }
        
        self.on_correlation_computed(correlation_map, stats)