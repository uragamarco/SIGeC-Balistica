"""
Widget Interactivo de Curvas CMC Mejorado
Proporciona visualización interactiva de curvas CMC con capacidades de zoom,
superposición y análisis detallado.
"""

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QFrame, QSlider, QCheckBox,
                             QGroupBox, QComboBox, QSpinBox, QListWidget,
                             QListWidgetItem, QSplitter, QTextEdit, QTabWidget)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from PyQt5.QtGui import (QPixmap, QPainter, QPen, QColor, QFont, QBrush, 
                         QLinearGradient, QIcon, QPalette)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.widgets import Cursor
import seaborn as sns


class CMCCurveData:
    """Clase para almacenar datos de una curva CMC"""
    
    def __init__(self, name, x_data, y_data, color='blue', style='-', metadata=None):
        self.name = name
        self.x_data = np.array(x_data)
        self.y_data = np.array(y_data)
        self.color = color
        self.style = style
        self.metadata = metadata or {}
        self.visible = True
        self.highlighted = False
        
    def get_auc(self):
        """Calcula el área bajo la curva"""
        return np.trapz(self.y_data, self.x_data)
        
    def get_eer(self):
        """Calcula la tasa de error igual (Equal Error Rate)"""
        # Encontrar el punto donde FAR = FRR
        far = self.x_data
        frr = 1 - self.y_data
        
        # Encontrar intersección
        diff = np.abs(far - frr)
        eer_idx = np.argmin(diff)
        
        return far[eer_idx], eer_idx
        
    def get_point_at_x(self, x_value):
        """Obtiene el valor Y para un valor X dado"""
        if x_value < self.x_data.min() or x_value > self.x_data.max():
            return None
            
        # Interpolación lineal
        y_interp = np.interp(x_value, self.x_data, self.y_data)
        return y_interp


class InteractiveCMCCanvas(FigureCanvas):
    """Canvas interactivo para curvas CMC"""
    
    pointHovered = pyqtSignal(str, float, float)  # curve_name, x, y
    pointClicked = pyqtSignal(str, float, float)  # curve_name, x, y
    zoomChanged = pyqtSignal(float, float, float, float)  # x_min, x_max, y_min, y_max
    
    def __init__(self, parent=None):
        self.figure = Figure(figsize=(10, 8))
        super().__init__(self.figure)
        self.setParent(parent)
        
        self.curves = {}
        self.show_grid = True
        self.show_legend = True
        self.show_statistics = True
        self.show_confidence_bands = False
        
        # Configurar subplot
        self.ax = self.figure.add_subplot(111)
        self.setup_plot()
        
        # Configurar interactividad
        self.mpl_connect('button_press_event', self.on_click)
        self.mpl_connect('motion_notify_event', self.on_hover)
        self.mpl_connect('scroll_event', self.on_scroll)
        self.mpl_connect('key_press_event', self.on_key_press)
        
        # Cursor para mostrar coordenadas
        self.cursor = Cursor(self.ax, useblit=True, color='red', linewidth=1)
        
        # Variables para zoom
        self.zoom_factor = 1.1
        self.pan_active = False
        self.zoom_active = False
        
    def setup_plot(self):
        """Configura el plot inicial"""
        self.ax.set_xlabel('Tasa de Falsos Positivos (FAR)', fontsize=12)
        self.ax.set_ylabel('Tasa de Verdaderos Positivos (TAR)', fontsize=12)
        self.ax.set_title('Curvas CMC Interactivas', fontsize=14, fontweight='bold')
        
        # Configurar límites
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        
        # Configurar grid
        if self.show_grid:
            self.ax.grid(True, alpha=0.3)
            
        # Línea diagonal de referencia
        self.ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Línea de Referencia')
        
        self.draw()
        
    def add_curve(self, curve_data):
        """Añade una nueva curva CMC"""
        self.curves[curve_data.name] = curve_data
        self.update_plot()
        
    def remove_curve(self, curve_name):
        """Elimina una curva CMC"""
        if curve_name in self.curves:
            del self.curves[curve_name]
            self.update_plot()
            
    def update_curve_visibility(self, curve_name, visible):
        """Actualiza la visibilidad de una curva"""
        if curve_name in self.curves:
            self.curves[curve_name].visible = visible
            self.update_plot()
            
    def highlight_curve(self, curve_name, highlighted=True):
        """Resalta una curva específica"""
        if curve_name in self.curves:
            self.curves[curve_name].highlighted = highlighted
            self.update_plot()
            
    def update_plot(self):
        """Actualiza la visualización del plot"""
        self.ax.clear()
        self.setup_plot()
        
        # Dibujar curvas
        for curve_name, curve in self.curves.items():
            if not curve.visible:
                continue
                
            # Configurar estilo
            linewidth = 3 if curve.highlighted else 2
            alpha = 1.0 if curve.highlighted else 0.8
            
            # Dibujar curva principal
            line = self.ax.plot(curve.x_data, curve.y_data, 
                              color=curve.color, linestyle=curve.style,
                              linewidth=linewidth, alpha=alpha, 
                              label=curve.name, picker=True)[0]
            
            # Añadir marcadores en puntos clave si está resaltada
            if curve.highlighted:
                # Marcar EER
                eer_value, eer_idx = curve.get_eer()
                self.ax.plot(curve.x_data[eer_idx], curve.y_data[eer_idx], 
                           'ro', markersize=8, label=f'EER: {eer_value:.3f}')
                
                # Marcar puntos de alta confianza
                high_conf_indices = np.where(curve.y_data > 0.9)[0]
                if len(high_conf_indices) > 0:
                    self.ax.plot(curve.x_data[high_conf_indices], 
                               curve.y_data[high_conf_indices], 
                               'go', markersize=6, alpha=0.7)
                               
            # Añadir bandas de confianza si están habilitadas
            if self.show_confidence_bands and 'confidence_lower' in curve.metadata:
                lower = curve.metadata['confidence_lower']
                upper = curve.metadata['confidence_upper']
                self.ax.fill_between(curve.x_data, lower, upper, 
                                   color=curve.color, alpha=0.2)
                                   
        # Configurar leyenda
        if self.show_legend and self.curves:
            self.ax.legend(loc='lower right', framealpha=0.9)
            
        # Añadir estadísticas si están habilitadas
        if self.show_statistics:
            self.add_statistics_text()
            
        self.draw()
        
    def add_statistics_text(self):
        """Añade texto con estadísticas de las curvas"""
        if not self.curves:
            return
            
        stats_text = "Estadísticas:\n"
        for curve_name, curve in self.curves.items():
            if curve.visible:
                auc = curve.get_auc()
                eer, _ = curve.get_eer()
                stats_text += f"{curve_name}: AUC={auc:.3f}, EER={eer:.3f}\n"
                
        # Posicionar texto en esquina superior izquierda
        self.ax.text(0.02, 0.98, stats_text, transform=self.ax.transAxes,
                    verticalalignment='top', bbox=dict(boxstyle='round', 
                    facecolor='white', alpha=0.8), fontsize=9)
                    
    def on_click(self, event):
        """Maneja clics en el canvas"""
        if event.inaxes != self.ax:
            return
            
        # Encontrar la curva más cercana al clic
        closest_curve = None
        min_distance = float('inf')
        
        for curve_name, curve in self.curves.items():
            if not curve.visible:
                continue
                
            # Calcular distancia al clic
            distances = np.sqrt((curve.x_data - event.xdata)**2 + 
                              (curve.y_data - event.ydata)**2)
            min_dist = np.min(distances)
            
            if min_dist < min_distance:
                min_distance = min_dist
                closest_curve = curve_name
                
        if closest_curve and min_distance < 0.05:  # Tolerancia de clic
            curve = self.curves[closest_curve]
            # Encontrar punto más cercano
            distances = np.sqrt((curve.x_data - event.xdata)**2 + 
                              (curve.y_data - event.ydata)**2)
            closest_idx = np.argmin(distances)
            
            x_val = curve.x_data[closest_idx]
            y_val = curve.y_data[closest_idx]
            
            self.pointClicked.emit(closest_curve, x_val, y_val)
            
    def on_hover(self, event):
        """Maneja el hover sobre el canvas"""
        if event.inaxes != self.ax:
            return
            
        # Actualizar tooltip con información del punto
        if event.xdata is not None and event.ydata is not None:
            tooltip_text = f"FAR: {event.xdata:.3f}, TAR: {event.ydata:.3f}"
            
            # Buscar curva más cercana
            for curve_name, curve in self.curves.items():
                if not curve.visible:
                    continue
                    
                y_interp = curve.get_point_at_x(event.xdata)
                if y_interp is not None:
                    distance = abs(y_interp - event.ydata)
                    if distance < 0.05:
                        tooltip_text += f"\nCurva: {curve_name}"
                        self.pointHovered.emit(curve_name, event.xdata, y_interp)
                        break
                        
            self.setToolTip(tooltip_text)
            
    def on_scroll(self, event):
        """Maneja el zoom con scroll del mouse"""
        if event.inaxes != self.ax:
            return
            
        # Obtener límites actuales
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        
        # Calcular factor de zoom
        if event.button == 'up':
            scale_factor = 1 / self.zoom_factor
        else:
            scale_factor = self.zoom_factor
            
        # Calcular nuevos límites centrados en el cursor
        x_center = event.xdata
        y_center = event.ydata
        
        x_range = (xlim[1] - xlim[0]) * scale_factor
        y_range = (ylim[1] - ylim[0]) * scale_factor
        
        new_xlim = [x_center - x_range/2, x_center + x_range/2]
        new_ylim = [y_center - y_range/2, y_center + y_range/2]
        
        # Aplicar límites
        self.ax.set_xlim(new_xlim)
        self.ax.set_ylim(new_ylim)
        
        self.draw()
        self.zoomChanged.emit(new_xlim[0], new_xlim[1], new_ylim[0], new_ylim[1])
        
    def on_key_press(self, event):
        """Maneja teclas presionadas"""
        if event.key == 'r':  # Reset zoom
            self.ax.set_xlim(0, 1)
            self.ax.set_ylim(0, 1)
            self.draw()
        elif event.key == 'g':  # Toggle grid
            self.show_grid = not self.show_grid
            self.update_plot()
        elif event.key == 'l':  # Toggle legend
            self.show_legend = not self.show_legend
            self.update_plot()
            
    def export_plot(self, filename, dpi=300):
        """Exporta el plot a archivo"""
        self.figure.savefig(filename, dpi=dpi, bbox_inches='tight')
        
    def set_zoom_region(self, x_min, x_max, y_min, y_max):
        """Establece una región de zoom específica"""
        self.ax.set_xlim(x_min, x_max)
        self.ax.set_ylim(y_min, y_max)
        self.draw()


class CMCCurveManager(QWidget):
    """Widget para gestionar múltiples curvas CMC"""
    
    curveSelectionChanged = pyqtSignal(str)  # curve_name
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        
    def setup_ui(self):
        """Configura la interfaz del gestor"""
        layout = QVBoxLayout(self)
        
        # Título
        title = QLabel("Gestión de Curvas")
        title.setFont(QFont("Arial", 12, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        
        # Lista de curvas
        self.curve_list = QListWidget()
        self.curve_list.itemChanged.connect(self.on_item_changed)
        self.curve_list.itemClicked.connect(self.on_item_clicked)
        
        # Botones de control
        buttons_layout = QHBoxLayout()
        
        self.add_btn = QPushButton("Añadir")
        self.add_btn.setIcon(QIcon("➕"))
        
        self.remove_btn = QPushButton("Eliminar")
        self.remove_btn.setIcon(QIcon("➖"))
        
        self.highlight_btn = QPushButton("Resaltar")
        self.highlight_btn.setIcon(QIcon("🔍"))
        self.highlight_btn.setCheckable(True)
        
        buttons_layout.addWidget(self.add_btn)
        buttons_layout.addWidget(self.remove_btn)
        buttons_layout.addWidget(self.highlight_btn)
        
        layout.addWidget(title)
        layout.addWidget(self.curve_list)
        layout.addLayout(buttons_layout)
        
    def add_curve_item(self, curve_name, color):
        """Añade un elemento de curva a la lista"""
        item = QListWidgetItem(curve_name)
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
        item.setCheckState(Qt.Checked)
        
        # Crear indicador de color
        pixmap = QPixmap(16, 16)
        pixmap.fill(QColor(color))
        item.setIcon(QIcon(pixmap))
        
        self.curve_list.addItem(item)
        
    def remove_curve_item(self, curve_name):
        """Elimina un elemento de curva de la lista"""
        for i in range(self.curve_list.count()):
            item = self.curve_list.item(i)
            if item.text() == curve_name:
                self.curve_list.takeItem(i)
                break
                
    def on_item_changed(self, item):
        """Maneja cambios en los elementos de la lista"""
        curve_name = item.text()
        visible = item.checkState() == Qt.Checked
        # Emitir señal de cambio de visibilidad
        
    def on_item_clicked(self, item):
        """Maneja clics en elementos de la lista"""
        curve_name = item.text()
        self.curveSelectionChanged.emit(curve_name)


class InteractiveCMCWidget(QWidget):
    """Widget principal para visualización interactiva de curvas CMC"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self.connect_signals()
        
        # Generar datos de ejemplo
        self.generate_sample_curves()
        
    def setup_ui(self):
        """Configura la interfaz principal"""
        layout = QHBoxLayout(self)
        
        # Splitter principal
        main_splitter = QSplitter(Qt.Horizontal)
        
        # Panel izquierdo - Canvas y controles
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Controles de visualización
        viz_controls = QGroupBox("Controles de Visualización")
        viz_layout = QHBoxLayout(viz_controls)
        
        self.grid_cb = QCheckBox("Grid")
        self.grid_cb.setChecked(True)
        
        self.legend_cb = QCheckBox("Leyenda")
        self.legend_cb.setChecked(True)
        
        self.stats_cb = QCheckBox("Estadísticas")
        self.stats_cb.setChecked(True)
        
        self.confidence_cb = QCheckBox("Bandas de Confianza")
        
        self.reset_zoom_btn = QPushButton("Reset Zoom")
        self.export_btn = QPushButton("Exportar")
        
        viz_layout.addWidget(self.grid_cb)
        viz_layout.addWidget(self.legend_cb)
        viz_layout.addWidget(self.stats_cb)
        viz_layout.addWidget(self.confidence_cb)
        viz_layout.addStretch()
        viz_layout.addWidget(self.reset_zoom_btn)
        viz_layout.addWidget(self.export_btn)
        
        # Canvas CMC
        self.cmc_canvas = InteractiveCMCCanvas()
        
        left_layout.addWidget(viz_controls)
        left_layout.addWidget(self.cmc_canvas)
        
        # Panel derecho - Gestión y análisis
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Tabs para diferentes funciones
        tabs = QTabWidget()
        
        # Tab de gestión de curvas
        self.curve_manager = CMCCurveManager()
        tabs.addTab(self.curve_manager, "Curvas")
        
        # Tab de análisis detallado
        analysis_tab = QWidget()
        analysis_layout = QVBoxLayout(analysis_tab)
        
        # Información del punto seleccionado
        point_info_group = QGroupBox("Información del Punto")
        point_info_layout = QVBoxLayout(point_info_group)
        
        self.point_info_text = QTextEdit()
        self.point_info_text.setMaximumHeight(100)
        self.point_info_text.setReadOnly(True)
        
        point_info_layout.addWidget(self.point_info_text)
        
        # Comparación de curvas
        comparison_group = QGroupBox("Comparación de Curvas")
        comparison_layout = QVBoxLayout(comparison_group)
        
        self.comparison_text = QTextEdit()
        self.comparison_text.setReadOnly(True)
        
        comparison_layout.addWidget(self.comparison_text)
        
        analysis_layout.addWidget(point_info_group)
        analysis_layout.addWidget(comparison_group)
        
        tabs.addTab(analysis_tab, "Análisis")
        
        right_layout.addWidget(tabs)
        
        # Configurar splitter
        main_splitter.addWidget(left_panel)
        main_splitter.addWidget(right_panel)
        main_splitter.setSizes([700, 300])
        
        layout.addWidget(main_splitter)
        
    def connect_signals(self):
        """Conecta las señales de los widgets"""
        # Controles de visualización
        self.grid_cb.toggled.connect(self.toggle_grid)
        self.legend_cb.toggled.connect(self.toggle_legend)
        self.stats_cb.toggled.connect(self.toggle_statistics)
        self.confidence_cb.toggled.connect(self.toggle_confidence_bands)
        
        self.reset_zoom_btn.clicked.connect(self.reset_zoom)
        self.export_btn.clicked.connect(self.export_plot)
        
        # Canvas
        self.cmc_canvas.pointClicked.connect(self.on_point_clicked)
        self.cmc_canvas.pointHovered.connect(self.on_point_hovered)
        
        # Gestor de curvas
        self.curve_manager.curveSelectionChanged.connect(self.on_curve_selected)
        
    def generate_sample_curves(self):
        """Genera curvas CMC de ejemplo"""
        # Curva 1 - Alto rendimiento
        x1 = np.linspace(0, 1, 100)
        y1 = 1 - np.exp(-5 * x1) + 0.1 * np.random.random(100)
        y1 = np.clip(y1, 0, 1)
        
        curve1 = CMCCurveData("Algoritmo A", x1, y1, color='blue', 
                             metadata={'algorithm': 'CNN', 'dataset': 'Test1'})
        
        # Curva 2 - Rendimiento medio
        x2 = np.linspace(0, 1, 100)
        y2 = 0.8 * (1 - np.exp(-3 * x2)) + 0.15 * np.random.random(100)
        y2 = np.clip(y2, 0, 1)
        
        curve2 = CMCCurveData("Algoritmo B", x2, y2, color='red',
                             metadata={'algorithm': 'SVM', 'dataset': 'Test1'})
        
        # Curva 3 - Bajo rendimiento
        x3 = np.linspace(0, 1, 100)
        y3 = 0.6 * x3 + 0.2 * np.random.random(100)
        y3 = np.clip(y3, 0, 1)
        
        curve3 = CMCCurveData("Algoritmo C", x3, y3, color='green',
                             metadata={'algorithm': 'Traditional', 'dataset': 'Test1'})
        
        # Añadir curvas al canvas
        self.cmc_canvas.add_curve(curve1)
        self.cmc_canvas.add_curve(curve2)
        self.cmc_canvas.add_curve(curve3)
        
        # Añadir a la lista del gestor
        self.curve_manager.add_curve_item("Algoritmo A", 'blue')
        self.curve_manager.add_curve_item("Algoritmo B", 'red')
        self.curve_manager.add_curve_item("Algoritmo C", 'green')
        
    def toggle_grid(self, show_grid):
        """Alterna la visualización del grid"""
        self.cmc_canvas.show_grid = show_grid
        self.cmc_canvas.update_plot()
        
    def toggle_legend(self, show_legend):
        """Alterna la visualización de la leyenda"""
        self.cmc_canvas.show_legend = show_legend
        self.cmc_canvas.update_plot()
        
    def toggle_statistics(self, show_stats):
        """Alterna la visualización de estadísticas"""
        self.cmc_canvas.show_statistics = show_stats
        self.cmc_canvas.update_plot()
        
    def toggle_confidence_bands(self, show_bands):
        """Alterna la visualización de bandas de confianza"""
        self.cmc_canvas.show_confidence_bands = show_bands
        self.cmc_canvas.update_plot()
        
    def reset_zoom(self):
        """Reinicia el zoom"""
        self.cmc_canvas.set_zoom_region(0, 1, 0, 1)
        
    def export_plot(self):
        """Exporta el plot"""
        filename = "cmc_curves_export.png"
        self.cmc_canvas.export_plot(filename)
        print(f"Plot exportado como {filename}")
        
    def on_point_clicked(self, curve_name, x, y):
        """Maneja clics en puntos de las curvas"""
        info_text = f"Curva: {curve_name}\n"
        info_text += f"FAR: {x:.4f}\n"
        info_text += f"TAR: {y:.4f}\n"
        
        # Información adicional de la curva
        if curve_name in self.cmc_canvas.curves:
            curve = self.cmc_canvas.curves[curve_name]
            auc = curve.get_auc()
            eer, _ = curve.get_eer()
            
            info_text += f"AUC: {auc:.4f}\n"
            info_text += f"EER: {eer:.4f}\n"
            
            if curve.metadata:
                info_text += "\nMetadatos:\n"
                for key, value in curve.metadata.items():
                    info_text += f"{key}: {value}\n"
                    
        self.point_info_text.setPlainText(info_text)
        
    def on_point_hovered(self, curve_name, x, y):
        """Maneja hover sobre puntos de las curvas"""
        # Actualizar información en tiempo real si es necesario
        pass
        
    def on_curve_selected(self, curve_name):
        """Maneja la selección de curvas"""
        # Resaltar la curva seleccionada
        for name in self.cmc_canvas.curves:
            self.cmc_canvas.highlight_curve(name, name == curve_name)
            
        # Actualizar comparación
        self.update_curve_comparison()
        
    def update_cmc_data(self, cmc_data: dict):
        """Actualiza los datos CMC del widget"""
        if not cmc_data:
            return
            
        # Limpiar curvas existentes
        self.cmc_canvas.curves.clear()
        
        # Procesar datos CMC
        if 'curves' in cmc_data:
            for curve_info in cmc_data['curves']:
                name = curve_info.get('name', 'Curva CMC')
                x_data = np.array(curve_info.get('x_data', []))
                y_data = np.array(curve_info.get('y_data', []))
                color = curve_info.get('color', 'blue')
                style = curve_info.get('style', '-')
                metadata = curve_info.get('metadata', {})
                
                if len(x_data) > 0 and len(y_data) > 0:
                    curve_data = CMCCurveData(name, x_data, y_data, color, style, metadata)
                    self.cmc_canvas.add_curve(curve_data)
                    self.curve_manager.add_curve_item(name, color)
        
        # Si no hay curvas específicas, generar una curva básica con los datos disponibles
        elif 'similarity_scores' in cmc_data or 'match_scores' in cmc_data:
            scores = cmc_data.get('similarity_scores', cmc_data.get('match_scores', []))
            if scores:
                # Generar curva CMC básica a partir de scores
                scores = np.array(scores)
                thresholds = np.linspace(0, 1, 100)
                tpr_values = []
                fpr_values = []
                
                for threshold in thresholds:
                    tp = np.sum(scores >= threshold)
                    fp = np.sum(scores < threshold)
                    total_positive = len(scores)
                    total_negative = len(scores)
                    
                    tpr = tp / total_positive if total_positive > 0 else 0
                    fpr = fp / total_negative if total_negative > 0 else 0
                    
                    tpr_values.append(tpr)
                    fpr_values.append(fpr)
                
                curve_data = CMCCurveData("Análisis CMC", fpr_values, tpr_values, 'blue', '-')
                self.cmc_canvas.add_curve(curve_data)
                self.curve_manager.add_curve_item("Análisis CMC", 'blue')
        
        # Actualizar visualización
        self.cmc_canvas.update_plot()
        self.update_curve_comparison()

    def update_curve_comparison(self):
        """Actualiza la comparación entre curvas"""
        comparison_text = "Comparación de Curvas:\n\n"
        
        curves_data = []
        for name, curve in self.cmc_canvas.curves.items():
            if curve.visible:
                auc = curve.get_auc()
                eer, _ = curve.get_eer()
                curves_data.append((name, auc, eer))
                
        # Ordenar por AUC
        curves_data.sort(key=lambda x: x[1], reverse=True)
        
        comparison_text += "Ranking por AUC:\n"
        for i, (name, auc, eer) in enumerate(curves_data, 1):
            comparison_text += f"{i}. {name}: AUC={auc:.4f}, EER={eer:.4f}\n"
            
        self.comparison_text.setPlainText(comparison_text)