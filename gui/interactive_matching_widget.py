"""
Widget de Visualización Interactiva de Coincidencias
Muestra las características coincidentes entre dos muestras balísticas
con líneas conectoras y detalles interactivos al pasar el cursor.
"""

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QScrollArea, QFrame, QToolTip, QGroupBox,
                             QPushButton, QSlider, QCheckBox, QSpinBox)
from PyQt5.QtCore import Qt, pyqtSignal, QPoint, QRect, QTimer
from PyQt5.QtGui import (QPixmap, QPainter, QPen, QColor, QFont, 
                         QBrush, QPolygon, QCursor)
import numpy as np
import math


class MatchingPoint:
    """Representa un punto de coincidencia entre dos imágenes"""
    
    def __init__(self, point1, point2, similarity_score, feature_type="unknown"):
        self.point1 = point1  # QPoint en imagen 1
        self.point2 = point2  # QPoint en imagen 2
        self.similarity_score = similarity_score  # 0.0 - 1.0
        self.feature_type = feature_type  # "striation", "impression", "breach_face", etc.
        self.is_highlighted = False
        self.is_selected = False
        
    def get_color(self):
        """Retorna el color basado en el score de similitud"""
        if self.similarity_score >= 0.8:
            return QColor(0, 255, 0, 180)  # Verde para alta similitud
        elif self.similarity_score >= 0.6:
            return QColor(255, 255, 0, 180)  # Amarillo para similitud media
        else:
            return QColor(255, 0, 0, 180)  # Rojo para baja similitud
            
    def get_line_width(self):
        """Retorna el grosor de línea basado en el score"""
        base_width = 1
        if self.is_highlighted:
            base_width += 2
        if self.is_selected:
            base_width += 1
        return base_width + int(self.similarity_score * 3)


class InteractiveMatchingLabel(QLabel):
    """Label que muestra imagen con puntos de coincidencia interactivos"""
    
    pointHovered = pyqtSignal(object)  # MatchingPoint
    pointClicked = pyqtSignal(object)  # MatchingPoint
    
    def __init__(self, is_left_image=True, parent=None):
        super().__init__(parent)
        self.is_left_image = is_left_image
        self.matching_points = []
        self.hovered_point = None
        self.selected_points = []
        self.show_points = True
        self.show_labels = True
        self.point_size = 8
        
        self.setMinimumSize(400, 400)
        self.setStyleSheet("border: 1px solid #ccc; background-color: #f0f0f0;")
        self.setAlignment(Qt.AlignCenter)
        self.setMouseTracking(True)
        
    def set_matching_points(self, points):
        """Establece los puntos de coincidencia"""
        self.matching_points = points
        self.update()
        
    def set_hovered_point(self, point):
        """Establece el punto resaltado"""
        if self.hovered_point != point:
            self.hovered_point = point
            self.update()
            
    def toggle_point_selection(self, point):
        """Alterna la selección de un punto"""
        if point in self.selected_points:
            self.selected_points.remove(point)
        else:
            self.selected_points.append(point)
        self.update()
        
    def mouseMoveEvent(self, event):
        """Maneja el movimiento del mouse para detectar hover"""
        if not self.matching_points:
            return
            
        mouse_pos = event.pos()
        closest_point = None
        min_distance = float('inf')
        
        for point in self.matching_points:
            # Obtener la posición del punto en esta imagen
            point_pos = point.point1 if self.is_left_image else point.point2
            
            # Calcular distancia
            distance = math.sqrt((mouse_pos.x() - point_pos.x())**2 + 
                               (mouse_pos.y() - point_pos.y())**2)
            
            if distance < self.point_size * 2 and distance < min_distance:
                min_distance = distance
                closest_point = point
                
        # Actualizar punto resaltado
        if closest_point != self.hovered_point:
            self.set_hovered_point(closest_point)
            if closest_point:
                self.pointHovered.emit(closest_point)
                self.setCursor(QCursor(Qt.PointingHandCursor))
            else:
                self.setCursor(QCursor(Qt.ArrowCursor))
                
    def mousePressEvent(self, event):
        """Maneja el clic del mouse"""
        if event.button() == Qt.LeftButton and self.hovered_point:
            self.toggle_point_selection(self.hovered_point)
            self.pointClicked.emit(self.hovered_point)
            
    def paintEvent(self, event):
        """Dibuja la imagen y los puntos de coincidencia"""
        super().paintEvent(event)
        
        if not self.show_points or not self.matching_points:
            return
            
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        for point in self.matching_points:
            # Obtener posición del punto en esta imagen
            point_pos = point.point1 if self.is_left_image else point.point2
            
            # Configurar estilo del punto
            color = point.get_color()
            if point == self.hovered_point:
                color.setAlpha(255)
                point.is_highlighted = True
            else:
                point.is_highlighted = False
                
            if point in self.selected_points:
                point.is_selected = True
            else:
                point.is_selected = False
                
            # Dibujar punto
            painter.setBrush(QBrush(color))
            painter.setPen(QPen(QColor(0, 0, 0), 1))
            
            size = self.point_size
            if point.is_highlighted:
                size += 4
            if point.is_selected:
                size += 2
                
            painter.drawEllipse(point_pos.x() - size//2, 
                              point_pos.y() - size//2, 
                              size, size)
            
            # Dibujar etiqueta si está habilitada
            if self.show_labels and (point.is_highlighted or point.is_selected):
                self.draw_point_label(painter, point, point_pos)
                
    def draw_point_label(self, painter, point, position):
        """Dibuja la etiqueta de información del punto"""
        # Configurar fuente
        font = QFont()
        font.setPointSize(8)
        painter.setFont(font)
        
        # Crear texto de la etiqueta
        label_text = f"{point.feature_type}\nScore: {point.similarity_score:.3f}"
        
        # Calcular tamaño del texto
        metrics = painter.fontMetrics()
        text_rect = metrics.boundingRect(label_text)
        
        # Posicionar etiqueta
        label_x = position.x() + self.point_size + 5
        label_y = position.y() - text_rect.height() // 2
        
        # Ajustar si se sale de los bordes
        if label_x + text_rect.width() > self.width():
            label_x = position.x() - text_rect.width() - self.point_size - 5
        if label_y < 0:
            label_y = 5
        if label_y + text_rect.height() > self.height():
            label_y = self.height() - text_rect.height() - 5
            
        # Dibujar fondo de la etiqueta
        bg_rect = QRect(label_x - 3, label_y - 3, 
                       text_rect.width() + 6, text_rect.height() + 6)
        painter.setBrush(QBrush(QColor(255, 255, 255, 200)))
        painter.setPen(QPen(QColor(0, 0, 0), 1))
        painter.drawRect(bg_rect)
        
        # Dibujar texto
        painter.setPen(QPen(QColor(0, 0, 0)))
        painter.drawText(label_x, label_y + metrics.ascent(), label_text)


class ConnectionLinesWidget(QWidget):
    """Widget que dibuja las líneas de conexión entre puntos coincidentes"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.matching_points = []
        self.hovered_point = None
        self.selected_points = []
        self.show_all_lines = True
        self.show_selected_only = False
        self.line_opacity = 0.7
        
        self.setMinimumHeight(400)
        
    def set_matching_points(self, points):
        """Establece los puntos de coincidencia"""
        self.matching_points = points
        self.update()
        
    def set_hovered_point(self, point):
        """Establece el punto resaltado"""
        self.hovered_point = point
        self.update()
        
    def set_selected_points(self, points):
        """Establece los puntos seleccionados"""
        self.selected_points = points
        self.update()
        
    def paintEvent(self, event):
        """Dibuja las líneas de conexión"""
        if not self.matching_points:
            return
            
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Calcular posiciones relativas
        widget_width = self.width()
        left_image_width = widget_width * 0.4
        right_image_start = widget_width * 0.6
        
        for point in self.matching_points:
            # Decidir si dibujar esta línea
            should_draw = True
            if self.show_selected_only and point not in self.selected_points:
                should_draw = False
            if not self.show_all_lines and point != self.hovered_point:
                should_draw = False
                
            if not should_draw:
                continue
                
            # Calcular posiciones de inicio y fin
            start_x = left_image_width * (point.point1.x() / 400)  # Normalizar
            start_y = point.point1.y()
            
            end_x = right_image_start + (widget_width * 0.4) * (point.point2.x() / 400)
            end_y = point.point2.y()
            
            # Configurar estilo de línea
            color = point.get_color()
            color.setAlpha(int(255 * self.line_opacity))
            
            if point == self.hovered_point:
                color.setAlpha(255)
                
            line_width = point.get_line_width()
            
            painter.setPen(QPen(color, line_width))
            
            # Dibujar línea
            painter.drawLine(int(start_x), int(start_y), int(end_x), int(end_y))
            
            # Dibujar flecha en el extremo
            if point == self.hovered_point or point in self.selected_points:
                self.draw_arrow_head(painter, start_x, start_y, end_x, end_y, color)
                
    def draw_arrow_head(self, painter, start_x, start_y, end_x, end_y, color):
        """Dibuja una punta de flecha al final de la línea"""
        # Calcular ángulo de la línea
        angle = math.atan2(end_y - start_y, end_x - start_x)
        
        # Tamaño de la flecha
        arrow_size = 8
        
        # Calcular puntos de la flecha
        arrow_p1_x = end_x - arrow_size * math.cos(angle - math.pi/6)
        arrow_p1_y = end_y - arrow_size * math.sin(angle - math.pi/6)
        
        arrow_p2_x = end_x - arrow_size * math.cos(angle + math.pi/6)
        arrow_p2_y = end_y - arrow_size * math.sin(angle + math.pi/6)
        
        # Dibujar flecha
        painter.setBrush(QBrush(color))
        arrow_points = [
            QPoint(int(end_x), int(end_y)),
            QPoint(int(arrow_p1_x), int(arrow_p1_y)),
            QPoint(int(arrow_p2_x), int(arrow_p2_y))
        ]
        painter.drawPolygon(QPolygon(arrow_points))


class InteractiveMatchingWidget(QWidget):
    """Widget principal para visualización interactiva de coincidencias"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self.connect_signals()
        self.matching_points = []
        
    def setup_ui(self):
        """Configura la interfaz del widget"""
        layout = QVBoxLayout(self)
        
        # Panel de controles
        controls_group = QGroupBox("Controles de Visualización")
        controls_layout = QHBoxLayout(controls_group)
        
        # Controles de visualización
        self.show_all_lines_cb = QCheckBox("Mostrar todas las líneas")
        self.show_selected_only_cb = QCheckBox("Solo líneas seleccionadas")
        self.show_points_cb = QCheckBox("Mostrar puntos")
        self.show_labels_cb = QCheckBox("Mostrar etiquetas")
        
        self.show_all_lines_cb.setChecked(True)
        self.show_points_cb.setChecked(True)
        self.show_labels_cb.setChecked(True)
        
        # Control de opacidad
        opacity_layout = QVBoxLayout()
        opacity_layout.addWidget(QLabel("Opacidad de líneas:"))
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(10, 100)
        self.opacity_slider.setValue(70)
        opacity_layout.addWidget(self.opacity_slider)
        
        # Control de tamaño de puntos
        size_layout = QVBoxLayout()
        size_layout.addWidget(QLabel("Tamaño de puntos:"))
        self.point_size_spin = QSpinBox()
        self.point_size_spin.setRange(4, 20)
        self.point_size_spin.setValue(8)
        size_layout.addWidget(self.point_size_spin)
        
        controls_layout.addWidget(self.show_all_lines_cb)
        controls_layout.addWidget(self.show_selected_only_cb)
        controls_layout.addWidget(self.show_points_cb)
        controls_layout.addWidget(self.show_labels_cb)
        controls_layout.addLayout(opacity_layout)
        controls_layout.addLayout(size_layout)
        
        # Área principal de visualización
        main_area = QHBoxLayout()
        
        # Imagen izquierda
        left_panel = QVBoxLayout()
        left_panel.addWidget(QLabel("Muestra 1"))
        self.left_image = InteractiveMatchingLabel(is_left_image=True)
        left_panel.addWidget(self.left_image)
        
        # Área de líneas de conexión
        self.connection_lines = ConnectionLinesWidget()
        
        # Imagen derecha
        right_panel = QVBoxLayout()
        right_panel.addWidget(QLabel("Muestra 2"))
        self.right_image = InteractiveMatchingLabel(is_left_image=False)
        right_panel.addWidget(self.right_image)
        
        main_area.addLayout(left_panel, 2)
        main_area.addWidget(self.connection_lines, 1)
        main_area.addLayout(right_panel, 2)
        
        # Panel de información
        info_group = QGroupBox("Información de Coincidencias")
        info_layout = QVBoxLayout(info_group)
        
        self.info_label = QLabel("Pase el cursor sobre un punto para ver detalles")
        self.info_label.setWordWrap(True)
        self.info_label.setStyleSheet("padding: 10px; background-color: #f9f9f9; border: 1px solid #ddd;")
        
        info_layout.addWidget(self.info_label)
        
        # Agregar todo al layout principal
        layout.addWidget(controls_group)
        layout.addLayout(main_area, 1)
        layout.addWidget(info_group)
        
    def connect_signals(self):
        """Conecta las señales entre componentes"""
        # Señales de hover
        self.left_image.pointHovered.connect(self.on_point_hovered)
        self.right_image.pointHovered.connect(self.on_point_hovered)
        
        # Señales de clic
        self.left_image.pointClicked.connect(self.on_point_clicked)
        self.right_image.pointClicked.connect(self.on_point_clicked)
        
        # Controles
        self.show_all_lines_cb.toggled.connect(self.update_line_visibility)
        self.show_selected_only_cb.toggled.connect(self.update_line_visibility)
        self.show_points_cb.toggled.connect(self.update_point_visibility)
        self.show_labels_cb.toggled.connect(self.update_label_visibility)
        self.opacity_slider.valueChanged.connect(self.update_opacity)
        self.point_size_spin.valueChanged.connect(self.update_point_size)
        
    def set_images_and_matches(self, image1, image2, matching_points):
        """Establece las imágenes y los puntos de coincidencia"""
        self.left_image.setPixmap(image1.scaled(400, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.right_image.setPixmap(image2.scaled(400, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        
        self.matching_points = matching_points
        self.left_image.set_matching_points(matching_points)
        self.right_image.set_matching_points(matching_points)
        self.connection_lines.set_matching_points(matching_points)
        
    def on_point_hovered(self, point):
        """Maneja el evento de hover sobre un punto"""
        # Actualizar visualización
        self.left_image.set_hovered_point(point)
        self.right_image.set_hovered_point(point)
        self.connection_lines.set_hovered_point(point)
        
        # Actualizar información
        if point:
            info_text = f"""
            <b>Tipo de Característica:</b> {point.feature_type}<br>
            <b>Score de Similitud:</b> {point.similarity_score:.4f}<br>
            <b>Posición Muestra 1:</b> ({point.point1.x()}, {point.point1.y()})<br>
            <b>Posición Muestra 2:</b> ({point.point2.x()}, {point.point2.y()})<br>
            <b>Calidad:</b> {'Alta' if point.similarity_score >= 0.8 else 'Media' if point.similarity_score >= 0.6 else 'Baja'}
            """
            self.info_label.setText(info_text)
        else:
            self.info_label.setText("Pase el cursor sobre un punto para ver detalles")
            
    def on_point_clicked(self, point):
        """Maneja el clic sobre un punto"""
        # Actualizar selección en todas las vistas
        selected_points = list(self.left_image.selected_points)
        self.connection_lines.set_selected_points(selected_points)
        
        # Mostrar información detallada
        if point in selected_points:
            self.info_label.setText(f"<b>Punto seleccionado:</b> {point.feature_type} (Score: {point.similarity_score:.4f})")
            
    def update_line_visibility(self):
        """Actualiza la visibilidad de las líneas"""
        self.connection_lines.show_all_lines = self.show_all_lines_cb.isChecked()
        self.connection_lines.show_selected_only = self.show_selected_only_cb.isChecked()
        self.connection_lines.update()
        
    def update_point_visibility(self):
        """Actualiza la visibilidad de los puntos"""
        show_points = self.show_points_cb.isChecked()
        self.left_image.show_points = show_points
        self.right_image.show_points = show_points
        self.left_image.update()
        self.right_image.update()
        
    def update_label_visibility(self):
        """Actualiza la visibilidad de las etiquetas"""
        show_labels = self.show_labels_cb.isChecked()
        self.left_image.show_labels = show_labels
        self.right_image.show_labels = show_labels
        self.left_image.update()
        self.right_image.update()
        
    def update_opacity(self, value):
        """Actualiza la opacidad de las líneas"""
        self.connection_lines.line_opacity = value / 100.0
        self.connection_lines.update()
        
    def update_point_size(self, size):
        """Actualiza el tamaño de los puntos"""
        self.left_image.point_size = size
        self.right_image.point_size = size
        self.left_image.update()
        self.right_image.update()
        
    def generate_sample_matches(self):
        """Genera puntos de coincidencia de ejemplo para pruebas"""
        sample_points = []
        
        # Generar algunos puntos de ejemplo
        for i in range(10):
            x1 = np.random.randint(50, 350)
            y1 = np.random.randint(50, 350)
            x2 = np.random.randint(50, 350)
            y2 = np.random.randint(50, 350)
            
            score = np.random.uniform(0.4, 0.95)
            feature_types = ["striation", "impression", "breach_face", "firing_pin"]
            feature_type = np.random.choice(feature_types)
            
            point = MatchingPoint(
                QPoint(x1, y1), QPoint(x2, y2), 
                score, feature_type
            )
            sample_points.append(point)
            
        return sample_points