"""
Sistema de Alineación Asistida por Puntos Clave
Permite al usuario marcar puntos de correspondencia para alinear automáticamente
dos imágenes de muestras balísticas.
"""

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QFrame, QSpinBox, QCheckBox,
                             QGroupBox, QSlider, QComboBox, QTextEdit)
from PyQt5.QtCore import Qt, pyqtSignal, QPoint, QPointF, QTimer
from PyQt5.QtGui import (QPixmap, QPainter, QPen, QColor, QFont, QBrush, 
                         QCursor, QTransform, QPolygonF)
import numpy as np
import cv2


class KeyPoint:
    """Representa un punto clave marcado por el usuario"""
    
    def __init__(self, x, y, point_id, description=""):
        self.x = x
        self.y = y
        self.point_id = point_id
        self.description = description
        self.is_selected = False
        
    def distance_to(self, x, y):
        """Calcula la distancia a otro punto"""
        return np.sqrt((self.x - x)**2 + (self.y - y)**2)


class AlignmentImageLabel(QLabel):
    """Label de imagen con capacidad de marcar puntos clave"""
    
    pointAdded = pyqtSignal(float, float, int)  # x, y, point_id
    pointRemoved = pyqtSignal(int)  # point_id
    pointSelected = pyqtSignal(int)  # point_id
    
    def __init__(self, image_name, parent=None):
        super().__init__(parent)
        self.image_name = image_name
        self.original_pixmap = None
        self.scaled_pixmap = None
        self.scale_factor = 1.0
        
        self.keypoints = []
        self.next_point_id = 1
        self.point_radius = 8
        self.selected_point_id = -1
        
        self.marking_enabled = True
        self.show_point_ids = True
        self.show_connections = False
        
        self.setMinimumSize(400, 300)
        self.setStyleSheet("border: 2px solid #ccc; background-color: #f9f9f9;")
        self.setAlignment(Qt.AlignCenter)
        self.setCursor(Qt.CrossCursor)
        
    def set_image(self, pixmap):
        """Establece la imagen a mostrar"""
        self.original_pixmap = pixmap
        self.update_display()
        
    def update_display(self):
        """Actualiza la visualización con puntos clave"""
        if not self.original_pixmap:
            return
            
        # Escalar imagen al tamaño del widget
        widget_size = self.size()
        self.scaled_pixmap = self.original_pixmap.scaled(
            widget_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        
        # Calcular factor de escala
        self.scale_factor = min(
            widget_size.width() / self.original_pixmap.width(),
            widget_size.height() / self.original_pixmap.height()
        )
        
        # Crear pixmap con puntos dibujados
        display_pixmap = QPixmap(self.scaled_pixmap)
        painter = QPainter(display_pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        
        self.draw_keypoints(painter)
        
        painter.end()
        self.setPixmap(display_pixmap)
        
    def draw_keypoints(self, painter):
        """Dibuja los puntos clave en la imagen"""
        for point in self.keypoints:
            # Convertir coordenadas a escala de visualización
            display_x = point.x * self.scale_factor
            display_y = point.y * self.scale_factor
            
            # Color del punto
            if point.point_id == self.selected_point_id:
                color = QColor(255, 0, 0)  # Rojo para seleccionado
                radius = self.point_radius + 2
            else:
                color = QColor(0, 255, 0)  # Verde para normal
                radius = self.point_radius
                
            # Dibujar círculo del punto
            painter.setPen(QPen(color, 2))
            painter.setBrush(QBrush(color, Qt.SolidPattern))
            painter.drawEllipse(QPoint(int(display_x), int(display_y)), radius, radius)
            
            # Dibujar ID del punto si está habilitado
            if self.show_point_ids:
                painter.setPen(QPen(QColor(255, 255, 255), 1))
                painter.setFont(QFont("Arial", 8, QFont.Bold))
                painter.drawText(
                    int(display_x - 5), int(display_y + 3), 
                    str(point.point_id)
                )
                
    def mousePressEvent(self, event):
        """Maneja clics del mouse para marcar puntos"""
        if not self.marking_enabled or not self.scaled_pixmap:
            return
            
        if event.button() == Qt.LeftButton:
            # Convertir coordenadas de visualización a imagen original
            click_x = event.x() / self.scale_factor
            click_y = event.y() / self.scale_factor
            
            # Verificar si se hizo clic cerca de un punto existente
            clicked_point = self.find_point_at(click_x, click_y)
            
            if clicked_point:
                # Seleccionar punto existente
                self.selected_point_id = clicked_point.point_id
                self.pointSelected.emit(clicked_point.point_id)
            else:
                # Crear nuevo punto
                new_point = KeyPoint(click_x, click_y, self.next_point_id)
                self.keypoints.append(new_point)
                self.selected_point_id = self.next_point_id
                
                self.pointAdded.emit(click_x, click_y, self.next_point_id)
                self.next_point_id += 1
                
            self.update_display()
            
        elif event.button() == Qt.RightButton:
            # Eliminar punto con clic derecho
            click_x = event.x() / self.scale_factor
            click_y = event.y() / self.scale_factor
            
            clicked_point = self.find_point_at(click_x, click_y)
            if clicked_point:
                self.remove_point(clicked_point.point_id)
                
    def find_point_at(self, x, y):
        """Encuentra un punto cerca de las coordenadas dadas"""
        for point in self.keypoints:
            if point.distance_to(x, y) <= self.point_radius / self.scale_factor:
                return point
        return None
        
    def remove_point(self, point_id):
        """Elimina un punto por su ID"""
        self.keypoints = [p for p in self.keypoints if p.point_id != point_id]
        if self.selected_point_id == point_id:
            self.selected_point_id = -1
        self.pointRemoved.emit(point_id)
        self.update_display()
        
    def clear_points(self):
        """Elimina todos los puntos"""
        self.keypoints.clear()
        self.selected_point_id = -1
        self.next_point_id = 1
        self.update_display()
        
    def get_keypoints(self):
        """Retorna lista de puntos clave"""
        return [(p.x, p.y, p.point_id) for p in self.keypoints]
        
    def set_marking_enabled(self, enabled):
        """Habilita/deshabilita el marcado de puntos"""
        self.marking_enabled = enabled
        self.setCursor(Qt.CrossCursor if enabled else Qt.ArrowCursor)
        
    def resizeEvent(self, event):
        """Maneja el redimensionamiento del widget"""
        super().resizeEvent(event)
        if self.original_pixmap:
            self.update_display()


class CorrespondenceTable(QWidget):
    """Tabla para mostrar correspondencias entre puntos"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.correspondences = []
        self.setup_ui()
        
    def setup_ui(self):
        """Configura la interfaz de la tabla"""
        layout = QVBoxLayout(self)
        
        # Título
        title = QLabel("Correspondencias de Puntos")
        title.setFont(QFont("Arial", 12, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        
        # Área de texto para mostrar correspondencias
        self.correspondence_text = QTextEdit()
        self.correspondence_text.setMaximumHeight(150)
        self.correspondence_text.setReadOnly(True)
        
        layout.addWidget(title)
        layout.addWidget(self.correspondence_text)
        
    def add_correspondence(self, point1_id, point2_id, description=""):
        """Añade una correspondencia entre dos puntos"""
        correspondence = {
            'point1_id': point1_id,
            'point2_id': point2_id,
            'description': description
        }
        self.correspondences.append(correspondence)
        self.update_display()
        
    def remove_correspondence(self, point1_id, point2_id):
        """Elimina una correspondencia"""
        self.correspondences = [
            c for c in self.correspondences 
            if not (c['point1_id'] == point1_id and c['point2_id'] == point2_id)
        ]
        self.update_display()
        
    def update_display(self):
        """Actualiza la visualización de correspondencias"""
        text = ""
        for i, corr in enumerate(self.correspondences, 1):
            text += f"{i}. Punto {corr['point1_id']} ↔ Punto {corr['point2_id']}"
            if corr['description']:
                text += f" ({corr['description']})"
            text += "\n"
            
        self.correspondence_text.setPlainText(text)
        
    def clear_correspondences(self):
        """Limpia todas las correspondencias"""
        self.correspondences.clear()
        self.update_display()


class AssistedAlignmentWidget(QWidget):
    """Widget principal para alineación asistida"""
    
    alignmentComputed = pyqtSignal(object)  # transformation_matrix
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.image1_points = []
        self.image2_points = []
        self.transformation_matrix = None
        
        self.setup_ui()
        self.connect_signals()
        
    def setup_ui(self):
        """Configura la interfaz principal"""
        layout = QVBoxLayout(self)
        
        # Panel de control superior
        controls_group = QGroupBox("Controles de Alineación")
        controls_layout = QHBoxLayout(controls_group)
        
        # Botones de acción
        self.mark_points_btn = QPushButton("Marcar Puntos")
        self.mark_points_btn.setCheckable(True)
        self.mark_points_btn.setChecked(True)
        
        self.auto_align_btn = QPushButton("Alinear Automáticamente")
        self.auto_align_btn.clicked.connect(self.compute_alignment)
        
        self.clear_points_btn = QPushButton("Limpiar Puntos")
        self.clear_points_btn.clicked.connect(self.clear_all_points)
        
        self.reset_btn = QPushButton("Reiniciar")
        self.reset_btn.clicked.connect(self.reset_alignment)
        
        # Configuraciones
        self.show_ids_cb = QCheckBox("Mostrar IDs")
        self.show_ids_cb.setChecked(True)
        self.show_ids_cb.toggled.connect(self.toggle_point_ids)
        
        controls_layout.addWidget(self.mark_points_btn)
        controls_layout.addWidget(self.auto_align_btn)
        controls_layout.addWidget(self.clear_points_btn)
        controls_layout.addWidget(self.reset_btn)
        controls_layout.addStretch()
        controls_layout.addWidget(self.show_ids_cb)
        
        # Panel de imágenes
        images_layout = QHBoxLayout()
        
        # Imagen 1
        image1_group = QGroupBox("Imagen de Referencia")
        image1_layout = QVBoxLayout(image1_group)
        
        self.image1_label = AlignmentImageLabel("Imagen 1")
        image1_layout.addWidget(self.image1_label)
        
        # Imagen 2
        image2_group = QGroupBox("Imagen a Alinear")
        image2_layout = QVBoxLayout(image2_group)
        
        self.image2_label = AlignmentImageLabel("Imagen 2")
        image2_layout.addWidget(self.image2_label)
        
        images_layout.addWidget(image1_group)
        images_layout.addWidget(image2_group)
        
        # Panel de correspondencias
        self.correspondence_table = CorrespondenceTable()
        
        # Panel de información
        info_group = QGroupBox("Información de Alineación")
        info_layout = QVBoxLayout(info_group)
        
        self.info_text = QTextEdit()
        self.info_text.setMaximumHeight(100)
        self.info_text.setReadOnly(True)
        self.info_text.setPlainText("Marque al menos 3 puntos correspondientes en ambas imágenes para calcular la alineación.")
        
        info_layout.addWidget(self.info_text)
        
        # Layout principal
        layout.addWidget(controls_group)
        layout.addLayout(images_layout)
        layout.addWidget(self.correspondence_table)
        layout.addWidget(info_group)
        
    def connect_signals(self):
        """Conecta las señales de los widgets"""
        self.image1_label.pointAdded.connect(self.on_point_added_image1)
        self.image1_label.pointRemoved.connect(self.on_point_removed_image1)
        
        self.image2_label.pointAdded.connect(self.on_point_added_image2)
        self.image2_label.pointRemoved.connect(self.on_point_removed_image2)
        
        self.mark_points_btn.toggled.connect(self.toggle_marking_mode)
        
    def set_images(self, image1_pixmap, image2_pixmap):
        """Establece las imágenes a alinear"""
        self.image1_label.set_image(image1_pixmap)
        self.image2_label.set_image(image2_pixmap)
        
    def on_point_added_image1(self, x, y, point_id):
        """Maneja la adición de punto en imagen 1"""
        self.image1_points.append((x, y, point_id))
        self.update_correspondences()
        
    def on_point_removed_image1(self, point_id):
        """Maneja la eliminación de punto en imagen 1"""
        self.image1_points = [p for p in self.image1_points if p[2] != point_id]
        self.update_correspondences()
        
    def on_point_added_image2(self, x, y, point_id):
        """Maneja la adición de punto en imagen 2"""
        self.image2_points.append((x, y, point_id))
        self.update_correspondences()
        
    def on_point_removed_image2(self, point_id):
        """Maneja la eliminación de punto en imagen 2"""
        self.image2_points = [p for p in self.image2_points if p[2] != point_id]
        self.update_correspondences()
        
    def update_correspondences(self):
        """Actualiza la tabla de correspondencias"""
        self.correspondence_table.clear_correspondences()
        
        # Crear correspondencias automáticas basadas en orden de creación
        min_points = min(len(self.image1_points), len(self.image2_points))
        
        for i in range(min_points):
            point1_id = self.image1_points[i][2]
            point2_id = self.image2_points[i][2]
            self.correspondence_table.add_correspondence(point1_id, point2_id)
            
    def compute_alignment(self):
        """Calcula la matriz de transformación para alineación"""
        if len(self.image1_points) < 3 or len(self.image2_points) < 3:
            self.info_text.setPlainText("Error: Se necesitan al menos 3 puntos en cada imagen.")
            return
            
        if len(self.image1_points) != len(self.image2_points):
            self.info_text.setPlainText("Error: Debe haber el mismo número de puntos en ambas imágenes.")
            return
            
        try:
            # Preparar puntos para OpenCV
            src_points = np.array([(p[0], p[1]) for p in self.image1_points], dtype=np.float32)
            dst_points = np.array([(p[0], p[1]) for p in self.image2_points], dtype=np.float32)
            
            # Calcular transformación afín
            self.transformation_matrix = cv2.getAffineTransform(src_points[:3], dst_points[:3])
            
            # Si hay más de 3 puntos, usar transformación de perspectiva
            if len(src_points) >= 4:
                self.transformation_matrix = cv2.getPerspectiveTransform(src_points[:4], dst_points[:4])
                
            # Calcular error de alineación
            error = self.calculate_alignment_error(src_points, dst_points)
            
            info_text = f"Alineación calculada exitosamente.\n"
            info_text += f"Puntos utilizados: {len(src_points)}\n"
            info_text += f"Error promedio: {error:.2f} píxeles"
            
            self.info_text.setPlainText(info_text)
            self.alignmentComputed.emit(self.transformation_matrix)
            
        except Exception as e:
            self.info_text.setPlainText(f"Error al calcular alineación: {str(e)}")
            
    def calculate_alignment_error(self, src_points, dst_points):
        """Calcula el error promedio de alineación"""
        if self.transformation_matrix is None:
            return float('inf')
            
        # Transformar puntos fuente
        if self.transformation_matrix.shape[0] == 2:  # Transformación afín
            ones = np.ones((src_points.shape[0], 1))
            src_homogeneous = np.hstack([src_points, ones])
            transformed_points = (self.transformation_matrix @ src_homogeneous.T).T
        else:  # Transformación de perspectiva
            transformed_points = cv2.perspectiveTransform(
                src_points.reshape(-1, 1, 2), self.transformation_matrix
            ).reshape(-1, 2)
            
        # Calcular distancias
        distances = np.linalg.norm(transformed_points - dst_points, axis=1)
        return np.mean(distances)
        
    def clear_all_points(self):
        """Limpia todos los puntos marcados"""
        self.image1_label.clear_points()
        self.image2_label.clear_points()
        self.image1_points.clear()
        self.image2_points.clear()
        self.correspondence_table.clear_correspondences()
        self.info_text.setPlainText("Puntos eliminados. Marque nuevos puntos para alinear.")
        
    def reset_alignment(self):
        """Reinicia completamente la alineación"""
        self.clear_all_points()
        self.transformation_matrix = None
        self.info_text.setPlainText("Alineación reiniciada. Marque puntos correspondientes en ambas imágenes.")
        
    def toggle_marking_mode(self, enabled):
        """Alterna el modo de marcado de puntos"""
        self.image1_label.set_marking_enabled(enabled)
        self.image2_label.set_marking_enabled(enabled)
        
    def toggle_point_ids(self, show_ids):
        """Alterna la visualización de IDs de puntos"""
        self.image1_label.show_point_ids = show_ids
        self.image2_label.show_point_ids = show_ids
        self.image1_label.update_display()
        self.image2_label.update_display()
        
    def get_transformation_matrix(self):
        """Retorna la matriz de transformación calculada"""
        return self.transformation_matrix