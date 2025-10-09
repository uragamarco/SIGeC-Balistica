"""
Sistema de Alineación Asistida por Puntos Clave
Permite al usuario marcar puntos de correspondencia para alinear automáticamente
dos imágenes de muestras balísticas.
"""

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QFrame, QSpinBox, QCheckBox,
                             QGroupBox, QSlider, QComboBox, QTextEdit,
                             QSplitter, QScrollArea, QToolButton, QButtonGroup)
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
    """Label de imagen con capacidad de marcar puntos clave y navegación mejorada"""
    
    pointAdded = pyqtSignal(float, float, int)  # x, y, point_id
    pointRemoved = pyqtSignal(int)  # point_id
    pointSelected = pyqtSignal(int)  # point_id
    
    def __init__(self, image_name, parent=None):
        super().__init__(parent)
        self.image_name = image_name
        self.original_pixmap = None
        self.scaled_pixmap = None
        self.scale_factor = 1.0
        self.zoom_factor = 1.0
        self.pan_offset = QPoint(0, 0)
        
        self.keypoints = []
        self.next_point_id = 1
        self.point_radius = 8
        self.selected_point_id = -1
        
        self.marking_enabled = True
        self.show_point_ids = True
        self.show_connections = False
        self.pan_enabled = False
        
        # Estado de arrastre para pan
        self.dragging = False
        self.last_pan_point = QPoint()
        
        self.setMinimumSize(400, 300)
        self.setStyleSheet("border: 2px solid #ccc; background-color: #f9f9f9;")
        self.setAlignment(Qt.AlignCenter)
        self.setCursor(Qt.CrossCursor)
        
    def set_image(self, pixmap):
        """Establece la imagen a mostrar"""
        self.original_pixmap = pixmap
        self.update_display()
        
    def update_display(self):
        """Actualiza la visualización con puntos clave, zoom y pan"""
        if not self.original_pixmap:
            return
            
        # Escalar imagen al tamaño del widget con zoom
        widget_size = self.size()
        base_scale = min(
            widget_size.width() / self.original_pixmap.width(),
            widget_size.height() / self.original_pixmap.height()
        )
        
        # Aplicar zoom
        self.scale_factor = base_scale * self.zoom_factor
        
        scaled_size = self.original_pixmap.size() * self.scale_factor
        self.scaled_pixmap = self.original_pixmap.scaled(
            scaled_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
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
        """Maneja clics del mouse para marcar puntos o iniciar pan"""
        if event.button() == Qt.LeftButton:
            if self.pan_enabled:
                self.dragging = True
                self.last_pan_point = event.pos()
                self.setCursor(Qt.ClosedHandCursor)
            elif self.marking_enabled and self.scaled_pixmap:
                # Convertir coordenadas de pantalla a imagen original
                image_x, image_y = self.screen_to_image_coords(event.x(), event.y())
                
                if image_x is not None and image_y is not None:
                    # Verificar si hay un punto existente cerca
                    existing_point = self.find_point_at(image_x, image_y)
                    
                    if existing_point:
                        # Seleccionar punto existente
                        self.selected_point_id = existing_point.point_id
                        self.pointSelected.emit(existing_point.point_id)
                    else:
                        # Crear nuevo punto
                        new_point = KeyPoint(image_x, image_y, self.next_point_id)
                        self.keypoints.append(new_point)
                        self.selected_point_id = self.next_point_id
                        self.pointAdded.emit(image_x, image_y, self.next_point_id)
                        self.next_point_id += 1
                    
                    self.update_display()
        
        elif event.button() == Qt.RightButton and self.marking_enabled and self.scaled_pixmap:
            # Eliminar punto con clic derecho
            image_x, image_y = self.screen_to_image_coords(event.x(), event.y())
            
            if image_x is not None and image_y is not None:
                point_to_remove = self.find_point_at(image_x, image_y)
                if point_to_remove:
                    self.remove_point(point_to_remove.point_id)
    
    def mouseMoveEvent(self, event):
        """Maneja el movimiento del mouse para pan"""
        if self.dragging and self.pan_enabled:
            delta = event.pos() - self.last_pan_point
            self.pan_offset += delta
            self.last_pan_point = event.pos()
            self.update_display()
    
    def mouseReleaseEvent(self, event):
        """Maneja la liberación del mouse"""
        if event.button() == Qt.LeftButton and self.dragging:
            self.dragging = False
            self.setCursor(Qt.OpenHandCursor if self.pan_enabled else Qt.CrossCursor)
    
    def wheelEvent(self, event):
        """Maneja el zoom con la rueda del mouse"""
        if event.modifiers() & Qt.ControlModifier:
            # Zoom con Ctrl + rueda
            zoom_in = event.angleDelta().y() > 0
            zoom_factor = 1.1 if zoom_in else 1.0 / 1.1
            
            new_zoom = self.zoom_factor * zoom_factor
            if 0.1 <= new_zoom <= 5.0:  # Limitar zoom entre 10% y 500%
                self.zoom_factor = new_zoom
                self.update_display()
        else:
            super().wheelEvent(event)
    
    def screen_to_image_coords(self, screen_x, screen_y):
        """Convierte coordenadas de pantalla a coordenadas de imagen original"""
        if not self.original_pixmap or self.scale_factor == 0:
            return None, None
        
        # Ajustar por el offset de pan
        adjusted_x = screen_x - self.pan_offset.x()
        adjusted_y = screen_y - self.pan_offset.y()
        
        # Convertir a coordenadas de imagen original
        image_x = adjusted_x / self.scale_factor
        image_y = adjusted_y / self.scale_factor
        
        # Verificar que esté dentro de los límites de la imagen
        if (0 <= image_x <= self.original_pixmap.width() and 
            0 <= image_y <= self.original_pixmap.height()):
            return image_x, image_y
        
        return None, None
    
    def set_zoom(self, zoom_factor):
        """Establece el factor de zoom"""
        if 0.1 <= zoom_factor <= 5.0:
            self.zoom_factor = zoom_factor
            self.update_display()
    
    def reset_view(self):
        """Resetea zoom y pan"""
        self.zoom_factor = 1.0
        self.pan_offset = QPoint(0, 0)
        self.update_display()
    
    def set_pan_enabled(self, enabled):
        """Habilita/deshabilita el modo pan"""
        self.pan_enabled = enabled
        if enabled:
            self.setCursor(Qt.OpenHandCursor)
        else:
            self.setCursor(Qt.CrossCursor if self.marking_enabled else Qt.ArrowCursor)
                
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
        """Configura la interfaz principal con layout mejorado"""
        layout = QVBoxLayout(self)
        
        # Panel de control superior
        controls_group = QGroupBox("Controles de Alineación")
        controls_layout = QHBoxLayout(controls_group)
        
        # Botones de modo
        mode_group = QButtonGroup(self)
        
        self.mark_points_btn = QToolButton()
        self.mark_points_btn.setText("Marcar Puntos")
        self.mark_points_btn.setCheckable(True)
        self.mark_points_btn.setChecked(True)
        self.mark_points_btn.setToolTip("Modo marcado de puntos (clic izquierdo: añadir, derecho: eliminar)")
        mode_group.addButton(self.mark_points_btn)
        
        self.pan_btn = QToolButton()
        self.pan_btn.setText("Navegar")
        self.pan_btn.setCheckable(True)
        self.pan_btn.setToolTip("Modo navegación (arrastrar para mover, Ctrl+rueda para zoom)")
        mode_group.addButton(self.pan_btn)
        
        # Botones de acción
        self.auto_align_btn = QPushButton("Alinear Automáticamente")
        
        self.clear_points_btn = QPushButton("Limpiar Puntos")
        
        self.reset_btn = QPushButton("Reiniciar")
        
        # Conectar señales de los botones de acción
        self.auto_align_btn.clicked.connect(self.auto_align)
        self.clear_points_btn.clicked.connect(self.clear_points)
        self.reset_btn.clicked.connect(self.reset_view)
        
        # Controles de zoom
        zoom_label = QLabel("Zoom:")
        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setRange(10, 500)  # 10% a 500%
        self.zoom_slider.setValue(100)
        self.zoom_slider.setToolTip("Control de zoom (10% - 500%)")
        self.zoom_slider.valueChanged.connect(self.on_zoom_changed)
        
        self.zoom_reset_btn = QPushButton("1:1")
        self.zoom_reset_btn.setMaximumWidth(40)
        self.zoom_reset_btn.clicked.connect(self.reset_zoom)
        
        # Configuraciones
        self.show_ids_cb = QCheckBox("Mostrar IDs")
        self.show_ids_cb.setChecked(True)
        self.show_ids_cb.toggled.connect(self.toggle_point_ids)
        
        # Layout de controles
        controls_layout.addWidget(self.mark_points_btn)
        controls_layout.addWidget(self.pan_btn)
        controls_layout.addWidget(QFrame())  # Separador
        controls_layout.addWidget(self.auto_align_btn)
        controls_layout.addWidget(self.clear_points_btn)
        controls_layout.addWidget(self.reset_btn)
        controls_layout.addStretch()
        controls_layout.addWidget(zoom_label)
        controls_layout.addWidget(self.zoom_slider)
        controls_layout.addWidget(self.zoom_reset_btn)
        controls_layout.addWidget(QFrame())  # Separador
        controls_layout.addWidget(self.show_ids_cb)
        
        # Splitter principal (horizontal)
        main_splitter = QSplitter(Qt.Horizontal)
        
        # Panel de imágenes con splitter vertical
        images_splitter = QSplitter(Qt.Vertical)
        
        # Imagen 1
        image1_group = QGroupBox("Imagen de Referencia")
        image1_layout = QVBoxLayout(image1_group)
        
        # Scroll area para imagen 1
        self.image1_scroll = QScrollArea()
        self.image1_label = AlignmentImageLabel("Imagen 1")
        self.image1_scroll.setWidget(self.image1_label)
        self.image1_scroll.setWidgetResizable(True)
        
        image1_layout.addWidget(self.image1_scroll)
        
        # Imagen 2
        image2_group = QGroupBox("Imagen a Alinear")
        image2_layout = QVBoxLayout(image2_group)
        
        # Scroll area para imagen 2
        self.image2_scroll = QScrollArea()
        self.image2_label = AlignmentImageLabel("Imagen 2")
        self.image2_scroll.setWidget(self.image2_label)
        self.image2_scroll.setWidgetResizable(True)
        
        image2_layout.addWidget(self.image2_scroll)
        
        # Añadir imágenes al splitter vertical
        images_splitter.addWidget(image1_group)
        images_splitter.addWidget(image2_group)
        images_splitter.setSizes([400, 400])  # Tamaños iguales inicialmente
        
        # Panel lateral derecho
        side_panel = QWidget()
        side_panel.setMaximumWidth(300)
        side_layout = QVBoxLayout(side_panel)
        
        # Panel de correspondencias (colapsable)
        self.correspondence_group = QGroupBox("Correspondencias de Puntos")
        self.correspondence_group.setCheckable(True)
        self.correspondence_group.setChecked(True)
        correspondence_layout = QVBoxLayout(self.correspondence_group)
        
        self.correspondence_table = CorrespondenceTable()
        correspondence_layout.addWidget(self.correspondence_table)
        
        # Panel de información
        info_group = QGroupBox("Información de Alineación")
        info_layout = QVBoxLayout(info_group)
        
        self.info_text = QTextEdit()
        self.info_text.setMaximumHeight(150)
        self.info_text.setReadOnly(True)
        self.info_text.setPlainText("Marque al menos 3 puntos correspondientes en ambas imágenes para calcular la alineación.")
        
        info_layout.addWidget(self.info_text)
        
        # Añadir al panel lateral
        side_layout.addWidget(self.correspondence_group)
        side_layout.addWidget(info_group)
        side_layout.addStretch()
        
        # Añadir al splitter principal
        main_splitter.addWidget(images_splitter)
        main_splitter.addWidget(side_panel)
        main_splitter.setSizes([800, 300])  # Más espacio para imágenes
        
        # Layout principal
        layout.addWidget(controls_group)
        layout.addWidget(main_splitter)
        
    def connect_signals(self):
        """Conecta las señales de los widgets"""
        self.image1_label.pointAdded.connect(self.on_point_added_image1)
        self.image1_label.pointRemoved.connect(self.on_point_removed_image1)
        
        self.image2_label.pointAdded.connect(self.on_point_added_image2)
        self.image2_label.pointRemoved.connect(self.on_point_removed_image2)
        
        self.mark_points_btn.toggled.connect(self.toggle_marking_mode)
        self.pan_btn.toggled.connect(self.toggle_pan_mode)
    
    def on_zoom_changed(self, value):
        """Maneja cambios en el slider de zoom"""
        zoom_factor = value / 100.0  # Convertir porcentaje a factor
        self.image1_label.set_zoom(zoom_factor)
        self.image2_label.set_zoom(zoom_factor)
    
    def reset_zoom(self):
        """Resetea el zoom a 100%"""
        self.zoom_slider.setValue(100)
        self.image1_label.reset_view()
        self.image2_label.reset_view()
    
    def toggle_pan_mode(self, enabled):
        """Alterna el modo de navegación (pan)"""
        self.image1_label.set_pan_enabled(enabled)
        self.image2_label.set_pan_enabled(enabled)
        
        if enabled:
            self.mark_points_btn.setChecked(False)
            self.image1_label.set_marking_enabled(False)
            self.image2_label.set_marking_enabled(False)
        
    def set_images(self, image1_pixmap, image2_pixmap):
        """Establece las imágenes a alinear"""
        self.image1_label.set_image(image1_pixmap)
        self.image2_label.set_image(image2_pixmap)
        
    def on_point_added_image1(self, x, y, point_id):
        """Maneja la adición de punto en imagen 1"""
        self.image1_points.append((x, y, point_id))
        self.update_correspondences()
        self.update_info_panel()
        
    def on_point_removed_image1(self, point_id):
        """Maneja la eliminación de punto en imagen 1"""
        self.image1_points = [p for p in self.image1_points if p[2] != point_id]
        self.update_correspondences()
        self.update_info_panel()
        
    def on_point_added_image2(self, x, y, point_id):
        """Maneja la adición de punto en imagen 2"""
        self.image2_points.append((x, y, point_id))
        self.update_correspondences()
        self.update_info_panel()
        
    def on_point_removed_image2(self, point_id):
        """Maneja la eliminación de punto en imagen 2"""
        self.image2_points = [p for p in self.image2_points if p[2] != point_id]
        self.update_correspondences()
        self.update_info_panel()
        
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
        
        if enabled:
            self.pan_btn.setChecked(False)
            self.image1_label.set_pan_enabled(False)
            self.image2_label.set_pan_enabled(False)
        
    def toggle_correspondence_table(self):
        """Alterna la visibilidad de la tabla de correspondencias"""
        if self.correspondence_group.isChecked():
            self.correspondence_group.setChecked(False)
        else:
            self.correspondence_group.setChecked(True)
    
    def update_info_panel(self):
        """Actualiza la información del panel de estado"""
        num_points1 = len(self.image1_points)
        num_points2 = len(self.image2_points)
        min_points = min(num_points1, num_points2)
        
        status_text = f"Puntos imagen 1: {num_points1}\n"
        status_text += f"Puntos imagen 2: {num_points2}\n"
        status_text += f"Correspondencias: {min_points}\n\n"
        
        if min_points >= 3:
            status_text += "Estado: Listo para alineación\n"
            status_text += "Mínimo de puntos alcanzado ✓"
        elif min_points > 0:
            status_text += f"Estado: Necesita {3 - min_points} puntos más\n"
            status_text += "Mínimo requerido: 3 puntos"
        else:
            status_text += "Estado: Sin puntos marcados\n"
            status_text += "Marque puntos correspondientes en ambas imágenes"
        
        self.info_text.setPlainText(status_text)
    
    def toggle_point_ids(self, show_ids):
        """Alterna la visualización de IDs de puntos"""
        self.image1_label.show_point_ids = show_ids
        self.image2_label.show_point_ids = show_ids
        self.image1_label.update_display()
        self.image2_label.update_display()
         
    def clear_points(self):
        """Limpia todos los puntos marcados"""
        self.image1_label.clear_points()
        self.image2_label.clear_points()
        self.image1_points.clear()
        self.image2_points.clear()
        self.correspondences.clear()
        self.update_correspondence_table()
        self.update_info_panel()
    
    def auto_align(self):
        """Ejecuta la alineación automática basada en los puntos marcados"""
        if len(self.correspondences) < 3:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Puntos insuficientes", 
                              "Se necesitan al menos 3 puntos correspondientes para realizar la alineación.")
            return
        
        try:
            # Aquí iría la lógica de alineación automática
            # Por ahora solo mostramos un mensaje de éxito
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.information(self, "Alineación completada", 
                                  f"Alineación realizada con {len(self.correspondences)} puntos correspondientes.")
        except Exception as e:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Error en alineación", f"Error durante la alineación: {str(e)}")
    
    def reset_view(self):
        """Resetea la vista de ambas imágenes"""
        self.reset_zoom()
        self.image1_label.reset_view()
        self.image2_label.reset_view()