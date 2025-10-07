"""
Widget de minimapa para navegación en imágenes de alta resolución
"""

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QFrame, QSizePolicy)
from PyQt5.QtCore import Qt, pyqtSignal, QRect, QPoint, QSize
from PyQt5.QtGui import QPixmap, QPainter, QColor, QPen, QBrush


class MinimapWidget(QFrame):
    """Widget de minimapa para navegación en imágenes grandes"""
    
    viewportChanged = pyqtSignal(object)  # QRect - nueva área visible
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.original_pixmap = None
        self.viewport_rect = QRect()
        self.minimap_size = QSize(200, 150)
        self.scale_factor = 1.0
        self.dragging = False
        self.drag_start = QPoint()
        self.setup_ui()
        
    def setup_ui(self):
        """Configura la interfaz del minimapa"""
        self.setProperty("class", "card")
        self.setFixedSize(self.minimap_size.width() + 20, self.minimap_size.height() + 50)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Título
        title_label = QLabel("Minimapa")
        title_label.setStyleSheet("font-weight: bold; color: #333; font-size: 12px;")
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        # Área del minimapa
        self.minimap_label = MinimapLabel(self.minimap_size)
        self.minimap_label.setStyleSheet("border: 1px solid #ccc; background-color: #f9f9f9;")
        self.minimap_label.mousePressed.connect(self.on_minimap_click)
        self.minimap_label.mouseDragged.connect(self.on_minimap_drag)
        self.minimap_label.mouseReleased.connect(self.on_minimap_release)
        layout.addWidget(self.minimap_label)
        
        # Información de zoom
        self.info_label = QLabel("Sin imagen")
        self.info_label.setStyleSheet("color: #666; font-size: 10px;")
        self.info_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.info_label)
        
    def set_image(self, pixmap):
        """Establece la imagen para el minimapa"""
        self.original_pixmap = pixmap
        if pixmap and not pixmap.isNull():
            # Calcular factor de escala para el minimapa
            image_size = pixmap.size()
            self.scale_factor = min(
                self.minimap_size.width() / image_size.width(),
                self.minimap_size.height() / image_size.height()
            )
            
            # Crear miniatura
            thumbnail = pixmap.scaled(
                self.minimap_size,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            
            self.minimap_label.set_thumbnail(thumbnail)
            self.update_info()
        else:
            self.minimap_label.clear()
            self.info_label.setText("Sin imagen")
            
    def set_viewport(self, viewport_rect, image_size, zoom_factor):
        """Actualiza el rectángulo del viewport visible"""
        if not self.original_pixmap or self.original_pixmap.isNull():
            return
            
        # Convertir viewport a coordenadas del minimapa
        minimap_rect = QRect(
            int(viewport_rect.x() * self.scale_factor),
            int(viewport_rect.y() * self.scale_factor),
            int(viewport_rect.width() * self.scale_factor),
            int(viewport_rect.height() * self.scale_factor)
        )
        
        self.viewport_rect = minimap_rect
        self.minimap_label.set_viewport_rect(minimap_rect)
        
        # Actualizar información
        self.info_label.setText(f"Zoom: {int(zoom_factor * 100)}%")
        
    def update_info(self):
        """Actualiza la información mostrada"""
        if self.original_pixmap and not self.original_pixmap.isNull():
            size = self.original_pixmap.size()
            self.info_label.setText(f"{size.width()}x{size.height()}")
        else:
            self.info_label.setText("Sin imagen")
            
    def on_minimap_click(self, pos):
        """Maneja clicks en el minimapa"""
        if not self.original_pixmap or self.original_pixmap.isNull():
            return
            
        self.dragging = True
        self.drag_start = pos
        self.move_viewport_to(pos)
        
    def on_minimap_drag(self, pos):
        """Maneja arrastre en el minimapa"""
        if self.dragging:
            self.move_viewport_to(pos)
            
    def on_minimap_release(self, pos):
        """Maneja liberación del mouse en el minimapa"""
        self.dragging = False
        
    def move_viewport_to(self, minimap_pos):
        """Mueve el viewport a la posición especificada en el minimapa"""
        if not self.original_pixmap or self.original_pixmap.isNull():
            return
            
        # Convertir posición del minimapa a coordenadas de imagen original
        original_x = minimap_pos.x() / self.scale_factor
        original_y = minimap_pos.y() / self.scale_factor
        
        # Centrar el viewport en esta posición
        viewport_width = self.viewport_rect.width() / self.scale_factor
        viewport_height = self.viewport_rect.height() / self.scale_factor
        
        new_viewport = QRect(
            int(original_x - viewport_width / 2),
            int(original_y - viewport_height / 2),
            int(viewport_width),
            int(viewport_height)
        )
        
        # Asegurar que el viewport esté dentro de los límites de la imagen
        image_size = self.original_pixmap.size()
        new_viewport = new_viewport.intersected(QRect(0, 0, image_size.width(), image_size.height()))
        
        # Emitir señal de cambio de viewport
        self.viewportChanged.emit(new_viewport)


class MinimapLabel(QLabel):
    """Label personalizado para el minimapa con manejo de eventos"""
    
    mousePressed = pyqtSignal(object)  # QPoint
    mouseDragged = pyqtSignal(object)  # QPoint
    mouseReleased = pyqtSignal(object) # QPoint
    
    def __init__(self, size):
        super().__init__()
        self.setFixedSize(size)
        self.thumbnail = None
        self.viewport_rect = QRect()
        self.setMouseTracking(True)
        
    def set_thumbnail(self, pixmap):
        """Establece la miniatura"""
        self.thumbnail = pixmap
        self.update()
        
    def set_viewport_rect(self, rect):
        """Establece el rectángulo del viewport"""
        self.viewport_rect = rect
        self.update()
        
    def paintEvent(self, event):
        """Dibuja el minimapa con el viewport"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Dibujar fondo
        painter.fillRect(self.rect(), QColor("#f9f9f9"))
        
        # Dibujar miniatura si existe
        if self.thumbnail and not self.thumbnail.isNull():
            # Centrar la miniatura
            thumb_rect = self.thumbnail.rect()
            x = (self.width() - thumb_rect.width()) // 2
            y = (self.height() - thumb_rect.height()) // 2
            painter.drawPixmap(x, y, self.thumbnail)
            
            # Dibujar rectángulo del viewport
            if not self.viewport_rect.isEmpty():
                # Ajustar posición del viewport según el centrado de la miniatura
                adjusted_rect = QRect(
                    self.viewport_rect.x() + x,
                    self.viewport_rect.y() + y,
                    self.viewport_rect.width(),
                    self.viewport_rect.height()
                )
                
                # Dibujar rectángulo del viewport
                painter.setPen(QPen(QColor("#007acc"), 2))
                painter.setBrush(QBrush(QColor(0, 122, 204, 50)))
                painter.drawRect(adjusted_rect)
                
                # Dibujar esquinas para indicar que es arrastrable
                corner_size = 6
                painter.setBrush(QBrush(QColor("#007acc")))
                painter.drawRect(adjusted_rect.topLeft().x() - corner_size//2, 
                               adjusted_rect.topLeft().y() - corner_size//2, 
                               corner_size, corner_size)
                painter.drawRect(adjusted_rect.topRight().x() - corner_size//2, 
                               adjusted_rect.topRight().y() - corner_size//2, 
                               corner_size, corner_size)
                painter.drawRect(adjusted_rect.bottomLeft().x() - corner_size//2, 
                               adjusted_rect.bottomLeft().y() - corner_size//2, 
                               corner_size, corner_size)
                painter.drawRect(adjusted_rect.bottomRight().x() - corner_size//2, 
                               adjusted_rect.bottomRight().y() - corner_size//2, 
                               corner_size, corner_size)
        else:
            # Mostrar texto cuando no hay imagen
            painter.setPen(QColor("#999"))
            painter.drawText(self.rect(), Qt.AlignCenter, "Sin imagen")
            
    def mousePressEvent(self, event):
        """Maneja el evento de presionar mouse"""
        if event.button() == Qt.LeftButton:
            self.mousePressed.emit(event.pos())
        super().mousePressEvent(event)
        
    def mouseMoveEvent(self, event):
        """Maneja el movimiento del mouse"""
        if event.buttons() & Qt.LeftButton:
            self.mouseDragged.emit(event.pos())
        super().mouseMoveEvent(event)
        
    def mouseReleaseEvent(self, event):
        """Maneja el evento de soltar mouse"""
        if event.button() == Qt.LeftButton:
            self.mouseReleased.emit(event.pos())
        super().mouseReleaseEvent(event)