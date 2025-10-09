"""
Visor Sincronizado para Comparación de Muestras Balísticas
Implementa visualización lado a lado con zoom y panorámica vinculados,
cursores espejo y funcionalidades avanzadas de comparación.
"""

from PyQt5.QtWidgets import (QWidget, QHBoxLayout, QVBoxLayout, QLabel, 
                             QScrollArea, QSlider, QPushButton, QFrame,
                             QButtonGroup, QCheckBox, QSpinBox, QGroupBox)
from PyQt5.QtCore import Qt, pyqtSignal, QPoint, QRect, QTimer
from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor, QCursor, QFont
import numpy as np
from PIL import Image, ImageChops
import cv2


class SynchronizedImageLabel(QLabel):
    """Label de imagen con capacidades de sincronización y cursor espejo"""
    
    mousePositionChanged = pyqtSignal(QPoint)
    zoomChanged = pyqtSignal(float)
    panChanged = pyqtSignal(QPoint)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 400)
        self.setStyleSheet("border: 1px solid #ccc; background-color: #f0f0f0;")
        self.setAlignment(Qt.AlignCenter)
        
        # Estado de la imagen
        self.original_pixmap = None
        self.current_pixmap = None
        self.zoom_factor = 1.0
        self.pan_offset = QPoint(0, 0)
        self.is_panning = False
        self.last_pan_point = QPoint()
        
        # Cursor espejo
        self.mirror_position = None
        self.show_mirror_cursor = True
        self.mirror_color = QColor(255, 0, 0, 180)
        
        # Configuración de eventos
        self.setMouseTracking(True)
        self.setAttribute(Qt.WA_OpaquePaintEvent)
        
    def set_image(self, pixmap):
        """Establece la imagen a mostrar"""
        self.original_pixmap = pixmap
        self.current_pixmap = pixmap
        self.zoom_factor = 1.0
        self.pan_offset = QPoint(0, 0)
        self.update_display()
        
    def set_zoom(self, zoom_factor):
        """Establece el factor de zoom"""
        if self.original_pixmap and zoom_factor > 0:
            self.zoom_factor = zoom_factor
            self.update_display()
            self.zoomChanged.emit(zoom_factor)
            
    def set_pan(self, offset):
        """Establece el desplazamiento de panorámica"""
        self.pan_offset = offset
        self.update_display()
        self.panChanged.emit(offset)
        
    def set_mirror_position(self, position):
        """Establece la posición del cursor espejo"""
        self.mirror_position = position
        self.update()
        
    def update_display(self):
        """Actualiza la visualización con zoom y pan"""
        if not self.original_pixmap:
            return
            
        # Aplicar zoom
        scaled_size = self.original_pixmap.size() * self.zoom_factor
        self.current_pixmap = self.original_pixmap.scaled(
            scaled_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        
        self.setPixmap(self.current_pixmap)
        
    def mousePressEvent(self, event):
        """Maneja el inicio del arrastre para panorámica"""
        if event.button() == Qt.LeftButton:
            self.is_panning = True
            self.last_pan_point = event.pos()
            self.setCursor(QCursor(Qt.ClosedHandCursor))
            
    def mouseMoveEvent(self, event):
        """Maneja el movimiento del mouse para panorámica y cursor espejo"""
        # Emitir posición para cursor espejo
        self.mousePositionChanged.emit(event.pos())
        
        # Manejar panorámica
        if self.is_panning and event.buttons() & Qt.LeftButton:
            delta = event.pos() - self.last_pan_point
            self.pan_offset += delta
            self.last_pan_point = event.pos()
            self.panChanged.emit(self.pan_offset)
            
    def mouseReleaseEvent(self, event):
        """Maneja el final del arrastre"""
        if event.button() == Qt.LeftButton:
            self.is_panning = False
            self.setCursor(QCursor(Qt.ArrowCursor))
            
    def wheelEvent(self, event):
        """Maneja el zoom con la rueda del mouse"""
        if self.original_pixmap:
            # Calcular nuevo factor de zoom
            zoom_delta = 0.1 if event.angleDelta().y() > 0 else -0.1
            new_zoom = max(0.1, min(5.0, self.zoom_factor + zoom_delta))
            
            if new_zoom != self.zoom_factor:
                self.set_zoom(new_zoom)
                
    def paintEvent(self, event):
        """Dibuja la imagen y el cursor espejo"""
        super().paintEvent(event)
        
        # Dibujar cursor espejo si está habilitado
        if self.show_mirror_cursor and self.mirror_position:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)
            
            # Configurar el pincel para el cursor
            pen = QPen(self.mirror_color, 2)
            painter.setPen(pen)
            
            # Dibujar cruz del cursor espejo
            x, y = self.mirror_position.x(), self.mirror_position.y()
            size = 10
            
            painter.drawLine(x - size, y, x + size, y)  # Línea horizontal
            painter.drawLine(x, y - size, x, y + size)  # Línea vertical
            
            # Dibujar círculo alrededor
            painter.drawEllipse(x - size//2, y - size//2, size, size)


class ImageOverlayWidget(QWidget):
    """Widget para superponer imágenes con control de transparencia"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self.image1 = None
        self.image2 = None
        self.overlay_opacity = 0.5
        
    def setup_ui(self):
        """Configura la interfaz del widget de superposición"""
        layout = QVBoxLayout(self)
        
        # Control de transparencia
        transparency_group = QGroupBox("Control de Transparencia")
        transparency_layout = QVBoxLayout(transparency_group)
        
        self.transparency_slider = QSlider(Qt.Horizontal)
        self.transparency_slider.setRange(0, 100)
        self.transparency_slider.setValue(50)
        self.transparency_slider.valueChanged.connect(self.on_transparency_changed)
        
        transparency_layout.addWidget(QLabel("Transparencia de Superposición:"))
        transparency_layout.addWidget(self.transparency_slider)
        
        # Área de visualización
        self.overlay_label = QLabel()
        self.overlay_label.setMinimumSize(400, 400)
        self.overlay_label.setStyleSheet("border: 1px solid #ccc; background-color: #f0f0f0;")
        self.overlay_label.setAlignment(Qt.AlignCenter)
        
        layout.addWidget(transparency_group)
        layout.addWidget(self.overlay_label)
        
    def set_images(self, image1, image2):
        """Establece las imágenes para superponer"""
        self.image1 = image1
        self.image2 = image2
        self.update_overlay()
        
    def on_transparency_changed(self, value):
        """Maneja el cambio de transparencia"""
        self.overlay_opacity = value / 100.0
        self.update_overlay()
        
    def update_overlay(self):
        """Actualiza la superposición de imágenes"""
        if not self.image1 or not self.image2:
            return
            
        try:
            # Convertir QPixmap a numpy arrays
            img1_array = self.pixmap_to_array(self.image1)
            img2_array = self.pixmap_to_array(self.image2)
            
            # Redimensionar para que coincidan
            if img1_array.shape != img2_array.shape:
                h, w = min(img1_array.shape[0], img2_array.shape[0]), min(img1_array.shape[1], img2_array.shape[1])
                img1_array = cv2.resize(img1_array, (w, h))
                img2_array = cv2.resize(img2_array, (w, h))
            
            # Crear superposición
            overlay = cv2.addWeighted(img1_array, 1 - self.overlay_opacity, 
                                    img2_array, self.overlay_opacity, 0)
            
            # Convertir de vuelta a QPixmap
            overlay_pixmap = self.array_to_pixmap(overlay)
            self.overlay_label.setPixmap(overlay_pixmap.scaled(
                self.overlay_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            ))
            
        except Exception as e:
            print(f"Error en superposición: {e}")
            
    def pixmap_to_array(self, pixmap):
        """Convierte QPixmap a numpy array"""
        image = pixmap.toImage()
        width, height = image.width(), image.height()
        
        ptr = image.bits()
        ptr.setsize(image.byteCount())
        arr = np.array(ptr).reshape(height, width, 4)  # RGBA
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
        
    def array_to_pixmap(self, array):
        """Convierte numpy array a QPixmap"""
        height, width, channel = array.shape
        bytes_per_line = 3 * width
        q_image = QImage(array.data, width, height, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(q_image)


class BlendingModeWidget(QWidget):
    """Widget para modos de fusión y diferencia"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self.image1 = None
        self.image2 = None
        self.current_mode = "normal"
        self.blink_timer = QTimer()
        self.blink_timer.timeout.connect(self.toggle_blink)
        self.blink_state = False
        
    def setup_ui(self):
        """Configura la interfaz del widget de fusión"""
        layout = QVBoxLayout(self)
        
        # Controles de modo
        mode_group = QGroupBox("Modos de Visualización")
        mode_layout = QVBoxLayout(mode_group)
        
        self.mode_buttons = QButtonGroup()
        
        # Botones de modo
        modes = [
            ("Normal", "normal"),
            ("Diferencia", "difference"),
            ("Comparador Parpadeo", "blink")
        ]
        
        for text, mode in modes:
            btn = QPushButton(text)
            btn.setCheckable(True)
            btn.clicked.connect(lambda checked, m=mode: self.set_mode(m))
            self.mode_buttons.addButton(btn)
            mode_layout.addWidget(btn)
            
        # Seleccionar modo normal por defecto
        self.mode_buttons.buttons()[0].setChecked(True)
        
        # Control de velocidad de parpadeo
        blink_group = QGroupBox("Control de Parpadeo")
        blink_layout = QVBoxLayout(blink_group)
        
        self.blink_speed = QSpinBox()
        self.blink_speed.setRange(100, 2000)
        self.blink_speed.setValue(500)
        self.blink_speed.setSuffix(" ms")
        self.blink_speed.valueChanged.connect(self.update_blink_speed)
        
        blink_layout.addWidget(QLabel("Velocidad de Parpadeo:"))
        blink_layout.addWidget(self.blink_speed)
        
        # Área de visualización
        self.blend_label = QLabel()
        self.blend_label.setMinimumSize(400, 400)
        self.blend_label.setStyleSheet("border: 1px solid #ccc; background-color: #f0f0f0;")
        self.blend_label.setAlignment(Qt.AlignCenter)
        
        layout.addWidget(mode_group)
        layout.addWidget(blink_group)
        layout.addWidget(self.blend_label)
        
    def set_images(self, image1, image2):
        """Establece las imágenes para fusión"""
        self.image1 = image1
        self.image2 = image2
        self.update_blend()
        
    def set_mode(self, mode):
        """Establece el modo de fusión"""
        self.current_mode = mode
        
        # Detener parpadeo si no es modo blink
        if mode != "blink":
            self.blink_timer.stop()
        else:
            self.blink_timer.start(self.blink_speed.value())
            
        self.update_blend()
        
    def update_blink_speed(self, speed):
        """Actualiza la velocidad de parpadeo"""
        if self.current_mode == "blink" and self.blink_timer.isActive():
            self.blink_timer.setInterval(speed)
            
    def toggle_blink(self):
        """Alterna entre las dos imágenes en modo parpadeo"""
        self.blink_state = not self.blink_state
        self.update_blend()
        
    def update_blend(self):
        """Actualiza la visualización según el modo seleccionado"""
        if not self.image1 or not self.image2:
            return
            
        try:
            if self.current_mode == "normal":
                self.blend_label.setPixmap(self.image1.scaled(
                    self.blend_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
                ))
                
            elif self.current_mode == "difference":
                self.show_difference()
                
            elif self.current_mode == "blink":
                current_image = self.image1 if self.blink_state else self.image2
                self.blend_label.setPixmap(current_image.scaled(
                    self.blend_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
                ))
                
        except Exception as e:
            print(f"Error en modo de fusión: {e}")
            
    def show_difference(self):
        """Muestra la diferencia entre las dos imágenes"""
        try:
            # Convertir a arrays
            img1_array = self.pixmap_to_array(self.image1)
            img2_array = self.pixmap_to_array(self.image2)
            
            # Redimensionar para que coincidan
            if img1_array.shape != img2_array.shape:
                h, w = min(img1_array.shape[0], img2_array.shape[0]), min(img1_array.shape[1], img2_array.shape[1])
                img1_array = cv2.resize(img1_array, (w, h))
                img2_array = cv2.resize(img2_array, (w, h))
            
            # Calcular diferencia
            diff = cv2.absdiff(img1_array, img2_array)
            
            # Aplicar mapa de color para resaltar diferencias
            diff_gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
            diff_colored = cv2.applyColorMap(diff_gray, cv2.COLORMAP_JET)
            diff_colored = cv2.cvtColor(diff_colored, cv2.COLOR_BGR2RGB)
            
            # Convertir de vuelta a QPixmap
            diff_pixmap = self.array_to_pixmap(diff_colored)
            self.blend_label.setPixmap(diff_pixmap.scaled(
                self.blend_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            ))
            
        except Exception as e:
            print(f"Error calculando diferencia: {e}")
            
    def pixmap_to_array(self, pixmap):
        """Convierte QPixmap a numpy array"""
        image = pixmap.toImage()
        width, height = image.width(), image.height()
        
        ptr = image.bits()
        ptr.setsize(image.byteCount())
        arr = np.array(ptr).reshape(height, width, 4)  # RGBA
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
        
    def array_to_pixmap(self, array):
        """Convierte numpy array a QPixmap"""
        height, width, channel = array.shape
        bytes_per_line = 3 * width
        q_image = QImage(array.data, width, height, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(q_image)


class SynchronizedViewer(QWidget):
    """Visor principal sincronizado para comparación de muestras"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self.connect_signals()
        
    def setup_ui(self):
        """Configura la interfaz principal"""
        main_layout = QVBoxLayout(self)
        
        # Layout superior para las imágenes
        images_layout = QHBoxLayout()
        
        # Panel izquierdo - Primera imagen
        left_widget = QWidget()
        left_panel = QVBoxLayout(left_widget)
        left_panel.addWidget(QLabel("Muestra 1"))
        
        self.image1_scroll = QScrollArea()
        self.image1_label = SynchronizedImageLabel()
        self.image1_scroll.setWidget(self.image1_label)
        self.image1_scroll.setWidgetResizable(True)
        
        left_panel.addWidget(self.image1_scroll)
        
        # Panel derecho - Segunda imagen
        right_widget = QWidget()
        right_panel = QVBoxLayout(right_widget)
        right_panel.addWidget(QLabel("Muestra 2"))
        
        self.image2_scroll = QScrollArea()
        self.image2_label = SynchronizedImageLabel()
        self.image2_scroll.setWidget(self.image2_label)
        self.image2_scroll.setWidgetResizable(True)
        
        right_panel.addWidget(self.image2_scroll)
        
        # Agregar paneles de imágenes al layout superior
        images_layout.addWidget(left_widget)
        images_layout.addWidget(right_widget)
        
        # Panel inferior compacto con controles
        controls_widget = QWidget()
        controls_widget.setMaximumHeight(120)  # Altura máxima compacta
        controls_layout = QHBoxLayout(controls_widget)
        
        # Controles de zoom (compactos)
        zoom_group = QGroupBox("Control de Zoom")
        zoom_layout = QHBoxLayout(zoom_group)  # Cambio a horizontal para compactar
        
        self.zoom_label = QLabel("100%")
        self.zoom_label.setAlignment(Qt.AlignCenter)
        self.zoom_label.setMinimumWidth(50)
        
        self.zoom_slider = QSlider(Qt.Horizontal)  # Cambio a horizontal
        self.zoom_slider.setRange(10, 500)  # 0.1x a 5.0x
        self.zoom_slider.setValue(100)  # 1.0x
        self.zoom_slider.valueChanged.connect(self.on_zoom_changed)
        
        zoom_buttons_layout = QVBoxLayout()  # Botones en vertical para ahorrar espacio
        self.zoom_in_btn = QPushButton("+")
        self.zoom_out_btn = QPushButton("-")
        self.zoom_reset_btn = QPushButton("Reset")
        
        # Hacer botones más pequeños
        for btn in [self.zoom_in_btn, self.zoom_out_btn, self.zoom_reset_btn]:
            btn.setMaximumSize(40, 25)
        
        self.zoom_in_btn.clicked.connect(lambda: self.adjust_zoom(10))
        self.zoom_out_btn.clicked.connect(lambda: self.adjust_zoom(-10))
        self.zoom_reset_btn.clicked.connect(self.reset_zoom)
        
        zoom_buttons_layout.addWidget(self.zoom_in_btn)
        zoom_buttons_layout.addWidget(self.zoom_out_btn)
        zoom_buttons_layout.addWidget(self.zoom_reset_btn)
        
        zoom_layout.addWidget(self.zoom_label)
        zoom_layout.addWidget(self.zoom_slider)
        zoom_layout.addLayout(zoom_buttons_layout)
        
        # Controles de sincronización (compactos)
        sync_group = QGroupBox("Sincronización")
        sync_layout = QVBoxLayout(sync_group)
        
        self.sync_zoom_cb = QCheckBox("Sincronizar Zoom")
        self.sync_pan_cb = QCheckBox("Sincronizar Panorámica")
        self.mirror_cursor_cb = QCheckBox("Cursor Espejo")
        
        self.sync_zoom_cb.setChecked(True)
        self.sync_pan_cb.setChecked(True)
        self.mirror_cursor_cb.setChecked(True)
        
        # Hacer checkboxes más compactos
        for cb in [self.sync_zoom_cb, self.sync_pan_cb, self.mirror_cursor_cb]:
            cb.setStyleSheet("QCheckBox { font-size: 11px; }")
        
        sync_layout.addWidget(self.sync_zoom_cb)
        sync_layout.addWidget(self.sync_pan_cb)
        sync_layout.addWidget(self.mirror_cursor_cb)
        
        # Agregar grupos de controles al layout inferior
        controls_layout.addWidget(zoom_group)
        controls_layout.addWidget(sync_group)
        controls_layout.addStretch()  # Espacio flexible al final
        
        # Agregar layouts al layout principal
        main_layout.addLayout(images_layout, 1)  # Las imágenes ocupan la mayor parte del espacio
        main_layout.addWidget(controls_widget, 0)  # Los controles ocupan espacio mínimo
        
    def connect_signals(self):
        """Conecta las señales entre los componentes"""
        # Sincronización de posición del mouse
        self.image1_label.mousePositionChanged.connect(self.on_mouse1_moved)
        self.image2_label.mousePositionChanged.connect(self.on_mouse2_moved)
        
        # Sincronización de zoom
        self.image1_label.zoomChanged.connect(self.on_image1_zoom_changed)
        self.image2_label.zoomChanged.connect(self.on_image2_zoom_changed)
        
        # Sincronización de panorámica
        self.image1_label.panChanged.connect(self.on_image1_pan_changed)
        self.image2_label.panChanged.connect(self.on_image2_pan_changed)
        
    def set_images(self, image1, image2):
        """Establece las imágenes a comparar"""
        self.image1_label.set_image(image1)
        self.image2_label.set_image(image2)
        
    def on_mouse1_moved(self, position):
        """Maneja el movimiento del mouse en la imagen 1"""
        if self.mirror_cursor_cb.isChecked():
            self.image2_label.set_mirror_position(position)
            
    def on_mouse2_moved(self, position):
        """Maneja el movimiento del mouse en la imagen 2"""
        if self.mirror_cursor_cb.isChecked():
            self.image1_label.set_mirror_position(position)
            
    def on_zoom_changed(self, value):
        """Maneja el cambio de zoom desde el slider"""
        zoom_factor = value / 100.0
        self.zoom_label.setText(f"{value}%")
        
        if self.sync_zoom_cb.isChecked():
            self.image1_label.set_zoom(zoom_factor)
            self.image2_label.set_zoom(zoom_factor)
            
    def on_image1_zoom_changed(self, zoom_factor):
        """Maneja el cambio de zoom desde la imagen 1"""
        if self.sync_zoom_cb.isChecked():
            self.image2_label.set_zoom(zoom_factor)
            self.zoom_slider.setValue(int(zoom_factor * 100))
            
    def on_image2_zoom_changed(self, zoom_factor):
        """Maneja el cambio de zoom desde la imagen 2"""
        if self.sync_zoom_cb.isChecked():
            self.image1_label.set_zoom(zoom_factor)
            self.zoom_slider.setValue(int(zoom_factor * 100))
            
    def on_image1_pan_changed(self, offset):
        """Maneja el cambio de panorámica desde la imagen 1"""
        if self.sync_pan_cb.isChecked():
            self.image2_label.set_pan(offset)
            
    def on_image2_pan_changed(self, offset):
        """Maneja el cambio de panorámica desde la imagen 2"""
        if self.sync_pan_cb.isChecked():
            self.image1_label.set_pan(offset)
            
    def adjust_zoom(self, delta):
        """Ajusta el zoom por un delta dado"""
        current_value = self.zoom_slider.value()
        new_value = max(10, min(500, current_value + delta))
        self.zoom_slider.setValue(new_value)
        
    def reset_zoom(self):
        """Resetea el zoom a 100%"""
        self.zoom_slider.setValue(100)