#!/usr/bin/env python3
"""
Widgets compartidos para la interfaz SIGeC-Balistica
Componentes reutilizables con diseño moderno
"""

import os
from typing import Optional, Callable, List, Any
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QFrame, QScrollArea, QProgressBar, QFileDialog, QApplication,
    QSizePolicy, QSpacerItem, QGridLayout, QTextEdit, QCheckBox,
    QComboBox, QDoubleSpinBox, QSpinBox, QSlider, QTabWidget
)
from PyQt5.QtCore import Qt, pyqtSignal, QPropertyAnimation, QEasingCurve, QRect, QTimer, QFileInfo
from PyQt5.QtGui import QPixmap, QFont, QIcon, QPainter, QColor, QDragEnterEvent, QDropEvent
from PIL import Image
import numpy as np

class ImageDropZone(QFrame):
    """Drop zone moderno para cargar imágenes con preview"""
    
    imageLoaded = pyqtSignal(str)  # Señal cuando se carga una imagen
    
    def __init__(self, title: str = "Arrastrar imagen aquí", subtitle: str = "o hacer clic para seleccionar"):
        super().__init__()
        self.title = title
        self.subtitle = subtitle
        self.image_path = None
        self.setup_ui()
        self.setup_drag_drop()
        
    def setup_ui(self):
        """Configura la interfaz del drop zone"""
        self.setProperty("class", "drop-zone")
        self.setMinimumHeight(200)
        self.setMaximumHeight(300)
        
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        
        # Icono de imagen
        self.icon_label = QLabel("🖼️")
        self.icon_label.setAlignment(Qt.AlignCenter)
        self.icon_label.setStyleSheet("font-size: 48px; margin: 16px;")
        layout.addWidget(self.icon_label)
        
        # Título
        self.title_label = QLabel(self.title)
        self.title_label.setProperty("class", "subtitle")
        self.title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.title_label)
        
        # Subtítulo
        self.subtitle_label = QLabel(self.subtitle)
        self.subtitle_label.setProperty("class", "caption")
        self.subtitle_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.subtitle_label)
        
        # Botón de selección
        self.select_button = QPushButton("Seleccionar Archivo")
        self.select_button.clicked.connect(self.select_file)
        layout.addWidget(self.select_button)
        
        # Preview de imagen (inicialmente oculto)
        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMaximumHeight(150)
        self.preview_label.setScaledContents(True)
        self.preview_label.hide()
        layout.addWidget(self.preview_label)
        
        # Info de archivo
        self.file_info_label = QLabel()
        self.file_info_label.setProperty("class", "caption")
        self.file_info_label.setAlignment(Qt.AlignCenter)
        self.file_info_label.hide()
        layout.addWidget(self.file_info_label)
        
    def setup_drag_drop(self):
        """Configura drag and drop"""
        self.setAcceptDrops(True)
        
    def dragEnterEvent(self, event: QDragEnterEvent):
        """Maneja el evento de arrastrar sobre el widget"""
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if urls and self.is_image_file(urls[0].toLocalFile()):
                event.acceptProposedAction()
                self.setProperty("class", "drop-zone-active")
                self.style().unpolish(self)
                self.style().polish(self)
                return
        
        self.setProperty("class", "drop-zone-error")
        self.style().unpolish(self)
        self.style().polish(self)
        event.ignore()
        
    def dragLeaveEvent(self, event):
        """Maneja el evento de salir del área de drop"""
        self.setProperty("class", "drop-zone")
        self.style().unpolish(self)
        self.style().polish(self)
        
    def dropEvent(self, event: QDropEvent):
        """Maneja el evento de soltar archivo"""
        self.setProperty("class", "drop-zone")
        self.style().unpolish(self)
        self.style().polish(self)
        
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if urls:
                file_path = urls[0].toLocalFile()
                if self.is_image_file(file_path):
                    self.load_image(file_path)
                    event.acceptProposedAction()
                    
    def mousePressEvent(self, event):
        """Permite hacer clic para seleccionar archivo"""
        if event.button() == Qt.LeftButton:
            self.select_file()
            
    def select_file(self):
        """Abre diálogo para seleccionar archivo"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Seleccionar Imagen",
            "",
            "Imágenes (*.png *.jpg *.jpeg *.bmp *.tiff *.tif);;Todos los archivos (*)"
        )
        
        if file_path:
            self.load_image(file_path)
            
    def load_image(self, file_path: str):
        """Carga y muestra preview de la imagen"""
        try:
            self.image_path = file_path
            
            # Cargar imagen para preview
            pixmap = QPixmap(file_path)
            if not pixmap.isNull():
                # Redimensionar para preview
                scaled_pixmap = pixmap.scaled(200, 150, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.preview_label.setPixmap(scaled_pixmap)
                self.preview_label.show()
                
                # Mostrar info del archivo
                file_info = os.path.basename(file_path)
                file_size = os.path.getsize(file_path)
                size_mb = file_size / (1024 * 1024)
                
                # Obtener dimensiones de la imagen
                with Image.open(file_path) as img:
                    width, height = img.size
                    
                info_text = f"{file_info}\n{width}x{height} px • {size_mb:.1f} MB"
                self.file_info_label.setText(info_text)
                self.file_info_label.show()
                
                # Ocultar elementos de drop zone
                self.icon_label.hide()
                self.title_label.hide()
                self.subtitle_label.hide()
                self.select_button.setText("Cambiar Imagen")
                
                # Emitir señal
                self.imageLoaded.emit(file_path)
                
        except Exception as e:
            print(f"Error cargando imagen: {e}")
            
    def is_image_file(self, file_path: str) -> bool:
        """Verifica si el archivo es una imagen válida"""
        valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
        return os.path.splitext(file_path.lower())[1] in valid_extensions
        
    def clear(self):
        """Limpia el drop zone"""
        self.image_path = None
        self.preview_label.hide()
        self.file_info_label.hide()
        self.icon_label.show()
        self.title_label.show()
        self.subtitle_label.show()
        self.select_button.setText("Seleccionar Archivo")
        
    def get_image_path(self) -> Optional[str]:
        """Retorna la ruta de la imagen cargada"""
        return self.image_path

class ResultCard(QFrame):
    """Tarjeta de resultado con diseño prominente y colores"""
    
    def __init__(self, title: str, value: str = "", result_type: str = "info"):
        super().__init__()
        self.title = title
        self.value = value
        self.result_type = result_type  # "success", "warning", "error", "info"
        self.setup_ui()
        
    def setup_ui(self):
        """Configura la interfaz de la tarjeta"""
        self.setProperty("class", "card-elevated")
        self.setMinimumHeight(120)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        
        # Título
        title_label = QLabel(self.title)
        title_label.setProperty("class", "body")
        layout.addWidget(title_label)
        
        # Valor principal
        self.value_label = QLabel(self.value)
        self.value_label.setProperty("class", "title")
        self.update_value_style()
        layout.addWidget(self.value_label)
        
        # Descripción adicional
        self.description_label = QLabel()
        self.description_label.setProperty("class", "caption")
        self.description_label.hide()
        layout.addWidget(self.description_label)
        
        # Spacer
        layout.addItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))
        
    def update_value_style(self):
        """Actualiza el estilo del valor según el tipo"""
        style_class = {
            "success": "success",
            "warning": "warning", 
            "error": "error",
            "info": "body"
        }.get(self.result_type, "body")
        
        self.value_label.setProperty("class", f"title {style_class}")
        
    def set_value(self, value: str, result_type: str = None):
        """Actualiza el valor y tipo de resultado"""
        self.value = value
        if result_type:
            self.result_type = result_type
            
        self.value_label.setText(value)
        self.update_value_style()
        
        # Forzar actualización de estilo
        self.value_label.style().unpolish(self.value_label)
        self.value_label.style().polish(self.value_label)
        
    def set_description(self, description: str):
        """Establece descripción adicional"""
        if description:
            self.description_label.setText(description)
            self.description_label.show()
        else:
            self.description_label.hide()

class CollapsiblePanel(QFrame):
    """Panel colapsable con animación suave"""
    
    def __init__(self, title: str, content_widget: QWidget = None):
        super().__init__()
        self.title = title
        self.content_widget = content_widget
        self.is_expanded = False
        self.setup_ui()
        
    def setup_ui(self):
        """Configura la interfaz del panel"""
        self.setProperty("class", "collapsible")
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Header clickeable
        self.header = QFrame()
        self.header.setProperty("class", "collapsible-header")
        self.header.setCursor(Qt.PointingHandCursor)
        
        header_layout = QHBoxLayout(self.header)
        
        # Icono de expansión
        self.expand_icon = QLabel("▶")
        self.expand_icon.setFixedWidth(20)
        header_layout.addWidget(self.expand_icon)
        
        # Título
        title_label = QLabel(self.title)
        title_label.setProperty("class", "subtitle")
        header_layout.addWidget(title_label)
        
        # Spacer
        header_layout.addItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))
        
        layout.addWidget(self.header)
        
        # Contenido
        self.content_frame = QFrame()
        self.content_frame.setProperty("class", "collapsible-content")
        self.content_frame.hide()
        
        if self.content_widget:
            content_layout = QVBoxLayout(self.content_frame)
            content_layout.addWidget(self.content_widget)
        else:
            self.content_layout = QVBoxLayout(self.content_frame)
            
        layout.addWidget(self.content_frame)
        
        # Conectar evento de clic
        self.header.mousePressEvent = self.toggle_expansion
        
    def toggle_expansion(self, event=None):
        """Alterna la expansión del panel"""
        self.is_expanded = not self.is_expanded
        
        if self.is_expanded:
            self.expand_icon.setText("▼")
            self.content_frame.show()
        else:
            self.expand_icon.setText("▶")
            self.content_frame.hide()
            
    def add_content_widget(self, widget: QWidget):
        """Añade un widget al contenido"""
        if hasattr(self, 'content_layout'):
            self.content_layout.addWidget(widget)
        
    def set_expanded(self, expanded: bool):
        """Establece el estado de expansión"""
        if expanded != self.is_expanded:
            self.toggle_expansion()

class StepIndicator(QFrame):
    """Indicador de pasos para flujos guiados"""
    
    def __init__(self, steps: List[str]):
        super().__init__()
        self.steps = steps
        self.current_step = 0
        self.setup_ui()
        
    def setup_ui(self):
        """Configura la interfaz del indicador"""
        layout = QHBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        
        self.step_widgets = []
        
        for i, step in enumerate(self.steps):
            # Contenedor del paso
            step_container = QVBoxLayout()
            step_container.setAlignment(Qt.AlignCenter)
            
            # Círculo del paso
            step_circle = QLabel(str(i + 1))
            step_circle.setProperty("class", "step-indicator")
            step_circle.setAlignment(Qt.AlignCenter)
            step_circle.setFixedSize(40, 40)
            
            # Texto del paso
            step_text = QLabel(step)
            step_text.setProperty("class", "caption")
            step_text.setAlignment(Qt.AlignCenter)
            step_text.setWordWrap(True)
            step_text.setMaximumWidth(100)
            
            step_container.addWidget(step_circle)
            step_container.addWidget(step_text)
            
            # Añadir al layout principal
            step_widget = QWidget()
            step_widget.setLayout(step_container)
            layout.addWidget(step_widget)
            
            self.step_widgets.append((step_circle, step_text))
            
            # Línea conectora (excepto para el último paso)
            if i < len(self.steps) - 1:
                line = QFrame()
                line.setProperty("class", "separator")
                line.setFixedHeight(2)
                line.setMinimumWidth(50)
                layout.addWidget(line)
                
        self.update_step_styles()
        
    def set_current_step(self, step: int):
        """Establece el paso actual"""
        if 0 <= step < len(self.steps):
            self.current_step = step
            self.update_step_styles()
            
    def update_step_styles(self):
        """Actualiza los estilos de los pasos"""
        for i, (circle, text) in enumerate(self.step_widgets):
            if i < self.current_step:
                # Paso completado
                circle.setProperty("class", "step-indicator-completed")
                circle.setText("✓")
            elif i == self.current_step:
                # Paso actual
                circle.setProperty("class", "step-indicator-active")
                circle.setText(str(i + 1))
            else:
                # Paso pendiente
                circle.setProperty("class", "step-indicator")
                circle.setText(str(i + 1))
                
            # Forzar actualización de estilo
            circle.style().unpolish(circle)
            circle.style().polish(circle)

class ProgressCard(QFrame):
    """Tarjeta de progreso con información detallada"""
    
    def __init__(self, title: str = "Procesando..."):
        super().__init__()
        self.title = title
        self.setup_ui()
        
    def setup_ui(self):
        """Configura la interfaz de la tarjeta"""
        self.setProperty("class", "card")
        
        layout = QVBoxLayout(self)
        
        # Título
        self.title_label = QLabel(self.title)
        self.title_label.setProperty("class", "subtitle")
        layout.addWidget(self.title_label)
        
        # Barra de progreso
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)
        
        # Información de estado
        self.status_label = QLabel("Iniciando...")
        self.status_label.setProperty("class", "caption")
        layout.addWidget(self.status_label)
        
        # Información de tiempo
        self.time_label = QLabel()
        self.time_label.setProperty("class", "caption")
        self.time_label.hide()
        layout.addWidget(self.time_label)
        
    def set_progress(self, value: int, status: str = ""):
        """Actualiza el progreso"""
        self.progress_bar.setValue(value)
        if status:
            self.status_label.setText(status)
            
    def set_time_info(self, elapsed: str = "", remaining: str = ""):
        """Establece información de tiempo"""
        if elapsed or remaining:
            time_text = []
            if elapsed:
                time_text.append(f"Transcurrido: {elapsed}")
            if remaining:
                time_text.append(f"Restante: {remaining}")
            
            self.time_label.setText(" • ".join(time_text))
            self.time_label.show()
        else:
            self.time_label.hide()
            
    def set_completed(self, success: bool = True, message: str = ""):
        """Marca como completado"""
        if success:
            self.progress_bar.setValue(100)
            self.status_label.setText(message or "Completado exitosamente")
            self.status_label.setProperty("class", "caption success")
        else:
            self.status_label.setText(message or "Error en el procesamiento")
            self.status_label.setProperty("class", "caption error")
            
        # Forzar actualización de estilo
        self.status_label.style().unpolish(self.status_label)
        self.status_label.style().polish(self.status_label)

class InteractiveImageViewer(QFrame):
    """Visor de imágenes interactivo con zoom, panorámica y overlays"""
    
    imageClicked = pyqtSignal(int, int)  # Señal para clicks en la imagen (x, y)
    
    def __init__(self):
        super().__init__()
        self.image_path = None
        self.zoom_factor = 1.0
        self.min_zoom = 0.1
        self.max_zoom = 10.0
        self.pan_start_point = None
        self.panning = False
        self.overlays = {}  # Diccionario para almacenar overlays
        self.overlay_opacity = 0.7
        self.setup_ui()
        
    def setup_ui(self):
        """Configura la interfaz del visor"""
        self.setProperty("class", "card")
        
        layout = QVBoxLayout(self)
        
        # Controles superiores
        controls_layout = QHBoxLayout()
        
        # Controles de zoom
        self.zoom_out_btn = QPushButton("🔍-")
        self.zoom_out_btn.setMaximumWidth(40)
        self.zoom_out_btn.setToolTip("Alejar (Ctrl + -)")
        self.zoom_out_btn.clicked.connect(self.zoom_out)
        controls_layout.addWidget(self.zoom_out_btn)
        
        self.zoom_label = QLabel("100%")
        self.zoom_label.setAlignment(Qt.AlignCenter)
        self.zoom_label.setMinimumWidth(60)
        controls_layout.addWidget(self.zoom_label)
        
        self.zoom_in_btn = QPushButton("🔍+")
        self.zoom_in_btn.setMaximumWidth(40)
        self.zoom_in_btn.setToolTip("Acercar (Ctrl + +)")
        self.zoom_in_btn.clicked.connect(self.zoom_in)
        controls_layout.addWidget(self.zoom_in_btn)
        
        self.fit_btn = QPushButton("Ajustar")
        self.fit_btn.setToolTip("Ajustar a ventana (Ctrl + 0)")
        self.fit_btn.clicked.connect(self.fit_to_window)
        controls_layout.addWidget(self.fit_btn)
        
        self.reset_btn = QPushButton("1:1")
        self.reset_btn.setToolTip("Tamaño real (Ctrl + 1)")
        self.reset_btn.clicked.connect(self.reset_zoom)
        controls_layout.addWidget(self.reset_btn)
        
        controls_layout.addItem(QSpacerItem(20, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))
        
        # Controles de overlay
        self.overlay_opacity_label = QLabel("Overlay:")
        controls_layout.addWidget(self.overlay_opacity_label)
        
        from PyQt5.QtWidgets import QSlider
        self.overlay_opacity_slider = QSlider(Qt.Horizontal)
        self.overlay_opacity_slider.setRange(0, 100)
        self.overlay_opacity_slider.setValue(70)
        self.overlay_opacity_slider.setMaximumWidth(100)
        self.overlay_opacity_slider.setToolTip("Opacidad de overlays")
        self.overlay_opacity_slider.valueChanged.connect(self.update_overlay_opacity)
        controls_layout.addWidget(self.overlay_opacity_slider)
        
        layout.addLayout(controls_layout)
        
        # Área de imagen con scroll personalizada
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(False)  # Importante para el zoom
        self.scroll_area.setAlignment(Qt.AlignCenter)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        # Label personalizado para la imagen
        self.image_label = InteractiveImageLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("color: #757575; font-size: 14px; background-color: #f5f5f5;")
        self.image_label.setMinimumSize(300, 200)
        self.image_label.setText("No hay imagen cargada")
        
        # Conectar señales del label
        self.image_label.mousePressed.connect(self.start_pan)
        self.image_label.mouseMoved.connect(self.pan_image)
        self.image_label.mouseReleased.connect(self.end_pan)
        self.image_label.wheelScrolled.connect(self.wheel_zoom)
        self.image_label.imageClicked.connect(self.imageClicked)
        
        self.scroll_area.setWidget(self.image_label)
        layout.addWidget(self.scroll_area)
        
        # Información de coordenadas
        self.coords_label = QLabel("Posición: -")
        self.coords_label.setStyleSheet("color: #666; font-size: 11px;")
        layout.addWidget(self.coords_label)
        
    def load_image(self, image_path: str):
        """Carga una imagen en el visor"""
        try:
            self.image_path = image_path
            self.original_pixmap = QPixmap(image_path)
            
            if not self.original_pixmap.isNull():
                self.image_label.set_original_pixmap(self.original_pixmap)
                self.fit_to_window()
                self.image_label.setText("")
            else:
                self.image_label.setText("Error: No se pudo cargar la imagen")
                
        except Exception as e:
            self.image_label.setText(f"Error: {str(e)}")
            
    def update_image_display(self):
        """Actualiza la visualización de la imagen"""
        if hasattr(self, 'original_pixmap') and not self.original_pixmap.isNull():
            # Calcular nuevo tamaño
            new_size = self.original_pixmap.size() * self.zoom_factor
            
            # Escalar imagen
            scaled_pixmap = self.original_pixmap.scaled(
                new_size,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            
            # Aplicar overlays si existen
            if self.overlays:
                scaled_pixmap = self.apply_overlays(scaled_pixmap)
            
            self.image_label.setPixmap(scaled_pixmap)
            self.image_label.resize(scaled_pixmap.size())
            
            # Actualizar etiqueta de zoom
            self.zoom_label.setText(f"{int(self.zoom_factor * 100)}%")
            
    def apply_overlays(self, pixmap):
        """Aplica overlays a la imagen"""
        if not self.overlays:
            return pixmap
            
        # Crear una copia del pixmap para dibujar encima
        result_pixmap = QPixmap(pixmap)
        painter = QPainter(result_pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Aplicar cada overlay
        for overlay_name, overlay_data in self.overlays.items():
            if overlay_data.get('visible', True):
                self.draw_overlay(painter, overlay_data, result_pixmap.size())
        
        painter.end()
        return result_pixmap
        
    def draw_overlay(self, painter, overlay_data, image_size):
        """Dibuja un overlay específico"""
        overlay_type = overlay_data.get('type', 'roi')
        color = QColor(overlay_data.get('color', '#ff0000'))
        color.setAlphaF(self.overlay_opacity)
        
        painter.setPen(color)
        painter.setBrush(QColor(color.red(), color.green(), color.blue(), int(50 * self.overlay_opacity)))
        
        if overlay_type == 'roi':
            # Dibujar rectángulo ROI
            rect = overlay_data.get('rect', QRect(10, 10, 100, 100))
            # Escalar rectángulo según zoom
            scaled_rect = QRect(
                int(rect.x() * self.zoom_factor),
                int(rect.y() * self.zoom_factor),
                int(rect.width() * self.zoom_factor),
                int(rect.height() * self.zoom_factor)
            )
            painter.drawRect(scaled_rect)
            
        elif overlay_type == 'point':
            # Dibujar punto
            point = overlay_data.get('point', (50, 50))
            scaled_point = (int(point[0] * self.zoom_factor), int(point[1] * self.zoom_factor))
            painter.drawEllipse(scaled_point[0] - 5, scaled_point[1] - 5, 10, 10)
            
    def add_overlay(self, name, overlay_type, **kwargs):
        """Agrega un overlay"""
        self.overlays[name] = {
            'type': overlay_type,
            'visible': True,
            **kwargs
        }
        self.update_image_display()
        
    def remove_overlay(self, name):
        """Remueve un overlay"""
        if name in self.overlays:
            del self.overlays[name]
            self.update_image_display()
            
    def set_overlay_visible(self, name, visible):
        """Cambia la visibilidad de un overlay"""
        if name in self.overlays:
            self.overlays[name]['visible'] = visible
            self.update_image_display()
            
    def update_overlay_opacity(self, value):
        """Actualiza la opacidad de los overlays"""
        self.overlay_opacity = value / 100.0
        self.update_image_display()
        
    def start_pan(self, pos):
        """Inicia el paneo"""
        self.pan_start_point = pos
        self.panning = True
        self.image_label.setCursor(Qt.ClosedHandCursor)
        
    def pan_image(self, pos):
        """Realiza el paneo de la imagen"""
        if self.panning and self.pan_start_point:
            # Calcular desplazamiento
            delta = pos - self.pan_start_point
            
            # Obtener scrollbars
            h_scroll = self.scroll_area.horizontalScrollBar()
            v_scroll = self.scroll_area.verticalScrollBar()
            
            # Aplicar desplazamiento
            h_scroll.setValue(h_scroll.value() - delta.x())
            v_scroll.setValue(v_scroll.value() - delta.y())
            
            self.pan_start_point = pos
            
    def end_pan(self, pos):
        """Termina el paneo"""
        self.panning = False
        self.pan_start_point = None
        self.image_label.setCursor(Qt.ArrowCursor)
        
    def wheel_zoom(self, delta, pos):
        """Zoom con rueda del mouse"""
        if hasattr(self, 'original_pixmap') and not self.original_pixmap.isNull():
            # Calcular factor de zoom
            zoom_in = delta > 0
            zoom_factor = 1.25 if zoom_in else 1/1.25
            
            old_zoom = self.zoom_factor
            self.zoom_factor = max(self.min_zoom, min(self.max_zoom, self.zoom_factor * zoom_factor))
            
            if old_zoom != self.zoom_factor:
                # Zoom centrado en la posición del mouse
                self.zoom_at_point(pos, old_zoom)
                
    def zoom_at_point(self, point, old_zoom):
        """Realiza zoom centrado en un punto específico"""
        # Obtener scrollbars
        h_scroll = self.scroll_area.horizontalScrollBar()
        v_scroll = self.scroll_area.verticalScrollBar()
        
        # Calcular posición relativa antes del zoom
        old_pos_x = (h_scroll.value() + point.x()) / old_zoom
        old_pos_y = (v_scroll.value() + point.y()) / old_zoom
        
        # Actualizar imagen
        self.update_image_display()
        
        # Calcular nueva posición del scroll
        new_scroll_x = old_pos_x * self.zoom_factor - point.x()
        new_scroll_y = old_pos_y * self.zoom_factor - point.y()
        
        # Aplicar nueva posición
        h_scroll.setValue(int(new_scroll_x))
        v_scroll.setValue(int(new_scroll_y))
        
    def zoom_in(self):
        """Aumenta el zoom"""
        self.zoom_factor = min(self.zoom_factor * 1.25, self.max_zoom)
        self.update_image_display()
        
    def zoom_out(self):
        """Disminuye el zoom"""
        self.zoom_factor = max(self.zoom_factor / 1.25, self.min_zoom)
        self.update_image_display()
        
    def reset_zoom(self):
        """Resetea el zoom a 100%"""
        self.zoom_factor = 1.0
        self.update_image_display()
        
    def fit_to_window(self):
        """Ajusta la imagen al tamaño de la ventana"""
        if hasattr(self, 'original_pixmap') and not self.original_pixmap.isNull():
            scroll_size = self.scroll_area.size()
            image_size = self.original_pixmap.size()
            
            # Calcular factor de escala para ajustar
            scale_x = (scroll_size.width() - 20) / image_size.width()
            scale_y = (scroll_size.height() - 20) / image_size.height()
            
            self.zoom_factor = min(scale_x, scale_y, 1.0)
            self.update_image_display()
            
    def clear(self):
        """Limpia el visor"""
        self.image_path = None
        self.zoom_factor = 1.0
        self.overlays.clear()
        self.image_label.clear()
        self.image_label.setText("No hay imagen cargada")
        self.zoom_label.setText("100%")
        self.coords_label.setText("Posición: -")


class InteractiveImageLabel(QLabel):
    """Label personalizado para manejar eventos de mouse en la imagen"""
    
    mousePressed = pyqtSignal(object)  # QPoint
    mouseMoved = pyqtSignal(object)    # QPoint
    mouseReleased = pyqtSignal(object) # QPoint
    wheelScrolled = pyqtSignal(int, object)  # delta, QPoint
    imageClicked = pyqtSignal(int, int)  # x, y en coordenadas de imagen
    
    def __init__(self):
        super().__init__()
        self.setMouseTracking(True)
        self.original_pixmap = None
        
    def set_original_pixmap(self, pixmap):
        """Establece el pixmap original para cálculos de coordenadas"""
        self.original_pixmap = pixmap
        
    def mousePressEvent(self, event):
        """Maneja el evento de presionar mouse"""
        if event.button() == Qt.LeftButton:
            self.mousePressed.emit(event.pos())
        super().mousePressEvent(event)
        
    def mouseMoveEvent(self, event):
        """Maneja el movimiento del mouse"""
        self.mouseMoved.emit(event.pos())
        
        # Emitir coordenadas de imagen si hay pixmap
        if self.original_pixmap and not self.original_pixmap.isNull():
            # Calcular coordenadas en la imagen original
            label_size = self.size()
            pixmap_size = self.pixmap().size() if self.pixmap() else label_size
            
            if pixmap_size.width() > 0 and pixmap_size.height() > 0:
                # Calcular offset del pixmap en el label
                offset_x = (label_size.width() - pixmap_size.width()) // 2
                offset_y = (label_size.height() - pixmap_size.height()) // 2
                
                # Coordenadas relativas al pixmap
                rel_x = event.pos().x() - offset_x
                rel_y = event.pos().y() - offset_y
                
                if 0 <= rel_x < pixmap_size.width() and 0 <= rel_y < pixmap_size.height():
                    # Escalar a coordenadas de imagen original
                    scale_x = self.original_pixmap.width() / pixmap_size.width()
                    scale_y = self.original_pixmap.height() / pixmap_size.height()
                    
                    orig_x = int(rel_x * scale_x)
                    orig_y = int(rel_y * scale_y)
                    
                    # Emitir coordenadas actualizadas
                    parent = self.parent()
                    while parent and not hasattr(parent, 'coords_label'):
                        parent = parent.parent()
                    if parent and hasattr(parent, 'coords_label'):
                        parent.coords_label.setText(f"Posición: ({orig_x}, {orig_y})")
        
        super().mouseMoveEvent(event)
        
    def mouseReleaseEvent(self, event):
        """Maneja el evento de soltar mouse"""
        if event.button() == Qt.LeftButton:
            self.mouseReleased.emit(event.pos())
            
            # Emitir click en coordenadas de imagen
            if self.original_pixmap and not self.original_pixmap.isNull():
                label_size = self.size()
                pixmap_size = self.pixmap().size() if self.pixmap() else label_size
                
                if pixmap_size.width() > 0 and pixmap_size.height() > 0:
                    offset_x = (label_size.width() - pixmap_size.width()) // 2
                    offset_y = (label_size.height() - pixmap_size.height()) // 2
                    
                    rel_x = event.pos().x() - offset_x
                    rel_y = event.pos().y() - offset_y
                    
                    if 0 <= rel_x < pixmap_size.width() and 0 <= rel_y < pixmap_size.height():
                        scale_x = self.original_pixmap.width() / pixmap_size.width()
                        scale_y = self.original_pixmap.height() / pixmap_size.height()
                        
                        orig_x = int(rel_x * scale_x)
                        orig_y = int(rel_y * scale_y)
                        
                        self.imageClicked.emit(orig_x, orig_y)
        
        super().mouseReleaseEvent(event)
        
    def wheelEvent(self, event):
        """Maneja el evento de rueda del mouse"""
        self.wheelScrolled.emit(event.angleDelta().y(), event.pos())
        super().wheelEvent(event)


# Mantener compatibilidad con el ImageViewer original
class ImageViewer(InteractiveImageViewer):
    """Alias para mantener compatibilidad"""
    pass


class ImageSelector(QWidget):
    """Widget selector de imágenes con preview y validación"""
    
    imageSelected = pyqtSignal(str)  # Señal cuando se selecciona una imagen
    imageChanged = pyqtSignal(str)   # Señal cuando cambia la imagen
    imagesChanged = pyqtSignal(list)  # Señal cuando cambian las imágenes seleccionadas
    
    def __init__(self, title="Seleccionar Imagen", parent=None):
        super().__init__(parent)
        self.title = title
        self.current_image_path = None
        self.selected_images = []
        self._selection_mode = 'single'  # 'single' or 'multiple'
        self._minimum_images = 1
        self.setup_ui()
        
    def setup_ui(self):
        """Configura la interfaz del selector"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # Título
        title_label = QLabel(self.title)
        title_label.setFont(QFont("Arial", 12, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        # Drop zone para imágenes
        self.drop_zone = ImageDropZone(
            title="Arrastrar imagen aquí",
            subtitle="o hacer clic para seleccionar"
        )
        self.drop_zone.imageLoaded.connect(self.on_image_loaded)
        layout.addWidget(self.drop_zone)
        
        # Información de la imagen
        self.info_frame = QFrame()
        self.info_frame.setFrameStyle(QFrame.StyledPanel)
        self.info_frame.hide()
        
        info_layout = QVBoxLayout(self.info_frame)
        
        self.filename_label = QLabel()
        self.filename_label.setFont(QFont("Arial", 10, QFont.Bold))
        self.filename_label.setStyleSheet("color: #2196f3;")
        
        self.size_label = QLabel()
        self.size_label.setFont(QFont("Arial", 9))
        self.size_label.setStyleSheet("color: #666;")
        
        self.dimensions_label = QLabel()
        self.dimensions_label.setFont(QFont("Arial", 9))
        self.dimensions_label.setStyleSheet("color: #666;")
        
        info_layout.addWidget(self.filename_label)
        info_layout.addWidget(self.size_label)
        info_layout.addWidget(self.dimensions_label)
        
        layout.addWidget(self.info_frame)
        
        # Botones de acción
        buttons_layout = QHBoxLayout()
        
        self.change_btn = QPushButton("Cambiar Imagen")
        self.change_btn.clicked.connect(self.change_image)
        self.change_btn.hide()
        
        self.clear_btn = QPushButton("Limpiar")
        self.clear_btn.clicked.connect(self.clear_selection)
        self.clear_btn.hide()
        
        buttons_layout.addWidget(self.change_btn)
        buttons_layout.addWidget(self.clear_btn)
        buttons_layout.addStretch()
        
        layout.addLayout(buttons_layout)
        
    def on_image_loaded(self, image_path):
        """Maneja cuando se carga una imagen"""
        self.current_image_path = image_path
        self.update_image_info()
        self.show_image_info()
        self.imageSelected.emit(image_path)
        self.imageChanged.emit(image_path)
        
    def update_image_info(self):
        """Actualiza la información de la imagen"""
        if not self.current_image_path:
            return
            
        try:
            # Verificar que el archivo existe
            if not os.path.exists(self.current_image_path):
                raise FileNotFoundError(f"El archivo no existe: {self.current_image_path}")
            
            # Información del archivo
            file_info = QFileInfo(self.current_image_path)
            filename = file_info.fileName()
            file_size = file_info.size()
            
            # Verificar que el archivo es válido
            if file_size == 0:
                raise ValueError("El archivo está vacío")
            
            # Dimensiones de la imagen
            pixmap = QPixmap(self.current_image_path)
            if pixmap.isNull():
                raise ValueError("No se pudo cargar la imagen como QPixmap")
                
            width = pixmap.width()
            height = pixmap.height()
            
            # Actualizar labels
            self.filename_label.setText(f"📁 {filename}")
            
            # Formatear tamaño del archivo
            if file_size < 1024:
                size_str = f"{file_size} bytes"
            elif file_size < 1024 * 1024:
                size_str = f"{file_size / 1024:.1f} KB"
            else:
                size_str = f"{file_size / (1024 * 1024):.1f} MB"
                
            self.size_label.setText(f"📏 Tamaño: {size_str}")
            self.dimensions_label.setText(f"🖼️ Dimensiones: {width} × {height} px")
            
        except Exception as e:
            print(f"Error al actualizar información de imagen: {e}")
            self.filename_label.setText("❌ Error al cargar información")
            self.size_label.setText("")
            self.dimensions_label.setText("")
            
    def show_image_info(self):
        """Muestra la información de la imagen"""
        self.info_frame.show()
        self.change_btn.show()
        self.clear_btn.show()
        
    def hide_image_info(self):
        """Oculta la información de la imagen"""
        self.info_frame.hide()
        self.change_btn.hide()
        self.clear_btn.hide()
        
    def change_image(self):
        """Permite cambiar la imagen actual"""
        self.drop_zone.select_file()
        
    def clear_selection(self):
        """Limpia la selección actual"""
        self.current_image_path = None
        self.drop_zone.clear()
        self.hide_image_info()
        self.imageChanged.emit("")
        
    def get_selected_image(self):
        """Retorna la ruta de la imagen seleccionada"""
        return self.current_image_path
        
    def set_image(self, image_path):
        """Establece una imagen programáticamente"""
        if image_path and os.path.exists(image_path):
            self.drop_zone.load_image(image_path)
        else:
            self.clear_selection()

    def set_images(self, images):
        """Establece múltiples imágenes y emite actualización.
        - Si la lista está vacía, limpia la selección.
        - Si hay imágenes válidas, muestra la primera y emite `imagesChanged`.
        """
        try:
            valid = [p for p in (images or []) if isinstance(p, str) and os.path.exists(p)]
        except Exception:
            valid = []

        self.selected_images = valid

        if not valid:
            self.clear_selection()
            self.imagesChanged.emit([])
            return

        # Mostrar la primera imagen y actualizar info
        first = valid[0]
        # Cargar en drop_zone (esto disparará imageSelected/imageChanged)
        self.drop_zone.load_image(first)
        self.current_image_path = first
        # Emitir señal de múltiples imágenes
        self.imagesChanged.emit(valid)

    # --- Métodos de compatibilidad usados por ComparisonTab ---
    def set_selection_mode(self, mode: str):
        """Establece el modo de selección: 'single' o 'multiple'."""
        if mode not in ('single', 'multiple'):
            mode = 'single'
        self._selection_mode = mode

    def set_minimum_images(self, count: int):
        """Establece el mínimo de imágenes requeridas para la selección."""
        try:
            self._minimum_images = max(1, int(count))
        except Exception:
            self._minimum_images = 1

    def add_image(self, image_path: str):
        """Agrega una imagen a la selección cuando el modo es múltiple."""
        if not image_path or not os.path.exists(image_path):
            return
        if self._selection_mode == 'single':
            # En modo single, reemplazar selección
            self.set_images([image_path])
            return
        # Modo multiple: agregar si no existe
        if image_path not in self.selected_images:
            self.selected_images.append(image_path)
            # Mantener preview en la primera imagen seleccionada
            if not self.current_image_path:
                self.current_image_path = image_path
                self.drop_zone.load_image(image_path)
            # Emitir lista actualizada
            self.imagesChanged.emit(self.selected_images[:])


# Alias para compatibilidad
ImageDropWidget = ImageDropZone


# ============================================================================
# ANALYSIS WIDGETS - Widgets especializados para análisis balístico
# ============================================================================

class NISTStandardsWidget(QWidget):
    """Widget para configuración de estándares NIST"""
    
    configuration_changed = pyqtSignal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.config = {
            'enabled': True,
            'quality_validation': True,
            'metadata_compliance': True,
            'generate_report': True,
            'validation_level': 'standard',
            'image_quality_threshold': 0.7,
            'spatial_resolution_min': 300,  # DPI mínimo
            'bit_depth_min': 8
        }
        self.setup_ui()
        
    def setup_ui(self):
        """Configura la interfaz del widget NIST"""
        layout = QVBoxLayout(self)
        
        # Título
        title = QLabel("Configuración NIST")
        title.setProperty("class", "section-title")
        layout.addWidget(title)
        
        # Habilitar NIST
        self.enabled_checkbox = QCheckBox("Habilitar validación NIST")
        self.enabled_checkbox.setChecked(self.config['enabled'])
        self.enabled_checkbox.toggled.connect(self._on_config_changed)
        layout.addWidget(self.enabled_checkbox)
        
        # Configuraciones principales
        config_frame = QFrame()
        config_layout = QGridLayout(config_frame)
        
        # Validación de calidad
        self.quality_checkbox = QCheckBox("Validación de calidad de imagen")
        self.quality_checkbox.setChecked(self.config['quality_validation'])
        self.quality_checkbox.toggled.connect(self._on_config_changed)
        config_layout.addWidget(self.quality_checkbox, 0, 0, 1, 2)
        
        # Cumplimiento de metadatos
        self.metadata_checkbox = QCheckBox("Cumplimiento de metadatos")
        self.metadata_checkbox.setChecked(self.config['metadata_compliance'])
        self.metadata_checkbox.toggled.connect(self._on_config_changed)
        config_layout.addWidget(self.metadata_checkbox, 1, 0, 1, 2)
        
        # Generar reporte
        self.report_checkbox = QCheckBox("Generar reporte de cumplimiento")
        self.report_checkbox.setChecked(self.config['generate_report'])
        self.report_checkbox.toggled.connect(self._on_config_changed)
        config_layout.addWidget(self.report_checkbox, 2, 0, 1, 2)
        
        # Nivel de validación
        config_layout.addWidget(QLabel("Nivel de validación:"), 3, 0)
        self.validation_combo = QComboBox()
        self.validation_combo.addItems(['basic', 'standard', 'strict'])
        self.validation_combo.setCurrentText(self.config['validation_level'])
        self.validation_combo.currentTextChanged.connect(self._on_config_changed)
        config_layout.addWidget(self.validation_combo, 3, 1)
        
        # Umbral de calidad
        config_layout.addWidget(QLabel("Umbral de calidad:"), 4, 0)
        self.quality_spinbox = QDoubleSpinBox()
        self.quality_spinbox.setRange(0.1, 1.0)
        self.quality_spinbox.setSingleStep(0.1)
        self.quality_spinbox.setValue(self.config['image_quality_threshold'])
        self.quality_spinbox.valueChanged.connect(self._on_config_changed)
        config_layout.addWidget(self.quality_spinbox, 4, 1)
        
        # Resolución espacial mínima
        config_layout.addWidget(QLabel("Resolución mínima (DPI):"), 5, 0)
        self.resolution_spinbox = QSpinBox()
        self.resolution_spinbox.setRange(72, 1200)
        self.resolution_spinbox.setValue(self.config['spatial_resolution_min'])
        self.resolution_spinbox.valueChanged.connect(self._on_config_changed)
        config_layout.addWidget(self.resolution_spinbox, 5, 1)
        
        layout.addWidget(config_frame)
        
        # Información adicional
        info_text = QTextEdit()
        info_text.setMaximumHeight(80)
        info_text.setPlainText("Los estándares NIST garantizan la calidad y trazabilidad de los análisis balísticos forenses.")
        info_text.setReadOnly(True)
        layout.addWidget(info_text)
        
    def _on_config_changed(self):
        """Maneja cambios en la configuración"""
        self.config.update({
            'enabled': self.enabled_checkbox.isChecked(),
            'quality_validation': self.quality_checkbox.isChecked(),
            'metadata_compliance': self.metadata_checkbox.isChecked(),
            'generate_report': self.report_checkbox.isChecked(),
            'validation_level': self.validation_combo.currentText(),
            'image_quality_threshold': self.quality_spinbox.value(),
            'spatial_resolution_min': self.resolution_spinbox.value()
        })
        self.configuration_changed.emit(self.config.copy())
        
    def get_configuration(self) -> dict:
        """Obtiene la configuración actual"""
        return self.config.copy()
        
    def set_configuration(self, config: dict):
        """Establece la configuración"""
        self.config.update(config)
        self._update_ui_from_config()
        
    def _update_ui_from_config(self):
        """Actualiza la UI desde la configuración"""
        self.enabled_checkbox.setChecked(self.config['enabled'])
        self.quality_checkbox.setChecked(self.config['quality_validation'])
        self.metadata_checkbox.setChecked(self.config['metadata_compliance'])
        self.report_checkbox.setChecked(self.config['generate_report'])
        self.validation_combo.setCurrentText(self.config['validation_level'])
        self.quality_spinbox.setValue(self.config['image_quality_threshold'])
        self.resolution_spinbox.setValue(self.config['spatial_resolution_min'])


class AFTEAnalysisWidget(QWidget):
    """Widget para configuración de análisis AFTE"""
    
    configuration_changed = pyqtSignal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.config = {
            'enabled': True,
            'generate_conclusions': True,
            'confidence_threshold': 0.8,
            'feature_analysis': True,
            'statistical_analysis': True,
            'conclusion_categories': ['identification', 'elimination', 'inconclusive'],
            'minimum_features': 6,
            'quality_assessment': True
        }
        self.setup_ui()
        
    def setup_ui(self):
        """Configura la interfaz del widget AFTE"""
        layout = QVBoxLayout(self)
        
        # Título
        title = QLabel("Configuración AFTE")
        title.setProperty("class", "section-title")
        layout.addWidget(title)
        
        # Habilitar AFTE
        self.enabled_checkbox = QCheckBox("Habilitar análisis AFTE")
        self.enabled_checkbox.setChecked(self.config['enabled'])
        self.enabled_checkbox.toggled.connect(self._on_config_changed)
        layout.addWidget(self.enabled_checkbox)
        
        # Configuraciones principales
        config_frame = QFrame()
        config_layout = QGridLayout(config_frame)
        
        # Generar conclusiones
        self.conclusions_checkbox = QCheckBox("Generar conclusiones automáticas")
        self.conclusions_checkbox.setChecked(self.config['generate_conclusions'])
        self.conclusions_checkbox.toggled.connect(self._on_config_changed)
        config_layout.addWidget(self.conclusions_checkbox, 0, 0, 1, 2)
        
        # Análisis de características
        self.features_checkbox = QCheckBox("Análisis de características")
        self.features_checkbox.setChecked(self.config['feature_analysis'])
        self.features_checkbox.toggled.connect(self._on_config_changed)
        config_layout.addWidget(self.features_checkbox, 1, 0, 1, 2)
        
        # Análisis estadístico
        self.stats_checkbox = QCheckBox("Análisis estadístico")
        self.stats_checkbox.setChecked(self.config['statistical_analysis'])
        self.stats_checkbox.toggled.connect(self._on_config_changed)
        config_layout.addWidget(self.stats_checkbox, 2, 0, 1, 2)
        
        # Evaluación de calidad
        self.quality_checkbox = QCheckBox("Evaluación de calidad")
        self.quality_checkbox.setChecked(self.config['quality_assessment'])
        self.quality_checkbox.toggled.connect(self._on_config_changed)
        config_layout.addWidget(self.quality_checkbox, 3, 0, 1, 2)
        
        # Umbral de confianza
        config_layout.addWidget(QLabel("Umbral de confianza:"), 4, 0)
        self.confidence_spinbox = QDoubleSpinBox()
        self.confidence_spinbox.setRange(0.1, 1.0)
        self.confidence_spinbox.setSingleStep(0.1)
        self.confidence_spinbox.setValue(self.config['confidence_threshold'])
        self.confidence_spinbox.valueChanged.connect(self._on_config_changed)
        config_layout.addWidget(self.confidence_spinbox, 4, 1)
        
        # Características mínimas
        config_layout.addWidget(QLabel("Características mínimas:"), 5, 0)
        self.features_spinbox = QSpinBox()
        self.features_spinbox.setRange(1, 20)
        self.features_spinbox.setValue(self.config['minimum_features'])
        self.features_spinbox.valueChanged.connect(self._on_config_changed)
        config_layout.addWidget(self.features_spinbox, 5, 1)
        
        layout.addWidget(config_frame)
        
        # Información adicional
        info_text = QTextEdit()
        info_text.setMaximumHeight(80)
        info_text.setPlainText("AFTE (Association of Firearm and Tool Mark Examiners) proporciona estándares para conclusiones forenses.")
        info_text.setReadOnly(True)
        layout.addWidget(info_text)
        
    def _on_config_changed(self):
        """Maneja cambios en la configuración"""
        self.config.update({
            'enabled': self.enabled_checkbox.isChecked(),
            'generate_conclusions': self.conclusions_checkbox.isChecked(),
            'feature_analysis': self.features_checkbox.isChecked(),
            'statistical_analysis': self.stats_checkbox.isChecked(),
            'quality_assessment': self.quality_checkbox.isChecked(),
            'confidence_threshold': self.confidence_spinbox.value(),
            'minimum_features': self.features_spinbox.value()
        })
        self.configuration_changed.emit(self.config.copy())
        
    def get_configuration(self) -> dict:
        """Obtiene la configuración actual"""
        return self.config.copy()
        
    def set_configuration(self, config: dict):
        """Establece la configuración"""
        self.config.update(config)
        self._update_ui_from_config()
        
    def _update_ui_from_config(self):
        """Actualiza la UI desde la configuración"""
        self.enabled_checkbox.setChecked(self.config['enabled'])
        self.conclusions_checkbox.setChecked(self.config['generate_conclusions'])
        self.features_checkbox.setChecked(self.config['feature_analysis'])
        self.stats_checkbox.setChecked(self.config['statistical_analysis'])
        self.quality_checkbox.setChecked(self.config['quality_assessment'])
        self.confidence_spinbox.setValue(self.config['confidence_threshold'])
        self.features_spinbox.setValue(self.config['minimum_features'])


class DeepLearningWidget(QWidget):
    """Widget para configuración de Deep Learning"""
    
    configuration_changed = pyqtSignal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.config = {
            'enabled': False,  # Deshabilitado por defecto según contexto
            'model_type': 'CNN',
            'use_pretrained': True,
            'confidence_threshold': 0.75,
            'batch_size': 16,
            'feature_extraction': True,
            'similarity_matching': True,
            'gpu_acceleration': True,
            'model_path': '',
            'preprocessing_enabled': True
        }
        self.setup_ui()
        
    def setup_ui(self):
        """Configura la interfaz del widget Deep Learning"""
        layout = QVBoxLayout(self)
        
        # Título
        title = QLabel("Configuración Deep Learning")
        title.setProperty("class", "section-title")
        layout.addWidget(title)
        
        # Advertencia de disponibilidad
        warning_frame = QFrame()
        warning_frame.setProperty("class", "warning-frame")
        warning_layout = QHBoxLayout(warning_frame)
        warning_label = QLabel("⚠️ Funcionalidad en desarrollo - Disponibilidad limitada")
        warning_label.setProperty("class", "warning-text")
        warning_layout.addWidget(warning_label)
        layout.addWidget(warning_frame)
        
        # Habilitar Deep Learning
        self.enabled_checkbox = QCheckBox("Habilitar Deep Learning")
        self.enabled_checkbox.setChecked(self.config['enabled'])
        self.enabled_checkbox.toggled.connect(self._on_config_changed)
        layout.addWidget(self.enabled_checkbox)
        
        # Configuraciones principales
        config_frame = QFrame()
        config_layout = QGridLayout(config_frame)
        
        # Tipo de modelo
        config_layout.addWidget(QLabel("Tipo de modelo:"), 0, 0)
        self.model_combo = QComboBox()
        self.model_combo.addItems(['CNN', 'ResNet', 'EfficientNet', 'Siamese'])
        self.model_combo.setCurrentText(self.config['model_type'])
        self.model_combo.currentTextChanged.connect(self._on_config_changed)
        config_layout.addWidget(self.model_combo, 0, 1)
        
        # Usar modelo preentrenado
        self.pretrained_checkbox = QCheckBox("Usar modelo preentrenado")
        self.pretrained_checkbox.setChecked(self.config['use_pretrained'])
        self.pretrained_checkbox.toggled.connect(self._on_config_changed)
        config_layout.addWidget(self.pretrained_checkbox, 1, 0, 1, 2)
        
        # Extracción de características
        self.features_checkbox = QCheckBox("Extracción de características")
        self.features_checkbox.setChecked(self.config['feature_extraction'])
        self.features_checkbox.toggled.connect(self._on_config_changed)
        config_layout.addWidget(self.features_checkbox, 2, 0, 1, 2)
        
        # Matching de similitud
        self.matching_checkbox = QCheckBox("Matching de similitud")
        self.matching_checkbox.setChecked(self.config['similarity_matching'])
        self.matching_checkbox.toggled.connect(self._on_config_changed)
        config_layout.addWidget(self.matching_checkbox, 3, 0, 1, 2)
        
        # Aceleración GPU
        self.gpu_checkbox = QCheckBox("Aceleración GPU")
        self.gpu_checkbox.setChecked(self.config['gpu_acceleration'])
        self.gpu_checkbox.toggled.connect(self._on_config_changed)
        config_layout.addWidget(self.gpu_checkbox, 4, 0, 1, 2)
        
        # Umbral de confianza
        config_layout.addWidget(QLabel("Umbral de confianza:"), 5, 0)
        self.confidence_spinbox = QDoubleSpinBox()
        self.confidence_spinbox.setRange(0.1, 1.0)
        self.confidence_spinbox.setSingleStep(0.05)
        self.confidence_spinbox.setValue(self.config['confidence_threshold'])
        self.confidence_spinbox.valueChanged.connect(self._on_config_changed)
        config_layout.addWidget(self.confidence_spinbox, 5, 1)
        
        # Tamaño de lote
        config_layout.addWidget(QLabel("Tamaño de lote:"), 6, 0)
        self.batch_spinbox = QSpinBox()
        self.batch_spinbox.setRange(1, 64)
        self.batch_spinbox.setValue(self.config['batch_size'])
        self.batch_spinbox.valueChanged.connect(self._on_config_changed)
        config_layout.addWidget(self.batch_spinbox, 6, 1)
        
        layout.addWidget(config_frame)
        
        # Información adicional
        info_text = QTextEdit()
        info_text.setMaximumHeight(80)
        info_text.setPlainText("Deep Learning proporciona análisis avanzado mediante redes neuronales especializadas en balística.")
        info_text.setReadOnly(True)
        layout.addWidget(info_text)
        
    def _on_config_changed(self):
        """Maneja cambios en la configuración"""
        self.config.update({
            'enabled': self.enabled_checkbox.isChecked(),
            'model_type': self.model_combo.currentText(),
            'use_pretrained': self.pretrained_checkbox.isChecked(),
            'feature_extraction': self.features_checkbox.isChecked(),
            'similarity_matching': self.matching_checkbox.isChecked(),
            'gpu_acceleration': self.gpu_checkbox.isChecked(),
            'confidence_threshold': self.confidence_spinbox.value(),
            'batch_size': self.batch_spinbox.value()
        })
        self.configuration_changed.emit(self.config.copy())
        
    def get_configuration(self) -> dict:
        """Obtiene la configuración actual"""
        return self.config.copy()
        
    def set_configuration(self, config: dict):
        """Establece la configuración"""
        self.config.update(config)
        self._update_ui_from_config()
        
    def _update_ui_from_config(self):
        """Actualiza la UI desde la configuración"""
        self.enabled_checkbox.setChecked(self.config['enabled'])
        self.model_combo.setCurrentText(self.config['model_type'])
        self.pretrained_checkbox.setChecked(self.config['use_pretrained'])
        self.features_checkbox.setChecked(self.config['feature_extraction'])
        self.matching_checkbox.setChecked(self.config['similarity_matching'])
        self.gpu_checkbox.setChecked(self.config['gpu_acceleration'])
        self.confidence_spinbox.setValue(self.config['confidence_threshold'])
        self.batch_spinbox.setValue(self.config['batch_size'])


class ImageProcessingWidget(QWidget):
    """Widget para configuración de procesamiento de imágenes"""
    
    configuration_changed = pyqtSignal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.config = {
            'enabled': True,
            'enhance_contrast': True,
            'denoise': True,
            'edge_enhancement': True,
            'illumination_correction': True,
            'orientation_normalization': True,
            'roi_detection': True,
            'spatial_calibration': True,
            'contrast_method': 'adaptive',
            'denoise_method': 'bilateral',
            'edge_method': 'unsharp_mask',
            'roi_method': 'automatic'
        }
        self.setup_ui()
        
    def setup_ui(self):
        """Configura la interfaz del widget de procesamiento"""
        layout = QVBoxLayout(self)
        
        # Título
        title = QLabel("Configuración Procesamiento de Imágenes")
        title.setProperty("class", "section-title")
        layout.addWidget(title)
        
        # Habilitar procesamiento
        self.enabled_checkbox = QCheckBox("Habilitar procesamiento de imágenes")
        self.enabled_checkbox.setChecked(self.config['enabled'])
        self.enabled_checkbox.toggled.connect(self._on_config_changed)
        layout.addWidget(self.enabled_checkbox)
        
        # Configuraciones principales
        config_frame = QFrame()
        config_layout = QGridLayout(config_frame)
        
        # Mejora de contraste
        self.contrast_checkbox = QCheckBox("Mejora de contraste")
        self.contrast_checkbox.setChecked(self.config['enhance_contrast'])
        self.contrast_checkbox.toggled.connect(self._on_config_changed)
        config_layout.addWidget(self.contrast_checkbox, 0, 0)
        
        self.contrast_combo = QComboBox()
        self.contrast_combo.addItems(['adaptive', 'histogram', 'clahe'])
        self.contrast_combo.setCurrentText(self.config['contrast_method'])
        self.contrast_combo.currentTextChanged.connect(self._on_config_changed)
        config_layout.addWidget(self.contrast_combo, 0, 1)
        
        # Reducción de ruido
        self.denoise_checkbox = QCheckBox("Reducción de ruido")
        self.denoise_checkbox.setChecked(self.config['denoise'])
        self.denoise_checkbox.toggled.connect(self._on_config_changed)
        config_layout.addWidget(self.denoise_checkbox, 1, 0)
        
        self.denoise_combo = QComboBox()
        self.denoise_combo.addItems(['bilateral', 'gaussian', 'median', 'nlm'])
        self.denoise_combo.setCurrentText(self.config['denoise_method'])
        self.denoise_combo.currentTextChanged.connect(self._on_config_changed)
        config_layout.addWidget(self.denoise_combo, 1, 1)
        
        # Mejora de bordes
        self.edge_checkbox = QCheckBox("Mejora de bordes")
        self.edge_checkbox.setChecked(self.config['edge_enhancement'])
        self.edge_checkbox.toggled.connect(self._on_config_changed)
        config_layout.addWidget(self.edge_checkbox, 2, 0)
        
        self.edge_combo = QComboBox()
        self.edge_combo.addItems(['unsharp_mask', 'laplacian', 'sobel'])
        self.edge_combo.setCurrentText(self.config['edge_method'])
        self.edge_combo.currentTextChanged.connect(self._on_config_changed)
        config_layout.addWidget(self.edge_combo, 2, 1)
        
        # Corrección de iluminación
        self.illumination_checkbox = QCheckBox("Corrección de iluminación")
        self.illumination_checkbox.setChecked(self.config['illumination_correction'])
        self.illumination_checkbox.toggled.connect(self._on_config_changed)
        config_layout.addWidget(self.illumination_checkbox, 3, 0, 1, 2)
        
        # Normalización de orientación
        self.orientation_checkbox = QCheckBox("Normalización de orientación")
        self.orientation_checkbox.setChecked(self.config['orientation_normalization'])
        self.orientation_checkbox.toggled.connect(self._on_config_changed)
        config_layout.addWidget(self.orientation_checkbox, 4, 0, 1, 2)
        
        # Detección de ROI
        self.roi_checkbox = QCheckBox("Detección de ROI")
        self.roi_checkbox.setChecked(self.config['roi_detection'])
        self.roi_checkbox.toggled.connect(self._on_config_changed)
        config_layout.addWidget(self.roi_checkbox, 5, 0)
        
        self.roi_combo = QComboBox()
        self.roi_combo.addItems(['automatic', 'manual', 'template'])
        self.roi_combo.setCurrentText(self.config['roi_method'])
        self.roi_combo.currentTextChanged.connect(self._on_config_changed)
        config_layout.addWidget(self.roi_combo, 5, 1)
        
        # Calibración espacial
        self.calibration_checkbox = QCheckBox("Calibración espacial")
        self.calibration_checkbox.setChecked(self.config['spatial_calibration'])
        self.calibration_checkbox.toggled.connect(self._on_config_changed)
        config_layout.addWidget(self.calibration_checkbox, 6, 0, 1, 2)
        
        layout.addWidget(config_frame)
        
        # Información adicional
        info_text = QTextEdit()
        info_text.setMaximumHeight(80)
        info_text.setPlainText("Procesamiento de imágenes optimizado para análisis balístico forense con técnicas avanzadas.")
        info_text.setReadOnly(True)
        layout.addWidget(info_text)
        
    def _on_config_changed(self):
        """Maneja cambios en la configuración"""
        self.config.update({
            'enabled': self.enabled_checkbox.isChecked(),
            'enhance_contrast': self.contrast_checkbox.isChecked(),
            'denoise': self.denoise_checkbox.isChecked(),
            'edge_enhancement': self.edge_checkbox.isChecked(),
            'illumination_correction': self.illumination_checkbox.isChecked(),
            'orientation_normalization': self.orientation_checkbox.isChecked(),
            'roi_detection': self.roi_checkbox.isChecked(),
            'spatial_calibration': self.calibration_checkbox.isChecked(),
            'contrast_method': self.contrast_combo.currentText(),
            'denoise_method': self.denoise_combo.currentText(),
            'edge_method': self.edge_combo.currentText(),
            'roi_method': self.roi_combo.currentText()
        })
        self.configuration_changed.emit(self.config.copy())
        
    def get_configuration(self) -> dict:
        """Obtiene la configuración actual"""
        return self.config.copy()
        
    def set_configuration(self, config: dict):
        """Establece la configuración"""
        self.config.update(config)
        self._update_ui_from_config()
        
    def _update_ui_from_config(self):
        """Actualiza la UI desde la configuración"""
        self.enabled_checkbox.setChecked(self.config['enabled'])
        self.contrast_checkbox.setChecked(self.config['enhance_contrast'])
        self.denoise_checkbox.setChecked(self.config['denoise'])
        self.edge_checkbox.setChecked(self.config['edge_enhancement'])
        self.illumination_checkbox.setChecked(self.config['illumination_correction'])
        self.orientation_checkbox.setChecked(self.config['orientation_normalization'])
        self.roi_checkbox.setChecked(self.config['roi_detection'])
        self.calibration_checkbox.setChecked(self.config['spatial_calibration'])
        self.contrast_combo.setCurrentText(self.config['contrast_method'])
        self.denoise_combo.setCurrentText(self.config['denoise_method'])
        self.edge_combo.setCurrentText(self.config['edge_method'])
        self.roi_combo.setCurrentText(self.config['roi_method'])


# ============================================================================
# VISUALIZATION WIDGETS - Widgets para visualización y paneles
# ============================================================================

class VisualizationPanel(QWidget):
    """Panel de visualización para resultados de análisis"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        
    def setup_ui(self):
        """Configura la interfaz del panel de visualización"""
        layout = QVBoxLayout(self)
        
        # Título
        title = QLabel("Panel de Visualización")
        title.setProperty("class", "section-title")
        layout.addWidget(title)
        
        # Área de contenido
        content_frame = QFrame()
        content_frame.setProperty("class", "content-frame")
        content_layout = QVBoxLayout(content_frame)
        
        # Placeholder para visualizaciones
        placeholder = QLabel("Visualizaciones aparecerán aquí")
        placeholder.setAlignment(Qt.AlignCenter)
        placeholder.setProperty("class", "placeholder-text")
        content_layout.addWidget(placeholder)
        
        layout.addWidget(content_frame)


class ResultsPanel(QWidget):
    """Panel de resultados para mostrar análisis completados"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        
    def setup_ui(self):
        """Configura la interfaz del panel de resultados"""
        layout = QVBoxLayout(self)
        
        # Título
        title = QLabel("Panel de Resultados")
        title.setProperty("class", "section-title")
        layout.addWidget(title)
        
        # Área de contenido
        content_frame = QFrame()
        content_frame.setProperty("class", "content-frame")
        content_layout = QVBoxLayout(content_frame)
        
        # Placeholder para resultados
        placeholder = QLabel("Los resultados aparecerán aquí")
        placeholder.setAlignment(Qt.AlignCenter)
        placeholder.setProperty("class", "placeholder-text")
        content_layout.addWidget(placeholder)
        
        layout.addWidget(content_frame)