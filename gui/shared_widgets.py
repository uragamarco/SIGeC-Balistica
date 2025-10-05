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
    QSizePolicy, QSpacerItem, QGridLayout, QTextEdit
)
from PyQt5.QtCore import Qt, pyqtSignal, QPropertyAnimation, QEasingCurve, QRect, QTimer
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

class ImageViewer(QFrame):
    """Visor de imágenes con zoom y controles"""
    
    def __init__(self):
        super().__init__()
        self.image_path = None
        self.zoom_factor = 1.0
        self.setup_ui()
        
    def setup_ui(self):
        """Configura la interfaz del visor"""
        self.setProperty("class", "card")
        
        layout = QVBoxLayout(self)
        
        # Controles superiores
        controls_layout = QHBoxLayout()
        
        self.zoom_out_btn = QPushButton("🔍-")
        self.zoom_out_btn.setMaximumWidth(40)
        self.zoom_out_btn.clicked.connect(self.zoom_out)
        controls_layout.addWidget(self.zoom_out_btn)
        
        self.zoom_label = QLabel("100%")
        self.zoom_label.setAlignment(Qt.AlignCenter)
        self.zoom_label.setMinimumWidth(60)
        controls_layout.addWidget(self.zoom_label)
        
        self.zoom_in_btn = QPushButton("🔍+")
        self.zoom_in_btn.setMaximumWidth(40)
        self.zoom_in_btn.clicked.connect(self.zoom_in)
        controls_layout.addWidget(self.zoom_in_btn)
        
        self.fit_btn = QPushButton("Ajustar")
        self.fit_btn.clicked.connect(self.fit_to_window)
        controls_layout.addWidget(self.fit_btn)
        
        controls_layout.addItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))
        
        layout.addLayout(controls_layout)
        
        # Área de imagen con scroll
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setAlignment(Qt.AlignCenter)
        
        self.image_label = QLabel("No hay imagen cargada")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("color: #757575; font-size: 14px;")
        self.image_label.setMinimumSize(300, 200)
        
        self.scroll_area.setWidget(self.image_label)
        layout.addWidget(self.scroll_area)
        
    def load_image(self, image_path: str):
        """Carga una imagen en el visor"""
        try:
            self.image_path = image_path
            self.original_pixmap = QPixmap(image_path)
            
            if not self.original_pixmap.isNull():
                self.fit_to_window()
            else:
                self.image_label.setText("Error: No se pudo cargar la imagen")
                
        except Exception as e:
            self.image_label.setText(f"Error: {str(e)}")
            
    def update_image_display(self):
        """Actualiza la visualización de la imagen"""
        if hasattr(self, 'original_pixmap') and not self.original_pixmap.isNull():
            scaled_pixmap = self.original_pixmap.scaled(
                self.original_pixmap.size() * self.zoom_factor,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.image_label.setPixmap(scaled_pixmap)
            self.image_label.resize(scaled_pixmap.size())
            
            # Actualizar etiqueta de zoom
            self.zoom_label.setText(f"{int(self.zoom_factor * 100)}%")
            
    def zoom_in(self):
        """Aumenta el zoom"""
        self.zoom_factor = min(self.zoom_factor * 1.25, 5.0)
        self.update_image_display()
        
    def zoom_out(self):
        """Disminuye el zoom"""
        self.zoom_factor = max(self.zoom_factor / 1.25, 0.1)
        self.update_image_display()
        
    def fit_to_window(self):
        """Ajusta la imagen al tamaño de la ventana"""
        if hasattr(self, 'original_pixmap') and not self.original_pixmap.isNull():
            scroll_size = self.scroll_area.size()
            image_size = self.original_pixmap.size()
            
            # Calcular factor de escala para ajustar
            scale_x = (scroll_size.width() - 20) / image_size.width()
            scale_y = (scroll_size.height() - 20) / image_size.height()
            
            self.zoom_factor = min(scale_x, scale_y, 1.0)  # No hacer zoom in más allá del 100%
            self.update_image_display()
            
    def clear(self):
        """Limpia el visor"""
        self.image_path = None
        self.zoom_factor = 1.0
        self.image_label.clear()
        self.image_label.setText("No hay imagen cargada")
        self.zoom_label.setText("100%")