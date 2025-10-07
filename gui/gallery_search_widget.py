"""
Widget de Galería para Resultados de Búsqueda
Proporciona una vista de galería con miniaturas para resultados de búsqueda
en base de datos con comparación instantánea al hacer hover.
"""

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QFrame, QSlider, QCheckBox,
                             QGroupBox, QComboBox, QSpinBox, QScrollArea,
                             QGridLayout, QSizePolicy, QProgressBar,
                             QLineEdit, QSplitter, QTextEdit, QTabWidget,
                             QApplication, QMenu, QAction)
from PyQt5.QtCore import (Qt, pyqtSignal, QTimer, QThread, pyqtSlot,
                          QPropertyAnimation, QEasingCurve, QRect, QSize)
from PyQt5.QtGui import (QPixmap, QPainter, QPen, QColor, QFont, QBrush, 
                         QLinearGradient, QIcon, QPalette, QMovie,
                         QCursor, QMouseEvent)
import numpy as np
import os
import json
from datetime import datetime


class SearchResult:
    """Clase para representar un resultado de búsqueda"""
    
    def __init__(self, sample_id, image_path, cmc_score, confidence, 
                 match_type="identification", metadata=None):
        self.sample_id = sample_id
        self.image_path = image_path
        self.cmc_score = cmc_score
        self.confidence = confidence
        self.match_type = match_type  # identification, inconclusive, elimination
        self.metadata = metadata or {}
        self.thumbnail = None
        self.selected = False
        
    def get_confidence_color(self):
        """Retorna el color basado en la confianza"""
        if self.confidence >= 0.8:
            return QColor(76, 175, 80)  # Verde
        elif self.confidence >= 0.6:
            return QColor(255, 193, 7)  # Amarillo
        else:
            return QColor(244, 67, 54)  # Rojo
            
    def get_match_type_icon(self):
        """Retorna el ícono basado en el tipo de coincidencia"""
        icons = {
            "identification": "✓",
            "inconclusive": "?",
            "elimination": "✗"
        }
        return icons.get(self.match_type, "?")


class ThumbnailWidget(QLabel):
    """Widget para mostrar miniatura de resultado con información"""
    
    clicked = pyqtSignal(str)  # sample_id
    hovered = pyqtSignal(str)  # sample_id
    doubleClicked = pyqtSignal(str)  # sample_id
    contextMenuRequested = pyqtSignal(str, object)  # sample_id, position
    
    def __init__(self, search_result, thumbnail_size=150, parent=None):
        super().__init__(parent)
        self.search_result = search_result
        self.thumbnail_size = thumbnail_size
        self.is_hovered = False
        self.is_selected = False
        
        self.setup_ui()
        self.load_thumbnail()
        
        # Configurar eventos
        self.setMouseTracking(True)
        self.setAttribute(Qt.WA_Hover, True)
        
        # Animación para hover
        self.hover_animation = QPropertyAnimation(self, b"geometry")
        self.hover_animation.setDuration(200)
        self.hover_animation.setEasingCurve(QEasingCurve.OutCubic)
        
    def setup_ui(self):
        """Configura la interfaz del thumbnail"""
        self.setFixedSize(self.thumbnail_size + 20, self.thumbnail_size + 60)
        self.setFrameStyle(QFrame.Box)
        self.setLineWidth(2)
        self.setAlignment(Qt.AlignCenter)
        
        # Configurar estilo base
        self.update_style()
        
        # Tooltip con información detallada
        tooltip_text = f"ID: {self.search_result.sample_id}\n"
        tooltip_text += f"Score CMC: {self.search_result.cmc_score:.4f}\n"
        tooltip_text += f"Confianza: {self.search_result.confidence:.2%}\n"
        tooltip_text += f"Tipo: {self.search_result.match_type.title()}"
        
        if self.search_result.metadata:
            tooltip_text += "\n\nMetadatos:"
            for key, value in self.search_result.metadata.items():
                tooltip_text += f"\n{key}: {value}"
                
        self.setToolTip(tooltip_text)
        
    def update_style(self):
        """Actualiza el estilo del widget"""
        confidence_color = self.search_result.get_confidence_color()
        
        if self.is_selected:
            border_color = "#2196F3"
            border_width = 3
            background = "#E3F2FD"
        elif self.is_hovered:
            border_color = confidence_color.name()
            border_width = 2
            background = "#F5F5F5"
        else:
            border_color = confidence_color.name()
            border_width = 1
            background = "#FFFFFF"
            
        style = f"""
        ThumbnailWidget {{
            border: {border_width}px solid {border_color};
            background-color: {background};
            border-radius: 8px;
            padding: 5px;
        }}
        """
        self.setStyleSheet(style)
        
    def load_thumbnail(self):
        """Carga la miniatura de la imagen"""
        if os.path.exists(self.search_result.image_path):
            pixmap = QPixmap(self.search_result.image_path)
            if not pixmap.isNull():
                # Redimensionar manteniendo aspecto
                scaled_pixmap = pixmap.scaled(
                    self.thumbnail_size - 10, self.thumbnail_size - 40,
                    Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
                
                # Crear pixmap compuesto con información
                composite = QPixmap(self.thumbnail_size - 10, self.thumbnail_size - 10)
                composite.fill(Qt.white)
                
                painter = QPainter(composite)
                
                # Dibujar imagen
                x = (composite.width() - scaled_pixmap.width()) // 2
                y = 5
                painter.drawPixmap(x, y, scaled_pixmap)
                
                # Dibujar información
                painter.setFont(QFont("Arial", 8, QFont.Bold))
                
                # ID de muestra
                painter.setPen(QColor(33, 33, 33))
                painter.drawText(5, composite.height() - 25, 
                               f"ID: {self.search_result.sample_id}")
                
                # Score CMC
                painter.setPen(QColor(33, 150, 243))
                painter.drawText(5, composite.height() - 15, 
                               f"CMC: {self.search_result.cmc_score:.3f}")
                
                # Confianza con color
                confidence_color = self.search_result.get_confidence_color()
                painter.setPen(confidence_color)
                painter.drawText(5, composite.height() - 5, 
                               f"{self.search_result.confidence:.1%}")
                
                # Ícono de tipo de coincidencia
                painter.setFont(QFont("Arial", 12, QFont.Bold))
                icon = self.search_result.get_match_type_icon()
                painter.drawText(composite.width() - 20, 20, icon)
                
                painter.end()
                
                self.setPixmap(composite)
                self.search_result.thumbnail = composite
        else:
            # Imagen placeholder
            self.create_placeholder()
            
    def create_placeholder(self):
        """Crea una imagen placeholder"""
        placeholder = QPixmap(self.thumbnail_size - 10, self.thumbnail_size - 10)
        placeholder.fill(QColor(240, 240, 240))
        
        painter = QPainter(placeholder)
        painter.setPen(QColor(150, 150, 150))
        painter.setFont(QFont("Arial", 10))
        
        # Dibujar texto placeholder
        painter.drawText(placeholder.rect(), Qt.AlignCenter, 
                        "Imagen\nno disponible")
        
        # Información básica
        painter.setFont(QFont("Arial", 8, QFont.Bold))
        painter.setPen(QColor(33, 33, 33))
        painter.drawText(5, placeholder.height() - 25, 
                       f"ID: {self.search_result.sample_id}")
        
        painter.setPen(QColor(33, 150, 243))
        painter.drawText(5, placeholder.height() - 15, 
                       f"CMC: {self.search_result.cmc_score:.3f}")
        
        confidence_color = self.search_result.get_confidence_color()
        painter.setPen(confidence_color)
        painter.drawText(5, placeholder.height() - 5, 
                       f"{self.search_result.confidence:.1%}")
        
        painter.end()
        self.setPixmap(placeholder)
        
    def mousePressEvent(self, event):
        """Maneja clics del mouse"""
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self.search_result.sample_id)
        elif event.button() == Qt.RightButton:
            self.contextMenuRequested.emit(self.search_result.sample_id, 
                                         event.globalPos())
        super().mousePressEvent(event)
        
    def mouseDoubleClickEvent(self, event):
        """Maneja doble clic"""
        if event.button() == Qt.LeftButton:
            self.doubleClicked.emit(self.search_result.sample_id)
        super().mouseDoubleClickEvent(event)
        
    def enterEvent(self, event):
        """Maneja entrada del mouse"""
        self.is_hovered = True
        self.update_style()
        self.hovered.emit(self.search_result.sample_id)
        
        # Animación de hover
        current_rect = self.geometry()
        hover_rect = QRect(current_rect.x() - 2, current_rect.y() - 2,
                          current_rect.width() + 4, current_rect.height() + 4)
        
        self.hover_animation.setStartValue(current_rect)
        self.hover_animation.setEndValue(hover_rect)
        self.hover_animation.start()
        
        super().enterEvent(event)
        
    def leaveEvent(self, event):
        """Maneja salida del mouse"""
        self.is_hovered = False
        self.update_style()
        
        # Revertir animación
        current_rect = self.geometry()
        normal_rect = QRect(current_rect.x() + 2, current_rect.y() + 2,
                           current_rect.width() - 4, current_rect.height() - 4)
        
        self.hover_animation.setStartValue(current_rect)
        self.hover_animation.setEndValue(normal_rect)
        self.hover_animation.start()
        
        super().leaveEvent(event)
        
    def set_selected(self, selected):
        """Establece el estado de selección"""
        self.is_selected = selected
        self.search_result.selected = selected
        self.update_style()


class GalleryScrollArea(QScrollArea):
    """Área de scroll personalizada para la galería"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        # Widget contenedor
        self.gallery_widget = QWidget()
        self.gallery_layout = QGridLayout(self.gallery_widget)
        self.gallery_layout.setSpacing(10)
        self.gallery_layout.setContentsMargins(10, 10, 10, 10)
        
        self.setWidget(self.gallery_widget)
        
        # Lista de thumbnails
        self.thumbnails = []
        self.columns = 4  # Número de columnas por defecto
        
    def add_thumbnail(self, thumbnail_widget):
        """Añade un thumbnail a la galería"""
        self.thumbnails.append(thumbnail_widget)
        self.update_layout()
        
    def remove_thumbnail(self, sample_id):
        """Elimina un thumbnail de la galería"""
        for i, thumbnail in enumerate(self.thumbnails):
            if thumbnail.search_result.sample_id == sample_id:
                self.gallery_layout.removeWidget(thumbnail)
                thumbnail.deleteLater()
                del self.thumbnails[i]
                break
        self.update_layout()
        
    def clear_thumbnails(self):
        """Limpia todos los thumbnails"""
        for thumbnail in self.thumbnails:
            self.gallery_layout.removeWidget(thumbnail)
            thumbnail.deleteLater()
        self.thumbnails.clear()
        
    def update_layout(self):
        """Actualiza el layout de la galería"""
        # Limpiar layout
        for i in reversed(range(self.gallery_layout.count())):
            self.gallery_layout.itemAt(i).widget().setParent(None)
            
        # Reorganizar thumbnails
        for i, thumbnail in enumerate(self.thumbnails):
            row = i // self.columns
            col = i % self.columns
            self.gallery_layout.addWidget(thumbnail, row, col)
            
        # Añadir stretch al final
        self.gallery_layout.setRowStretch(len(self.thumbnails) // self.columns + 1, 1)
        
    def set_columns(self, columns):
        """Establece el número de columnas"""
        self.columns = max(1, columns)
        self.update_layout()
        
    def get_thumbnail_by_id(self, sample_id):
        """Obtiene un thumbnail por ID de muestra"""
        for thumbnail in self.thumbnails:
            if thumbnail.search_result.sample_id == sample_id:
                return thumbnail
        return None


class SearchFilterWidget(QWidget):
    """Widget para filtros de búsqueda"""
    
    filterChanged = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        
    def setup_ui(self):
        """Configura la interfaz de filtros"""
        layout = QHBoxLayout(self)
        
        # Filtro por confianza mínima
        conf_label = QLabel("Confianza mín:")
        self.confidence_slider = QSlider(Qt.Horizontal)
        self.confidence_slider.setRange(0, 100)
        self.confidence_slider.setValue(60)
        self.confidence_slider.setTickPosition(QSlider.TicksBelow)
        self.confidence_slider.setTickInterval(20)
        
        self.confidence_value = QLabel("60%")
        self.confidence_slider.valueChanged.connect(
            lambda v: self.confidence_value.setText(f"{v}%"))
        self.confidence_slider.valueChanged.connect(self.filterChanged.emit)
        
        # Filtro por tipo de coincidencia
        type_label = QLabel("Tipo:")
        self.match_type_combo = QComboBox()
        self.match_type_combo.addItems(["Todos", "Identificación", 
                                       "Inconclusivo", "Eliminación"])
        self.match_type_combo.currentTextChanged.connect(self.filterChanged.emit)
        
        # Filtro por score CMC
        score_label = QLabel("Score CMC mín:")
        self.score_slider = QSlider(Qt.Horizontal)
        self.score_slider.setRange(0, 100)
        self.score_slider.setValue(0)
        self.score_slider.setTickPosition(QSlider.TicksBelow)
        self.score_slider.setTickInterval(25)
        
        self.score_value = QLabel("0.00")
        self.score_slider.valueChanged.connect(
            lambda v: self.score_value.setText(f"{v/100:.2f}"))
        self.score_slider.valueChanged.connect(self.filterChanged.emit)
        
        # Búsqueda por texto
        search_label = QLabel("Buscar:")
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("ID de muestra...")
        self.search_edit.textChanged.connect(self.filterChanged.emit)
        
        # Añadir widgets al layout
        layout.addWidget(conf_label)
        layout.addWidget(self.confidence_slider)
        layout.addWidget(self.confidence_value)
        layout.addWidget(type_label)
        layout.addWidget(self.match_type_combo)
        layout.addWidget(score_label)
        layout.addWidget(self.score_slider)
        layout.addWidget(self.score_value)
        layout.addWidget(search_label)
        layout.addWidget(self.search_edit)
        
    def get_filters(self):
        """Retorna los filtros actuales"""
        return {
            'min_confidence': self.confidence_slider.value() / 100.0,
            'match_type': self.match_type_combo.currentText(),
            'min_score': self.score_slider.value() / 100.0,
            'search_text': self.search_edit.text().strip()
        }


class ComparisonPanel(QWidget):
    """Panel para mostrar comparación instantánea"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.query_image = None
        self.candidate_image = None
        self.setup_ui()
        
    def setup_ui(self):
        """Configura la interfaz del panel de comparación"""
        layout = QVBoxLayout(self)
        
        # Título
        title = QLabel("Comparación Instantánea")
        title.setFont(QFont("Arial", 12, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        
        # Layout de imágenes
        images_layout = QHBoxLayout()
        
        # Imagen de consulta
        query_group = QGroupBox("Muestra de Consulta")
        query_layout = QVBoxLayout(query_group)
        
        self.query_label = QLabel()
        self.query_label.setFixedSize(200, 200)
        self.query_label.setAlignment(Qt.AlignCenter)
        self.query_label.setFrameStyle(QFrame.Box)
        self.query_label.setText("Imagen de\nConsulta")
        
        query_layout.addWidget(self.query_label)
        
        # Imagen candidata
        candidate_group = QGroupBox("Candidato")
        candidate_layout = QVBoxLayout(candidate_group)
        
        self.candidate_label = QLabel()
        self.candidate_label.setFixedSize(200, 200)
        self.candidate_label.setAlignment(Qt.AlignCenter)
        self.candidate_label.setFrameStyle(QFrame.Box)
        self.candidate_label.setText("Seleccione un\ncandidato")
        
        self.candidate_info = QLabel("Información del candidato")
        self.candidate_info.setWordWrap(True)
        self.candidate_info.setMaximumHeight(60)
        
        candidate_layout.addWidget(self.candidate_label)
        candidate_layout.addWidget(self.candidate_info)
        
        images_layout.addWidget(query_group)
        images_layout.addWidget(candidate_group)
        
        # Botones de acción
        buttons_layout = QHBoxLayout()
        
        self.detailed_btn = QPushButton("Análisis Detallado")
        self.detailed_btn.setIcon(QIcon("🔍"))
        
        self.export_btn = QPushButton("Exportar Comparación")
        self.export_btn.setIcon(QIcon("💾"))
        
        buttons_layout.addWidget(self.detailed_btn)
        buttons_layout.addWidget(self.export_btn)
        buttons_layout.addStretch()
        
        layout.addWidget(title)
        layout.addLayout(images_layout)
        layout.addLayout(buttons_layout)
        
    def set_query_image(self, image_path):
        """Establece la imagen de consulta"""
        self.query_image = image_path
        if os.path.exists(image_path):
            pixmap = QPixmap(image_path)
            scaled_pixmap = pixmap.scaled(190, 190, Qt.KeepAspectRatio, 
                                        Qt.SmoothTransformation)
            self.query_label.setPixmap(scaled_pixmap)
        else:
            self.query_label.setText("Imagen de\nConsulta\nno disponible")
            
    def set_candidate_result(self, search_result):
        """Establece el resultado candidato"""
        if search_result and os.path.exists(search_result.image_path):
            pixmap = QPixmap(search_result.image_path)
            scaled_pixmap = pixmap.scaled(190, 190, Qt.KeepAspectRatio, 
                                        Qt.SmoothTransformation)
            self.candidate_label.setPixmap(scaled_pixmap)
            
            # Actualizar información
            info_text = f"ID: {search_result.sample_id}\n"
            info_text += f"Score CMC: {search_result.cmc_score:.4f}\n"
            info_text += f"Confianza: {search_result.confidence:.2%}\n"
            info_text += f"Tipo: {search_result.match_type.title()}"
            
            self.candidate_info.setText(info_text)
        else:
            self.candidate_label.setText("Imagen\nno disponible")
            self.candidate_info.setText("Sin información disponible")


class GallerySearchWidget(QWidget):
    """Widget principal de galería para resultados de búsqueda"""
    
    resultSelected = pyqtSignal(str)  # sample_id
    comparisonRequested = pyqtSignal(str, str)  # query_id, candidate_id
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.search_results = []
        self.filtered_results = []
        self.selected_result = None
        
        self.setup_ui()
        self.connect_signals()
        
        # Generar datos de ejemplo
        self.generate_sample_results()
        
    def setup_ui(self):
        """Configura la interfaz principal"""
        layout = QVBoxLayout(self)
        
        # Título y controles superiores
        header_layout = QHBoxLayout()
        
        title = QLabel("Galería de Resultados de Búsqueda")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        
        # Controles de vista
        view_controls = QHBoxLayout()
        
        columns_label = QLabel("Columnas:")
        self.columns_spin = QSpinBox()
        self.columns_spin.setRange(2, 8)
        self.columns_spin.setValue(4)
        
        self.sort_combo = QComboBox()
        self.sort_combo.addItems(["Score CMC ↓", "Score CMC ↑", 
                                 "Confianza ↓", "Confianza ↑", 
                                 "ID ↑", "ID ↓"])
        
        view_controls.addWidget(columns_label)
        view_controls.addWidget(self.columns_spin)
        view_controls.addWidget(QLabel("Ordenar:"))
        view_controls.addWidget(self.sort_combo)
        view_controls.addStretch()
        
        header_layout.addWidget(title)
        header_layout.addStretch()
        header_layout.addLayout(view_controls)
        
        # Filtros
        self.filter_widget = SearchFilterWidget()
        
        # Splitter principal
        main_splitter = QSplitter(Qt.Horizontal)
        
        # Panel izquierdo - Galería
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Información de resultados
        self.results_info = QLabel("0 resultados encontrados")
        self.results_info.setFont(QFont("Arial", 10))
        
        # Galería
        self.gallery_scroll = GalleryScrollArea()
        
        left_layout.addWidget(self.results_info)
        left_layout.addWidget(self.gallery_scroll)
        
        # Panel derecho - Comparación
        self.comparison_panel = ComparisonPanel()
        
        # Configurar splitter
        main_splitter.addWidget(left_panel)
        main_splitter.addWidget(self.comparison_panel)
        main_splitter.setSizes([600, 400])
        
        layout.addLayout(header_layout)
        layout.addWidget(self.filter_widget)
        layout.addWidget(main_splitter)
        
    def connect_signals(self):
        """Conecta las señales de los widgets"""
        self.columns_spin.valueChanged.connect(self.gallery_scroll.set_columns)
        self.sort_combo.currentTextChanged.connect(self.sort_results)
        self.filter_widget.filterChanged.connect(self.apply_filters)
        
    def generate_sample_results(self):
        """Genera resultados de ejemplo"""
        sample_results = []
        
        # Generar 20 resultados de ejemplo
        for i in range(20):
            sample_id = f"SAMPLE_{i+1:03d}"
            image_path = f"samples/sample_{i+1}.jpg"  # Ruta de ejemplo
            cmc_score = np.random.beta(2, 1)  # Sesgo hacia scores altos
            confidence = np.random.uniform(0.3, 0.95)
            
            # Determinar tipo de coincidencia basado en confianza
            if confidence >= 0.8:
                match_type = "identification"
            elif confidence >= 0.5:
                match_type = "inconclusive"
            else:
                match_type = "elimination"
                
            metadata = {
                'caliber': np.random.choice(['9mm', '.40', '.45', '.38']),
                'manufacturer': np.random.choice(['Glock', 'Smith & Wesson', 'Sig Sauer']),
                'date_analyzed': datetime.now().strftime('%Y-%m-%d'),
                'analyst': f'Analyst_{np.random.randint(1, 5)}'
            }
            
            result = SearchResult(sample_id, image_path, cmc_score, 
                                confidence, match_type, metadata)
            sample_results.append(result)
            
        self.set_search_results(sample_results)
        
    def set_search_results(self, results):
        """Establece los resultados de búsqueda"""
        self.search_results = results
        self.apply_filters()
        
    def apply_filters(self):
        """Aplica los filtros actuales"""
        filters = self.filter_widget.get_filters()
        
        self.filtered_results = []
        for result in self.search_results:
            # Filtro por confianza
            if result.confidence < filters['min_confidence']:
                continue
                
            # Filtro por tipo
            if filters['match_type'] != "Todos":
                type_map = {
                    "Identificación": "identification",
                    "Inconclusivo": "inconclusive", 
                    "Eliminación": "elimination"
                }
                if result.match_type != type_map.get(filters['match_type']):
                    continue
                    
            # Filtro por score
            if result.cmc_score < filters['min_score']:
                continue
                
            # Filtro por texto
            if filters['search_text']:
                if filters['search_text'].lower() not in result.sample_id.lower():
                    continue
                    
            self.filtered_results.append(result)
            
        self.sort_results()
        
    def sort_results(self):
        """Ordena los resultados según el criterio seleccionado"""
        sort_option = self.sort_combo.currentText()
        
        if "Score CMC" in sort_option:
            reverse = "↓" in sort_option
            self.filtered_results.sort(key=lambda x: x.cmc_score, reverse=reverse)
        elif "Confianza" in sort_option:
            reverse = "↓" in sort_option
            self.filtered_results.sort(key=lambda x: x.confidence, reverse=reverse)
        elif "ID" in sort_option:
            reverse = "↓" in sort_option
            self.filtered_results.sort(key=lambda x: x.sample_id, reverse=reverse)
            
        self.update_gallery()
        
    def update_gallery(self):
        """Actualiza la galería con los resultados filtrados"""
        # Limpiar galería actual
        self.gallery_scroll.clear_thumbnails()
        
        # Añadir nuevos thumbnails
        for result in self.filtered_results:
            thumbnail = ThumbnailWidget(result)
            thumbnail.clicked.connect(self.on_thumbnail_clicked)
            thumbnail.hovered.connect(self.on_thumbnail_hovered)
            thumbnail.doubleClicked.connect(self.on_thumbnail_double_clicked)
            thumbnail.contextMenuRequested.connect(self.on_context_menu)
            
            self.gallery_scroll.add_thumbnail(thumbnail)
            
        # Actualizar información
        count = len(self.filtered_results)
        total = len(self.search_results)
        self.results_info.setText(f"{count} de {total} resultados mostrados")
        
    def on_thumbnail_clicked(self, sample_id):
        """Maneja clic en thumbnail"""
        # Deseleccionar anterior
        if self.selected_result:
            prev_thumbnail = self.gallery_scroll.get_thumbnail_by_id(
                self.selected_result.sample_id)
            if prev_thumbnail:
                prev_thumbnail.set_selected(False)
                
        # Seleccionar nuevo
        thumbnail = self.gallery_scroll.get_thumbnail_by_id(sample_id)
        if thumbnail:
            thumbnail.set_selected(True)
            self.selected_result = thumbnail.search_result
            
        self.resultSelected.emit(sample_id)
        
    def on_thumbnail_hovered(self, sample_id):
        """Maneja hover en thumbnail"""
        # Encontrar resultado
        result = None
        for r in self.filtered_results:
            if r.sample_id == sample_id:
                result = r
                break
                
        if result:
            # Actualizar panel de comparación
            self.comparison_panel.set_candidate_result(result)
            
    def on_thumbnail_double_clicked(self, sample_id):
        """Maneja doble clic en thumbnail"""
        # Abrir análisis detallado
        self.comparisonRequested.emit("QUERY_SAMPLE", sample_id)
        
    def on_context_menu(self, sample_id, position):
        """Maneja menú contextual"""
        menu = QMenu(self)
        
        # Acciones del menú
        view_action = QAction("Ver Detalles", self)
        view_action.triggered.connect(
            lambda: self.show_result_details(sample_id))
        
        compare_action = QAction("Comparar", self)
        compare_action.triggered.connect(
            lambda: self.comparisonRequested.emit("QUERY_SAMPLE", sample_id))
        
        export_action = QAction("Exportar", self)
        export_action.triggered.connect(
            lambda: self.export_result(sample_id))
        
        menu.addAction(view_action)
        menu.addAction(compare_action)
        menu.addSeparator()
        menu.addAction(export_action)
        
        menu.exec_(position)
        
    def show_result_details(self, sample_id):
        """Muestra detalles del resultado"""
        # Implementar ventana de detalles
        print(f"Mostrando detalles para {sample_id}")
        
    def export_result(self, sample_id):
        """Exporta el resultado"""
        # Implementar exportación
        print(f"Exportando resultado {sample_id}")
        
    def set_query_image(self, image_path):
        """Establece la imagen de consulta para comparación"""
        self.comparison_panel.set_query_image(image_path)