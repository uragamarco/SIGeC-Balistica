"""
Panel Dinámico de Resultados
Presenta los resultados de análisis balístico con tarjetas de resultado
color-coded y conclusiones AFTE prominentes.
"""

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QFrame, QScrollArea, QProgressBar, QPushButton,
                             QGroupBox, QGridLayout, QTextEdit, QSplitter)
from PyQt5.QtCore import Qt, pyqtSignal, QPropertyAnimation, QEasingCurve, QRect
from PyQt5.QtGui import (QPixmap, QPainter, QPen, QColor, QFont, QBrush, 
                         QLinearGradient, QIcon, QPalette)
import numpy as np


class ResultCard(QFrame):
    """Tarjeta de resultado individual con indicadores visuales"""
    
    cardClicked = pyqtSignal(str)  # feature_name
    
    def __init__(self, feature_name, score, confidence_level, details="", parent=None):
        super().__init__(parent)
        self.feature_name = feature_name
        self.score = score
        self.confidence_level = confidence_level
        self.details = details
        self.is_expanded = False
        
        self.setup_ui()
        self.setup_style()
        
    def setup_ui(self):
        """Configura la interfaz de la tarjeta"""
        self.setFixedHeight(120)
        self.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        self.setLineWidth(2)
        self.setCursor(Qt.PointingHandCursor)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 10, 15, 10)
        
        # Header con nombre y score
        header_layout = QHBoxLayout()
        
        # Nombre de la característica
        self.name_label = QLabel(self.feature_name)
        self.name_label.setFont(QFont("Arial", 12, QFont.Bold))
        
        # Score
        self.score_label = QLabel(f"{self.score:.3f}")
        self.score_label.setFont(QFont("Arial", 14, QFont.Bold))
        self.score_label.setAlignment(Qt.AlignRight)
        
        header_layout.addWidget(self.name_label)
        header_layout.addStretch()
        header_layout.addWidget(self.score_label)
        
        # Barra de progreso visual
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1000)
        self.progress_bar.setValue(int(self.score * 1000))
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(8)
        
        # Nivel de confianza
        self.confidence_label = QLabel(f"Confianza: {self.confidence_level}")
        self.confidence_label.setFont(QFont("Arial", 9))
        
        # Detalles (inicialmente ocultos)
        self.details_label = QLabel(self.details)
        self.details_label.setWordWrap(True)
        self.details_label.setFont(QFont("Arial", 8))
        self.details_label.hide()
        
        layout.addLayout(header_layout)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.confidence_label)
        layout.addWidget(self.details_label)
        layout.addStretch()
        
    def setup_style(self):
        """Configura el estilo basado en el nivel de confianza"""
        if self.confidence_level == "Alta":
            bg_color = "#e8f5e8"
            border_color = "#4caf50"
            progress_color = "#4caf50"
        elif self.confidence_level == "Media":
            bg_color = "#fff8e1"
            border_color = "#ff9800"
            progress_color = "#ff9800"
        else:  # Baja
            bg_color = "#ffebee"
            border_color = "#f44336"
            progress_color = "#f44336"
            
        self.setStyleSheet(f"""
            ResultCard {{
                background-color: {bg_color};
                border: 2px solid {border_color};
                border-radius: 8px;
            }}
            ResultCard:hover {{
                background-color: {bg_color.replace('e8', 'f0')};
                border: 3px solid {border_color};
            }}
        """)
        
        # Estilo de la barra de progreso
        self.progress_bar.setStyleSheet(f"""
            QProgressBar {{
                border: 1px solid #ccc;
                border-radius: 4px;
                background-color: #f0f0f0;
            }}
            QProgressBar::chunk {{
                background-color: {progress_color};
                border-radius: 3px;
            }}
        """)
        
    def mousePressEvent(self, event):
        """Maneja el clic en la tarjeta"""
        if event.button() == Qt.LeftButton:
            self.toggle_expansion()
            self.cardClicked.emit(self.feature_name)
            
    def toggle_expansion(self):
        """Alterna la expansión de la tarjeta"""
        if self.is_expanded:
            self.setFixedHeight(120)
            self.details_label.hide()
        else:
            self.setFixedHeight(180)
            self.details_label.show()
            
        self.is_expanded = not self.is_expanded


class AFTEConclusionWidget(QFrame):
    """Widget prominente para mostrar la conclusión AFTE"""
    
    def __init__(self, conclusion="Inconclusivo", confidence=0.5, parent=None):
        super().__init__(parent)
        self.conclusion = conclusion
        self.confidence = confidence
        self.setup_ui()
        self.setup_style()
        
    def setup_ui(self):
        """Configura la interfaz del widget de conclusión"""
        self.setFixedHeight(150)
        self.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        self.setLineWidth(3)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 15, 20, 15)
        
        # Título
        title_label = QLabel("CONCLUSIÓN AFTE")
        title_label.setFont(QFont("Arial", 14, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        
        # Conclusión principal
        self.conclusion_label = QLabel(self.conclusion.upper())
        self.conclusion_label.setFont(QFont("Arial", 24, QFont.Bold))
        self.conclusion_label.setAlignment(Qt.AlignCenter)
        
        # Nivel de confianza
        confidence_layout = QHBoxLayout()
        confidence_layout.addWidget(QLabel("Nivel de Confianza:"))
        
        self.confidence_bar = QProgressBar()
        self.confidence_bar.setRange(0, 100)
        self.confidence_bar.setValue(int(self.confidence * 100))
        self.confidence_bar.setFixedHeight(20)
        
        confidence_percentage = QLabel(f"{int(self.confidence * 100)}%")
        confidence_percentage.setFont(QFont("Arial", 12, QFont.Bold))
        
        confidence_layout.addWidget(self.confidence_bar)
        confidence_layout.addWidget(confidence_percentage)
        
        layout.addWidget(title_label)
        layout.addWidget(self.conclusion_label)
        layout.addLayout(confidence_layout)
        
    def setup_style(self):
        """Configura el estilo basado en la conclusión"""
        if self.conclusion.lower() == "identificación":
            bg_color = "#e8f5e8"
            border_color = "#4caf50"
            text_color = "#2e7d32"
            confidence_color = "#4caf50"
        elif self.conclusion.lower() == "eliminación":
            bg_color = "#ffebee"
            border_color = "#f44336"
            text_color = "#c62828"
            confidence_color = "#f44336"
        else:  # Inconclusivo
            bg_color = "#fff8e1"
            border_color = "#ff9800"
            text_color = "#ef6c00"
            confidence_color = "#ff9800"
            
        self.setStyleSheet(f"""
            AFTEConclusionWidget {{
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 {bg_color}, stop: 1 {bg_color.replace('e8', 'd0')});
                border: 3px solid {border_color};
                border-radius: 12px;
            }}
        """)
        
        self.conclusion_label.setStyleSheet(f"color: {text_color};")
        
        self.confidence_bar.setStyleSheet(f"""
            QProgressBar {{
                border: 2px solid #ccc;
                border-radius: 10px;
                background-color: #f0f0f0;
                text-align: center;
            }}
            QProgressBar::chunk {{
                background-color: {confidence_color};
                border-radius: 8px;
            }}
        """)
        
    def update_conclusion(self, conclusion, confidence):
        """Actualiza la conclusión y confianza"""
        self.conclusion = conclusion
        self.confidence = confidence
        
        self.conclusion_label.setText(conclusion.upper())
        self.confidence_bar.setValue(int(confidence * 100))
        self.setup_style()


class StatisticsWidget(QWidget):
    """Widget para mostrar estadísticas del análisis"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        
    def setup_ui(self):
        """Configura la interfaz de estadísticas"""
        layout = QVBoxLayout(self)
        
        # Título
        title = QLabel("Estadísticas del Análisis")
        title.setFont(QFont("Arial", 12, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        
        # Grid de estadísticas
        stats_grid = QGridLayout()
        
        self.stats_labels = {}
        stats_items = [
            ("Características Analizadas", "0"),
            ("Coincidencias Encontradas", "0"),
            ("Score Promedio", "0.000"),
            ("Tiempo de Análisis", "0s"),
            ("Algoritmo Utilizado", "N/A"),
            ("Calidad de Imagen", "N/A")
        ]
        
        for i, (label, value) in enumerate(stats_items):
            label_widget = QLabel(f"{label}:")
            label_widget.setFont(QFont("Arial", 9))
            
            value_widget = QLabel(value)
            value_widget.setFont(QFont("Arial", 9, QFont.Bold))
            value_widget.setStyleSheet("color: #2196f3;")
            
            stats_grid.addWidget(label_widget, i, 0)
            stats_grid.addWidget(value_widget, i, 1)
            
            self.stats_labels[label] = value_widget
            
        layout.addWidget(title)
        layout.addLayout(stats_grid)
        
    def update_statistics(self, stats_dict):
        """Actualiza las estadísticas mostradas"""
        for label, value in stats_dict.items():
            if label in self.stats_labels:
                self.stats_labels[label].setText(str(value))


class DynamicResultsPanel(QWidget):
    """Panel principal dinámico de resultados"""
    
    featureSelected = pyqtSignal(str)
    exportRequested = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.result_cards = []
        self.setup_ui()
        
    def setup_ui(self):
        """Configura la interfaz principal"""
        layout = QVBoxLayout(self)
        
        # Splitter principal
        main_splitter = QSplitter(Qt.Vertical)
        
        # Panel superior - Conclusión AFTE
        top_panel = QWidget()
        top_layout = QVBoxLayout(top_panel)
        
        self.afte_conclusion = AFTEConclusionWidget()
        top_layout.addWidget(self.afte_conclusion)
        
        # Panel medio - Tarjetas de resultados
        middle_panel = QWidget()
        middle_layout = QVBoxLayout(middle_panel)
        
        # Título de resultados
        results_title = QLabel("Resultados por Característica")
        results_title.setFont(QFont("Arial", 14, QFont.Bold))
        results_title.setAlignment(Qt.AlignCenter)
        
        # Área scrollable para tarjetas
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        self.cards_widget = QWidget()
        self.cards_layout = QVBoxLayout(self.cards_widget)
        self.cards_layout.setSpacing(10)
        
        self.scroll_area.setWidget(self.cards_widget)
        
        middle_layout.addWidget(results_title)
        middle_layout.addWidget(self.scroll_area)
        
        # Panel inferior - Estadísticas y controles
        bottom_panel = QWidget()
        bottom_layout = QHBoxLayout(bottom_panel)
        
        # Estadísticas
        self.statistics_widget = StatisticsWidget()
        
        # Controles de acción
        actions_group = QGroupBox("Acciones")
        actions_layout = QVBoxLayout(actions_group)
        
        self.export_btn = QPushButton("Exportar Resultados")
        self.export_btn.setIcon(QIcon("📊"))
        self.export_btn.clicked.connect(self.exportRequested.emit)
        
        self.save_report_btn = QPushButton("Generar Reporte")
        self.save_report_btn.setIcon(QIcon("📄"))
        
        self.compare_btn = QPushButton("Comparar con Otros")
        self.compare_btn.setIcon(QIcon("🔍"))
        
        actions_layout.addWidget(self.export_btn)
        actions_layout.addWidget(self.save_report_btn)
        actions_layout.addWidget(self.compare_btn)
        actions_layout.addStretch()
        
        bottom_layout.addWidget(self.statistics_widget, 2)
        bottom_layout.addWidget(actions_group, 1)
        
        # Agregar paneles al splitter
        main_splitter.addWidget(top_panel)
        main_splitter.addWidget(middle_panel)
        main_splitter.addWidget(bottom_panel)
        
        # Configurar proporciones del splitter
        main_splitter.setSizes([200, 400, 200])
        
        layout.addWidget(main_splitter)
        
    def add_result_card(self, feature_name, score, confidence_level, details=""):
        """Añade una nueva tarjeta de resultado"""
        card = ResultCard(feature_name, score, confidence_level, details)
        card.cardClicked.connect(self.featureSelected.emit)
        
        self.result_cards.append(card)
        self.cards_layout.addWidget(card)
        
    def clear_results(self):
        """Limpia todos los resultados"""
        for card in self.result_cards:
            card.deleteLater()
        self.result_cards.clear()
        
    def update_afte_conclusion(self, conclusion, confidence):
        """Actualiza la conclusión AFTE"""
        self.afte_conclusion.update_conclusion(conclusion, confidence)
        
    def update_statistics(self, stats_dict):
        """Actualiza las estadísticas"""
        self.statistics_widget.update_statistics(stats_dict)
        
    def set_sample_results(self):
        """Establece resultados de ejemplo para demostración"""
        # Limpiar resultados anteriores
        self.clear_results()
        
        # Añadir tarjetas de ejemplo
        sample_results = [
            ("Marca del Percutor", 0.892, "Alta", "Coincidencia significativa en forma y dimensiones del percutor"),
            ("Cara de Recámara", 0.756, "Media", "Patrones de estrías parcialmente coincidentes"),
            ("Marca del Eyector", 0.634, "Media", "Similitudes en posición y morfología"),
            ("Estrías del Cañón", 0.423, "Baja", "Coincidencias limitadas en patrones de estrías"),
            ("Impresión de Cápsula", 0.789, "Alta", "Patrones de impresión altamente compatibles")
        ]
        
        for feature, score, confidence, details in sample_results:
            self.add_result_card(feature, score, confidence, details)
            
        # Actualizar conclusión AFTE
        overall_score = np.mean([r[1] for r in sample_results])
        if overall_score >= 0.8:
            conclusion = "Identificación"
        elif overall_score >= 0.6:
            conclusion = "Inconclusivo"
        else:
            conclusion = "Eliminación"
            
        self.update_afte_conclusion(conclusion, overall_score)
        
        # Actualizar estadísticas
        stats = {
            "Características Analizadas": len(sample_results),
            "Coincidencias Encontradas": sum(1 for r in sample_results if r[1] > 0.6),
            "Score Promedio": f"{overall_score:.3f}",
            "Tiempo de Análisis": "2.3s",
            "Algoritmo Utilizado": "CNN + LBP",
            "Calidad de Imagen": "Excelente"
        }
        self.update_statistics(stats)