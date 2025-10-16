"""
Estilos unificados para SIGeC Balística GUI
Combina lo mejor de ambas implementaciones con mejoras para paneles flotantes y estándares NIST
"""

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QPalette, QColor
import re
from pathlib import Path

class SIGeCBallisticaTheme:
    """Tema unificado para SIGeC Balística con soporte completo para análisis forense"""
    
    # Colores primarios y secundarios
    PRIMARY = "#1976D2"          # Azul profesional
    PRIMARY_DARK = "#1565C0"     # Azul oscuro
    PRIMARY_LIGHT = "#42A5F5"    # Azul claro
    PRIMARY_VARIANT = "#1E88E5"  # Variante azul
    
    SECONDARY = "#388E3C"        # Verde éxito
    SECONDARY_DARK = "#2E7D32"   # Verde oscuro
    SECONDARY_LIGHT = "#66BB6A"  # Verde claro
    SECONDARY_VARIANT = "#43A047" # Variante verde
    
    # Colores de estado
    SUCCESS = "#4CAF50"          # Verde éxito
    WARNING = "#FF9800"          # Naranja advertencia
    ERROR = "#F44336"            # Rojo error
    INFO = "#2196F3"             # Azul información
    
    WARNING_DARK = "#F57C00"     # Naranja oscuro
    WARNING_LIGHT = "#FFB74D"    # Naranja claro
    ERROR_DARK = "#D32F2F"       # Rojo oscuro
    ERROR_LIGHT = "#EF5350"      # Rojo claro
    
    # Colores de superficie
    SURFACE = "#FFFFFF"          # Blanco
    SURFACE_VARIANT = "#F5F5F5"  # Gris muy claro
    SURFACE_CONTAINER = "#F8F9FA" # Contenedor
    BACKGROUND = "#FAFAFA"       # Gris de fondo
    BACKGROUND_VARIANT = "#F0F0F0" # Variante de fondo
    
    # Colores de texto
    TEXT_PRIMARY = "#212121"     # Negro principal
    TEXT_SECONDARY = "#757575"   # Gris secundario
    TEXT_DISABLED = "#BDBDBD"    # Gris deshabilitado
    TEXT_ON_PRIMARY = "#FFFFFF"  # Blanco sobre primario
    TEXT_ON_SURFACE = "#1C1B1F"  # Texto sobre superficie
    
    # Colores específicos para análisis balístico
    BALLISTIC_PRIMARY = "#2E3440"    # Azul oscuro profesional
    BALLISTIC_SECONDARY = "#3B4252"  # Gris azulado
    BALLISTIC_ACCENT = "#5E81AC"     # Azul medio
    BALLISTIC_HIGHLIGHT = "#88C0D0"  # Azul claro destacado
    
    # Colores NIST
    NIST_PRIMARY = "#003366"     # Azul institucional
    NIST_SECONDARY = "#0066CC"   # Azul NIST
    NIST_ACCENT = "#4A90E2"      # Azul claro NIST
    NIST_SUCCESS = "#28A745"     # Verde cumplimiento
    NIST_WARNING = "#FFC107"     # Amarillo advertencia
    NIST_ERROR = "#DC3545"       # Rojo no cumplimiento
    
    # Colores Deep Learning
    DL_PRIMARY = "#673AB7"       # Morado Deep Learning
    DL_PRIMARY_DARK = "#512DA8"  # Morado oscuro
    DL_PRIMARY_LIGHT = "#9575CD" # Morado claro
    DL_ACCENT = "#E91E63"        # Rosa accent para DL
    DL_ACCENT_DARK = "#C2185B"   # Rosa oscuro
    DL_ACCENT_LIGHT = "#F06292"  # Rosa claro
    
    # Colores de calidad de imagen
    QUALITY_EXCELLENT = "#4CAF50" # Verde excelente
    QUALITY_GOOD = "#8BC34A"      # Verde claro bueno
    QUALITY_FAIR = "#FF9800"      # Naranja regular
    QUALITY_POOR = "#F44336"      # Rojo pobre
    QUALITY_UNKNOWN = "#9E9E9E"   # Gris desconocido
    
    # Colores de contorno y sombra
    DIVIDER = "#E0E0E0"         # Gris divisor
    OUTLINE = "#9E9E9E"         # Gris contorno
    OUTLINE_VARIANT = "#CAC4D0" # Variante contorno
    SHADOW = "#00000020"        # Sombra sutil
    
    # Colores para paneles flotantes
    DOCK_BACKGROUND = "#F8F9FA"  # Fondo de dock
    DOCK_BORDER = "#DEE2E6"      # Borde de dock
    DOCK_TITLE = "#495057"       # Título de dock
    DOCK_ACTIVE = "#007BFF"      # Dock activo
    DOCK_HOVER = "#E9ECEF"       # Hover en dock

    @classmethod
    def get_stylesheet(cls) -> str:
        """Retorna la hoja de estilos QSS completa"""
        return f"""
        /* === ESTILOS BASE DE LA APLICACIÓN === */
        QMainWindow {{
            background-color: {cls.BACKGROUND};
            color: {cls.TEXT_PRIMARY};
            font-family: 'Segoe UI', 'Arial', sans-serif;
        }}
        
        /* === TABS PRINCIPALES === */
        QTabWidget::pane {{
            border: 1px solid {cls.DIVIDER};
            background-color: {cls.SURFACE};
            border-radius: 8px;
            margin-top: -1px;
        }}
        
        QTabBar::tab {{
            background-color: {cls.SURFACE_VARIANT};
            color: {cls.TEXT_SECONDARY};
            padding: 12px 24px;
            margin-right: 2px;
            border-top-left-radius: 8px;
            border-top-right-radius: 8px;
            border: 1px solid {cls.DIVIDER};
            border-bottom: none;
            font-weight: 500;
            min-width: 120px;
            min-height: 20px;
        }}
        
        QTabBar::tab:selected {{
            background-color: {cls.PRIMARY};
            color: {cls.TEXT_ON_PRIMARY};
            font-weight: 600;
            border-color: {cls.PRIMARY};
            margin-bottom: -1px;
        }}
        
        QTabBar::tab:hover:!selected {{
            background-color: {cls.PRIMARY_LIGHT};
            color: {cls.TEXT_ON_PRIMARY};
            border-color: {cls.PRIMARY_LIGHT};
        }}
        
        /* === QDOCKWIDGET PARA PANELES FLOTANTES === */
        QDockWidget {{
            background-color: {cls.DOCK_BACKGROUND};
            border: 1px solid {cls.DOCK_BORDER};
            border-radius: 8px;
        }}
        
        QDockWidget::title {{
            background-color: {cls.DOCK_BACKGROUND};
            color: {cls.DOCK_TITLE};
            padding: 8px 12px;
            border-bottom: 1px solid {cls.DOCK_BORDER};
            font-weight: 600;
            font-size: 13px;
        }}
        
        QDockWidget::close-button, QDockWidget::float-button {{
            background-color: transparent;
            border: none;
            padding: 4px;
            margin: 2px;
            border-radius: 4px;
        }}
        
        QDockWidget::close-button:hover, QDockWidget::float-button:hover {{
            background-color: {cls.DOCK_HOVER};
        }}
        
        QDockWidget::close-button:pressed, QDockWidget::float-button:pressed {{
            background-color: {cls.DOCK_ACTIVE};
        }}
        
        /* === BOTONES PRINCIPALES === */
        QPushButton {{
            background-color: {cls.PRIMARY};
            color: {cls.TEXT_ON_PRIMARY};
            border: none;
            padding: 12px 24px;
            border-radius: 6px;
            font-weight: 500;
            font-size: 14px;
            min-height: 20px;
        }}
        
        QPushButton:hover {{
            background-color: {cls.PRIMARY_DARK};
        }}
        
        QPushButton:pressed {{
            background-color: {cls.PRIMARY_DARK};
            padding: 13px 23px 11px 25px;
        }}
        
        QPushButton:disabled {{
            background-color: {cls.TEXT_DISABLED};
            color: {cls.SURFACE};
        }}
        
        /* === BOTONES ESPECIALIZADOS === */
        QPushButton[class="secondary"] {{
            background-color: {cls.SECONDARY};
        }}
        
        QPushButton[class="secondary"]:hover {{
            background-color: {cls.SECONDARY_DARK};
        }}
        
        QPushButton[class="success"] {{
            background-color: {cls.SUCCESS};
        }}
        
        QPushButton[class="success"]:hover {{
            background-color: {cls.SECONDARY_DARK};
        }}
        
        QPushButton[class="warning"] {{
            background-color: {cls.WARNING};
        }}
        
        QPushButton[class="warning"]:hover {{
            background-color: {cls.WARNING_DARK};
        }}
        
        QPushButton[class="error"] {{
            background-color: {cls.ERROR};
        }}
        
        QPushButton[class="error"]:hover {{
            background-color: {cls.ERROR_DARK};
        }}
        
        /* === BOTONES BALÍSTICOS === */
        QPushButton[class="ballistic"] {{
            background-color: {cls.BALLISTIC_PRIMARY};
            color: {cls.TEXT_ON_PRIMARY};
            border: 1px solid {cls.BALLISTIC_ACCENT};
        }}
        
        QPushButton[class="ballistic"]:hover {{
            background-color: {cls.BALLISTIC_SECONDARY};
            border-color: {cls.BALLISTIC_HIGHLIGHT};
        }}
        
        /* === BOTONES NIST === */
        QPushButton[class="nist"] {{
            background-color: {cls.NIST_PRIMARY};
            color: {cls.TEXT_ON_PRIMARY};
            border: 1px solid {cls.NIST_ACCENT};
        }}
        
        QPushButton[class="nist"]:hover {{
            background-color: {cls.NIST_SECONDARY};
        }}
        
        QPushButton[class="nist-success"] {{
            background-color: {cls.NIST_SUCCESS};
        }}
        
        QPushButton[class="nist-warning"] {{
            background-color: {cls.NIST_WARNING};
            color: {cls.TEXT_PRIMARY};
        }}
        
        QPushButton[class="nist-error"] {{
            background-color: {cls.NIST_ERROR};
        }}
        
        /* === CAMPOS DE ENTRADA === */
        QLineEdit, QTextEdit, QPlainTextEdit {{
            border: 2px solid {cls.OUTLINE};
            border-radius: 6px;
            padding: 8px 12px;
            background-color: {cls.SURFACE};
            color: {cls.TEXT_PRIMARY};
            font-size: 14px;
        }}
        
        QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus {{
            border-color: {cls.PRIMARY};
        }}
        
        QLineEdit:disabled, QTextEdit:disabled, QPlainTextEdit:disabled {{
            background-color: {cls.SURFACE_VARIANT};
            color: {cls.TEXT_DISABLED};
        }}
        
        /* === COMBOBOX === */
        QComboBox {{
            border: 2px solid {cls.OUTLINE};
            border-radius: 6px;
            padding: 8px 12px;
            background-color: {cls.SURFACE};
            color: {cls.TEXT_PRIMARY};
            font-size: 14px;
            min-width: 120px;
        }}
        
        QComboBox:focus {{
            border-color: {cls.PRIMARY};
        }}
        
        QComboBox::drop-down {{
            border: none;
            width: 30px;
        }}
        
        QComboBox::down-arrow {{
            width: 16px;
            height: 16px;
        }}
        
        QComboBox QAbstractItemView {{
            border: 1px solid {cls.DIVIDER};
            background-color: {cls.SURFACE};
            selection-background-color: {cls.PRIMARY_LIGHT};
        }}
        
        /* === TARJETAS === */
        QFrame[class="card"] {{
            background-color: {cls.SURFACE};
            border: 1px solid {cls.DIVIDER};
            border-radius: 12px;
            padding: 16px;
        }}
        
        QFrame[class="card-elevated"] {{
            background-color: {cls.SURFACE};
            border: none;
            border-radius: 12px;
            padding: 16px;
        }}
        
        /* === PANELES COLAPSABLES === */
        QFrame[class="collapsible"] {{
            background-color: {cls.SURFACE_VARIANT};
            border: 1px solid {cls.DIVIDER};
            border-radius: 8px;
            margin: 4px 0px;
        }}
        
        QFrame[class="collapsible-header"] {{
            background-color: {cls.SURFACE_VARIANT};
            border: none;
            border-radius: 8px 8px 0px 0px;
            padding: 12px 16px;
        }}
        
        QFrame[class="collapsible-content"] {{
            background-color: {cls.SURFACE};
            border: none;
            border-radius: 0px 0px 8px 8px;
            padding: 16px;
        }}
        
        /* === ZONAS DE ARRASTRE === */
        QFrame[class="drop-zone"] {{
            border: 3px dashed {cls.OUTLINE};
            border-radius: 12px;
            background-color: {cls.SURFACE_VARIANT};
            padding: 32px;
            margin: 8px;
        }}
        
        QFrame[class="drop-zone-active"] {{
            border-color: {cls.PRIMARY};
            background-color: rgba(66, 165, 245, 0.1);
        }}
        
        QFrame[class="drop-zone-error"] {{
            border-color: {cls.ERROR};
            background-color: rgba(244, 67, 54, 0.1);
        }}
        
        /* === BARRAS DE PROGRESO === */
        QProgressBar {{
            border: 2px solid {cls.OUTLINE};
            border-radius: 8px;
            text-align: center;
            background-color: {cls.SURFACE_VARIANT};
            color: {cls.TEXT_PRIMARY};
            font-weight: 500;
        }}
        
        QProgressBar::chunk {{
            background-color: {cls.PRIMARY};
            border-radius: 6px;
        }}
        
        QProgressBar[class="success"]::chunk {{
            background-color: {cls.SUCCESS};
        }}
        
        QProgressBar[class="warning"]::chunk {{
            background-color: {cls.WARNING};
        }}
        
        QProgressBar[class="error"]::chunk {{
            background-color: {cls.ERROR};
        }}
        
        /* === ETIQUETAS DE TEXTO === */
        QLabel[class="title"] {{
            font-size: 24px;
            font-weight: 600;
            color: {cls.TEXT_PRIMARY};
            margin: 8px 0px;
        }}
        
        QLabel[class="subtitle"] {{
            font-size: 18px;
            font-weight: 500;
            color: {cls.TEXT_PRIMARY};
            margin: 6px 0px;
        }}
        
        QLabel[class="body"] {{
            font-size: 14px;
            color: {cls.TEXT_PRIMARY};
            margin: 4px 0px;
        }}
        
        QLabel[class="caption"] {{
            font-size: 12px;
            color: {cls.TEXT_SECONDARY};
            margin: 2px 0px;
        }}
        
        QLabel[class="overline"] {{
            font-size: 10px;
            font-weight: 500;
            color: {cls.TEXT_SECONDARY};
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        
        /* === INDICADORES DE CALIDAD === */
        QFrame[class="quality-indicator"] {{
            border-radius: 6px;
            padding: 8px 12px;
            margin: 2px;
            font-weight: 600;
        }}
        
        QFrame[class="quality-excellent"] {{
            background-color: rgba(76, 175, 80, 0.2);
            border: 2px solid {cls.QUALITY_EXCELLENT};
        }}
        
        QFrame[class="quality-good"] {{
            background-color: rgba(139, 195, 74, 0.2);
            border: 2px solid {cls.QUALITY_GOOD};
        }}
        
        QFrame[class="quality-fair"] {{
            background-color: rgba(255, 152, 0, 0.2);
            border: 2px solid {cls.QUALITY_FAIR};
        }}
        
        QFrame[class="quality-poor"] {{
            background-color: rgba(244, 67, 54, 0.2);
            border: 2px solid {cls.QUALITY_POOR};
        }}
        
        QFrame[class="quality-unknown"] {{
            background-color: rgba(158, 158, 158, 0.2);
            border: 2px solid {cls.QUALITY_UNKNOWN};
        }}
        
        /* === SCROLLBARS === */
        QScrollBar:vertical {{
            background-color: {cls.SURFACE_VARIANT};
            width: 12px;
            border-radius: 6px;
        }}
        
        QScrollBar::handle:vertical {{
            background-color: {cls.OUTLINE};
            border-radius: 6px;
            min-height: 20px;
        }}
        
        QScrollBar::handle:vertical:hover {{
            background-color: {cls.PRIMARY_LIGHT};
        }}
        
        QScrollBar:horizontal {{
            background-color: {cls.SURFACE_VARIANT};
            height: 12px;
            border-radius: 6px;
        }}
        
        QScrollBar::handle:horizontal {{
            background-color: {cls.OUTLINE};
            border-radius: 6px;
            min-width: 20px;
        }}
        
        QScrollBar::handle:horizontal:hover {{
            background-color: {cls.PRIMARY_LIGHT};
        }}
        
        /* === TOOLTIPS === */
        QToolTip {{
            background-color: {cls.BALLISTIC_PRIMARY};
            color: {cls.TEXT_ON_PRIMARY};
            border: 1px solid {cls.BALLISTIC_ACCENT};
            border-radius: 6px;
            padding: 8px 12px;
            font-size: 12px;
        }}
        
        /* === SEPARADORES === */
        QFrame[class="separator"] {{
            background-color: {cls.DIVIDER};
            max-height: 1px;
            margin: 8px 0px;
        }}
        
        /* === GROUPBOX === */
        QGroupBox {{
            background-color: {cls.SURFACE};
            border: 2px solid {cls.OUTLINE};
            border-radius: 8px;
            padding: 16px;
            margin: 8px 0px;
            font-weight: 600;
            color: {cls.TEXT_PRIMARY};
        }}
        
        QGroupBox::title {{
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 4px 8px;
            background-color: {cls.PRIMARY};
            color: {cls.TEXT_ON_PRIMARY};
            border-radius: 4px;
            font-weight: 600;
        }}
        
        /* === CHECKBOXES === */
        QCheckBox {{
            color: {cls.TEXT_PRIMARY};
            font-weight: 500;
            spacing: 8px;
        }}
        
        QCheckBox::indicator {{
            width: 18px;
            height: 18px;
            border: 2px solid {cls.OUTLINE};
            border-radius: 4px;
            background-color: {cls.SURFACE};
        }}
        
        QCheckBox::indicator:checked {{
            background-color: {cls.PRIMARY};
            border-color: {cls.PRIMARY};
        }}
        
        /* === RADIOBUTTONS === */
        QRadioButton {{
            color: {cls.TEXT_PRIMARY};
            font-weight: 500;
            spacing: 8px;
        }}
        
        QRadioButton::indicator {{
            width: 18px;
            height: 18px;
            border: 2px solid {cls.OUTLINE};
            border-radius: 9px;
            background-color: {cls.SURFACE};
        }}
        
        QRadioButton::indicator:checked {{
            background-color: {cls.PRIMARY};
            border-color: {cls.PRIMARY};
        }}
        
        /* === SPINBOX === */
        QSpinBox, QDoubleSpinBox {{
            border: 2px solid {cls.OUTLINE};
            border-radius: 6px;
            padding: 6px 8px;
            background-color: {cls.SURFACE};
            color: {cls.TEXT_PRIMARY};
            font-size: 14px;
            min-width: 80px;
        }}
        
        QSpinBox:focus, QDoubleSpinBox:focus {{
            border-color: {cls.PRIMARY};
        }}
        
        /* === SLIDERS === */
        QSlider::groove:horizontal {{
            border: 1px solid {cls.OUTLINE};
            height: 6px;
            background: {cls.SURFACE_VARIANT};
            border-radius: 3px;
        }}
        
        QSlider::handle:horizontal {{
            background: {cls.PRIMARY};
            border: 2px solid {cls.PRIMARY_DARK};
            width: 18px;
            margin: -6px 0;
            border-radius: 9px;
        }}
        
        /* === TABLAS === */
        QTableWidget {{
            background-color: {cls.SURFACE};
            alternate-background-color: {cls.SURFACE_VARIANT};
            gridline-color: {cls.DIVIDER};
            border: 1px solid {cls.DIVIDER};
            border-radius: 8px;
        }}
        
        QTableWidget::item {{
            padding: 8px;
            border: none;
        }}
        
        QTableWidget::item:selected {{
            background-color: {cls.PRIMARY_LIGHT};
            color: {cls.TEXT_ON_PRIMARY};
        }}
        
        QHeaderView::section {{
            background-color: {cls.PRIMARY};
            color: {cls.TEXT_ON_PRIMARY};
            padding: 8px 12px;
            border: none;
            font-weight: 600;
        }}
        
        /* === MENÚS === */
        QMenuBar {{
            background-color: {cls.SURFACE};
            color: {cls.TEXT_PRIMARY};
            border-bottom: 1px solid {cls.DIVIDER};
        }}
        
        QMenuBar::item {{
            background-color: transparent;
            padding: 8px 12px;
            border-radius: 4px;
        }}
        
        QMenuBar::item:selected {{
            background-color: {cls.PRIMARY_DARK};
        }}
        
        QMenu {{
            background-color: {cls.SURFACE};
            border: 1px solid {cls.DIVIDER};
            border-radius: 6px;
            padding: 4px;
        }}
        
        QMenu::item {{
            padding: 8px 20px;
            border-radius: 4px;
        }}
        
        QMenu::item:selected {{
            background-color: {cls.PRIMARY_LIGHT};
            color: {cls.TEXT_ON_PRIMARY};
        }}
        
        /* === STATUSBAR === */
        QStatusBar {{
            background-color: {cls.SURFACE};
            border-top: 1px solid {cls.DIVIDER};
            color: {cls.TEXT_PRIMARY};
            padding: 4px 8px;
        }}
        
        QStatusBar::item {{
            border: none;
        }}
        
        /* === ESTILOS ESPECÍFICOS PARA DEEP LEARNING === */
        QPushButton[class="dl-button"] {{
            background-color: {cls.DL_PRIMARY};
            color: {cls.TEXT_ON_PRIMARY};
            border: none;
            padding: 10px 20px;
            border-radius: 6px;
            font-weight: 500;
            font-size: 13px;
        }}
        
        QPushButton[class="dl-button"]:hover {{
            background-color: {cls.DL_PRIMARY_DARK};
        }}
        
        QGroupBox[class="dl-group"] {{
            background-color: {cls.SURFACE};
            border: 2px solid {cls.DL_PRIMARY_LIGHT};
            border-radius: 12px;
            padding: 16px;
            margin: 8px 0px;
            font-weight: 600;
            color: {cls.DL_PRIMARY_DARK};
        }}
        
        QGroupBox[class="dl-group"]::title {{
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 4px 12px;
            background-color: {cls.DL_PRIMARY};
            color: {cls.TEXT_ON_PRIMARY};
            border-radius: 6px;
            font-weight: 600;
            font-size: 13px;
        }}
        
        /* === ESTILOS PARA PANELES FLOTANTES === */
        QWidget[class="floating-panel"] {{
            background-color: {cls.DOCK_BACKGROUND};
            border: 2px solid {cls.DOCK_BORDER};
            border-radius: 12px;
        }}
        
        QWidget[class="floating-panel-header"] {{
            background-color: {cls.DOCK_BACKGROUND};
            border-bottom: 1px solid {cls.DOCK_BORDER};
            padding: 8px 12px;
            font-weight: 600;
            color: {cls.DOCK_TITLE};
        }}
        
        QWidget[class="floating-panel-content"] {{
            background-color: {cls.SURFACE};
            border: none;
            padding: 12px;
        }}
        
        /* === ESTILOS PARA ANÁLISIS CIENTÍFICO === */
        QFrame[class="scientific-step"] {{
            background-color: {cls.SURFACE};
            border: 1px solid {cls.BALLISTIC_ACCENT};
            border-radius: 8px;
            padding: 12px;
            margin: 4px 0px;
        }}
        
        QFrame[class="scientific-step-active"] {{
            border-color: {cls.BALLISTIC_PRIMARY};
            background-color: rgba(94, 129, 172, 0.1);
        }}
        
        QFrame[class="scientific-step-completed"] {{
            border-color: {cls.SUCCESS};
            background-color: rgba(76, 175, 80, 0.1);
        }}
        
        QLabel[class="scientific-step-title"] {{
            font-size: 16px;
            font-weight: 600;
            color: {cls.BALLISTIC_PRIMARY};
        }}
        
        QLabel[class="scientific-step-description"] {{
            font-size: 13px;
            color: {cls.TEXT_SECONDARY};
            margin: 4px 0px;
        }}
        """

def apply_SIGeC_Balistica_theme(app: QApplication):
    """Aplica el tema SIGeC Balística a la aplicación"""
    
    # Configurar fuente por defecto
    font = QFont("Segoe UI", 10)
    app.setFont(font)
    
    # Aplicar hoja de estilos
    app.setStyleSheet(SIGeCBallisticaTheme.get_stylesheet())
    
    # Configurar paleta de colores
    palette = QPalette()
    
    # Colores de ventana
    palette.setColor(QPalette.Window, QColor(SIGeCBallisticaTheme.BACKGROUND))
    palette.setColor(QPalette.WindowText, QColor(SIGeCBallisticaTheme.TEXT_PRIMARY))
    
    # Colores de base
    palette.setColor(QPalette.Base, QColor(SIGeCBallisticaTheme.SURFACE))
    palette.setColor(QPalette.AlternateBase, QColor(SIGeCBallisticaTheme.SURFACE_VARIANT))
    
    # Colores de texto
    palette.setColor(QPalette.Text, QColor(SIGeCBallisticaTheme.TEXT_PRIMARY))
    palette.setColor(QPalette.BrightText, QColor(SIGeCBallisticaTheme.TEXT_ON_PRIMARY))
    
    # Colores de botón
    palette.setColor(QPalette.Button, QColor(SIGeCBallisticaTheme.PRIMARY))
    palette.setColor(QPalette.ButtonText, QColor(SIGeCBallisticaTheme.TEXT_ON_PRIMARY))
    
    # Colores de selección
    palette.setColor(QPalette.Highlight, QColor(SIGeCBallisticaTheme.PRIMARY))
    palette.setColor(QPalette.HighlightedText, QColor(SIGeCBallisticaTheme.TEXT_ON_PRIMARY))
    
    app.setPalette(palette)


# ===== CARGA SEGURA DE QSS EXTERNOS =====
def sanitize_qss(qss_text: str) -> str:
    """Elimina reglas CSS no soportadas por Qt para evitar warnings."""
    def remove_balanced_blocks(text: str, token: str) -> str:
        """Elimina bloques balanceados comenzando por 'token' hasta su '}' correspondiente."""
        i = 0
        token_lower = token.lower()
        while True:
            idx = text.lower().find(token_lower, i)
            if idx == -1:
                break
            brace_idx = text.find("{", idx)
            if brace_idx == -1:
                break
            depth = 1
            j = brace_idx + 1
            while j < len(text) and depth > 0:
                ch = text[j]
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                j += 1
            # Eliminar desde idx hasta j
            text = text[:idx] + text[j:]
            i = idx
        return text

    # Eliminar propiedades individuales conocidas no soportadas
    prop_patterns = [
        r"\boutline\s*:\s*[^;]+;",
        r"\bbox-shadow\s*:\s*[^;]+;",
        r"\bcursor\s*:\s*[^;]+;",
        r"\btransition\s*:[^;]+;",
        r"\bcontent\s*:\s*[^;]+;",
    ]
    for p in prop_patterns:
        qss_text = re.sub(p, "", qss_text, flags=re.IGNORECASE)

    # Eliminar bloques @media completos de forma balanceada
    qss_text = remove_balanced_blocks(qss_text, "@media")

    # Eliminar bloques de pseudo-elementos ::after de forma balanceada
    qss_text = remove_balanced_blocks(qss_text, "::after")

    # Eliminar selectores con clase CSS (no soportados por Qt): .clase {...}
    # Incluye variantes como .clase.subclase:hover { ... }
    qss_text = re.sub(r"\n\s*\.[^{]+\{[^{}]*\}", "\n", qss_text, flags=re.IGNORECASE)

    return qss_text


def load_qss_safe(qss_path: Path) -> str:
    """Lee y sanea un archivo QSS, devolviendo texto listo para aplicar.
    Si el archivo no existe, devuelve cadena vacía.
    """
    try:
        if not qss_path or not qss_path.exists():
            return ""
        text = qss_path.read_text(encoding="utf-8")
        return sanitize_qss(text)
    except Exception:
        return ""


def apply_modern_qss_to_widget(widget) -> bool:
    """Aplica el QSS moderno saneado al widget si está disponible."""
    theme_path = Path(__file__).parent / "styles" / "modern_theme.qss"
    qss = load_qss_safe(theme_path)
    if qss:
        try:
            widget.setStyleSheet(qss)
            return True
        except Exception:
            return False
    return False