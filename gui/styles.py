#!/usr/bin/env python3
"""
Sistema de estilos moderno para SEACABAr
Implementa Material Design con colores profesionales
"""

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QPalette, QColor

class SEACABArTheme:
    """Tema visual moderno para SEACABAr"""
    
    # Colores principales (Material Design)
    PRIMARY = "#1976D2"          # Azul profesional
    PRIMARY_DARK = "#1565C0"     # Azul oscuro
    PRIMARY_LIGHT = "#42A5F5"    # Azul claro
    
    SECONDARY = "#388E3C"        # Verde éxito
    SECONDARY_DARK = "#2E7D32"   # Verde oscuro
    SECONDARY_LIGHT = "#66BB6A"  # Verde claro
    
    WARNING = "#F57C00"          # Naranja advertencia
    WARNING_DARK = "#EF6C00"     # Naranja oscuro
    WARNING_LIGHT = "#FFB74D"    # Naranja claro
    
    ERROR = "#D32F2F"            # Rojo error
    ERROR_DARK = "#C62828"       # Rojo oscuro
    ERROR_LIGHT = "#EF5350"      # Rojo claro
    
    # Colores de superficie
    SURFACE = "#FFFFFF"          # Blanco
    SURFACE_VARIANT = "#F5F5F5"  # Gris muy claro
    BACKGROUND = "#FAFAFA"       # Gris de fondo
    
    # Colores de texto
    TEXT_PRIMARY = "#212121"     # Negro principal
    TEXT_SECONDARY = "#757575"   # Gris secundario
    TEXT_DISABLED = "#BDBDBD"    # Gris deshabilitado
    TEXT_ON_PRIMARY = "#FFFFFF"  # Blanco sobre primario
    
    # Colores de Deep Learning (tonos morados/violetas para diferenciación)
    DL_PRIMARY = "#673AB7"       # Morado Deep Learning
    DL_PRIMARY_DARK = "#512DA8"  # Morado oscuro
    DL_PRIMARY_LIGHT = "#9575CD" # Morado claro
    
    DL_ACCENT = "#E91E63"        # Rosa accent para DL
    DL_ACCENT_DARK = "#C2185B"   # Rosa oscuro
    DL_ACCENT_LIGHT = "#F06292"  # Rosa claro
    
    # Colores de bordes y divisores
    DIVIDER = "#E0E0E0"         # Gris divisor
    OUTLINE = "#9E9E9E"         # Gris contorno
    
    @classmethod
    def get_stylesheet(cls) -> str:
        """Retorna el stylesheet completo de la aplicación"""
        return f"""
        /* Estilo base de la aplicación */
        QMainWindow {{
            background-color: {cls.BACKGROUND};
            color: {cls.TEXT_PRIMARY};
            font-family: 'Segoe UI', 'Arial', sans-serif;
        }}
        
        /* Tabs principales */
        QTabWidget::pane {{
            border: 1px solid {cls.DIVIDER};
            background-color: {cls.SURFACE};
            border-radius: 8px;
        }}
        
        QTabBar::tab {{
            background-color: {cls.SURFACE_VARIANT};
            color: {cls.TEXT_SECONDARY};
            padding: 12px 24px;
            margin-right: 2px;
            border-top-left-radius: 8px;
            border-top-right-radius: 8px;
            font-weight: 500;
            min-width: 120px;
        }}
        
        QTabBar::tab:selected {{
            background-color: {cls.PRIMARY};
            color: {cls.TEXT_ON_PRIMARY};
            font-weight: 600;
        }}
        
        QTabBar::tab:hover:!selected {{
            background-color: {cls.PRIMARY_LIGHT};
            color: {cls.TEXT_ON_PRIMARY};
        }}
        
        /* Botones principales */
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
        
        /* Botones secundarios */
        QPushButton[class="secondary"] {{
            background-color: {cls.SECONDARY};
        }}
        
        QPushButton[class="secondary"]:hover {{
            background-color: {cls.SECONDARY_DARK};
        }}
        
        /* Botones de advertencia */
        QPushButton[class="warning"] {{
            background-color: {cls.WARNING};
        }}
        
        QPushButton[class="warning"]:hover {{
            background-color: {cls.WARNING_DARK};
        }}
        
        /* Botones de error */
        QPushButton[class="error"] {{
            background-color: {cls.ERROR};
        }}
        
        QPushButton[class="error"]:hover {{
            background-color: {cls.ERROR_DARK};
        }}
        
        /* Campos de entrada */
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
        
        /* ComboBox */
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
            image: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTYiIGhlaWdodD0iMTYiIHZpZXdCb3g9IjAgMCAxNiAxNiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTQgNkw4IDEwTDEyIDYiIHN0cm9rZT0iIzc1NzU3NSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiLz4KPC9zdmc+);
        }}
        
        /* Tarjetas */
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
        
        /* Paneles colapsables */
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
        
        /* Drop zones */
        QFrame[class="drop-zone"] {{
            border: 3px dashed {cls.OUTLINE};
            border-radius: 12px;
            background-color: {cls.SURFACE_VARIANT};
            padding: 32px;
            margin: 8px;
        }}
        
        QFrame[class="drop-zone-active"] {{
            border-color: {cls.PRIMARY};
            background-color: {cls.PRIMARY_LIGHT}20;
        }}
        
        QFrame[class="drop-zone-error"] {{
            border-color: {cls.ERROR};
            background-color: {cls.ERROR_LIGHT}20;
        }}
        
        /* Barras de progreso */
        QProgressBar {{
            border: none;
            border-radius: 6px;
            background-color: {cls.SURFACE_VARIANT};
            text-align: center;
            font-weight: 500;
            height: 12px;
        }}
        
        QProgressBar::chunk {{
            background-color: {cls.PRIMARY};
            border-radius: 6px;
        }}
        
        /* Labels */
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
        
        QLabel[class="success"] {{
            color: {cls.SECONDARY};
            font-weight: 500;
        }}
        
        QLabel[class="warning"] {{
            color: {cls.WARNING};
            font-weight: 500;
        }}
        
        QLabel[class="error"] {{
            color: {cls.ERROR};
            font-weight: 500;
        }}
        
        /* Scrollbars */
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
            background-color: {cls.TEXT_SECONDARY};
        }}
        
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
            border: none;
            background: none;
        }}
        
        /* Tooltips */
        QToolTip {{
            background-color: {cls.TEXT_PRIMARY};
            color: {cls.TEXT_ON_PRIMARY};
            border: none;
            border-radius: 6px;
            padding: 8px 12px;
            font-size: 12px;
        }}
        
        /* Separadores */
        QFrame[class="separator"] {{
            background-color: {cls.DIVIDER};
            max-height: 1px;
            margin: 8px 0px;
        }}
        
        /* Indicadores de paso */
        QFrame[class="step-indicator"] {{
            background-color: {cls.SURFACE};
            border: 2px solid {cls.OUTLINE};
            border-radius: 20px;
            min-width: 40px;
            max-width: 40px;
            min-height: 40px;
            max-height: 40px;
        }}
        
        QFrame[class="step-indicator-active"] {{
            background-color: {cls.PRIMARY};
            border-color: {cls.PRIMARY};
        }}
        
        QFrame[class="step-indicator-completed"] {{
            background-color: {cls.SECONDARY};
            border-color: {cls.SECONDARY};
        }}
        
        /* Estilos específicos para Deep Learning */
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
        
        QPushButton[class="dl-button"]:pressed {{
            background-color: {cls.DL_PRIMARY_DARK};
            padding: 11px 19px 9px 21px;
        }}
        
        QPushButton[class="dl-button"]:disabled {{
            background-color: {cls.TEXT_DISABLED};
            color: {cls.SURFACE};
        }}
        
        QPushButton[class="dl-advanced"] {{
            background-color: {cls.DL_ACCENT};
            color: {cls.TEXT_ON_PRIMARY};
            border: none;
            padding: 8px 16px;
            border-radius: 6px;
            font-weight: 500;
            font-size: 12px;
        }}
        
        QPushButton[class="dl-advanced"]:hover {{
            background-color: {cls.DL_ACCENT_DARK};
        }}
        
        QPushButton[class="dl-advanced"]:disabled {{
            background-color: {cls.TEXT_DISABLED};
            color: {cls.SURFACE};
        }}
        
        QCheckBox[class="dl-checkbox"] {{
            color: {cls.DL_PRIMARY_DARK};
            font-weight: 500;
            spacing: 8px;
        }}
        
        QCheckBox[class="dl-checkbox"]::indicator {{
            width: 18px;
            height: 18px;
            border: 2px solid {cls.DL_PRIMARY_LIGHT};
            border-radius: 4px;
            background-color: {cls.SURFACE};
        }}
        
        QCheckBox[class="dl-checkbox"]::indicator:checked {{
            background-color: {cls.DL_PRIMARY};
            border-color: {cls.DL_PRIMARY};
            image: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTIiIGhlaWdodD0iMTIiIHZpZXdCb3g9IjAgMCAxMiAxMiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTEwIDNMNC41IDguNUwyIDYiIHN0cm9rZT0id2hpdGUiIHN0cm9rZS13aWR0aD0iMiIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBzdHJva2UtbGluZWpvaW49InJvdW5kIi8+Cjwvc3ZnPg==);
        }}
        
        QComboBox[class="dl-combo"] {{
            border: 2px solid {cls.DL_PRIMARY_LIGHT};
            border-radius: 6px;
            padding: 8px 12px;
            background-color: {cls.SURFACE};
            color: {cls.DL_PRIMARY_DARK};
            font-size: 13px;
            font-weight: 500;
            min-width: 150px;
        }}
        
        QComboBox[class="dl-combo"]:focus {{
            border-color: {cls.DL_PRIMARY};
        }}
        
        QSpinBox[class="dl-spin"], QDoubleSpinBox[class="dl-spin"] {{
            border: 2px solid {cls.DL_PRIMARY_LIGHT};
            border-radius: 6px;
            padding: 6px 8px;
            background-color: {cls.SURFACE};
            color: {cls.DL_PRIMARY_DARK};
            font-size: 13px;
            font-weight: 500;
            min-width: 80px;
        }}
        
        QSpinBox[class="dl-spin"]:focus, QDoubleSpinBox[class="dl-spin"]:focus {{
            border-color: {cls.DL_PRIMARY};
        }}
        
        QLabel[class="dl-label"] {{
            color: {cls.DL_PRIMARY_DARK};
            font-weight: 500;
            font-size: 13px;
        }}
        
        QFrame[class="dl-separator"] {{
            background-color: {cls.DL_PRIMARY_LIGHT};
            max-height: 2px;
            margin: 12px 0px;
            border-radius: 1px;
        }}
        
        /* Indicador de estado DL */
        QFrame[class="dl-status-active"] {{
            background-color: {cls.DL_PRIMARY};
            border: 2px solid {cls.DL_PRIMARY_DARK};
            border-radius: 6px;
            padding: 4px 8px;
        }}
        
        QFrame[class="dl-status-inactive"] {{
            background-color: {cls.SURFACE_VARIANT};
            border: 2px solid {cls.OUTLINE};
            border-radius: 6px;
            padding: 4px 8px;
        }}
        """

def apply_seacaba_theme(app: QApplication):
    """Aplica el tema SEACABAr a la aplicación"""
    
    # Configurar fuente por defecto
    font = QFont("Segoe UI", 10)
    app.setFont(font)
    
    # Aplicar stylesheet
    app.setStyleSheet(SEACABArTheme.get_stylesheet())
    
    # Configurar paleta de colores
    palette = QPalette()
    
    # Colores de ventana
    palette.setColor(QPalette.Window, QColor(SEACABArTheme.BACKGROUND))
    palette.setColor(QPalette.WindowText, QColor(SEACABArTheme.TEXT_PRIMARY))
    
    # Colores de base
    palette.setColor(QPalette.Base, QColor(SEACABArTheme.SURFACE))
    palette.setColor(QPalette.AlternateBase, QColor(SEACABArTheme.SURFACE_VARIANT))
    
    # Colores de texto
    palette.setColor(QPalette.Text, QColor(SEACABArTheme.TEXT_PRIMARY))
    palette.setColor(QPalette.BrightText, QColor(SEACABArTheme.TEXT_ON_PRIMARY))
    
    # Colores de botones
    palette.setColor(QPalette.Button, QColor(SEACABArTheme.PRIMARY))
    palette.setColor(QPalette.ButtonText, QColor(SEACABArTheme.TEXT_ON_PRIMARY))
    
    # Colores de highlight
    palette.setColor(QPalette.Highlight, QColor(SEACABArTheme.PRIMARY))
    palette.setColor(QPalette.HighlightedText, QColor(SEACABArTheme.TEXT_ON_PRIMARY))
    
    app.setPalette(palette)