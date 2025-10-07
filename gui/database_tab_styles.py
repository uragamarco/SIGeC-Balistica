# -*- coding: utf-8 -*-
"""
Estilos CSS mejorados para database_tab.py
"""

DATABASE_TAB_STYLES = """
/* Estilos principales para DatabaseTab */
QTabWidget::pane {
    border: 1px solid #bdc3c7;
    background-color: #ffffff;
    border-radius: 8px;
}

QTabWidget::tab-bar {
    alignment: center;
}

QTabBar::tab {
    background-color: #ecf0f1;
    color: #2c3e50;
    padding: 12px 20px;
    margin-right: 2px;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
    font-weight: 500;
    min-width: 120px;
}

QTabBar::tab:selected {
    background-color: #3498db;
    color: white;
    font-weight: bold;
}

QTabBar::tab:hover:!selected {
    background-color: #d5dbdb;
}

/* Estilos para Dashboard */
#dashboardContainer {
    background-color: #f8f9fa;
    border-radius: 8px;
    padding: 15px;
}

#statsCard {
    background-color: white;
    border: 1px solid #e9ecef;
    border-radius: 8px;
    padding: 20px;
    margin: 10px;
}

#chartContainer {
    background-color: white;
    border: 1px solid #dee2e6;
    border-radius: 8px;
    padding: 15px;
    margin: 5px;
}

/* Estilos para gestión de casos */
#casesPanel {
    background-color: #ffffff;
    border: 1px solid #dee2e6;
    border-radius: 8px;
    padding: 15px;
}

#detailsPanel {
    background-color: #f8f9fa;
    border: 1px solid #dee2e6;
    border-radius: 8px;
    padding: 15px;
}

QTreeWidget {
    background-color: white;
    border: 1px solid #dee2e6;
    border-radius: 6px;
    selection-background-color: #3498db;
    selection-color: white;
    font-size: 13px;
}

QTreeWidget::item {
    padding: 8px;
    border-bottom: 1px solid #f1f3f4;
}

QTreeWidget::item:selected {
    background-color: #3498db;
    color: white;
}

QTreeWidget::item:hover {
    background-color: #e3f2fd;
}

/* Estilos para acciones por lotes */
#batchActionsContainer {
    background-color: #ffffff;
    border: 1px solid #dee2e6;
    border-radius: 8px;
    padding: 20px;
}

#selectionInfo {
    background-color: #e8f4fd;
    border: 1px solid #bee5eb;
    border-radius: 6px;
    padding: 15px;
    margin-bottom: 15px;
}

/* Estilos para búsqueda avanzada */
#advancedSearchContainer {
    background-color: #ffffff;
    border: 1px solid #dee2e6;
    border-radius: 8px;
    padding: 15px;
}

#facetFilters {
    background-color: #f8f9fa;
    border: 1px solid #e9ecef;
    border-radius: 6px;
    padding: 15px;
    margin: 10px 0;
}

#temporalFilters {
    background-color: #fff3cd;
    border: 1px solid #ffeaa7;
    border-radius: 6px;
    padding: 15px;
    margin: 10px 0;
}

#tagFilters {
    background-color: #d1ecf1;
    border: 1px solid #bee5eb;
    border-radius: 6px;
    padding: 15px;
    margin: 10px 0;
}

/* Botones mejorados */
QPushButton {
    background-color: #3498db;
    color: white;
    border: none;
    padding: 10px 20px;
    border-radius: 6px;
    font-weight: 500;
    font-size: 13px;
}

QPushButton:hover {
    background-color: #2980b9;
}

QPushButton:pressed {
    background-color: #21618c;
}

QPushButton:disabled {
    background-color: #bdc3c7;
    color: #7f8c8d;
}

/* Botones de acción específicos */
QPushButton#primaryButton {
    background-color: #27ae60;
}

QPushButton#primaryButton:hover {
    background-color: #229954;
}

QPushButton#warningButton {
    background-color: #f39c12;
}

QPushButton#warningButton:hover {
    background-color: #e67e22;
}

QPushButton#dangerButton {
    background-color: #e74c3c;
}

QPushButton#dangerButton:hover {
    background-color: #c0392b;
}

/* Campos de entrada mejorados */
QLineEdit, QTextEdit, QComboBox, QSpinBox, QDoubleSpinBox {
    border: 2px solid #e9ecef;
    border-radius: 6px;
    padding: 8px 12px;
    font-size: 13px;
    background-color: white;
}

QLineEdit:focus, QTextEdit:focus, QComboBox:focus {
    border-color: #3498db;
    outline: none;
}

/* Grupos de controles */
QGroupBox {
    font-weight: bold;
    font-size: 14px;
    color: #2c3e50;
    border: 2px solid #bdc3c7;
    border-radius: 8px;
    margin-top: 10px;
    padding-top: 15px;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 8px 0 8px;
    background-color: white;
}

/* Barras de progreso */
QProgressBar {
    border: 2px solid #bdc3c7;
    border-radius: 6px;
    text-align: center;
    font-weight: bold;
    background-color: #ecf0f1;
}

QProgressBar::chunk {
    background-color: #3498db;
    border-radius: 4px;
}

/* Scrollbars personalizadas */
QScrollBar:vertical {
    background-color: #f8f9fa;
    width: 12px;
    border-radius: 6px;
}

QScrollBar::handle:vertical {
    background-color: #bdc3c7;
    border-radius: 6px;
    min-height: 20px;
}

QScrollBar::handle:vertical:hover {
    background-color: #95a5a6;
}

QScrollBar:horizontal {
    background-color: #f8f9fa;
    height: 12px;
    border-radius: 6px;
}

QScrollBar::handle:horizontal {
    background-color: #bdc3c7;
    border-radius: 6px;
    min-width: 20px;
}

QScrollBar::handle:horizontal:hover {
    background-color: #95a5a6;
}

/* Tooltips mejorados */
QToolTip {
    background-color: #2c3e50;
    color: white;
    border: none;
    padding: 8px;
    border-radius: 6px;
    font-size: 12px;
}

/* Menús contextuales */
QMenu {
    background-color: white;
    border: 1px solid #bdc3c7;
    border-radius: 6px;
    padding: 5px;
}

QMenu::item {
    padding: 8px 20px;
    border-radius: 4px;
}

QMenu::item:selected {
    background-color: #3498db;
    color: white;
}

/* Splitters */
QSplitter::handle {
    background-color: #bdc3c7;
    border-radius: 2px;
}

QSplitter::handle:horizontal {
    width: 4px;
}

QSplitter::handle:vertical {
    height: 4px;
}

QSplitter::handle:hover {
    background-color: #95a5a6;
}

/* Etiquetas de información */
QLabel#infoLabel {
    color: #7f8c8d;
    font-size: 12px;
    font-style: italic;
}

QLabel#sectionTitle {
    color: #2c3e50;
    font-size: 16px;
    font-weight: bold;
    margin-bottom: 10px;
}

QLabel#warningLabel {
    color: #e67e22;
    font-weight: bold;
}

QLabel#errorLabel {
    color: #e74c3c;
    font-weight: bold;
}

QLabel#successLabel {
    color: #27ae60;
    font-weight: bold;
}

/* Responsive design para diferentes tamaños - Nota: PyQt5 no soporta media queries */
/* Los siguientes estilos se aplicarán por defecto */
"""

def apply_database_tab_styles(widget):
    """Aplica los estilos CSS al widget DatabaseTab"""
    widget.setStyleSheet(DATABASE_TAB_STYLES)