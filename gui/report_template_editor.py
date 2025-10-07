#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Editor de Plantillas de Reportes - SIGeC-Balística
================================================

Editor visual para crear y personalizar plantillas de reportes:
- Personalización de logos y encabezados
- Configuración de secciones y orden
- Estilos CSS personalizables
- Vista previa en tiempo real
- Gestión de plantillas (guardar/cargar)

Autor: SIGeC-BalisticaTeam
Fecha: Octubre 2025
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from pathlib import Path

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout,
    QLabel, QLineEdit, QTextEdit, QPushButton, QComboBox, QCheckBox,
    QSpinBox, QGroupBox, QTabWidget, QWidget, QSplitter, QFrame,
    QScrollArea, QListWidget, QListWidgetItem, QTreeWidget, QTreeWidgetItem,
    QColorDialog, QFontDialog, QFileDialog, QMessageBox, QSlider,
    QProgressBar, QTableWidget, QTableWidgetItem, QHeaderView
)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer, QThread, QSize
from PyQt5.QtGui import QPixmap, QFont, QColor, QPalette, QIcon
from PyQt5.QtWebEngineWidgets import QWebEngineView

from .shared_widgets import CollapsiblePanel, ResultCard

logger = logging.getLogger(__name__)

class TemplateSection:
    """Representa una sección de la plantilla"""
    
    def __init__(self, name: str, title: str, enabled: bool = True, order: int = 0):
        self.name = name
        self.title = title
        self.enabled = enabled
        self.order = order
        self.content_type = "text"  # text, image, table, chart
        self.style = {}
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'title': self.title,
            'enabled': self.enabled,
            'order': self.order,
            'content_type': self.content_type,
            'style': self.style
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TemplateSection':
        section = cls(
            data['name'],
            data['title'],
            data.get('enabled', True),
            data.get('order', 0)
        )
        section.content_type = data.get('content_type', 'text')
        section.style = data.get('style', {})
        return section

class ReportTemplate:
    """Clase para manejar plantillas de reportes"""
    
    def __init__(self, name: str = "Nueva Plantilla"):
        self.name = name
        self.description = ""
        self.author = ""
        self.created_date = datetime.now()
        self.modified_date = datetime.now()
        
        # Configuración de página
        self.page_config = {
            'format': 'A4',
            'orientation': 'portrait',
            'margins': {'top': 2.5, 'bottom': 2.5, 'left': 2.0, 'right': 2.0}
        }
        
        # Configuración de encabezado y pie
        self.header_config = {
            'enabled': True,
            'height': 80,
            'logo_path': '',
            'logo_position': 'left',  # left, center, right
            'title': 'Reporte Forense',
            'subtitle': 'Sistema SIGeC-Balística',
            'show_date': True,
            'show_page_numbers': True
        }
        
        self.footer_config = {
            'enabled': True,
            'height': 60,
            'text': 'Documento confidencial - Uso restringido',
            'show_organization': True,
            'show_signature_line': False
        }
        
        # Secciones de la plantilla
        self.sections = self._create_default_sections()
        
        # Estilos CSS
        self.css_styles = self._create_default_styles()
        
    def _create_default_sections(self) -> List[TemplateSection]:
        """Crea las secciones por defecto"""
        return [
            TemplateSection("metadata", "Información del Documento", True, 0),
            TemplateSection("executive_summary", "Resumen Ejecutivo", True, 1),
            TemplateSection("methodology", "Metodología", True, 2),
            TemplateSection("results", "Resultados", True, 3),
            TemplateSection("analysis_details", "Detalles del Análisis", True, 4),
            TemplateSection("conclusions", "Conclusiones", True, 5),
            TemplateSection("recommendations", "Recomendaciones", False, 6),
            TemplateSection("appendices", "Apéndices", True, 7),
            TemplateSection("technical_specs", "Especificaciones Técnicas", False, 8),
            TemplateSection("quality_metrics", "Métricas de Calidad", False, 9)
        ]
    
    def _create_default_styles(self) -> str:
        """Crea los estilos CSS por defecto"""
        return """
        body {
            font-family: 'Arial', sans-serif;
            font-size: 12pt;
            line-height: 1.6;
            color: #333;
            margin: 0;
            padding: 20px;
        }
        
        .header {
            border-bottom: 2px solid #2c3e50;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }
        
        .logo {
            max-height: 60px;
            float: left;
            margin-right: 20px;
        }
        
        .header-text {
            overflow: hidden;
        }
        
        .title {
            font-size: 24pt;
            font-weight: bold;
            color: #2c3e50;
            margin: 0;
        }
        
        .subtitle {
            font-size: 14pt;
            color: #7f8c8d;
            margin: 5px 0 0 0;
        }
        
        .section {
            margin-bottom: 30px;
            page-break-inside: avoid;
        }
        
        .section h1 {
            font-size: 18pt;
            color: #2c3e50;
            border-bottom: 1px solid #bdc3c7;
            padding-bottom: 5px;
            margin-bottom: 15px;
        }
        
        .section h2 {
            font-size: 16pt;
            color: #34495e;
            margin-bottom: 10px;
        }
        
        .section h3 {
            font-size: 14pt;
            color: #34495e;
            margin-bottom: 8px;
        }
        
        .metadata-table {
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 20px;
        }
        
        .metadata-table th,
        .metadata-table td {
            border: 1px solid #bdc3c7;
            padding: 8px;
            text-align: left;
        }
        
        .metadata-table th {
            background-color: #ecf0f1;
            font-weight: bold;
        }
        
        .result-card {
            border: 1px solid #bdc3c7;
            border-radius: 5px;
            padding: 15px;
            margin-bottom: 15px;
            background-color: #f8f9fa;
        }
        
        .result-card h4 {
            margin-top: 0;
            color: #2c3e50;
        }
        
        .footer {
            border-top: 1px solid #bdc3c7;
            padding-top: 20px;
            margin-top: 40px;
            font-size: 10pt;
            color: #7f8c8d;
            text-align: center;
        }
        
        .page-break {
            page-break-before: always;
        }
        
        @media print {
            body { margin: 0; }
            .no-print { display: none; }
        }
        """
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte la plantilla a diccionario"""
        return {
            'name': self.name,
            'description': self.description,
            'author': self.author,
            'created_date': self.created_date.isoformat(),
            'modified_date': self.modified_date.isoformat(),
            'page_config': self.page_config,
            'header_config': self.header_config,
            'footer_config': self.footer_config,
            'sections': [section.to_dict() for section in self.sections],
            'css_styles': self.css_styles
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ReportTemplate':
        """Crea plantilla desde diccionario"""
        template = cls(data.get('name', 'Plantilla Sin Nombre'))
        template.description = data.get('description', '')
        template.author = data.get('author', '')
        
        if 'created_date' in data:
            template.created_date = datetime.fromisoformat(data['created_date'])
        if 'modified_date' in data:
            template.modified_date = datetime.fromisoformat(data['modified_date'])
            
        template.page_config = data.get('page_config', template.page_config)
        template.header_config = data.get('header_config', template.header_config)
        template.footer_config = data.get('footer_config', template.footer_config)
        template.css_styles = data.get('css_styles', template.css_styles)
        
        # Cargar secciones
        if 'sections' in data:
            template.sections = [
                TemplateSection.from_dict(section_data) 
                for section_data in data['sections']
            ]
        
        return template

class ReportTemplateEditor(QDialog):
    """Editor visual de plantillas de reportes"""
    
    template_saved = pyqtSignal(str)  # nombre de la plantilla guardada
    
    def __init__(self, template: ReportTemplate = None, parent=None):
        super().__init__(parent)
        
        self.template = template or ReportTemplate()
        self.templates_dir = Path("templates/reports")
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        
        self.setWindowTitle("Editor de Plantillas de Reportes")
        self.setModal(True)
        self.resize(1200, 800)
        
        self._setup_ui()
        self._connect_signals()
        self._load_template_data()
        
        logger.info("Editor de plantillas inicializado")
    
    def _setup_ui(self):
        """Configura la interfaz de usuario"""
        layout = QVBoxLayout(self)
        
        # Barra de herramientas
        toolbar_layout = QHBoxLayout()
        
        self.template_combo = QComboBox()
        self.template_combo.setMinimumWidth(200)
        toolbar_layout.addWidget(QLabel("Plantilla:"))
        toolbar_layout.addWidget(self.template_combo)
        
        self.new_template_btn = QPushButton("Nueva")
        self.load_template_btn = QPushButton("Cargar")
        self.save_template_btn = QPushButton("Guardar")
        self.save_as_template_btn = QPushButton("Guardar Como...")
        
        toolbar_layout.addWidget(self.new_template_btn)
        toolbar_layout.addWidget(self.load_template_btn)
        toolbar_layout.addWidget(self.save_template_btn)
        toolbar_layout.addWidget(self.save_as_template_btn)
        toolbar_layout.addStretch()
        
        layout.addLayout(toolbar_layout)
        
        # Splitter principal
        main_splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(main_splitter)
        
        # Panel izquierdo - Configuración
        config_panel = self._create_configuration_panel()
        main_splitter.addWidget(config_panel)
        
        # Panel derecho - Vista previa
        preview_panel = self._create_preview_panel()
        main_splitter.addWidget(preview_panel)
        
        # Configurar proporciones
        main_splitter.setSizes([500, 700])
        
        # Botones de diálogo
        buttons_layout = QHBoxLayout()
        buttons_layout.addStretch()
        
        self.apply_btn = QPushButton("Aplicar")
        self.cancel_btn = QPushButton("Cancelar")
        self.ok_btn = QPushButton("Aceptar")
        
        buttons_layout.addWidget(self.apply_btn)
        buttons_layout.addWidget(self.cancel_btn)
        buttons_layout.addWidget(self.ok_btn)
        
        layout.addLayout(buttons_layout)
    
    def _create_configuration_panel(self) -> QWidget:
        """Crea el panel de configuración"""
        panel = QFrame()
        panel.setMaximumWidth(550)
        
        layout = QVBoxLayout(panel)
        
        # Tabs de configuración
        self.config_tabs = QTabWidget()
        
        # Tab 1: Información general
        general_tab = self._create_general_tab()
        self.config_tabs.addTab(general_tab, "General")
        
        # Tab 2: Encabezado y pie
        header_footer_tab = self._create_header_footer_tab()
        self.config_tabs.addTab(header_footer_tab, "Encabezado/Pie")
        
        # Tab 3: Secciones
        sections_tab = self._create_sections_tab()
        self.config_tabs.addTab(sections_tab, "Secciones")
        
        # Tab 4: Estilos
        styles_tab = self._create_styles_tab()
        self.config_tabs.addTab(styles_tab, "Estilos")
        
        layout.addWidget(self.config_tabs)
        
        return panel
    
    def _create_preview_panel(self) -> QWidget:
        """Crea el panel de vista previa"""
        panel = QFrame()
        layout = QVBoxLayout(panel)
        
        # Título
        title = QLabel("Vista Previa")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)
        
        # Botones de vista previa
        preview_buttons = QHBoxLayout()
        
        self.refresh_preview_btn = QPushButton("Actualizar Vista Previa")
        self.export_preview_btn = QPushButton("Exportar Vista Previa")
        
        preview_buttons.addWidget(self.refresh_preview_btn)
        preview_buttons.addWidget(self.export_preview_btn)
        preview_buttons.addStretch()
        
        layout.addLayout(preview_buttons)
        
        # Vista previa web
        self.preview_web = QWebEngineView()
        layout.addWidget(self.preview_web)
        
        return panel
    
    def _create_general_tab(self) -> QWidget:
        """Crea la pestaña de configuración general"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Información de la plantilla
        info_group = QGroupBox("Información de la Plantilla")
        info_layout = QFormLayout(info_group)
        
        self.name_input = QLineEdit()
        self.description_input = QTextEdit()
        self.description_input.setMaximumHeight(80)
        self.author_input = QLineEdit()
        
        info_layout.addRow("Nombre:", self.name_input)
        info_layout.addRow("Descripción:", self.description_input)
        info_layout.addRow("Autor:", self.author_input)
        
        layout.addWidget(info_group)
        
        # Configuración de página
        page_group = QGroupBox("Configuración de Página")
        page_layout = QFormLayout(page_group)
        
        self.page_format_combo = QComboBox()
        self.page_format_combo.addItems(["A4", "Letter", "Legal", "A3"])
        
        self.orientation_combo = QComboBox()
        self.orientation_combo.addItems(["portrait", "landscape"])
        
        page_layout.addRow("Formato:", self.page_format_combo)
        page_layout.addRow("Orientación:", self.orientation_combo)
        
        layout.addWidget(page_group)
        
        layout.addStretch()
        
        return widget
    
    def _create_header_footer_tab(self) -> QWidget:
        """Crea la pestaña de encabezado y pie de página"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Configuración de encabezado
        header_group = QGroupBox("Encabezado")
        header_layout = QFormLayout(header_group)
        
        self.header_enabled_check = QCheckBox("Habilitar encabezado")
        self.header_title_input = QLineEdit()
        self.header_subtitle_input = QLineEdit()
        
        # Logo
        logo_layout = QHBoxLayout()
        self.logo_path_input = QLineEdit()
        self.browse_logo_btn = QPushButton("Examinar...")
        logo_layout.addWidget(self.logo_path_input)
        logo_layout.addWidget(self.browse_logo_btn)
        
        self.logo_position_combo = QComboBox()
        self.logo_position_combo.addItems(["left", "center", "right"])
        
        header_layout.addRow("", self.header_enabled_check)
        header_layout.addRow("Título:", self.header_title_input)
        header_layout.addRow("Subtítulo:", self.header_subtitle_input)
        header_layout.addRow("Logo:", logo_layout)
        header_layout.addRow("Posición del logo:", self.logo_position_combo)
        
        layout.addWidget(header_group)
        
        # Configuración de pie de página
        footer_group = QGroupBox("Pie de Página")
        footer_layout = QFormLayout(footer_group)
        
        self.footer_enabled_check = QCheckBox("Habilitar pie de página")
        self.footer_text_input = QTextEdit()
        self.footer_text_input.setMaximumHeight(60)
        self.show_organization_check = QCheckBox("Mostrar organización")
        self.show_signature_check = QCheckBox("Mostrar línea de firma")
        
        footer_layout.addRow("", self.footer_enabled_check)
        footer_layout.addRow("Texto:", self.footer_text_input)
        footer_layout.addRow("", self.show_organization_check)
        footer_layout.addRow("", self.show_signature_check)
        
        layout.addWidget(footer_group)
        
        layout.addStretch()
        
        return widget
    
    def _create_sections_tab(self) -> QWidget:
        """Crea la pestaña de configuración de secciones"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Título
        title = QLabel("Configuración de Secciones del Reporte")
        title.setFont(QFont("Arial", 12, QFont.Bold))
        layout.addWidget(title)
        
        # Lista de secciones
        self.sections_list = QListWidget()
        self.sections_list.setDragDropMode(QListWidget.InternalMove)
        layout.addWidget(self.sections_list)
        
        # Botones de sección
        sections_buttons = QHBoxLayout()
        
        self.add_section_btn = QPushButton("Agregar Sección")
        self.edit_section_btn = QPushButton("Editar Sección")
        self.remove_section_btn = QPushButton("Eliminar Sección")
        self.move_up_btn = QPushButton("↑")
        self.move_down_btn = QPushButton("↓")
        
        sections_buttons.addWidget(self.add_section_btn)
        sections_buttons.addWidget(self.edit_section_btn)
        sections_buttons.addWidget(self.remove_section_btn)
        sections_buttons.addStretch()
        sections_buttons.addWidget(self.move_up_btn)
        sections_buttons.addWidget(self.move_down_btn)
        
        layout.addLayout(sections_buttons)
        
        return widget
    
    def _create_styles_tab(self) -> QWidget:
        """Crea la pestaña de estilos CSS"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Título
        title = QLabel("Estilos CSS Personalizados")
        title.setFont(QFont("Arial", 12, QFont.Bold))
        layout.addWidget(title)
        
        # Editor CSS
        self.css_editor = QTextEdit()
        self.css_editor.setFont(QFont("Courier New", 10))
        layout.addWidget(self.css_editor)
        
        # Botones de estilos
        styles_buttons = QHBoxLayout()
        
        self.reset_styles_btn = QPushButton("Restablecer por Defecto")
        self.load_css_btn = QPushButton("Cargar CSS...")
        self.save_css_btn = QPushButton("Guardar CSS...")
        
        styles_buttons.addWidget(self.reset_styles_btn)
        styles_buttons.addStretch()
        styles_buttons.addWidget(self.load_css_btn)
        styles_buttons.addWidget(self.save_css_btn)
        
        layout.addLayout(styles_buttons)
        
        return widget
    
    def _connect_signals(self):
        """Conecta las señales de la interfaz"""
        # Botones principales
        self.new_template_btn.clicked.connect(self._new_template)
        self.load_template_btn.clicked.connect(self._load_template)
        self.save_template_btn.clicked.connect(self._save_template)
        self.save_as_template_btn.clicked.connect(self._save_template_as)
        
        # Vista previa
        self.refresh_preview_btn.clicked.connect(self._update_preview)
        self.export_preview_btn.clicked.connect(self._export_preview)
        
        # Encabezado/pie
        self.browse_logo_btn.clicked.connect(self._browse_logo)
        
        # Secciones
        self.add_section_btn.clicked.connect(self._add_section)
        self.edit_section_btn.clicked.connect(self._edit_section)
        self.remove_section_btn.clicked.connect(self._remove_section)
        self.move_up_btn.clicked.connect(self._move_section_up)
        self.move_down_btn.clicked.connect(self._move_section_down)
        
        # Estilos
        self.reset_styles_btn.clicked.connect(self._reset_styles)
        self.load_css_btn.clicked.connect(self._load_css)
        self.save_css_btn.clicked.connect(self._save_css)
        
        # Botones de diálogo
        self.apply_btn.clicked.connect(self._apply_changes)
        self.cancel_btn.clicked.connect(self.reject)
        self.ok_btn.clicked.connect(self._accept_changes)
        
        # Cambios automáticos
        self.name_input.textChanged.connect(self._on_template_changed)
        self.css_editor.textChanged.connect(self._on_template_changed)
    
    def _load_template_data(self):
        """Carga los datos de la plantilla en la interfaz"""
        # Información general
        self.name_input.setText(self.template.name)
        self.description_input.setPlainText(self.template.description)
        self.author_input.setText(self.template.author)
        
        # Configuración de página
        self.page_format_combo.setCurrentText(self.template.page_config.get('format', 'A4'))
        self.orientation_combo.setCurrentText(self.template.page_config.get('orientation', 'portrait'))
        
        # Encabezado
        header_config = self.template.header_config
        self.header_enabled_check.setChecked(header_config.get('enabled', True))
        self.header_title_input.setText(header_config.get('title', ''))
        self.header_subtitle_input.setText(header_config.get('subtitle', ''))
        self.logo_path_input.setText(header_config.get('logo_path', ''))
        self.logo_position_combo.setCurrentText(header_config.get('logo_position', 'left'))
        
        # Pie de página
        footer_config = self.template.footer_config
        self.footer_enabled_check.setChecked(footer_config.get('enabled', True))
        self.footer_text_input.setPlainText(footer_config.get('text', ''))
        self.show_organization_check.setChecked(footer_config.get('show_organization', True))
        self.show_signature_check.setChecked(footer_config.get('show_signature_line', False))
        
        # Secciones
        self._load_sections()
        
        # Estilos CSS
        self.css_editor.setPlainText(self.template.css_styles)
        
        # Cargar plantillas disponibles
        self._load_available_templates()
        
        # Actualizar vista previa
        self._update_preview()
    
    def _load_sections(self):
        """Carga las secciones en la lista"""
        self.sections_list.clear()
        
        # Ordenar secciones por orden
        sorted_sections = sorted(self.template.sections, key=lambda s: s.order)
        
        for section in sorted_sections:
            item = QListWidgetItem()
            item.setText(f"{'✓' if section.enabled else '✗'} {section.title}")
            item.setData(Qt.UserRole, section)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if section.enabled else Qt.Unchecked)
            self.sections_list.addItem(item)
    
    def _load_available_templates(self):
        """Carga las plantillas disponibles"""
        self.template_combo.clear()
        self.template_combo.addItem("Nueva Plantilla")
        
        if self.templates_dir.exists():
            for template_file in self.templates_dir.glob("*.json"):
                self.template_combo.addItem(template_file.stem)
    
    def _new_template(self):
        """Crea una nueva plantilla"""
        self.template = ReportTemplate()
        self._load_template_data()
    
    def _load_template(self):
        """Carga una plantilla existente"""
        template_name = self.template_combo.currentText()
        if template_name == "Nueva Plantilla":
            return
        
        template_path = self.templates_dir / f"{template_name}.json"
        if template_path.exists():
            try:
                with open(template_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.template = ReportTemplate.from_dict(data)
                self._load_template_data()
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Error cargando plantilla: {e}")
    
    def _save_template(self):
        """Guarda la plantilla actual"""
        if not self.template.name or self.template.name == "Nueva Plantilla":
            self._save_template_as()
            return
        
        self._update_template_from_ui()
        template_path = self.templates_dir / f"{self.template.name}.json"
        
        try:
            with open(template_path, 'w', encoding='utf-8') as f:
                json.dump(self.template.to_dict(), f, indent=2, ensure_ascii=False)
            
            QMessageBox.information(self, "Éxito", "Plantilla guardada correctamente")
            self.template_saved.emit(self.template.name)
            
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Error guardando plantilla: {e}")
    
    def _save_template_as(self):
        """Guarda la plantilla con un nuevo nombre"""
        name, ok = QInputDialog.getText(
            self, "Guardar Plantilla Como", 
            "Nombre de la plantilla:", 
            text=self.template.name
        )
        
        if ok and name:
            self.template.name = name
            self._save_template()
    
    def _update_template_from_ui(self):
        """Actualiza la plantilla con los datos de la interfaz"""
        # Información general
        self.template.name = self.name_input.text()
        self.template.description = self.description_input.toPlainText()
        self.template.author = self.author_input.text()
        self.template.modified_date = datetime.now()
        
        # Configuración de página
        self.template.page_config.update({
            'format': self.page_format_combo.currentText(),
            'orientation': self.orientation_combo.currentText()
        })
        
        # Encabezado
        self.template.header_config.update({
            'enabled': self.header_enabled_check.isChecked(),
            'title': self.header_title_input.text(),
            'subtitle': self.header_subtitle_input.text(),
            'logo_path': self.logo_path_input.text(),
            'logo_position': self.logo_position_combo.currentText()
        })
        
        # Pie de página
        self.template.footer_config.update({
            'enabled': self.footer_enabled_check.isChecked(),
            'text': self.footer_text_input.toPlainText(),
            'show_organization': self.show_organization_check.isChecked(),
            'show_signature_line': self.show_signature_check.isChecked()
        })
        
        # Secciones
        for i in range(self.sections_list.count()):
            item = self.sections_list.item(i)
            section = item.data(Qt.UserRole)
            section.enabled = item.checkState() == Qt.Checked
            section.order = i
        
        # Estilos CSS
        self.template.css_styles = self.css_editor.toPlainText()
    
    def _browse_logo(self):
        """Examina archivos de logo"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar Logo",
            "", "Imágenes (*.png *.jpg *.jpeg *.svg *.bmp)"
        )
        
        if file_path:
            self.logo_path_input.setText(file_path)
    
    def _add_section(self):
        """Agrega una nueva sección"""
        name, ok = QInputDialog.getText(self, "Nueva Sección", "Nombre de la sección:")
        if ok and name:
            section = TemplateSection(
                name.lower().replace(' ', '_'),
                name,
                True,
                len(self.template.sections)
            )
            self.template.sections.append(section)
            self._load_sections()
    
    def _edit_section(self):
        """Edita la sección seleccionada"""
        current_item = self.sections_list.currentItem()
        if current_item:
            section = current_item.data(Qt.UserRole)
            new_title, ok = QInputDialog.getText(
                self, "Editar Sección", 
                "Título de la sección:", 
                text=section.title
            )
            if ok and new_title:
                section.title = new_title
                self._load_sections()
    
    def _remove_section(self):
        """Elimina la sección seleccionada"""
        current_item = self.sections_list.currentItem()
        if current_item:
            section = current_item.data(Qt.UserRole)
            reply = QMessageBox.question(
                self, "Confirmar", 
                f"¿Eliminar la sección '{section.title}'?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self.template.sections.remove(section)
                self._load_sections()
    
    def _move_section_up(self):
        """Mueve la sección hacia arriba"""
        current_row = self.sections_list.currentRow()
        if current_row > 0:
            item = self.sections_list.takeItem(current_row)
            self.sections_list.insertItem(current_row - 1, item)
            self.sections_list.setCurrentRow(current_row - 1)
    
    def _move_section_down(self):
        """Mueve la sección hacia abajo"""
        current_row = self.sections_list.currentRow()
        if current_row < self.sections_list.count() - 1:
            item = self.sections_list.takeItem(current_row)
            self.sections_list.insertItem(current_row + 1, item)
            self.sections_list.setCurrentRow(current_row + 1)
    
    def _reset_styles(self):
        """Restablece los estilos por defecto"""
        self.css_editor.setPlainText(ReportTemplate()._create_default_styles())
    
    def _load_css(self):
        """Carga estilos CSS desde archivo"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Cargar CSS", "", "Archivos CSS (*.css)"
        )
        
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    css_content = f.read()
                self.css_editor.setPlainText(css_content)
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Error cargando CSS: {e}")
    
    def _save_css(self):
        """Guarda los estilos CSS a archivo"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Guardar CSS", "", "Archivos CSS (*.css)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(self.css_editor.toPlainText())
                QMessageBox.information(self, "Éxito", "CSS guardado correctamente")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Error guardando CSS: {e}")
    
    def _update_preview(self):
        """Actualiza la vista previa del reporte"""
        self._update_template_from_ui()
        
        # Generar HTML de ejemplo
        sample_html = self._generate_sample_html()
        self.preview_web.setHtml(sample_html)
    
    def _generate_sample_html(self) -> str:
        """Genera HTML de ejemplo para la vista previa"""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>{self.template.name}</title>
            <style>
                {self.template.css_styles}
            </style>
        </head>
        <body>
        """
        
        # Encabezado
        if self.template.header_config.get('enabled'):
            html += self._generate_header_html()
        
        # Secciones habilitadas
        enabled_sections = [s for s in self.template.sections if s.enabled]
        enabled_sections.sort(key=lambda s: s.order)
        
        for section in enabled_sections:
            html += f"""
            <div class="section">
                <h1>{section.title}</h1>
                <p>Contenido de ejemplo para la sección {section.title}. 
                Este es un texto de muestra que demuestra cómo se verá 
                el contenido en esta sección del reporte.</p>
            </div>
            """
        
        # Pie de página
        if self.template.footer_config.get('enabled'):
            html += self._generate_footer_html()
        
        html += """
        </body>
        </html>
        """
        
        return html
    
    def _generate_header_html(self) -> str:
        """Genera HTML del encabezado"""
        header_config = self.template.header_config
        
        html = '<div class="header">'
        
        # Logo
        if header_config.get('logo_path'):
            logo_class = f"logo logo-{header_config.get('logo_position', 'left')}"
            html += f'<img src="{header_config["logo_path"]}" class="{logo_class}" alt="Logo">'
        
        # Texto del encabezado
        html += '<div class="header-text">'
        if header_config.get('title'):
            html += f'<h1 class="title">{header_config["title"]}</h1>'
        if header_config.get('subtitle'):
            html += f'<p class="subtitle">{header_config["subtitle"]}</p>'
        html += '</div>'
        
        html += '</div>'
        
        return html
    
    def _generate_footer_html(self) -> str:
        """Genera HTML del pie de página"""
        footer_config = self.template.footer_config
        
        html = '<div class="footer">'
        
        if footer_config.get('text'):
            html += f'<p>{footer_config["text"]}</p>'
        
        if footer_config.get('show_organization'):
            html += '<p>SIGeC-Balística - Sistema de Evaluación Automatizada</p>'
        
        if footer_config.get('show_signature_line'):
            html += '<div style="margin-top: 40px; border-top: 1px solid #000; width: 200px;"></div>'
            html += '<p>Firma del Perito</p>'
        
        html += '</div>'
        
        return html
    
    def _export_preview(self):
        """Exporta la vista previa a archivo HTML"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Exportar Vista Previa", 
            f"{self.template.name}_preview.html",
            "Archivos HTML (*.html)"
        )
        
        if file_path:
            try:
                html_content = self._generate_sample_html()
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(html_content)
                QMessageBox.information(self, "Éxito", "Vista previa exportada correctamente")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Error exportando vista previa: {e}")
    
    def _on_template_changed(self):
        """Maneja cambios en la plantilla"""
        # Actualizar vista previa automáticamente después de un breve delay
        if hasattr(self, '_preview_timer'):
            self._preview_timer.stop()
        
        self._preview_timer = QTimer()
        self._preview_timer.setSingleShot(True)
        self._preview_timer.timeout.connect(self._update_preview)
        self._preview_timer.start(1000)  # 1 segundo de delay
    
    def _apply_changes(self):
        """Aplica los cambios sin cerrar el diálogo"""
        self._update_template_from_ui()
        self._update_preview()
    
    def _accept_changes(self):
        """Acepta los cambios y cierra el diálogo"""
        self._update_template_from_ui()
        self.accept()
    
    def get_template(self) -> ReportTemplate:
        """Obtiene la plantilla configurada"""
        self._update_template_from_ui()
        return self.template

# Importación tardía para evitar problemas circulares
from PyQt5.QtWidgets import QInputDialog