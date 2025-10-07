#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generador de Reportes HTML Interactivos - SIGeC-Balística
========================================================

Sistema avanzado para generar reportes HTML interactivos con:
- Imágenes con zoom y comparación lado a lado
- Gráficos interactivos con tooltips y detalles
- Tablas ordenables y filtrables
- Navegación fluida entre secciones
- Exportación a PDF manteniendo interactividad

Autor: SIGeC-BalisticaTeam
Fecha: Octubre 2025
"""

import os
import json
import base64
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path
import uuid

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QProgressBar, QTextEdit, QGroupBox, QCheckBox, QSpinBox,
    QComboBox, QFileDialog, QMessageBox, QSplitter, QFrame
)
from PyQt5.QtCore import Qt, pyqtSignal, QThread, QTimer
from PyQt5.QtGui import QFont, QPixmap
from PyQt5.QtWebEngineWidgets import QWebEngineView

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.offline import plot
    import plotly.io as pio
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from .report_template_editor import ReportTemplate

logger = logging.getLogger(__name__)

class InteractiveReportData:
    """Estructura de datos para reportes interactivos"""
    
    def __init__(self):
        self.metadata = {}
        self.images = []  # Lista de imágenes con metadatos
        self.charts = []  # Lista de gráficos interactivos
        self.tables = []  # Lista de tablas de datos
        self.comparisons = []  # Lista de comparaciones
        self.analysis_results = {}
        self.quality_metrics = {}
        self.nist_compliance = {}
        
    def add_image(self, path: str, title: str, description: str = "", 
                  metadata: Dict = None, comparison_group: str = None):
        """Agrega una imagen al reporte"""
        image_data = {
            'id': str(uuid.uuid4()),
            'path': path,
            'title': title,
            'description': description,
            'metadata': metadata or {},
            'comparison_group': comparison_group,
            'timestamp': datetime.now().isoformat()
        }
        self.images.append(image_data)
        return image_data['id']
    
    def add_chart(self, chart_type: str, data: Dict, title: str, 
                  description: str = "", interactive: bool = True):
        """Agrega un gráfico al reporte"""
        chart_data = {
            'id': str(uuid.uuid4()),
            'type': chart_type,
            'data': data,
            'title': title,
            'description': description,
            'interactive': interactive,
            'timestamp': datetime.now().isoformat()
        }
        self.charts.append(chart_data)
        return chart_data['id']
    
    def add_table(self, data: List[Dict], title: str, description: str = "",
                  sortable: bool = True, filterable: bool = True):
        """Agrega una tabla al reporte"""
        table_data = {
            'id': str(uuid.uuid4()),
            'data': data,
            'title': title,
            'description': description,
            'sortable': sortable,
            'filterable': filterable,
            'timestamp': datetime.now().isoformat()
        }
        self.tables.append(table_data)
        return table_data['id']

class InteractiveHTMLGenerator:
    """Generador de HTML interactivo"""
    
    def __init__(self, template: ReportTemplate):
        self.template = template
        self.report_data = InteractiveReportData()
        
    def generate_html(self, output_path: str = None) -> str:
        """Genera el HTML completo del reporte interactivo"""
        html_content = self._generate_html_structure()
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
        
        return html_content
    
    def _generate_html_structure(self) -> str:
        """Genera la estructura HTML completa"""
        html = f"""
        <!DOCTYPE html>
        <html lang="es">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{self.template.name} - Reporte Interactivo</title>
            
            <!-- Estilos CSS -->
            <style>
                {self._generate_css_styles()}
            </style>
            
            <!-- Librerías JavaScript -->
            {self._generate_javascript_libraries()}
        </head>
        <body>
            <!-- Navegación -->
            {self._generate_navigation()}
            
            <!-- Contenido principal -->
            <div class="main-content">
                {self._generate_header()}
                {self._generate_sections()}
                {self._generate_footer()}
            </div>
            
            <!-- Modales -->
            {self._generate_modals()}
            
            <!-- Scripts JavaScript -->
            <script>
                {self._generate_javascript_code()}
            </script>
        </body>
        </html>
        """
        
        return html
    
    def _generate_css_styles(self) -> str:
        """Genera los estilos CSS completos"""
        base_styles = self.template.css_styles
        
        interactive_styles = """
        /* Estilos para funcionalidades interactivas */
        
        /* Navegación lateral */
        .sidebar {
            position: fixed;
            left: 0;
            top: 0;
            width: 250px;
            height: 100vh;
            background: #2c3e50;
            color: white;
            padding: 20px;
            box-sizing: border-box;
            overflow-y: auto;
            z-index: 1000;
            transform: translateX(-100%);
            transition: transform 0.3s ease;
        }
        
        .sidebar.open {
            transform: translateX(0);
        }
        
        .sidebar h3 {
            margin-top: 0;
            color: #ecf0f1;
            border-bottom: 1px solid #34495e;
            padding-bottom: 10px;
        }
        
        .sidebar ul {
            list-style: none;
            padding: 0;
        }
        
        .sidebar li {
            margin: 10px 0;
        }
        
        .sidebar a {
            color: #bdc3c7;
            text-decoration: none;
            display: block;
            padding: 8px 12px;
            border-radius: 4px;
            transition: background-color 0.3s;
        }
        
        .sidebar a:hover,
        .sidebar a.active {
            background-color: #34495e;
            color: #ecf0f1;
        }
        
        /* Botón de navegación */
        .nav-toggle {
            position: fixed;
            top: 20px;
            left: 20px;
            z-index: 1001;
            background: #3498db;
            color: white;
            border: none;
            padding: 10px 15px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
        }
        
        .nav-toggle:hover {
            background: #2980b9;
        }
        
        /* Contenido principal */
        .main-content {
            margin-left: 0;
            transition: margin-left 0.3s ease;
            min-height: 100vh;
        }
        
        .main-content.shifted {
            margin-left: 250px;
        }
        
        /* Imágenes interactivas */
        .interactive-image {
            position: relative;
            display: inline-block;
            cursor: zoom-in;
            border: 2px solid transparent;
            transition: border-color 0.3s;
        }
        
        .interactive-image:hover {
            border-color: #3498db;
        }
        
        .interactive-image img {
            max-width: 100%;
            height: auto;
            display: block;
        }
        
        .image-overlay {
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0, 0, 0, 0.7);
            color: white;
            display: flex;
            align-items: center;
            justify-content: center;
            opacity: 0;
            transition: opacity 0.3s;
            font-size: 18px;
        }
        
        .interactive-image:hover .image-overlay {
            opacity: 1;
        }
        
        /* Comparación de imágenes */
        .image-comparison {
            display: flex;
            gap: 20px;
            margin: 20px 0;
        }
        
        .comparison-item {
            flex: 1;
            text-align: center;
        }
        
        .comparison-item h4 {
            margin-bottom: 10px;
            color: #2c3e50;
        }
        
        /* Tablas interactivas */
        .interactive-table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: white;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        }
        
        .interactive-table th {
            background: #34495e;
            color: white;
            padding: 12px;
            text-align: left;
            cursor: pointer;
            user-select: none;
            position: relative;
        }
        
        .interactive-table th:hover {
            background: #2c3e50;
        }
        
        .interactive-table th.sortable::after {
            content: '↕';
            position: absolute;
            right: 8px;
            opacity: 0.5;
        }
        
        .interactive-table th.sort-asc::after {
            content: '↑';
            opacity: 1;
        }
        
        .interactive-table th.sort-desc::after {
            content: '↓';
            opacity: 1;
        }
        
        .interactive-table td {
            padding: 10px 12px;
            border-bottom: 1px solid #ecf0f1;
        }
        
        .interactive-table tr:hover {
            background: #f8f9fa;
        }
        
        /* Filtros de tabla */
        .table-filters {
            margin-bottom: 15px;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 5px;
        }
        
        .filter-input {
            padding: 8px 12px;
            border: 1px solid #bdc3c7;
            border-radius: 4px;
            margin-right: 10px;
            font-size: 14px;
        }
        
        /* Gráficos interactivos */
        .chart-container {
            margin: 20px 0;
            padding: 20px;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        }
        
        .chart-title {
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 15px;
            color: #2c3e50;
        }
        
        .chart-description {
            color: #7f8c8d;
            margin-bottom: 20px;
            font-style: italic;
        }
        
        /* Modales */
        .modal {
            display: none;
            position: fixed;
            z-index: 2000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0, 0, 0, 0.9);
        }
        
        .modal-content {
            position: relative;
            margin: auto;
            padding: 20px;
            width: 90%;
            max-width: 1200px;
            top: 50%;
            transform: translateY(-50%);
            background: white;
            border-radius: 8px;
        }
        
        .modal-image {
            width: 100%;
            height: auto;
            max-height: 80vh;
            object-fit: contain;
        }
        
        .close {
            position: absolute;
            top: 15px;
            right: 25px;
            color: #aaa;
            font-size: 28px;
            font-weight: bold;
            cursor: pointer;
        }
        
        .close:hover {
            color: #000;
        }
        
        /* Controles de zoom */
        .zoom-controls {
            position: absolute;
            top: 20px;
            left: 20px;
            display: flex;
            gap: 10px;
        }
        
        .zoom-btn {
            background: #3498db;
            color: white;
            border: none;
            padding: 8px 12px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
        }
        
        .zoom-btn:hover {
            background: #2980b9;
        }
        
        /* Tooltips */
        .tooltip {
            position: absolute;
            background: rgba(0, 0, 0, 0.8);
            color: white;
            padding: 8px 12px;
            border-radius: 4px;
            font-size: 12px;
            pointer-events: none;
            z-index: 1500;
            opacity: 0;
            transition: opacity 0.3s;
        }
        
        .tooltip.show {
            opacity: 1;
        }
        
        /* Indicadores de progreso */
        .progress-indicator {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 3px;
            background: #ecf0f1;
            z-index: 1500;
        }
        
        .progress-bar {
            height: 100%;
            background: #3498db;
            width: 0%;
            transition: width 0.3s ease;
        }
        
        /* Responsive */
        @media (max-width: 768px) {
            .sidebar {
                width: 100%;
                transform: translateX(-100%);
            }
            
            .main-content.shifted {
                margin-left: 0;
            }
            
            .image-comparison {
                flex-direction: column;
            }
            
            .modal-content {
                width: 95%;
                padding: 10px;
            }
        }
        
        /* Animaciones */
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .fade-in {
            animation: fadeIn 0.6s ease-out;
        }
        
        /* Estilos de impresión */
        @media print {
            .sidebar,
            .nav-toggle,
            .zoom-controls,
            .table-filters {
                display: none !important;
            }
            
            .main-content {
                margin-left: 0 !important;
            }
            
            .modal {
                display: none !important;
            }
            
            .interactive-table {
                box-shadow: none;
            }
        }
        """
        
        return base_styles + interactive_styles
    
    def _generate_javascript_libraries(self) -> str:
        """Genera las referencias a librerías JavaScript"""
        return """
        <!-- Chart.js para gráficos -->
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        
        <!-- Plotly.js para gráficos interactivos -->
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        
        <!-- DataTables para tablas interactivas -->
        <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
        <script src="https://cdn.datatables.net/1.11.5/js/jquery.dataTables.min.js"></script>
        <link rel="stylesheet" href="https://cdn.datatables.net/1.11.5/css/jquery.dataTables.min.css">
        
        <!-- Lightbox para imágenes -->
        <script src="https://cdnjs.cloudflare.com/ajax/libs/lightbox2/2.11.3/js/lightbox.min.js"></script>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/lightbox2/2.11.3/css/lightbox.min.css">
        
        <!-- Hammer.js para gestos táctiles -->
        <script src="https://cdnjs.cloudflare.com/ajax/libs/hammer.js/2.0.8/hammer.min.js"></script>
        """
    
    def _generate_navigation(self) -> str:
        """Genera la navegación lateral"""
        nav_html = """
        <button class="nav-toggle" onclick="toggleSidebar()">☰</button>
        
        <nav class="sidebar" id="sidebar">
            <h3>Navegación</h3>
            <ul>
        """
        
        # Agregar enlaces a secciones habilitadas
        enabled_sections = [s for s in self.template.sections if s.enabled]
        enabled_sections.sort(key=lambda s: s.order)
        
        for section in enabled_sections:
            nav_html += f"""
                <li><a href="#{section.name}" onclick="scrollToSection('{section.name}')">{section.title}</a></li>
            """
        
        nav_html += """
            </ul>
            
            <h3>Herramientas</h3>
            <ul>
                <li><a href="#" onclick="exportToPDF()">Exportar PDF</a></li>
                <li><a href="#" onclick="printReport()">Imprimir</a></li>
                <li><a href="#" onclick="toggleFullscreen()">Pantalla Completa</a></li>
            </ul>
        </nav>
        """
        
        return nav_html
    
    def _generate_header(self) -> str:
        """Genera el encabezado del reporte"""
        if not self.template.header_config.get('enabled'):
            return ""
        
        header_config = self.template.header_config
        
        header_html = '<header class="report-header fade-in">'
        
        # Logo
        if header_config.get('logo_path'):
            logo_position = header_config.get('logo_position', 'left')
            header_html += f"""
                <div class="logo-container logo-{logo_position}">
                    <img src="{self._encode_image_to_base64(header_config['logo_path'])}" 
                         alt="Logo" class="header-logo">
                </div>
            """
        
        # Título y subtítulo
        header_html += '<div class="header-text">'
        if header_config.get('title'):
            header_html += f'<h1 class="report-title">{header_config["title"]}</h1>'
        if header_config.get('subtitle'):
            header_html += f'<p class="report-subtitle">{header_config["subtitle"]}</p>'
        
        # Información adicional
        header_html += f"""
            <div class="report-info">
                <p><strong>Fecha de generación:</strong> {datetime.now().strftime('%d/%m/%Y %H:%M')}</p>
                <p><strong>Sistema:</strong> SIGeC-Balística</p>
            </div>
        """
        
        header_html += '</div></header>'
        
        return header_html
    
    def _generate_sections(self) -> str:
        """Genera todas las secciones del reporte"""
        sections_html = ""
        
        enabled_sections = [s for s in self.template.sections if s.enabled]
        enabled_sections.sort(key=lambda s: s.order)
        
        for section in enabled_sections:
            sections_html += self._generate_section(section)
        
        return sections_html
    
    def _generate_section(self, section) -> str:
        """Genera una sección específica"""
        section_html = f"""
        <section id="{section.name}" class="report-section fade-in">
            <h2 class="section-title">{section.title}</h2>
        """
        
        # Generar contenido específico según el tipo de sección
        if section.name == "metadata":
            section_html += self._generate_metadata_section()
        elif section.name == "results":
            section_html += self._generate_results_section()
        elif section.name == "analysis_details":
            section_html += self._generate_analysis_details_section()
        elif section.name == "quality_metrics":
            section_html += self._generate_quality_metrics_section()
        else:
            # Sección genérica
            section_html += f"""
                <div class="section-content">
                    <p>Contenido de la sección {section.title}.</p>
                </div>
            """
        
        section_html += "</section>"
        
        return section_html
    
    def _generate_metadata_section(self) -> str:
        """Genera la sección de metadatos"""
        metadata = self.report_data.metadata
        
        html = """
        <div class="section-content">
            <div class="metadata-grid">
        """
        
        # Tabla de metadatos básicos
        html += """
            <table class="interactive-table metadata-table">
                <thead>
                    <tr>
                        <th>Campo</th>
                        <th>Valor</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        basic_metadata = {
            "Fecha de análisis": metadata.get('analysis_date', 'No especificada'),
            "Tipo de análisis": metadata.get('analysis_type', 'No especificado'),
            "Perito responsable": metadata.get('expert', 'No especificado'),
            "Caso número": metadata.get('case_number', 'No especificado'),
            "Evidencia": metadata.get('evidence_id', 'No especificada')
        }
        
        for field, value in basic_metadata.items():
            html += f"""
                <tr>
                    <td><strong>{field}</strong></td>
                    <td>{value}</td>
                </tr>
            """
        
        html += """
                </tbody>
            </table>
        </div>
        </div>
        """
        
        return html
    
    def _generate_results_section(self) -> str:
        """Genera la sección de resultados con elementos interactivos"""
        html = """
        <div class="section-content">
        """
        
        # Imágenes interactivas
        if self.report_data.images:
            html += self._generate_interactive_images()
        
        # Gráficos interactivos
        if self.report_data.charts:
            html += self._generate_interactive_charts()
        
        # Tablas interactivas
        if self.report_data.tables:
            html += self._generate_interactive_tables()
        
        html += """
        </div>
        """
        
        return html
    
    def _generate_interactive_images(self) -> str:
        """Genera imágenes interactivas con zoom y comparación"""
        html = """
        <div class="images-section">
            <h3>Imágenes de Análisis</h3>
        """
        
        # Agrupar imágenes por grupo de comparación
        comparison_groups = {}
        standalone_images = []
        
        for image in self.report_data.images:
            if image.get('comparison_group'):
                group = image['comparison_group']
                if group not in comparison_groups:
                    comparison_groups[group] = []
                comparison_groups[group].append(image)
            else:
                standalone_images.append(image)
        
        # Generar comparaciones
        for group_name, images in comparison_groups.items():
            html += f"""
            <div class="image-comparison">
                <h4>Comparación: {group_name}</h4>
                <div class="comparison-container">
            """
            
            for image in images:
                html += f"""
                <div class="comparison-item">
                    <h5>{image['title']}</h5>
                    <div class="interactive-image" onclick="openImageModal('{image['id']}')">
                        <img src="{self._encode_image_to_base64(image['path'])}" 
                             alt="{image['title']}" 
                             id="img_{image['id']}">
                        <div class="image-overlay">
                            <span>🔍 Click para ampliar</span>
                        </div>
                    </div>
                    <p class="image-description">{image['description']}</p>
                </div>
                """
            
            html += """
                </div>
            </div>
            """
        
        # Generar imágenes independientes
        for image in standalone_images:
            html += f"""
            <div class="standalone-image">
                <h4>{image['title']}</h4>
                <div class="interactive-image" onclick="openImageModal('{image['id']}')">
                    <img src="{self._encode_image_to_base64(image['path'])}" 
                         alt="{image['title']}" 
                         id="img_{image['id']}">
                    <div class="image-overlay">
                        <span>🔍 Click para ampliar</span>
                    </div>
                </div>
                <p class="image-description">{image['description']}</p>
            </div>
            """
        
        html += """
        </div>
        """
        
        return html
    
    def _generate_interactive_charts(self) -> str:
        """Genera gráficos interactivos"""
        html = """
        <div class="charts-section">
            <h3>Gráficos de Análisis</h3>
        """
        
        for chart in self.report_data.charts:
            chart_id = f"chart_{chart['id']}"
            
            html += f"""
            <div class="chart-container">
                <div class="chart-title">{chart['title']}</div>
                <div class="chart-description">{chart['description']}</div>
                <div id="{chart_id}" class="chart-plot"></div>
            </div>
            """
        
        html += """
        </div>
        """
        
        return html
    
    def _generate_interactive_tables(self) -> str:
        """Genera tablas interactivas"""
        html = """
        <div class="tables-section">
            <h3>Datos de Análisis</h3>
        """
        
        for table in self.report_data.tables:
            table_id = f"table_{table['id']}"
            
            html += f"""
            <div class="table-container">
                <h4>{table['title']}</h4>
                <p class="table-description">{table['description']}</p>
            """
            
            # Filtros si están habilitados
            if table.get('filterable'):
                html += f"""
                <div class="table-filters">
                    <input type="text" class="filter-input" 
                           placeholder="Filtrar tabla..." 
                           onkeyup="filterTable('{table_id}', this.value)">
                </div>
                """
            
            # Tabla
            html += f"""
                <table id="{table_id}" class="interactive-table">
                    <thead>
                        <tr>
            """
            
            # Encabezados
            if table['data']:
                for key in table['data'][0].keys():
                    sortable_class = "sortable" if table.get('sortable') else ""
                    html += f"""
                        <th class="{sortable_class}" 
                            onclick="sortTable('{table_id}', '{key}')">
                            {key}
                        </th>
                    """
            
            html += """
                        </tr>
                    </thead>
                    <tbody>
            """
            
            # Filas de datos
            for row in table['data']:
                html += "<tr>"
                for value in row.values():
                    html += f"<td>{value}</td>"
                html += "</tr>"
            
            html += """
                    </tbody>
                </table>
            </div>
            """
        
        html += """
        </div>
        """
        
        return html
    
    def _generate_analysis_details_section(self) -> str:
        """Genera la sección de detalles del análisis"""
        html = """
        <div class="section-content">
            <div class="analysis-details">
                <h3>Detalles Técnicos del Análisis</h3>
        """
        
        # Información de algoritmos utilizados
        html += """
                <div class="algorithm-info">
                    <h4>Algoritmos Utilizados</h4>
                    <ul>
                        <li><strong>Detección de características:</strong> SIFT + ORB</li>
                        <li><strong>Matching:</strong> FLANN + Ratio Test</li>
                        <li><strong>Filtrado:</strong> RANSAC</li>
                        <li><strong>Scoring:</strong> Weighted similarity</li>
                    </ul>
                </div>
        """
        
        # Parámetros de configuración
        html += """
                <div class="parameters-info">
                    <h4>Parámetros de Configuración</h4>
                    <table class="interactive-table">
                        <thead>
                            <tr>
                                <th>Parámetro</th>
                                <th>Valor</th>
                                <th>Descripción</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td>Umbral de similitud</td>
                                <td>0.75</td>
                                <td>Umbral mínimo para considerar una coincidencia válida</td>
                            </tr>
                            <tr>
                                <td>Número de características</td>
                                <td>5000</td>
                                <td>Máximo número de puntos característicos a extraer</td>
                            </tr>
                            <tr>
                                <td>Ratio test</td>
                                <td>0.7</td>
                                <td>Ratio para filtrar coincidencias ambiguas</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
        """
        
        html += """
            </div>
        </div>
        """
        
        return html
    
    def _generate_quality_metrics_section(self) -> str:
        """Genera la sección de métricas de calidad"""
        html = """
        <div class="section-content">
            <div class="quality-metrics">
                <h3>Métricas de Calidad</h3>
        """
        
        # Gráfico de métricas de calidad
        html += """
                <div class="chart-container">
                    <div class="chart-title">Métricas de Calidad del Análisis</div>
                    <div id="quality_chart" class="chart-plot"></div>
                </div>
        """
        
        # Tabla de métricas detalladas
        html += """
                <div class="metrics-table">
                    <h4>Detalles de Métricas</h4>
                    <table class="interactive-table">
                        <thead>
                            <tr>
                                <th>Métrica</th>
                                <th>Valor</th>
                                <th>Estado</th>
                                <th>Descripción</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td>Calidad de imagen</td>
                                <td>85%</td>
                                <td><span class="status-good">Buena</span></td>
                                <td>Calidad general de las imágenes analizadas</td>
                            </tr>
                            <tr>
                                <td>Número de características</td>
                                <td>3247</td>
                                <td><span class="status-good">Suficiente</span></td>
                                <td>Puntos característicos detectados</td>
                            </tr>
                            <tr>
                                <td>Coincidencias válidas</td>
                                <td>156</td>
                                <td><span class="status-good">Alto</span></td>
                                <td>Coincidencias que pasaron el filtrado</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
        """
        
        html += """
            </div>
        </div>
        """
        
        return html
    
    def _generate_footer(self) -> str:
        """Genera el pie de página"""
        if not self.template.footer_config.get('enabled'):
            return ""
        
        footer_config = self.template.footer_config
        
        footer_html = '<footer class="report-footer">'
        
        if footer_config.get('text'):
            footer_html += f'<p class="footer-text">{footer_config["text"]}</p>'
        
        if footer_config.get('show_organization'):
            footer_html += '<p class="organization">SIGeC-Balística - Sistema de Evaluación Automatizada</p>'
        
        footer_html += f'<p class="generation-info">Reporte generado el {datetime.now().strftime("%d/%m/%Y a las %H:%M")}</p>'
        
        if footer_config.get('show_signature_line'):
            footer_html += """
            <div class="signature-section">
                <div class="signature-line"></div>
                <p class="signature-label">Firma del Perito</p>
            </div>
            """
        
        footer_html += '</footer>'
        
        return footer_html
    
    def _generate_modals(self) -> str:
        """Genera los modales para imágenes"""
        html = """
        <!-- Modal para imágenes -->
        <div id="imageModal" class="modal">
            <div class="modal-content">
                <span class="close" onclick="closeImageModal()">&times;</span>
                <div class="zoom-controls">
                    <button class="zoom-btn" onclick="zoomIn()">🔍+</button>
                    <button class="zoom-btn" onclick="zoomOut()">🔍-</button>
                    <button class="zoom-btn" onclick="resetZoom()">⟲</button>
                    <button class="zoom-btn" onclick="fitToScreen()">⛶</button>
                </div>
                <img id="modalImage" class="modal-image" src="" alt="">
                <div id="imageInfo" class="image-info"></div>
            </div>
        </div>
        """
        
        return html
    
    def _generate_javascript_code(self) -> str:
        """Genera el código JavaScript para interactividad"""
        js_code = """
        // Variables globales
        let currentZoom = 1;
        let isDragging = false;
        let startX, startY, scrollLeft, scrollTop;
        
        // Navegación
        function toggleSidebar() {
            const sidebar = document.getElementById('sidebar');
            const mainContent = document.querySelector('.main-content');
            
            sidebar.classList.toggle('open');
            mainContent.classList.toggle('shifted');
        }
        
        function scrollToSection(sectionId) {
            const section = document.getElementById(sectionId);
            if (section) {
                section.scrollIntoView({ behavior: 'smooth' });
                
                // Actualizar navegación activa
                document.querySelectorAll('.sidebar a').forEach(a => a.classList.remove('active'));
                document.querySelector(`a[href="#${sectionId}"]`).classList.add('active');
            }
        }
        
        // Modales de imágenes
        function openImageModal(imageId) {
            const modal = document.getElementById('imageModal');
            const modalImg = document.getElementById('modalImage');
            const img = document.getElementById(`img_${imageId}`);
            
            modal.style.display = 'block';
            modalImg.src = img.src;
            modalImg.alt = img.alt;
            
            resetZoom();
            setupImageDragging();
        }
        
        function closeImageModal() {
            document.getElementById('imageModal').style.display = 'none';
        }
        
        // Controles de zoom
        function zoomIn() {
            currentZoom *= 1.2;
            applyZoom();
        }
        
        function zoomOut() {
            currentZoom /= 1.2;
            applyZoom();
        }
        
        function resetZoom() {
            currentZoom = 1;
            applyZoom();
        }
        
        function fitToScreen() {
            const modalImg = document.getElementById('modalImage');
            const container = modalImg.parentElement;
            
            const containerWidth = container.clientWidth - 40;
            const containerHeight = container.clientHeight - 40;
            const imgWidth = modalImg.naturalWidth;
            const imgHeight = modalImg.naturalHeight;
            
            const scaleX = containerWidth / imgWidth;
            const scaleY = containerHeight / imgHeight;
            currentZoom = Math.min(scaleX, scaleY);
            
            applyZoom();
        }
        
        function applyZoom() {
            const modalImg = document.getElementById('modalImage');
            modalImg.style.transform = `scale(${currentZoom})`;
            modalImg.style.cursor = currentZoom > 1 ? 'grab' : 'zoom-in';
        }
        
        // Arrastrar imagen
        function setupImageDragging() {
            const modalImg = document.getElementById('modalImage');
            
            modalImg.addEventListener('mousedown', startDragging);
            modalImg.addEventListener('mousemove', drag);
            modalImg.addEventListener('mouseup', stopDragging);
            modalImg.addEventListener('mouseleave', stopDragging);
        }
        
        function startDragging(e) {
            if (currentZoom > 1) {
                isDragging = true;
                startX = e.pageX - e.target.offsetLeft;
                startY = e.pageY - e.target.offsetTop;
                e.target.style.cursor = 'grabbing';
            }
        }
        
        function drag(e) {
            if (!isDragging) return;
            
            e.preventDefault();
            const x = e.pageX - startX;
            const y = e.pageY - startY;
            
            e.target.style.left = x + 'px';
            e.target.style.top = y + 'px';
        }
        
        function stopDragging(e) {
            isDragging = false;
            e.target.style.cursor = currentZoom > 1 ? 'grab' : 'zoom-in';
        }
        
        // Tablas interactivas
        function sortTable(tableId, column) {
            const table = document.getElementById(tableId);
            const tbody = table.querySelector('tbody');
            const rows = Array.from(tbody.querySelectorAll('tr'));
            const headerCell = table.querySelector(`th[onclick*="${column}"]`);
            
            // Determinar dirección de ordenamiento
            let ascending = true;
            if (headerCell.classList.contains('sort-asc')) {
                ascending = false;
                headerCell.classList.remove('sort-asc');
                headerCell.classList.add('sort-desc');
            } else {
                // Limpiar otros encabezados
                table.querySelectorAll('th').forEach(th => {
                    th.classList.remove('sort-asc', 'sort-desc');
                });
                headerCell.classList.add('sort-asc');
            }
            
            // Obtener índice de columna
            const columnIndex = Array.from(headerCell.parentElement.children).indexOf(headerCell);
            
            // Ordenar filas
            rows.sort((a, b) => {
                const aValue = a.children[columnIndex].textContent.trim();
                const bValue = b.children[columnIndex].textContent.trim();
                
                // Intentar conversión numérica
                const aNum = parseFloat(aValue);
                const bNum = parseFloat(bValue);
                
                if (!isNaN(aNum) && !isNaN(bNum)) {
                    return ascending ? aNum - bNum : bNum - aNum;
                } else {
                    return ascending ? 
                        aValue.localeCompare(bValue) : 
                        bValue.localeCompare(aValue);
                }
            });
            
            // Reordenar en el DOM
            rows.forEach(row => tbody.appendChild(row));
        }
        
        function filterTable(tableId, filterValue) {
            const table = document.getElementById(tableId);
            const rows = table.querySelectorAll('tbody tr');
            const filter = filterValue.toLowerCase();
            
            rows.forEach(row => {
                const text = row.textContent.toLowerCase();
                row.style.display = text.includes(filter) ? '' : 'none';
            });
        }
        
        // Gráficos interactivos
        function initializeCharts() {
            // Inicializar gráficos de Plotly si están disponibles
            if (typeof Plotly !== 'undefined') {
                initializePlotlyCharts();
            }
            
            // Inicializar Chart.js si está disponible
            if (typeof Chart !== 'undefined') {
                initializeChartJSCharts();
            }
        }
        
        function initializePlotlyCharts() {
            // Gráfico de métricas de calidad
            const qualityData = [{
                x: ['Calidad de Imagen', 'Características', 'Coincidencias', 'Precisión'],
                y: [85, 92, 78, 88],
                type: 'bar',
                marker: {
                    color: ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']
                }
            }];
            
            const qualityLayout = {
                title: 'Métricas de Calidad (%)',
                xaxis: { title: 'Métricas' },
                yaxis: { title: 'Porcentaje' },
                showlegend: false
            };
            
            if (document.getElementById('quality_chart')) {
                Plotly.newPlot('quality_chart', qualityData, qualityLayout, {responsive: true});
            }
        }
        
        // Utilidades
        function exportToPDF() {
            window.print();
        }
        
        function printReport() {
            window.print();
        }
        
        function toggleFullscreen() {
            if (!document.fullscreenElement) {
                document.documentElement.requestFullscreen();
            } else {
                document.exitFullscreen();
            }
        }
        
        // Tooltips
        function createTooltip(element, text) {
            const tooltip = document.createElement('div');
            tooltip.className = 'tooltip';
            tooltip.textContent = text;
            document.body.appendChild(tooltip);
            
            element.addEventListener('mouseenter', (e) => {
                tooltip.style.left = e.pageX + 10 + 'px';
                tooltip.style.top = e.pageY - 30 + 'px';
                tooltip.classList.add('show');
            });
            
            element.addEventListener('mouseleave', () => {
                tooltip.classList.remove('show');
            });
            
            element.addEventListener('mousemove', (e) => {
                tooltip.style.left = e.pageX + 10 + 'px';
                tooltip.style.top = e.pageY - 30 + 'px';
            });
        }
        
        // Inicialización
        document.addEventListener('DOMContentLoaded', function() {
            // Inicializar gráficos
            initializeCharts();
            
            // Configurar tooltips
            document.querySelectorAll('[data-tooltip]').forEach(element => {
                createTooltip(element, element.getAttribute('data-tooltip'));
            });
            
            // Configurar gestos táctiles si Hammer.js está disponible
            if (typeof Hammer !== 'undefined') {
                setupTouchGestures();
            }
            
            // Animaciones de entrada
            const observer = new IntersectionObserver((entries) => {
                entries.forEach(entry => {
                    if (entry.isIntersecting) {
                        entry.target.classList.add('fade-in');
                    }
                });
            });
            
            document.querySelectorAll('.report-section').forEach(section => {
                observer.observe(section);
            });
        });
        
        function setupTouchGestures() {
            const modalImg = document.getElementById('modalImage');
            if (modalImg) {
                const hammer = new Hammer(modalImg);
                
                hammer.get('pinch').set({ enable: true });
                hammer.on('pinch', function(e) {
                    currentZoom *= e.scale;
                    applyZoom();
                });
            }
        }
        
        // Cerrar modal con Escape
        document.addEventListener('keydown', function(e) {
            if (e.key === 'Escape') {
                closeImageModal();
            }
        });
        """
        
        return js_code
    
    def _encode_image_to_base64(self, image_path: str) -> str:
        """Codifica una imagen a base64 para embeber en HTML"""
        try:
            if not os.path.exists(image_path):
                return ""
            
            with open(image_path, 'rb') as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                
            # Determinar tipo MIME
            ext = Path(image_path).suffix.lower()
            mime_types = {
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.png': 'image/png',
                '.gif': 'image/gif',
                '.svg': 'image/svg+xml',
                '.bmp': 'image/bmp'
            }
            
            mime_type = mime_types.get(ext, 'image/jpeg')
            
            return f"data:{mime_type};base64,{encoded_string}"
            
        except Exception as e:
            logger.error(f"Error codificando imagen {image_path}: {e}")
            return ""
    
    def set_report_data(self, report_data: InteractiveReportData):
        """Establece los datos del reporte"""
        self.report_data = report_data
    
    def add_sample_data(self):
        """Agrega datos de ejemplo para demostración"""
        # Imágenes de ejemplo
        self.report_data.add_image(
            "/path/to/evidence1.jpg",
            "Evidencia Principal",
            "Imagen de la bala evidencia encontrada en la escena",
            {"resolution": "2048x1536", "format": "JPEG"},
            "comparison_1"
        )
        
        self.report_data.add_image(
            "/path/to/reference1.jpg",
            "Muestra de Referencia",
            "Imagen de la bala de referencia del arma sospechosa",
            {"resolution": "2048x1536", "format": "JPEG"},
            "comparison_1"
        )
        
        # Gráfico de ejemplo
        chart_data = {
            "labels": ["Similitud", "Confianza", "Precisión"],
            "values": [85, 92, 78]
        }
        
        self.report_data.add_chart(
            "bar",
            chart_data,
            "Métricas de Análisis",
            "Resultados principales del análisis comparativo"
        )
        
        # Tabla de ejemplo
        table_data = [
            {"Característica": "Estrías", "Coincidencias": 15, "Calidad": "Alta"},
            {"Característica": "Campos", "Coincidencias": 8, "Calidad": "Media"},
            {"Característica": "Defectos", "Coincidencias": 3, "Calidad": "Alta"}
        ]
        
        self.report_data.add_table(
            table_data,
            "Análisis de Características",
            "Detalle de las características analizadas"
        )
        
        # Metadatos
        self.report_data.metadata = {
            "analysis_date": datetime.now().strftime("%d/%m/%Y"),
            "analysis_type": "Comparación Balística",
            "expert": "Dr. Juan Pérez",
            "case_number": "CASE-2025-001",
            "evidence_id": "EV-001"
        }

class InteractiveReportWidget(QWidget):
    """Widget para generar reportes interactivos"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.template = ReportTemplate()
        self.generator = InteractiveHTMLGenerator(self.template)
        
        self._setup_ui()
        self._connect_signals()
        
        logger.info("Widget de reportes interactivos inicializado")
    
    def _setup_ui(self):
        """Configura la interfaz de usuario"""
        layout = QVBoxLayout(self)
        
        # Título
        title = QLabel("Generador de Reportes Interactivos")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        layout.addWidget(title)
        
        # Controles
        controls_layout = QHBoxLayout()
        
        self.template_btn = QPushButton("Configurar Plantilla")
        self.preview_btn = QPushButton("Vista Previa")
        self.export_btn = QPushButton("Exportar HTML")
        
        controls_layout.addWidget(self.template_btn)
        controls_layout.addWidget(self.preview_btn)
        controls_layout.addWidget(self.export_btn)
        controls_layout.addStretch()
        
        layout.addLayout(controls_layout)
        
        # Vista previa
        self.preview_web = QWebEngineView()
        layout.addWidget(self.preview_web)
        
        # Barra de progreso
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
    
    def _connect_signals(self):
        """Conecta las señales"""
        self.template_btn.clicked.connect(self._configure_template)
        self.preview_btn.clicked.connect(self._generate_preview)
        self.export_btn.clicked.connect(self._export_html)
    
    def _configure_template(self):
        """Abre el editor de plantillas"""
        from .report_template_editor import ReportTemplateEditor
        
        editor = ReportTemplateEditor(self.template, self)
        if editor.exec_() == QDialog.Accepted:
            self.template = editor.get_template()
            self.generator = InteractiveHTMLGenerator(self.template)
    
    def _generate_preview(self):
        """Genera la vista previa del reporte"""
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminado
        
        try:
            # Agregar datos de ejemplo
            self.generator.add_sample_data()
            
            # Generar HTML
            html_content = self.generator.generate_html()
            
            # Mostrar en vista previa
            self.preview_web.setHtml(html_content)
            
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Error generando vista previa: {e}")
        finally:
            self.progress_bar.setVisible(False)
    
    def _export_html(self):
        """Exporta el reporte a archivo HTML"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Exportar Reporte HTML",
            f"reporte_interactivo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
            "Archivos HTML (*.html)"
        )
        
        if file_path:
            try:
                self.progress_bar.setVisible(True)
                self.progress_bar.setRange(0, 0)
                
                # Agregar datos de ejemplo si no hay datos
                if not self.generator.report_data.images:
                    self.generator.add_sample_data()
                
                # Generar y guardar HTML
                self.generator.generate_html(file_path)
                
                QMessageBox.information(
                    self, "Éxito", 
                    f"Reporte exportado correctamente a:\n{file_path}"
                )
                
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Error exportando reporte: {e}")
            finally:
                self.progress_bar.setVisible(False)
    
    def set_report_data(self, report_data: InteractiveReportData):
        """Establece los datos del reporte"""
        self.generator.set_report_data(report_data)
    
    def get_generator(self) -> InteractiveHTMLGenerator:
        """Obtiene el generador de reportes"""
        return self.generator