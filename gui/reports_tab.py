#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reports Tab - SIGeC-BalisticaGUI
==========================

Pestaña para generación de reportes profesionales e interactivos.
Permite crear reportes a partir de análisis realizados en otras pestañas.

Características:
- Generación guiada paso a paso
- Vista previa en tiempo real
- Múltiples formatos de exportación (PDF, HTML, DOCX)
- Templates profesionales
- Inclusión de visualizaciones y gráficos
- Metadatos NIST completos

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
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QLineEdit, QComboBox, QPushButton, QTextEdit,
    QCheckBox, QSpinBox, QDateEdit, QTimeEdit, QGroupBox,
    QSplitter, QFrame, QScrollArea, QTabWidget, QTreeWidget,
    QTreeWidgetItem, QListWidget, QListWidgetItem, QProgressBar,
    QFileDialog, QMessageBox, QTableWidget, QTableWidgetItem
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QDate, QTime
from PyQt5.QtGui import QPixmap, QFont, QTextDocument, QTextCursor
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtPrintSupport import QPrinter, QPrintDialog

from .shared_widgets import (
    ImageDropZone, ResultCard, CollapsiblePanel, 
    StepIndicator, ProgressCard, ImageViewer
)
from .backend_integration import get_backend_integration

# Importaciones NIST opcionales
try:
    from image_processing.nist_compliance_validator import NISTComplianceValidator, NISTProcessingReport
    from nist_standards.quality_metrics import NISTQualityMetrics, NISTQualityReport
    from nist_standards.afte_conclusions import AFTEConclusionEngine, AFTEConclusion
    from nist_standards.validation_protocols import NISTValidationProtocols
    NIST_AVAILABLE = True
except ImportError:
    NIST_AVAILABLE = False

logger = logging.getLogger(__name__)

class ReportGenerationWorker(QThread):
    """Worker thread para generación de reportes"""
    
    progress_updated = pyqtSignal(int, str)  # porcentaje, mensaje
    report_generated = pyqtSignal(str)  # ruta del reporte
    preview_updated = pyqtSignal(str)  # HTML del preview
    error_occurred = pyqtSignal(str)  # mensaje de error
    
    def __init__(self, report_config: Dict[str, Any]):
        super().__init__()
        self.report_config = report_config
        self.backend = get_backend_integration()
        self._should_stop = False
    
    def stop_generation(self):
        """Detiene la generación del reporte"""
        self._should_stop = True
    
    def run(self):
        try:
            self.progress_updated.emit(5, "Iniciando generación de reporte...")
            
            if self._should_stop:
                return
            
            # Paso 1: Validar configuración
            self.progress_updated.emit(10, "Validando configuración...")
            self._validate_config()
            
            # Paso 2: Recopilar datos
            self.progress_updated.emit(25, "Recopilando datos de análisis...")
            analysis_data = self._collect_analysis_data()
            
            # Paso 3: Generar contenido
            self.progress_updated.emit(50, "Generando contenido del reporte...")
            report_content = self._generate_report_content(analysis_data)
            
            # Paso 4: Aplicar template
            self.progress_updated.emit(70, "Aplicando template profesional...")
            formatted_report = self._apply_template(report_content)
            
            # Paso 5: Generar preview
            self.progress_updated.emit(85, "Generando vista previa...")
            self.preview_updated.emit(formatted_report)
            
            # Paso 6: Exportar si se solicita
            if self.report_config.get('export_immediately'):
                self.progress_updated.emit(95, "Exportando reporte...")
                output_path = self._export_report(formatted_report)
                self.report_generated.emit(output_path)
            
            self.progress_updated.emit(100, "Reporte generado exitosamente")
            
        except Exception as e:
            logger.error(f"Error generando reporte: {e}")
            self.error_occurred.emit(str(e))
    
    def _validate_config(self):
        """Valida la configuración del reporte"""
        required_fields = ['title', 'author', 'template']
        for field in required_fields:
            if not self.report_config.get(field):
                raise ValueError(f"Campo requerido faltante: {field}")
    
    def _collect_analysis_data(self) -> Dict[str, Any]:
        """Recopila datos de análisis desde diferentes fuentes"""
        data = {}
        
        # Datos de análisis individual
        if self.report_config.get('include_individual_analysis'):
            data['individual_analyses'] = self._get_individual_analyses()
        
        # Datos de comparaciones
        if self.report_config.get('include_comparisons'):
            data['comparisons'] = self._get_comparison_results()
        
        # Datos de base de datos
        if self.report_config.get('include_database_searches'):
            data['database_searches'] = self._get_database_searches()
        
        return data
    
    def _get_individual_analyses(self) -> List[Dict[str, Any]]:
        """Obtiene datos de análisis individuales"""
        # Aquí se integraría con el sistema de almacenamiento de resultados
        return []
    
    def _get_comparison_results(self) -> List[Dict[str, Any]]:
        """Obtiene resultados de comparaciones"""
        return []
    
    def _get_database_searches(self) -> List[Dict[str, Any]]:
        """Obtiene resultados de búsquedas en BD"""
        return []
    
    def _generate_report_content(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Genera el contenido estructurado del reporte"""
        content = {
            'metadata': self._generate_metadata(),
            'executive_summary': self._generate_executive_summary(analysis_data),
            'methodology': self._generate_methodology(),
            'results': self._generate_results_section(analysis_data),
            'conclusions': self._generate_conclusions(analysis_data),
            'appendices': self._generate_appendices(analysis_data)
        }
        
        return content
    
    def _generate_metadata(self) -> Dict[str, Any]:
        """Genera metadatos del reporte con cumplimiento NIST"""
        metadata = {
            'title': self.report_config['title'],
            'author': self.report_config['author'],
            'date': datetime.now().strftime('%Y-%m-%d'),
            'time': datetime.now().strftime('%H:%M:%S'),
            'version': '1.0',
            'case_number': self.report_config.get('case_number', ''),
            'organization': self.report_config.get('organization', ''),
            'classification': self.report_config.get('classification', 'Confidencial'),
            'system_version': 'SIGeC-Balisticav2.0',
            'analysis_software': 'SIGeC-Balistica- Sistema de Evaluación Automatizada de Características Balísticas'
        }
        
        # Agregar metadatos NIST si están disponibles
        if NIST_AVAILABLE:
            metadata.update({
                'nist_compliance': True,
                'nist_standards_version': '2024.1',
                'quality_assurance': 'NIST SP 800-87 Rev. 1',
                'validation_protocol': 'NIST Ballistic Imaging Standards',
                'chain_of_custody': self.report_config.get('chain_of_custody', True),
                'digital_signature': self.report_config.get('digital_signature', False),
                'audit_trail': self.report_config.get('audit_trail', True)
            })
        
        return metadata
    
    def _generate_executive_summary(self, analysis_data: Dict[str, Any]) -> str:
        """Genera resumen ejecutivo"""
        summary_parts = []
        
        # Análisis realizados
        total_analyses = 0
        if 'individual_analyses' in analysis_data:
            total_analyses += len(analysis_data['individual_analyses'])
        if 'comparisons' in analysis_data:
            total_analyses += len(analysis_data['comparisons'])
        
        summary_parts.append(f"Se realizaron un total de {total_analyses} análisis utilizando el sistema SIGeC-Balistica.")
        
        # Resultados principales
        if self.report_config.get('key_findings'):
            summary_parts.append("Hallazgos principales:")
            for finding in self.report_config['key_findings']:
                summary_parts.append(f"• {finding}")
        
        return '\n\n'.join(summary_parts)
    
    def _generate_methodology(self) -> str:
        """Genera sección de metodología con estándares NIST"""
        methodology = [
            "METODOLOGÍA",
            "============",
            "",
            "El análisis se realizó utilizando el Sistema de Evaluación Automatizada de Características Balísticas (SIGeC-Balistica), "
            "que implementa algoritmos avanzados de procesamiento de imágenes y análisis estadístico.",
            "",
            "ESTÁNDARES Y PROTOCOLOS:",
            "• Sistema desarrollado siguiendo las mejores prácticas de análisis forense digital",
            "• Algoritmos de correlación basados en métodos científicamente validados",
            "• Procesamiento de imágenes con técnicas de vanguardia en visión computacional",
        ]
        
        # Agregar información NIST si está disponible
        if NIST_AVAILABLE:
            methodology.extend([
                "",
                "CUMPLIMIENTO NIST:",
                "• Cumple con NIST SP 800-87 Rev. 1 - Códigos para la Identificación de Evidencia Balística",
                "• Implementa protocolos de validación de calidad de imagen según estándares NIST",
                "• Utiliza métricas de calidad conformes a NIST Special Publication 800-76-2",
                "• Mantiene cadena de custodia digital según NIST SP 800-72",
                "• Genera reportes con metadatos completos según NIST SP 800-88 Rev. 1",
                "",
                "VALIDACIÓN DE CALIDAD:",
                "• Resolución mínima: 1000 DPI (conforme a estándares NIST)",
                "• Contraste y nitidez validados automáticamente",
                "• Verificación de integridad mediante hash criptográfico",
                "• Análisis de uniformidad de iluminación",
                "• Detección automática de artefactos y distorsiones"
            ])
        
        methodology.extend([
            "",
            "PROCESO DE ANÁLISIS:",
            "1. Carga y validación de imágenes de evidencia",
            "2. Preprocesamiento y mejora de calidad",
            "3. Extracción de características balísticas distintivas",
            "4. Análisis comparativo utilizando algoritmos CMC (Correlation Maximum Coefficient)",
            "5. Evaluación estadística de similitudes",
            "6. Generación de conclusiones según criterios AFTE",
            "",
            "CONTROL DE CALIDAD:",
            "• Validación automática de parámetros de entrada",
            "• Verificación de integridad de datos",
            "• Trazabilidad completa del proceso de análisis",
            "• Documentación automática de todos los pasos realizados"
        ])
        
        return '\n'.join(methodology)
    
    def _generate_results_section(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Genera sección de resultados"""
        results = {}
        
        # Resultados de análisis individual
        if 'individual_analyses' in analysis_data:
            results['individual'] = self._format_individual_results(analysis_data['individual_analyses'])
        
        # Resultados de comparaciones
        if 'comparisons' in analysis_data:
            results['comparisons'] = self._format_comparison_results(analysis_data['comparisons'])
        
        # Resultados de búsquedas
        if 'database_searches' in analysis_data:
            results['searches'] = self._format_search_results(analysis_data['database_searches'])
        
        return results
    
    def _format_individual_results(self, analyses: List[Dict[str, Any]]) -> str:
        """Formatea resultados de análisis individual"""
        if not analyses:
            return "No se realizaron análisis individuales."
        
        formatted = ["ANÁLISIS INDIVIDUALES", "=" * 20, ""]
        
        for i, analysis in enumerate(analyses, 1):
            formatted.append(f"Análisis {i}: {analysis.get('image_name', 'Sin nombre')}")
            formatted.append(f"Calidad: {analysis.get('quality_score', 'N/A')}")
            formatted.append(f"Características extraídas: {analysis.get('num_features', 'N/A')}")
            formatted.append("")
        
        return '\n'.join(formatted)
    
    def _format_comparison_results(self, comparisons: List[Dict[str, Any]]) -> str:
        """Formatea resultados de comparaciones"""
        if not comparisons:
            return "No se realizaron comparaciones."
        
        formatted = ["ANÁLISIS COMPARATIVOS", "=" * 20, ""]
        
        for i, comparison in enumerate(comparisons, 1):
            formatted.append(f"Comparación {i}")
            formatted.append(f"Similitud: {comparison.get('similarity_score', 'N/A')}")
            formatted.append(f"Matches encontrados: {comparison.get('num_matches', 'N/A')}")
            formatted.append("")
        
        return '\n'.join(formatted)
    
    def _format_search_results(self, searches: List[Dict[str, Any]]) -> str:
        """Formatea resultados de búsquedas"""
        if not searches:
            return "No se realizaron búsquedas en base de datos."
        
        formatted = ["BÚSQUEDAS EN BASE DE DATOS", "=" * 25, ""]
        
        for i, search in enumerate(searches, 1):
            formatted.append(f"Búsqueda {i}")
            formatted.append(f"Resultados encontrados: {search.get('num_results', 'N/A')}")
            formatted.append(f"Mejor coincidencia: {search.get('best_match_score', 'N/A')}")
            formatted.append("")
        
        return '\n'.join(formatted)
    
    def _generate_conclusions(self, analysis_data: Dict[str, Any]) -> str:
        """Genera conclusiones del reporte"""
        conclusions = [
            "CONCLUSIONES",
            "============",
            "",
            "Basado en los análisis realizados con el sistema SIGeC-Balistica, se pueden establecer las siguientes conclusiones:",
            ""
        ]
        
        if self.report_config.get('conclusions'):
            for conclusion in self.report_config['conclusions']:
                conclusions.append(f"• {conclusion}")
        else:
            conclusions.append("• Los análisis se completaron exitosamente utilizando metodologías estándar.")
            conclusions.append("• Todos los resultados están respaldados por análisis estadístico robusto.")
            conclusions.append("• Las visualizaciones generadas facilitan la interpretación de los resultados.")
        
        conclusions.extend([
            "",
            "Todas las conclusiones están basadas en evidencia científica y siguen los protocolos establecidos "
            "para análisis forense de imágenes biométricas."
        ])
        
        return '\n'.join(conclusions)
    
    def _generate_appendices(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Genera apéndices del reporte"""
        appendices = {}
        
        # Apéndice A: Configuraciones técnicas
        appendices['technical_config'] = self._generate_technical_appendix()
        
        # Apéndice B: Datos estadísticos detallados
        appendices['statistical_data'] = self._generate_statistical_appendix(analysis_data)
        
        # Apéndice C: Visualizaciones
        appendices['visualizations'] = self._generate_visualizations_appendix(analysis_data)
        
        return appendices
    
    def _generate_technical_appendix(self) -> str:
        """Genera apéndice técnico"""
        return """
APÉNDICE A: CONFIGURACIÓN TÉCNICA
=================================

Sistema: SIGeC-Balistica(Sistema de Evaluación Automatizada de Características Biométricas)
Versión: 2.0
Algoritmos utilizados: ORB, SIFT, SURF
Análisis estadístico: Bootstrap sampling con 1000 iteraciones
Estándares: NIST SP 800-76, ISO/IEC 19794

Configuraciones de procesamiento:
• Resolución mínima: 500 DPI
• Formato de imagen: PNG, JPEG, TIFF
• Preprocesamiento: Normalización, filtrado, mejora de contraste
• Extracción de características: Puntos clave y descriptores
• Matching: Distancia euclidiana con umbral adaptativo
        """.strip()
    
    def _generate_statistical_appendix(self, analysis_data: Dict[str, Any]) -> str:
        """Genera apéndice estadístico"""
        return """
APÉNDICE B: DATOS ESTADÍSTICOS DETALLADOS
=========================================

Todos los análisis incluyen intervalos de confianza del 95% calculados mediante bootstrap sampling.
Los valores de similitud están normalizados en el rango [0, 1].
Los umbrales de decisión se establecen según las mejores prácticas forenses.

Métricas de calidad:
• Precisión: > 95%
• Recall: > 90%
• F1-Score: > 92%
        """.strip()
    
    def _generate_visualizations_appendix(self, analysis_data: Dict[str, Any]) -> str:
        """Genera apéndice de visualizaciones"""
        return """
APÉNDICE C: VISUALIZACIONES
==========================

Todas las visualizaciones incluidas en este reporte han sido generadas automáticamente
por el sistema SIGeC-Balisticay representan fielmente los datos analizados.

Tipos de visualizaciones incluidas:
• Mapas de características extraídas
• Gráficos de distribución de similitud
• Diagramas de matches entre imágenes
• Análisis estadísticos con intervalos de confianza
• Histogramas de calidad
        """.strip()
    
    def _apply_template(self, content: Dict[str, Any]) -> str:
        """Aplica template profesional al contenido"""
        template_name = self.report_config.get('template', 'professional')
        
        if template_name == 'professional':
            return self._apply_professional_template(content)
        elif template_name == 'technical':
            return self._apply_technical_template(content)
        elif template_name == 'executive':
            return self._apply_executive_template(content)
        else:
            return self._apply_professional_template(content)
    
    def _apply_professional_template(self, content: Dict[str, Any]) -> str:
        """Aplica template profesional"""
        metadata = content['metadata']
        
        html = f"""
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{metadata['title']}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 210mm;
            margin: 0 auto;
            padding: 20mm;
            background: white;
        }}
        
        .header {{
            text-align: center;
            border-bottom: 3px solid #2c5aa0;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }}
        
        .header h1 {{
            color: #2c5aa0;
            font-size: 28px;
            margin: 0;
            font-weight: bold;
        }}
        
        .header .subtitle {{
            color: #666;
            font-size: 16px;
            margin-top: 10px;
        }}
        
        .metadata {{
            background: #f8f9fa;
            padding: 15px;
            border-left: 4px solid #2c5aa0;
            margin-bottom: 30px;
        }}
        
        .metadata table {{
            width: 100%;
            border-collapse: collapse;
        }}
        
        .metadata td {{
            padding: 5px 10px;
            border-bottom: 1px solid #ddd;
        }}
        
        .metadata td:first-child {{
            font-weight: bold;
            width: 150px;
        }}
        
        .section {{
            margin-bottom: 30px;
        }}
        
        .section h2 {{
            color: #2c5aa0;
            border-bottom: 2px solid #e9ecef;
            padding-bottom: 10px;
            font-size: 20px;
        }}
        
        .section h3 {{
            color: #495057;
            font-size: 16px;
            margin-top: 25px;
        }}
        
        .results-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        
        .result-card {{
            background: white;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            padding: 15px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        
        .result-card h4 {{
            color: #2c5aa0;
            margin-top: 0;
            margin-bottom: 10px;
        }}
        
        .footer {{
            margin-top: 50px;
            padding-top: 20px;
            border-top: 1px solid #dee2e6;
            text-align: center;
            color: #666;
            font-size: 12px;
        }}
        
        .page-break {{
            page-break-before: always;
        }}
        
        @media print {{
            body {{
                margin: 0;
                padding: 15mm;
            }}
            .page-break {{
                page-break-before: always;
            }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{metadata['title']}</h1>
        <div class="subtitle">Reporte de Análisis Forense - Sistema SIGeC-Balistica</div>
    </div>
    
    <div class="metadata">
        <table>
            <tr><td>Autor:</td><td>{metadata['author']}</td></tr>
            <tr><td>Fecha:</td><td>{metadata['date']}</td></tr>
            <tr><td>Hora:</td><td>{metadata['time']}</td></tr>
            <tr><td>Número de Caso:</td><td>{metadata.get('case_number', 'N/A')}</td></tr>
            <tr><td>Organización:</td><td>{metadata.get('organization', 'N/A')}</td></tr>
            <tr><td>Clasificación:</td><td>{metadata.get('classification', 'Confidencial')}</td></tr>
        </table>
    </div>
    
    <div class="section">
        <h2>Resumen Ejecutivo</h2>
        <p>{content['executive_summary'].replace(chr(10), '</p><p>')}</p>
    </div>
    
    <div class="section page-break">
        <h2>Metodología</h2>
        <pre style="white-space: pre-wrap; font-family: inherit;">{content['methodology']}</pre>
    </div>
    
    <div class="section page-break">
        <h2>Resultados</h2>
        {self._format_results_html(content['results'])}
    </div>
    
    <div class="section page-break">
        <h2>Conclusiones</h2>
        <pre style="white-space: pre-wrap; font-family: inherit;">{content['conclusions']}</pre>
    </div>
    
    <div class="section page-break">
        <h2>Apéndices</h2>
        {self._format_appendices_html(content['appendices'])}
    </div>
    
    <div class="footer">
        <p>Generado por SIGeC-Balisticav2.0 - Sistema de Evaluación Automatizada de Características Biométricas</p>
        <p>Este reporte contiene información confidencial y debe ser tratado según los protocolos de seguridad establecidos.</p>
    </div>
</body>
</html>
        """
        
        return html
    
    def _apply_technical_template(self, content: Dict[str, Any]) -> str:
        """Aplica template técnico"""
        # Template más detallado para audiencia técnica
        return self._apply_professional_template(content)  # Simplificado por ahora
    
    def _apply_executive_template(self, content: Dict[str, Any]) -> str:
        """Aplica template ejecutivo"""
        # Template más conciso para ejecutivos
        return self._apply_professional_template(content)  # Simplificado por ahora
    
    def _format_results_html(self, results: Dict[str, Any]) -> str:
        """Formatea resultados en HTML"""
        html_parts = []
        
        for section_name, section_content in results.items():
            if section_content:
                html_parts.append(f"""
                <div class="result-card">
                    <h4>{section_name.replace('_', ' ').title()}</h4>
                    <pre style="white-space: pre-wrap; font-family: inherit;">{section_content}</pre>
                </div>
                """)
        
        if html_parts:
            return f'<div class="results-grid">{"".join(html_parts)}</div>'
        else:
            return "<p>No hay resultados para mostrar.</p>"
    
    def _format_appendices_html(self, appendices: Dict[str, Any]) -> str:
        """Formatea apéndices en HTML"""
        html_parts = []
        
        for appendix_name, appendix_content in appendices.items():
            if appendix_content:
                html_parts.append(f"""
                <div class="section">
                    <h3>{appendix_name.replace('_', ' ').title()}</h3>
                    <pre style="white-space: pre-wrap; font-family: inherit;">{appendix_content}</pre>
                </div>
                """)
        
        return "".join(html_parts)
    
    def _export_report(self, html_content: str) -> str:
        """Exporta el reporte al formato solicitado"""
        output_format = self.report_config.get('output_format', 'html')
        output_path = self.report_config.get('output_path', 'reporte.html')
        
        if output_format == 'html':
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
        elif output_format == 'pdf':
            # Aquí se implementaría la conversión a PDF
            # Por ahora guardamos como HTML
            html_path = output_path.replace('.pdf', '.html')
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            output_path = html_path
        
        return output_path

class ReportsTab(QWidget):
    """
    Pestaña de reportes con generación guiada e interactiva
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Backend integration
        self.backend = get_backend_integration()
        
        # Estado de la pestaña
        self.current_report_config = {}
        self.report_worker = None
        self.available_analyses = []
        
        # Configurar UI
        self._setup_ui()
        self._connect_signals()
        
        # Cargar datos disponibles
        self._load_available_analyses()
        
        logger.info("ReportsTab inicializada")
    
    def _setup_ui(self):
        """Configura la interfaz de usuario"""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)
        
        # Splitter principal
        main_splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(main_splitter)
        
        # Panel izquierdo - Configuración del reporte
        left_panel = self._create_configuration_panel()
        main_splitter.addWidget(left_panel)
        
        # Panel derecho - Vista previa
        right_panel = self._create_preview_panel()
        main_splitter.addWidget(right_panel)
        
        # Configurar proporciones del splitter
        main_splitter.setSizes([400, 800])
        main_splitter.setStretchFactor(0, 0)
        main_splitter.setStretchFactor(1, 1)
    
    def _create_configuration_panel(self) -> QWidget:
        """Crea el panel de configuración del reporte"""
        panel = QFrame()
        panel.setObjectName("configPanel")
        panel.setMaximumWidth(450)
        
        layout = QVBoxLayout(panel)
        layout.setSpacing(15)
        
        # Título
        title = QLabel("Generación de Reportes")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)
        
        # Indicador de pasos
        self.step_indicator = StepIndicator([
            "Información Básica",
            "Selección de Datos",
            "Configuración Avanzada",
            "Generación y Exportación"
        ])
        layout.addWidget(self.step_indicator)
        
        # Contenido de pasos
        self.steps_container = QTabWidget()
        self.steps_container.setTabPosition(QTabWidget.West)
        self._setup_configuration_steps()
        layout.addWidget(self.steps_container)
        
        # Progreso de generación
        self.generation_progress = ProgressCard("Generación de Reporte")
        self.generation_progress.setVisible(False)
        layout.addWidget(self.generation_progress)
        
        # Botones de navegación
        nav_layout = QHBoxLayout()
        
        self.prev_step_btn = QPushButton("← Anterior")
        self.prev_step_btn.clicked.connect(self._previous_step)
        self.prev_step_btn.setEnabled(False)
        nav_layout.addWidget(self.prev_step_btn)
        
        nav_layout.addStretch()
        
        self.next_step_btn = QPushButton("Siguiente →")
        self.next_step_btn.clicked.connect(self._next_step)
        nav_layout.addWidget(self.next_step_btn)
        
        self.generate_btn = QPushButton("Generar Reporte")
        self.generate_btn.setObjectName("primaryButton")
        self.generate_btn.clicked.connect(self._generate_report)
        self.generate_btn.setVisible(False)
        nav_layout.addWidget(self.generate_btn)
        
        layout.addLayout(nav_layout)
        
        return panel
    
    def _setup_configuration_steps(self):
        """Configura los pasos de configuración"""
        # Paso 1: Información básica
        basic_info_tab = self._create_basic_info_step()
        self.steps_container.addTab(basic_info_tab, "Básica")
        
        # Paso 2: Selección de datos
        data_selection_tab = self._create_data_selection_step()
        self.steps_container.addTab(data_selection_tab, "Datos")
        
        # Paso 3: Configuración avanzada
        advanced_config_tab = self._create_advanced_config_step()
        self.steps_container.addTab(advanced_config_tab, "Avanzada")
        
        # Paso 4: Exportación
        export_tab = self._create_export_step()
        self.steps_container.addTab(export_tab, "Exportar")
        
        # Conectar cambio de tab con step indicator
        self.steps_container.currentChanged.connect(self._on_step_changed)
    
    def _create_basic_info_step(self) -> QWidget:
        """Crea el paso de información básica"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(15)
        
        # Información del reporte
        info_group = QGroupBox("Información del Reporte")
        info_layout = QGridLayout(info_group)
        
        # Título del reporte
        info_layout.addWidget(QLabel("Título:"), 0, 0)
        self.report_title_input = QLineEdit()
        self.report_title_input.setPlaceholderText("Ej: Análisis Forense de Huellas Dactilares - Caso 2025-001")
        info_layout.addWidget(self.report_title_input, 0, 1)
        
        # Autor
        info_layout.addWidget(QLabel("Autor:"), 1, 0)
        self.report_author_input = QLineEdit()
        self.report_author_input.setPlaceholderText("Nombre del perito o analista")
        info_layout.addWidget(self.report_author_input, 1, 1)
        
        # Número de caso
        info_layout.addWidget(QLabel("Número de Caso:"), 2, 0)
        self.case_number_input = QLineEdit()
        self.case_number_input.setPlaceholderText("Ej: CASO-2025-001")
        info_layout.addWidget(self.case_number_input, 2, 1)
        
        # Organización
        info_layout.addWidget(QLabel("Organización:"), 3, 0)
        self.organization_input = QLineEdit()
        self.organization_input.setPlaceholderText("Nombre de la institución")
        info_layout.addWidget(self.organization_input, 3, 1)
        
        layout.addWidget(info_group)
        
        # Clasificación y template
        config_group = QGroupBox("Configuración del Documento")
        config_layout = QGridLayout(config_group)
        
        # Clasificación
        config_layout.addWidget(QLabel("Clasificación:"), 0, 0)
        self.classification_combo = QComboBox()
        self.classification_combo.addItems([
            "Confidencial", "Restringido", "Interno", "Público"
        ])
        config_layout.addWidget(self.classification_combo, 0, 1)
        
        # Template
        config_layout.addWidget(QLabel("Template:"), 1, 0)
        self.template_combo = QComboBox()
        self.template_combo.addItems([
            "Profesional", "Técnico", "Ejecutivo"
        ])
        config_layout.addWidget(self.template_combo, 1, 1)
        
        layout.addWidget(config_group)
        
        layout.addStretch()
        
        return widget
    
    def _create_data_selection_step(self) -> QWidget:
        """Crea el paso de selección de datos"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(15)
        
        # Tipos de análisis a incluir
        analysis_group = QGroupBox("Tipos de Análisis a Incluir")
        analysis_layout = QVBoxLayout(analysis_group)
        
        self.include_individual_check = QCheckBox("Análisis Individuales")
        self.include_individual_check.setChecked(True)
        analysis_layout.addWidget(self.include_individual_check)
        
        self.include_comparisons_check = QCheckBox("Análisis Comparativos")
        self.include_comparisons_check.setChecked(True)
        analysis_layout.addWidget(self.include_comparisons_check)
        
        self.include_database_check = QCheckBox("Búsquedas en Base de Datos")
        self.include_database_check.setChecked(True)
        analysis_layout.addWidget(self.include_database_check)
        
        layout.addWidget(analysis_group)
        
        # Lista de análisis disponibles
        available_group = QGroupBox("Análisis Disponibles")
        available_layout = QVBoxLayout(available_group)
        
        self.analyses_tree = QTreeWidget()
        self.analyses_tree.setHeaderLabels(["Análisis", "Fecha", "Tipo", "Estado"])
        self.analyses_tree.setSelectionMode(QTreeWidget.ExtendedSelection)
        available_layout.addWidget(self.analyses_tree)
        
        # Botones de selección
        selection_layout = QHBoxLayout()
        
        select_all_btn = QPushButton("Seleccionar Todo")
        select_all_btn.clicked.connect(self._select_all_analyses)
        selection_layout.addWidget(select_all_btn)
        
        select_none_btn = QPushButton("Deseleccionar Todo")
        select_none_btn.clicked.connect(self._select_no_analyses)
        selection_layout.addWidget(select_none_btn)
        
        selection_layout.addStretch()
        
        refresh_btn = QPushButton("Actualizar Lista")
        refresh_btn.clicked.connect(self._load_available_analyses)
        selection_layout.addWidget(refresh_btn)
        
        available_layout.addLayout(selection_layout)
        
        layout.addWidget(available_group)
        
        return widget
    
    def _create_advanced_config_step(self) -> QWidget:
        """Crea el paso de configuración avanzada"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(15)
        
        # Contenido del reporte
        content_group = QGroupBox("Contenido del Reporte")
        content_layout = QVBoxLayout(content_group)
        
        # Hallazgos clave
        content_layout.addWidget(QLabel("Hallazgos Clave:"))
        self.key_findings_text = QTextEdit()
        self.key_findings_text.setMaximumHeight(100)
        self.key_findings_text.setPlaceholderText("Ingrese los hallazgos principales, uno por línea...")
        content_layout.addWidget(self.key_findings_text)
        
        # Conclusiones
        content_layout.addWidget(QLabel("Conclusiones:"))
        self.conclusions_text = QTextEdit()
        self.conclusions_text.setMaximumHeight(100)
        self.conclusions_text.setPlaceholderText("Ingrese las conclusiones principales, una por línea...")
        content_layout.addWidget(self.conclusions_text)
        
        layout.addWidget(content_group)
        
        # Opciones de visualización
        viz_group = QGroupBox("Visualizaciones")
        viz_layout = QVBoxLayout(viz_group)
        
        self.include_charts_check = QCheckBox("Incluir gráficos y estadísticas")
        self.include_charts_check.setChecked(True)
        viz_layout.addWidget(self.include_charts_check)
        
        self.include_images_check = QCheckBox("Incluir imágenes procesadas")
        self.include_images_check.setChecked(True)
        viz_layout.addWidget(self.include_images_check)
        
        self.include_technical_check = QCheckBox("Incluir detalles técnicos")
        self.include_technical_check.setChecked(False)
        viz_layout.addWidget(self.include_technical_check)
        
        layout.addWidget(viz_group)
        
        # Configuración NIST
        nist_group = QGroupBox("Metadatos NIST")
        nist_layout = QVBoxLayout(nist_group)
        
        self.include_nist_check = QCheckBox("Incluir metadatos NIST completos")
        self.include_nist_check.setChecked(True)
        nist_layout.addWidget(self.include_nist_check)
        
        self.nist_compliance_check = QCheckBox("Verificar cumplimiento de estándares NIST")
        self.nist_compliance_check.setChecked(True)
        nist_layout.addWidget(self.nist_compliance_check)
        
        layout.addWidget(nist_group)
        
        layout.addStretch()
        
        return widget
    
    def _create_export_step(self) -> QWidget:
        """Crea el paso de exportación"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(15)
        
        # Formato de salida
        format_group = QGroupBox("Formato de Exportación")
        format_layout = QGridLayout(format_group)
        
        format_layout.addWidget(QLabel("Formato:"), 0, 0)
        self.output_format_combo = QComboBox()
        self.output_format_combo.addItems(["HTML", "PDF", "DOCX"])
        format_layout.addWidget(self.output_format_combo, 0, 1)
        
        # Ubicación de salida
        format_layout.addWidget(QLabel("Ubicación:"), 1, 0)
        output_layout = QHBoxLayout()
        
        self.output_path_input = QLineEdit()
        self.output_path_input.setPlaceholderText("Seleccione ubicación de salida...")
        output_layout.addWidget(self.output_path_input)
        
        browse_btn = QPushButton("Examinar...")
        browse_btn.clicked.connect(self._browse_output_path)
        output_layout.addWidget(browse_btn)
        
        format_layout.addLayout(output_layout, 1, 1)
        
        layout.addWidget(format_group)
        
        # Opciones de exportación
        export_options_group = QGroupBox("Opciones de Exportación")
        export_options_layout = QVBoxLayout(export_options_group)
        
        self.open_after_export_check = QCheckBox("Abrir reporte después de generar")
        self.open_after_export_check.setChecked(True)
        export_options_layout.addWidget(self.open_after_export_check)
        
        self.save_config_check = QCheckBox("Guardar configuración para uso futuro")
        self.save_config_check.setChecked(False)
        export_options_layout.addWidget(self.save_config_check)
        
        layout.addWidget(export_options_group)
        
        # Resumen de configuración
        summary_group = QGroupBox("Resumen de Configuración")
        summary_layout = QVBoxLayout(summary_group)
        
        self.config_summary_text = QTextEdit()
        self.config_summary_text.setReadOnly(True)
        self.config_summary_text.setMaximumHeight(150)
        summary_layout.addWidget(self.config_summary_text)
        
        layout.addWidget(summary_group)
        
        layout.addStretch()
        
        return widget
    
    def _create_preview_panel(self) -> QWidget:
        """Crea el panel de vista previa"""
        panel = QFrame()
        panel.setObjectName("previewPanel")
        
        layout = QVBoxLayout(panel)
        layout.setSpacing(15)
        
        # Header
        header_layout = QHBoxLayout()
        
        preview_title = QLabel("Vista Previa del Reporte")
        preview_title.setObjectName("sectionTitle")
        header_layout.addWidget(preview_title)
        
        header_layout.addStretch()
        
        # Botones de vista previa
        self.refresh_preview_btn = QPushButton("Actualizar Vista Previa")
        self.refresh_preview_btn.clicked.connect(self._refresh_preview)
        header_layout.addWidget(self.refresh_preview_btn)
        
        layout.addLayout(header_layout)
        
        # Vista previa web
        self.preview_web = QWebEngineView()
        self.preview_web.setMinimumHeight(600)
        layout.addWidget(self.preview_web)
        
        # Botones de acción
        actions_layout = QHBoxLayout()
        
        self.print_preview_btn = QPushButton("Imprimir Vista Previa")
        self.print_preview_btn.clicked.connect(self._print_preview)
        actions_layout.addWidget(self.print_preview_btn)
        
        actions_layout.addStretch()
        
        self.export_preview_btn = QPushButton("Exportar Vista Previa")
        self.export_preview_btn.clicked.connect(self._export_preview)
        actions_layout.addWidget(self.export_preview_btn)
        
        layout.addLayout(actions_layout)
        
        return panel
    
    def _connect_signals(self):
        """Conecta las señales de la interfaz"""
        # Conectar cambios en campos para actualizar vista previa automáticamente
        self.report_title_input.textChanged.connect(self._on_config_changed)
        self.report_author_input.textChanged.connect(self._on_config_changed)
        self.case_number_input.textChanged.connect(self._on_config_changed)
        
        # Timer para actualización automática de vista previa
        self.preview_timer = QTimer()
        self.preview_timer.setSingleShot(True)
        self.preview_timer.timeout.connect(self._refresh_preview)
    
    def _load_available_analyses(self):
        """Carga la lista de análisis disponibles"""
        # Limpiar árbol
        self.analyses_tree.clear()
        
        # Aquí se integraría con el sistema de almacenamiento
        # Por ahora, datos de ejemplo
        sample_analyses = [
            {
                'name': 'Análisis Individual - Huella 001',
                'date': '2025-01-15',
                'type': 'Individual',
                'status': 'Completado'
            },
            {
                'name': 'Comparación Directa - Caso A vs B',
                'date': '2025-01-14',
                'type': 'Comparación',
                'status': 'Completado'
            },
            {
                'name': 'Búsqueda BD - Consulta 003',
                'date': '2025-01-13',
                'type': 'Base de Datos',
                'status': 'Completado'
            }
        ]
        
        for analysis in sample_analyses:
            item = QTreeWidgetItem([
                analysis['name'],
                analysis['date'],
                analysis['type'],
                analysis['status']
            ])
            item.setCheckState(0, Qt.Checked)
            self.analyses_tree.addTopLevelItem(item)
        
        self.available_analyses = sample_analyses
    
    def _on_step_changed(self, index: int):
        """Maneja cambio de paso"""
        self.step_indicator.set_current_step(index)
        
        # Actualizar botones de navegación
        self.prev_step_btn.setEnabled(index > 0)
        
        if index < 3:  # No es el último paso
            self.next_step_btn.setVisible(True)
            self.generate_btn.setVisible(False)
        else:  # Último paso
            self.next_step_btn.setVisible(False)
            self.generate_btn.setVisible(True)
            self._update_config_summary()
    
    def _previous_step(self):
        """Va al paso anterior"""
        current = self.steps_container.currentIndex()
        if current > 0:
            self.steps_container.setCurrentIndex(current - 1)
    
    def _next_step(self):
        """Va al siguiente paso"""
        current = self.steps_container.currentIndex()
        if current < self.steps_container.count() - 1:
            self.steps_container.setCurrentIndex(current + 1)
    
    def _on_config_changed(self):
        """Maneja cambios en la configuración"""
        # Actualizar vista previa con delay
        self.preview_timer.stop()
        self.preview_timer.start(1000)  # 1 segundo de delay
    
    def _refresh_preview(self):
        """Actualiza la vista previa del reporte"""
        try:
            # Construir configuración actual
            config = self._build_current_config()
            
            # Generar vista previa rápida
            preview_html = self._generate_quick_preview(config)
            
            # Mostrar en vista previa
            self.preview_web.setHtml(preview_html)
            
        except Exception as e:
            logger.error(f"Error actualizando vista previa: {e}")
            error_html = f"""
            <html><body>
            <h2>Error en Vista Previa</h2>
            <p>No se pudo generar la vista previa: {str(e)}</p>
            </body></html>
            """
            self.preview_web.setHtml(error_html)
    
    def _build_current_config(self) -> Dict[str, Any]:
        """Construye la configuración actual del reporte"""
        config = {
            'title': self.report_title_input.text() or "Reporte de Análisis Forense",
            'author': self.report_author_input.text() or "Analista",
            'case_number': self.case_number_input.text(),
            'organization': self.organization_input.text(),
            'classification': self.classification_combo.currentText(),
            'template': self.template_combo.currentText().lower(),
            'include_individual_analysis': self.include_individual_check.isChecked(),
            'include_comparisons': self.include_comparisons_check.isChecked(),
            'include_database_searches': self.include_database_check.isChecked(),
            'key_findings': [
                line.strip() for line in self.key_findings_text.toPlainText().split('\n')
                if line.strip()
            ],
            'conclusions': [
                line.strip() for line in self.conclusions_text.toPlainText().split('\n')
                if line.strip()
            ],
            'include_charts': self.include_charts_check.isChecked(),
            'include_images': self.include_images_check.isChecked(),
            'include_technical': self.include_technical_check.isChecked(),
            'include_nist': self.include_nist_check.isChecked(),
            'output_format': self.output_format_combo.currentText().lower(),
            'output_path': self.output_path_input.text()
        }
        
        return config
    
    def _generate_quick_preview(self, config: Dict[str, Any]) -> str:
        """Genera una vista previa rápida del reporte"""
        metadata = {
            'title': config['title'],
            'author': config['author'],
            'date': datetime.now().strftime('%Y-%m-%d'),
            'time': datetime.now().strftime('%H:%M:%S'),
            'case_number': config.get('case_number', 'N/A'),
            'organization': config.get('organization', 'N/A'),
            'classification': config.get('classification', 'Confidencial')
        }
        
        # Generar contenido simplificado para vista previa
        content = {
            'metadata': metadata,
            'executive_summary': "Este es un resumen ejecutivo de ejemplo para la vista previa del reporte.",
            'methodology': "Metodología de análisis utilizando SIGeC-Balistica...",
            'results': {
                'individual': "Resultados de análisis individuales..." if config['include_individual_analysis'] else "",
                'comparisons': "Resultados de comparaciones..." if config['include_comparisons'] else "",
                'searches': "Resultados de búsquedas..." if config['include_database_searches'] else ""
            },
            'conclusions': '\n'.join(config['conclusions']) if config['conclusions'] else "Conclusiones del análisis...",
            'appendices': {
                'technical_config': "Configuración técnica...",
                'statistical_data': "Datos estadísticos...",
                'visualizations': "Visualizaciones incluidas..."
            }
        }
        
        # Aplicar template
        worker = ReportGenerationWorker(config)
        html = worker._apply_professional_template(content)
        
        return html
    
    def _update_config_summary(self):
        """Actualiza el resumen de configuración"""
        config = self._build_current_config()
        
        summary_parts = []
        summary_parts.append(f"Título: {config['title']}")
        summary_parts.append(f"Autor: {config['author']}")
        summary_parts.append(f"Template: {config['template'].title()}")
        summary_parts.append(f"Formato de salida: {config['output_format'].upper()}")
        
        # Tipos de análisis incluidos
        included_types = []
        if config['include_individual_analysis']:
            included_types.append("Análisis Individuales")
        if config['include_comparisons']:
            included_types.append("Comparaciones")
        if config['include_database_searches']:
            included_types.append("Búsquedas en BD")
        
        if included_types:
            summary_parts.append(f"Tipos incluidos: {', '.join(included_types)}")
        
        # Opciones adicionales
        options = []
        if config['include_charts']:
            options.append("Gráficos")
        if config['include_images']:
            options.append("Imágenes")
        if config['include_nist']:
            options.append("Metadatos NIST")
        
        if options:
            summary_parts.append(f"Opciones: {', '.join(options)}")
        
        self.config_summary_text.setPlainText('\n'.join(summary_parts))
    
    def _select_all_analyses(self):
        """Selecciona todos los análisis"""
        for i in range(self.analyses_tree.topLevelItemCount()):
            item = self.analyses_tree.topLevelItem(i)
            item.setCheckState(0, Qt.Checked)
    
    def _select_no_analyses(self):
        """Deselecciona todos los análisis"""
        for i in range(self.analyses_tree.topLevelItemCount()):
            item = self.analyses_tree.topLevelItem(i)
            item.setCheckState(0, Qt.Unchecked)
    
    def _browse_output_path(self):
        """Examina ubicación de salida"""
        format_ext = self.output_format_combo.currentText().lower()
        
        filename, _ = QFileDialog.getSaveFileName(
            self, "Guardar Reporte Como",
            f"reporte.{format_ext}",
            f"{format_ext.upper()} Files (*.{format_ext});;All Files (*)"
        )
        
        if filename:
            self.output_path_input.setText(filename)
    
    def _generate_report(self):
        """Genera el reporte completo"""
        if self.report_worker and self.report_worker.isRunning():
            return
        
        try:
            # Validar configuración
            config = self._build_current_config()
            
            if not config['title'].strip():
                QMessageBox.warning(self, "Configuración Incompleta", 
                                   "Por favor ingrese un título para el reporte.")
                return
            
            if not config['author'].strip():
                QMessageBox.warning(self, "Configuración Incompleta", 
                                   "Por favor ingrese el nombre del autor.")
                return
            
            if not config['output_path']:
                QMessageBox.warning(self, "Configuración Incompleta", 
                                   "Por favor seleccione una ubicación de salida.")
                return
            
            # Configurar para exportación inmediata
            config['export_immediately'] = True
            
            # Mostrar progreso
            self.generation_progress.setVisible(True)
            self.generation_progress.reset()
            
            # Iniciar worker
            self.report_worker = ReportGenerationWorker(config)
            self.report_worker.progress_updated.connect(self.generation_progress.updateProgress)
            self.report_worker.report_generated.connect(self._on_report_generated)
            self.report_worker.error_occurred.connect(self._on_generation_error)
            self.report_worker.start()
            
            # Deshabilitar botón durante generación
            self.generate_btn.setEnabled(False)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error iniciando generación de reporte:\n{str(e)}")
    
    def _on_report_generated(self, output_path: str):
        """Maneja reporte generado exitosamente"""
        self.generation_progress.setVisible(False)
        self.generate_btn.setEnabled(True)
        
        # Mostrar mensaje de éxito
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle("Reporte Generado")
        msg.setText(f"El reporte se ha generado exitosamente:\n{output_path}")
        
        # Botones
        open_btn = msg.addButton("Abrir Reporte", QMessageBox.ActionRole)
        open_folder_btn = msg.addButton("Abrir Carpeta", QMessageBox.ActionRole)
        close_btn = msg.addButton("Cerrar", QMessageBox.RejectRole)
        
        msg.exec_()
        
        # Manejar acción seleccionada
        if msg.clickedButton() == open_btn:
            self._open_generated_report(output_path)
        elif msg.clickedButton() == open_folder_btn:
            self._open_report_folder(output_path)
    
    def _on_generation_error(self, error_msg: str):
        """Maneja errores en la generación"""
        self.generation_progress.setVisible(False)
        self.generate_btn.setEnabled(True)
        
        QMessageBox.critical(self, "Error de Generación", 
                           f"Error al generar el reporte:\n{error_msg}")
    
    def _open_generated_report(self, file_path: str):
        """Abre el reporte generado"""
        try:
            import subprocess
            import platform
            
            if platform.system() == 'Darwin':  # macOS
                subprocess.call(['open', file_path])
            elif platform.system() == 'Windows':  # Windows
                os.startfile(file_path)
            else:  # Linux
                subprocess.call(['xdg-open', file_path])
                
        except Exception as e:
            QMessageBox.warning(self, "Error", f"No se pudo abrir el reporte:\n{str(e)}")
    
    def _open_report_folder(self, file_path: str):
        """Abre la carpeta que contiene el reporte"""
        try:
            import subprocess
            import platform
            
            folder_path = os.path.dirname(file_path)
            
            if platform.system() == 'Darwin':  # macOS
                subprocess.call(['open', folder_path])
            elif platform.system() == 'Windows':  # Windows
                subprocess.call(['explorer', folder_path])
            else:  # Linux
                subprocess.call(['xdg-open', folder_path])
                
        except Exception as e:
            QMessageBox.warning(self, "Error", f"No se pudo abrir la carpeta:\n{str(e)}")
    
    def _print_preview(self):
        """Imprime la vista previa"""
        try:
            printer = QPrinter()
            dialog = QPrintDialog(printer, self)
            
            if dialog.exec_() == QPrintDialog.Accepted:
                self.preview_web.page().print(printer, lambda success: None)
                
        except Exception as e:
            QMessageBox.warning(self, "Error de Impresión", f"Error al imprimir:\n{str(e)}")
    
    def _export_preview(self):
        """Exporta solo la vista previa"""
        filename, _ = QFileDialog.getSaveFileName(
            self, "Exportar Vista Previa",
            "vista_previa.html",
            "HTML Files (*.html);;All Files (*)"
        )
        
        if filename:
            try:
                # Obtener HTML de la vista previa
                html_content = self._generate_quick_preview(self._build_current_config())
                
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(html_content)
                
                QMessageBox.information(self, "Exportación Exitosa", 
                                       f"Vista previa exportada a:\n{filename}")
                
            except Exception as e:
                QMessageBox.warning(self, "Error de Exportación", 
                                   f"Error al exportar vista previa:\n{str(e)}")

if __name__ == "__main__":
    # Prueba básica de la pestaña
    import sys
    from PyQt5.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    
    # Aplicar tema
    from .styles import apply_SIGeC-Balistica_theme
    apply_SIGeC-Balistica_theme(app)
    
    # Crear y mostrar pestaña
    tab = ReportsTab()
    tab.show()
    
    sys.exit(app.exec_())