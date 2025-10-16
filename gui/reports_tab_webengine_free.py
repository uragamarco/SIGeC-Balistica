"""
WebEngine-free Reports Tab for SIGeC-Balisticar GUI.

This module provides a reports tab that doesn't depend on QtWebEngine,
using native Qt widgets for report generation and display.
"""

import os
import json
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QSplitter,
    QLabel, QPushButton, QTextEdit, QComboBox, QGroupBox,
    QScrollArea, QFrame, QFileDialog, QMessageBox, QProgressBar,
    QCheckBox, QSpinBox, QTabWidget, QTableWidget, QTableWidgetItem,
    QHeaderView, QLineEdit
)
from PyQt5.QtCore import Qt, pyqtSignal, QThread, QTimer
from PyQt5.QtGui import QFont, QPixmap, QTextDocument, QPainter
try:
    from PyQt5.QtPrintSupport import QPrinter
except ImportError:
    # Fallback if QPrinter is not available
    QPrinter = None

# Import shared widgets
from .shared_widgets import StepIndicator, ProgressCard


class ReportGenerationWorker(QThread):
    """Worker thread for generating reports without blocking the UI."""
    
    progress_updated = pyqtSignal(int, str)
    report_generated = pyqtSignal(str)  # file_path
    error_occurred = pyqtSignal(str)
    
    def __init__(self, report_data: Dict[str, Any], output_path: str, report_type: str):
        super().__init__()
        self.report_data = report_data
        self.output_path = output_path
        self.report_type = report_type
    
    def run(self):
        """Generate the report."""
        try:
            self.progress_updated.emit(10, "Preparando datos del reporte...")
            
            # Simulate report generation steps
            self.progress_updated.emit(30, "Generando contenido HTML...")
            html_content = self._generate_html_content()
            
            self.progress_updated.emit(60, "Aplicando estilos...")
            styled_html = self._apply_styles(html_content)
            
            self.progress_updated.emit(80, "Guardando archivo...")
            self._save_report(styled_html)
            
            self.progress_updated.emit(100, "Reporte generado exitosamente")
            self.report_generated.emit(self.output_path)
            
        except Exception as e:
            self.error_occurred.emit(f"Error generando reporte: {str(e)}")
    
    def _generate_html_content(self) -> str:
        """Generate HTML content for the report."""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>Reporte de Análisis Balístico - SIGeC</title>
        </head>
        <body>
            <div class="header">
                <h1>Reporte de Análisis Balístico</h1>
                <p>Generado el: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}</p>
            </div>
            
            <div class="content">
                <h2>Información del Caso</h2>
                <p><strong>Tipo de Análisis:</strong> {self.report_type}</p>
                <p><strong>Número de Imágenes:</strong> {self.report_data.get('image_count', 'N/A')}</p>
                
                <h2>Resultados del Análisis</h2>
                <p>Los resultados del análisis se presentan a continuación...</p>
                
                <h2>Conclusiones</h2>
                <p>Basado en el análisis realizado, se pueden establecer las siguientes conclusiones...</p>
            </div>
            
            <div class="footer">
                <p>Generado por SIGeC-Balisticar v1.0</p>
            </div>
        </body>
        </html>
        """
        return html
    
    def _apply_styles(self, html_content: str) -> str:
        """Apply CSS styles to the HTML content."""
        css = """
        <style>
        body {
            font-family: Arial, sans-serif;
            margin: 40px;
            line-height: 1.6;
        }
        .header {
            border-bottom: 2px solid #333;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }
        .header h1 {
            color: #2c3e50;
            margin: 0;
        }
        .content {
            margin-bottom: 40px;
        }
        .content h2 {
            color: #34495e;
            border-left: 4px solid #3498db;
            padding-left: 15px;
        }
        .footer {
            border-top: 1px solid #bdc3c7;
            padding-top: 20px;
            text-align: center;
            color: #7f8c8d;
        }
        </style>
        """
        return html_content.replace('<head>', f'<head>{css}')
    
    def _save_report(self, html_content: str):
        """Save the report to file."""
        with open(self.output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)


class ReportsTab(QWidget):
    """
    Reports tab for generating professional ballistic analysis reports.
    This version doesn't use QtWebEngine to avoid initialization issues.
    """
    
    # Signals
    report_generated = pyqtSignal(str)  # file_path
    report_requested = pyqtSignal(dict)  # report_config
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # State
        self.current_data = {}
        self.report_worker = None
        
        # Initialize UI
        self.init_ui()
        self.setup_connections()
        self.apply_modern_theme()
    
    def init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)
        
        # Create main splitter
        splitter = QSplitter(Qt.Horizontal)
        
        # Left panel - Report configuration
        left_panel = self.create_configuration_panel()
        splitter.addWidget(left_panel)
        
        # Right panel - Preview and generation
        right_panel = self.create_preview_panel()
        splitter.addWidget(right_panel)
        
        # Set splitter proportions
        splitter.setSizes([400, 600])
        
        layout.addWidget(splitter)
        
        # Status bar
        status_bar = self.create_status_bar()
        layout.addWidget(status_bar)
    
    def create_configuration_panel(self) -> QWidget:
        """Create the report configuration panel."""
        panel = QWidget()
        panel.setObjectName("configPanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(15)
        
        # Title
        title = QLabel("Configuración del Reporte")
        title.setObjectName("panelTitle")
        layout.addWidget(title)
        
        # Report type selection
        type_group = QGroupBox("Tipo de Reporte")
        type_layout = QVBoxLayout(type_group)
        
        self.report_type_combo = QComboBox()
        self.report_type_combo.addItems([
            "Análisis Individual",
            "Comparación Balística",
            "Análisis Estadístico",
            "Reporte Completo"
        ])
        type_layout.addWidget(self.report_type_combo)
        
        layout.addWidget(type_group)
        
        # Content selection
        content_group = QGroupBox("Contenido a Incluir")
        content_layout = QVBoxLayout(content_group)
        
        self.include_images = QCheckBox("Incluir imágenes")
        self.include_images.setChecked(True)
        content_layout.addWidget(self.include_images)
        
        self.include_statistics = QCheckBox("Incluir estadísticas")
        self.include_statistics.setChecked(True)
        content_layout.addWidget(self.include_statistics)
        
        self.include_metadata = QCheckBox("Incluir metadatos NIST")
        self.include_metadata.setChecked(False)
        content_layout.addWidget(self.include_metadata)
        
        self.include_conclusions = QCheckBox("Incluir conclusiones AFTE")
        self.include_conclusions.setChecked(True)
        content_layout.addWidget(self.include_conclusions)
        
        layout.addWidget(content_group)
        
        # Format options
        format_group = QGroupBox("Opciones de Formato")
        format_layout = QVBoxLayout(format_group)
        
        # Page size
        page_layout = QHBoxLayout()
        page_layout.addWidget(QLabel("Tamaño de página:"))
        self.page_size_combo = QComboBox()
        self.page_size_combo.addItems(["A4", "Letter", "Legal"])
        page_layout.addWidget(self.page_size_combo)
        format_layout.addLayout(page_layout)
        
        # Language
        lang_layout = QHBoxLayout()
        lang_layout.addWidget(QLabel("Idioma:"))
        self.language_combo = QComboBox()
        self.language_combo.addItems(["Español", "English"])
        lang_layout.addWidget(self.language_combo)
        format_layout.addLayout(lang_layout)
        
        layout.addWidget(format_group)
        
        # Generate button
        self.generate_btn = QPushButton("Generar Reporte")
        self.generate_btn.setObjectName("primaryButton")
        layout.addWidget(self.generate_btn)
        
        layout.addStretch()
        
        return panel
    
    def create_preview_panel(self) -> QWidget:
        """Create the preview and generation panel."""
        panel = QWidget()
        panel.setObjectName("previewPanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(15)
        
        # Title
        title = QLabel("Vista Previa del Reporte")
        title.setObjectName("panelTitle")
        layout.addWidget(title)
        
        # Preview area
        self.preview_area = QTextEdit()
        self.preview_area.setObjectName("previewArea")
        self.preview_area.setReadOnly(True)
        self.preview_area.setHtml(self.get_default_preview())
        layout.addWidget(self.preview_area)
        
        # Action buttons
        buttons_layout = QHBoxLayout()
        
        self.preview_btn = QPushButton("Actualizar Vista Previa")
        self.preview_btn.setObjectName("secondaryButton")
        buttons_layout.addWidget(self.preview_btn)
        
        self.export_btn = QPushButton("Exportar HTML")
        self.export_btn.setObjectName("secondaryButton")
        buttons_layout.addWidget(self.export_btn)
        
        self.print_btn = QPushButton("Imprimir")
        self.print_btn.setObjectName("secondaryButton")
        buttons_layout.addWidget(self.print_btn)
        
        buttons_layout.addStretch()
        layout.addLayout(buttons_layout)
        
        return panel
    
    def create_status_bar(self) -> QWidget:
        """Create the status bar."""
        status_widget = QWidget()
        status_widget.setObjectName("statusBar")
        layout = QHBoxLayout(status_widget)
        layout.setContentsMargins(10, 5, 10, 5)
        
        self.status_label = QLabel("Listo para generar reportes")
        layout.addWidget(self.status_label)
        
        layout.addStretch()
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setMaximumWidth(200)
        layout.addWidget(self.progress_bar)
        
        return status_widget
    
    def setup_connections(self):
        """Setup signal connections."""
        self.generate_btn.clicked.connect(self.generate_report)
        self.preview_btn.clicked.connect(self.update_preview)
        self.export_btn.clicked.connect(self.export_html)
        self.print_btn.clicked.connect(self.print_report)
        self.report_type_combo.currentTextChanged.connect(self.update_preview)
    
    def apply_modern_theme(self):
        """Apply modern theme to the widget."""
        self.setStyleSheet("""
            QWidget#configPanel, QWidget#previewPanel {
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 8px;
            }
            
            QLabel#panelTitle {
                font-size: 16px;
                font-weight: bold;
                color: #2c3e50;
                padding: 10px 0;
            }
            
            QGroupBox {
                font-weight: bold;
                border: 2px solid #bdc3c7;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            
            QPushButton#primaryButton {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                font-weight: bold;
            }
            
            QPushButton#primaryButton:hover {
                background-color: #2980b9;
            }
            
            QPushButton#secondaryButton {
                background-color: #95a5a6;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 5px;
            }
            
            QPushButton#secondaryButton:hover {
                background-color: #7f8c8d;
            }
            
            QTextEdit#previewArea {
                border: 1px solid #bdc3c7;
                border-radius: 5px;
                background-color: white;
            }
            
            QWidget#statusBar {
                background-color: #ecf0f1;
                border-top: 1px solid #bdc3c7;
            }
        """)
    
    def get_default_preview(self) -> str:
        """Get default preview content."""
        return """
        <h2>Vista Previa del Reporte</h2>
        <p>Seleccione las opciones de configuración y haga clic en "Actualizar Vista Previa" para ver el contenido del reporte.</p>
        <hr>
        <h3>Información del Caso</h3>
        <p><strong>Tipo de Análisis:</strong> Análisis Individual</p>
        <p><strong>Fecha de Generación:</strong> {}</p>
        <h3>Contenido del Reporte</h3>
        <ul>
            <li>Imágenes procesadas</li>
            <li>Estadísticas de análisis</li>
            <li>Conclusiones técnicas</li>
        </ul>
        """.format(datetime.now().strftime('%d/%m/%Y %H:%M:%S'))
    
    def update_preview(self):
        """Update the preview content."""
        report_type = self.report_type_combo.currentText()
        
        preview_content = f"""
        <h2>Vista Previa - {report_type}</h2>
        <p><strong>Fecha:</strong> {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}</p>
        <hr>
        
        <h3>Configuración Seleccionada</h3>
        <ul>
            <li><strong>Tipo:</strong> {report_type}</li>
            <li><strong>Incluir imágenes:</strong> {'Sí' if self.include_images.isChecked() else 'No'}</li>
            <li><strong>Incluir estadísticas:</strong> {'Sí' if self.include_statistics.isChecked() else 'No'}</li>
            <li><strong>Incluir metadatos NIST:</strong> {'Sí' if self.include_metadata.isChecked() else 'No'}</li>
            <li><strong>Incluir conclusiones AFTE:</strong> {'Sí' if self.include_conclusions.isChecked() else 'No'}</li>
        </ul>
        
        <h3>Contenido del Reporte</h3>
        <p>El reporte incluirá los siguientes elementos basados en su configuración...</p>
        """
        
        self.preview_area.setHtml(preview_content)
        self.status_label.setText("Vista previa actualizada")
    
    def generate_report(self):
        """Generate the report."""
        if self.report_worker and self.report_worker.isRunning():
            QMessageBox.warning(self, "Advertencia", "Ya se está generando un reporte. Por favor espere.")
            return
        
        # Get output file path
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Guardar Reporte",
            f"reporte_balistico_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
            "HTML Files (*.html);;All Files (*)"
        )
        
        if not file_path:
            return
        
        # Prepare report data
        report_data = {
            'report_type': self.report_type_combo.currentText(),
            'include_images': self.include_images.isChecked(),
            'include_statistics': self.include_statistics.isChecked(),
            'include_metadata': self.include_metadata.isChecked(),
            'include_conclusions': self.include_conclusions.isChecked(),
            'page_size': self.page_size_combo.currentText(),
            'language': self.language_combo.currentText(),
            'image_count': len(self.current_data.get('images', [])),
            'generation_date': datetime.now().isoformat()
        }
        
        # Start report generation
        self.report_worker = ReportGenerationWorker(
            report_data, file_path, self.report_type_combo.currentText()
        )
        self.report_worker.progress_updated.connect(self.on_progress_updated)
        self.report_worker.report_generated.connect(self.on_report_generated)
        self.report_worker.error_occurred.connect(self.on_error_occurred)
        
        self.progress_bar.setVisible(True)
        self.generate_btn.setEnabled(False)
        self.report_worker.start()
    
    def export_html(self):
        """Export current preview as HTML."""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Exportar Vista Previa",
            f"vista_previa_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
            "HTML Files (*.html);;All Files (*)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(self.preview_area.toHtml())
                QMessageBox.information(self, "Éxito", f"Vista previa exportada a:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error exportando vista previa:\n{str(e)}")
    
    def print_report(self):
        """Print the current preview."""
        if QPrinter is None:
            QMessageBox.warning(self, "Advertencia", "La funcionalidad de impresión no está disponible.")
            return
            
        printer = QPrinter()
        document = QTextDocument()
        document.setHtml(self.preview_area.toHtml())
        document.print_(printer)
        self.status_label.setText("Reporte enviado a impresora")
    
    def on_progress_updated(self, progress: int, message: str):
        """Handle progress updates."""
        self.progress_bar.setValue(progress)
        self.status_label.setText(message)
    
    def on_report_generated(self, file_path: str):
        """Handle successful report generation."""
        self.progress_bar.setVisible(False)
        self.generate_btn.setEnabled(True)
        self.status_label.setText(f"Reporte generado: {os.path.basename(file_path)}")
        
        QMessageBox.information(
            self,
            "Reporte Generado",
            f"El reporte se ha generado exitosamente:\n{file_path}"
        )
        
        self.report_generated.emit(file_path)
    
    def on_error_occurred(self, error_message: str):
        """Handle report generation errors."""
        self.progress_bar.setVisible(False)
        self.generate_btn.setEnabled(True)
        self.status_label.setText("Error generando reporte")
        
        QMessageBox.critical(self, "Error", error_message)
    
    def set_analysis_data(self, data: Dict[str, Any]):
        """Set analysis data for report generation."""
        self.current_data = data
        self.status_label.setText("Datos de análisis cargados")
        self.update_preview()
    
    def get_report_configuration(self) -> Dict[str, Any]:
        """Get current report configuration."""
        return {
            'report_type': self.report_type_combo.currentText(),
            'include_images': self.include_images.isChecked(),
            'include_statistics': self.include_statistics.isChecked(),
            'include_metadata': self.include_metadata.isChecked(),
            'include_conclusions': self.include_conclusions.isChecked(),
            'page_size': self.page_size_combo.currentText(),
            'language': self.language_combo.currentText()
        }