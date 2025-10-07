#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced WebView - SIGeC-Balística
==================================

Componente QWebEngineView mejorado para soportar interactividad completa
en reportes HTML, incluyendo comunicación bidireccional con JavaScript.

Funcionalidades:
- Comunicación JavaScript ↔ Python
- Zoom y navegación de imágenes
- Interacción con gráficos
- Exportación mejorada
- Gestión de recursos

Autor: SIGeC-BalisticaTeam
Fecha: Octubre 2025
"""

import os
import json
import logging
import tempfile
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
from urllib.parse import urljoin
from urllib.request import pathname2url

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QToolBar, QAction,
    QProgressBar, QLabel, QMessageBox, QFileDialog, QMenu,
    QSplitter, QTextEdit, QGroupBox, QPushButton, QSpinBox,
    QSlider, QCheckBox, QComboBox
)
from PyQt5.QtWebEngineWidgets import QWebEngineView, QWebEnginePage, QWebEngineProfile
from PyQt5.QtWebChannel import QWebChannel
from PyQt5.QtCore import (
    Qt, pyqtSignal, pyqtSlot, QObject, QUrl, QTimer,
    QThread, QMutex, QWaitCondition
)
from PyQt5.QtGui import QIcon, QKeySequence, QFont
from PyQt5.QtPrintSupport import QPrinter, QPrintDialog

logger = logging.getLogger(__name__)

class JavaScriptBridge(QObject):
    """Puente de comunicación entre JavaScript y Python"""
    
    # Señales para comunicación con Python
    image_zoom_requested = pyqtSignal(str, float)  # image_id, zoom_level
    chart_interaction = pyqtSignal(str, str, dict)  # chart_id, action, data
    table_sort_requested = pyqtSignal(str, str, bool)  # table_id, column, ascending
    export_requested = pyqtSignal(str, dict)  # format, options
    navigation_requested = pyqtSignal(str)  # target_id
    
    def __init__(self):
        super().__init__()
        self.callbacks = {}
        
    @pyqtSlot(str, float)
    def onImageZoom(self, image_id: str, zoom_level: float):
        """Maneja eventos de zoom de imagen desde JavaScript"""
        logger.debug(f"Zoom de imagen: {image_id} -> {zoom_level}")
        self.image_zoom_requested.emit(image_id, zoom_level)
    
    @pyqtSlot(str, str, str)
    def onChartInteraction(self, chart_id: str, action: str, data_json: str):
        """Maneja interacciones con gráficos desde JavaScript"""
        try:
            data = json.loads(data_json) if data_json else {}
            logger.debug(f"Interacción con gráfico: {chart_id} -> {action}")
            self.chart_interaction.emit(chart_id, action, data)
        except json.JSONDecodeError as e:
            logger.error(f"Error decodificando datos de gráfico: {e}")
    
    @pyqtSlot(str, str, bool)
    def onTableSort(self, table_id: str, column: str, ascending: bool):
        """Maneja eventos de ordenamiento de tabla desde JavaScript"""
        logger.debug(f"Ordenamiento de tabla: {table_id} -> {column} ({'ASC' if ascending else 'DESC'})")
        self.table_sort_requested.emit(table_id, column, ascending)
    
    @pyqtSlot(str, str)
    def onExportRequest(self, format_type: str, options_json: str):
        """Maneja solicitudes de exportación desde JavaScript"""
        try:
            options = json.loads(options_json) if options_json else {}
            logger.debug(f"Solicitud de exportación: {format_type}")
            self.export_requested.emit(format_type, options)
        except json.JSONDecodeError as e:
            logger.error(f"Error decodificando opciones de exportación: {e}")
    
    @pyqtSlot(str)
    def onNavigationRequest(self, target_id: str):
        """Maneja solicitudes de navegación desde JavaScript"""
        logger.debug(f"Navegación solicitada: {target_id}")
        self.navigation_requested.emit(target_id)
    
    @pyqtSlot(str, str)
    def registerCallback(self, event_type: str, callback_id: str):
        """Registra un callback desde JavaScript"""
        if event_type not in self.callbacks:
            self.callbacks[event_type] = []
        self.callbacks[event_type].append(callback_id)
    
    def call_javascript(self, function_name: str, *args):
        """Llama una función JavaScript desde Python"""
        args_json = json.dumps(args)
        script = f"{function_name}(...{args_json})"
        return script

class EnhancedWebEnginePage(QWebEnginePage):
    """Página web mejorada con manejo de errores y recursos"""
    
    def __init__(self, profile=None, parent=None):
        super().__init__(profile, parent)
        
        self.resource_cache = {}
        self.base_url = None
        
    def acceptNavigationRequest(self, url, navigation_type, is_main_frame):
        """Controla las solicitudes de navegación"""
        # Permitir navegación interna y recursos locales
        if url.scheme() in ['file', 'data', 'about']:
            return True
        
        # Bloquear navegación externa por seguridad
        if url.scheme() in ['http', 'https']:
            logger.warning(f"Navegación externa bloqueada: {url.toString()}")
            return False
        
        return super().acceptNavigationRequest(url, navigation_type, is_main_frame)
    
    def javaScriptAlert(self, security_origin, msg):
        """Maneja alertas de JavaScript"""
        logger.info(f"JavaScript Alert: {msg}")
        # Opcional: mostrar en UI en lugar de alert nativo
        
    def javaScriptConsoleMessage(self, level, message, line_number, source_id):
        """Maneja mensajes de consola JavaScript"""
        level_names = {0: 'INFO', 1: 'WARNING', 2: 'ERROR'}
        level_name = level_names.get(level, 'UNKNOWN')
        logger.debug(f"JS Console [{level_name}] {source_id}:{line_number} - {message}")

class EnhancedWebView(QWebEngineView):
    """WebView mejorado con funcionalidades interactivas avanzadas"""
    
    # Señales para comunicación con el exterior
    content_loaded = pyqtSignal()
    loading_progress = pyqtSignal(int)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.js_bridge = JavaScriptBridge()
        self.web_channel = QWebChannel()
        self.temp_files = []
        self.current_content = ""
        self.zoom_factor = 1.0
        
        self._setup_web_engine()
        self._setup_javascript_bridge()
        self._connect_signals()
        
        logger.info("Enhanced WebView inicializado")
    
    def _setup_web_engine(self):
        """Configura el motor web"""
        # Crear perfil personalizado
        profile = QWebEngineProfile.defaultProfile()
        profile.setHttpCacheType(QWebEngineProfile.MemoryHttpCache)
        profile.setPersistentCookiesPolicy(QWebEngineProfile.NoPersistentCookies)
        
        # Crear página personalizada
        page = EnhancedWebEnginePage(profile, self)
        self.setPage(page)
        
        # Configurar zoom inicial
        self.setZoomFactor(1.0)
        
        # Habilitar características
        settings = page.settings()
        settings.setAttribute(settings.JavascriptEnabled, True)
        settings.setAttribute(settings.LocalContentCanAccessRemoteUrls, False)
        settings.setAttribute(settings.LocalContentCanAccessFileUrls, True)
        settings.setAttribute(settings.AllowRunningInsecureContent, False)
    
    def _setup_javascript_bridge(self):
        """Configura el puente JavaScript"""
        # Registrar el objeto bridge
        self.web_channel.registerObject("pyBridge", self.js_bridge)
        self.page().setWebChannel(self.web_channel)
        
        # Conectar señales del bridge
        self.js_bridge.image_zoom_requested.connect(self._handle_image_zoom)
        self.js_bridge.chart_interaction.connect(self._handle_chart_interaction)
        self.js_bridge.table_sort_requested.connect(self._handle_table_sort)
        self.js_bridge.export_requested.connect(self._handle_export_request)
        self.js_bridge.navigation_requested.connect(self._handle_navigation)
    
    def _connect_signals(self):
        """Conecta las señales internas"""
        self.loadFinished.connect(self._on_load_finished)
        self.loadProgress.connect(self.loading_progress.emit)
        
        # Timer para inyección de JavaScript
        self.js_injection_timer = QTimer()
        self.js_injection_timer.setSingleShot(True)
        self.js_injection_timer.timeout.connect(self._inject_javascript_bridge)
    
    def set_html_content(self, html_content: str, base_url: str = None):
        """Establece el contenido HTML con soporte para recursos locales"""
        try:
            self.current_content = html_content
            
            # Crear archivo temporal si es necesario
            if base_url is None:
                temp_file = tempfile.NamedTemporaryFile(
                    mode='w', suffix='.html', delete=False, encoding='utf-8'
                )
                temp_file.write(html_content)
                temp_file.close()
                
                self.temp_files.append(temp_file.name)
                base_url = f"file://{temp_file.name}"
            
            # Cargar contenido
            if base_url.startswith('file://'):
                self.load(QUrl(base_url))
            else:
                self.setHtml(html_content, QUrl(base_url))
            
            logger.info("Contenido HTML cargado en WebView")
            
        except Exception as e:
            logger.error(f"Error cargando contenido HTML: {e}")
            self.error_occurred.emit(str(e))
    
    def _on_load_finished(self, success: bool):
        """Maneja la finalización de carga"""
        if success:
            # Programar inyección de JavaScript después de un breve retraso
            self.js_injection_timer.start(100)
            self.content_loaded.emit()
            logger.debug("Contenido cargado exitosamente")
        else:
            error_msg = "Error cargando contenido web"
            logger.error(error_msg)
            self.error_occurred.emit(error_msg)
    
    def _inject_javascript_bridge(self):
        """Inyecta el código JavaScript del puente"""
        bridge_js = """
        // Verificar si QWebChannel está disponible
        if (typeof QWebChannel !== 'undefined') {
            new QWebChannel(qt.webChannelTransport, function(channel) {
                window.pyBridge = channel.objects.pyBridge;
                
                // Funciones de utilidad para comunicación
                window.reportInteraction = {
                    imageZoom: function(imageId, zoomLevel) {
                        if (window.pyBridge) {
                            window.pyBridge.onImageZoom(imageId, zoomLevel);
                        }
                    },
                    
                    chartInteraction: function(chartId, action, data) {
                        if (window.pyBridge) {
                            window.pyBridge.onChartInteraction(chartId, action, JSON.stringify(data || {}));
                        }
                    },
                    
                    tableSort: function(tableId, column, ascending) {
                        if (window.pyBridge) {
                            window.pyBridge.onTableSort(tableId, column, ascending);
                        }
                    },
                    
                    exportRequest: function(format, options) {
                        if (window.pyBridge) {
                            window.pyBridge.onExportRequest(format, JSON.stringify(options || {}));
                        }
                    },
                    
                    navigate: function(targetId) {
                        if (window.pyBridge) {
                            window.pyBridge.onNavigationRequest(targetId);
                        }
                    }
                };
                
                // Notificar que el puente está listo
                console.log('JavaScript bridge ready');
                
                // Activar funcionalidades interactivas
                if (typeof activateInteractiveFeatures === 'function') {
                    activateInteractiveFeatures();
                }
            });
        } else {
            console.warn('QWebChannel not available');
        }
        """
        
        self.page().runJavaScript(bridge_js)
        logger.debug("JavaScript bridge inyectado")
    
    def execute_javascript(self, script: str, callback: Callable = None):
        """Ejecuta JavaScript y opcionalmente maneja el resultado"""
        if callback:
            self.page().runJavaScript(script, callback)
        else:
            self.page().runJavaScript(script)
    
    def _handle_image_zoom(self, image_id: str, zoom_level: float):
        """Maneja eventos de zoom de imagen"""
        logger.info(f"Zoom de imagen {image_id}: {zoom_level}")
        
        # Actualizar zoom en JavaScript
        script = f"""
        if (typeof updateImageZoom === 'function') {{
            updateImageZoom('{image_id}', {zoom_level});
        }}
        """
        self.execute_javascript(script)
    
    def _handle_chart_interaction(self, chart_id: str, action: str, data: Dict):
        """Maneja interacciones con gráficos"""
        logger.info(f"Interacción con gráfico {chart_id}: {action}")
        
        # Procesar según el tipo de acción
        if action == 'hover':
            # Mostrar tooltip o información adicional
            pass
        elif action == 'click':
            # Manejar clics en elementos del gráfico
            pass
        elif action == 'zoom':
            # Manejar zoom en gráficos
            pass
    
    def _handle_table_sort(self, table_id: str, column: str, ascending: bool):
        """Maneja ordenamiento de tablas"""
        logger.info(f"Ordenamiento de tabla {table_id}: {column} ({'ASC' if ascending else 'DESC'})")
        
        # Ejecutar ordenamiento en JavaScript
        script = f"""
        if (typeof sortTable === 'function') {{
            sortTable('{table_id}', '{column}', {str(ascending).lower()});
        }}
        """
        self.execute_javascript(script)
    
    def _handle_export_request(self, format_type: str, options: Dict):
        """Maneja solicitudes de exportación"""
        logger.info(f"Solicitud de exportación: {format_type}")
        
        if format_type == 'pdf':
            self._export_to_pdf(options)
        elif format_type == 'html':
            self._export_to_html(options)
        elif format_type == 'image':
            self._export_to_image(options)
    
    def _handle_navigation(self, target_id: str):
        """Maneja navegación interna"""
        logger.info(f"Navegación a: {target_id}")
        
        # Scroll suave al elemento
        script = f"""
        var element = document.getElementById('{target_id}');
        if (element) {{
            element.scrollIntoView({{ behavior: 'smooth', block: 'start' }});
        }}
        """
        self.execute_javascript(script)
    
    def _export_to_pdf(self, options: Dict):
        """Exporta el contenido a PDF"""
        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Exportar a PDF", "reporte.pdf", "PDF Files (*.pdf)"
            )
            
            if file_path:
                # Usar QPrinter para exportación
                printer = QPrinter(QPrinter.HighResolution)
                printer.setOutputFormat(QPrinter.PdfFormat)
                printer.setOutputFileName(file_path)
                
                # Configurar opciones
                if 'page_size' in options:
                    printer.setPageSize(getattr(QPrinter, options['page_size'], QPrinter.A4))
                
                if 'orientation' in options:
                    orientation = QPrinter.Portrait if options['orientation'] == 'portrait' else QPrinter.Landscape
                    printer.setOrientation(orientation)
                
                # Imprimir
                self.print_(printer)
                
                logger.info(f"Reporte exportado a PDF: {file_path}")
                
        except Exception as e:
            logger.error(f"Error exportando a PDF: {e}")
            QMessageBox.critical(self, "Error", f"Error exportando a PDF: {e}")
    
    def _export_to_html(self, options: Dict):
        """Exporta el contenido a HTML"""
        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Exportar a HTML", "reporte.html", "HTML Files (*.html)"
            )
            
            if file_path:
                # Obtener HTML completo
                def save_html(html_content):
                    try:
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(html_content)
                        logger.info(f"Reporte exportado a HTML: {file_path}")
                    except Exception as e:
                        logger.error(f"Error guardando HTML: {e}")
                
                self.page().toHtml(save_html)
                
        except Exception as e:
            logger.error(f"Error exportando a HTML: {e}")
            QMessageBox.critical(self, "Error", f"Error exportando a HTML: {e}")
    
    def _export_to_image(self, options: Dict):
        """Exporta el contenido a imagen"""
        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Exportar a Imagen", "reporte.png", "PNG Files (*.png);;JPG Files (*.jpg)"
            )
            
            if file_path:
                # Capturar screenshot
                pixmap = self.grab()
                pixmap.save(file_path)
                
                logger.info(f"Reporte exportado a imagen: {file_path}")
                
        except Exception as e:
            logger.error(f"Error exportando a imagen: {e}")
            QMessageBox.critical(self, "Error", f"Error exportando a imagen: {e}")
    
    def zoom_in(self):
        """Aumenta el zoom"""
        self.zoom_factor = min(self.zoom_factor * 1.2, 3.0)
        self.setZoomFactor(self.zoom_factor)
    
    def zoom_out(self):
        """Disminuye el zoom"""
        self.zoom_factor = max(self.zoom_factor / 1.2, 0.3)
        self.setZoomFactor(self.zoom_factor)
    
    def reset_zoom(self):
        """Resetea el zoom"""
        self.zoom_factor = 1.0
        self.setZoomFactor(self.zoom_factor)
    
    def cleanup(self):
        """Limpia recursos temporales"""
        for temp_file in self.temp_files:
            try:
                os.unlink(temp_file)
            except OSError:
                pass
        self.temp_files.clear()
        
        logger.debug("Recursos temporales limpiados")

class InteractiveReportViewer(QWidget):
    """Visor de reportes interactivos con controles avanzados"""
    
    export_requested = pyqtSignal(str, dict)  # format, options
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.web_view = None
        self.current_report_data = None
        
        self._setup_ui()
        self._connect_signals()
        
        logger.info("Interactive Report Viewer inicializado")
    
    def _setup_ui(self):
        """Configura la interfaz de usuario"""
        layout = QVBoxLayout(self)
        
        # WebView primero
        self.web_view = EnhancedWebView()
        
        # Barra de herramientas
        self.toolbar = self._create_toolbar()
        layout.addWidget(self.toolbar)
        
        # Splitter principal
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)
        
        # Panel de control
        control_panel = self._create_control_panel()
        splitter.addWidget(control_panel)
        
        # WebView
        splitter.addWidget(self.web_view)
        
        # Configurar proporciones del splitter
        splitter.setSizes([200, 800])
        
        # Barra de estado
        self.status_bar = self._create_status_bar()
        layout.addWidget(self.status_bar)
    
    def _create_toolbar(self) -> QToolBar:
        """Crea la barra de herramientas"""
        toolbar = QToolBar()
        
        # Navegación
        toolbar.addAction("⬅️", self.web_view.back)
        toolbar.addAction("➡️", self.web_view.forward)
        toolbar.addAction("🔄", self.web_view.reload)
        
        toolbar.addSeparator()
        
        # Zoom
        toolbar.addAction("🔍+", self.web_view.zoom_in)
        toolbar.addAction("🔍-", self.web_view.zoom_out)
        toolbar.addAction("🔍=", self.web_view.reset_zoom)
        
        toolbar.addSeparator()
        
        # Exportación
        export_menu = QMenu("Exportar")
        export_menu.addAction("📄 PDF", lambda: self._request_export('pdf'))
        export_menu.addAction("🌐 HTML", lambda: self._request_export('html'))
        export_menu.addAction("🖼️ Imagen", lambda: self._request_export('image'))
        
        export_action = toolbar.addAction("📤 Exportar")
        export_action.setMenu(export_menu)
        
        return toolbar
    
    def _create_control_panel(self) -> QWidget:
        """Crea el panel de control"""
        panel = QWidget()
        panel.setMaximumWidth(250)
        layout = QVBoxLayout(panel)
        
        # Grupo de zoom
        zoom_group = QGroupBox("Control de Zoom")
        zoom_layout = QVBoxLayout(zoom_group)
        
        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setRange(30, 300)
        self.zoom_slider.setValue(100)
        self.zoom_slider.valueChanged.connect(self._on_zoom_changed)
        
        self.zoom_label = QLabel("100%")
        
        zoom_layout.addWidget(QLabel("Zoom:"))
        zoom_layout.addWidget(self.zoom_slider)
        zoom_layout.addWidget(self.zoom_label)
        
        layout.addWidget(zoom_group)
        
        # Grupo de navegación
        nav_group = QGroupBox("Navegación")
        nav_layout = QVBoxLayout(nav_group)
        
        self.nav_combo = QComboBox()
        nav_layout.addWidget(QLabel("Ir a sección:"))
        nav_layout.addWidget(self.nav_combo)
        
        layout.addWidget(nav_group)
        
        # Grupo de opciones de visualización
        display_group = QGroupBox("Visualización")
        display_layout = QVBoxLayout(display_group)
        
        self.show_tooltips_cb = QCheckBox("Mostrar tooltips")
        self.show_tooltips_cb.setChecked(True)
        
        self.enable_animations_cb = QCheckBox("Habilitar animaciones")
        self.enable_animations_cb.setChecked(True)
        
        self.high_contrast_cb = QCheckBox("Alto contraste")
        
        display_layout.addWidget(self.show_tooltips_cb)
        display_layout.addWidget(self.enable_animations_cb)
        display_layout.addWidget(self.high_contrast_cb)
        
        layout.addWidget(display_group)
        
        # Información del reporte
        info_group = QGroupBox("Información")
        info_layout = QVBoxLayout(info_group)
        
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setMaximumHeight(150)
        
        info_layout.addWidget(self.info_text)
        layout.addWidget(info_group)
        
        layout.addStretch()
        
        return panel
    
    def _create_status_bar(self) -> QWidget:
        """Crea la barra de estado"""
        status_widget = QWidget()
        layout = QHBoxLayout(status_widget)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        
        self.status_label = QLabel("Listo")
        
        layout.addWidget(self.status_label)
        layout.addStretch()
        layout.addWidget(self.progress_bar)
        
        return status_widget
    
    def _connect_signals(self):
        """Conecta las señales"""
        # Conectar señales del WebView cuando esté disponible
        if self.web_view:
            self.web_view.loading_progress.connect(self._on_loading_progress)
            self.web_view.content_loaded.connect(self._on_content_loaded)
            self.web_view.error_occurred.connect(self._on_error_occurred)
        
        # Conectar controles
        self.nav_combo.currentTextChanged.connect(self._on_navigation_selected)
        self.show_tooltips_cb.toggled.connect(self._on_display_option_changed)
        self.enable_animations_cb.toggled.connect(self._on_display_option_changed)
        self.high_contrast_cb.toggled.connect(self._on_display_option_changed)
    
    def set_report_content(self, html_content: str, report_data=None):
        """Establece el contenido del reporte"""
        try:
            self.current_report_data = report_data
            
            # Actualizar información
            if report_data and hasattr(report_data, 'metadata'):
                info_text = "Información del Reporte:\n\n"
                for key, value in report_data.metadata.items():
                    info_text += f"• {key}: {value}\n"
                self.info_text.setPlainText(info_text)
            
            # Actualizar navegación
            self._update_navigation_options(html_content)
            
            # Cargar contenido en WebView
            self.web_view.set_html_content(html_content)
            
            logger.info("Contenido del reporte establecido")
            
        except Exception as e:
            logger.error(f"Error estableciendo contenido del reporte: {e}")
            self._on_error_occurred(str(e))
    
    def _update_navigation_options(self, html_content: str):
        """Actualiza las opciones de navegación basadas en el contenido"""
        try:
            # Extraer secciones del HTML (buscar elementos con id)
            import re
            
            # Buscar elementos con id que representen secciones
            id_pattern = r'id=["\']([^"\']*section[^"\']*)["\']'
            section_ids = re.findall(id_pattern, html_content, re.IGNORECASE)
            
            # Buscar headers con id
            header_pattern = r'<h[1-6][^>]*id=["\']([^"\']*)["\'][^>]*>([^<]*)</h[1-6]>'
            headers = re.findall(header_pattern, html_content, re.IGNORECASE)
            
            # Combinar opciones
            nav_options = ["-- Seleccionar sección --"]
            
            for section_id in section_ids:
                nav_options.append(f"Sección: {section_id}")
            
            for header_id, header_text in headers:
                nav_options.append(f"{header_text.strip()}")
            
            # Actualizar combo
            self.nav_combo.clear()
            self.nav_combo.addItems(nav_options)
            
        except Exception as e:
            logger.warning(f"Error actualizando navegación: {e}")
    
    def _on_zoom_changed(self, value: int):
        """Maneja cambios en el zoom"""
        zoom_factor = value / 100.0
        self.web_view.setZoomFactor(zoom_factor)
        self.zoom_label.setText(f"{value}%")
    
    def _on_navigation_selected(self, text: str):
        """Maneja selección de navegación"""
        if text and not text.startswith("--"):
            # Extraer ID si es necesario
            if ":" in text:
                section_id = text.split(":", 1)[1].strip()
            else:
                section_id = text.replace(" ", "_").lower()
            
            # Navegar usando JavaScript
            script = f"""
            var element = document.getElementById('{section_id}') || 
                         document.querySelector('h1, h2, h3, h4, h5, h6').filter(h => h.textContent.includes('{text}'))[0];
            if (element) {{
                element.scrollIntoView({{ behavior: 'smooth', block: 'start' }});
            }}
            """
            self.web_view.execute_javascript(script)
    
    def _on_display_option_changed(self):
        """Maneja cambios en las opciones de visualización"""
        options = {
            'show_tooltips': self.show_tooltips_cb.isChecked(),
            'enable_animations': self.enable_animations_cb.isChecked(),
            'high_contrast': self.high_contrast_cb.isChecked()
        }
        
        # Aplicar opciones via JavaScript
        script = f"""
        if (typeof updateDisplayOptions === 'function') {{
            updateDisplayOptions({json.dumps(options)});
        }}
        """
        self.web_view.execute_javascript(script)
    
    def _on_loading_progress(self, progress: int):
        """Maneja el progreso de carga"""
        self.progress_bar.setVisible(progress < 100)
        self.progress_bar.setValue(progress)
        
        if progress < 100:
            self.status_label.setText(f"Cargando... {progress}%")
        else:
            self.status_label.setText("Listo")
    
    def _on_content_loaded(self):
        """Maneja la carga completa del contenido"""
        self.status_label.setText("Contenido cargado")
        self.progress_bar.setVisible(False)
        
        # Aplicar opciones de visualización
        self._on_display_option_changed()
    
    def _on_error_occurred(self, error_message: str):
        """Maneja errores"""
        self.status_label.setText(f"Error: {error_message}")
        self.progress_bar.setVisible(False)
        
        QMessageBox.warning(self, "Error", f"Error en el visor: {error_message}")
    
    def _request_export(self, format_type: str):
        """Solicita exportación"""
        options = {
            'high_quality': True,
            'include_interactive': format_type == 'html'
        }
        
        self.export_requested.emit(format_type, options)
    
    def cleanup(self):
        """Limpia recursos"""
        if self.web_view:
            self.web_view.cleanup()
        
        logger.debug("Interactive Report Viewer limpiado")