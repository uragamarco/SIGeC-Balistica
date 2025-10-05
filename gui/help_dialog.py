#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diálogo de Ayuda - SEACABAr
===========================

Sistema completo de ayuda y documentación que incluye:
- Guía de usuario paso a paso
- Documentación técnica
- Tutoriales interactivos
- FAQ (Preguntas frecuentes)
- Información del sistema
- Soporte técnico

Autor: SEACABAr Team
Fecha: Octubre 2025
"""

import os
import sys
import platform
import webbrowser
from typing import Dict, Any, List, Optional
from pathlib import Path

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTabWidget, QWidget,
    QLabel, QTextEdit, QTreeWidget, QTreeWidgetItem, QSplitter,
    QPushButton, QGroupBox, QFormLayout, QScrollArea, QFrame,
    QMessageBox, QLineEdit, QComboBox, QCheckBox, QProgressBar,
    QListWidget, QListWidgetItem, QStackedWidget, QGridLayout,
    QTextBrowser, QApplication
)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer, QThread, pyqtSlot, QUrl
from PyQt5.QtGui import QFont, QIcon, QPixmap, QPalette, QDesktopServices

from .shared_widgets import CollapsiblePanel, ResultCard
from .backend_integration import get_backend_integration

class HelpContentWidget(QWidget):
    """Widget base para contenido de ayuda"""
    
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.title = title
        self.setup_ui()
    
    def setup_ui(self):
        """Configura la interfaz básica"""
        layout = QVBoxLayout(self)
        
        # Título
        title_label = QLabel(self.title)
        title_label.setObjectName("sectionTitle")
        layout.addWidget(title_label)
        
        # Contenido específico
        self.setup_content(layout)
    
    def setup_content(self, layout):
        """Implementar en subclases"""
        pass

class UserGuideWidget(HelpContentWidget):
    """Guía de usuario paso a paso"""
    
    def __init__(self, parent=None):
        super().__init__("Guía de Usuario", parent)
    
    def setup_content(self, layout):
        """Configura el contenido de la guía"""
        
        # Crear splitter para navegación y contenido
        splitter = QSplitter(Qt.Horizontal)
        
        # Panel de navegación
        nav_widget = QWidget()
        nav_widget.setMaximumWidth(250)
        nav_layout = QVBoxLayout(nav_widget)
        
        nav_label = QLabel("Navegación")
        nav_label.setObjectName("subsectionTitle")
        nav_layout.addWidget(nav_label)
        
        self.nav_tree = QTreeWidget()
        self.nav_tree.setHeaderHidden(True)
        self.nav_tree.itemClicked.connect(self.on_nav_item_clicked)
        nav_layout.addWidget(self.nav_tree)
        
        # Poblar navegación
        self.populate_navigation()
        
        splitter.addWidget(nav_widget)
        
        # Panel de contenido
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        
        self.content_browser = QTextBrowser()
        self.content_browser.setOpenExternalLinks(True)
        content_layout.addWidget(self.content_browser)
        
        splitter.addWidget(content_widget)
        
        # Configurar proporciones
        splitter.setSizes([250, 600])
        
        layout.addWidget(splitter)
        
        # Mostrar contenido inicial
        self.show_introduction()
    
    def populate_navigation(self):
        """Pobla el árbol de navegación"""
        
        # Introducción
        intro_item = QTreeWidgetItem(["Introducción"])
        intro_item.setData(0, Qt.UserRole, "introduction")
        self.nav_tree.addTopLevelItem(intro_item)
        
        # Primeros pasos
        getting_started = QTreeWidgetItem(["Primeros Pasos"])
        getting_started.addChild(QTreeWidgetItem(["Instalación y Configuración"]))
        getting_started.addChild(QTreeWidgetItem(["Interfaz de Usuario"]))
        getting_started.addChild(QTreeWidgetItem(["Configuración Inicial"]))
        self.nav_tree.addTopLevelItem(getting_started)
        
        # Análisis Individual
        individual_analysis = QTreeWidgetItem(["Análisis Individual"])
        individual_analysis.addChild(QTreeWidgetItem(["Cargar Imagen"]))
        individual_analysis.addChild(QTreeWidgetItem(["Configurar Parámetros"]))
        individual_analysis.addChild(QTreeWidgetItem(["Ejecutar Análisis"]))
        individual_analysis.addChild(QTreeWidgetItem(["Interpretar Resultados"]))
        self.nav_tree.addTopLevelItem(individual_analysis)
        
        # Análisis Comparativo
        comparative_analysis = QTreeWidgetItem(["Análisis Comparativo"])
        comparative_analysis.addChild(QTreeWidgetItem(["Comparación Directa"]))
        comparative_analysis.addChild(QTreeWidgetItem(["Búsqueda en Base de Datos"]))
        comparative_analysis.addChild(QTreeWidgetItem(["Interpretación de Similitudes"]))
        self.nav_tree.addTopLevelItem(comparative_analysis)
        
        # Base de Datos
        database = QTreeWidgetItem(["Gestión de Base de Datos"])
        database.addChild(QTreeWidgetItem(["Búsqueda y Filtros"]))
        database.addChild(QTreeWidgetItem(["Visualización de Resultados"]))
        database.addChild(QTreeWidgetItem(["Exportación de Datos"]))
        self.nav_tree.addTopLevelItem(database)
        
        # Reportes
        reports = QTreeWidgetItem(["Generación de Reportes"])
        reports.addChild(QTreeWidgetItem(["Configuración de Reportes"]))
        reports.addChild(QTreeWidgetItem(["Personalización"]))
        reports.addChild(QTreeWidgetItem(["Exportación"]))
        self.nav_tree.addTopLevelItem(reports)
        
        # Configuración Avanzada
        advanced = QTreeWidgetItem(["Configuración Avanzada"])
        advanced.addChild(QTreeWidgetItem(["Parámetros de Procesamiento"]))
        advanced.addChild(QTreeWidgetItem(["Optimización de Rendimiento"]))
        advanced.addChild(QTreeWidgetItem(["Integración NIST"]))
        self.nav_tree.addTopLevelItem(advanced)
        
        # Expandir primer nivel
        self.nav_tree.expandAll()
    
    def on_nav_item_clicked(self, item, column):
        """Maneja clics en elementos de navegación"""
        item_text = item.text(0)
        self.show_content_for_item(item_text)
    
    def show_content_for_item(self, item_text: str):
        """Muestra contenido específico para un elemento"""
        content_map = {
            "Introducción": self.get_introduction_content(),
            "Instalación y Configuración": self.get_installation_content(),
            "Interfaz de Usuario": self.get_interface_content(),
            "Cargar Imagen": self.get_load_image_content(),
            "Configurar Parámetros": self.get_configure_params_content(),
            "Ejecutar Análisis": self.get_execute_analysis_content(),
            "Comparación Directa": self.get_direct_comparison_content(),
            "Búsqueda en Base de Datos": self.get_database_search_content(),
            "Configuración de Reportes": self.get_report_config_content(),
        }
        
        content = content_map.get(item_text, self.get_default_content(item_text))
        self.content_browser.setHtml(content)
    
    def show_introduction(self):
        """Muestra la introducción"""
        self.content_browser.setHtml(self.get_introduction_content())
    
    def get_introduction_content(self) -> str:
        """Contenido de introducción"""
        return """
        <h2>Bienvenido a SEACABAr</h2>
        
        <p><strong>SEACABAr</strong> es un sistema avanzado de análisis de huellas dactilares que combina 
        técnicas estadísticas modernas con estándares NIST para proporcionar análisis forenses precisos 
        y confiables.</p>
        
        <h3>Características Principales</h3>
        <ul>
            <li><strong>Análisis Individual:</strong> Procesamiento completo de huellas individuales</li>
            <li><strong>Análisis Comparativo:</strong> Comparación entre huellas y búsqueda en base de datos</li>
            <li><strong>Cumplimiento NIST:</strong> Integración completa con estándares NIST</li>
            <li><strong>Reportes Profesionales:</strong> Generación de reportes detallados</li>
            <li><strong>Interfaz Intuitiva:</strong> Diseño moderno y fácil de usar</li>
        </ul>
        
        <h3>Flujo de Trabajo Típico</h3>
        <ol>
            <li>Cargar imagen de huella dactilar</li>
            <li>Configurar parámetros de análisis</li>
            <li>Ejecutar procesamiento</li>
            <li>Revisar resultados y visualizaciones</li>
            <li>Generar reporte profesional</li>
        </ol>
        
        <p><em>Seleccione un tema en el panel de navegación para obtener información detallada.</em></p>
        """
    
    def get_installation_content(self) -> str:
        """Contenido de instalación"""
        return """
        <h2>Instalación y Configuración</h2>
        
        <h3>Requisitos del Sistema</h3>
        <ul>
            <li>Python 3.8 o superior</li>
            <li>4 GB de RAM mínimo (8 GB recomendado)</li>
            <li>2 GB de espacio libre en disco</li>
            <li>Sistema operativo: Windows 10+, macOS 10.14+, Linux Ubuntu 18.04+</li>
        </ul>
        
        <h3>Instalación</h3>
        <ol>
            <li>Descargar el paquete de instalación</li>
            <li>Ejecutar el instalador como administrador</li>
            <li>Seguir las instrucciones del asistente</li>
            <li>Reiniciar el sistema si es necesario</li>
        </ol>
        
        <h3>Primera Configuración</h3>
        <p>Al iniciar SEACABAr por primera vez:</p>
        <ol>
            <li>Se ejecutará el asistente de configuración inicial</li>
            <li>Configure la ruta de la base de datos</li>
            <li>Ajuste las preferencias de procesamiento</li>
            <li>Verifique la conectividad con servicios NIST</li>
        </ol>
        """
    
    def get_interface_content(self) -> str:
        """Contenido de interfaz"""
        return """
        <h2>Interfaz de Usuario</h2>
        
        <h3>Componentes Principales</h3>
        
        <h4>Barra de Menú</h4>
        <ul>
            <li><strong>Archivo:</strong> Operaciones de archivo y configuración</li>
            <li><strong>Análisis:</strong> Acceso rápido a funciones de análisis</li>
            <li><strong>Herramientas:</strong> Utilidades adicionales</li>
            <li><strong>Ayuda:</strong> Documentación y soporte</li>
        </ul>
        
        <h4>Pestañas Principales</h4>
        <ul>
            <li><strong>Análisis:</strong> Procesamiento de huellas individuales</li>
            <li><strong>Comparación:</strong> Análisis comparativo entre huellas</li>
            <li><strong>Base de Datos:</strong> Búsqueda y gestión de datos</li>
            <li><strong>Reportes:</strong> Generación de documentos profesionales</li>
        </ul>
        
        <h4>Paneles de Trabajo</h4>
        <ul>
            <li><strong>Panel de Configuración:</strong> Ajustes de procesamiento</li>
            <li><strong>Panel de Visualización:</strong> Resultados y gráficos</li>
            <li><strong>Panel de Progreso:</strong> Estado de operaciones</li>
        </ul>
        """
    
    def get_load_image_content(self) -> str:
        """Contenido de carga de imagen"""
        return """
        <h2>Cargar Imagen</h2>
        
        <h3>Formatos Soportados</h3>
        <ul>
            <li>PNG (recomendado)</li>
            <li>JPEG/JPG</li>
            <li>TIFF</li>
            <li>BMP</li>
            <li>WSQ (formato forense)</li>
        </ul>
        
        <h3>Métodos de Carga</h3>
        
        <h4>Arrastrar y Soltar</h4>
        <ol>
            <li>Arrastre el archivo desde el explorador</li>
            <li>Suéltelo en la zona de carga</li>
            <li>La imagen se cargará automáticamente</li>
        </ol>
        
        <h4>Botón Examinar</h4>
        <ol>
            <li>Haga clic en "Examinar..."</li>
            <li>Seleccione el archivo en el diálogo</li>
            <li>Confirme la selección</li>
        </ol>
        
        <h3>Validación de Imagen</h3>
        <p>El sistema validará automáticamente:</p>
        <ul>
            <li>Formato de archivo válido</li>
            <li>Resolución mínima (300 DPI recomendado)</li>
            <li>Calidad de imagen suficiente</li>
            <li>Presencia de características dactilares</li>
        </ul>
        """
    
    def get_default_content(self, item_text: str) -> str:
        """Contenido por defecto para elementos no implementados"""
        return f"""
        <h2>{item_text}</h2>
        <p>Documentación para <strong>{item_text}</strong> estará disponible próximamente.</p>
        <p>Para más información, consulte la documentación técnica o contacte al soporte técnico.</p>
        """

class TechnicalDocsWidget(HelpContentWidget):
    """Documentación técnica"""
    
    def __init__(self, parent=None):
        super().__init__("Documentación Técnica", parent)
    
    def setup_content(self, layout):
        """Configura el contenido técnico"""
        
        # Crear pestañas para diferentes secciones técnicas
        tech_tabs = QTabWidget()
        
        # API Reference
        api_tab = QWidget()
        api_layout = QVBoxLayout(api_tab)
        api_browser = QTextBrowser()
        api_browser.setHtml(self.get_api_documentation())
        api_layout.addWidget(api_browser)
        tech_tabs.addTab(api_tab, "API Reference")
        
        # Algoritmos
        algo_tab = QWidget()
        algo_layout = QVBoxLayout(algo_tab)
        algo_browser = QTextBrowser()
        algo_browser.setHtml(self.get_algorithms_documentation())
        algo_layout.addWidget(algo_browser)
        tech_tabs.addTab(algo_tab, "Algoritmos")
        
        # Configuración
        config_tab = QWidget()
        config_layout = QVBoxLayout(config_tab)
        config_browser = QTextBrowser()
        config_browser.setHtml(self.get_configuration_documentation())
        config_layout.addWidget(config_browser)
        tech_tabs.addTab(config_tab, "Configuración")
        
        # Integración
        integration_tab = QWidget()
        integration_layout = QVBoxLayout(integration_tab)
        integration_browser = QTextBrowser()
        integration_browser.setHtml(self.get_integration_documentation())
        integration_layout.addWidget(integration_browser)
        tech_tabs.addTab(integration_tab, "Integración")
        
        layout.addWidget(tech_tabs)
    
    def get_api_documentation(self) -> str:
        """Documentación de API"""
        return """
        <h2>API Reference</h2>
        
        <h3>BackendIntegration</h3>
        <p>Clase principal para integración con el backend.</p>
        
        <h4>Métodos Principales</h4>
        <pre><code>
# Análisis individual
result = backend.analyze_image(
    image_path: str,
    config: Dict[str, Any]
) -> AnalysisResult

# Comparación de imágenes
comparison = backend.compare_images(
    image_a: str,
    image_b: str,
    config: Dict[str, Any]
) -> ComparisonResult

# Búsqueda en base de datos
results = backend.search_database(
    query_image: str,
    filters: Dict[str, Any]
) -> List[SearchResult]
        </code></pre>
        
        <h3>Configuración</h3>
        <pre><code>
config = {
    'quality_threshold': 0.7,
    'min_minutiae': 12,
    'enhancement_enabled': True,
    'algorithm': 'auto'
}
        </code></pre>
        
        <h3>Resultados</h3>
        <pre><code>
class AnalysisResult:
    quality_score: float
    minutiae_count: int
    statistical_analysis: Dict
    nist_compliance: Dict
    visualizations: List[str]
        </code></pre>
        """
    
    def get_algorithms_documentation(self) -> str:
        """Documentación de algoritmos"""
        return """
        <h2>Algoritmos Implementados</h2>
        
        <h3>Mejora de Imagen</h3>
        <ul>
            <li><strong>Filtro Gabor:</strong> Mejora de crestas y valles</li>
            <li><strong>Ecualización de Histograma:</strong> Mejora de contraste</li>
            <li><strong>CLAHE:</strong> Ecualización adaptativa</li>
            <li><strong>Filtro Wiener:</strong> Reducción de ruido</li>
        </ul>
        
        <h3>Extracción de Características</h3>
        <ul>
            <li><strong>Crossing Number:</strong> Detección de minucias por número de cruces</li>
            <li><strong>Ridge Ending:</strong> Detección de terminaciones</li>
            <li><strong>Hybrid:</strong> Combinación de métodos</li>
            <li><strong>Deep Learning:</strong> Redes neuronales especializadas</li>
        </ul>
        
        <h3>Análisis Estadístico</h3>
        <ul>
            <li><strong>Bootstrap Sampling:</strong> Estimación de confianza</li>
            <li><strong>PCA:</strong> Análisis de componentes principales</li>
            <li><strong>Clustering:</strong> Agrupación de características</li>
            <li><strong>Correlación:</strong> Análisis de similitudes</li>
        </ul>
        
        <h3>Comparación</h3>
        <ul>
            <li><strong>Minutiae Matching:</strong> Comparación de minucias</li>
            <li><strong>Pattern Matching:</strong> Comparación de patrones</li>
            <li><strong>Statistical Similarity:</strong> Similitud estadística</li>
        </ul>
        """
    
    def get_configuration_documentation(self) -> str:
        """Documentación de configuración"""
        return """
        <h2>Configuración del Sistema</h2>
        
        <h3>Archivo de Configuración</h3>
        <p>La configuración se almacena en <code>config/unified_config.yaml</code></p>
        
        <h3>Secciones Principales</h3>
        
        <h4>Procesamiento de Imagen</h4>
        <pre><code>
image_processing:
  quality_threshold: 0.7
  min_minutiae: 12
  enhancement:
    enabled: true
    method: "gabor"
  preprocessing:
    resize: true
    normalize: true
        </code></pre>
        
        <h4>Base de Datos</h4>
        <pre><code>
database:
  path: "data/seacabar.db"
  backup:
    enabled: true
    interval: 24  # horas
  search:
    threshold: 0.8
    max_results: 100
        </code></pre>
        
        <h4>NIST Integration</h4>
        <pre><code>
nist:
  enabled: true
  standards_path: "standards/"
  compliance_level: "strict"
  reporting:
    include_metadata: true
    format: "json"
        </code></pre>
        
        <h3>Variables de Entorno</h3>
        <ul>
            <li><code>SEACABAR_CONFIG_PATH</code>: Ruta del archivo de configuración</li>
            <li><code>SEACABAR_DATA_PATH</code>: Directorio de datos</li>
            <li><code>SEACABAR_LOG_LEVEL</code>: Nivel de logging</li>
        </ul>
        """
    
    def get_integration_documentation(self) -> str:
        """Documentación de integración"""
        return """
        <h2>Integración con Sistemas Externos</h2>
        
        <h3>API REST</h3>
        <p>SEACABAr puede exponerse como servicio REST:</p>
        <pre><code>
# Iniciar servidor API
python -m seacabar.api --port 8080

# Endpoints disponibles
POST /api/v1/analyze
POST /api/v1/compare
GET  /api/v1/database/search
GET  /api/v1/status
        </code></pre>
        
        <h3>Integración con AFIS</h3>
        <p>Conectores disponibles para sistemas AFIS:</p>
        <ul>
            <li>Morpho AFIS</li>
            <li>NEC AFIS</li>
            <li>Cogent AFIS</li>
            <li>Generic ANSI/NIST</li>
        </ul>
        
        <h3>Formatos de Intercambio</h3>
        <ul>
            <li><strong>ANSI/NIST-ITL:</strong> Estándar para intercambio</li>
            <li><strong>ISO/IEC 19794-2:</strong> Formato de minucias</li>
            <li><strong>WSQ:</strong> Compresión de imágenes</li>
            <li><strong>JSON:</strong> Metadatos y resultados</li>
        </ul>
        
        <h3>SDK y Librerías</h3>
        <pre><code>
# Instalación del SDK
pip install seacabar-sdk

# Uso básico
from seacabar_sdk import SEACABArClient

client = SEACABArClient("http://localhost:8080")
result = client.analyze_image("fingerprint.png")
        </code></pre>
        """

class FAQWidget(HelpContentWidget):
    """Preguntas frecuentes"""
    
    def __init__(self, parent=None):
        super().__init__("Preguntas Frecuentes", parent)
    
    def setup_content(self, layout):
        """Configura el contenido de FAQ"""
        
        # Crear lista de categorías y contenido
        splitter = QSplitter(Qt.Horizontal)
        
        # Lista de categorías
        categories_widget = QWidget()
        categories_widget.setMaximumWidth(200)
        categories_layout = QVBoxLayout(categories_widget)
        
        categories_label = QLabel("Categorías")
        categories_label.setObjectName("subsectionTitle")
        categories_layout.addWidget(categories_label)
        
        self.categories_list = QListWidget()
        self.categories_list.addItems([
            "General",
            "Instalación",
            "Análisis",
            "Comparación",
            "Base de Datos",
            "Reportes",
            "Errores Comunes",
            "Rendimiento"
        ])
        self.categories_list.currentTextChanged.connect(self.show_faq_category)
        categories_layout.addWidget(self.categories_list)
        
        splitter.addWidget(categories_widget)
        
        # Contenido de FAQ
        self.faq_browser = QTextBrowser()
        splitter.addWidget(self.faq_browser)
        
        layout.addWidget(splitter)
        
        # Mostrar FAQ general por defecto
        self.show_faq_category("General")
    
    def show_faq_category(self, category: str):
        """Muestra FAQ para una categoría específica"""
        faq_content = {
            "General": self.get_general_faq(),
            "Instalación": self.get_installation_faq(),
            "Análisis": self.get_analysis_faq(),
            "Comparación": self.get_comparison_faq(),
            "Base de Datos": self.get_database_faq(),
            "Reportes": self.get_reports_faq(),
            "Errores Comunes": self.get_errors_faq(),
            "Rendimiento": self.get_performance_faq()
        }
        
        content = faq_content.get(category, "<h2>Categoría no encontrada</h2>")
        self.faq_browser.setHtml(content)
    
    def get_general_faq(self) -> str:
        """FAQ general"""
        return """
        <h2>Preguntas Generales</h2>
        
        <h3>¿Qué es SEACABAr?</h3>
        <p>SEACABAr es un sistema avanzado de análisis estadístico de huellas dactilares que combina 
        técnicas modernas de procesamiento de imágenes con estándares NIST para análisis forenses.</p>
        
        <h3>¿Qué formatos de imagen soporta?</h3>
        <p>SEACABAr soporta PNG, JPEG, TIFF, BMP y WSQ. Se recomienda PNG para mejor calidad.</p>
        
        <h3>¿Es compatible con estándares NIST?</h3>
        <p>Sí, SEACABAr está completamente integrado con estándares NIST y puede generar reportes 
        compatibles con requisitos forenses.</p>
        
        <h3>¿Puedo usar SEACABAr en modo offline?</h3>
        <p>Sí, SEACABAr funciona completamente offline. Solo requiere conexión para actualizaciones 
        y sincronización con servicios externos opcionales.</p>
        
        <h3>¿Qué precisión tiene el sistema?</h3>
        <p>La precisión depende de la calidad de la imagen, pero típicamente alcanza >95% en 
        condiciones óptimas con imágenes de alta calidad.</p>
        """
    
    def get_installation_faq(self) -> str:
        """FAQ de instalación"""
        return """
        <h2>Instalación y Configuración</h2>
        
        <h3>¿Cuáles son los requisitos mínimos del sistema?</h3>
        <p>Python 3.8+, 4GB RAM, 2GB espacio libre, Windows 10+/macOS 10.14+/Ubuntu 18.04+</p>
        
        <h3>¿Cómo instalo las dependencias?</h3>
        <p>Ejecute <code>pip install -r requirements.txt</code> en el directorio del proyecto.</p>
        
        <h3>¿Qué hago si falla la instalación?</h3>
        <p>Verifique que tiene permisos de administrador y que Python está correctamente instalado. 
        Consulte los logs de instalación para errores específicos.</p>
        
        <h3>¿Cómo configuro la base de datos?</h3>
        <p>La base de datos se configura automáticamente en el primer inicio. Puede cambiar la 
        ubicación en Configuración > Base de Datos.</p>
        
        <h3>¿Puedo instalar en un servidor?</h3>
        <p>Sí, SEACABAr puede ejecutarse en modo servidor. Use <code>python main.py --headless</code> 
        para modo sin interfaz gráfica.</p>
        """
    
    def get_analysis_faq(self) -> str:
        """FAQ de análisis"""
        return """
        <h2>Análisis de Huellas</h2>
        
        <h3>¿Por qué mi imagen no se procesa correctamente?</h3>
        <p>Verifique que la imagen tenga suficiente resolución (mínimo 300 DPI) y que las 
        características dactilares sean visibles.</p>
        
        <h3>¿Qué significa el puntaje de calidad?</h3>
        <p>El puntaje de calidad (0-1) indica qué tan clara y procesable es la huella. 
        Valores >0.7 son considerados buenos para análisis.</p>
        
        <h3>¿Cuántas minucias necesito para un análisis válido?</h3>
        <p>Se recomiendan al menos 12 minucias para análisis confiable, aunque esto puede 
        configurarse según sus necesidades.</p>
        
        <h3>¿Qué hago si el análisis toma mucho tiempo?</h3>
        <p>Reduzca la resolución de la imagen o ajuste los parámetros de procesamiento en 
        Configuración > Procesamiento.</p>
        
        <h3>¿Puedo procesar múltiples imágenes a la vez?</h3>
        <p>Actualmente no, pero esta funcionalidad está planificada para futuras versiones.</p>
        """
    
    def get_comparison_faq(self) -> str:
        """FAQ de comparación"""
        return """
        <h2>Análisis Comparativo</h2>
        
        <h3>¿Qué significa el porcentaje de similitud?</h3>
        <p>Indica qué tan similares son las huellas comparadas. >80% sugiere alta probabilidad 
        de coincidencia, pero siempre debe considerarse el contexto forense.</p>
        
        <h3>¿Cómo interpreto los resultados de comparación?</h3>
        <p>Considere tanto el porcentaje de similitud como el número de minucias coincidentes 
        y la calidad de ambas imágenes.</p>
        
        <h3>¿Puedo comparar huellas de diferentes calidades?</h3>
        <p>Sí, pero la precisión puede verse afectada. El sistema ajusta automáticamente 
        los parámetros según la calidad detectada.</p>
        
        <h3>¿Qué es la búsqueda en base de datos?</h3>
        <p>Permite comparar una huella contra todas las almacenadas en la base de datos 
        para encontrar posibles coincidencias.</p>
        
        <h3>¿Cómo mejoro la precisión de las comparaciones?</h3>
        <p>Use imágenes de alta calidad, ajuste los umbrales de similitud y verifique 
        que las configuraciones sean apropiadas para su caso de uso.</p>
        """
    
    def get_database_faq(self) -> str:
        """FAQ de base de datos"""
        return """
        <h2>Gestión de Base de Datos</h2>
        
        <h3>¿Dónde se almacenan mis datos?</h3>
        <p>Por defecto en el directorio <code>data/</code> del proyecto. Puede cambiar 
        la ubicación en la configuración.</p>
        
        <h3>¿Cómo hago respaldo de la base de datos?</h3>
        <p>Use Herramientas > Respaldo de Base de Datos o configure respaldos automáticos 
        en Configuración > Base de Datos.</p>
        
        <h3>¿Puedo importar datos de otros sistemas?</h3>
        <p>Sí, SEACABAr soporta importación desde formatos ANSI/NIST y CSV. 
        Use Archivo > Importar Datos.</p>
        
        <h3>¿Cómo busco registros específicos?</h3>
        <p>Use los filtros en la pestaña Base de Datos. Puede filtrar por fecha, 
        tipo de evidencia, calidad y otros criterios.</p>
        
        <h3>¿Qué hago si la base de datos se corrompe?</h3>
        <p>Restaure desde el último respaldo válido. Si no tiene respaldo, 
        contacte al soporte técnico.</p>
        """
    
    def get_reports_faq(self) -> str:
        """FAQ de reportes"""
        return """
        <h2>Generación de Reportes</h2>
        
        <h3>¿Qué formatos de reporte están disponibles?</h3>
        <p>PDF, HTML y DOCX. PDF es recomendado para documentos oficiales.</p>
        
        <h3>¿Puedo personalizar el contenido del reporte?</h3>
        <p>Sí, puede seleccionar qué secciones incluir, añadir conclusiones personalizadas 
        y configurar el formato.</p>
        
        <h3>¿Los reportes cumplen con estándares forenses?</h3>
        <p>Sí, los reportes incluyen toda la información requerida por estándares NIST 
        y pueden usarse en contextos legales.</p>
        
        <h3>¿Cómo incluyo múltiples análisis en un reporte?</h3>
        <p>En la configuración del reporte, seleccione múltiples análisis de la lista 
        de datos disponibles.</p>
        
        <h3>¿Puedo añadir mi logo institucional?</h3>
        <p>Sí, configure su logo en Configuración > Exportación > Marca de Agua.</p>
        """
    
    def get_errors_faq(self) -> str:
        """FAQ de errores comunes"""
        return """
        <h2>Errores Comunes</h2>
        
        <h3>"Error de memoria insuficiente"</h3>
        <p>Reduzca el tamaño de la imagen o aumente el límite de memoria en 
        Configuración > Procesamiento > Límite de Memoria.</p>
        
        <h3>"No se pueden detectar minucias"</h3>
        <p>La imagen puede tener baja calidad. Intente con mejora de imagen activada 
        o ajuste los parámetros de detección.</p>
        
        <h3>"Error de base de datos bloqueada"</h3>
        <p>Cierre otras instancias de SEACABAr o reinicie la aplicación. 
        Si persiste, verifique permisos de archivo.</p>
        
        <h3>"Fallo en la comparación"</h3>
        <p>Verifique que ambas imágenes sean válidas y tengan suficientes características 
        para comparar.</p>
        
        <h3>"Error de exportación de reporte"</h3>
        <p>Verifique que tiene permisos de escritura en el directorio de destino 
        y que no hay archivos abiertos con el mismo nombre.</p>
        """
    
    def get_performance_faq(self) -> str:
        """FAQ de rendimiento"""
        return """
        <h2>Optimización de Rendimiento</h2>
        
        <h3>¿Cómo acelero el procesamiento?</h3>
        <p>Active la aceleración GPU si está disponible, aumente el número de hilos 
        de procesamiento y use imágenes de resolución apropiada.</p>
        
        <h3>¿Por qué la aplicación consume mucha memoria?</h3>
        <p>El procesamiento de imágenes de alta resolución requiere memoria. 
        Ajuste el límite de memoria en la configuración.</p>
        
        <h3>¿Puedo procesar en segundo plano?</h3>
        <p>Sí, todos los análisis se ejecutan en hilos separados para mantener 
        la interfaz responsiva.</p>
        
        <h3>¿Cómo optimizo para múltiples usuarios?</h3>
        <p>Use el modo servidor con balanceador de carga y configure múltiples 
        instancias de procesamiento.</p>
        
        <h3>¿Qué hardware recomiendan?</h3>
        <p>CPU multi-core, 16GB+ RAM, SSD para almacenamiento, GPU compatible 
        con CUDA para aceleración opcional.</p>
        """

class SystemInfoWidget(HelpContentWidget):
    """Información del sistema"""
    
    def __init__(self, parent=None):
        super().__init__("Información del Sistema", parent)
    
    def setup_content(self, layout):
        """Configura el contenido de información del sistema"""
        
        # Crear pestañas para diferentes tipos de información
        info_tabs = QTabWidget()
        
        # Información general
        general_tab = QWidget()
        general_layout = QVBoxLayout(general_tab)
        
        self.general_info = QTextEdit()
        self.general_info.setReadOnly(True)
        general_layout.addWidget(self.general_info)
        
        refresh_general_btn = QPushButton("Actualizar Información")
        refresh_general_btn.clicked.connect(self.refresh_general_info)
        general_layout.addWidget(refresh_general_btn)
        
        info_tabs.addTab(general_tab, "General")
        
        # Información de dependencias
        deps_tab = QWidget()
        deps_layout = QVBoxLayout(deps_tab)
        
        self.deps_info = QTextEdit()
        self.deps_info.setReadOnly(True)
        deps_layout.addWidget(self.deps_info)
        
        refresh_deps_btn = QPushButton("Verificar Dependencias")
        refresh_deps_btn.clicked.connect(self.refresh_dependencies_info)
        deps_layout.addWidget(refresh_deps_btn)
        
        info_tabs.addTab(deps_tab, "Dependencias")
        
        # Logs del sistema
        logs_tab = QWidget()
        logs_layout = QVBoxLayout(logs_tab)
        
        self.logs_info = QTextEdit()
        self.logs_info.setReadOnly(True)
        logs_layout.addWidget(self.logs_info)
        
        logs_buttons = QHBoxLayout()
        refresh_logs_btn = QPushButton("Actualizar Logs")
        refresh_logs_btn.clicked.connect(self.refresh_logs_info)
        logs_buttons.addWidget(refresh_logs_btn)
        
        clear_logs_btn = QPushButton("Limpiar Logs")
        clear_logs_btn.clicked.connect(self.clear_logs)
        logs_buttons.addWidget(clear_logs_btn)
        
        logs_buttons.addStretch()
        logs_layout.addLayout(logs_buttons)
        
        info_tabs.addTab(logs_tab, "Logs")
        
        layout.addWidget(info_tabs)
        
        # Cargar información inicial
        self.refresh_general_info()
        self.refresh_dependencies_info()
        self.refresh_logs_info()
    
    def refresh_general_info(self):
        """Actualiza la información general del sistema"""
        try:
            info = f"""
INFORMACIÓN DEL SISTEMA
======================

Aplicación: SEACABAr v1.0.0
Fecha de compilación: Octubre 2025

SISTEMA OPERATIVO
-----------------
Sistema: {platform.system()}
Versión: {platform.version()}
Arquitectura: {platform.architecture()[0]}
Procesador: {platform.processor()}

PYTHON
------
Versión: {sys.version}
Ejecutable: {sys.executable}
Ruta: {sys.path[0]}

MEMORIA
-------
Memoria total: {self.get_memory_info()}

DIRECTORIOS
-----------
Directorio de trabajo: {os.getcwd()}
Directorio de datos: {os.path.join(os.getcwd(), 'data')}
Directorio de configuración: {os.path.join(os.getcwd(), 'config')}

BACKEND
-------
Estado: {self.get_backend_status()}
            """.strip()
            
            self.general_info.setPlainText(info)
            
        except Exception as e:
            self.general_info.setPlainText(f"Error obteniendo información del sistema: {str(e)}")
    
    def refresh_dependencies_info(self):
        """Actualiza la información de dependencias"""
        try:
            import pkg_resources
            
            # Lista de dependencias críticas
            critical_deps = [
                'PyQt5', 'numpy', 'opencv-python', 'scikit-learn',
                'matplotlib', 'pillow', 'scipy', 'pandas'
            ]
            
            info = "DEPENDENCIAS CRÍTICAS\n"
            info += "====================\n\n"
            
            for dep in critical_deps:
                try:
                    version = pkg_resources.get_distribution(dep).version
                    info += f"✓ {dep}: {version}\n"
                except pkg_resources.DistributionNotFound:
                    info += f"✗ {dep}: NO INSTALADO\n"
            
            info += "\n\nTODAS LAS DEPENDENCIAS\n"
            info += "=====================\n\n"
            
            installed_packages = [d for d in pkg_resources.working_set]
            installed_packages.sort(key=lambda x: x.project_name.lower())
            
            for package in installed_packages:
                info += f"{package.project_name}: {package.version}\n"
            
            self.deps_info.setPlainText(info)
            
        except Exception as e:
            self.deps_info.setPlainText(f"Error obteniendo información de dependencias: {str(e)}")
    
    def refresh_logs_info(self):
        """Actualiza la información de logs"""
        try:
            log_file = "seacabar.log"
            if os.path.exists(log_file):
                with open(log_file, 'r', encoding='utf-8') as f:
                    # Leer las últimas 100 líneas
                    lines = f.readlines()
                    recent_lines = lines[-100:] if len(lines) > 100 else lines
                    log_content = ''.join(recent_lines)
            else:
                log_content = "No se encontró archivo de log."
            
            self.logs_info.setPlainText(log_content)
            
        except Exception as e:
            self.logs_info.setPlainText(f"Error leyendo logs: {str(e)}")
    
    def clear_logs(self):
        """Limpia los logs del sistema"""
        reply = QMessageBox.question(
            self,
            "Confirmar Limpieza",
            "¿Está seguro de que desea limpiar los logs del sistema?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            try:
                log_file = "seacabar.log"
                if os.path.exists(log_file):
                    open(log_file, 'w').close()  # Vaciar archivo
                
                self.logs_info.setPlainText("Logs limpiados.")
                QMessageBox.information(self, "Éxito", "Logs limpiados exitosamente.")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error limpiando logs: {str(e)}")
    
    def get_memory_info(self) -> str:
        """Obtiene información de memoria"""
        try:
            import psutil
            memory = psutil.virtual_memory()
            return f"{memory.total // (1024**3)} GB ({memory.percent}% usado)"
        except ImportError:
            return "No disponible (instale psutil)"
    
    def get_backend_status(self) -> str:
        """Obtiene el estado del backend"""
        try:
            backend = get_backend_integration()
            return "Conectado ✓"
        except Exception:
            return "Error de conexión ✗"

class SupportWidget(HelpContentWidget):
    """Widget de soporte técnico"""
    
    def __init__(self, parent=None):
        super().__init__("Soporte Técnico", parent)
    
    def setup_content(self, layout):
        """Configura el contenido de soporte"""
        
        # Información de contacto
        contact_group = QGroupBox("Información de Contacto")
        contact_layout = QFormLayout(contact_group)
        
        contact_layout.addRow("Email:", QLabel("soporte@seacabar.com"))
        contact_layout.addRow("Teléfono:", QLabel("+1-800-SEACABAR"))
        contact_layout.addRow("Web:", QLabel('<a href="https://seacabar.com">https://seacabar.com</a>'))
        
        layout.addWidget(contact_group)
        
        # Formulario de reporte de problemas
        report_group = QGroupBox("Reportar Problema")
        report_layout = QVBoxLayout(report_group)
        
        form_layout = QFormLayout()
        
        self.problem_type = QComboBox()
        self.problem_type.addItems([
            "Error de aplicación",
            "Problema de rendimiento",
            "Error de análisis",
            "Problema de instalación",
            "Solicitud de función",
            "Otro"
        ])
        form_layout.addRow("Tipo de Problema:", self.problem_type)
        
        self.problem_description = QTextEdit()
        self.problem_description.setMaximumHeight(150)
        self.problem_description.setPlaceholderText("Describa el problema en detalle...")
        form_layout.addRow("Descripción:", self.problem_description)
        
        self.include_system_info = QCheckBox("Incluir información del sistema")
        self.include_system_info.setChecked(True)
        form_layout.addRow(self.include_system_info)
        
        report_layout.addLayout(form_layout)
        
        # Botones de acción
        buttons_layout = QHBoxLayout()
        
        generate_report_btn = QPushButton("Generar Reporte")
        generate_report_btn.clicked.connect(self.generate_support_report)
        buttons_layout.addWidget(generate_report_btn)
        
        send_email_btn = QPushButton("Enviar por Email")
        send_email_btn.clicked.connect(self.send_support_email)
        buttons_layout.addWidget(send_email_btn)
        
        buttons_layout.addStretch()
        report_layout.addLayout(buttons_layout)
        
        layout.addWidget(report_group)
        
        # Enlaces útiles
        links_group = QGroupBox("Enlaces Útiles")
        links_layout = QVBoxLayout(links_group)
        
        links = [
            ("Documentación Online", "https://docs.seacabar.com"),
            ("Foro de Usuarios", "https://forum.seacabar.com"),
            ("Base de Conocimiento", "https://kb.seacabar.com"),
            ("Actualizaciones", "https://updates.seacabar.com"),
            ("Canal de YouTube", "https://youtube.com/seacabar")
        ]
        
        for text, url in links:
            link_btn = QPushButton(text)
            link_btn.clicked.connect(lambda checked, u=url: QDesktopServices.openUrl(QUrl(u)))
            links_layout.addWidget(link_btn)
        
        layout.addWidget(links_group)
    
    def generate_support_report(self):
        """Genera un reporte de soporte"""
        try:
            report_content = f"""
REPORTE DE SOPORTE TÉCNICO
==========================

Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Tipo de Problema: {self.problem_type.currentText()}

DESCRIPCIÓN DEL PROBLEMA
------------------------
{self.problem_description.toPlainText()}

"""
            
            if self.include_system_info.isChecked():
                report_content += f"""
INFORMACIÓN DEL SISTEMA
-----------------------
Sistema Operativo: {platform.system()} {platform.version()}
Arquitectura: {platform.architecture()[0]}
Python: {sys.version}
Directorio de trabajo: {os.getcwd()}

"""
            
            # Guardar reporte
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Guardar Reporte de Soporte",
                f"reporte_soporte_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                "Archivos de texto (*.txt);;Todos los archivos (*)"
            )
            
            if file_path:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(report_content)
                
                QMessageBox.information(
                    self,
                    "Éxito",
                    f"Reporte de soporte generado exitosamente:\n{file_path}"
                )
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error generando reporte: {str(e)}")
    
    def send_support_email(self):
        """Abre el cliente de email para enviar soporte"""
        try:
            subject = f"Soporte SEACABAr - {self.problem_type.currentText()}"
            body = f"""
Tipo de Problema: {self.problem_type.currentText()}

Descripción:
{self.problem_description.toPlainText()}

---
Sistema: {platform.system()} {platform.version()}
Python: {sys.version.split()[0]}
            """.strip()
            
            mailto_url = f"mailto:soporte@seacabar.com?subject={subject}&body={body}"
            QDesktopServices.openUrl(QUrl(mailto_url))
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error abriendo cliente de email: {str(e)}")

class HelpDialog(QDialog):
    """Diálogo principal de ayuda"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Ayuda - SEACABAr")
        self.setModal(False)
        self.resize(1000, 700)
        
        self.setup_ui()
    
    def setup_ui(self):
        """Configura la interfaz de usuario"""
        layout = QVBoxLayout(self)
        
        # Crear pestañas principales
        self.tab_widget = QTabWidget()
        
        # Añadir pestañas
        self.user_guide = UserGuideWidget()
        self.technical_docs = TechnicalDocsWidget()
        self.faq = FAQWidget()
        self.system_info = SystemInfoWidget()
        self.support = SupportWidget()
        
        self.tab_widget.addTab(self.user_guide, "Guía de Usuario")
        self.tab_widget.addTab(self.technical_docs, "Documentación Técnica")
        self.tab_widget.addTab(self.faq, "FAQ")
        self.tab_widget.addTab(self.system_info, "Información del Sistema")
        self.tab_widget.addTab(self.support, "Soporte Técnico")
        
        layout.addWidget(self.tab_widget)
        
        # Botones
        buttons_layout = QHBoxLayout()
        
        self.print_btn = QPushButton("Imprimir")
        self.print_btn.clicked.connect(self.print_help)
        buttons_layout.addWidget(self.print_btn)
        
        buttons_layout.addStretch()
        
        self.close_btn = QPushButton("Cerrar")
        self.close_btn.clicked.connect(self.accept)
        buttons_layout.addWidget(self.close_btn)
        
        layout.addLayout(buttons_layout)
    
    def print_help(self):
        """Imprime la ayuda actual"""
        try:
            current_widget = self.tab_widget.currentWidget()
            if hasattr(current_widget, 'content_browser'):
                # Para widgets con QTextBrowser
                current_widget.content_browser.print_()
            elif hasattr(current_widget, 'faq_browser'):
                # Para FAQ
                current_widget.faq_browser.print_()
            else:
                QMessageBox.information(
                    self,
                    "Información",
                    "La impresión no está disponible para esta sección."
                )
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error imprimiendo: {str(e)}")
    
    def show_section(self, section: str):
        """Muestra una sección específica"""
        section_map = {
            "user_guide": 0,
            "technical": 1,
            "faq": 2,
            "system": 3,
            "support": 4
        }
        
        if section in section_map:
            self.tab_widget.setCurrentIndex(section_map[section])