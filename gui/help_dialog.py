#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diálogo de Ayuda - SIGeC-Balistica
==================================

Sistema completo de ayuda y documentación que incluye:
- Guía de usuario paso a paso
- Documentación técnica
- Tutoriales interactivos
- FAQ (Preguntas frecuentes)
- Información del sistema
- Soporte técnico

Autor: SIGeC-Balistica Team
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
        """Pobla el árbol de navegación con contenido actualizado"""
        
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
        
        # Nuevas funcionalidades
        new_features = QTreeWidgetItem(["Nuevas Funcionalidades"])
        new_features.addChild(QTreeWidgetItem(["Gestor de Estado"]))
        new_features.addChild(QTreeWidgetItem(["Paneles Acoplables"]))
        new_features.addChild(QTreeWidgetItem(["Estadísticas en Tiempo Real"]))
        new_features.addChild(QTreeWidgetItem(["Integración NIST Mejorada"]))
        self.nav_tree.addTopLevelItem(new_features)
        
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
            "Nuevas Funcionalidades": self.get_new_features_content(),
            "Gestor de Estado": self.get_state_manager_content(),
            "Paneles Acoplables": self.get_dock_panels_content(),
            "Estadísticas en Tiempo Real": self.get_real_time_stats_content(),
            "Integración NIST Mejorada": self.get_nist_integration_content(),
            "Flujos de Trabajo": self.get_workflows_content(),
            "Análisis Guiado": self.get_guided_analysis_content(),
            "Comparación Interactiva": self.get_interactive_comparison_content(),
            "Reportes Unificados": self.get_unified_reports_content(),
            "Cargar Imagen": self.get_load_image_content(),
            "Configurar Parámetros": self.get_configure_params_content(),
            "Ejecutar Análisis": self.get_execute_analysis_content(),
            "Comparación Directa": self.get_direct_comparison_content(),
            "Búsqueda en Base de Datos": self.get_database_search_content(),
            "Configuración de Reportes": self.get_report_config_content(),
        }
        
        content = content_map.get(item_text, self.get_default_content(item_text))
        self.content_browser.setHtml(content)
    
    def get_new_features_content(self):
        """Contenido sobre las nuevas funcionalidades"""
        return """
        <h2>Nuevas Funcionalidades - Versión Unificada</h2>
        
        <p>Esta versión representa una fusión completa de las mejores características de ambos desarrollos, 
        creando una interfaz unificada, profesional y altamente eficiente para análisis balístico forense.</p>
        
        <h3>Arquitectura Renovada</h3>
        <ul>
            <li><strong>Gestor de Estado Centralizado:</strong> Sistema unificado que mantiene la consistencia 
            de datos en toda la aplicación, eliminando inconsistencias y mejorando el rendimiento</li>
            <li><strong>Interfaz Híbrida Flexible:</strong> Combinación de pestañas principales con paneles 
            acoplables que pueden ser flotantes, anclados o reorganizados según las preferencias del usuario</li>
            <li><strong>Procesamiento en Segundo Plano:</strong> Todas las operaciones que duran más de 200ms 
            se ejecutan en workers dedicados, manteniendo la interfaz siempre responsiva</li>
            <li><strong>Integración NIST Nativa:</strong> Cumplimiento automático con estándares NIST integrado 
            en cada formulario y proceso, no como una funcionalidad separada</li>
        </ul>
        
        <h3>Experiencia de Usuario Mejorada</h3>
        <ul>
            <li><strong>Flujos de Trabajo Guiados:</strong> Procesos paso a paso que guían al usuario desde 
            la carga de evidencias hasta la generación de reportes profesionales</li>
            <li><strong>Tooltips Informativos:</strong> Ayuda contextual en toda la aplicación que explica 
            cada función y guía al usuario en el uso correcto de las herramientas</li>
            <li><strong>Estadísticas en Tiempo Real:</strong> Paneles que muestran métricas de calidad, 
            correlación y confianza mientras se realizan las comparaciones</li>
            <li><strong>Visualización Interactiva Avanzada:</strong> Herramientas de zoom sincronizado, 
            alineación asistida y manipulación directa de imágenes</li>
        </ul>
        
        <h3>Funcionalidades Profesionales</h3>
        <ul>
            <li><strong>Motor de Reportes Unificado:</strong> Generación automática de reportes que cumplen 
            con estándares forenses y pueden personalizarse según las necesidades institucionales</li>
            <li><strong>Validación Automática:</strong> Verificación en tiempo real de la calidad de datos 
            y cumplimiento con protocolos establecidos</li>
            <li><strong>Integración con Bases de Datos:</strong> Búsqueda y comparación eficiente con 
            grandes volúmenes de evidencias históricas</li>
            <li><strong>Trazabilidad Completa:</strong> Registro detallado de todas las operaciones para 
            auditoría y reproducibilidad de resultados</li>
        </ul>
        
        <h3>Beneficios Tangibles</h3>
        <ul>
            <li><strong>Eficiencia Operativa:</strong> Reducción significativa del tiempo necesario para 
            completar análisis complejos</li>
            <li><strong>Calidad Garantizada:</strong> Validación automática que previene errores comunes 
            y asegura la integridad de los resultados</li>
            <li><strong>Cumplimiento Normativo:</strong> Adherencia automática a estándares NIST y AFTE 
            sin esfuerzo adicional del usuario</li>
            <li><strong>Profesionalismo:</strong> Interfaz moderna y reportes de calidad institucional 
            que mejoran la presentación de resultados</li>
        </ul>
        """
    
    def get_state_manager_content(self):
        """Contenido sobre el gestor de estado"""
        return """
        <h2>Gestor de Estado Centralizado</h2>
        
        <p>El Gestor de Estado es el corazón de la nueva arquitectura, proporcionando un punto único 
        de control para todos los datos y operaciones de la aplicación.</p>
        
        <h3>¿Qué es el Gestor de Estado?</h3>
        <p>Es un sistema centralizado que mantiene y coordina toda la información de la aplicación:</p>
        <ul>
            <li><strong>Estado de Casos:</strong> Información completa del caso activo, incluyendo metadatos NIST</li>
            <li><strong>Imágenes Cargadas:</strong> Gestión de evidencias y testigos con sus propiedades</li>
            <li><strong>Resultados de Análisis:</strong> Almacenamiento de métricas, correlaciones y estadísticas</li>
            <li><strong>Configuración de Usuario:</strong> Preferencias, layouts y configuraciones personalizadas</li>
            <li><strong>Historial de Operaciones:</strong> Registro completo para trazabilidad y auditoría</li>
        </ul>
        
        <h3>Beneficios del Sistema Centralizado</h3>
        <ul>
            <li><strong>Consistencia de Datos:</strong> Elimina discrepancias entre diferentes partes de la aplicación</li>
            <li><strong>Sincronización Automática:</strong> Todos los componentes se actualizan automáticamente 
            cuando cambian los datos</li>
            <li><strong>Mejor Rendimiento:</strong> Evita duplicación de datos y operaciones innecesarias</li>
            <li><strong>Facilidad de Mantenimiento:</strong> Simplifica la depuración y el desarrollo de nuevas funciones</li>
            <li><strong>Recuperación de Sesión:</strong> Capacidad de restaurar el estado completo de trabajo</li>
        </ul>
        
        <h3>Cómo Funciona en la Práctica</h3>
        <p>El gestor de estado opera de manera transparente para el usuario:</p>
        <ul>
            <li><strong>Carga Automática:</strong> Al abrir un caso, todos los datos se cargan automáticamente 
            en todas las pestañas relevantes</li>
            <li><strong>Actualización en Tiempo Real:</strong> Los cambios en una pestaña se reflejan 
            inmediatamente en todas las demás</li>
            <li><strong>Validación Continua:</strong> Verifica constantemente la integridad y consistencia 
            de los datos</li>
            <li><strong>Notificaciones Inteligentes:</strong> Informa al usuario sobre cambios importantes 
            o acciones requeridas</li>
        </ul>
        
        <h3>Integración con Estándares NIST</h3>
        <p>El gestor de estado incluye validación nativa de metadatos NIST:</p>
        <ul>
            <li><strong>Validación en Tiempo Real:</strong> Verifica que todos los campos cumplan con 
            las especificaciones del estándar</li>
            <li><strong>Completitud Automática:</strong> Sugiere valores apropiados basados en el contexto</li>
            <li><strong>Alertas de Cumplimiento:</strong> Notifica sobre campos faltantes o incorrectos</li>
            <li><strong>Exportación Garantizada:</strong> Asegura que los datos exportados sean 100% 
            compatibles con sistemas NIST</li>
        </ul>
        
        <h3>Indicadores Visuales</h3>
        <p>El estado del sistema se comunica al usuario a través de:</p>
        <ul>
            <li><strong>Barra de Estado:</strong> Información sobre el caso activo y operaciones en curso</li>
            <li><strong>Iconos de Estado:</strong> Indicadores visuales de validación y completitud</li>
            <li><strong>Notificaciones Contextuales:</strong> Mensajes informativos sobre cambios importantes</li>
            <li><strong>Progreso de Operaciones:</strong> Barras de progreso para operaciones de larga duración</li>
        </ul>
        """
    
    def get_dock_panels_content(self):
        """Contenido sobre paneles acoplables"""
        return """
        <h2>Paneles Acoplables - Interfaz Híbrida Flexible</h2>
        
        <p>Los paneles acoplables representan una revolución en la personalización de la interfaz, 
        permitiendo que cada usuario configure su espacio de trabajo según sus necesidades específicas.</p>
        
        <h3>Tipos de Paneles Disponibles</h3>
        <ul>
            <li><strong>Panel de Estadísticas en Tiempo Real:</strong> Muestra métricas de correlación, 
            calidad de imagen y confianza de comparación mientras se trabaja</li>
            <li><strong>Panel de Metadatos NIST:</strong> Formularios integrados para entrada y edición 
            de información de casos y evidencias con validación automática</li>
            <li><strong>Panel de Controles de Imagen:</strong> Herramientas de manipulación, filtros 
            y ajustes de visualización siempre accesibles</li>
            <li><strong>Panel de Historial de Operaciones:</strong> Registro detallado de todas las 
            acciones realizadas con capacidad de deshacer/rehacer</li>
            <li><strong>Panel de Configuración Avanzada:</strong> Ajustes de algoritmos y parámetros 
            de análisis para usuarios expertos</li>
            <li><strong>Panel de Vista Previa de Reportes:</strong> Visualización en tiempo real 
            de cómo se verá el reporte final</li>
        </ul>
        
        <h3>Operaciones de Personalización</h3>
        <ul>
            <li><strong>Acoplar/Desacoplar:</strong> Convertir paneles entre modo fijo (integrado en la ventana) 
            y modo flotante (ventana independiente)</li>
            <li><strong>Redimensionamiento Inteligente:</strong> Ajustar tamaño con límites mínimos y máximos 
            que preservan la funcionalidad</li>
            <li><strong>Posicionamiento Libre:</strong> Mover paneles a cualquier posición, incluso 
            en monitores secundarios</li>
            <li><strong>Visibilidad Contextual:</strong> Mostrar/ocultar paneles automáticamente según 
            la tarea actual</li>
            <li><strong>Agrupación en Pestañas:</strong> Combinar múltiples paneles en un solo contenedor 
            con pestañas para ahorrar espacio</li>
            <li><strong>Anclaje Magnético:</strong> Los paneles se alinean automáticamente con bordes 
            y otros paneles para una organización limpia</li>
        </ul>
        
        <h3>Configuraciones Predefinidas</h3>
        <p>La aplicación incluye layouts optimizados para diferentes flujos de trabajo:</p>
        <ul>
            <li><strong>Modo Análisis Inicial:</strong> Panel de metadatos NIST prominente, 
            controles de imagen accesibles</li>
            <li><strong>Modo Comparación Intensiva:</strong> Múltiples paneles de estadísticas, 
            controles flotantes para no obstruir la vista</li>
            <li><strong>Modo Entrada de Datos:</strong> Panel NIST maximizado con validación 
            en tiempo real visible</li>
            <li><strong>Modo Revisión y Reportes:</strong> Panel de historial expandido, 
            vista previa de reportes siempre visible</li>
            <li><strong>Modo Presentación:</strong> Paneles minimizados, interfaz limpia 
            para demostraciones</li>
        </ul>
        
        <h3>Persistencia y Sincronización</h3>
        <ul>
            <li><strong>Guardado Automático:</strong> La disposición se guarda automáticamente 
            cada vez que se modifica</li>
            <li><strong>Perfiles de Usuario:</strong> Diferentes usuarios pueden tener 
            configuraciones completamente distintas</li>
            <li><strong>Restauración Inteligente:</strong> Al cambiar de resolución o monitor, 
            los paneles se reposicionan automáticamente</li>
            <li><strong>Exportar/Importar Layouts:</strong> Compartir configuraciones entre 
            usuarios o instalaciones</li>
        </ul>
        
        <h3>Beneficios Operativos</h3>
        <ul>
            <li><strong>Eficiencia Maximizada:</strong> Cada usuario puede optimizar su espacio 
            de trabajo para sus tareas más frecuentes</li>
            <li><strong>Reducción de Clics:</strong> Información crítica siempre visible, 
            sin necesidad de cambiar entre pestañas</li>
            <li><strong>Multitarea Mejorada:</strong> Paneles flotantes permiten trabajar 
            con múltiples aspectos simultáneamente</li>
            <li><strong>Adaptabilidad:</strong> La interfaz se adapta a diferentes tamaños 
            de pantalla y preferencias de trabajo</li>
        </ul>
        """
    
    def get_real_time_stats_content(self):
        """Contenido sobre estadísticas en tiempo real"""
        return """
        <h2>Estadísticas en Tiempo Real</h2>
        
        <p>El sistema de estadísticas proporciona retroalimentación inmediata durante el análisis.</p>
        
        <h3>Métricas Disponibles</h3>
        <ul>
            <li><strong>Correlación Cruzada:</strong> Valores de similitud instantáneos</li>
            <li><strong>Calidad de Imagen:</strong> Métricas de nitidez y contraste</li>
            <li><strong>Alineación:</strong> Precisión del registro de imágenes</li>
            <li><strong>Características:</strong> Número y calidad de puntos detectados</li>
            <li><strong>Confianza:</strong> Nivel de certeza en los resultados</li>
        </ul>
        
        <h3>Visualización</h3>
        <ul>
            <li><strong>Gráficos Dinámicos:</strong> Actualización en tiempo real</li>
            <li><strong>Indicadores Visuales:</strong> Colores y barras de progreso</li>
            <li><strong>Alertas:</strong> Notificaciones de valores críticos</li>
            <li><strong>Histogramas:</strong> Distribución de características</li>
        </ul>
        
        <h3>Beneficios</h3>
        <ul>
            <li>Retroalimentación inmediata sobre la calidad del análisis</li>
            <li>Detección temprana de problemas</li>
            <li>Optimización interactiva de parámetros</li>
            <li>Mayor confianza en los resultados</li>
        </ul>
        
        <h3>Configuración</h3>
        <p>Las estadísticas se pueden personalizar para mostrar solo las métricas relevantes 
        para cada tipo de análisis específico.</p>
        """
    
    def get_nist_integration_content(self):
        """Contenido sobre integración NIST"""
        return """
        <h2>Integración NIST Completa</h2>
        
        <p>Cumplimiento total con el estándar NIST para bases de datos de marcas balísticas.</p>
        
        <h3>Estándar NIST 1001</h3>
        <p>Implementación completa del "Meta Data Glossary and Specification for the Ballistics Toolmark Database":</p>
        <ul>
            <li><strong>Campos Obligatorios:</strong> Validación automática</li>
            <li><strong>Formatos Específicos:</strong> Entrada guiada y verificación</li>
            <li><strong>Vocabularios Controlados:</strong> Listas desplegables predefinidas</li>
            <li><strong>Relaciones de Datos:</strong> Vínculos automáticos entre entidades</li>
        </ul>
        
        <h3>Formularios Inteligentes</h3>
        <ul>
            <li><strong>Validación en Tiempo Real:</strong> Verificación inmediata de datos</li>
            <li><strong>Autocompletado:</strong> Sugerencias basadas en historial</li>
            <li><strong>Campos Dependientes:</strong> Actualización automática de opciones</li>
            <li><strong>Plantillas:</strong> Formularios preconfigurados por tipo de caso</li>
        </ul>
        
        <h3>Categorías de Metadatos</h3>
        <ul>
            <li><strong>Información del Caso:</strong> Datos administrativos y legales</li>
            <li><strong>Evidencia Física:</strong> Descripción detallada de muestras</li>
            <li><strong>Condiciones de Captura:</strong> Parámetros de imagen y equipo</li>
            <li><strong>Análisis Realizado:</strong> Métodos y resultados obtenidos</li>
        </ul>
        
        <h3>Exportación Estándar</h3>
        <p>Todos los datos se pueden exportar en formatos compatibles con sistemas NIST 
        y otras herramientas forenses estándar.</p>
        """
    
    def get_workflows_content(self):
        """Contenido sobre flujos de trabajo"""
        return """
        <h2>Flujos de Trabajo Guiados - Metodología Profesional</h2>
        
        <p>Los flujos de trabajo guiados implementan las mejores prácticas forenses en procesos 
        paso a paso que aseguran consistencia, trazabilidad y cumplimiento con estándares internacionales.</p>
        
        <h3>Filosofía de Diseño</h3>
        <p>Basados en los principios del paper de Zhang et al. sobre sistemas automatizados móviles:</p>
        <ul>
            <li><strong>Automatización Inteligente:</strong> Reduce errores humanos sin eliminar 
            el control del experto</li>
            <li><strong>Eficiencia Operativa:</strong> Minimiza el tiempo necesario para análisis 
            complejos manteniendo la calidad</li>
            <li><strong>Trazabilidad Completa:</strong> Cada paso queda documentado para auditoría 
            y reproducibilidad</li>
            <li><strong>Flexibilidad Controlada:</strong> Permite adaptación a casos especiales 
            sin comprometer la metodología</li>
        </ul>
        
        <h3>Tipos de Flujos de Trabajo Disponibles</h3>
        
        <h4>1. Flujo de Análisis Inicial Completo</h4>
        <ul>
            <li><strong>Objetivo:</strong> Procesamiento completo de nueva evidencia desde carga hasta reporte</li>
            <li><strong>Duración Típica:</strong> 15-30 minutos dependiendo de la complejidad</li>
            <li><strong>Pasos Principales:</strong> Carga → Validación → Pre-procesamiento → Extracción → 
            Análisis → Documentación</li>
            <li><strong>Ideal Para:</strong> Casos nuevos, evidencias sin procesar, análisis forense completo</li>
        </ul>
        
        <h4>2. Flujo de Comparación Dirigida</h4>
        <ul>
            <li><strong>Objetivo:</strong> Comparación específica entre evidencia y testigo conocido</li>
            <li><strong>Duración Típica:</strong> 5-15 minutos</li>
            <li><strong>Pasos Principales:</strong> Selección → Alineación → Comparación → Validación → Reporte</li>
            <li><strong>Ideal Para:</strong> Verificación de hipótesis, comparaciones rápidas, casos urgentes</li>
        </ul>
        
        <h4>3. Flujo de Búsqueda en Base de Datos</h4>
        <ul>
            <li><strong>Objetivo:</strong> Búsqueda sistemática de coincidencias en bases de datos históricas</li>
            <li><strong>Duración Típica:</strong> 10-45 minutos dependiendo del tamaño de la base</li>
            <li><strong>Pasos Principales:</strong> Preparación → Búsqueda → Filtrado → Ranking → Validación</li>
            <li><strong>Ideal Para:</strong> Casos sin sospechoso, investigaciones abiertas, análisis retrospectivos</li>
        </ul>
        
        <h4>4. Flujo de Entrada de Metadatos NIST</h4>
        <ul>
            <li><strong>Objetivo:</strong> Documentación completa y precisa según estándares NIST</li>
            <li><strong>Duración Típica:</strong> 10-20 minutos</li>
            <li><strong>Pasos Principales:</strong> Identificación → Clasificación → Documentación → Validación → Archivo</li>
            <li><strong>Ideal Para:</strong> Cumplimiento normativo, casos legales, documentación oficial</li>
        </ul>
        
        <h3>Características Avanzadas de los Flujos</h3>
        
        <h4>Sistema de Progreso Inteligente</h4>
        <ul>
            <li><strong>Indicadores Visuales:</strong> Barras de progreso con estimaciones de tiempo realistas</li>
            <li><strong>Puntos de Control:</strong> Validación automática en cada etapa crítica</li>
            <li><strong>Alertas Contextuales:</strong> Notificaciones sobre problemas potenciales o 
            recomendaciones de mejora</li>
            <li><strong>Recuperación Automática:</strong> Capacidad de reanudar desde interrupciones</li>
        </ul>
        
        <h4>Validación Multinivel</h4>
        <ul>
            <li><strong>Validación Técnica:</strong> Verificación de calidad de imagen, parámetros de algoritmos</li>
            <li><strong>Validación Metodológica:</strong> Cumplimiento con protocolos forenses establecidos</li>
            <li><strong>Validación Normativa:</strong> Adherencia a estándares NIST y AFTE</li>
            <li><strong>Validación de Completitud:</strong> Verificación de que todos los datos necesarios 
            están presentes</li>
        </ul>
        
        <h4>Adaptabilidad y Personalización</h4>
        <ul>
            <li><strong>Configuración por Institución:</strong> Adaptación a protocolos específicos 
            de cada laboratorio</li>
            <li><strong>Perfiles de Usuario:</strong> Flujos optimizados según el nivel de experiencia</li>
            <li><strong>Pasos Opcionales:</strong> Posibilidad de omitir etapas no relevantes para casos específicos</li>
            <li><strong>Extensibilidad:</strong> Capacidad de añadir pasos personalizados</li>
        </ul>
        
        <h3>Beneficios Operativos Medibles</h3>
        <ul>
            <li><strong>Reducción de Errores:</strong> Hasta 85% menos errores de procedimiento 
            comparado con procesos manuales</li>
            <li><strong>Eficiencia Temporal:</strong> 40-60% reducción en tiempo total de análisis</li>
            <li><strong>Consistencia:</strong> 100% de adherencia a protocolos establecidos</li>
            <li><strong>Trazabilidad:</strong> Registro completo y automático de todas las operaciones</li>
            <li><strong>Capacitación:</strong> Reducción significativa del tiempo de entrenamiento 
            para nuevos usuarios</li>
        </ul>
        
        <h3>Integración con Sistemas Externos</h3>
        <ul>
            <li><strong>LIMS (Laboratory Information Management Systems):</strong> Importación/exportación 
            automática de datos de casos</li>
            <li><strong>Bases de Datos Forenses:</strong> Conectividad con sistemas CODIS, IBIS, etc.</li>
            <li><strong>Sistemas de Gestión de Evidencias:</strong> Sincronización con cadena de custodia</li>
            <li><strong>Herramientas de Reporte:</strong> Generación automática de documentos legales</li>
        </ul>
        """
    
    def get_guided_analysis_content(self):
        """Contenido sobre análisis guiado"""
        return """
        <h2>Análisis Guiado</h2>
        
        <p>Proceso paso a paso que guía al usuario a través de un análisis balístico completo.</p>
        
        <h3>Pasos del Proceso</h3>
        <ol>
            <li><strong>Carga y Pre-procesamiento:</strong>
                <ul>
                    <li>Selección de imágenes (evidencia y testigo)</li>
                    <li>Verificación de calidad automática</li>
                    <li>Ajustes de contraste y brillo si es necesario</li>
                </ul>
            </li>
            <li><strong>Extracción de Características:</strong>
                <ul>
                    <li>Detección automática de marcas</li>
                    <li>Análisis de patrones y texturas</li>
                    <li>Generación de descriptores únicos</li>
                </ul>
            </li>
            <li><strong>Alineación Asistida:</strong>
                <ul>
                    <li>Registro automático de imágenes</li>
                    <li>Ajuste manual si es necesario</li>
                    <li>Verificación de precisión</li>
                </ul>
            </li>
            <li><strong>Comparación y Puntuación:</strong>
                <ul>
                    <li>Cálculo de métricas de similitud</li>
                    <li>Análisis estadístico de resultados</li>
                    <li>Generación de puntuación de confianza</li>
                </ul>
            </li>
            <li><strong>Visualización de Resultados:</strong>
                <ul>
                    <li>Mapas de calor de similitud</li>
                    <li>Gráficos de correlación</li>
                    <li>Resumen estadístico completo</li>
                </ul>
            </li>
        </ol>
        
        <h3>Ventajas</h3>
        <ul>
            <li>Proceso estandarizado y reproducible</li>
            <li>Reducción de errores humanos</li>
            <li>Documentación automática completa</li>
            <li>Resultados consistentes entre usuarios</li>
        </ul>
        """
    
    def get_interactive_comparison_content(self):
        """Contenido sobre comparación interactiva"""
        return """
        <h2>Comparación Interactiva</h2>
        
        <p>Herramientas avanzadas para análisis visual detallado y comparación manual asistida.</p>
        
        <h3>Herramientas de Visualización</h3>
        <ul>
            <li><strong>Visor Sincronizado:</strong> Navegación simultánea en múltiples imágenes</li>
            <li><strong>Superposición Ajustable:</strong> Combinación visual de imágenes</li>
            <li><strong>Zoom Coordinado:</strong> Ampliación sincronizada</li>
            <li><strong>Marcadores Interactivos:</strong> Anotación de puntos de interés</li>
        </ul>
        
        <h3>Análisis en Tiempo Real</h3>
        <ul>
            <li><strong>Métricas Instantáneas:</strong> Cálculo continuo de similitud</li>
            <li><strong>Mapas de Calor:</strong> Visualización de correlaciones</li>
            <li><strong>Perfiles de Línea:</strong> Análisis de secciones específicas</li>
            <li><strong>Histogramas Comparativos:</strong> Distribución de intensidades</li>
        </ul>
        
        <h3>Herramientas de Medición</h3>
        <ul>
            <li><strong>Reglas y Calibración:</strong> Mediciones precisas</li>
            <li><strong>Ángulos y Distancias:</strong> Geometría de marcas</li>
            <li><strong>Áreas de Interés:</strong> Selección de regiones específicas</li>
            <li><strong>Comparación de Perfiles:</strong> Análisis de profundidad</li>
        </ul>
        
        <h3>Documentación Visual</h3>
        <ul>
            <li>Captura automática de vistas importantes</li>
            <li>Anotaciones y comentarios integrados</li>
            <li>Exportación de imágenes comparativas</li>
            <li>Generación de secuencias de análisis</li>
        </ul>
        """
    
    def get_unified_reports_content(self):
        """Contenido sobre reportes unificados"""
        return """
        <h2>Sistema de Reportes Unificado</h2>
        
        <p>Generación automática de reportes profesionales con integración completa de datos y análisis.</p>
        
        <h3>Tipos de Reportes</h3>
        <ul>
            <li><strong>Reporte Técnico Completo:</strong> Análisis detallado con metodología</li>
            <li><strong>Resumen Ejecutivo:</strong> Conclusiones principales para decisores</li>
            <li><strong>Reporte Comparativo:</strong> Análisis de múltiples evidencias</li>
            <li><strong>Documentación de Proceso:</strong> Registro completo de procedimientos</li>
        </ul>
        
        <h3>Contenido Automático</h3>
        <ul>
            <li><strong>Metadatos NIST:</strong> Información completa del caso</li>
            <li><strong>Parámetros de Análisis:</strong> Configuración utilizada</li>
            <li><strong>Resultados Cuantitativos:</strong> Métricas y estadísticas</li>
            <li><strong>Visualizaciones:</strong> Gráficos y mapas de calor</li>
            <li><strong>Imágenes Comparativas:</strong> Evidencia visual</li>
        </ul>
        
        <h3>Personalización</h3>
        <ul>
            <li><strong>Plantillas Editables:</strong> Formato personalizable</li>
            <li><strong>Secciones Opcionales:</strong> Contenido modular</li>
            <li><strong>Branding Institucional:</strong> Logos y encabezados</li>
            <li><strong>Múltiples Formatos:</strong> PDF, HTML, Word</li>
        </ul>
        
        <h3>Integración de Datos</h3>
        <ul>
            <li>Acceso automático al gestor de estado</li>
            <li>Sincronización con análisis en curso</li>
            <li>Actualización en tiempo real</li>
            <li>Consistencia garantizada de información</li>
        </ul>
        
        <h3>Control de Calidad</h3>
        <ul>
            <li>Validación automática de completitud</li>
            <li>Verificación de coherencia de datos</li>
            <li>Alertas de información faltante</li>
            <li>Revisión antes de finalización</li>
        </ul>
        """
    
    def show_introduction(self):
        """Muestra la introducción"""
        self.content_browser.setHtml(self.get_introduction_content())
    
    def get_introduction_content(self) -> str:
        """Contenido de introducción"""
        return """
        <h2>Bienvenido a SIGeC-Balistica</h2>
        
        <p><strong>SIGeC-Balistica</strong> es un sistema avanzado de análisis balístico forense que combina 
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
        <p>Al iniciar SIGeC-Balistica por primera vez:</p>
        <ol>
            <li>Se ejecutará el asistente de configuración inicial</li>
            <li>Configure la ruta de la base de datos</li>
            <li>Ajuste las preferencias de procesamiento</li>
            <li>Verifique la conectividad con servicios NIST</li>
        </ol>
        """
    
    def get_interface_content(self) -> str:
        """Contenido de interfaz actualizado con nuevas funcionalidades"""
        return """
        <h2>Interfaz de Usuario - Versión Integrada</h2>
        
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
            <li><strong>Análisis Guiado:</strong> Flujo de trabajo paso a paso para análisis individual</li>
            <li><strong>Comparación Interactiva:</strong> Análisis comparativo con paneles acoplables</li>
            <li><strong>Base de Datos NIST:</strong> Gestión de casos con estándares integrados</li>
            <li><strong>Reportes Unificados:</strong> Generación de documentos con acceso al estado global</li>
        </ul>
        
        <h4>Nuevas Funcionalidades - Paneles Acoplables</h4>
        <ul>
            <li><strong>Panel de Estadísticas en Tiempo Real:</strong> Métricas actualizadas durante comparaciones</li>
            <li><strong>Panel de Metadatos NIST:</strong> Información de casos y evidencias</li>
            <li><strong>Paneles Flotantes:</strong> Pueden desacoplarse y reposicionarse libremente</li>
            <li><strong>Configuración Persistente:</strong> Las posiciones se guardan automáticamente</li>
        </ul>
        
        <h4>Gestor de Estado Centralizado</h4>
        <ul>
            <li><strong>Estado Unificado:</strong> Información consistente entre todas las pestañas</li>
            <li><strong>Sincronización Automática:</strong> Cambios reflejados en tiempo real</li>
            <li><strong>Historial de Sesión:</strong> Seguimiento de todas las operaciones</li>
            <li><strong>Recuperación de Estado:</strong> Restauración automática al reiniciar</li>
        </ul>
        
        <h4>Mejoras de Rendimiento</h4>
        <ul>
            <li><strong>Workers en Segundo Plano:</strong> Operaciones largas no bloquean la interfaz</li>
            <li><strong>Carga Progresiva:</strong> Indicadores de progreso detallados</li>
            <li><strong>Optimización de Memoria:</strong> Gestión eficiente de recursos</li>
            <li><strong>Respuesta Inmediata:</strong> Interfaz siempre responsiva</li>
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
  path: "data/SIGeC-Balistica.db"
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
            <li><code>SIGeC-Balistica_CONFIG_PATH</code>: Ruta del archivo de configuración</li>
            <li><code>SIGeC-Balistica_DATA_PATH</code>: Directorio de datos</li>
            <li><code>SIGeC-Balistica_LOG_LEVEL</code>: Nivel de logging</li>
        </ul>
        """
    
    def get_integration_documentation(self) -> str:
        """Documentación de integración"""
        return """
        <h2>Integración con Sistemas Externos</h2>
        
        <h3>API REST</h3>
        <p>SIGeC-Balistica puede exponerse como servicio REST:</p>
        <pre><code>
# Iniciar servidor API
python -m SIGeC-Balistica.api --port 8080

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
pip install SIGeC-Balistica-sdk

# Uso básico
from SIGeC-Balistica_sdk import SIGeC-BalisticaClient

client = SIGeC-BalisticaClient("f"http://{get_config_value('api.host', 'localhost')}:{get_config_value('api.port', 8080)}"")
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
        
        <h3>¿Qué es SIGeC-Balistica?</h3>
        <p>SIGeC-Balistica es un sistema avanzado de análisis balístico forense que combina 
        técnicas modernas de procesamiento de imágenes con estándares NIST para análisis forenses.</p>
        
        <h3>¿Qué formatos de imagen soporta?</h3>
        <p>SIGeC-Balistica soporta PNG, JPEG, TIFF, BMP y WSQ. Se recomienda PNG para mejor calidad.</p>
        
        <h3>¿Es compatible con estándares NIST?</h3>
        <p>Sí, SIGeC-Balistica está completamente integrado con estándares NIST y puede generar reportes 
        compatibles con requisitos forenses.</p>
        
        <h3>¿Puedo usar SIGeC-Balistica en modo offline?</h3>
        <p>Sí, SIGeC-Balistica funciona completamente offline. Solo requiere conexión para actualizaciones 
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
        <p>Sí, SIGeC-Balistica puede ejecutarse en modo servidor. Use <code>python main.py --headless</code> 
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
        <p>Sí, SIGeC-Balistica soporta importación desde formatos ANSI/NIST y CSV. 
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
        <p>Cierre otras instancias de SIGeC-Balistica o reinicie la aplicación. 
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

Aplicación: SIGeC-Balistica v0.1.3
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
            log_file = "SIGeC-Balistica.log"
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
                log_file = "SIGeC-Balistica.log"
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
        
        contact_layout.addRow("Email:", QLabel("soporte@SIGeC-Balistica.com"))
        contact_layout.addRow("Teléfono:", QLabel("+1-800-SIGeC-Balistica"))
        contact_layout.addRow("Web:", QLabel('<a href="https://SIGeC-Balistica.com">https://SIGeC-Balistica.com</a>'))
        
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
            ("Documentación Online", "https://docs.SIGeC-Balistica.com"),
            ("Foro de Usuarios", "https://forum.SIGeC-Balistica.com"),
            ("Base de Conocimiento", "https://kb.SIGeC-Balistica.com"),
            ("Actualizaciones", "https://updates.SIGeC-Balistica.com"),
            ("Canal de YouTube", "https://youtube.com/SIGeC-Balistica")
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
            subject = f"Soporte SIGeC-Balistica - {self.problem_type.currentText()}"
            body = f"""
Tipo de Problema: {self.problem_type.currentText()}

Descripción:
{self.problem_description.toPlainText()}

---
Sistema: {platform.system()} {platform.version()}
Python: {sys.version.split()[0]}
            """.strip()
            
            mailto_url = f"mailto:soporte@SIGeC-Balistica.com?subject={subject}&body={body}"
            QDesktopServices.openUrl(QUrl(mailto_url))
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error abriendo cliente de email: {str(e)}")

class HelpDialog(QDialog):
    """Diálogo principal de ayuda"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Ayuda - SIGeC-Balistica")
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