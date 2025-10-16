# -*- coding: utf-8 -*-
"""
Gestor de Tooltips para la Aplicación GUI
Proporciona tooltips informativos y contextuales para mejorar la experiencia del usuario.
"""

from PyQt5.QtWidgets import QWidget, QToolTip, QApplication
from PyQt5.QtCore import QTimer, QPoint
from PyQt5.QtGui import QFont, QPalette
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

class TooltipManager:
    """Gestor centralizado de tooltips para la aplicación"""
    
    def __init__(self):
        self.tooltips = {}
        self.setup_tooltip_style()
        
    def setup_tooltip_style(self):
        """Configura el estilo global de los tooltips"""
        # Configurar fuente para tooltips
        font = QFont()
        font.setPointSize(9)
        QToolTip.setFont(font)
        
    def register_widget_tooltips(self, widget_class_name: str, tooltips_dict: Dict[str, str]):
        """Registra tooltips para una clase de widget específica"""
        self.tooltips[widget_class_name] = tooltips_dict
        
    def apply_tooltips_to_widget(self, widget: QWidget, widget_class_name: str):
        """Aplica tooltips a un widget específico basado en su clase"""
        if widget_class_name not in self.tooltips:
            return
            
        tooltips_dict = self.tooltips[widget_class_name]
        
        for object_name, tooltip_text in tooltips_dict.items():
            child_widget = widget.findChild(QWidget, object_name)
            if child_widget:
                child_widget.setToolTip(tooltip_text)
                logger.debug(f"Tooltip aplicado a {object_name}: {tooltip_text[:50]}...")

# Instancia global del gestor de tooltips
tooltip_manager = TooltipManager()

# Definición de tooltips por clase de widget
MAIN_WINDOW_TOOLTIPS = {
    "analysis_tab": "Realizar análisis balístico individual de muestras",
    "comparison_tab": "Comparar múltiples muestras y buscar coincidencias",
    "database_tab": "Gestionar casos, evidencias y metadatos NIST",
    "reports_tab": "Generar reportes profesionales de análisis",
    "alignment_tab": "Alinear manualmente imágenes usando puntos de correspondencia",
    "statistical_visualizations_tab": "Visualizar estadísticas y métricas de análisis"
}

ANALYSIS_TAB_TOOLTIPS = {
    "load_image_btn": "Cargar imagen de muestra balística para análisis",
    "preprocess_btn": "Aplicar filtros y mejoras a la imagen cargada",
    "extract_features_btn": "Extraer características distintivas de la muestra",
    "analyze_btn": "Ejecutar análisis completo de la muestra",
    "save_results_btn": "Guardar resultados del análisis actual",
    "clear_btn": "Limpiar todos los datos y resultados actuales",
    "zoom_slider": "Ajustar nivel de zoom de la imagen",
    "contrast_slider": "Modificar contraste de la imagen",
    "brightness_slider": "Ajustar brillo de la imagen",
    "roi_selector": "Seleccionar región de interés para análisis detallado"
}

COMPARISON_TAB_TOOLTIPS = {
    "load_evidence_btn": "Cargar imagen de evidencia balística",
    "load_reference_btn": "Cargar imagen de referencia para comparación",
    "align_images_btn": "Alinear automáticamente las imágenes cargadas",
    "compare_btn": "Ejecutar comparación entre las muestras",
    "sync_zoom_cb": "Sincronizar zoom entre las imágenes",
    "overlay_mode_cb": "Activar modo de superposición de imágenes",
    "similarity_threshold_slider": "Ajustar umbral de similitud para coincidencias",
    "correlation_method_combo": "Seleccionar método de correlación a utilizar",
    "show_matches_cb": "Mostrar puntos de coincidencia detectados",
    "export_comparison_btn": "Exportar resultados de comparación"
}

DATABASE_TAB_TOOLTIPS = {
    "new_case_btn": "Crear nuevo caso balístico",
    "edit_case_btn": "Editar caso seleccionado",
    "delete_case_btn": "Eliminar caso seleccionado (irreversible)",
    "add_evidence_btn": "Añadir nueva evidencia al caso actual",
    "search_btn": "Buscar casos o evidencias en la base de datos",
    "clear_filters_btn": "Limpiar todos los filtros de búsqueda",
    "export_data_btn": "Exportar datos en formato NIST estándar",
    "import_data_btn": "Importar datos desde archivo NIST",
    "case_id_edit": "Identificador único del caso (formato: YYYY-NNNN)",
    "investigator_edit": "Nombre del investigador responsable",
    "evidence_type_combo": "Tipo de evidencia balística (vaina, proyectil, etc.)",
    "date_filter": "Filtrar casos por rango de fechas",
    "status_filter": "Filtrar casos por estado (activo, cerrado, etc.)"
}

REPORTS_TAB_TOOLTIPS = {
    "generate_report_btn": "Generar reporte con configuración actual",
    "preview_report_btn": "Vista previa del reporte antes de generar",
    "save_template_btn": "Guardar configuración actual como plantilla",
    "load_template_btn": "Cargar plantilla de reporte guardada",
    "export_pdf_btn": "Exportar reporte en formato PDF",
    "export_html_btn": "Exportar reporte en formato HTML",
    "include_images_cb": "Incluir imágenes en el reporte",
    "include_stats_cb": "Incluir estadísticas detalladas",
    "include_metadata_cb": "Incluir metadatos NIST completos",
    "report_title_edit": "Título del reporte a generar",
    "author_edit": "Autor del reporte",
    "institution_edit": "Institución que emite el reporte",
    "template_combo": "Seleccionar plantilla de reporte predefinida"
}

FLOATING_STATS_TOOLTIPS = {
    "correlation_value": "Valor de correlación cruzada entre muestras",
    "similarity_score": "Puntuación de similitud calculada",
    "match_confidence": "Nivel de confianza en la coincidencia",
    "feature_count": "Número de características detectadas",
    "alignment_error": "Error de alineación entre imágenes",
    "processing_time": "Tiempo de procesamiento del análisis",
    "image_quality": "Métrica de calidad de la imagen",
    "roi_coverage": "Porcentaje de cobertura de la región de interés",
    "noise_level": "Nivel de ruido detectado en la imagen",
    "contrast_ratio": "Relación de contraste de la imagen"
}

INTERACTIVE_WIDGETS_TOOLTIPS = {
    "zoom_in_btn": "Aumentar zoom (Ctrl + rueda del mouse)",
    "zoom_out_btn": "Disminuir zoom (Ctrl + rueda del mouse)",
    "pan_tool": "Herramienta de desplazamiento (mantener clic y arrastrar)",
    "measure_tool": "Herramienta de medición de distancias",
    "annotation_tool": "Añadir anotaciones a la imagen",
    "reset_view_btn": "Restablecer vista original de la imagen",
    "fullscreen_btn": "Ver imagen en pantalla completa (F11)",
    "save_view_btn": "Guardar vista actual como imagen",
    "grid_overlay_cb": "Mostrar cuadrícula de referencia",
    "ruler_overlay_cb": "Mostrar regla de medición"
}

NIST_METADATA_TOOLTIPS = {
    "case_number": "Número de caso según estándar NIST (obligatorio)",
    "evidence_id": "Identificador único de la evidencia",
    "acquisition_date": "Fecha de adquisición de la muestra",
    "examiner_name": "Nombre del examinador certificado",
    "laboratory": "Laboratorio que realiza el análisis",
    "firearm_type": "Tipo de arma de fuego utilizada",
    "ammunition_type": "Tipo de munición empleada",
    "barrel_length": "Longitud del cañón en milímetros",
    "caliber": "Calibre del arma de fuego",
    "manufacturer": "Fabricante del arma de fuego",
    "serial_number": "Número de serie del arma (si disponible)",
    "chain_of_custody": "Cadena de custodia de la evidencia"
}

# Registrar todos los tooltips en el gestor
tooltip_manager.register_widget_tooltips("MainWindow", MAIN_WINDOW_TOOLTIPS)
tooltip_manager.register_widget_tooltips("AnalysisTab", ANALYSIS_TAB_TOOLTIPS)
tooltip_manager.register_widget_tooltips("ComparisonTab", COMPARISON_TAB_TOOLTIPS)
tooltip_manager.register_widget_tooltips("DatabaseTab", DATABASE_TAB_TOOLTIPS)
tooltip_manager.register_widget_tooltips("ReportsTab", REPORTS_TAB_TOOLTIPS)
tooltip_manager.register_widget_tooltips("FloatingStatsPanel", FLOATING_STATS_TOOLTIPS)
tooltip_manager.register_widget_tooltips("InteractiveWidgets", INTERACTIVE_WIDGETS_TOOLTIPS)
tooltip_manager.register_widget_tooltips("NISTMetadata", NIST_METADATA_TOOLTIPS)

def apply_tooltips_to_reports_tab(reports_tab):
    """
    Aplica tooltips específicos a la pestaña de reportes
    
    Args:
        reports_tab: Instancia de ReportsTab
    """
    try:
        # Tooltips para controles de configuración de reportes
        if hasattr(reports_tab, 'step_indicator'):
            reports_tab.step_indicator.setToolTip(
                "Indicador de progreso del proceso de generación de reportes. "
                "Sigue estos pasos para crear un reporte completo y profesional."
            )
        
        # Tooltips para campos de información básica
        if hasattr(reports_tab, 'case_name_edit'):
            reports_tab.case_name_edit.setToolTip(
                "Nombre del caso balístico. Este nombre aparecerá en el encabezado del reporte."
            )
        
        if hasattr(reports_tab, 'investigator_edit'):
            reports_tab.investigator_edit.setToolTip(
                "Nombre del investigador o perito responsable del análisis balístico."
            )
        
        if hasattr(reports_tab, 'report_type_combo'):
            reports_tab.report_type_combo.setToolTip(
                "Tipo de reporte a generar:\n"
                "• Técnico: Incluye detalles metodológicos completos\n"
                "• Ejecutivo: Resumen para audiencia no técnica\n"
                "• Forense: Formato estándar para procedimientos judiciales"
            )
        
        # Tooltips para selección de datos
        if hasattr(reports_tab, 'analyses_tree'):
            reports_tab.analyses_tree.setToolTip(
                "Selecciona los análisis que deseas incluir en el reporte. "
                "Puedes incluir análisis individuales, comparaciones y búsquedas en base de datos."
            )
        
        # Tooltips para botones de acción
        if hasattr(reports_tab, 'select_all_btn'):
            reports_tab.select_all_btn.setToolTip(
                "Selecciona todos los análisis disponibles para incluir en el reporte."
            )
        
        if hasattr(reports_tab, 'select_none_btn'):
            reports_tab.select_none_btn.setToolTip(
                "Deselecciona todos los análisis. Útil para empezar una nueva selección."
            )
        
        # Tooltips para configuración avanzada
        if hasattr(reports_tab, 'include_images_check'):
            reports_tab.include_images_check.setToolTip(
                "Incluir imágenes de evidencias y resultados en el reporte. "
                "Recomendado para reportes técnicos y forenses."
            )
        
        if hasattr(reports_tab, 'include_statistics_check'):
            reports_tab.include_statistics_check.setToolTip(
                "Incluir gráficos estadísticos y métricas de calidad en el reporte."
            )
        
        if hasattr(reports_tab, 'include_metadata_check'):
            reports_tab.include_metadata_check.setToolTip(
                "Incluir metadatos NIST y información técnica detallada de las evidencias."
            )
        
        # Tooltips para generación y exportación
        if hasattr(reports_tab, 'output_format_combo'):
            reports_tab.output_format_combo.setToolTip(
                "Formato de salida del reporte:\n"
                "• PDF: Formato estándar para distribución\n"
                "• HTML: Formato interactivo con navegación\n"
                "• DOCX: Formato editable para Microsoft Word"
            )
        
        if hasattr(reports_tab, 'output_path_edit'):
            reports_tab.output_path_edit.setToolTip(
                "Ruta donde se guardará el reporte generado. "
                "Usa el botón 'Examinar' para seleccionar la ubicación."
            )
        
        if hasattr(reports_tab, 'browse_output_btn'):
            reports_tab.browse_output_btn.setToolTip(
                "Examinar y seleccionar la carpeta donde guardar el reporte."
            )
        
        if hasattr(reports_tab, 'generate_btn'):
            reports_tab.generate_btn.setToolTip(
                "Generar el reporte con la configuración actual. "
                "El proceso puede tomar varios minutos dependiendo de la cantidad de datos."
            )
        
        # Tooltips para vista previa
        if hasattr(reports_tab, 'preview_web_view'):
            reports_tab.preview_web_view.setToolTip(
                "Vista previa del reporte. Aquí puedes revisar el contenido antes de la generación final."
            )
        
        # Tooltips para botones de template y herramientas avanzadas
        if hasattr(reports_tab, 'template_editor_btn'):
            reports_tab.template_editor_btn.setToolTip(
                "Abrir el editor de plantillas para personalizar el formato del reporte."
            )
        
        if hasattr(reports_tab, 'history_import_btn'):
            reports_tab.history_import_btn.setToolTip(
                "Importar datos históricos de casos anteriores para incluir en el reporte."
            )
        
        # Tooltips para controles de progreso
        if hasattr(reports_tab, 'progress_bar'):
            reports_tab.progress_bar.setToolTip(
                "Progreso de la generación del reporte. El proceso incluye recopilación de datos, "
                "análisis y formateo del documento final."
            )
        
        if hasattr(reports_tab, 'progress_label'):
            reports_tab.progress_label.setToolTip(
                "Estado actual del proceso de generación del reporte."
            )
        
        logger.info("Tooltips aplicados exitosamente a ReportsTab")
        
    except Exception as e:
        logger.error(f"Error aplicando tooltips a ReportsTab: {e}")


def apply_tooltips_to_main_window(main_window):
    """Aplica tooltips a la ventana principal y sus pestañas"""
    tooltip_manager.apply_tooltips_to_widget(main_window, "MainWindow")
    
    # Aplicar tooltips específicos a cada pestaña
    if hasattr(main_window, 'analysis_tab'):
        tooltip_manager.apply_tooltips_to_widget(main_window.analysis_tab, "AnalysisTab")
    
    if hasattr(main_window, 'comparison_tab'):
        tooltip_manager.apply_tooltips_to_widget(main_window.comparison_tab, "ComparisonTab")
    
    if hasattr(main_window, 'database_tab'):
        tooltip_manager.apply_tooltips_to_widget(main_window.database_tab, "DatabaseTab")
    
    if hasattr(main_window, 'reports_tab'):
        tooltip_manager.apply_tooltips_to_widget(main_window.reports_tab, "ReportsTab")

def setup_contextual_tooltips(widget: QWidget, context: str):
    """Configura tooltips contextuales basados en el contexto de uso"""
    contextual_tips = {
        "analysis_mode": {
            "tip": "Modo de análisis individual activado. Use las herramientas de la izquierda para procesar la muestra.",
            "duration": 3000
        },
        "comparison_mode": {
            "tip": "Modo de comparación activado. Cargue dos muestras para realizar la comparación.",
            "duration": 3000
        },
        "database_mode": {
            "tip": "Modo de base de datos activado. Gestione casos y evidencias con cumplimiento NIST.",
            "duration": 3000
        }
    }
    
    if context in contextual_tips:
        tip_info = contextual_tips[context]
        QTimer.singleShot(500, lambda: QToolTip.showText(
            widget.mapToGlobal(QPoint(10, 10)),
            tip_info["tip"],
            widget,
            widget.rect(),
            tip_info["duration"]
        ))

def show_feature_tooltip(widget: QWidget, feature_name: str, description: str):
    """Muestra tooltip para una característica específica"""
    tooltip_text = f"<b>{feature_name}</b><br>{description}"
    QToolTip.showText(
        widget.mapToGlobal(QPoint(0, 0)),
        tooltip_text,
        widget
    )

def create_rich_tooltip(title: str, description: str, shortcut: str = None) -> str:
    """Crea un tooltip enriquecido con formato HTML"""
    tooltip = f"<b>{title}</b><br>{description}"
    if shortcut:
        tooltip += f"<br><i>Atajo: {shortcut}</i>"
    return tooltip

# Tooltips enriquecidos para funciones avanzadas
ADVANCED_TOOLTIPS = {
    "guided_analysis": create_rich_tooltip(
        "Análisis Guiado",
        "Proceso paso a paso que guía a través del análisis completo de muestras balísticas con validación automática en cada etapa.",
        "Ctrl+G"
    ),
    "interactive_comparison": create_rich_tooltip(
        "Comparación Interactiva",
        "Herramientas avanzadas de comparación visual con sincronización de zoom, superposición ajustable y métricas en tiempo real.",
        "Ctrl+I"
    ),
    "nist_compliance": create_rich_tooltip(
        "Cumplimiento NIST",
        "Formularios inteligentes que garantizan el cumplimiento total con el estándar NIST 1001 para bases de datos balísticas.",
        "Ctrl+N"
    ),
    "real_time_stats": create_rich_tooltip(
        "Estadísticas en Tiempo Real",
        "Panel flotante que muestra métricas de calidad, correlación y confianza actualizadas instantáneamente durante el análisis.",
        "F9"
    ),
    "unified_reports": create_rich_tooltip(
        "Reportes Unificados",
        "Sistema de generación automática de reportes profesionales con integración completa de datos, análisis y visualizaciones.",
        "Ctrl+R"
    )
}

def apply_advanced_tooltips(widget: QWidget):
    """Aplica tooltips avanzados a widgets específicos"""
    for object_name, tooltip_text in ADVANCED_TOOLTIPS.items():
        child_widget = widget.findChild(QWidget, object_name)
        if child_widget:
            child_widget.setToolTip(tooltip_text)