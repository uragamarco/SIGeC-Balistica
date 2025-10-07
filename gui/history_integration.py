#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integración con Historial de Análisis - SIGeC-Balística
======================================================

Módulo para integrar el historial de análisis con el generador de reportes,
permitiendo importar resultados pasados directamente a los reportes.

Funcionalidades:
- Importación de análisis desde history_dialog.py
- Conversión de datos históricos a formato de reporte
- Filtrado y selección de análisis relevantes
- Integración fluida con el generador de reportes

Autor: SIGeC-BalisticaTeam
Fecha: Octubre 2025
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView, QCheckBox,
    QGroupBox, QDateEdit, QComboBox, QLineEdit, QTextEdit,
    QSplitter, QFrame, QMessageBox, QProgressBar, QTabWidget,
    QScrollArea, QGridLayout, QWidget
)
from PyQt5.QtCore import Qt, pyqtSignal, QDate, QThread, pyqtSlot
from PyQt5.QtGui import QFont, QIcon, QPixmap

from .interactive_report_generator import InteractiveReportData

logger = logging.getLogger(__name__)

class AnalysisHistoryItem:
    """Elemento del historial de análisis"""
    
    def __init__(self, data: Dict):
        self.id = data.get('id', '')
        self.timestamp = data.get('timestamp', '')
        self.analysis_type = data.get('type', '')
        self.file_path = data.get('file', '')
        self.status = data.get('status', '')
        self.duration = data.get('duration', 0)
        self.results = data.get('results', {})
        self.metadata = data.get('metadata', {})
        self.images = data.get('images', [])
        self.quality_metrics = data.get('quality_metrics', {})
        self.parameters = data.get('parameters', {})
        
    def to_dict(self) -> Dict:
        """Convierte el elemento a diccionario"""
        return {
            'id': self.id,
            'timestamp': self.timestamp,
            'type': self.analysis_type,
            'file': self.file_path,
            'status': self.status,
            'duration': self.duration,
            'results': self.results,
            'metadata': self.metadata,
            'images': self.images,
            'quality_metrics': self.quality_metrics,
            'parameters': self.parameters
        }
    
    def get_display_name(self) -> str:
        """Obtiene el nombre para mostrar"""
        file_name = Path(self.file_path).name if self.file_path else "Sin archivo"
        return f"{self.analysis_type} - {file_name} ({self.timestamp})"
    
    def is_successful(self) -> bool:
        """Verifica si el análisis fue exitoso"""
        return self.status.lower() in ['completed', 'success', 'exitoso', 'completado']
    
    def has_results(self) -> bool:
        """Verifica si tiene resultados válidos"""
        return bool(self.results) and self.is_successful()
    
    def get_similarity_score(self) -> float:
        """Obtiene el puntaje de similitud si está disponible"""
        if 'similarity' in self.results:
            return float(self.results['similarity'])
        elif 'score' in self.results:
            return float(self.results['score'])
        elif 'confidence' in self.results:
            return float(self.results['confidence'])
        return 0.0

class HistoryDataConverter:
    """Conversor de datos históricos a formato de reporte"""
    
    @staticmethod
    def convert_to_report_data(history_items: List[AnalysisHistoryItem]) -> InteractiveReportData:
        """Convierte elementos del historial a datos de reporte"""
        report_data = InteractiveReportData()
        
        # Metadatos generales
        report_data.metadata = {
            'analysis_date': datetime.now().strftime('%d/%m/%Y'),
            'analysis_type': 'Reporte Consolidado',
            'total_analyses': len(history_items),
            'successful_analyses': len([item for item in history_items if item.is_successful()]),
            'date_range': HistoryDataConverter._get_date_range(history_items)
        }
        
        # Procesar cada elemento del historial
        for item in history_items:
            if not item.has_results():
                continue
                
            # Agregar imágenes
            HistoryDataConverter._add_images_from_item(report_data, item)
            
            # Agregar métricas de calidad
            HistoryDataConverter._add_quality_metrics_from_item(report_data, item)
            
            # Agregar resultados a tablas
            HistoryDataConverter._add_results_to_tables(report_data, item)
        
        # Generar gráficos consolidados
        HistoryDataConverter._generate_consolidated_charts(report_data, history_items)
        
        return report_data
    
    @staticmethod
    def _get_date_range(history_items: List[AnalysisHistoryItem]) -> str:
        """Obtiene el rango de fechas de los análisis"""
        if not history_items:
            return "Sin datos"
        
        try:
            dates = []
            for item in history_items:
                if item.timestamp:
                    # Intentar parsear diferentes formatos de fecha
                    for fmt in ['%Y-%m-%d %H:%M:%S', '%d/%m/%Y %H:%M', '%Y-%m-%d']:
                        try:
                            date = datetime.strptime(item.timestamp, fmt)
                            dates.append(date)
                            break
                        except ValueError:
                            continue
            
            if dates:
                min_date = min(dates).strftime('%d/%m/%Y')
                max_date = max(dates).strftime('%d/%m/%Y')
                return f"{min_date} - {max_date}" if min_date != max_date else min_date
            
        except Exception as e:
            logger.warning(f"Error procesando rango de fechas: {e}")
        
        return "Rango no disponible"
    
    @staticmethod
    def _add_images_from_item(report_data: InteractiveReportData, item: AnalysisHistoryItem):
        """Agrega imágenes del elemento histórico al reporte"""
        if not item.images:
            return
        
        for i, image_path in enumerate(item.images):
            if os.path.exists(image_path):
                title = f"{item.analysis_type} - Imagen {i+1}"
                description = f"Análisis realizado el {item.timestamp}"
                metadata = {
                    'analysis_id': item.id,
                    'analysis_type': item.analysis_type,
                    'timestamp': item.timestamp,
                    'similarity_score': item.get_similarity_score()
                }
                
                # Agrupar por tipo de análisis para comparación
                comparison_group = item.analysis_type if len(item.images) > 1 else None
                
                report_data.add_image(
                    image_path, title, description, metadata, comparison_group
                )
    
    @staticmethod
    def _add_quality_metrics_from_item(report_data: InteractiveReportData, item: AnalysisHistoryItem):
        """Agrega métricas de calidad del elemento al reporte"""
        if not item.quality_metrics:
            return
        
        # Consolidar métricas de calidad
        for metric_name, metric_value in item.quality_metrics.items():
            if metric_name not in report_data.quality_metrics:
                report_data.quality_metrics[metric_name] = []
            
            report_data.quality_metrics[metric_name].append({
                'value': metric_value,
                'analysis_id': item.id,
                'timestamp': item.timestamp,
                'analysis_type': item.analysis_type
            })
    
    @staticmethod
    def _add_results_to_tables(report_data: InteractiveReportData, item: AnalysisHistoryItem):
        """Agrega resultados del elemento a las tablas del reporte"""
        # Tabla de resultados principales
        result_row = {
            'ID': item.id,
            'Fecha': item.timestamp,
            'Tipo': item.analysis_type,
            'Archivo': Path(item.file_path).name if item.file_path else 'N/A',
            'Estado': item.status,
            'Duración': f"{item.duration:.2f}s" if item.duration else 'N/A',
            'Similitud': f"{item.get_similarity_score():.2f}%" if item.get_similarity_score() > 0 else 'N/A'
        }
        
        # Buscar tabla existente o crear nueva
        results_table = None
        for table in report_data.tables:
            if table['title'] == 'Historial de Análisis':
                results_table = table
                break
        
        if not results_table:
            report_data.add_table(
                [result_row],
                'Historial de Análisis',
                'Resumen de todos los análisis realizados'
            )
        else:
            results_table['data'].append(result_row)
        
        # Tabla de parámetros si están disponibles
        if item.parameters:
            param_rows = []
            for param_name, param_value in item.parameters.items():
                param_rows.append({
                    'Análisis': item.id,
                    'Parámetro': param_name,
                    'Valor': str(param_value),
                    'Tipo': item.analysis_type
                })
            
            # Buscar tabla de parámetros o crear nueva
            params_table = None
            for table in report_data.tables:
                if table['title'] == 'Parámetros de Análisis':
                    params_table = table
                    break
            
            if not params_table:
                report_data.add_table(
                    param_rows,
                    'Parámetros de Análisis',
                    'Configuración utilizada en cada análisis'
                )
            else:
                params_table['data'].extend(param_rows)
    
    @staticmethod
    def _generate_consolidated_charts(report_data: InteractiveReportData, history_items: List[AnalysisHistoryItem]):
        """Genera gráficos consolidados de los datos históricos"""
        successful_items = [item for item in history_items if item.is_successful()]
        
        if not successful_items:
            return
        
        # Gráfico de distribución por tipo de análisis
        analysis_types = {}
        for item in successful_items:
            analysis_types[item.analysis_type] = analysis_types.get(item.analysis_type, 0) + 1
        
        if analysis_types:
            chart_data = {
                'labels': list(analysis_types.keys()),
                'values': list(analysis_types.values())
            }
            
            report_data.add_chart(
                'pie',
                chart_data,
                'Distribución por Tipo de Análisis',
                'Cantidad de análisis realizados por tipo'
            )
        
        # Gráfico de evolución temporal de similitud
        similarity_over_time = []
        dates = []
        
        for item in successful_items:
            similarity = item.get_similarity_score()
            if similarity > 0:
                try:
                    # Intentar parsear fecha
                    for fmt in ['%Y-%m-%d %H:%M:%S', '%d/%m/%Y %H:%M', '%Y-%m-%d']:
                        try:
                            date = datetime.strptime(item.timestamp, fmt)
                            dates.append(date.strftime('%d/%m'))
                            similarity_over_time.append(similarity)
                            break
                        except ValueError:
                            continue
                except:
                    continue
        
        if similarity_over_time:
            chart_data = {
                'labels': dates,
                'values': similarity_over_time
            }
            
            report_data.add_chart(
                'line',
                chart_data,
                'Evolución de Similitud',
                'Puntajes de similitud a lo largo del tiempo'
            )
        
        # Gráfico de métricas de calidad promedio
        if report_data.quality_metrics:
            avg_metrics = {}
            for metric_name, metric_values in report_data.quality_metrics.items():
                if metric_values:
                    avg_value = sum(m['value'] for m in metric_values if isinstance(m['value'], (int, float))) / len(metric_values)
                    avg_metrics[metric_name] = avg_value
            
            if avg_metrics:
                chart_data = {
                    'labels': list(avg_metrics.keys()),
                    'values': list(avg_metrics.values())
                }
                
                report_data.add_chart(
                    'bar',
                    chart_data,
                    'Métricas de Calidad Promedio',
                    'Valores promedio de las métricas de calidad'
                )

class HistoryImportDialog(QDialog):
    """Diálogo para importar análisis desde el historial"""
    
    analysis_selected = pyqtSignal(list)  # Lista de AnalysisHistoryItem
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.history_items = []
        self.selected_items = []
        
        self.setWindowTitle("Importar desde Historial de Análisis")
        self.setModal(True)
        self.resize(1000, 700)
        
        self._setup_ui()
        self._connect_signals()
        self._load_history_data()
        
        logger.info("Diálogo de importación de historial inicializado")
    
    def _setup_ui(self):
        """Configura la interfaz de usuario"""
        layout = QVBoxLayout(self)
        
        # Título
        title = QLabel("Seleccionar Análisis para Importar")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        layout.addWidget(title)
        
        # Pestañas
        tab_widget = QTabWidget()
        layout.addWidget(tab_widget)
        
        # Pestaña de filtros
        filters_tab = self._create_filters_tab()
        tab_widget.addTab(filters_tab, "Filtros")
        
        # Pestaña de selección
        selection_tab = self._create_selection_tab()
        tab_widget.addTab(selection_tab, "Selección")
        
        # Pestaña de vista previa
        preview_tab = self._create_preview_tab()
        tab_widget.addTab(preview_tab, "Vista Previa")
        
        # Botones
        buttons_layout = QHBoxLayout()
        
        self.select_all_btn = QPushButton("Seleccionar Todo")
        self.clear_selection_btn = QPushButton("Limpiar Selección")
        self.import_btn = QPushButton("Importar Seleccionados")
        self.cancel_btn = QPushButton("Cancelar")
        
        buttons_layout.addWidget(self.select_all_btn)
        buttons_layout.addWidget(self.clear_selection_btn)
        buttons_layout.addStretch()
        buttons_layout.addWidget(self.import_btn)
        buttons_layout.addWidget(self.cancel_btn)
        
        layout.addLayout(buttons_layout)
        
        # Barra de progreso
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
    
    def _create_filters_tab(self) -> QWidget:
        """Crea la pestaña de filtros"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Filtros de fecha
        date_group = QGroupBox("Filtros de Fecha")
        date_layout = QGridLayout(date_group)
        
        date_layout.addWidget(QLabel("Desde:"), 0, 0)
        self.date_from = QDateEdit()
        self.date_from.setDate(QDate.currentDate().addDays(-30))
        self.date_from.setCalendarPopup(True)
        date_layout.addWidget(self.date_from, 0, 1)
        
        date_layout.addWidget(QLabel("Hasta:"), 0, 2)
        self.date_to = QDateEdit()
        self.date_to.setDate(QDate.currentDate())
        self.date_to.setCalendarPopup(True)
        date_layout.addWidget(self.date_to, 0, 3)
        
        layout.addWidget(date_group)
        
        # Filtros de tipo
        type_group = QGroupBox("Filtros de Tipo")
        type_layout = QGridLayout(type_group)
        
        type_layout.addWidget(QLabel("Tipo de Análisis:"), 0, 0)
        self.analysis_type_combo = QComboBox()
        self.analysis_type_combo.addItems([
            "Todos", "Comparación Individual", "Comparación Múltiple",
            "Búsqueda en Base de Datos", "Análisis de Calidad"
        ])
        type_layout.addWidget(self.analysis_type_combo, 0, 1)
        
        type_layout.addWidget(QLabel("Estado:"), 0, 2)
        self.status_combo = QComboBox()
        self.status_combo.addItems([
            "Todos", "Completado", "Error", "Cancelado"
        ])
        type_layout.addWidget(self.status_combo, 0, 3)
        
        layout.addWidget(type_group)
        
        # Filtros de contenido
        content_group = QGroupBox("Filtros de Contenido")
        content_layout = QGridLayout(content_group)
        
        content_layout.addWidget(QLabel("Archivo:"), 0, 0)
        self.file_filter = QLineEdit()
        self.file_filter.setPlaceholderText("Filtrar por nombre de archivo...")
        content_layout.addWidget(self.file_filter, 0, 1, 1, 2)
        
        self.only_successful_cb = QCheckBox("Solo análisis exitosos")
        self.only_successful_cb.setChecked(True)
        content_layout.addWidget(self.only_successful_cb, 1, 0)
        
        self.only_with_images_cb = QCheckBox("Solo con imágenes")
        content_layout.addWidget(self.only_with_images_cb, 1, 1)
        
        layout.addWidget(content_group)
        
        # Botón aplicar filtros
        self.apply_filters_btn = QPushButton("Aplicar Filtros")
        layout.addWidget(self.apply_filters_btn)
        
        layout.addStretch()
        
        return widget
    
    def _create_selection_tab(self) -> QWidget:
        """Crea la pestaña de selección"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Información
        info_label = QLabel("Seleccione los análisis que desea incluir en el reporte:")
        layout.addWidget(info_label)
        
        # Tabla de análisis
        self.analysis_table = QTableWidget()
        self.analysis_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.analysis_table.setAlternatingRowColors(True)
        
        # Configurar columnas
        columns = ["Seleccionar", "Fecha", "Tipo", "Archivo", "Estado", "Duración", "Similitud"]
        self.analysis_table.setColumnCount(len(columns))
        self.analysis_table.setHorizontalHeaderLabels(columns)
        
        # Ajustar tamaños de columna
        header = self.analysis_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Fixed)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.Stretch)
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(6, QHeaderView.ResizeToContents)
        
        self.analysis_table.setColumnWidth(0, 80)
        
        layout.addWidget(self.analysis_table)
        
        # Estadísticas de selección
        self.selection_stats = QLabel("0 análisis seleccionados")
        layout.addWidget(self.selection_stats)
        
        return widget
    
    def _create_preview_tab(self) -> QWidget:
        """Crea la pestaña de vista previa"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Información
        info_label = QLabel("Vista previa de los datos que se importarán:")
        layout.addWidget(info_label)
        
        # Área de vista previa
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        
        self.preview_content = QWidget()
        self.preview_layout = QVBoxLayout(self.preview_content)
        
        scroll_area.setWidget(self.preview_content)
        layout.addWidget(scroll_area)
        
        # Botón actualizar vista previa
        self.update_preview_btn = QPushButton("Actualizar Vista Previa")
        layout.addWidget(self.update_preview_btn)
        
        return widget
    
    def _connect_signals(self):
        """Conecta las señales"""
        self.apply_filters_btn.clicked.connect(self._apply_filters)
        self.select_all_btn.clicked.connect(self._select_all)
        self.clear_selection_btn.clicked.connect(self._clear_selection)
        self.import_btn.clicked.connect(self._import_selected)
        self.cancel_btn.clicked.connect(self.reject)
        self.update_preview_btn.clicked.connect(self._update_preview)
        
        # Conectar cambios en filtros
        self.date_from.dateChanged.connect(self._on_filter_changed)
        self.date_to.dateChanged.connect(self._on_filter_changed)
        self.analysis_type_combo.currentTextChanged.connect(self._on_filter_changed)
        self.status_combo.currentTextChanged.connect(self._on_filter_changed)
        self.file_filter.textChanged.connect(self._on_filter_changed)
        self.only_successful_cb.toggled.connect(self._on_filter_changed)
        self.only_with_images_cb.toggled.connect(self._on_filter_changed)
    
    def _load_history_data(self):
        """Carga los datos del historial"""
        try:
            # Intentar cargar desde history_dialog.py
            self._load_from_history_dialog()
            
            # Si no hay datos, cargar datos de ejemplo
            if not self.history_items:
                self._load_sample_data()
            
            self._populate_table()
            
        except Exception as e:
            logger.error(f"Error cargando datos del historial: {e}")
            QMessageBox.warning(self, "Error", f"Error cargando historial: {e}")
    
    def _load_from_history_dialog(self):
        """Carga datos desde history_dialog.py"""
        try:
            # Intentar importar y usar history_dialog
            from .history_dialog import HistoryTableModel
            
            # Crear instancia del modelo
            model = HistoryTableModel()
            
            # Obtener datos del modelo
            for row in range(model.rowCount()):
                item_data = {}
                for col in range(model.columnCount()):
                    header = model.headerData(col, Qt.Horizontal, Qt.DisplayRole)
                    value = model.data(model.index(row, col), Qt.DisplayRole)
                    item_data[header.lower().replace('/', '_')] = value
                
                # Crear elemento del historial
                history_item = AnalysisHistoryItem(item_data)
                self.history_items.append(history_item)
            
            logger.info(f"Cargados {len(self.history_items)} elementos del historial")
            
        except ImportError:
            logger.warning("No se pudo importar history_dialog, usando datos de ejemplo")
        except Exception as e:
            logger.error(f"Error accediendo a history_dialog: {e}")
    
    def _load_sample_data(self):
        """Carga datos de ejemplo"""
        sample_data = [
            {
                'id': 'ANAL-001',
                'timestamp': '2025-01-15 10:30:00',
                'type': 'Comparación Individual',
                'file': '/path/to/evidence1.jpg',
                'status': 'Completado',
                'duration': 45.2,
                'results': {'similarity': 85.5, 'matches': 156},
                'images': ['/path/to/evidence1.jpg', '/path/to/reference1.jpg'],
                'quality_metrics': {'image_quality': 92, 'feature_count': 3247}
            },
            {
                'id': 'ANAL-002',
                'timestamp': '2025-01-14 15:45:00',
                'type': 'Búsqueda en Base de Datos',
                'file': '/path/to/evidence2.jpg',
                'status': 'Completado',
                'duration': 120.8,
                'results': {'matches_found': 3, 'best_similarity': 78.2},
                'images': ['/path/to/evidence2.jpg'],
                'quality_metrics': {'image_quality': 88, 'feature_count': 2891}
            },
            {
                'id': 'ANAL-003',
                'timestamp': '2025-01-13 09:15:00',
                'type': 'Comparación Múltiple',
                'file': '/path/to/evidence3.jpg',
                'status': 'Error',
                'duration': 12.3,
                'results': {},
                'images': [],
                'quality_metrics': {}
            }
        ]
        
        for data in sample_data:
            self.history_items.append(AnalysisHistoryItem(data))
        
        logger.info(f"Cargados {len(self.history_items)} elementos de ejemplo")
    
    def _populate_table(self):
        """Puebla la tabla con los datos del historial"""
        filtered_items = self._get_filtered_items()
        
        self.analysis_table.setRowCount(len(filtered_items))
        
        for row, item in enumerate(filtered_items):
            # Checkbox de selección
            checkbox = QCheckBox()
            checkbox.setChecked(item.is_successful())  # Seleccionar exitosos por defecto
            checkbox.stateChanged.connect(self._on_selection_changed)
            self.analysis_table.setCellWidget(row, 0, checkbox)
            
            # Datos del análisis
            self.analysis_table.setItem(row, 1, QTableWidgetItem(item.timestamp))
            self.analysis_table.setItem(row, 2, QTableWidgetItem(item.analysis_type))
            self.analysis_table.setItem(row, 3, QTableWidgetItem(
                Path(item.file_path).name if item.file_path else 'N/A'
            ))
            
            # Estado con color
            status_item = QTableWidgetItem(item.status)
            if item.is_successful():
                status_item.setBackground(Qt.green)
            elif item.status.lower() in ['error', 'failed']:
                status_item.setBackground(Qt.red)
            else:
                status_item.setBackground(Qt.yellow)
            
            self.analysis_table.setItem(row, 4, status_item)
            
            # Duración
            duration_text = f"{item.duration:.1f}s" if item.duration else 'N/A'
            self.analysis_table.setItem(row, 5, QTableWidgetItem(duration_text))
            
            # Similitud
            similarity = item.get_similarity_score()
            similarity_text = f"{similarity:.1f}%" if similarity > 0 else 'N/A'
            self.analysis_table.setItem(row, 6, QTableWidgetItem(similarity_text))
            
            # Guardar referencia al item
            self.analysis_table.item(row, 1).setData(Qt.UserRole, item)
        
        self._update_selection_stats()
    
    def _get_filtered_items(self) -> List[AnalysisHistoryItem]:
        """Obtiene los elementos filtrados"""
        filtered = self.history_items.copy()
        
        # Filtro de fecha
        date_from = self.date_from.date().toPyDate()
        date_to = self.date_to.date().toPyDate()
        
        # Filtro de tipo
        analysis_type = self.analysis_type_combo.currentText()
        if analysis_type != "Todos":
            filtered = [item for item in filtered if item.analysis_type == analysis_type]
        
        # Filtro de estado
        status = self.status_combo.currentText()
        if status != "Todos":
            status_map = {
                "Completado": ["completed", "success", "exitoso", "completado"],
                "Error": ["error", "failed", "fallido"],
                "Cancelado": ["cancelled", "canceled", "cancelado"]
            }
            if status in status_map:
                filtered = [item for item in filtered 
                          if item.status.lower() in status_map[status]]
        
        # Filtro de archivo
        file_filter = self.file_filter.text().strip()
        if file_filter:
            filtered = [item for item in filtered 
                       if file_filter.lower() in Path(item.file_path).name.lower()]
        
        # Solo exitosos
        if self.only_successful_cb.isChecked():
            filtered = [item for item in filtered if item.is_successful()]
        
        # Solo con imágenes
        if self.only_with_images_cb.isChecked():
            filtered = [item for item in filtered if item.images]
        
        return filtered
    
    def _apply_filters(self):
        """Aplica los filtros y actualiza la tabla"""
        self._populate_table()
    
    def _on_filter_changed(self):
        """Maneja cambios en los filtros"""
        # Auto-aplicar filtros después de un breve retraso
        if hasattr(self, '_filter_timer'):
            self._filter_timer.stop()
        
        from PyQt5.QtCore import QTimer
        self._filter_timer = QTimer()
        self._filter_timer.setSingleShot(True)
        self._filter_timer.timeout.connect(self._apply_filters)
        self._filter_timer.start(500)  # 500ms de retraso
    
    def _select_all(self):
        """Selecciona todos los elementos"""
        for row in range(self.analysis_table.rowCount()):
            checkbox = self.analysis_table.cellWidget(row, 0)
            if checkbox:
                checkbox.setChecked(True)
    
    def _clear_selection(self):
        """Limpia la selección"""
        for row in range(self.analysis_table.rowCount()):
            checkbox = self.analysis_table.cellWidget(row, 0)
            if checkbox:
                checkbox.setChecked(False)
    
    def _on_selection_changed(self):
        """Maneja cambios en la selección"""
        self._update_selection_stats()
    
    def _update_selection_stats(self):
        """Actualiza las estadísticas de selección"""
        selected_count = 0
        total_count = self.analysis_table.rowCount()
        
        for row in range(total_count):
            checkbox = self.analysis_table.cellWidget(row, 0)
            if checkbox and checkbox.isChecked():
                selected_count += 1
        
        self.selection_stats.setText(f"{selected_count} de {total_count} análisis seleccionados")
        
        # Habilitar/deshabilitar botón de importar
        self.import_btn.setEnabled(selected_count > 0)
    
    def _update_preview(self):
        """Actualiza la vista previa"""
        selected_items = self._get_selected_items()
        
        if not selected_items:
            self._clear_preview()
            return
        
        # Limpiar vista previa anterior
        for i in reversed(range(self.preview_layout.count())):
            self.preview_layout.itemAt(i).widget().setParent(None)
        
        # Generar datos de reporte
        report_data = HistoryDataConverter.convert_to_report_data(selected_items)
        
        # Mostrar resumen
        summary_group = QGroupBox("Resumen de Importación")
        summary_layout = QGridLayout(summary_group)
        
        summary_layout.addWidget(QLabel("Total de análisis:"), 0, 0)
        summary_layout.addWidget(QLabel(str(len(selected_items))), 0, 1)
        
        summary_layout.addWidget(QLabel("Imágenes a importar:"), 1, 0)
        summary_layout.addWidget(QLabel(str(len(report_data.images))), 1, 1)
        
        summary_layout.addWidget(QLabel("Gráficos a generar:"), 2, 0)
        summary_layout.addWidget(QLabel(str(len(report_data.charts))), 2, 1)
        
        summary_layout.addWidget(QLabel("Tablas a crear:"), 3, 0)
        summary_layout.addWidget(QLabel(str(len(report_data.tables))), 3, 1)
        
        self.preview_layout.addWidget(summary_group)
        
        # Mostrar detalles de metadatos
        metadata_group = QGroupBox("Metadatos del Reporte")
        metadata_layout = QGridLayout(metadata_group)
        
        row = 0
        for key, value in report_data.metadata.items():
            metadata_layout.addWidget(QLabel(f"{key}:"), row, 0)
            metadata_layout.addWidget(QLabel(str(value)), row, 1)
            row += 1
        
        self.preview_layout.addWidget(metadata_group)
        
        # Mostrar lista de análisis seleccionados
        analyses_group = QGroupBox("Análisis Seleccionados")
        analyses_layout = QVBoxLayout(analyses_group)
        
        for item in selected_items:
            item_label = QLabel(f"• {item.get_display_name()}")
            analyses_layout.addWidget(item_label)
        
        self.preview_layout.addWidget(analyses_group)
        
        self.preview_layout.addStretch()
    
    def _clear_preview(self):
        """Limpia la vista previa"""
        for i in reversed(range(self.preview_layout.count())):
            widget = self.preview_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)
        
        no_selection_label = QLabel("No hay análisis seleccionados para mostrar vista previa.")
        no_selection_label.setAlignment(Qt.AlignCenter)
        self.preview_layout.addWidget(no_selection_label)
    
    def _get_selected_items(self) -> List[AnalysisHistoryItem]:
        """Obtiene los elementos seleccionados"""
        selected = []
        
        for row in range(self.analysis_table.rowCount()):
            checkbox = self.analysis_table.cellWidget(row, 0)
            if checkbox and checkbox.isChecked():
                item = self.analysis_table.item(row, 1).data(Qt.UserRole)
                if item:
                    selected.append(item)
        
        return selected
    
    def _import_selected(self):
        """Importa los elementos seleccionados"""
        selected_items = self._get_selected_items()
        
        if not selected_items:
            QMessageBox.warning(self, "Advertencia", "No hay análisis seleccionados para importar.")
            return
        
        try:
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 0)  # Indeterminado
            
            # Emitir señal con los elementos seleccionados
            self.analysis_selected.emit(selected_items)
            
            QMessageBox.information(
                self, "Éxito", 
                f"Se han importado {len(selected_items)} análisis correctamente."
            )
            
            self.accept()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error importando análisis: {e}")
        finally:
            self.progress_bar.setVisible(False)
    
    def get_selected_items(self) -> List[AnalysisHistoryItem]:
        """Obtiene los elementos seleccionados (método público)"""
        return self._get_selected_items()

class HistoryIntegrationWidget(QWidget):
    """Widget principal para la integración con el historial"""
    
    report_data_ready = pyqtSignal(object)  # InteractiveReportData
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self._setup_ui()
        self._connect_signals()
        
        logger.info("Widget de integración con historial inicializado")
    
    def _setup_ui(self):
        """Configura la interfaz de usuario"""
        layout = QVBoxLayout(self)
        
        # Título
        title = QLabel("Integración con Historial de Análisis")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        layout.addWidget(title)
        
        # Descripción
        description = QLabel(
            "Importe análisis previos desde el historial para incluirlos en reportes consolidados."
        )
        description.setWordWrap(True)
        layout.addWidget(description)
        
        # Botones
        buttons_layout = QHBoxLayout()
        
        self.import_btn = QPushButton("Importar desde Historial")
        self.import_btn.setIcon(QIcon(":/icons/import.png"))
        
        self.clear_btn = QPushButton("Limpiar Datos")
        self.clear_btn.setIcon(QIcon(":/icons/clear.png"))
        
        buttons_layout.addWidget(self.import_btn)
        buttons_layout.addWidget(self.clear_btn)
        buttons_layout.addStretch()
        
        layout.addLayout(buttons_layout)
        
        # Área de información
        self.info_area = QTextEdit()
        self.info_area.setReadOnly(True)
        self.info_area.setMaximumHeight(200)
        self.info_area.setPlainText("No hay datos importados.")
        
        layout.addWidget(self.info_area)
        
        layout.addStretch()
    
    def _connect_signals(self):
        """Conecta las señales"""
        self.import_btn.clicked.connect(self._open_import_dialog)
        self.clear_btn.clicked.connect(self._clear_data)
    
    def _open_import_dialog(self):
        """Abre el diálogo de importación"""
        dialog = HistoryImportDialog(self)
        dialog.analysis_selected.connect(self._on_analysis_imported)
        dialog.exec_()
    
    def _on_analysis_imported(self, history_items: List[AnalysisHistoryItem]):
        """Maneja la importación de análisis"""
        try:
            # Convertir a datos de reporte
            report_data = HistoryDataConverter.convert_to_report_data(history_items)
            
            # Actualizar información
            info_text = f"""Datos importados correctamente:

• Total de análisis: {len(history_items)}
• Imágenes importadas: {len(report_data.images)}
• Gráficos generados: {len(report_data.charts)}
• Tablas creadas: {len(report_data.tables)}

Rango de fechas: {report_data.metadata.get('date_range', 'N/A')}
Análisis exitosos: {report_data.metadata.get('successful_analyses', 0)}

Los datos están listos para ser incluidos en el reporte."""
            
            self.info_area.setPlainText(info_text)
            
            # Emitir señal con los datos del reporte
            self.report_data_ready.emit(report_data)
            
            logger.info(f"Importados {len(history_items)} análisis del historial")
            
        except Exception as e:
            logger.error(f"Error procesando análisis importados: {e}")
            QMessageBox.critical(self, "Error", f"Error procesando datos: {e}")
    
    def _clear_data(self):
        """Limpia los datos importados"""
        self.info_area.setPlainText("No hay datos importados.")
        
        # Emitir datos vacíos
        empty_data = InteractiveReportData()
        self.report_data_ready.emit(empty_data)
        
        logger.info("Datos de historial limpiados")