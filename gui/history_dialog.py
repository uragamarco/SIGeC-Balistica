#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diálogo de Historial - SIGeC-Balistica
===============================

Diálogo para visualizar y gestionar el historial de análisis y operaciones:
- Historial de análisis individuales
- Historial de comparaciones
- Historial de búsquedas en base de datos
- Historial de reportes generados
- Filtros y búsqueda en historial
- Exportación de historial

Autor: SIGeC-BalisticaTeam
Fecha: Octubre 2025
"""

import os
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union
from pathlib import Path

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTabWidget, QWidget,
    QLabel, QLineEdit, QTableWidget, QTableWidgetItem, QHeaderView,
    QComboBox, QDateEdit, QPushButton, QGroupBox, QFormLayout,
    QTextEdit, QSplitter, QTreeWidget, QTreeWidgetItem, QFrame,
    QMessageBox, QFileDialog, QProgressBar, QCheckBox, QSpinBox,
    QScrollArea, QGridLayout, QButtonGroup, QRadioButton, QMenu,
    QAction, QToolButton, QListWidget, QListWidgetItem
)
from PyQt5.QtCore import (
    Qt, pyqtSignal, QTimer, QThread, pyqtSlot, QDate, QDateTime,
    QAbstractTableModel, QModelIndex, QVariant, QSortFilterProxyModel
)
from PyQt5.QtGui import QFont, QIcon, QPixmap, QPalette, QColor, QBrush

from .shared_widgets import CollapsiblePanel, ResultCard
from .backend_integration import get_backend_integration

class HistoryTableModel(QAbstractTableModel):
    """Modelo de tabla para el historial"""
    
    def __init__(self, data: List[Dict[str, Any]] = None, parent=None):
        super().__init__(parent)
        self.history_data = data or []
        self.headers = [
            "Fecha/Hora", "Tipo", "Archivo", "Estado", 
            "Duración", "Resultados", "Acciones"
        ]
    
    def rowCount(self, parent=QModelIndex()):
        return len(self.history_data)
    
    def columnCount(self, parent=QModelIndex()):
        return len(self.headers)
    
    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid() or index.row() >= len(self.history_data):
            return QVariant()
        
        item = self.history_data[index.row()]
        column = index.column()
        
        if role == Qt.DisplayRole:
            if column == 0:  # Fecha/Hora
                return item.get('timestamp', 'N/A')
            elif column == 1:  # Tipo
                return item.get('type', 'N/A')
            elif column == 2:  # Archivo
                return item.get('filename', 'N/A')
            elif column == 3:  # Estado
                return item.get('status', 'N/A')
            elif column == 4:  # Duración
                duration = item.get('duration', 0)
                return f"{duration:.2f}s" if duration else 'N/A'
            elif column == 5:  # Resultados
                return item.get('results_summary', 'N/A')
        
        elif role == Qt.BackgroundRole:
            status = item.get('status', '')
            if status == 'Completado':
                return QBrush(QColor(200, 255, 200))  # Verde claro
            elif status == 'Error':
                return QBrush(QColor(255, 200, 200))  # Rojo claro
            elif status == 'En Progreso':
                return QBrush(QColor(255, 255, 200))  # Amarillo claro
        
        elif role == Qt.UserRole:
            return item
        
        return QVariant()
    
    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return self.headers[section]
        return QVariant()
    
    def update_data(self, new_data: List[Dict[str, Any]]):
        """Actualiza los datos del modelo"""
        self.beginResetModel()
        self.history_data = new_data
        self.endResetModel()
    
    def add_item(self, item: Dict[str, Any]):
        """Añade un elemento al historial"""
        self.beginInsertRows(QModelIndex(), 0, 0)
        self.history_data.insert(0, item)
        self.endInsertRows()
    
    def remove_item(self, row: int):
        """Elimina un elemento del historial"""
        if 0 <= row < len(self.history_data):
            self.beginRemoveRows(QModelIndex(), row, row)
            del self.history_data[row]
            self.endRemoveRows()

class HistoryFilterWidget(QWidget):
    """Widget para filtrar el historial"""
    
    filter_changed = pyqtSignal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
    
    def setup_ui(self):
        """Configura la interfaz de usuario"""
        layout = QVBoxLayout(self)
        
        # Filtros básicos
        basic_group = QGroupBox("Filtros Básicos")
        basic_layout = QFormLayout(basic_group)
        
        # Filtro por tipo
        self.type_combo = QComboBox()
        self.type_combo.addItems([
            "Todos", "Análisis Individual", "Comparación Directa",
            "Búsqueda en BD", "Reporte", "Exportación"
        ])
        self.type_combo.currentTextChanged.connect(self._emit_filter_changed)
        basic_layout.addRow("Tipo:", self.type_combo)
        
        # Filtro por estado
        self.status_combo = QComboBox()
        self.status_combo.addItems([
            "Todos", "Completado", "Error", "En Progreso", "Cancelado"
        ])
        self.status_combo.currentTextChanged.connect(self._emit_filter_changed)
        basic_layout.addRow("Estado:", self.status_combo)
        
        # Filtro por fecha
        self.date_from = QDateEdit()
        self.date_from.setDate(QDate.currentDate().addDays(-30))
        self.date_from.setCalendarPopup(True)
        self.date_from.dateChanged.connect(self._emit_filter_changed)
        basic_layout.addRow("Desde:", self.date_from)
        
        self.date_to = QDateEdit()
        self.date_to.setDate(QDate.currentDate())
        self.date_to.setCalendarPopup(True)
        self.date_to.dateChanged.connect(self._emit_filter_changed)
        basic_layout.addRow("Hasta:", self.date_to)
        
        layout.addWidget(basic_group)
        
        # Filtros avanzados
        advanced_group = CollapsiblePanel("Filtros Avanzados")
        advanced_content = QWidget()
        advanced_layout = QFormLayout(advanced_content)
        
        # Búsqueda por texto
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Buscar en archivos, resultados...")
        self.search_edit.textChanged.connect(self._emit_filter_changed)
        advanced_layout.addRow("Búsqueda:", self.search_edit)
        
        # Filtro por duración
        duration_layout = QHBoxLayout()
        self.min_duration = QSpinBox()
        self.min_duration.setRange(0, 3600)
        self.min_duration.setSuffix(" s")
        self.min_duration.valueChanged.connect(self._emit_filter_changed)
        
        self.max_duration = QSpinBox()
        self.max_duration.setRange(0, 3600)
        self.max_duration.setValue(3600)
        self.max_duration.setSuffix(" s")
        self.max_duration.valueChanged.connect(self._emit_filter_changed)
        
        duration_layout.addWidget(QLabel("Min:"))
        duration_layout.addWidget(self.min_duration)
        duration_layout.addWidget(QLabel("Max:"))
        duration_layout.addWidget(self.max_duration)
        
        advanced_layout.addRow("Duración:", duration_layout)
        
        # Filtros de calidad
        self.only_successful = QCheckBox("Solo exitosos")
        self.only_successful.stateChanged.connect(self._emit_filter_changed)
        advanced_layout.addRow(self.only_successful)
        
        self.only_with_results = QCheckBox("Solo con resultados")
        self.only_with_results.stateChanged.connect(self._emit_filter_changed)
        advanced_layout.addRow(self.only_with_results)
        
        advanced_group.set_content_widget(advanced_content)
        layout.addWidget(advanced_group)
        
        # Botones de acción
        buttons_layout = QHBoxLayout()
        
        self.clear_filters_btn = QPushButton("Limpiar Filtros")
        self.clear_filters_btn.clicked.connect(self.clear_filters)
        buttons_layout.addWidget(self.clear_filters_btn)
        
        self.refresh_btn = QPushButton("Actualizar")
        self.refresh_btn.clicked.connect(self._emit_filter_changed)
        buttons_layout.addWidget(self.refresh_btn)
        
        buttons_layout.addStretch()
        layout.addLayout(buttons_layout)
    
    def _emit_filter_changed(self):
        """Emite la señal de cambio de filtros"""
        filters = self.get_current_filters()
        self.filter_changed.emit(filters)
    
    def get_current_filters(self) -> Dict[str, Any]:
        """Obtiene los filtros actuales"""
        return {
            'type': self.type_combo.currentText(),
            'status': self.status_combo.currentText(),
            'date_from': self.date_from.date().toPyDate(),
            'date_to': self.date_to.date().toPyDate(),
            'search_text': self.search_edit.text(),
            'min_duration': self.min_duration.value(),
            'max_duration': self.max_duration.value(),
            'only_successful': self.only_successful.isChecked(),
            'only_with_results': self.only_with_results.isChecked(),
        }
    
    def clear_filters(self):
        """Limpia todos los filtros"""
        self.type_combo.setCurrentText("Todos")
        self.status_combo.setCurrentText("Todos")
        self.date_from.setDate(QDate.currentDate().addDays(-30))
        self.date_to.setDate(QDate.currentDate())
        self.search_edit.clear()
        self.min_duration.setValue(0)
        self.max_duration.setValue(3600)
        self.only_successful.setChecked(False)
        self.only_with_results.setChecked(False)
        self._emit_filter_changed()

class HistoryDetailWidget(QWidget):
    """Widget para mostrar detalles de un elemento del historial"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_item = None
        self.setup_ui()
    
    def setup_ui(self):
        """Configura la interfaz de usuario"""
        layout = QVBoxLayout(self)
        
        # Título
        self.title_label = QLabel("Seleccione un elemento para ver detalles")
        self.title_label.setObjectName("sectionTitle")
        layout.addWidget(self.title_label)
        
        # Área de scroll para detalles
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        self.details_widget = QWidget()
        self.details_layout = QVBoxLayout(self.details_widget)
        
        scroll_area.setWidget(self.details_widget)
        layout.addWidget(scroll_area)
        
        # Botones de acción
        actions_layout = QHBoxLayout()
        
        self.view_results_btn = QPushButton("Ver Resultados")
        self.view_results_btn.clicked.connect(self._view_results)
        self.view_results_btn.setEnabled(False)
        actions_layout.addWidget(self.view_results_btn)
        
        self.export_item_btn = QPushButton("Exportar")
        self.export_item_btn.clicked.connect(self._export_item)
        self.export_item_btn.setEnabled(False)
        actions_layout.addWidget(self.export_item_btn)
        
        self.delete_item_btn = QPushButton("Eliminar")
        self.delete_item_btn.clicked.connect(self._delete_item)
        self.delete_item_btn.setEnabled(False)
        actions_layout.addWidget(self.delete_item_btn)
        
        actions_layout.addStretch()
        layout.addLayout(actions_layout)
    
    def show_item_details(self, item: Dict[str, Any]):
        """Muestra los detalles de un elemento"""
        self.current_item = item
        
        # Limpiar layout anterior
        for i in reversed(range(self.details_layout.count())):
            child = self.details_layout.itemAt(i).widget()
            if child:
                child.setParent(None)
        
        if not item:
            self.title_label.setText("Seleccione un elemento para ver detalles")
            self.view_results_btn.setEnabled(False)
            self.export_item_btn.setEnabled(False)
            self.delete_item_btn.setEnabled(False)
            return
        
        # Actualizar título
        self.title_label.setText(f"Detalles: {item.get('type', 'N/A')} - {item.get('filename', 'N/A')}")
        
        # Información básica
        basic_group = QGroupBox("Información Básica")
        basic_layout = QFormLayout(basic_group)
        
        basic_layout.addRow("Fecha/Hora:", QLabel(item.get('timestamp', 'N/A')))
        basic_layout.addRow("Tipo:", QLabel(item.get('type', 'N/A')))
        basic_layout.addRow("Archivo:", QLabel(item.get('filename', 'N/A')))
        basic_layout.addRow("Estado:", QLabel(item.get('status', 'N/A')))
        basic_layout.addRow("Duración:", QLabel(f"{item.get('duration', 0):.2f}s"))
        
        self.details_layout.addWidget(basic_group)
        
        # Parámetros de configuración
        if 'config' in item and item['config']:
            config_group = QGroupBox("Configuración Utilizada")
            config_layout = QVBoxLayout(config_group)
            
            config_text = QTextEdit()
            config_text.setReadOnly(True)
            config_text.setMaximumHeight(150)
            config_text.setPlainText(json.dumps(item['config'], indent=2, ensure_ascii=False))
            config_layout.addWidget(config_text)
            
            self.details_layout.addWidget(config_group)
        
        # Resultados
        if 'results' in item and item['results']:
            results_group = QGroupBox("Resumen de Resultados")
            results_layout = QVBoxLayout(results_group)
            
            results_text = QTextEdit()
            results_text.setReadOnly(True)
            results_text.setMaximumHeight(200)
            
            # Formatear resultados según el tipo
            results_str = self._format_results(item['results'], item.get('type', ''))
            results_text.setPlainText(results_str)
            results_layout.addWidget(results_text)
            
            self.details_layout.addWidget(results_group)
        
        # Errores (si los hay)
        if 'error' in item and item['error']:
            error_group = QGroupBox("Información de Error")
            error_layout = QVBoxLayout(error_group)
            
            error_text = QTextEdit()
            error_text.setReadOnly(True)
            error_text.setMaximumHeight(100)
            error_text.setPlainText(str(item['error']))
            error_text.setStyleSheet("QTextEdit { background-color: #ffe6e6; }")
            error_layout.addWidget(error_text)
            
            self.details_layout.addWidget(error_group)
        
        # Archivos asociados
        if 'files' in item and item['files']:
            files_group = QGroupBox("Archivos Asociados")
            files_layout = QVBoxLayout(files_group)
            
            for file_path in item['files']:
                file_label = QLabel(file_path)
                file_label.setWordWrap(True)
                files_layout.addWidget(file_label)
            
            self.details_layout.addWidget(files_group)
        
        # Habilitar botones
        self.view_results_btn.setEnabled('results' in item and bool(item['results']))
        self.export_item_btn.setEnabled(True)
        self.delete_item_btn.setEnabled(True)
        
        # Espaciador
        self.details_layout.addStretch()
    
    def _format_results(self, results: Dict[str, Any], analysis_type: str) -> str:
        """Formatea los resultados según el tipo de análisis"""
        if analysis_type == "Análisis Individual":
            return f"""
Calidad de imagen: {results.get('quality_score', 'N/A')}
Minucias detectadas: {results.get('minutiae_count', 'N/A')}
Análisis estadístico: {results.get('statistical_summary', 'N/A')}
Cumplimiento NIST: {results.get('nist_compliance', 'N/A')}
            """.strip()
        
        elif analysis_type == "Comparación Directa":
            return f"""
Similitud: {results.get('similarity_score', 'N/A')}%
Minucias coincidentes: {results.get('matching_minutiae', 'N/A')}
Confianza: {results.get('confidence', 'N/A')}
Resultado: {results.get('match_result', 'N/A')}
            """.strip()
        
        elif analysis_type == "Búsqueda en BD":
            return f"""
Resultados encontrados: {results.get('total_results', 'N/A')}
Mejor coincidencia: {results.get('best_match_score', 'N/A')}%
Tiempo de búsqueda: {results.get('search_time', 'N/A')}s
            """.strip()
        
        else:
            return json.dumps(results, indent=2, ensure_ascii=False)
    
    def _view_results(self):
        """Visualiza los resultados completos"""
        if not self.current_item or 'results' not in self.current_item:
            return
        
        # Crear diálogo para mostrar resultados
        dialog = QDialog(self)
        dialog.setWindowTitle("Resultados Completos")
        dialog.resize(600, 400)
        
        layout = QVBoxLayout(dialog)
        
        text_edit = QTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setPlainText(json.dumps(self.current_item['results'], indent=2, ensure_ascii=False))
        layout.addWidget(text_edit)
        
        close_btn = QPushButton("Cerrar")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)
        
        dialog.exec_()
    
    def _export_item(self):
        """Exporta el elemento actual"""
        if not self.current_item:
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Exportar Elemento del Historial",
            f"historial_{self.current_item.get('id', 'item')}.json",
            "JSON (*.json);;Todos los archivos (*)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(self.current_item, f, indent=2, ensure_ascii=False, default=str)
                
                QMessageBox.information(self, "Éxito", "Elemento exportado exitosamente.")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error exportando elemento: {str(e)}")
    
    def _delete_item(self):
        """Elimina el elemento actual del historial"""
        if not self.current_item:
            return
        
        reply = QMessageBox.question(
            self,
            "Confirmar Eliminación",
            "¿Está seguro de que desea eliminar este elemento del historial?\n"
            "Esta acción no se puede deshacer.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # Emitir señal para eliminar (implementar en el diálogo principal)
            pass

class HistoryDialog(QDialog):
    """Diálogo principal del historial"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Historial de Análisis - SIGeC-Balistica")
        self.setModal(False)
        self.resize(1200, 800)
        
        # Modelo de datos
        self.history_model = HistoryTableModel()
        self.proxy_model = QSortFilterProxyModel()
        self.proxy_model.setSourceModel(self.history_model)
        
        self.setup_ui()
        self.load_history()
    
    def setup_ui(self):
        """Configura la interfaz de usuario"""
        layout = QHBoxLayout(self)
        
        # Panel izquierdo - Filtros
        left_panel = QWidget()
        left_panel.setMaximumWidth(300)
        left_panel.setMinimumWidth(250)
        left_layout = QVBoxLayout(left_panel)
        
        # Widget de filtros
        self.filter_widget = HistoryFilterWidget()
        self.filter_widget.filter_changed.connect(self.apply_filters)
        left_layout.addWidget(self.filter_widget)
        
        layout.addWidget(left_panel)
        
        # Panel central - Tabla y detalles
        central_splitter = QSplitter(Qt.Vertical)
        
        # Tabla de historial
        table_widget = QWidget()
        table_layout = QVBoxLayout(table_widget)
        
        # Barra de herramientas de tabla
        table_toolbar = QHBoxLayout()
        
        self.export_history_btn = QPushButton("Exportar Historial")
        self.export_history_btn.clicked.connect(self.export_history)
        table_toolbar.addWidget(self.export_history_btn)
        
        self.clear_history_btn = QPushButton("Limpiar Historial")
        self.clear_history_btn.clicked.connect(self.clear_history)
        table_toolbar.addWidget(self.clear_history_btn)
        
        table_toolbar.addStretch()
        
        self.refresh_btn = QPushButton("Actualizar")
        self.refresh_btn.clicked.connect(self.load_history)
        table_toolbar.addWidget(self.refresh_btn)
        
        table_layout.addLayout(table_toolbar)
        
        # Tabla
        self.history_table = QTableWidget()
        self.history_table.setModel(self.proxy_model)
        self.history_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.history_table.setAlternatingRowColors(True)
        self.history_table.setSortingEnabled(True)
        
        # Configurar encabezados
        header = self.history_table.horizontalHeader()
        header.setStretchLastSection(True)
        header.setSectionResizeMode(QHeaderView.Interactive)
        
        # Conectar selección
        self.history_table.selectionModel().currentRowChanged.connect(self.on_selection_changed)
        
        table_layout.addWidget(self.history_table)
        central_splitter.addWidget(table_widget)
        
        # Panel de detalles
        self.detail_widget = HistoryDetailWidget()
        central_splitter.addWidget(self.detail_widget)
        
        # Configurar proporciones del splitter
        central_splitter.setSizes([400, 300])
        
        layout.addWidget(central_splitter)
    
    def load_history(self):
        """Carga el historial desde el backend"""
        try:
            backend = get_backend_integration()
            # history_data = backend.get_analysis_history()
            
            # Por ahora, usar datos de ejemplo
            history_data = self.get_sample_history()
            
            self.history_model.update_data(history_data)
            
        except Exception as e:
            QMessageBox.warning(
                self,
                "Advertencia",
                f"No se pudo cargar el historial: {str(e)}"
            )
    
    def get_sample_history(self) -> List[Dict[str, Any]]:
        """Genera datos de ejemplo para el historial"""
        sample_data = []
        
        # Generar entradas de ejemplo
        for i in range(20):
            timestamp = datetime.now() - timedelta(days=i, hours=i*2)
            
            sample_data.append({
                'id': f'analysis_{i}',
                'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                'type': ['Análisis Individual', 'Comparación Directa', 'Búsqueda en BD'][i % 3],
                'filename': f'muestra_{i:03d}.png',
                'status': ['Completado', 'Error', 'En Progreso'][i % 3],
                'duration': 15.5 + i * 2.3,
                'results_summary': f'Calidad: {85 + i}%, Minucias: {12 + i}',
                'config': {
                    'quality_threshold': 0.7,
                    'enhancement': True,
                    'algorithm': 'Automático'
                },
                'results': {
                    'quality_score': 0.85 + i * 0.01,
                    'minutiae_count': 12 + i,
                    'processing_time': 15.5 + i * 2.3
                },
                'files': [f'/path/to/muestra_{i:03d}.png'],
                'error': 'Error de procesamiento' if i % 3 == 1 else None
            })
        
        return sample_data
    
    def apply_filters(self, filters: Dict[str, Any]):
        """Aplica filtros a la tabla"""
        # Implementar filtrado personalizado
        # Por ahora, filtro básico por texto
        search_text = filters.get('search_text', '')
        if search_text:
            self.proxy_model.setFilterRegExp(search_text)
            self.proxy_model.setFilterKeyColumn(-1)  # Buscar en todas las columnas
        else:
            self.proxy_model.setFilterRegExp('')
    
    def on_selection_changed(self, current, previous):
        """Maneja el cambio de selección en la tabla"""
        if current.isValid():
            source_index = self.proxy_model.mapToSource(current)
            item_data = self.history_model.data(source_index, Qt.UserRole)
            self.detail_widget.show_item_details(item_data)
        else:
            self.detail_widget.show_item_details(None)
    
    def export_history(self):
        """Exporta el historial completo"""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Exportar Historial",
            f"historial_SIGeC-Balistica_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            "JSON (*.json);;CSV (*.csv);;Todos los archivos (*)"
        )
        
        if file_path:
            try:
                if file_path.endswith('.csv'):
                    self._export_to_csv(file_path)
                else:
                    self._export_to_json(file_path)
                
                QMessageBox.information(self, "Éxito", "Historial exportado exitosamente.")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error exportando historial: {str(e)}")
    
    def _export_to_json(self, file_path: str):
        """Exporta a formato JSON"""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.history_model.history_data, f, indent=2, ensure_ascii=False, default=str)
    
    def _export_to_csv(self, file_path: str):
        """Exporta a formato CSV"""
        import csv
        
        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Escribir encabezados
            headers = ['Fecha/Hora', 'Tipo', 'Archivo', 'Estado', 'Duración', 'Resultados']
            writer.writerow(headers)
            
            # Escribir datos
            for item in self.history_model.history_data:
                row = [
                    item.get('timestamp', ''),
                    item.get('type', ''),
                    item.get('filename', ''),
                    item.get('status', ''),
                    f"{item.get('duration', 0):.2f}s",
                    item.get('results_summary', '')
                ]
                writer.writerow(row)
    
    def clear_history(self):
        """Limpia el historial"""
        reply = QMessageBox.question(
            self,
            "Confirmar Limpieza",
            "¿Está seguro de que desea limpiar todo el historial?\n"
            "Esta acción no se puede deshacer.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            try:
                backend = get_backend_integration()
                # backend.clear_analysis_history()
                
                self.history_model.update_data([])
                self.detail_widget.show_item_details(None)
                
                QMessageBox.information(self, "Éxito", "Historial limpiado exitosamente.")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error limpiando historial: {str(e)}")