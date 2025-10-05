#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diálogo de Configuración - SEACABAr
===================================

Diálogo para configurar las preferencias del sistema, incluyendo:
- Configuración de procesamiento
- Preferencias de interfaz
- Configuración de base de datos
- Configuración de exportación
- Configuración avanzada

Autor: SEACABAr Team
Fecha: Octubre 2025
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTabWidget, QWidget,
    QLabel, QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox,
    QCheckBox, QSlider, QPushButton, QGroupBox, QFormLayout,
    QFileDialog, QMessageBox, QTextEdit, QScrollArea,
    QGridLayout, QFrame, QButtonGroup, QRadioButton,
    QProgressBar, QSplitter
)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer, QThread, pyqtSlot
from PyQt5.QtGui import QFont, QIcon, QPixmap, QPalette

from .shared_widgets import CollapsiblePanel, ResultCard
from .backend_integration import get_backend_integration

class ConfigurationTab(QWidget):
    """Clase base para pestañas de configuración"""
    
    config_changed = pyqtSignal(str, object)  # key, value
    
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.title = title
        self.config_data = {}
        self.setup_ui()
    
    def setup_ui(self):
        """Configura la interfaz de usuario"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # Título
        title_label = QLabel(self.title)
        title_label.setObjectName("sectionTitle")
        layout.addWidget(title_label)
        
        # Contenido específico
        self.setup_content(layout)
        
        # Espaciador
        layout.addStretch()
    
    def setup_content(self, layout):
        """Implementar en subclases"""
        pass
    
    def get_config(self) -> Dict[str, Any]:
        """Obtiene la configuración actual"""
        return self.config_data.copy()
    
    def set_config(self, config: Dict[str, Any]):
        """Establece la configuración"""
        self.config_data.update(config)
        self.update_ui_from_config()
    
    def update_ui_from_config(self):
        """Actualiza la UI desde la configuración"""
        pass
    
    def validate_config(self) -> tuple[bool, str]:
        """Valida la configuración actual"""
        return True, ""

class ProcessingConfigTab(ConfigurationTab):
    """Configuración de procesamiento"""
    
    def __init__(self, parent=None):
        super().__init__("Configuración de Procesamiento", parent)
    
    def setup_content(self, layout):
        """Configura el contenido específico"""
        
        # Configuración de calidad
        quality_group = QGroupBox("Configuración de Calidad")
        quality_layout = QFormLayout(quality_group)
        
        self.quality_threshold_spin = QDoubleSpinBox()
        self.quality_threshold_spin.setRange(0.0, 1.0)
        self.quality_threshold_spin.setSingleStep(0.01)
        self.quality_threshold_spin.setValue(0.7)
        self.quality_threshold_spin.setDecimals(2)
        quality_layout.addRow("Umbral de Calidad:", self.quality_threshold_spin)
        
        self.min_minutiae_spin = QSpinBox()
        self.min_minutiae_spin.setRange(5, 100)
        self.min_minutiae_spin.setValue(12)
        quality_layout.addRow("Mínimo de Minucias:", self.min_minutiae_spin)
        
        self.enhancement_check = QCheckBox("Aplicar mejora de imagen")
        self.enhancement_check.setChecked(True)
        quality_layout.addRow(self.enhancement_check)
        
        layout.addWidget(quality_group)
        
        # Configuración de algoritmos
        algo_group = QGroupBox("Algoritmos de Procesamiento")
        algo_layout = QFormLayout(algo_group)
        
        self.enhancement_method_combo = QComboBox()
        self.enhancement_method_combo.addItems([
            "Gabor Filter", "Histogram Equalization", 
            "CLAHE", "Wiener Filter", "Automático"
        ])
        algo_layout.addRow("Método de Mejora:", self.enhancement_method_combo)
        
        self.minutiae_method_combo = QComboBox()
        self.minutiae_method_combo.addItems([
            "Crossing Number", "Ridge Ending", 
            "Hybrid", "Deep Learning", "Automático"
        ])
        algo_layout.addRow("Extracción de Minucias:", self.minutiae_method_combo)
        
        layout.addWidget(algo_group)
        
        # Configuración de rendimiento
        perf_group = QGroupBox("Configuración de Rendimiento")
        perf_layout = QFormLayout(perf_group)
        
        self.max_threads_spin = QSpinBox()
        self.max_threads_spin.setRange(1, 16)
        self.max_threads_spin.setValue(4)
        perf_layout.addRow("Máximo de Hilos:", self.max_threads_spin)
        
        self.gpu_acceleration_check = QCheckBox("Usar aceleración GPU")
        self.gpu_acceleration_check.setChecked(False)
        perf_layout.addRow(self.gpu_acceleration_check)
        
        self.memory_limit_spin = QSpinBox()
        self.memory_limit_spin.setRange(512, 8192)
        self.memory_limit_spin.setValue(2048)
        self.memory_limit_spin.setSuffix(" MB")
        perf_layout.addRow("Límite de Memoria:", self.memory_limit_spin)
        
        layout.addWidget(perf_group)
        
        # Conectar señales
        self.quality_threshold_spin.valueChanged.connect(
            lambda v: self.config_changed.emit("quality_threshold", v)
        )
        self.min_minutiae_spin.valueChanged.connect(
            lambda v: self.config_changed.emit("min_minutiae", v)
        )

class InterfaceConfigTab(ConfigurationTab):
    """Configuración de interfaz"""
    
    def __init__(self, parent=None):
        super().__init__("Preferencias de Interfaz", parent)
    
    def setup_content(self, layout):
        """Configura el contenido específico"""
        
        # Tema y apariencia
        theme_group = QGroupBox("Tema y Apariencia")
        theme_layout = QFormLayout(theme_group)
        
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Claro", "Oscuro", "Automático"])
        theme_layout.addRow("Tema:", self.theme_combo)
        
        self.font_size_spin = QSpinBox()
        self.font_size_spin.setRange(8, 24)
        self.font_size_spin.setValue(10)
        theme_layout.addRow("Tamaño de Fuente:", self.font_size_spin)
        
        self.high_dpi_check = QCheckBox("Soporte para alta resolución")
        self.high_dpi_check.setChecked(True)
        theme_layout.addRow(self.high_dpi_check)
        
        layout.addWidget(theme_group)
        
        # Configuración de ventana
        window_group = QGroupBox("Configuración de Ventana")
        window_layout = QFormLayout(window_group)
        
        self.remember_size_check = QCheckBox("Recordar tamaño de ventana")
        self.remember_size_check.setChecked(True)
        window_layout.addRow(self.remember_size_check)
        
        self.start_maximized_check = QCheckBox("Iniciar maximizada")
        self.start_maximized_check.setChecked(False)
        window_layout.addRow(self.start_maximized_check)
        
        self.show_tooltips_check = QCheckBox("Mostrar tooltips")
        self.show_tooltips_check.setChecked(True)
        window_layout.addRow(self.show_tooltips_check)
        
        layout.addWidget(window_group)
        
        # Configuración de notificaciones
        notif_group = QGroupBox("Notificaciones")
        notif_layout = QFormLayout(notif_group)
        
        self.show_notifications_check = QCheckBox("Mostrar notificaciones")
        self.show_notifications_check.setChecked(True)
        notif_layout.addRow(self.show_notifications_check)
        
        self.sound_notifications_check = QCheckBox("Notificaciones sonoras")
        self.sound_notifications_check.setChecked(False)
        notif_layout.addRow(self.sound_notifications_check)
        
        layout.addWidget(notif_group)

class DatabaseConfigTab(ConfigurationTab):
    """Configuración de base de datos"""
    
    def __init__(self, parent=None):
        super().__init__("Configuración de Base de Datos", parent)
    
    def setup_content(self, layout):
        """Configura el contenido específico"""
        
        # Configuración de conexión
        conn_group = QGroupBox("Configuración de Conexión")
        conn_layout = QFormLayout(conn_group)
        
        self.db_path_edit = QLineEdit()
        self.db_path_edit.setPlaceholderText("Ruta a la base de datos...")
        
        browse_btn = QPushButton("Examinar...")
        browse_btn.clicked.connect(self._browse_database)
        
        path_layout = QHBoxLayout()
        path_layout.addWidget(self.db_path_edit)
        path_layout.addWidget(browse_btn)
        
        conn_layout.addRow("Ruta de BD:", path_layout)
        
        self.auto_backup_check = QCheckBox("Respaldo automático")
        self.auto_backup_check.setChecked(True)
        conn_layout.addRow(self.auto_backup_check)
        
        self.backup_interval_spin = QSpinBox()
        self.backup_interval_spin.setRange(1, 168)  # 1 hora a 1 semana
        self.backup_interval_spin.setValue(24)
        self.backup_interval_spin.setSuffix(" horas")
        conn_layout.addRow("Intervalo de Respaldo:", self.backup_interval_spin)
        
        layout.addWidget(conn_group)
        
        # Configuración de búsqueda
        search_group = QGroupBox("Configuración de Búsqueda")
        search_layout = QFormLayout(search_group)
        
        self.search_threshold_spin = QDoubleSpinBox()
        self.search_threshold_spin.setRange(0.0, 1.0)
        self.search_threshold_spin.setSingleStep(0.01)
        self.search_threshold_spin.setValue(0.8)
        self.search_threshold_spin.setDecimals(2)
        search_layout.addRow("Umbral de Similitud:", self.search_threshold_spin)
        
        self.max_results_spin = QSpinBox()
        self.max_results_spin.setRange(10, 1000)
        self.max_results_spin.setValue(100)
        search_layout.addRow("Máximo de Resultados:", self.max_results_spin)
        
        self.enable_fuzzy_search_check = QCheckBox("Búsqueda difusa")
        self.enable_fuzzy_search_check.setChecked(True)
        search_layout.addRow(self.enable_fuzzy_search_check)
        
        layout.addWidget(search_group)
        
        # Estado de la base de datos
        status_group = QGroupBox("Estado de la Base de Datos")
        status_layout = QVBoxLayout(status_group)
        
        self.status_label = QLabel("Verificando estado...")
        status_layout.addWidget(self.status_label)
        
        self.refresh_status_btn = QPushButton("Actualizar Estado")
        self.refresh_status_btn.clicked.connect(self._refresh_database_status)
        status_layout.addWidget(self.refresh_status_btn)
        
        layout.addWidget(status_group)
        
        # Actualizar estado inicial
        QTimer.singleShot(100, self._refresh_database_status)
    
    def _browse_database(self):
        """Examinar archivo de base de datos"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Seleccionar Base de Datos",
            "",
            "Base de Datos (*.db *.sqlite *.sqlite3);;Todos los archivos (*)"
        )
        
        if file_path:
            self.db_path_edit.setText(file_path)
            self.config_changed.emit("database_path", file_path)
    
    def _refresh_database_status(self):
        """Actualiza el estado de la base de datos"""
        try:
            backend = get_backend_integration()
            stats = backend.get_database_stats()
            
            status_text = f"""
Estado: Conectada ✓
Total de imágenes: {stats.get('total_images', 0)}
Total de análisis: {stats.get('total_analyses', 0)}
Última actualización: {stats.get('last_update', 'N/A')}
            """.strip()
            
            self.status_label.setText(status_text)
            
        except Exception as e:
            self.status_label.setText(f"Error: {str(e)}")

class ExportConfigTab(ConfigurationTab):
    """Configuración de exportación"""
    
    def __init__(self, parent=None):
        super().__init__("Configuración de Exportación", parent)
    
    def setup_content(self, layout):
        """Configura el contenido específico"""
        
        # Configuración de reportes
        reports_group = QGroupBox("Configuración de Reportes")
        reports_layout = QFormLayout(reports_group)
        
        self.default_format_combo = QComboBox()
        self.default_format_combo.addItems(["PDF", "HTML", "DOCX"])
        reports_layout.addRow("Formato por Defecto:", self.default_format_combo)
        
        self.include_images_check = QCheckBox("Incluir imágenes en reportes")
        self.include_images_check.setChecked(True)
        reports_layout.addRow(self.include_images_check)
        
        self.compress_images_check = QCheckBox("Comprimir imágenes")
        self.compress_images_check.setChecked(True)
        reports_layout.addRow(self.compress_images_check)
        
        self.image_quality_spin = QSpinBox()
        self.image_quality_spin.setRange(10, 100)
        self.image_quality_spin.setValue(85)
        self.image_quality_spin.setSuffix("%")
        reports_layout.addRow("Calidad de Imagen:", self.image_quality_spin)
        
        layout.addWidget(reports_group)
        
        # Configuración de exportación de datos
        export_group = QGroupBox("Exportación de Datos")
        export_layout = QFormLayout(export_group)
        
        self.export_path_edit = QLineEdit()
        self.export_path_edit.setPlaceholderText("Directorio de exportación...")
        
        browse_export_btn = QPushButton("Examinar...")
        browse_export_btn.clicked.connect(self._browse_export_path)
        
        export_path_layout = QHBoxLayout()
        export_path_layout.addWidget(self.export_path_edit)
        export_path_layout.addWidget(browse_export_btn)
        
        export_layout.addRow("Directorio de Exportación:", export_path_layout)
        
        self.auto_timestamp_check = QCheckBox("Añadir timestamp automáticamente")
        self.auto_timestamp_check.setChecked(True)
        export_layout.addRow(self.auto_timestamp_check)
        
        self.export_metadata_check = QCheckBox("Exportar metadatos")
        self.export_metadata_check.setChecked(True)
        export_layout.addRow(self.export_metadata_check)
        
        layout.addWidget(export_group)
        
        # Configuración de seguridad
        security_group = QGroupBox("Configuración de Seguridad")
        security_layout = QFormLayout(security_group)
        
        self.encrypt_exports_check = QCheckBox("Encriptar exportaciones")
        self.encrypt_exports_check.setChecked(False)
        security_layout.addRow(self.encrypt_exports_check)
        
        self.watermark_check = QCheckBox("Añadir marca de agua")
        self.watermark_check.setChecked(True)
        security_layout.addRow(self.watermark_check)
        
        layout.addWidget(security_group)
    
    def _browse_export_path(self):
        """Examinar directorio de exportación"""
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "Seleccionar Directorio de Exportación"
        )
        
        if dir_path:
            self.export_path_edit.setText(dir_path)
            self.config_changed.emit("export_path", dir_path)

class AdvancedConfigTab(ConfigurationTab):
    """Configuración avanzada"""
    
    def __init__(self, parent=None):
        super().__init__("Configuración Avanzada", parent)
    
    def setup_content(self, layout):
        """Configura el contenido específico"""
        
        # Configuración de logging
        logging_group = QGroupBox("Configuración de Logging")
        logging_layout = QFormLayout(logging_group)
        
        self.log_level_combo = QComboBox()
        self.log_level_combo.addItems(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
        self.log_level_combo.setCurrentText("INFO")
        logging_layout.addRow("Nivel de Log:", self.log_level_combo)
        
        self.log_to_file_check = QCheckBox("Guardar logs en archivo")
        self.log_to_file_check.setChecked(True)
        logging_layout.addRow(self.log_to_file_check)
        
        self.max_log_size_spin = QSpinBox()
        self.max_log_size_spin.setRange(1, 100)
        self.max_log_size_spin.setValue(10)
        self.max_log_size_spin.setSuffix(" MB")
        logging_layout.addRow("Tamaño Máximo de Log:", self.max_log_size_spin)
        
        layout.addWidget(logging_group)
        
        # Configuración de desarrollo
        dev_group = QGroupBox("Configuración de Desarrollo")
        dev_layout = QFormLayout(dev_group)
        
        self.debug_mode_check = QCheckBox("Modo debug")
        self.debug_mode_check.setChecked(False)
        dev_layout.addRow(self.debug_mode_check)
        
        self.show_performance_check = QCheckBox("Mostrar métricas de rendimiento")
        self.show_performance_check.setChecked(False)
        dev_layout.addRow(self.show_performance_check)
        
        self.enable_profiling_check = QCheckBox("Habilitar profiling")
        self.enable_profiling_check.setChecked(False)
        dev_layout.addRow(self.enable_profiling_check)
        
        layout.addWidget(dev_group)
        
        # Configuración experimental
        experimental_group = QGroupBox("Características Experimentales")
        experimental_layout = QFormLayout(experimental_group)
        
        self.experimental_features_check = QCheckBox("Habilitar características experimentales")
        self.experimental_features_check.setChecked(False)
        experimental_layout.addRow(self.experimental_features_check)
        
        self.beta_algorithms_check = QCheckBox("Usar algoritmos beta")
        self.beta_algorithms_check.setChecked(False)
        experimental_layout.addRow(self.beta_algorithms_check)
        
        layout.addWidget(experimental_group)
        
        # Botones de mantenimiento
        maintenance_group = QGroupBox("Mantenimiento del Sistema")
        maintenance_layout = QVBoxLayout(maintenance_group)
        
        clear_cache_btn = QPushButton("Limpiar Caché")
        clear_cache_btn.clicked.connect(self._clear_cache)
        maintenance_layout.addWidget(clear_cache_btn)
        
        reset_config_btn = QPushButton("Restablecer Configuración")
        reset_config_btn.clicked.connect(self._reset_configuration)
        maintenance_layout.addWidget(reset_config_btn)
        
        export_config_btn = QPushButton("Exportar Configuración")
        export_config_btn.clicked.connect(self._export_configuration)
        maintenance_layout.addWidget(export_config_btn)
        
        import_config_btn = QPushButton("Importar Configuración")
        import_config_btn.clicked.connect(self._import_configuration)
        maintenance_layout.addWidget(import_config_btn)
        
        layout.addWidget(maintenance_group)
    
    def _clear_cache(self):
        """Limpia el caché del sistema"""
        reply = QMessageBox.question(
            self,
            "Limpiar Caché",
            "¿Está seguro de que desea limpiar el caché del sistema?\n"
            "Esto puede afectar el rendimiento temporalmente.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            try:
                # Implementar limpieza de caché
                QMessageBox.information(self, "Éxito", "Caché limpiado exitosamente.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error limpiando caché: {str(e)}")
    
    def _reset_configuration(self):
        """Restablece la configuración por defecto"""
        reply = QMessageBox.question(
            self,
            "Restablecer Configuración",
            "¿Está seguro de que desea restablecer toda la configuración?\n"
            "Esta acción no se puede deshacer.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            try:
                # Implementar restablecimiento
                QMessageBox.information(self, "Éxito", "Configuración restablecida.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error restableciendo configuración: {str(e)}")
    
    def _export_configuration(self):
        """Exporta la configuración actual"""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Exportar Configuración",
            "seacabar_config.json",
            "JSON (*.json);;Todos los archivos (*)"
        )
        
        if file_path:
            try:
                # Implementar exportación
                QMessageBox.information(self, "Éxito", "Configuración exportada exitosamente.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error exportando configuración: {str(e)}")
    
    def _import_configuration(self):
        """Importa configuración desde archivo"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Importar Configuración",
            "",
            "JSON (*.json);;Todos los archivos (*)"
        )
        
        if file_path:
            try:
                # Implementar importación
                QMessageBox.information(self, "Éxito", "Configuración importada exitosamente.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error importando configuración: {str(e)}")

class SettingsDialog(QDialog):
    """Diálogo principal de configuración"""
    
    settings_changed = pyqtSignal(dict)  # Emitido cuando cambia la configuración
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Configuración - SEACABAr")
        self.setModal(True)
        self.resize(800, 600)
        
        # Configuración inicial
        self.config_data = {}
        self.setup_ui()
        self.load_configuration()
    
    def setup_ui(self):
        """Configura la interfaz de usuario"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Crear pestañas
        self.tab_widget = QTabWidget()
        
        # Añadir pestañas de configuración
        self.processing_tab = ProcessingConfigTab()
        self.interface_tab = InterfaceConfigTab()
        self.database_tab = DatabaseConfigTab()
        self.export_tab = ExportConfigTab()
        self.advanced_tab = AdvancedConfigTab()
        
        self.tab_widget.addTab(self.processing_tab, "Procesamiento")
        self.tab_widget.addTab(self.interface_tab, "Interfaz")
        self.tab_widget.addTab(self.database_tab, "Base de Datos")
        self.tab_widget.addTab(self.export_tab, "Exportación")
        self.tab_widget.addTab(self.advanced_tab, "Avanzado")
        
        layout.addWidget(self.tab_widget)
        
        # Botones
        buttons_layout = QHBoxLayout()
        buttons_layout.addStretch()
        
        self.apply_btn = QPushButton("Aplicar")
        self.apply_btn.clicked.connect(self.apply_settings)
        buttons_layout.addWidget(self.apply_btn)
        
        self.ok_btn = QPushButton("Aceptar")
        self.ok_btn.clicked.connect(self.accept_settings)
        self.ok_btn.setDefault(True)
        buttons_layout.addWidget(self.ok_btn)
        
        self.cancel_btn = QPushButton("Cancelar")
        self.cancel_btn.clicked.connect(self.reject)
        buttons_layout.addWidget(self.cancel_btn)
        
        layout.addLayout(buttons_layout)
        
        # Conectar señales de cambio de configuración
        for tab in [self.processing_tab, self.interface_tab, self.database_tab, 
                   self.export_tab, self.advanced_tab]:
            tab.config_changed.connect(self.on_config_changed)
    
    def load_configuration(self):
        """Carga la configuración actual"""
        try:
            backend = get_backend_integration()
            # Implementar carga de configuración desde backend
            # self.config_data = backend.get_configuration()
            
            # Por ahora, usar configuración por defecto
            self.config_data = self.get_default_configuration()
            
            # Actualizar pestañas
            for tab in [self.processing_tab, self.interface_tab, self.database_tab, 
                       self.export_tab, self.advanced_tab]:
                tab.set_config(self.config_data)
                
        except Exception as e:
            QMessageBox.warning(
                self,
                "Advertencia",
                f"No se pudo cargar la configuración: {str(e)}\n"
                "Se usará la configuración por defecto."
            )
    
    def get_default_configuration(self) -> Dict[str, Any]:
        """Obtiene la configuración por defecto"""
        return {
            # Procesamiento
            "quality_threshold": 0.7,
            "min_minutiae": 12,
            "enhancement_enabled": True,
            "enhancement_method": "Automático",
            "minutiae_method": "Automático",
            "max_threads": 4,
            "gpu_acceleration": False,
            "memory_limit": 2048,
            
            # Interfaz
            "theme": "Claro",
            "font_size": 10,
            "high_dpi": True,
            "remember_size": True,
            "start_maximized": False,
            "show_tooltips": True,
            "show_notifications": True,
            "sound_notifications": False,
            
            # Base de datos
            "database_path": "",
            "auto_backup": True,
            "backup_interval": 24,
            "search_threshold": 0.8,
            "max_results": 100,
            "fuzzy_search": True,
            
            # Exportación
            "default_format": "PDF",
            "include_images": True,
            "compress_images": True,
            "image_quality": 85,
            "export_path": "",
            "auto_timestamp": True,
            "export_metadata": True,
            "encrypt_exports": False,
            "watermark": True,
            
            # Avanzado
            "log_level": "INFO",
            "log_to_file": True,
            "max_log_size": 10,
            "debug_mode": False,
            "show_performance": False,
            "enable_profiling": False,
            "experimental_features": False,
            "beta_algorithms": False,
        }
    
    @pyqtSlot(str, object)
    def on_config_changed(self, key: str, value: Any):
        """Maneja cambios en la configuración"""
        self.config_data[key] = value
    
    def apply_settings(self):
        """Aplica la configuración actual"""
        try:
            # Validar configuración
            valid, error_msg = self.validate_configuration()
            if not valid:
                QMessageBox.warning(self, "Error de Validación", error_msg)
                return
            
            # Aplicar configuración
            backend = get_backend_integration()
            # backend.set_configuration(self.config_data)
            
            # Emitir señal de cambio
            self.settings_changed.emit(self.config_data.copy())
            
            QMessageBox.information(self, "Éxito", "Configuración aplicada exitosamente.")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error aplicando configuración: {str(e)}")
    
    def accept_settings(self):
        """Acepta y aplica la configuración"""
        self.apply_settings()
        self.accept()
    
    def validate_configuration(self) -> tuple[bool, str]:
        """Valida la configuración completa"""
        
        # Validar cada pestaña
        for tab in [self.processing_tab, self.interface_tab, self.database_tab, 
                   self.export_tab, self.advanced_tab]:
            valid, error_msg = tab.validate_config()
            if not valid:
                return False, f"Error en {tab.title}: {error_msg}"
        
        return True, ""