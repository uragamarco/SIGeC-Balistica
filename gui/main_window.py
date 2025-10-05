#!/usr/bin/env python3
"""
Ventana principal de SIGeC-Balistica
Interfaz moderna con tabs para análisis, comparación, base de datos y reportes
"""

import sys
import os
from typing import Optional
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
    QMenuBar, QStatusBar, QAction, QMessageBox, QSplitter,
    QFrame, QLabel, QPushButton, QApplication, QToolBar
)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from PyQt5.QtGui import QIcon, QFont, QPixmap

# Importar configuración existente
try:
    from unified_config import GUIConfig
except ImportError:
    # Configuración por defecto si no existe
    class GUIConfig:
        WINDOW_WIDTH = 1200
        WINDOW_HEIGHT = 800
        WINDOW_TITLE = "SIGeC-Balistica- Sistema de Análisis Forense"
        THEME = "modern"
        LANGUAGE = "es"

# Importar estilos y widgets
from .styles import SIGeCBallisticaTheme, apply_SIGeC_Balistica_theme
from .shared_widgets import StepIndicator, ProgressCard
from .settings_dialog import SettingsDialog
from .history_dialog import HistoryDialog
from .help_dialog import HelpDialog
from .about_dialog import AboutDialog

class MainWindow(QMainWindow):
    """Ventana principal de la aplicación SIGeC-Balistica"""
    
    # Señales para comunicación entre componentes
    analysisRequested = pyqtSignal(dict)
    comparisonRequested = pyqtSignal(dict)
    databaseSearchRequested = pyqtSignal(dict)
    reportGenerationRequested = pyqtSignal(dict)
    
    def __init__(self):
        super().__init__()
        self.theme = SIGeC-BalisticaTheme()
        self.current_analysis = None
        self.setup_ui()
        self.setup_menu_bar()
        self.setup_status_bar()
        self.setup_connections()
        
    def setup_ui(self):
        """Configura la interfaz principal"""
        # Configuración de ventana
        self.setWindowTitle(GUIConfig.WINDOW_TITLE)
        self.setMinimumSize(1000, 700)
        self.resize(GUIConfig.WINDOW_WIDTH, GUIConfig.WINDOW_HEIGHT)
        
        # Widget central
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Layout principal
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Header con información del sistema
        self.setup_header()
        main_layout.addWidget(self.header_frame)
        
        # Tabs principales
        self.setup_tabs()
        main_layout.addWidget(self.tab_widget)
        
    def setup_header(self):
        """Configura el header con información del sistema"""
        self.header_frame = QFrame()
        self.header_frame.setProperty("class", "header")
        self.header_frame.setFixedHeight(60)
        
        header_layout = QHBoxLayout(self.header_frame)
        header_layout.setContentsMargins(20, 10, 20, 10)
        
        # Logo y título
        title_layout = QHBoxLayout()
        
        # Logo (placeholder)
        logo_label = QLabel("🔬")
        logo_label.setStyleSheet("font-size: 24px;")
        title_layout.addWidget(logo_label)
        
        # Título y subtítulo
        title_container = QVBoxLayout()
        title_container.setSpacing(0)
        
        title_label = QLabel("SIGeC-Balistica")
        title_label.setProperty("class", "title")
        title_container.addWidget(title_label)
        
        subtitle_label = QLabel("Sistema de Análisis Forense de Imágenes")
        subtitle_label.setProperty("class", "caption")
        title_container.addWidget(subtitle_label)
        
        title_layout.addLayout(title_container)
        header_layout.addLayout(title_layout)
        
        # Spacer
        header_layout.addStretch()
        
        # Información del sistema
        system_info_layout = QVBoxLayout()
        system_info_layout.setSpacing(0)
        system_info_layout.setAlignment(Qt.AlignRight)
        
        # Estado del sistema
        self.system_status_label = QLabel("Sistema Listo")
        self.system_status_label.setProperty("class", "caption success")
        system_info_layout.addWidget(self.system_status_label)
        
        # Información adicional
        self.system_info_label = QLabel("Base de Datos: Conectada")
        self.system_info_label.setProperty("class", "caption")
        system_info_layout.addWidget(self.system_info_label)
        
        header_layout.addLayout(system_info_layout)
        
    def setup_tabs(self):
        """Configura las pestañas principales"""
        self.tab_widget = QTabWidget()
        self.tab_widget.setProperty("class", "main-tabs")
        
        # Importar pestañas reales
        try:
            from .analysis_tab import AnalysisTab
            from .comparison_tab import ComparisonTab
            from .database_tab import DatabaseTab
            from .reports_tab import ReportsTab
            
            # Crear pestañas reales
            self.analysis_tab = AnalysisTab()
            self.comparison_tab = ComparisonTab()
            self.database_tab = DatabaseTab()
            self.reports_tab = ReportsTab()
            
        except ImportError as e:
            print(f"Error importando pestañas: {e}")
            # Fallback a pestañas placeholder
            self.analysis_tab = self.create_placeholder_tab(
                "Análisis Individual",
                "Procesar y analizar una imagen individual",
                "🔍"
            )
            
            self.comparison_tab = self.create_placeholder_tab(
                "Análisis Comparativo", 
                "Comparar dos imágenes o buscar en base de datos",
                "⚖️"
            )
            
            self.database_tab = self.create_placeholder_tab(
                "Base de Datos",
                "Explorar y gestionar la base de datos de imágenes",
                "🗄️"
            )
            
            self.reports_tab = self.create_placeholder_tab(
                "Reportes",
                "Generar reportes profesionales de análisis",
                "📊"
            )
        
        # Añadir pestañas
        self.tab_widget.addTab(self.analysis_tab, "🔍 Análisis")
        self.tab_widget.addTab(self.comparison_tab, "⚖️ Comparación")
        self.tab_widget.addTab(self.database_tab, "🗄️ Base de Datos")
        self.tab_widget.addTab(self.reports_tab, "📊 Reportes")
        
        # Conectar cambio de pestaña
        self.tab_widget.currentChanged.connect(self.on_tab_changed)
        
    def create_placeholder_tab(self, title: str, description: str, icon: str) -> QWidget:
        """Crea una pestaña placeholder temporal"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(20)
        
        # Icono grande
        icon_label = QLabel(icon)
        icon_label.setAlignment(Qt.AlignCenter)
        icon_label.setStyleSheet("font-size: 64px; margin: 20px;")
        layout.addWidget(icon_label)
        
        # Título
        title_label = QLabel(title)
        title_label.setProperty("class", "title")
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        # Descripción
        desc_label = QLabel(description)
        desc_label.setProperty("class", "body")
        desc_label.setAlignment(Qt.AlignCenter)
        desc_label.setWordWrap(True)
        layout.addWidget(desc_label)
        
        # Estado
        status_label = QLabel("En desarrollo...")
        status_label.setProperty("class", "caption")
        status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(status_label)
        
        return tab
        
    def setup_menu_bar(self):
        """Configura la barra de menú"""
        menubar = self.menuBar()
        
        # Menú Archivo
        file_menu = menubar.addMenu("&Archivo")
        
        new_action = QAction("&Nuevo Análisis", self)
        new_action.setShortcut("Ctrl+N")
        new_action.triggered.connect(self.new_analysis)
        file_menu.addAction(new_action)
        
        open_action = QAction("&Abrir Proyecto", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.open_project)
        file_menu.addAction(open_action)
        
        save_action = QAction("&Guardar", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.save_project)
        file_menu.addAction(save_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("&Salir", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Menú Herramientas
        tools_menu = menubar.addMenu("&Herramientas")
        
        backup_action = QAction("&Respaldo de Base de Datos", self)
        backup_action.setShortcut("Ctrl+B")
        backup_action.triggered.connect(self.backup_database)
        tools_menu.addAction(backup_action)
        
        tools_menu.addSeparator()
        
        history_action = QAction("&Historial de Análisis", self)
        history_action.setShortcut("Ctrl+H")
        history_action.triggered.connect(self.show_history)
        tools_menu.addAction(history_action)
        
        tools_menu.addSeparator()
        
        preferences_action = QAction("&Configuración", self)
        preferences_action.setShortcut("Ctrl+,")
        preferences_action.triggered.connect(self.show_preferences)
        tools_menu.addAction(preferences_action)
        
        database_action = QAction("Gestión de &Base de Datos", self)
        database_action.triggered.connect(self.show_database_management)
        tools_menu.addAction(database_action)
        
        # Menú Ayuda
        help_menu = menubar.addMenu("A&yuda")
        
        user_guide_action = QAction("&Guía de Usuario", self)
        user_guide_action.setShortcut("F1")
        user_guide_action.triggered.connect(self.show_user_guide)
        help_menu.addAction(user_guide_action)
        
        help_action = QAction("&Ayuda Completa", self)
        help_action.setShortcut("Ctrl+F1")
        help_action.triggered.connect(self.show_help)
        help_menu.addAction(help_action)
        
        help_menu.addSeparator()
        
        support_action = QAction("&Soporte Técnico", self)
        support_action.triggered.connect(self.show_support)
        help_menu.addAction(support_action)
        
        help_menu.addSeparator()
        
        about_action = QAction("&Acerca de SIGeC-Balistica", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
        
    def setup_status_bar(self):
        """Configura la barra de estado"""
        self.status_bar = self.statusBar()
        
        # Mensaje principal
        self.status_bar.showMessage("Listo")
        
        # Widgets adicionales en la barra de estado
        self.progress_label = QLabel("Inactivo")
        self.progress_label.setProperty("class", "caption")
        self.status_bar.addPermanentWidget(self.progress_label)
        
        # Información de memoria/rendimiento
        self.performance_label = QLabel("Memoria: OK")
        self.performance_label.setProperty("class", "caption")
        self.status_bar.addPermanentWidget(self.performance_label)
        
        # Timer para actualizar información del sistema
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.update_system_status)
        self.status_timer.start(5000)  # Actualizar cada 5 segundos
        
    def setup_connections(self):
        """Configura las conexiones de señales"""
        # Conectar señales de análisis (se implementarán cuando estén las pestañas)
        pass
        
    def on_tab_changed(self, index: int):
        """Maneja el cambio de pestaña"""
        tab_names = ["Análisis", "Comparación", "Base de Datos", "Reportes"]
        if 0 <= index < len(tab_names):
            self.status_bar.showMessage(f"Pestaña activa: {tab_names[index]}")
            
    def update_system_status(self):
        """Actualiza el estado del sistema"""
        try:
            # Aquí se podría verificar el estado real del sistema
            # Por ahora, simulamos información básica
            import psutil
            memory_percent = psutil.virtual_memory().percent
            
            if memory_percent < 80:
                self.performance_label.setText(f"Memoria: {memory_percent:.1f}% OK")
                self.performance_label.setProperty("class", "caption success")
            else:
                self.performance_label.setText(f"Memoria: {memory_percent:.1f}% Alta")
                self.performance_label.setProperty("class", "caption warning")
                
            # Forzar actualización de estilo
            self.performance_label.style().unpolish(self.performance_label)
            self.performance_label.style().polish(self.performance_label)
            
        except ImportError:
            # Si psutil no está disponible
            self.performance_label.setText("Memoria: N/A")
            
    def new_analysis(self):
        """Inicia un nuevo análisis"""
        self.tab_widget.setCurrentIndex(0)  # Ir a pestaña de análisis
        self.status_bar.showMessage("Nuevo análisis iniciado")
        
    def open_project(self):
        """Abre un proyecto existente"""
        # Placeholder - implementar cuando esté el backend
        QMessageBox.information(self, "Información", "Funcionalidad en desarrollo")
        
    def save_project(self):
        """Guarda el proyecto actual"""
        # Placeholder - implementar cuando esté el backend
        QMessageBox.information(self, "Información", "Funcionalidad en desarrollo")
        
    def show_configuration(self):
        """Muestra el diálogo de configuración"""
        # Placeholder - implementar cuando esté el diálogo
        QMessageBox.information(self, "Información", "Configuración en desarrollo")
        
    def show_database_management(self):
        """Muestra la gestión de base de datos"""
        self.tab_widget.setCurrentIndex(2)  # Ir a pestaña de base de datos
        
    def backup_database(self):
        """Realiza respaldo de la base de datos"""
        try:
            from PyQt5.QtWidgets import QFileDialog, QMessageBox, QProgressDialog
            from PyQt5.QtCore import QThread, pyqtSignal
            import shutil
            import os
            from datetime import datetime
            
            # Seleccionar ubicación de respaldo
            backup_path, _ = QFileDialog.getSaveFileName(
                self,
                "Guardar Respaldo de Base de Datos",
                f"SIGeC-Balistica_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db",
                "Base de Datos (*.db);;Todos los archivos (*)"
            )
            
            if backup_path:
                # Crear diálogo de progreso
                progress = QProgressDialog("Creando respaldo...", "Cancelar", 0, 0, self)
                progress.setWindowModality(Qt.WindowModal)
                progress.show()
                
                try:
                    # Obtener ruta de la base de datos actual
                    db_path = self.backend.get_database_path()
                    
                    # Copiar archivo
                    shutil.copy2(db_path, backup_path)
                    
                    progress.close()
                    QMessageBox.information(
                        self,
                        "Éxito",
                        f"Respaldo creado exitosamente:\n{backup_path}"
                    )
                    
                except Exception as e:
                    progress.close()
                    QMessageBox.critical(
                        self,
                        "Error",
                        f"Error creando respaldo:\n{str(e)}"
                    )
                    
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"Error en respaldo de base de datos:\n{str(e)}"
            )
    
    def show_history(self):
        """Muestra el historial de análisis"""
        try:
            dialog = HistoryDialog(self)
            dialog.exec_()
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"Error mostrando historial:\n{str(e)}"
            )
    
    def show_preferences(self):
        """Muestra el diálogo de configuración"""
        try:
            dialog = SettingsDialog(self)
            if dialog.exec_() == QDialog.Accepted:
                # Aplicar configuraciones si fueron aceptadas
                self.apply_new_settings()
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"Error mostrando configuración:\n{str(e)}"
            )
    
    def show_user_guide(self):
        """Muestra la guía de usuario"""
        try:
            dialog = HelpDialog(self)
            dialog.tab_widget.setCurrentIndex(0)  # Pestaña de guía de usuario
            dialog.show()
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"Error mostrando guía de usuario:\n{str(e)}"
            )
    
    def show_help(self):
        """Muestra la ayuda completa"""
        try:
            dialog = HelpDialog(self)
            dialog.show()
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"Error mostrando ayuda:\n{str(e)}"
            )
    
    def show_support(self):
        """Muestra el soporte técnico"""
        try:
            dialog = HelpDialog(self)
            dialog.tab_widget.setCurrentIndex(4)  # Pestaña de soporte técnico
            dialog.show()
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"Error mostrando soporte técnico:\n{str(e)}"
            )
    
    def show_about(self):
        """Muestra información acerca de la aplicación"""
        try:
            dialog = AboutDialog(self)
            dialog.exec_()
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"Error mostrando información:\n{str(e)}"
            )
    
    def apply_new_settings(self):
        """Aplica nuevas configuraciones"""
        try:
            # Recargar configuración del backend
            self.backend.reload_configuration()
            
            # Actualizar tema si cambió
            apply_SIGeC-Balistica_theme(self)
            
            # Notificar a las pestañas sobre cambios de configuración
            for i in range(self.tab_widget.count()):
                tab = self.tab_widget.widget(i)
                if hasattr(tab, 'on_settings_changed'):
                    tab.on_settings_changed()
            
            self.status_bar.showMessage("Configuración aplicada exitosamente", 3000)
            
        except Exception as e:
            QMessageBox.warning(
                self,
                "Advertencia",
                f"Algunas configuraciones no pudieron aplicarse:\n{str(e)}"
            )
        
    def set_analysis_progress(self, progress: int, message: str = ""):
        """Actualiza el progreso de análisis"""
        if progress >= 0:
            self.progress_label.setText(f"Progreso: {progress}%")
            if message:
                self.status_bar.showMessage(message)
        else:
            self.progress_label.setText("Inactivo")
            
    def show_error(self, title: str, message: str):
        """Muestra un mensaje de error"""
        QMessageBox.critical(self, title, message)
        
    def show_warning(self, title: str, message: str):
        """Muestra un mensaje de advertencia"""
        QMessageBox.warning(self, title, message)
        
    def show_info(self, title: str, message: str):
        """Muestra un mensaje informativo"""
        QMessageBox.information(self, title, message)
        
    def closeEvent(self, event):
        """Maneja el evento de cierre de la aplicación"""
        reply = QMessageBox.question(
            self,
            "Confirmar Salida",
            "¿Está seguro de que desea salir?\n\nSe perderán los cambios no guardados.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # Limpiar recursos si es necesario
            if hasattr(self, 'status_timer'):
                self.status_timer.stop()
            event.accept()
        else:
            event.ignore()

def main():
    """Función principal para ejecutar la aplicación"""
    app = QApplication(sys.argv)
    
    # Aplicar tema
    apply_SIGeC-Balistica_theme(app)
    
    # Crear y mostrar ventana principal
    window = MainWindow()
    window.show()
    
    return app.exec_()

if __name__ == "__main__":
    sys.exit(main())