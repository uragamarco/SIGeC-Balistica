#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test GUI Minimal - SIGeC-Balística
==================================

Test mínimo de la GUI sin componentes WebEngine para verificar
la funcionalidad básica de la interfaz.
"""

import sys
import os
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from PyQt5.QtWidgets import QApplication, QMainWindow, QTabWidget, QVBoxLayout, QWidget, QLabel
from PyQt5.QtCore import Qt

# Import only the basic tabs without WebEngine dependencies
from gui.analysis_tab import AnalysisTab
from gui.comparison_tab import ComparisonTab
from gui.database_tab import DatabaseTab

class MinimalMainWindow(QMainWindow):
    """Ventana principal mínima para testing"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SIGeC-Balística - Test Minimal")
        self.setGeometry(100, 100, 1200, 800)
        
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Create layout
        layout = QVBoxLayout(central_widget)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)
        
        # Add tabs
        try:
            self.analysis_tab = AnalysisTab()
            self.tab_widget.addTab(self.analysis_tab, "Análisis")
            print("✓ Analysis tab loaded successfully")
        except Exception as e:
            print(f"✗ Error loading Analysis tab: {e}")
            error_tab = QWidget()
            error_layout = QVBoxLayout(error_tab)
            error_layout.addWidget(QLabel(f"Error: {str(e)}"))
            self.tab_widget.addTab(error_tab, "Análisis (Error)")
        
        try:
            self.comparison_tab = ComparisonTab()
            self.tab_widget.addTab(self.comparison_tab, "Comparación")
            print("✓ Comparison tab loaded successfully")
        except Exception as e:
            print(f"✗ Error loading Comparison tab: {e}")
            error_tab = QWidget()
            error_layout = QVBoxLayout(error_tab)
            error_layout.addWidget(QLabel(f"Error: {str(e)}"))
            self.tab_widget.addTab(error_tab, "Comparación (Error)")
        
        try:
            self.database_tab = DatabaseTab()
            self.tab_widget.addTab(self.database_tab, "Base de Datos")
            print("✓ Database tab loaded successfully")
        except Exception as e:
            print(f"✗ Error loading Database tab: {e}")
            error_tab = QWidget()
            error_layout = QVBoxLayout(error_tab)
            error_layout.addWidget(QLabel(f"Error: {str(e)}"))
            self.tab_widget.addTab(error_tab, "Base de Datos (Error)")

def main():
    """Función principal del test"""
    print("Iniciando test GUI minimal...")
    
    # Create application
    app = QApplication(sys.argv)
    app.setApplicationName("SIGeC-Balística Test")
    
    try:
        # Create and show main window
        window = MinimalMainWindow()
        window.show()
        
        print("✓ GUI iniciada correctamente")
        print("Presiona Ctrl+C para salir")
        
        # Run for a short time to test initialization
        from PyQt5.QtCore import QTimer
        timer = QTimer()
        timer.timeout.connect(lambda: (print("✓ GUI funcionando correctamente"), app.quit()))
        timer.start(3000)  # Exit after 3 seconds
        
        return app.exec_()
        
    except Exception as e:
        print(f"✗ Error en la GUI: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())