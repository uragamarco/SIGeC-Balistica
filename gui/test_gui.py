#!/usr/bin/env python3
"""
Test script para verificar la funcionalidad básica del GUI después de la reversión
"""

import sys
import os

# Agregar el directorio raíz al path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
from gui.main_window import MainWindow

def main():
    """Función principal para probar la GUI"""
    # Configurar Qt antes de crear la aplicación
    QApplication.setAttribute(Qt.AA_ShareOpenGLContexts, True)
    
    app = QApplication(sys.argv)
    
    # Crear y mostrar la ventana principal
    window = MainWindow()
    window.show()
    
    print("GUI iniciada correctamente - versión básica sin mejoras UX/UI")
    print("Sistema listo para análisis balístico")
    
    return app.exec_()

if __name__ == "__main__":
    sys.exit(main())