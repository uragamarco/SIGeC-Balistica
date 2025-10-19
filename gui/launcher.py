#!/usr/bin/env python3
"""
Launcher para SIGeC-Balistica GUI
Este script configura el entorno de Python y lanza la aplicación principal.
"""

import sys
import os
from pathlib import Path

def setup_python_path():
    """Configura el path de Python para reconocer el paquete gui"""
    # Obtener el directorio padre del directorio gui
    current_dir = Path(__file__).parent
    parent_dir = current_dir.parent
    
    # Añadir el directorio padre al path de Python
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))
    
    # Añadir el directorio actual al path también
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))

def check_dependencies():
    """Verifica que las dependencias básicas estén disponibles"""
    try:
        import PyQt5
        print("✓ PyQt5 encontrado")
    except ImportError:
        print("✗ Error: PyQt5 no está instalado")
        print("  Instale con: pip install PyQt5")
        return False
    
    try:
        import numpy
        print("✓ NumPy encontrado")
    except ImportError:
        print("✗ Error: NumPy no está instalado")
        print("  Instale con: pip install numpy")
        return False
    
    try:
        import cv2
        print("✓ OpenCV encontrado")
    except ImportError:
        print("✗ Error: OpenCV no está instalado")
        print("  Instale con: pip install opencv-python")
        return False
    
    return True

def main():
    """Función principal del launcher"""
    print("=" * 60)
    print("SIGeC-Balistica GUI - Sistema de Análisis Forense")
    print("=" * 60)
    print()
    
    # Configurar el path de Python
    setup_python_path()
    
    # Verificar dependencias
    print("Verificando dependencias...")
    if not check_dependencies():
        print("\nError: Faltan dependencias requeridas.")
        print("Por favor, instale las dependencias faltantes y vuelva a intentar.")
        sys.exit(1)
    
    print("\n✓ Todas las dependencias están disponibles")
    print("Iniciando aplicación...")
    print()
    
    try:
        # Usar el lanzador consolidado con diagnósticos
        from launch_gui import launch_gui

        # Ejecutar la aplicación a través del lanzador unificado
        sys.exit(launch_gui())
        
    except ImportError as e:
        print(f"✗ Error de importación: {e}")
        print("\nVerifique que todos los archivos estén en su lugar:")
        print("- main_window.py")
        print("- styles.py")
        print("- app_state_manager.py")
        print("- Todos los archivos de pestañas (*_tab.py)")
        sys.exit(1)
        
    except Exception as e:
        import traceback
        print(f"✗ Error inesperado: {e}")
        traceback.print_exc() # Imprimir el traceback completo
        print("\nSi el problema persiste, contacte al soporte técnico.")
        sys.exit(1)

if __name__ == "__main__":
    main()