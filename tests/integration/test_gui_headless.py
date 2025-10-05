#!/usr/bin/env python3
"""
Test GUI Headless - Sistema Balístico Forense MVP
Prueba la interfaz gráfica en modo headless (sin pantalla)
"""

import sys
import os
import tempfile
from pathlib import Path

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent))

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtTest import QTest

from gui.main_window import MainWindow
from config.unified_config import get_unified_config
from utils.logger import setup_logging

def test_gui_initialization():
    """Prueba la inicialización de la GUI"""
    print("🔄 Probando inicialización de la GUI...")
    
    try:
        # Configurar logging
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = os.path.join(temp_dir, "test_gui.log")
            setup_logging(log_file=log_file, console_output=False)
            
            # Cargar configuración
            config = get_unified_config()
            
            # Crear aplicación Qt
            app = QApplication(sys.argv)
            app.setApplicationName("Test Sistema Balístico")
            app.setApplicationVersion("1.0.0")
            
            # Crear ventana principal
            main_window = MainWindow(config)
            
            # Verificar que la ventana se creó correctamente
            assert main_window is not None, "La ventana principal no se creó"
            assert main_window.windowTitle() == "Sistema Balístico Forense - MVP v1.0.0", "Título incorrecto"
            
            # Verificar que tiene las pestañas esperadas
            assert main_window.tab_widget.count() == 4, f"Número incorrecto de pestañas: {main_window.tab_widget.count()}"
            
            # Verificar nombres de pestañas
            expected_tabs = ["Cargar Imágenes", "Base de Datos", "Comparación", "Reportes"]
            for i, expected_name in enumerate(expected_tabs):
                actual_name = main_window.tab_widget.tabText(i)
                assert actual_name == expected_name, f"Pestaña {i}: esperado '{expected_name}', obtenido '{actual_name}'"
            
            # Verificar menú
            menubar = main_window.menuBar()
            assert menubar is not None, "Barra de menú no encontrada"
            
            # Verificar barra de estado
            statusbar = main_window.statusBar()
            assert statusbar is not None, "Barra de estado no encontrada"
            
            print("✅ Inicialización de GUI exitosa")
            return True
            
    except Exception as e:
        print(f"❌ Error en inicialización de GUI: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_gui_menu_actions():
    """Prueba las acciones del menú"""
    print("🔄 Probando acciones del menú...")
    
    try:
        # Crear aplicación Qt
        app = QApplication(sys.argv)
        config = get_unified_config()
        main_window = MainWindow(config)
        
        # Verificar que el menú existe y tiene las acciones esperadas
        menubar = main_window.menuBar()
        
        # Verificar menú Archivo
        file_menu = None
        for action in menubar.actions():
            if action.text() == "&Archivo":
                file_menu = action.menu()
                break
        
        assert file_menu is not None, "Menú Archivo no encontrado"
        
        # Verificar menú Herramientas
        tools_menu = None
        for action in menubar.actions():
            if action.text() == "&Herramientas":
                tools_menu = action.menu()
                break
        
        assert tools_menu is not None, "Menú Herramientas no encontrado"
        
        # Verificar menú Ayuda
        help_menu = None
        for action in menubar.actions():
            if action.text() == "&Ayuda":
                help_menu = action.menu()
                break
        
        assert help_menu is not None, "Menú Ayuda no encontrado"
        
        print("✅ Estructura del menú verificada correctamente")
        return True
        
    except Exception as e:
        print(f"❌ Error en verificación del menú: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_gui_status_and_progress():
    """Prueba la barra de estado y progreso"""
    print("🔄 Probando barra de estado y progreso...")
    
    try:
        # Crear aplicación Qt
        app = QApplication(sys.argv)
        config = get_unified_config()
        main_window = MainWindow(config)
        
        # Probar mensaje de estado
        main_window.show_status_message("Mensaje de prueba", 1000)
        
        # Probar barra de progreso
        main_window.show_progress(True)
        main_window.set_progress(50)
        main_window.set_progress(100)
        main_window.show_progress(False)
        
        print("✅ Barra de estado y progreso funcionan correctamente")
        return True
        
    except Exception as e:
        print(f"❌ Error en barra de estado y progreso: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Función principal de prueba"""
    print("🚀 Iniciando pruebas de GUI en modo headless...")
    print("=" * 60)
    
    # Configurar Qt para modo headless
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'
    
    tests = [
        test_gui_initialization,
        test_gui_menu_actions,
        test_gui_status_and_progress
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print("-" * 40)
    
    print(f"📊 Resultados: {passed}/{total} pruebas pasaron")
    
    if passed == total:
        print("🎉 ¡Todas las pruebas de GUI pasaron exitosamente!")
        return 0
    else:
        print("⚠️  Algunas pruebas de GUI fallaron")
        return 1

if __name__ == "__main__":
    sys.exit(main())