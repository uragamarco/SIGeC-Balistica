#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lanzador específico para la interfaz gráfica de SIGeC-Balisticar
Fuerza el modo GUI y proporciona diagnóstico detallado de errores
"""

import sys
import os
import traceback
from pathlib import Path

# Configurar el entorno para GUI
os.environ['QT_QPA_PLATFORM'] = 'xcb'  # Forzar uso de X11
if 'DISPLAY' not in os.environ:
    os.environ['DISPLAY'] = ':0'

# Agregar el directorio del proyecto al path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def check_gui_requirements():
    """Verifica que todos los requisitos para GUI estén disponibles"""
    print("🔍 Verificando requisitos para GUI...")
    
    # Verificar PyQt5
    try:
        import PyQt5
        from PyQt5.QtWidgets import QApplication
        from PyQt5.QtCore import Qt
        print("✅ PyQt5 disponible")
    except ImportError as e:
        print(f"❌ PyQt5 no disponible: {e}")
        return False
    
    # Verificar DISPLAY
    display = os.environ.get('DISPLAY')
    if not display:
        print("❌ Variable DISPLAY no configurada")
        return False
    print(f"✅ DISPLAY configurado: {display}")
    
    # Intentar crear aplicación Qt
    try:
        app = QApplication([])
        print("✅ QApplication se puede crear")
        app.quit()
        return True
    except Exception as e:
        print(f"❌ Error creando QApplication: {e}")
        return False

def launch_gui():
    """Lanza la interfaz gráfica"""
    print("🚀 Iniciando SIGeC-Balisticar GUI...")
    
    try:
        # Importar módulos necesarios
        from PyQt5.QtWidgets import QApplication
        from PyQt5.QtCore import Qt
        from gui.main_window import MainWindow
        from gui.styles import apply_SIGeC_Balistica_theme
        
        # Configurar atributos Qt
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
        QApplication.setAttribute(Qt.AA_ShareOpenGLContexts, True)
        
        # Crear aplicación
        app = QApplication(sys.argv)
        app.setApplicationName("SIGeC-Balistica")
        app.setApplicationVersion("2.0.0")
        
        print("✅ Aplicación Qt creada")
        
        # Aplicar tema
        apply_SIGeC_Balistica_theme(app)
        print("✅ Tema aplicado")
        
        # Crear ventana principal
        window = MainWindow()
        print("✅ Ventana principal creada")
        
        # Mostrar ventana
        window.show()
        print("✅ Ventana mostrada")
        
        # Centrar ventana
        screen = app.primaryScreen().geometry()
        window_size = window.geometry()
        x = (screen.width() - window_size.width()) // 2
        y = (screen.height() - window_size.height()) // 2
        window.move(x, y)
        
        print("🎉 SIGeC-Balisticar GUI iniciado exitosamente!")
        print("📋 Funcionalidades disponibles:")
        print("   • Análisis de imágenes balísticas")
        print("   • Comparación de características")
        print("   • Gestión de base de datos")
        print("   • Generación de reportes")
        print("   • Configuración del sistema")
        
        # Ejecutar aplicación
        return app.exec_()
        
    except Exception as e:
        print(f"❌ Error lanzando GUI: {e}")
        print("📋 Traceback completo:")
        traceback.print_exc()
        return 1

def main():
    """Función principal"""
    print("=" * 60)
    print("🔬 SIGeC-Balisticar - Sistema Balístico Forense")
    print("   Interfaz Gráfica de Usuario")
    print("=" * 60)
    
    # Verificar requisitos
    if not check_gui_requirements():
        print("\n❌ Los requisitos para GUI no están disponibles.")
        print("💡 Soluciones posibles:")
        print("   1. Instalar PyQt5: pip install PyQt5")
        print("   2. Configurar DISPLAY: export DISPLAY=:0")
        print("   3. Instalar servidor X: sudo apt-get install xorg")
        print("   4. Para entornos remotos: usar X11 forwarding (ssh -X)")
        return 1
    
    # Lanzar GUI
    return launch_gui()

if __name__ == "__main__":
    sys.exit(main())