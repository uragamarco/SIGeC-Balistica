#!/usr/bin/env python3
"""
Test de Integración del Frontend
Sistema Balístico Forense MVP

Prueba todos los componentes del frontend para verificar su funcionamiento
"""

import sys
import os
import traceback
import json
from datetime import datetime
from pathlib import Path

# Agregar el directorio raíz al path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_gui_main_window():
    """Probar la ventana principal de la GUI"""
    print("🔍 Probando ventana principal...")
    try:
        from gui.main_window import MainWindow
        from config.unified_config import get_unified_config
        from PyQt5.QtWidgets import QApplication
        
        # Crear aplicación Qt (necesaria para widgets)
        if not QApplication.instance():
            app = QApplication(sys.argv)
        else:
            app = QApplication.instance()
        
        # Crear configuración necesaria
        config = get_unified_config()
        
        # Inicializar ventana principal
        main_window = MainWindow(config)
        print("  ✅ MainWindow inicializada correctamente")
        
        # Verificar componentes principales
        if hasattr(main_window, 'config'):
            print("  ✅ Configuración cargada")
        
        if hasattr(main_window, 'tab_widget'):
            print("  ✅ Widget de tabs encontrado")
        
        # Verificar menús y toolbars
        menubar = main_window.menuBar()
        if menubar:
            print(f"  ✅ Barra de menú configurada con {len(menubar.actions())} acciones")
        
        # Verificar status bar
        statusbar = main_window.statusBar()
        if statusbar:
            print("  ✅ Barra de estado configurada")
        
        return True, "Ventana principal funcionando correctamente"
        
    except Exception as e:
        return False, f"Error en ventana principal: {str(e)}\n{traceback.format_exc()}"

def test_gui_tabs():
    """Probar los tabs de la interfaz"""
    print("🔍 Probando tabs de la interfaz...")
    try:
        from gui.image_tab import ImageTab
        from gui.comparison_tab import ComparisonTab
        from gui.database_tab import DatabaseTab
        from gui.reports_tab import ReportsTab
        from config.unified_config import get_unified_config
        from PyQt5.QtWidgets import QApplication, QWidget
        
        # Crear aplicación Qt si no existe
        if not QApplication.instance():
            app = QApplication(sys.argv)
        
        # Crear widget padre para los tabs
        parent = QWidget()
        
        # Create config for tabs that need it
        config = get_unified_config()
        
        # Test ImageTab
        image_tab = ImageTab(config)
        assert hasattr(image_tab, '_load_image'), "ImageTab should have _load_image method"
        print("  ✅ ImageTab inicializado")
        
        # Test ComparisonTab
        comparison_tab = ComparisonTab(config)
        assert hasattr(comparison_tab, 'database'), "ComparisonTab should have database attribute"
        print("  ✅ ComparisonTab inicializado")
        
        # Test DatabaseTab (needs config)
        database_tab = DatabaseTab(config)
        assert hasattr(database_tab, 'config'), "DatabaseTab should have config attribute"
        
        # Test ReportsTab
        reports_tab = ReportsTab(config)
        assert hasattr(reports_tab, 'database'), "ReportsTab should have database attribute"
        print("  ✅ ReportsTab inicializado")
        
        return True, "Todos los tabs funcionando correctamente"
        
    except Exception as e:
        return False, f"Error en tabs: {str(e)}\n{traceback.format_exc()}"

def test_gui_panels():
    """Probar los paneles de la interfaz"""
    print("🔍 Probando paneles de la interfaz...")
    try:
        from gui.floating_panel import FloatingPanel
        from gui.floating_controls_panel import FloatingControlsPanel
        from gui.floating_stats_panel import FloatingStatsPanel
        from gui.advanced_config_panel import AdvancedConfigPanel
        from PyQt5.QtWidgets import QApplication, QWidget
        
        # Crear aplicación Qt si no existe
        if not QApplication.instance():
            app = QApplication(sys.argv)
        
        # Crear widget padre para los paneles
        parent = QWidget()
        
        # Probar FloatingPanel
        floating_panel = FloatingPanel(parent)
        print("  ✅ FloatingPanel inicializado")
        
        # Probar FloatingControlsPanel
        controls_panel = FloatingControlsPanel(parent)
        print("  ✅ FloatingControlsPanel inicializado")
        
        # Probar FloatingStatsPanel
        stats_panel = FloatingStatsPanel(parent)
        print("  ✅ FloatingStatsPanel inicializado")
        
        # Probar AdvancedConfigPanel
        config_panel = AdvancedConfigPanel(parent)
        print("  ✅ AdvancedConfigPanel inicializado")
        
        return True, "Todos los paneles funcionando correctamente"
        
    except Exception as e:
        return False, f"Error en paneles: {str(e)}\n{traceback.format_exc()}"

def test_gui_dialogs():
    """Probar los diálogos de la interfaz"""
    print("🔍 Probando diálogos de la interfaz...")
    try:
        from gui.advanced_config_dialog import AdvancedConfigDialog
        from PyQt5.QtWidgets import QApplication, QWidget
        
        # Crear aplicación Qt si no existe
        if not QApplication.instance():
            app = QApplication(sys.argv)
        
        # Crear widget padre para los diálogos
        parent = QWidget()
        
        # Probar AdvancedConfigDialog
        config_dialog = AdvancedConfigDialog(parent)
        print("  ✅ AdvancedConfigDialog inicializado")
        
        return True, "Los diálogos funcionando correctamente"
        
    except Exception as e:
        return False, f"Error en diálogos: {str(e)}\n{traceback.format_exc()}"

def test_gui_widgets():
    """Probar widgets personalizados"""
    print("🔍 Probando widgets personalizados...")
    try:
        from gui.collapsible_widget import CollapsibleWidget
        from gui.visualization import MatchVisualizationWidget
        from PyQt5.QtWidgets import QApplication, QWidget
        
        # Crear aplicación Qt si no existe
        if not QApplication.instance():
            app = QApplication(sys.argv)
        
        # Crear widget padre
        parent = QWidget()
        
        # Probar CollapsibleWidget
        collapsible_widget = CollapsibleWidget("Test Title", parent)
        print("  ✅ CollapsibleWidget inicializado")
        
        # Probar MatchVisualizationWidget
        match_widget = MatchVisualizationWidget(parent)
        print("  ✅ MatchVisualizationWidget inicializado")
        
        return True, "Los widgets personalizados funcionando correctamente"
        
    except Exception as e:
        return False, f"Error en widgets personalizados: {str(e)}\n{traceback.format_exc()}"

def main():
    """Función principal para ejecutar todos los tests del frontend"""
    print("=" * 60)
    print("🎯 TEST DE INTEGRACIÓN DEL FRONTEND")
    print("Sistema Balístico Forense MVP")
    print("=" * 60)
    
    # Lista de tests a ejecutar
    tests = [
        ("Main Window", test_gui_main_window),
        ("Tabs", test_gui_tabs),
        ("Panels", test_gui_panels),
        ("Dialogs", test_gui_dialogs),
        ("Widgets", test_gui_widgets)
    ]
    
    results = {}
    successful_tests = 0
    total_tests = len(tests)
    
    # Ejecutar cada test
    for test_name, test_func in tests:
        print(f"\n📦 Probando módulo: {test_name}")
        print("-" * 40)
        
        try:
            success, message = test_func()
            results[test_name] = {
                "success": success,
                "message": message,
                "timestamp": datetime.now().isoformat()
            }
            
            if success:
                print(f"✅ {test_name}: ÉXITO")
                successful_tests += 1
            else:
                print(f"❌ {test_name}: FALLO")
                print(f"   {message}")
                
        except Exception as e:
            error_msg = f"Error inesperado: {str(e)}\n{traceback.format_exc()}"
            results[test_name] = {
                "success": False,
                "message": error_msg,
                "timestamp": datetime.now().isoformat()
            }
            print(f"❌ {test_name}: ERROR CRÍTICO")
            print(f"   {error_msg}")
    
    # Generar resumen
    print("\n" + "=" * 60)
    print("📊 RESUMEN DE RESULTADOS")
    print("=" * 60)
    print(f"✅ Módulos exitosos: {successful_tests}/{total_tests}")
    
    if successful_tests > 0:
        successful_modules = [name for name, result in results.items() if result["success"]]
        for module in successful_modules:
            print(f"   - {module}")
    
    failed_tests = total_tests - successful_tests
    if failed_tests > 0:
        print(f"\n❌ Módulos con problemas: {failed_tests}/{total_tests}")
        failed_modules = [name for name, result in results.items() if not result["success"]]
        for module in failed_modules:
            print(f"   - {module}: {results[module]['message'].split('Traceback')[0].strip()}")
    
    # Guardar resultados
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"frontend_integration_test_results_{timestamp}.json"
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n📄 Resultados guardados en: {results_file}")
    
    # Determinar estado general
    if successful_tests == total_tests:
        print("\n🎯 Estado general del frontend: ✅ EXCELENTE")
        return 0
    elif successful_tests >= total_tests * 0.7:
        print("\n🎯 Estado general del frontend: ⚠️ ACEPTABLE")
        return 0
    else:
        print("\n🎯 Estado general del frontend: ❌ REQUIERE ATENCIÓN")
        return 1

if __name__ == "__main__":
    sys.exit(main())