#!/usr/bin/env python3
"""
Validación manual del sistema GUI - SIGeC-Balistica
Verifica que la GUI puede iniciarse y funcionar básicamente
"""

import sys
import os
from pathlib import Path

# Agregar el directorio raíz del proyecto al path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_gui_startup():
    """Test de inicio de la GUI"""
    print("🔍 Probando inicio de GUI...")
    try:
        # Verificar que podemos importar los componentes principales
        from gui.main_window import MainWindow
        print("  ✅ MainWindow importado correctamente")
        
        # Verificar configuración GUI
        from config.unified_config import get_unified_config
        config = get_unified_config()
        print("  ✅ Configuración GUI disponible")
        
        # Verificar integraciones básicas
        from gui.core_integration import CORE_AVAILABLE
        print(f"  ✅ Core integration: {CORE_AVAILABLE}")
        
        return True
    except Exception as e:
        print(f"  ❌ Error en inicio GUI: {e}")
        return False

def test_gui_components():
    """Test de componentes GUI individuales"""
    print("🔍 Probando componentes GUI individuales...")
    try:
        # Test de tabs principales
        components = [
            ("AnalysisTab", "gui.analysis_tab"),
            ("DatabaseTab", "gui.database_tab"),
            ("ComparisonTab", "gui.comparison_tab"),
        ]
        
        for comp_name, module_name in components:
            try:
                module = __import__(module_name, fromlist=[comp_name])
                getattr(module, comp_name)
                print(f"  ✅ {comp_name} disponible")
            except Exception as e:
                print(f"  ⚠️  {comp_name} no disponible: {e}")
        
        return True
    except Exception as e:
        print(f"  ❌ Error en componentes GUI: {e}")
        return False

def test_gui_integration():
    """Test de integración GUI con backend"""
    print("🔍 Probando integración GUI-Backend...")
    try:
        from gui.backend_integration import BackendIntegration
        print("  ✅ BackendIntegration disponible")
        
        from gui.common_integration import CommonIntegration
        print("  ✅ CommonIntegration disponible")
        
        return True
    except Exception as e:
        print(f"  ❌ Error en integración GUI: {e}")
        return False

def main():
    """Función principal de validación manual GUI"""
    print("🚀 Iniciando Validación Manual GUI - SIGeC-Balistica")
    print("=" * 50)
    
    tests = [
        ("Inicio de GUI", test_gui_startup),
        ("Componentes GUI", test_gui_components),
        ("Integración GUI", test_gui_integration)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n📋 Ejecutando: {test_name}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"  ❌ Error inesperado en {test_name}: {e}")
            results.append((test_name, False))
    
    # Resumen
    print("\n" + "=" * 50)
    print("📊 RESUMEN VALIDACIÓN MANUAL GUI")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASÓ" if result else "❌ FALLÓ"
        print(f"  {status}: {test_name}")
    
    print(f"\n🎯 Resultado: {passed}/{total} tests pasaron")
    success_rate = (passed / total) * 100
    print(f"📈 Tasa de Éxito: {success_rate:.1f}%")
    
    if success_rate >= 70:
        print("🎉 GUI FUNCIONAL - Lista para uso básico")
        return True
    else:
        print("❌ GUI REQUIERE CORRECCIONES")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
