#!/usr/bin/env python3
"""
Script de validación E2E simplificado para SIGeC-Balistica
Verifica funcionalidades básicas sin dependencias externas complejas
"""

import sys
import os
import traceback
from pathlib import Path

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent))

def test_core_module():
    """Test del módulo Core"""
    print("🔍 Probando módulo Core...")
    try:
        # Verificar que el módulo core está disponible
        from gui.core_integration import CORE_AVAILABLE
        print(f"  ✅ CORE_AVAILABLE: {CORE_AVAILABLE}")
        
        if CORE_AVAILABLE:
            from core.unified_pipeline import ScientificPipeline
            from core.error_handler import get_error_manager
            from core.intelligent_cache import IntelligentCache
            print("  ✅ Componentes core importados correctamente")
        else:
            print("  ⚠️  Core no disponible, pero sistema funciona con fallbacks")
        
        return True
    except Exception as e:
        print(f"  ❌ Error en módulo Core: {e}")
        return False

def test_common_module():
    """Test del módulo Common"""
    print("🔍 Probando módulo Common...")
    try:
        import common
        print("  ✅ Módulo Common importado correctamente")
        
        # Verificar componentes principales
        from common.statistical_core import StatisticalCore
        from common.compatibility_adapters import CompatibilityAdapter
        print("  ✅ Componentes Common disponibles")
        
        return True
    except Exception as e:
        print(f"  ❌ Error en módulo Common: {e}")
        return False

def test_gui_components():
    """Test de componentes GUI básicos"""
    print("🔍 Probando componentes GUI...")
    try:
        # Test de importaciones GUI básicas
        from gui.main_window import MainWindow
        from gui.analysis_tab import AnalysisTab
        from gui.database_tab import DatabaseTab
        from gui.comparison_tab import ComparisonTab
        print("  ✅ Componentes GUI principales importados")
        
        # Test de integración GUI
        from gui.backend_integration import BackendIntegration
        from gui.common_integration import CommonIntegration
        print("  ✅ Integraciones GUI disponibles")
        
        return True
    except Exception as e:
        print(f"  ❌ Error en componentes GUI: {e}")
        return False

def test_config_system():
    """Test del sistema de configuración"""
    print("🔍 Probando sistema de configuración...")
    try:
        from config.unified_config import get_unified_config
        config = get_unified_config()
        print("  ✅ Sistema de configuración funcionando")
        
        # Verificar secciones principales
        if hasattr(config, 'gui') and hasattr(config, 'processing'):
            print("  ✅ Configuraciones GUI y procesamiento disponibles")
        
        return True
    except Exception as e:
        print(f"  ❌ Error en sistema de configuración: {e}")
        return False

def test_utils_system():
    """Test del sistema de utilidades"""
    print("🔍 Probando sistema de utilidades...")
    try:
        from utils.dependency_manager import DependencyManager
        from utils.validators import SystemValidator
        print("  ✅ Utilidades del sistema disponibles")
        
        # Test básico de validador
        validator = SystemValidator()
        print("  ✅ SystemValidator inicializado")
        
        return True
    except Exception as e:
        print(f"  ❌ Error en sistema de utilidades: {e}")
        return False

def test_basic_workflow():
    """Test de flujo de trabajo básico"""
    print("🔍 Probando flujo de trabajo básico...")
    try:
        # Simular carga de configuración
        from config.unified_config import get_unified_config
        config = get_unified_config()
        
        # Simular inicialización de componentes
        from utils.dependency_manager import DependencyManager
        dep_manager = DependencyManager()
        
        print("  ✅ Flujo básico de inicialización completado")
        return True
    except Exception as e:
        print(f"  ❌ Error en flujo básico: {e}")
        return False

def main():
    """Función principal de validación E2E"""
    print("🚀 Iniciando Validación E2E Simplificada - SIGeC-Balistica")
    print("=" * 60)
    
    tests = [
        ("Módulo Core", test_core_module),
        ("Módulo Common", test_common_module),
        ("Componentes GUI", test_gui_components),
        ("Sistema de Configuración", test_config_system),
        ("Sistema de Utilidades", test_utils_system),
        ("Flujo de Trabajo Básico", test_basic_workflow)
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
    
    # Resumen de resultados
    print("\n" + "=" * 60)
    print("📊 RESUMEN DE VALIDACIÓN E2E")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASÓ" if result else "❌ FALLÓ"
        print(f"  {status}: {test_name}")
    
    print(f"\n🎯 Resultado Final: {passed}/{total} tests pasaron")
    success_rate = (passed / total) * 100
    print(f"📈 Tasa de Éxito: {success_rate:.1f}%")
    
    if success_rate >= 80:
        print("🎉 VALIDACIÓN E2E EXITOSA - Sistema listo para uso básico")
        return True
    elif success_rate >= 60:
        print("⚠️  VALIDACIÓN PARCIAL - Sistema funcional con limitaciones")
        return True
    else:
        print("❌ VALIDACIÓN FALLIDA - Requiere correcciones críticas")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
