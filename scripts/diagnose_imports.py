#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de Diagnóstico de Importaciones
=====================================

Verifica qué módulos pueden importarse correctamente y cuáles tienen problemas.
"""

import sys
import traceback
from pathlib import Path

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_import(module_name, description=""):
    """Prueba importar un módulo y reporta el resultado"""
    try:
        __import__(module_name)
        print(f"✅ {module_name} - {description}")
        return True
    except Exception as e:
        print(f"❌ {module_name} - {description}")
        print(f"   Error: {str(e)}")
        if "--verbose" in sys.argv:
            print(f"   Traceback: {traceback.format_exc()}")
        return False

def main():
    """Ejecuta el diagnóstico de importaciones"""
    print("🔍 Diagnóstico de Importaciones - SIGeC-Balisticar")
    print("=" * 50)
    
    # Módulos principales a verificar
    modules_to_test = [
        # Core modules
        ("core.unified_pipeline", "Pipeline unificado"),
        ("core.performance_monitor", "Monitor de rendimiento"),
        ("core.pipeline_config", "Configuración del pipeline"),
        
        # Config modules
        ("config.unified_config", "Configuración unificada"),
        
        # Database modules
        ("database", "Módulo de base de datos"),
        ("database.unified_database", "Base de datos unificada"),
        ("database.vector_db", "Base de datos vectorial"),
        
        # Utils modules
        ("utils.dependency_manager", "Gestor de dependencias"),
        ("utils.logger", "Sistema de logging"),
        
        # Image processing modules
        ("image_processing.optimized_loader", "Cargador optimizado"),
        ("image_processing.unified_preprocessor", "Preprocesador unificado"),
        ("image_processing.unified_roi_detector", "Detector de ROI"),
        
        # Matching modules
        ("matching", "Módulo de matching"),
        ("matching.unified_matcher", "Matcher unificado"),
        ("matching.cmc_algorithm", "Algoritmo CMC"),
        
        # NIST standards
        ("nist_standards.quality_metrics", "Métricas de calidad NIST"),
        
        # Deep learning (opcional)
        ("deep_learning.ballistic_dl_models", "Modelos de Deep Learning"),
        
        # Security
        ("security.security_manager", "Gestor de seguridad"),
        
        # Common test helpers
        ("common.test_helpers", "Helpers de testing"),
    ]
    
    successful_imports = 0
    failed_imports = 0
    
    for module_name, description in modules_to_test:
        if test_import(module_name, description):
            successful_imports += 1
        else:
            failed_imports += 1
        print()  # Línea en blanco para separar
    
    # Resumen
    print("=" * 50)
    print(f"📊 Resumen del Diagnóstico:")
    print(f"   ✅ Importaciones exitosas: {successful_imports}")
    print(f"   ❌ Importaciones fallidas: {failed_imports}")
    print(f"   📈 Tasa de éxito: {successful_imports/(successful_imports+failed_imports)*100:.1f}%")
    
    if failed_imports > 0:
        print("\n⚠️  Hay módulos con problemas de importación.")
        print("   Ejecuta con --verbose para ver detalles completos.")
        return 1
    else:
        print("\n🎉 Todos los módulos se importaron correctamente!")
        return 0

if __name__ == "__main__":
    sys.exit(main())