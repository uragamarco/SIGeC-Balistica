#!/usr/bin/env python3
"""
Script para verificar que todos los servicios estén disponibles en la GUI
"""

import sys
import os
from pathlib import Path

# Añadir el directorio del proyecto al path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def verify_gui_services():
    """Verifica que todos los servicios estén disponibles en la GUI"""
    
    print("=" * 60)
    print("VERIFICACIÓN DE SERVICIOS EN GUI - SIGeC-Balisticar")
    print("=" * 60)
    
    services_status = {}
    
    try:
        # 1. Verificar configuración unificada
        print("\n1. Verificando configuración unificada...")
        from config.unified_config import UnifiedConfig
        config = UnifiedConfig()
        services_status['unified_config'] = "✓ OK"
        print("   ✓ Configuración unificada cargada")
        
        # 2. Verificar pipeline científico
        print("\n2. Verificando pipeline científico...")
        from core.unified_pipeline import ScientificPipeline
        pipeline = ScientificPipeline()
        services_status['scientific_pipeline'] = "✓ OK"
        print("   ✓ Pipeline científico inicializado")
        
        # 3. Verificar base de datos
        print("\n3. Verificando base de datos...")
        from database.unified_database import UnifiedDatabase
        db = UnifiedDatabase()
        services_status['database'] = "✓ OK"
        print("   ✓ Base de datos unificada inicializada")
        
        # 4. Verificar procesamiento de imágenes
        print("\n4. Verificando procesamiento de imágenes...")
        from image_processing.unified_preprocessor import UnifiedPreprocessor
        from image_processing.unified_roi_detector import UnifiedROIDetector
        preprocessor = UnifiedPreprocessor()
        roi_detector = UnifiedROIDetector()
        services_status['image_processing'] = "✓ OK"
        print("   ✓ Preprocesador unificado inicializado")
        print("   ✓ Detector de ROI unificado inicializado")
        
        # 5. Verificar sistema de matching
        print("\n5. Verificando sistema de matching...")
        from matching.unified_matcher import UnifiedMatcher
        matcher = UnifiedMatcher()
        services_status['matching'] = "✓ OK"
        print("   ✓ Matcher unificado inicializado")
        
        # 6. Verificar estándares NIST
        print("\n6. Verificando estándares NIST...")
        from nist_standards.quality_metrics import NISTQualityMetrics
        from nist_standards.afte_conclusions import AFTEConclusionEngine
        nist_metrics = NISTQualityMetrics()
        afte_engine = AFTEConclusionEngine()
        services_status['nist_standards'] = "✓ OK"
        print("   ✓ Métricas de calidad NIST inicializadas")
        print("   ✓ Motor de conclusiones AFTE inicializado")
        
        # 7. Verificar componentes GUI
        print("\n7. Verificando componentes GUI...")
        
        # Configurar Qt para modo headless para verificación
        os.environ['QT_QPA_PLATFORM'] = 'offscreen'
        
        from PyQt5.QtWidgets import QApplication
        from gui.main_window import MainWindow
        
        # Crear aplicación temporal
        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        
        # Crear ventana principal
        window = MainWindow()
        services_status['gui_components'] = "✓ OK"
        print("   ✓ Ventana principal creada")
        print("   ✓ Componentes GUI inicializados")
        
        # 8. Verificar integración backend
        print("\n8. Verificando integración backend...")
        from gui.backend_integration import BackendIntegration
        backend = BackendIntegration()
        services_status['backend_integration'] = "✓ OK"
        print("   ✓ Integración backend inicializada")
        
        # 9. Verificar sistema de reportes
        print("\n9. Verificando sistema de reportes...")
        from gui.enhanced_webview import EnhancedWebView
        from gui.interactive_report_generator import InteractiveReportViewer
        webview = EnhancedWebView()
        report_viewer = InteractiveReportViewer()
        services_status['reporting'] = "✓ OK"
        print("   ✓ WebView mejorado inicializado")
        print("   ✓ Generador de reportes interactivo inicializado")
        
    except Exception as e:
        print(f"   ✗ ERROR: {e}")
        services_status['error'] = str(e)
    
    # Resumen final
    print("\n" + "=" * 60)
    print("RESUMEN DE SERVICIOS")
    print("=" * 60)
    
    total_services = len([k for k in services_status.keys() if k != 'error'])
    successful_services = len([v for v in services_status.values() if v == "✓ OK"])
    
    for service, status in services_status.items():
        if service != 'error':
            print(f"{service:25} : {status}")
    
    if 'error' in services_status:
        print(f"\nERROR ENCONTRADO: {services_status['error']}")
    
    print(f"\nServicios verificados: {successful_services}/{total_services}")
    
    if successful_services == total_services:
        print("✓ TODOS LOS SERVICIOS ESTÁN DISPONIBLES")
        return True
    else:
        print("✗ ALGUNOS SERVICIOS NO ESTÁN DISPONIBLES")
        return False

if __name__ == "__main__":
    success = verify_gui_services()
    sys.exit(0 if success else 1)