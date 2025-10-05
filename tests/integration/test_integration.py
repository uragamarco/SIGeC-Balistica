#!/usr/bin/env python3
"""
Script de Pruebas de Integración Básicas
Sistema Balístico Forense MVP

Verifica que todos los módulos se integren correctamente
"""

import sys
import os
import traceback
from pathlib import Path
import numpy as np
import cv2

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Prueba que todos los módulos se importen correctamente"""
    print("🔍 Probando importaciones de módulos...")
    
    try:
        # Configuración y logging
        from config.unified_config import get_unified_config
        from utils.logger import setup_logging, get_logger
        from utils.validators import SystemValidator, SecurityUtils, FileUtils
        print("✅ Módulos de utilidades importados correctamente")
        
        # Procesamiento de imágenes
        from image_processing.feature_extractor import BallisticFeatureExtractor
        from image_processing.unified_roi_detector import UnifiedROIDetector
        from image_processing.statistical_analyzer import StatisticalAnalyzer
        from image_processing.unified_preprocessor import UnifiedPreprocessor
        print("✅ Módulos de procesamiento de imágenes importados correctamente")
        
        # Base de datos
        from database.vector_db import VectorDatabase, BallisticCase, BallisticImage, FeatureVector
        print("✅ Módulos de base de datos importados correctamente")
        
        # Matching
        from matching.unified_matcher import UnifiedMatcher
        print("✅ Módulos de matching importados correctamente")
        
        # GUI
        from gui.main_window import MainWindow
        print("✅ Módulos de GUI importados correctamente")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en importaciones: {e}")
        traceback.print_exc()
        return False

def test_config_system():
    """Prueba el sistema de configuración"""
    print("\n🔧 Probando sistema de configuración...")
    
    try:
        # Crear configuración
        config = get_unified_config()
        
        # Verificar que se carguen los valores por defecto
        assert config.database.db_path == "data/database"
        assert config.image_processing.orb_features == 5000
        assert config.matching.min_matches == 10
        
        # Crear directorios
        config.create_directories()
        
        # Verificar rutas
        db_path = config.get_database_path()
        faiss_path = config.get_faiss_index_path()
        
        assert db_path.endswith("ballistic_database.db")
        assert "faiss_index" in faiss_path
        
        print("✅ Sistema de configuración funcionando correctamente")
        return True
        
    except Exception as e:
        print(f"❌ Error en configuración: {e}")
        traceback.print_exc()
        return False

def test_logging_system():
    """Prueba el sistema de logging"""
    print("\n📝 Probando sistema de logging...")
    
    try:
        from utils.logger import setup_logging, get_logger
        
        # Configurar logging
        setup_logging()
        
        # Obtener logger
        logger = get_logger("test")
        
        # Probar diferentes niveles
        logger.info("Mensaje de prueba INFO")
        logger.warning("Mensaje de prueba WARNING")
        logger.error("Mensaje de prueba ERROR")
        
        print("✅ Sistema de logging funcionando correctamente")
        return True
        
    except Exception as e:
        print(f"❌ Error en logging: {e}")
        traceback.print_exc()
        return False

def test_validators():
    """Prueba los validadores del sistema"""
    print("\n🛡️ Probando validadores...")
    
    try:
        from utils.validators import SystemValidator
        
        validator = SystemValidator()
        
        # Probar validación de número de caso
        valid, msg = validator.validate_case_number("CASO-2024-001")
        assert valid, f"Validación de caso falló: {msg}"
        
        # Probar validación de investigador
        valid, msg = validator.validate_investigator_name("Juan Pérez")
        assert valid, f"Validación de investigador falló: {msg}"
        
        # Probar validación de tipo de evidencia
        valid, msg = validator.validate_evidence_type("vaina")
        assert valid, f"Validación de evidencia falló: {msg}"
        
        # Probar sanitización de nombre de archivo
        sanitized = validator.sanitize_filename("archivo<>peligroso.jpg")
        assert "<" not in sanitized and ">" not in sanitized
        
        print("✅ Validadores funcionando correctamente")
        return True
        
    except Exception as e:
        print(f"❌ Error en validadores: {e}")
        traceback.print_exc()
        return False

def test_image_processing():
    """Prueba el procesamiento básico de imágenes"""
    print("\n🖼️ Probando procesamiento de imágenes...")
    
    try:
        from config.unified_config import get_unified_config
        from image_processing.unified_preprocessor import UnifiedPreprocessor
        from image_processing.feature_extractor import BallisticFeatureExtractor
        
        config = get_unified_config()
        
        # Crear imagen de prueba
        test_image = np.random.randint(0, 255, (400, 400, 3), dtype=np.uint8)
        
        # Probar preprocesador unificado
        preprocessor = UnifiedPreprocessor()
        
        # Probar conversión a escala de grises
        gray = preprocessor.convert_to_grayscale(test_image)
        assert len(gray.shape) == 2, "Conversión a escala de grises falló"
        
        # Probar mejora de contraste
        enhanced = preprocessor.enhance_contrast(test_image)
        assert enhanced.shape == test_image.shape, "Mejora de contraste falló"
        
        # Probar extractor de características
        extractor = BallisticFeatureExtractor(config)
        
        # Extraer características ORB
        features = extractor.extract_orb_features(gray)
        assert "keypoints" in features, "Extracción ORB falló"
        assert "descriptors" in features, "Descriptores ORB faltantes"
        
        print("✅ Procesamiento de imágenes funcionando correctamente")
        return True
        
    except Exception as e:
        print(f"❌ Error en procesamiento de imágenes: {e}")
        traceback.print_exc()
        return False

def test_database_system():
    """Prueba el sistema de base de datos"""
    print("\n🗄️ Probando sistema de base de datos...")
    
    try:
        from config.unified_config import get_unified_config
        from database.vector_db import VectorDatabase, BallisticCase, BallisticImage
        
        config = get_unified_config()
        config.create_directories()
        
        # Crear base de datos
        db = VectorDatabase(config)
        
        # Probar agregar caso
        case = BallisticCase(
            case_number="TEST-001",
            investigator="Test User",
            weapon_type="Pistola",
            caliber="9mm"
        )
        
        case_id = db.add_case(case)
        assert case_id > 0, "Error agregando caso"
        
        # Probar recuperar caso
        retrieved_case = db.get_case_by_id(case_id)
        assert retrieved_case is not None, "Error recuperando caso"
        assert retrieved_case.case_number == "TEST-001"
        
        # Probar estadísticas
        stats = db.get_database_stats()
        assert "active_cases" in stats, "Error obteniendo estadísticas"
        assert stats["active_cases"] >= 1, "Caso no contabilizado"
        
        print("✅ Sistema de base de datos funcionando correctamente")
        return True
        
    except Exception as e:
        print(f"❌ Error en base de datos: {e}")
        traceback.print_exc()
        return False

def test_matching_system():
    """Prueba el sistema de matching"""
    print("\n🔍 Probando sistema de matching...")
    
    try:
        from config.unified_config import get_unified_config
        from matching.unified_matcher import UnifiedMatcher
        
        config = get_unified_config()
        matcher = UnifiedMatcher()
        
        # Crear datos de prueba
        # Simular keypoints y descriptores
        kp1 = [cv2.KeyPoint(x=10, y=10, size=5) for _ in range(10)]
        kp2 = [cv2.KeyPoint(x=12, y=12, size=5) for _ in range(8)]
        
        # Crear descriptores ORB simulados
        desc1 = np.random.randint(0, 256, (10, 32), dtype=np.uint8)
        desc2 = np.random.randint(0, 256, (8, 32), dtype=np.uint8)
        
        # Probar matching ORB
        result = matcher.match_features(desc1, desc2, kp1, kp2, algorithm="ORB")
        
        assert result["algorithm"] == "ORB", "Algoritmo incorrecto"
        assert result["total_keypoints1"] == 10, "Conteo de keypoints incorrecto"
        assert result["total_keypoints2"] == 8, "Conteo de keypoints incorrecto"
        assert result["similarity_score"] >= 0, "Score de similitud inválido"
        
        print("✅ Sistema de matching funcionando correctamente")
        return True
        
    except Exception as e:
        print(f"❌ Error en matching: {e}")
        traceback.print_exc()
        return False

def test_gui_initialization():
    """Prueba la inicialización de la GUI (sin mostrar ventana)"""
    print("\n🖥️ Probando inicialización de GUI...")
    
    try:
        from PyQt5.QtWidgets import QApplication
        from gui.main_window import MainWindow
        from config.unified_config import get_unified_config
        
        # Crear aplicación Qt (necesaria para widgets)
        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        
        config = get_unified_config()
        
        # Crear ventana principal (sin mostrar)
        window = MainWindow(config)
        
        # Verificar que se creó correctamente
        assert window.windowTitle() == "Sistema Balístico Forense - MVP"
        assert window.config == config
        
        print("✅ GUI se inicializa correctamente")
        return True
        
    except Exception as e:
        print(f"❌ Error en GUI: {e}")
        traceback.print_exc()
        return False

def run_integration_tests():
    """Ejecuta todas las pruebas de integración"""
    print("🚀 Iniciando pruebas de integración del Sistema Balístico Forense MVP")
    print("=" * 70)
    
    tests = [
        ("Importaciones", test_imports),
        ("Sistema de Configuración", test_config_system),
        ("Sistema de Logging", test_logging_system),
        ("Validadores", test_validators),
        ("Procesamiento de Imágenes", test_image_processing),
        ("Sistema de Base de Datos", test_database_system),
        ("Sistema de Matching", test_matching_system),
        ("Inicialización GUI", test_gui_initialization)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ {test_name} falló con excepción: {e}")
            failed += 1
    
    print("\n" + "=" * 70)
    print(f"📊 Resultados de las pruebas:")
    print(f"✅ Pruebas exitosas: {passed}")
    print(f"❌ Pruebas fallidas: {failed}")
    print(f"📈 Porcentaje de éxito: {(passed / (passed + failed)) * 100:.1f}%")
    
    if failed == 0:
        print("\n🎉 ¡Todas las pruebas de integración pasaron exitosamente!")
        print("El sistema está listo para la siguiente fase de desarrollo.")
        return True
    else:
        print(f"\n⚠️ {failed} pruebas fallaron. Revisar errores antes de continuar.")
        return False

if __name__ == "__main__":
    success = run_integration_tests()
    sys.exit(0 if success else 1)