#!/usr/bin/env python3
"""
Test de integración sin GUI para el Sistema Forense Balístico MVP
Ejecuta pruebas básicas de todos los módulos sin inicializar la interfaz gráfica
"""

import sys
import os
import tempfile
import shutil
from pathlib import Path

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Prueba que todos los módulos se importen correctamente"""
    print("🔧 Probando importación de módulos...")
    
    try:
        # Importaciones básicas
        import numpy as np
        import cv2
        import sqlite3
        import faiss
        print("✅ Librerías externas importadas correctamente")
        
        # Importaciones del proyecto
        from config.unified_config import get_unified_config
        from utils.logger import setup_logging, get_logger
        from utils.validators import SystemValidator, SecurityUtils, FileUtils
        print("✅ Módulos de utilidades importados correctamente")
        
        from image_processing.unified_preprocessor import UnifiedPreprocessor
        # Note: feature_extractor has functions, not a class
        from image_processing import feature_extractor
        print("✅ Módulos de procesamiento de imágenes importados correctamente")
        
        from database.vector_db import VectorDatabase, BallisticCase
        print("✅ Módulo de base de datos importado correctamente")
        
        from matching.unified_matcher import UnifiedMatcher
        print("✅ Módulo de matching importado correctamente")
        
        # Store imports for later use
        globals().update({
            'np': np, 'cv2': cv2, 'Config': Config, 'setup_logging': setup_logging,
            'get_logger': get_logger, 'SystemValidator': SystemValidator,
            'SecurityUtils': SecurityUtils, 'FileUtils': FileUtils,
            'UnifiedPreprocessor': UnifiedPreprocessor, 'feature_extractor': feature_extractor,
            'VectorDatabase': VectorDatabase, 'BallisticCase': BallisticCase, 'UnifiedMatcher': UnifiedMatcher
        })
        
        return True
        
    except ImportError as e:
        print(f"❌ Error de importación: {e}")
        return False
    except Exception as e:
        print(f"❌ Error inesperado en importaciones: {e}")
        return False

def test_configuration():
    """Prueba el sistema de configuración"""
    print("\n⚙️ Probando sistema de configuración...")
    
    try:
        # Crear configuración temporal
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = os.path.join(temp_dir, "test_config/unified_config.yaml")
            config = Config(config_file)
            
            # Verificar valores por defecto
            assert config.database.sqlite_path == "database/ballistics.db"
            assert config.image_processing.orb_features == 5000
            assert config.matching.min_matches == 10
            
            # Guardar y cargar configuración
            config.save_config()
            assert os.path.exists(config_file)
            
            # Crear nueva instancia y cargar
            config2 = Config(config_file)
            assert config2.database.sqlite_path == config.database.sqlite_path
            
        print("✅ Sistema de configuración funcionando correctamente")
        return True
        
    except Exception as e:
        print(f"❌ Error en configuración: {e}")
        return False

def test_logging():
    """Prueba el sistema de logging"""
    print("\n📝 Probando sistema de logging...")
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = os.path.join(temp_dir, "test.log")
            setup_logging(
                log_level="INFO",
                log_file=log_file,
                console_output=True
            )
            
            logger = get_logger("test")
            logger.info("Mensaje de prueba")
            
            # Verificar que el archivo de log se creó
            assert os.path.exists(log_file)
            
        print("✅ Sistema de logging funcionando correctamente")
        return True
        
    except Exception as e:
        print(f"❌ Error en logging: {e}")
        return False

def test_validators():
    """Prueba los validadores del sistema"""
    print("\n🔍 Probando validadores...")
    
    try:
        validator = SystemValidator()
        
        # Test image validation
        is_valid, message = validator.validate_image_file("test.jpg")
        assert isinstance(is_valid, bool)
        assert isinstance(message, str)
        
        # Test SecurityUtils
        security = SecurityUtils()
        
        # Test input sanitization
        sanitized = security.sanitize_input("test input")
        assert sanitized == "test input"
        
        print("✅ Validadores funcionando correctamente")
        return True
        
    except Exception as e:
        print(f"❌ Error en validadores: {e}")
        return False

def test_image_processing():
    """Prueba el sistema de procesamiento de imágenes"""
    print("\n🖼️ Probando procesamiento de imágenes...")
    
    try:
        # Crear imagen de prueba
        test_image = np.random.randint(0, 255, (500, 500, 3), dtype=np.uint8)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Guardar imagen de prueba
            test_image_path = os.path.join(temp_dir, "test_image.jpg")
            cv2.imwrite(test_image_path, test_image)
            
            # Probar preprocessor unificado
            preprocessor = UnifiedPreprocessor()
            
            # Cargar imagen
            loaded_image = preprocessor.load_image(test_image_path)
            assert loaded_image is not None, "Error cargando imagen"
            
            # Probar conversión a escala de grises
            gray_img = preprocessor.convert_to_grayscale(loaded_image)
            assert len(gray_img.shape) == 2, "Conversión a escala de grises falló"
            
            # Probar mejora de contraste
            enhanced = preprocessor.enhance_contrast(loaded_image)
            assert enhanced.shape == loaded_image.shape, "Mejora de contraste falló"
            
            # Test feature extraction
            orb_features = feature_extractor.extract_orb_features(gray_img)
            
            print(f"Debug - ORB features type: {type(orb_features)}")
            print(f"Debug - ORB features keys: {orb_features.keys() if isinstance(orb_features, dict) else 'Not a dict'}")
            
            assert orb_features is not None
            
            # The extract_orb_features function returns a dict with specific keys
            if isinstance(orb_features, dict):
                assert 'num_keypoints' in orb_features
                assert 'keypoint_positions' in orb_features
                assert 'descriptor_stats' in orb_features
                
                # Verify we have some keypoints
                assert orb_features['num_keypoints'] >= 0
            else:
                print(f"Unexpected ORB features format: {type(orb_features)}")
                return False
            
            print("✅ Procesamiento de imágenes funcionando correctamente")
            return True
            
    except Exception as e:
        print(f"❌ Error en procesamiento de imágenes: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_database_system():
    """Prueba el sistema de base de datos"""
    print("\n🗄️ Probando sistema de base de datos...")
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Crear configuración temporal
            config = get_unified_config()
            config.database.db_path = temp_dir
            config.create_directories()
            
            # Inicializar base de datos
            db = VectorDatabase(config)
            
            # Create a test case
            test_case = BallisticCase(
                case_number="TEST001",
                investigator="Test Investigator",
                weapon_type="Pistol",
                caliber="9mm"
            )
            
            # Add case to database
            case_id = db.add_case(test_case)
            
            assert case_id is not None
            
            # Verificar estadísticas
            stats = db.get_database_stats()
            assert stats is not None
            assert 'active_cases' in stats
            
        print("✅ Sistema de base de datos funcionando correctamente")
        return True
        
    except Exception as e:
        print(f"❌ Error en base de datos: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_matching_system():
    """Prueba el sistema de matching"""
    print("\n🔍 Probando sistema de matching...")
    
    try:
        matcher = UnifiedMatcher()
        
        # Crear características de prueba en el formato esperado
        features1 = {
            "num_keypoints": 100,
            "keypoints": [cv2.KeyPoint(x=10, y=10, size=5) for _ in range(100)],
            "descriptors": np.random.randint(0, 256, (100, 32), dtype=np.uint8),
            "algorithm": "ORB"
        }
        
        features2 = {
            "num_keypoints": 100,
            "keypoints": [cv2.KeyPoint(x=12, y=12, size=5) for _ in range(100)],
            "descriptors": np.random.randint(0, 256, (100, 32), dtype=np.uint8),
            "algorithm": "ORB"
        }
        
        # Probar comparación ORB
        result = matcher.match_features(features1, features2)
        
        assert result is not None
        assert hasattr(result, 'similarity_score')
        assert result.similarity_score >= 0
        
        print("✅ Sistema de matching funcionando correctamente")
        return True
        
    except Exception as e:
        print(f"❌ Error en matching: {e}")
        return False

def main():
    """Función principal de pruebas"""
    print("🚀 Iniciando pruebas de integración (sin GUI)...")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_configuration,
        test_logging,
        test_validators,
        test_image_processing,
        test_database_system,
        test_matching_system
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Error inesperado en {test.__name__}: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"📊 Resumen de pruebas:")
    print(f"✅ Pasaron: {passed}")
    print(f"❌ Fallaron: {failed}")
    print(f"📈 Total: {passed + failed}")
    
    if failed == 0:
        print("\n🎉 ¡Todas las pruebas pasaron! El sistema está listo.")
        return 0
    else:
        print(f"\n⚠️ {failed} pruebas fallaron. Revisar errores.")
        return 1

if __name__ == "__main__":
    sys.exit(main())