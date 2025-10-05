#!/usr/bin/env python3
"""
Script para probar las correcciones de integración entre módulos
"""

import sys
import os
import numpy as np
import cv2
from pathlib import Path

# Agregar el directorio raíz al path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_feature_extractor_fixes():
    """Prueba las correcciones en FeatureExtractor"""
    print("=== PROBANDO CORRECCIONES EN FEATUREEXTRACTOR ===")
    
    try:
        from image_processing.feature_extractor import FeatureExtractor
        
        # Crear imagen de prueba
        test_image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        gray_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        
        # Inicializar extractor
        extractor = FeatureExtractor()
        
        # Probar extract_all_features (método correcto)
        print("1. Probando extract_all_features...")
        features = extractor.extract_all_features(gray_image, ['orb'])
        print(f"   ✓ Características extraídas: {len(features.get('orb', {}).get('keypoints', []))}")
        
        # Verificar que extract_features no existe
        print("2. Verificando que extract_features no existe...")
        if hasattr(extractor, 'extract_features'):
            print("   ⚠ extract_features aún existe (puede causar confusión)")
        else:
            print("   ✓ extract_features no existe (correcto)")
            
        return True
        
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False

def test_database_fixes():
    """Prueba las correcciones en VectorDatabase"""
    print("\n=== PROBANDO CORRECCIONES EN VECTORDATABASE ===")
    
    try:
        from database.vector_db import VectorDatabase, BallisticCase
        from config.unified_config import get_unified_config
        
        # Probar inicialización con string
        print("1. Probando inicialización con string...")
        db_string = ":memory:"
        db1 = VectorDatabase(db_string)
        print("   ✓ Inicialización con string exitosa")
        
        # Probar inicialización con Config
        print("2. Probando inicialización con Config...")
        config = get_unified_config()
        db2 = VectorDatabase(config)
        print("   ✓ Inicialización con Config exitosa")
        
        # Probar agregar caso
        print("3. Probando agregar caso...")
        test_case = BallisticCase(
            case_number="TEST-001",
            investigator="Test User",
            weapon_type="Pistola",
            caliber="9mm",
            description="Caso de prueba"
        )
        
        case_id = db1.add_case(test_case)
        print(f"   ✓ Caso agregado con ID: {case_id}")
        
        return True
        
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False

def test_preprocessor_fixes():
    """Prueba las correcciones en UnifiedPreprocessor"""
    print("\n=== PROBANDO CORRECCIONES EN UNIFIEDPREPROCESSOR ===")
    
    try:
        from image_processing.unified_preprocessor import UnifiedPreprocessor
        
        # Crear imagen de prueba
        test_image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        
        # Inicializar preprocessor
        preprocessor = UnifiedPreprocessor()
        
        # Probar preprocess_image (método agregado)
        print("1. Probando preprocess_image...")
        result = preprocessor.preprocess_image(test_image)
        print(f"   ✓ Imagen preprocesada: {result.processed_image.shape}")
        
        # Probar preprocess_ballistic_image (método original)
        print("2. Probando preprocess_ballistic_image...")
        result2 = preprocessor.preprocess_ballistic_image(test_image, "vaina")
        print(f"   ✓ Imagen balística preprocesada: {result2.processed_image.shape}")
        
        return True
        
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False

def test_statistical_analyzer_fixes():
    """Prueba las correcciones en StatisticalAnalyzer"""
    print("\n=== PROBANDO CORRECCIONES EN STATISTICALANALYZER ===")
    
    try:
        from image_processing.statistical_analyzer import StatisticalAnalyzer
        
        # Crear imagen de prueba
        test_image = np.random.randint(0, 255, (200, 200), dtype=np.uint8)
        
        # Inicializar analyzer
        analyzer = StatisticalAnalyzer()
        
        # Probar analyze_image (método agregado)
        print("1. Probando analyze_image...")
        analysis = analyzer.analyze_image(test_image)
        print(f"   ✓ Análisis completado: {len(analysis)} métricas")
        print(f"   - Entropía: {analysis.get('entropy', 'N/A')}")
        print(f"   - Contraste: {analysis.get('contrast', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False

def test_unified_matcher():
    """Prueba UnifiedMatcher (debería funcionar correctamente)"""
    print("\n=== PROBANDO UNIFIEDMATCHER ===")
    
    try:
        from matching.unified_matcher import UnifiedMatcher
        
        # Crear imágenes de prueba
        img1 = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        img2 = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        
        # Inicializar matcher
        matcher = UnifiedMatcher()
        
        # Probar extract_features (método correcto en UnifiedMatcher)
        print("1. Probando extract_features...")
        features1 = matcher.extract_features(img1)
        features2 = matcher.extract_features(img2)
        print(f"   ✓ Características extraídas: {len(features1.get('keypoints', []))} / {len(features2.get('keypoints', []))}")
        
        # Probar matching
        print("2. Probando matching...")
        result = matcher.compare_images(img1, img2)
        print(f"   ✓ Matching completado: {result.similarity_score:.3f}")
        
        return True
        
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False

def main():
    """Función principal"""
    print("PRUEBAS DE CORRECCIONES DE INTEGRACIÓN")
    print("=" * 50)
    
    tests = [
        ("FeatureExtractor", test_feature_extractor_fixes),
        ("VectorDatabase", test_database_fixes),
        ("UnifiedPreprocessor", test_preprocessor_fixes),
        ("StatisticalAnalyzer", test_statistical_analyzer_fixes),
        ("UnifiedMatcher", test_unified_matcher)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n=== ERROR EN {test_name.upper()} ===")
            print(f"Error: {e}")
            results[test_name] = False
    
    # Resumen
    print("\n" + "=" * 50)
    print("RESUMEN DE PRUEBAS")
    print("=" * 50)
    
    passed = 0
    total = len(tests)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name:20} : {status}")
        if result:
            passed += 1
    
    print(f"\nResultado: {passed}/{total} pruebas exitosas")
    
    if passed == total:
        print("🎉 ¡Todas las correcciones funcionan correctamente!")
        return True
    else:
        print("⚠️  Algunas correcciones necesitan revisión")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)