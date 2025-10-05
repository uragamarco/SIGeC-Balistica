#!/usr/bin/env python3
"""
Test final de integración completa del sistema SEACABA
"""

import sys
import os
import numpy as np
import cv2
from pathlib import Path

# Agregar el directorio raíz al path
sys.path.insert(0, os.path.abspath('.'))

def test_complete_integration():
    """Prueba completa de integración del sistema"""
    print("=== PRUEBA FINAL DE INTEGRACIÓN SEACABA ===\n")
    
    # 1. Importar todos los módulos
    print("1. Importando módulos...")
    try:
        from image_processing.unified_preprocessor import UnifiedPreprocessor, PreprocessingConfig
        from image_processing.feature_extractor import FeatureExtractor
        from database.vector_db import VectorDatabase, BallisticCase, BallisticImage, FeatureVector
        from matching.unified_matcher import UnifiedMatcher
        from config.unified_config import get_unified_config
        print("✓ Todos los módulos importados correctamente")
    except Exception as e:
        print(f"✗ Error importando módulos: {e}")
        return False
    
    # 2. Crear imagen de prueba
    print("\n2. Creando imagen de prueba...")
    test_image = np.random.randint(0, 255, (400, 400, 3), dtype=np.uint8)
    print("✓ Imagen de prueba creada")
    
    # 3. Probar UnifiedPreprocessor
    print("\n3. Probando UnifiedPreprocessor...")
    try:
        preprocessor = UnifiedPreprocessor()
        result = preprocessor.preprocess_image(test_image)
        
        if result.success and result.processed_image is not None:
            print("✓ UnifiedPreprocessor funciona correctamente")
            processed_image = result.processed_image
        else:
            print(f"✗ Error en UnifiedPreprocessor: {result.error_message}")
            return False
    except Exception as e:
        print(f"✗ Error en UnifiedPreprocessor: {e}")
        return False
    
    # 4. Probar FeatureExtractor
    print("\n4. Probando FeatureExtractor...")
    try:
        extractor = FeatureExtractor()
        features = extractor.extract_all_features(processed_image, ['orb'])
        
        if features and 'orb' in features:
            print("✓ FeatureExtractor funciona correctamente")
            orb_features = features['orb']
        else:
            print("✗ Error en FeatureExtractor: no se extrajeron características")
            return False
    except Exception as e:
        print(f"✗ Error en FeatureExtractor: {e}")
        return False
    
    # 5. Probar VectorDatabase
    print("\n5. Probando VectorDatabase...")
    try:
        config = get_unified_config()
        db = VectorDatabase(config)
        
        # Crear caso de prueba con número único
        import time
        unique_number = f"TEST-{int(time.time())}"
        
        test_case = BallisticCase(
            case_number=unique_number,
            investigator="Test User",
            date_created="2024-01-01",
            weapon_type="Pistola",
            caliber="9mm"
        )
        
        case_id = db.add_case(test_case)
        print(f"✓ Caso agregado con ID: {case_id}")
        
        # Crear imagen de prueba con hash único
        import hashlib
        unique_hash = hashlib.md5(f"test_image_{int(time.time())}".encode()).hexdigest()
        
        test_image_record = BallisticImage(
            case_id=case_id,
            filename="test_image.jpg",
            file_path="/tmp/test_image.jpg",
            evidence_type="vaina",
            image_hash=unique_hash,
            width=400,
            height=400,
            file_size=50000
        )
        
        image_id = db.add_image(test_image_record)
        print(f"✓ Imagen agregada con ID: {image_id}")
        
        # Agregar vector de características
        if orb_features['descriptors'] is not None:
            # Crear un vector promedio de los descriptores
            vector = np.mean(orb_features['descriptors'], axis=0).astype(np.float32)
            
            # Crear objeto FeatureVector
            feature_vector = FeatureVector(
                image_id=image_id,
                algorithm="ORB",
                extraction_params="{}"
            )
            
            vector_id = db.add_feature_vector(feature_vector, vector)
            print(f"✓ Vector agregado con ID: {vector_id}")
        else:
            print("⚠ No se pudieron agregar vectores (descriptores vacíos)")
        
        print("✓ VectorDatabase funciona correctamente")
        
    except Exception as e:
        print(f"✗ Error en VectorDatabase: {e}")
        return False
    
    # 6. Probar UnifiedMatcher
    print("\n6. Probando UnifiedMatcher...")
    try:
        matcher = UnifiedMatcher()
        
        # Crear segunda imagen para comparar
        test_image2 = np.random.randint(0, 255, (400, 400, 3), dtype=np.uint8)
        
        # Comparar imágenes
        result = matcher.compare_images(test_image, test_image2)
        
        if result and hasattr(result, 'similarity_score'):
            print(f"✓ UnifiedMatcher funciona correctamente (similitud: {result.similarity_score:.2f})")
        else:
            print("✗ Error en UnifiedMatcher: resultado inválido")
            return False
            
    except Exception as e:
        print(f"✗ Error en UnifiedMatcher: {e}")
        return False
    
    # 7. Probar estadísticas de base de datos
    print("\n7. Probando estadísticas de base de datos...")
    try:
        stats = db.get_database_stats()
        print(f"✓ Estadísticas obtenidas: {stats}")
    except Exception as e:
        print(f"✗ Error obteniendo estadísticas: {e}")
        return False
    
    print("\n=== INTEGRACIÓN COMPLETA EXITOSA ===")
    print("✓ Todos los módulos funcionan correctamente")
    print("✓ La integración entre módulos es exitosa")
    print("✓ El sistema está listo para uso")
    
    return True

if __name__ == "__main__":
    success = test_complete_integration()
    sys.exit(0 if success else 1)