#!/usr/bin/env python3
"""
Test completo del sistema consolidado de procesamiento balístico
Verifica que todas las funcionalidades estén integradas correctamente
"""

import sys
import os
import numpy as np
import cv2
import time
import logging
from typing import Dict, Any

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configurar el path para importaciones
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Importar módulos del sistema
try:
    # Importar directamente con rutas correctas desde tests
    sys.path.append(os.path.join(current_dir, '..'))
    
    # Importar ballistic_features
    from image_processing import ballistic_features
    from image_processing.ballistic_features import BallisticFeatureExtractor, ParallelConfig
    
    # Importar feature_extractor
    from image_processing import feature_extractor
    
    # Importar unified_preprocessor
    from image_processing import unified_preprocessor
    from image_processing.unified_preprocessor import UnifiedPreprocessor
    
    # Importar unified_roi_detector
    from image_processing import unified_roi_detector
    from image_processing.unified_roi_detector import UnifiedROIDetector
    
    MODULES_AVAILABLE = True
    logger.info("✓ Todos los módulos importados correctamente")
    
except ImportError as e:
    logger.error(f"Error importando módulos: {e}")
    MODULES_AVAILABLE = False

def create_synthetic_cartridge_image(size=(512, 512)) -> np.ndarray:
    """Crea una imagen sintética de casquillo para pruebas"""
    image = np.zeros(size, dtype=np.uint8)
    center = (size[0] // 2, size[1] // 2)
    
    # Círculo principal del casquillo
    cv2.circle(image, center, size[0] // 3, 180, -1)
    
    # Percutor (círculo pequeño en el centro)
    cv2.circle(image, center, 15, 220, -1)
    
    # Estrías radiales
    for angle in range(0, 360, 30):
        x1 = center[0] + int(20 * np.cos(np.radians(angle)))
        y1 = center[1] + int(20 * np.sin(np.radians(angle)))
        x2 = center[0] + int(80 * np.cos(np.radians(angle)))
        y2 = center[1] + int(80 * np.sin(np.radians(angle)))
        cv2.line(image, (x1, y1), (x2, y2), 200, 2)
    
    # Añadir ruido realista
    noise = np.random.normal(0, 10, size).astype(np.uint8)
    image = cv2.add(image, noise)
    
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

def test_ballistic_feature_extractor():
    """Test del extractor de características balísticas consolidado"""
    logger.info("=== Test: BallisticFeatureExtractor Consolidado ===")
    
    try:
        # Configuración paralela
        parallel_config = ParallelConfig(
            max_workers_process=2,
            max_workers_thread=4,
            enable_gabor_parallel=True,
            enable_roi_parallel=True
        )
        
        extractor = BallisticFeatureExtractor(parallel_config=parallel_config)
        image = create_synthetic_cartridge_image()
        
        # Test procesamiento secuencial
        start_time = time.time()
        features_seq = extractor.extract_ballistic_features(image, use_parallel=False)
        seq_time = time.time() - start_time
        
        # Test procesamiento paralelo
        start_time = time.time()
        features_par = extractor.extract_ballistic_features(image, use_parallel=True)
        par_time = time.time() - start_time
        
        # Verificar que ambos métodos funcionan
        assert features_seq is not None, "Extracción secuencial falló"
        assert features_par is not None, "Extracción paralela falló"
        
        # Verificar nuevos campos de estrías
        assert hasattr(features_seq, 'striation_num_lines'), "Campo striation_num_lines faltante"
        assert hasattr(features_seq, 'striation_dominant_directions'), "Campo striation_dominant_directions faltante"
        assert hasattr(features_seq, 'striation_parallelism_score'), "Campo striation_parallelism_score faltante"
        
        # Test de benchmark
        benchmark_results = extractor.benchmark_performance(image, iterations=2)
        assert 'average_speedup_factor' in benchmark_results, "Benchmark falló"
        
        logger.info(f"✓ Extracción secuencial: {seq_time:.3f}s")
        logger.info(f"✓ Extracción paralela: {par_time:.3f}s")
        logger.info(f"✓ Factor de aceleración promedio: {benchmark_results['average_speedup_factor']:.2f}x")
        logger.info(f"✓ Nuevos campos de estrías implementados correctamente")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Error en test de BallisticFeatureExtractor: {e}")
        return False

def test_feature_extractor_api():
    """Test del API de feature_extractor especializado"""
    logger.info("=== Test: Feature Extractor API ===")
    
    try:
        import feature_extractor
        flask_app = feature_extractor.app
        
        with flask_app.test_client() as client:
            # Test health check
            response = client.get('/health')
            assert response.status_code == 200, "Health check falló"
            
            # Crear imagen de prueba
            image = create_synthetic_cartridge_image()
            _, buffer = cv2.imencode('.jpg', image)
            
            # Test extracción balística
            response = client.post('/api/extract', 
                                 data={'image': (buffer.tobytes(), 'test.jpg')},
                                 content_type='multipart/form-data')
            
            assert response.status_code == 200, f"API falló: {response.status_code}"
            
            result = response.get_json()
            assert 'features' in result, "Respuesta API sin features"
            assert 'processing_time' in result, "Respuesta API sin processing_time"
            
            logger.info("✓ API Health check funcional")
            logger.info("✓ Endpoint de extracción balística funcional")
            logger.info(f"✓ Tiempo de procesamiento API: {result['processing_time']:.3f}s")
            
            return True
            
    except Exception as e:
        logger.error(f"✗ Error en test de Feature Extractor API: {e}")
        return False

def test_unified_modules():
    """Test de módulos unificados"""
    logger.info("=== Test: Módulos Unificados ===")
    
    try:
        image = create_synthetic_cartridge_image()
        
        # Test UnifiedPreprocessor
        preprocessor = UnifiedPreprocessor()
        preprocessing_result = preprocessor.preprocess_image(image)
        processed_image = preprocessing_result.processed_image
        assert preprocessing_result is not None, "UnifiedPreprocessor falló"
        assert processed_image is not None, "Imagen procesada es None"
        assert len(processed_image.shape) >= 2, "Imagen procesada debe tener al menos 2 dimensiones"
        assert processed_image.shape[0] > 0 and processed_image.shape[1] > 0, "Dimensiones de imagen procesada deben ser positivas"
        
        # Test UnifiedROIDetector
        roi_detector = UnifiedROIDetector()
        roi_regions = roi_detector.detect_roi_regions(processed_image, 'cartridge_case')
        assert isinstance(roi_regions, list), "UnifiedROIDetector no retornó lista"
        
        logger.info("✓ UnifiedPreprocessor funcional")
        logger.info("✓ UnifiedROIDetector funcional")
        logger.info(f"✓ ROI detectadas: {len(roi_regions)}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Error en test de módulos unificados: {e}")
        return False

def test_system_integration():
    """Test de integración completa del sistema"""
    logger.info("=== Test: Integración Completa del Sistema ===")
    
    try:
        # Configurar componentes
        parallel_config = ParallelConfig(
            max_workers_process=2,
            max_workers_thread=4,
            enable_gabor_parallel=True,
            enable_roi_parallel=True
        )
        
        extractor = BallisticFeatureExtractor(parallel_config=parallel_config)
        preprocessor = UnifiedPreprocessor()
        roi_detector = UnifiedROIDetector()
        
        # Imagen de prueba
        image = create_synthetic_cartridge_image()
        
        # Pipeline completo
        start_time = time.time()
        
        # 1. Preprocesamiento
        preprocessing_result = preprocessor.preprocess_image(image)
        processed_image = preprocessing_result.processed_image
        
        # 2. Detección ROI
        roi_regions = roi_detector.detect_roi_regions(processed_image, 'cartridge_case')
        
        # 3. Extracción de características
        features = extractor.extract_ballistic_features(processed_image, use_parallel=True)
        
        total_time = time.time() - start_time
        
        # Verificaciones
        assert preprocessing_result is not None, "Preprocesamiento falló"
        assert processed_image is not None, "Imagen procesada es None"
        assert isinstance(roi_regions, list), "Detección ROI falló"
        assert features is not None, "Extracción de características falló"
        
        # Verificar calidad de resultados
        assert features.quality_score > 0, "Score de calidad inválido"
        assert features.confidence > 0, "Confianza inválida"
        
        logger.info(f"✓ Pipeline completo ejecutado en {total_time:.3f}s")
        logger.info(f"✓ Calidad de imagen: {features.quality_score:.3f}")
        logger.info(f"✓ Confianza: {features.confidence:.3f}")
        logger.info(f"✓ ROI detectadas: {len(roi_regions)}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Error en test de integración: {e}")
        return False

def run_all_tests():
    """Ejecuta todos los tests del sistema consolidado"""
    logger.info("🚀 Iniciando tests del sistema consolidado...")
    
    if not MODULES_AVAILABLE:
        logger.error("❌ No se pudieron importar los módulos necesarios")
        return False
    
    tests = [
        ("Ballistic Feature Extractor", test_ballistic_feature_extractor),
        ("Feature Extractor API", test_feature_extractor_api),
        ("Módulos Unificados", test_unified_modules),
        ("Integración Completa", test_system_integration)
    ]
    
    results = []
    total_start = time.time()
    
    for test_name, test_func in tests:
        logger.info(f"\n--- Ejecutando: {test_name} ---")
        try:
            result = test_func()
            results.append((test_name, result))
            status = "✅ PASÓ" if result else "❌ FALLÓ"
            logger.info(f"{status}: {test_name}")
        except Exception as e:
            logger.error(f"❌ ERROR en {test_name}: {e}")
            results.append((test_name, False))
    
    total_time = time.time() - total_start
    
    # Resumen final
    logger.info(f"\n{'='*60}")
    logger.info("📊 RESUMEN DE TESTS")
    logger.info(f"{'='*60}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASÓ" if result else "❌ FALLÓ"
        logger.info(f"{status} {test_name}")
    
    logger.info(f"\n🎯 Resultado: {passed}/{total} tests pasaron")
    logger.info(f"⏱️  Tiempo total: {total_time:.2f}s")
    
    if passed == total:
        logger.info("🎉 ¡Todos los tests pasaron! Sistema consolidado funcional.")
        return True
    else:
        logger.error(f"⚠️  {total - passed} tests fallaron. Revisar implementación.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)