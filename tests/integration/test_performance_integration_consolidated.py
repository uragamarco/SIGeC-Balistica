#!/usr/bin/env python3
"""
Tests de Integración de Rendimiento Consolidados
Sistema Balístico Forense SIGeC-Balisticar

Consolida todos los tests de rendimiento e integración de performance
Migrado desde: test_performance_*.py, test_benchmark_*.py archivos
"""

import sys
import os
import time
import psutil
import unittest
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, patch
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc
import tracemalloc

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Imports del sistema
from config.unified_config import get_unified_config
from utils.logger import get_logger
from utils.memory_manager import MemoryManager
from utils.cache_manager import CacheManager

# Imports de módulos de análisis
try:
    from analysis.image_processor import ImageProcessor
    from analysis.feature_extractor import FeatureExtractor
    from analysis.similarity_calculator import SimilarityCalculator
    from analysis.bootstrap_similarity import BootstrapSimilarity
    from analysis.comparison_worker import ComparisonWorker
    from analysis.unified_matcher import UnifiedMatcher
    ANALYSIS_AVAILABLE = True
except ImportError as e:
    ANALYSIS_AVAILABLE = False
    print(f"⚠️ Módulos de análisis no disponibles: {e}")

# Imports de base de datos
try:
    from database.db_manager import DatabaseManager
    from database.case_manager import CaseManager
    DATABASE_AVAILABLE = True
except ImportError as e:
    DATABASE_AVAILABLE = False
    print(f"⚠️ Módulos de base de datos no disponibles: {e}")


class PerformanceIntegrationTestSuite(unittest.TestCase):
    """Suite consolidada de tests de integración de rendimiento"""
    
    @classmethod
    def setUpClass(cls):
        """Configuración inicial para toda la suite"""
        cls.config = get_unified_config()
        cls.logger = get_logger(__name__)
        cls.test_assets_path = Path(__file__).parent.parent / "assets"
        
        # Configurar límites de rendimiento
        cls.performance_limits = {
            'image_processing_time': 5.0,  # segundos
            'feature_extraction_time': 3.0,  # segundos
            'similarity_calculation_time': 2.0,  # segundos
            'database_query_time': 1.0,  # segundos
            'memory_usage_mb': 500,  # MB
            'cpu_usage_percent': 80  # %
        }
        
        # Inicializar gestores
        cls.memory_manager = MemoryManager()
        cls.cache_manager = CacheManager()
        
        cls.logger.info("Performance Integration Test Suite initialized")
    
    @classmethod
    def tearDownClass(cls):
        """Limpieza final de la suite"""
        # Limpiar cache
        if hasattr(cls, 'cache_manager'):
            cls.cache_manager.clear_all()
        
        # Forzar garbage collection
        gc.collect()
    
    def setUp(self):
        """Configuración para cada test individual"""
        self.start_time = time.time()
        self.start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # Iniciar monitoreo de memoria
        tracemalloc.start()
        
    def tearDown(self):
        """Limpieza después de cada test"""
        # Detener monitoreo de memoria
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        execution_time = time.time() - self.start_time
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        memory_delta = end_memory - self.start_memory
        
        self.logger.debug(f"Test executed in {execution_time:.2f}s, "
                         f"Memory delta: {memory_delta:.2f}MB, "
                         f"Peak memory: {peak / 1024 / 1024:.2f}MB")

    def test_image_processing_performance(self):
        """Test de rendimiento del procesamiento de imágenes"""
        if not ANALYSIS_AVAILABLE:
            self.skipTest("Módulos de análisis no disponibles")
        
        self.logger.info("🖼️ Testing image processing performance...")
        
        try:
            processor = ImageProcessor()
            test_image_path = self._find_test_image()
            
            if not test_image_path:
                self.skipTest("No hay imágenes de test disponibles")
            
            # Test de carga de imagen
            start_time = time.time()
            image = processor.load_image(test_image_path)
            load_time = time.time() - start_time
            
            self.assertIsNotNone(image, "Image should be loaded")
            self.assertLess(load_time, self.performance_limits['image_processing_time'],
                           f"Image loading took {load_time:.2f}s, limit is {self.performance_limits['image_processing_time']}s")
            
            # Test de preprocesamiento
            start_time = time.time()
            processed_image = processor.preprocess(image)
            preprocess_time = time.time() - start_time
            
            self.assertIsNotNone(processed_image, "Processed image should not be None")
            self.assertLess(preprocess_time, self.performance_limits['image_processing_time'],
                           f"Image preprocessing took {preprocess_time:.2f}s")
            
            # Test de múltiples imágenes en paralelo
            test_images = self._get_multiple_test_images(5)
            if len(test_images) > 1:
                start_time = time.time()
                
                with ThreadPoolExecutor(max_workers=4) as executor:
                    futures = [executor.submit(processor.load_image, img_path) 
                              for img_path in test_images]
                    results = [future.result() for future in as_completed(futures)]
                
                parallel_time = time.time() - start_time
                
                self.assertEqual(len(results), len(test_images), 
                               "All images should be processed")
                self.assertLess(parallel_time, self.performance_limits['image_processing_time'] * 2,
                               f"Parallel processing took {parallel_time:.2f}s")
            
        except Exception as e:
            self.fail(f"Image processing performance test failed: {e}")

    def test_feature_extraction_performance(self):
        """Test de rendimiento de extracción de características"""
        if not ANALYSIS_AVAILABLE:
            self.skipTest("Módulos de análisis no disponibles")
        
        self.logger.info("🔍 Testing feature extraction performance...")
        
        try:
            processor = ImageProcessor()
            extractor = FeatureExtractor()
            test_image_path = self._find_test_image()
            
            if not test_image_path:
                self.skipTest("No hay imágenes de test disponibles")
            
            # Cargar imagen
            image = processor.load_image(test_image_path)
            processed_image = processor.preprocess(image)
            
            # Test de extracción de características
            start_time = time.time()
            features = extractor.extract_features(processed_image)
            extraction_time = time.time() - start_time
            
            self.assertIsNotNone(features, "Features should be extracted")
            self.assertLess(extraction_time, self.performance_limits['feature_extraction_time'],
                           f"Feature extraction took {extraction_time:.2f}s")
            
            # Test de extracción en lote
            test_images = self._get_multiple_test_images(3)
            if len(test_images) > 1:
                processed_images = []
                for img_path in test_images:
                    img = processor.load_image(img_path)
                    processed_images.append(processor.preprocess(img))
                
                start_time = time.time()
                batch_features = extractor.extract_features_batch(processed_images)
                batch_time = time.time() - start_time
                
                self.assertEqual(len(batch_features), len(processed_images),
                               "All features should be extracted")
                self.assertLess(batch_time, self.performance_limits['feature_extraction_time'] * 2,
                               f"Batch feature extraction took {batch_time:.2f}s")
            
        except Exception as e:
            self.fail(f"Feature extraction performance test failed: {e}")

    def test_similarity_calculation_performance(self):
        """Test de rendimiento del cálculo de similitud"""
        if not ANALYSIS_AVAILABLE:
            self.skipTest("Módulos de análisis no disponibles")
        
        self.logger.info("⚖️ Testing similarity calculation performance...")
        
        try:
            calculator = SimilarityCalculator()
            
            # Generar datos de test
            features_a = self._generate_mock_features()
            features_b = self._generate_mock_features()
            
            # Test de cálculo de similitud básico
            start_time = time.time()
            similarity = calculator.calculate_similarity(features_a, features_b)
            calc_time = time.time() - start_time
            
            self.assertIsInstance(similarity, (int, float), "Similarity should be numeric")
            self.assertLess(calc_time, self.performance_limits['similarity_calculation_time'],
                           f"Similarity calculation took {calc_time:.2f}s")
            
            # Test de múltiples comparaciones
            feature_sets = [self._generate_mock_features() for _ in range(10)]
            
            start_time = time.time()
            similarities = []
            for i in range(len(feature_sets)):
                for j in range(i + 1, len(feature_sets)):
                    sim = calculator.calculate_similarity(feature_sets[i], feature_sets[j])
                    similarities.append(sim)
            
            multi_calc_time = time.time() - start_time
            
            self.assertEqual(len(similarities), 45, "Should have 45 comparisons (10 choose 2)")
            self.assertLess(multi_calc_time, self.performance_limits['similarity_calculation_time'] * 10,
                           f"Multiple similarity calculations took {multi_calc_time:.2f}s")
            
        except Exception as e:
            self.fail(f"Similarity calculation performance test failed: {e}")

    def test_bootstrap_similarity_performance(self):
        """Test de rendimiento del bootstrap de similitud"""
        if not ANALYSIS_AVAILABLE:
            self.skipTest("Módulos de análisis no disponibles")
        
        self.logger.info("🎲 Testing bootstrap similarity performance...")
        
        try:
            bootstrap = BootstrapSimilarity()
            
            # Generar datos de test
            features_a = self._generate_mock_features()
            features_b = self._generate_mock_features()
            
            # Test de bootstrap con pocas iteraciones
            start_time = time.time()
            result = bootstrap.calculate_confidence_interval(
                features_a, features_b, n_bootstrap=100
            )
            bootstrap_time = time.time() - start_time
            
            self.assertIsInstance(result, dict, "Bootstrap result should be dict")
            self.assertIn('confidence_interval', result, "Should have confidence interval")
            self.assertLess(bootstrap_time, self.performance_limits['similarity_calculation_time'] * 2,
                           f"Bootstrap calculation took {bootstrap_time:.2f}s")
            
            # Test de bootstrap paralelo
            start_time = time.time()
            parallel_result = bootstrap.calculate_confidence_interval_parallel(
                features_a, features_b, n_bootstrap=100, n_workers=4
            )
            parallel_time = time.time() - start_time
            
            self.assertIsInstance(parallel_result, dict, "Parallel bootstrap result should be dict")
            self.assertLess(parallel_time, bootstrap_time,
                           "Parallel bootstrap should be faster than sequential")
            
        except Exception as e:
            self.fail(f"Bootstrap similarity performance test failed: {e}")

    def test_database_performance(self):
        """Test de rendimiento de operaciones de base de datos"""
        if not DATABASE_AVAILABLE:
            self.skipTest("Módulos de base de datos no disponibles")
        
        self.logger.info("🗄️ Testing database performance...")
        
        try:
            db_manager = DatabaseManager()
            case_manager = CaseManager()
            
            # Test de conexión
            start_time = time.time()
            connection = db_manager.get_connection()
            connection_time = time.time() - start_time
            
            self.assertIsNotNone(connection, "Database connection should be established")
            self.assertLess(connection_time, self.performance_limits['database_query_time'],
                           f"Database connection took {connection_time:.2f}s")
            
            # Test de consulta simple
            start_time = time.time()
            cases = case_manager.get_all_cases(limit=100)
            query_time = time.time() - start_time
            
            self.assertIsInstance(cases, list, "Cases should be a list")
            self.assertLess(query_time, self.performance_limits['database_query_time'],
                           f"Database query took {query_time:.2f}s")
            
            # Test de inserción en lote (simulada)
            test_cases = [
                {"name": f"test_case_{i}", "description": f"Test case {i}"}
                for i in range(10)
            ]
            
            start_time = time.time()
            with patch.object(case_manager, 'create_case') as mock_create:
                mock_create.return_value = {"id": 1, "status": "created"}
                
                for case_data in test_cases:
                    case_manager.create_case(case_data)
            
            batch_insert_time = time.time() - start_time
            
            self.assertLess(batch_insert_time, self.performance_limits['database_query_time'] * 5,
                           f"Batch insert took {batch_insert_time:.2f}s")
            
        except Exception as e:
            self.fail(f"Database performance test failed: {e}")

    def test_memory_usage_performance(self):
        """Test de uso de memoria durante operaciones"""
        self.logger.info("💾 Testing memory usage performance...")
        
        try:
            initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            # Simular operaciones que consumen memoria
            large_data = []
            for i in range(1000):
                # Simular datos de características
                features = {
                    'keypoints': [(j, j+1) for j in range(100)],
                    'descriptors': [j * 0.1 for j in range(500)],
                    'metadata': {'id': i, 'timestamp': time.time()}
                }
                large_data.append(features)
            
            peak_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            memory_usage = peak_memory - initial_memory
            
            self.assertLess(memory_usage, self.performance_limits['memory_usage_mb'],
                           f"Memory usage was {memory_usage:.2f}MB, limit is {self.performance_limits['memory_usage_mb']}MB")
            
            # Test de liberación de memoria
            del large_data
            gc.collect()
            
            final_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            memory_freed = peak_memory - final_memory
            
            self.assertGreater(memory_freed, memory_usage * 0.5,
                             f"Should free at least 50% of used memory, freed {memory_freed:.2f}MB")
            
        except Exception as e:
            self.fail(f"Memory usage performance test failed: {e}")

    def test_cpu_usage_performance(self):
        """Test de uso de CPU durante operaciones intensivas"""
        self.logger.info("⚡ Testing CPU usage performance...")
        
        try:
            # Monitorear CPU antes de la operación
            cpu_before = psutil.cpu_percent(interval=1)
            
            # Simular operación intensiva de CPU
            def cpu_intensive_task():
                result = 0
                for i in range(1000000):
                    result += i ** 0.5
                return result
            
            start_time = time.time()
            
            # Ejecutar tarea en múltiples threads
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(cpu_intensive_task) for _ in range(4)]
                results = [future.result() for future in as_completed(futures)]
            
            execution_time = time.time() - start_time
            
            # Monitorear CPU después de la operación
            cpu_after = psutil.cpu_percent(interval=1)
            
            self.assertEqual(len(results), 4, "All CPU tasks should complete")
            self.assertLess(execution_time, 10.0, "CPU intensive task should complete in reasonable time")
            
            # Verificar que el CPU no se sature completamente
            max_cpu_usage = max(cpu_before, cpu_after)
            self.assertLess(max_cpu_usage, self.performance_limits['cpu_usage_percent'],
                           f"CPU usage was {max_cpu_usage}%, limit is {self.performance_limits['cpu_usage_percent']}%")
            
        except Exception as e:
            self.fail(f"CPU usage performance test failed: {e}")

    def test_cache_performance(self):
        """Test de rendimiento del sistema de cache"""
        self.logger.info("🚀 Testing cache performance...")
        
        try:
            cache_manager = CacheManager()
            
            # Test de escritura en cache
            test_data = {"features": [i for i in range(1000)], "metadata": {"test": True}}
            
            start_time = time.time()
            cache_manager.set("test_key", test_data)
            write_time = time.time() - start_time
            
            self.assertLess(write_time, 0.1, f"Cache write took {write_time:.4f}s")
            
            # Test de lectura desde cache
            start_time = time.time()
            cached_data = cache_manager.get("test_key")
            read_time = time.time() - start_time
            
            self.assertEqual(cached_data, test_data, "Cached data should match original")
            self.assertLess(read_time, 0.01, f"Cache read took {read_time:.4f}s")
            
            # Test de múltiples operaciones de cache
            start_time = time.time()
            for i in range(100):
                cache_manager.set(f"key_{i}", {"data": i})
                retrieved = cache_manager.get(f"key_{i}")
                self.assertEqual(retrieved["data"], i)
            
            multi_ops_time = time.time() - start_time
            
            self.assertLess(multi_ops_time, 1.0, f"100 cache operations took {multi_ops_time:.2f}s")
            
        except Exception as e:
            self.fail(f"Cache performance test failed: {e}")

    def test_concurrent_operations_performance(self):
        """Test de rendimiento de operaciones concurrentes"""
        self.logger.info("🔄 Testing concurrent operations performance...")
        
        try:
            def simulate_analysis_task(task_id):
                """Simular tarea de análisis"""
                time.sleep(0.1)  # Simular procesamiento
                return {"task_id": task_id, "result": f"analysis_{task_id}"}
            
            # Test de operaciones concurrentes
            start_time = time.time()
            
            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = [executor.submit(simulate_analysis_task, i) for i in range(20)]
                results = [future.result() for future in as_completed(futures)]
            
            concurrent_time = time.time() - start_time
            
            self.assertEqual(len(results), 20, "All concurrent tasks should complete")
            self.assertLess(concurrent_time, 5.0, f"Concurrent operations took {concurrent_time:.2f}s")
            
            # Verificar que la concurrencia es efectiva
            sequential_time_estimate = 20 * 0.1  # 20 tareas * 0.1s cada una
            efficiency = sequential_time_estimate / concurrent_time
            
            self.assertGreater(efficiency, 2.0, f"Concurrency efficiency should be > 2x, got {efficiency:.2f}x")
            
        except Exception as e:
            self.fail(f"Concurrent operations performance test failed: {e}")

    def _find_test_image(self) -> Optional[str]:
        """Buscar imagen de test disponible"""
        test_images = [
            self.test_assets_path / "test_image.png",
            self.test_assets_path / "FBI 58A008995 RP1_BFR.png",
            self.test_assets_path / "FBI B240793 RP1_BFR.png",
            self.test_assets_path / "SS007_CCI BF R.png"
        ]
        
        for img_path in test_images:
            if img_path.exists():
                return str(img_path)
        
        return None

    def _get_multiple_test_images(self, count: int) -> List[str]:
        """Obtener múltiples imágenes de test"""
        test_images = []
        base_image = self._find_test_image()
        
        if base_image:
            test_images.append(base_image)
            # Duplicar la imagen base para simular múltiples imágenes
            for i in range(1, count):
                test_images.append(base_image)
        
        return test_images

    def _generate_mock_features(self) -> Dict[str, Any]:
        """Generar características mock para testing"""
        import random
        
        return {
            'keypoints': [(random.random() * 100, random.random() * 100) for _ in range(50)],
            'descriptors': [random.random() for _ in range(128)],
            'metadata': {
                'image_size': (640, 480),
                'extraction_time': random.random(),
                'algorithm': 'SIFT'
            }
        }


def run_performance_integration_tests():
    """Ejecutar todos los tests de integración de rendimiento"""
    print("🚀 Ejecutando Tests de Integración de Rendimiento Consolidados")
    print("=" * 70)
    
    # Configurar test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(PerformanceIntegrationTestSuite)
    
    # Ejecutar tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Resumen de resultados
    print("\n" + "=" * 70)
    print(f"Tests ejecutados: {result.testsRun}")
    print(f"Errores: {len(result.errors)}")
    print(f"Fallos: {len(result.failures)}")
    print(f"Saltados: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
    
    success = len(result.errors) == 0 and len(result.failures) == 0
    print(f"Estado: {'✅ ÉXITO' if success else '❌ FALLÓ'}")
    
    return success


if __name__ == "__main__":
    success = run_performance_integration_tests()
    sys.exit(0 if success else 1)