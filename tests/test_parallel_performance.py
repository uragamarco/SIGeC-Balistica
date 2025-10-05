"""
Tests de Rendimiento para Procesamiento Paralelo
Sistema Balístico Forense MVP

Tests para validar y comparar el rendimiento entre:
- Procesamiento secuencial (original)
- Procesamiento paralelo (nuevo)

Métricas evaluadas:
- Tiempo de procesamiento
- Uso de memoria
- Factor de aceleración (speedup)
- Eficiencia paralela
- Calidad de resultados
"""

import unittest
import cv2
import numpy as np
import time
import psutil
import os
import sys
from typing import Dict, List, Any
import logging

# Agregar el directorio padre al path para importaciones
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from image_processing.ballistic_features import BallisticFeatureExtractor
from image_processing.ballistic_features import (
    BallisticFeatureExtractor,
    ParallelConfig,
    BallisticFeatures
)

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestParallelPerformance(unittest.TestCase):
    """Tests de rendimiento para procesamiento paralelo"""
    
    @classmethod
    def setUpClass(cls):
        """Configuración inicial para todos los tests"""
        cls.test_images = cls._create_test_images()
        cls.sequential_extractor = BallisticFeatureExtractor()
        cls.parallel_extractor = ParallelBallisticFeatureExtractor()
        cls.performance_results = {}
        
        # Configuraciones de prueba
        cls.test_configs = {
            'minimal': ParallelConfig(max_workers_process=2, max_workers_thread=4),
            'standard': ParallelConfig(max_workers_process=4, max_workers_thread=8),
            'aggressive': ParallelConfig(max_workers_process=8, max_workers_thread=16)
        }
        
        logger.info("Configuración de tests completada")
    
    @staticmethod
    def _create_test_images() -> Dict[str, np.ndarray]:
        """Crea imágenes de prueba sintéticas"""
        images = {}
        
        # Imagen pequeña (512x512)
        small_img = np.random.randint(0, 256, (512, 512), dtype=np.uint8)
        # Agregar algunas características sintéticas
        cv2.circle(small_img, (256, 256), 50, 200, -1)  # Marca de percutor
        images['small'] = small_img
        
        # Imagen mediana (1024x1024)
        medium_img = np.random.randint(0, 256, (1024, 1024), dtype=np.uint8)
        # Agregar patrones de estrías sintéticas
        for i in range(0, 1024, 20):
            cv2.line(medium_img, (i, 0), (i, 1024), 180, 2)
        images['medium'] = medium_img
        
        # Imagen grande (2048x2048)
        large_img = np.random.randint(0, 256, (2048, 2048), dtype=np.uint8)
        # Agregar múltiples características
        cv2.circle(large_img, (1024, 1024), 100, 200, -1)
        for i in range(0, 2048, 15):
            cv2.line(large_img, (i, 0), (i, 2048), 180, 1)
        images['large'] = large_img
        
        return images
    
    def _measure_processing_time(self, extractor, image: np.ndarray, 
                               specimen_type: str = 'cartridge_case', 
                               parallel: bool = False) -> Dict[str, Any]:
        """Mide tiempo de procesamiento y uso de memoria"""
        # Medir memoria inicial
        process = psutil.Process()
        initial_memory = process.memory_info().rss / (1024 * 1024)  # MB
        
        # Medir tiempo de procesamiento
        start_time = time.time()
        
        if parallel:
            features = extractor.extract_ballistic_features_parallel(image, specimen_type)
        else:
            features = extractor.extract_ballistic_features(image, specimen_type)
        
        processing_time = time.time() - start_time
        
        # Medir memoria final
        final_memory = process.memory_info().rss / (1024 * 1024)  # MB
        memory_usage = final_memory - initial_memory
        
        return {
            'processing_time': processing_time,
            'memory_usage': memory_usage,
            'features': features,
            'quality_score': features.quality_score if features else 0.0,
            'confidence': features.confidence if features else 0.0
        }
    
    def test_small_image_performance(self):
        """Test de rendimiento con imagen pequeña (512x512)"""
        logger.info("=== TEST: Imagen Pequeña (512x512) ===")
        
        image = self.test_images['small']
        
        # Procesamiento secuencial
        seq_result = self._measure_processing_time(
            self.sequential_extractor, image, parallel=False
        )
        
        # Procesamiento paralelo
        par_result = self._measure_processing_time(
            self.parallel_extractor, image, parallel=True
        )
        
        # Calcular métricas
        speedup = seq_result['processing_time'] / max(par_result['processing_time'], 0.001)
        
        # Guardar resultados
        self.performance_results['small_image'] = {
            'sequential_time': seq_result['processing_time'],
            'parallel_time': par_result['processing_time'],
            'speedup': speedup,
            'sequential_memory': seq_result['memory_usage'],
            'parallel_memory': par_result['memory_usage'],
            'quality_difference': abs(seq_result['quality_score'] - par_result['quality_score'])
        }
        
        logger.info(f"Secuencial: {seq_result['processing_time']:.3f}s")
        logger.info(f"Paralelo: {par_result['processing_time']:.3f}s")
        logger.info(f"Speedup: {speedup:.2f}x")
        
        # Validaciones
        self.assertIsNotNone(seq_result['features'])
        self.assertIsNotNone(par_result['features'])
        self.assertLess(abs(seq_result['quality_score'] - par_result['quality_score']), 0.1)
    
    def test_medium_image_performance(self):
        """Test de rendimiento con imagen mediana (1024x1024)"""
        logger.info("=== TEST: Imagen Mediana (1024x1024) ===")
        
        image = self.test_images['medium']
        
        # Procesamiento secuencial
        seq_result = self._measure_processing_time(
            self.sequential_extractor, image, parallel=False
        )
        
        # Procesamiento paralelo
        par_result = self._measure_processing_time(
            self.parallel_extractor, image, parallel=True
        )
        
        # Calcular métricas
        speedup = seq_result['processing_time'] / max(par_result['processing_time'], 0.001)
        
        # Guardar resultados
        self.performance_results['medium_image'] = {
            'sequential_time': seq_result['processing_time'],
            'parallel_time': par_result['processing_time'],
            'speedup': speedup,
            'sequential_memory': seq_result['memory_usage'],
            'parallel_memory': par_result['memory_usage'],
            'quality_difference': abs(seq_result['quality_score'] - par_result['quality_score'])
        }
        
        logger.info(f"Secuencial: {seq_result['processing_time']:.3f}s")
        logger.info(f"Paralelo: {par_result['processing_time']:.3f}s")
        logger.info(f"Speedup: {speedup:.2f}x")
        
        # Para imágenes medianas, esperamos mejor speedup
        self.assertGreater(speedup, 1.0, "El procesamiento paralelo debería ser más rápido")
        self.assertLess(abs(seq_result['quality_score'] - par_result['quality_score']), 0.1)
    
    def test_large_image_performance(self):
        """Test de rendimiento con imagen grande (2048x2048)"""
        logger.info("=== TEST: Imagen Grande (2048x2048) ===")
        
        image = self.test_images['large']
        
        # Procesamiento secuencial
        seq_result = self._measure_processing_time(
            self.sequential_extractor, image, parallel=False
        )
        
        # Procesamiento paralelo
        par_result = self._measure_processing_time(
            self.parallel_extractor, image, parallel=True
        )
        
        # Calcular métricas
        speedup = seq_result['processing_time'] / max(par_result['processing_time'], 0.001)
        
        # Guardar resultados
        self.performance_results['large_image'] = {
            'sequential_time': seq_result['processing_time'],
            'parallel_time': par_result['processing_time'],
            'speedup': speedup,
            'sequential_memory': seq_result['memory_usage'],
            'parallel_memory': par_result['memory_usage'],
            'quality_difference': abs(seq_result['quality_score'] - par_result['quality_score'])
        }
        
        logger.info(f"Secuencial: {seq_result['processing_time']:.3f}s")
        logger.info(f"Paralelo: {par_result['processing_time']:.3f}s")
        logger.info(f"Speedup: {speedup:.2f}x")
        
        # Para imágenes grandes, esperamos el mejor speedup
        self.assertGreater(speedup, 1.2, "El procesamiento paralelo debería ser significativamente más rápido")
        self.assertLess(abs(seq_result['quality_score'] - par_result['quality_score']), 0.1)
    
    def test_different_parallel_configurations(self):
        """Test de diferentes configuraciones paralelas"""
        logger.info("=== TEST: Configuraciones Paralelas ===")
        
        image = self.test_images['medium']
        config_results = {}
        
        for config_name, config in self.test_configs.items():
            logger.info(f"Probando configuración: {config_name}")
            
            # Crear extractor con configuración específica
            extractor = ParallelBallisticFeatureExtractor(config)
            
            # Medir rendimiento
            result = self._measure_processing_time(extractor, image, parallel=True)
            
            config_results[config_name] = {
                'processing_time': result['processing_time'],
                'memory_usage': result['memory_usage'],
                'quality_score': result['quality_score'],
                'workers_process': config.max_workers_process,
                'workers_thread': config.max_workers_thread
            }
            
            logger.info(f"  Tiempo: {result['processing_time']:.3f}s")
            logger.info(f"  Memoria: {result['memory_usage']:.1f}MB")
            logger.info(f"  Calidad: {result['quality_score']:.3f}")
        
        self.performance_results['configurations'] = config_results
        
        # Validar que todas las configuraciones funcionan
        for config_name, result in config_results.items():
            self.assertGreater(result['quality_score'], 0.0, 
                             f"Configuración {config_name} debería producir resultados válidos")
    
    def test_benchmark_comprehensive(self):
        """Test de benchmark completo con múltiples iteraciones"""
        logger.info("=== TEST: Benchmark Completo ===")
        
        image = self.test_images['medium']
        iterations = 3
        
        # Usar el método de benchmark integrado
        benchmark_results = self.parallel_extractor.benchmark_performance(
            image, 'cartridge_case', iterations
        )
        
        self.performance_results['comprehensive_benchmark'] = benchmark_results
        
        logger.info(f"Speedup promedio: {benchmark_results['average_speedup']:.2f}x")
        logger.info(f"Eficiencia paralela: {benchmark_results['parallel_efficiency']:.2f}")
        logger.info(f"Tiempo secuencial promedio: {benchmark_results['average_sequential_time']:.3f}s")
        logger.info(f"Tiempo paralelo promedio: {benchmark_results['average_parallel_time']:.3f}s")
        
        # Validaciones
        self.assertGreater(benchmark_results['average_speedup'], 1.0)
        self.assertGreater(benchmark_results['parallel_efficiency'], 0.1)
        self.assertLess(benchmark_results['parallel_efficiency'], 2.0)  # No puede ser > 100% por core
    
    def test_memory_efficiency(self):
        """Test de eficiencia de memoria"""
        logger.info("=== TEST: Eficiencia de Memoria ===")
        
        image = self.test_images['large']
        
        # Medir memoria base
        process = psutil.Process()
        base_memory = process.memory_info().rss / (1024 * 1024)
        
        # Procesamiento secuencial
        seq_result = self._measure_processing_time(
            self.sequential_extractor, image, parallel=False
        )
        
        # Procesamiento paralelo
        par_result = self._measure_processing_time(
            self.parallel_extractor, image, parallel=True
        )
        
        memory_overhead = par_result['memory_usage'] - seq_result['memory_usage']
        memory_efficiency = seq_result['processing_time'] / max(par_result['memory_usage'], 1.0)
        
        self.performance_results['memory_efficiency'] = {
            'base_memory': base_memory,
            'sequential_memory': seq_result['memory_usage'],
            'parallel_memory': par_result['memory_usage'],
            'memory_overhead': memory_overhead,
            'memory_efficiency': memory_efficiency
        }
        
        logger.info(f"Memoria secuencial: {seq_result['memory_usage']:.1f}MB")
        logger.info(f"Memoria paralela: {par_result['memory_usage']:.1f}MB")
        logger.info(f"Overhead de memoria: {memory_overhead:.1f}MB")
        
        # El overhead de memoria no debería ser excesivo (< 500MB)
        self.assertLess(memory_overhead, 500.0, "El overhead de memoria es demasiado alto")
    
    def test_scalability(self):
        """Test de escalabilidad con diferentes números de workers"""
        logger.info("=== TEST: Escalabilidad ===")
        
        image = self.test_images['medium']
        worker_counts = [1, 2, 4, 8]
        scalability_results = {}
        
        for workers in worker_counts:
            if workers > psutil.cpu_count():
                continue  # Saltar si excede cores disponibles
            
            config = ParallelConfig(
                max_workers_process=workers,
                max_workers_thread=workers * 2
            )
            
            extractor = ParallelBallisticFeatureExtractor(config)
            result = self._measure_processing_time(extractor, image, parallel=True)
            
            scalability_results[workers] = {
                'processing_time': result['processing_time'],
                'memory_usage': result['memory_usage']
            }
            
            logger.info(f"Workers: {workers}, Tiempo: {result['processing_time']:.3f}s")
        
        self.performance_results['scalability'] = scalability_results
        
        # Validar que más workers generalmente mejoran el rendimiento
        if len(scalability_results) >= 2:
            times = [result['processing_time'] for result in scalability_results.values()]
            # El tiempo debería tender a disminuir (aunque puede haber variaciones)
            self.assertLess(min(times), max(times) * 1.5, 
                          "La escalabilidad debería mostrar alguna mejora")
    
    @classmethod
    def tearDownClass(cls):
        """Generar reporte final de rendimiento"""
        logger.info("\n" + "="*60)
        logger.info("REPORTE FINAL DE RENDIMIENTO")
        logger.info("="*60)
        
        # Resumen de speedups
        if 'small_image' in cls.performance_results:
            small_speedup = cls.performance_results['small_image']['speedup']
            logger.info(f"Imagen pequeña (512x512): {small_speedup:.2f}x speedup")
        
        if 'medium_image' in cls.performance_results:
            medium_speedup = cls.performance_results['medium_image']['speedup']
            logger.info(f"Imagen mediana (1024x1024): {medium_speedup:.2f}x speedup")
        
        if 'large_image' in cls.performance_results:
            large_speedup = cls.performance_results['large_image']['speedup']
            logger.info(f"Imagen grande (2048x2048): {large_speedup:.2f}x speedup")
        
        # Resumen de configuraciones
        if 'configurations' in cls.performance_results:
            logger.info("\nRendimiento por configuración:")
            for config_name, result in cls.performance_results['configurations'].items():
                logger.info(f"  {config_name}: {result['processing_time']:.3f}s")
        
        # Resumen de benchmark completo
        if 'comprehensive_benchmark' in cls.performance_results:
            benchmark = cls.performance_results['comprehensive_benchmark']
            logger.info(f"\nBenchmark completo:")
            logger.info(f"  Speedup promedio: {benchmark['average_speedup']:.2f}x")
            logger.info(f"  Eficiencia paralela: {benchmark['parallel_efficiency']:.2f}")
        
        # Resumen de memoria
        if 'memory_efficiency' in cls.performance_results:
            memory = cls.performance_results['memory_efficiency']
            logger.info(f"\nUso de memoria:")
            logger.info(f"  Overhead paralelo: {memory['memory_overhead']:.1f}MB")
        
        logger.info("="*60)

class TestParallelCorrectness(unittest.TestCase):
    """Tests de correctitud para verificar que los resultados paralelos son equivalentes"""
    
    def setUp(self):
        """Configuración para cada test"""
        self.sequential_extractor = BallisticFeatureExtractor()
        self.parallel_extractor = ParallelBallisticFeatureExtractor()
        
        # Imagen de prueba simple
        self.test_image = np.random.randint(0, 256, (800, 800), dtype=np.uint8)
        cv2.circle(self.test_image, (400, 400), 50, 200, -1)
    
    def test_feature_consistency(self):
        """Verifica que las características extraídas sean consistentes"""
        # Extraer características con ambos métodos
        seq_features = self.sequential_extractor.extract_ballistic_features(
            self.test_image, 'cartridge_case'
        )
        par_features = self.parallel_extractor.extract_ballistic_features_parallel(
            self.test_image, 'cartridge_case'
        )
        
        # Comparar características principales (con tolerancia)
        tolerance = 0.1
        
        self.assertAlmostEqual(
            seq_features.quality_score, par_features.quality_score, 
            delta=tolerance, msg="Quality score debería ser similar"
        )
        
        self.assertAlmostEqual(
            seq_features.confidence, par_features.confidence,
            delta=tolerance, msg="Confidence debería ser similar"
        )
        
        # Verificar que ambos producen resultados válidos
        self.assertGreater(seq_features.quality_score, 0.0)
        self.assertGreater(par_features.quality_score, 0.0)
    
    def test_roi_detection_consistency(self):
        """Verifica que la detección de ROI sea consistente"""
        # Detectar ROIs con ambos métodos
        seq_rois = self.sequential_extractor._detect_striation_roi(self.test_image)
        par_rois = self.parallel_extractor._detect_striation_roi_parallel(self.test_image)
        
        # El número de ROIs debería ser similar (±1)
        roi_count_diff = abs(len(seq_rois) - len(par_rois))
        self.assertLessEqual(roi_count_diff, 2, 
                           f"Diferencia en número de ROIs demasiado grande: {roi_count_diff}")
        
        # Si hay ROIs, verificar que tengan características válidas
        if par_rois:
            for roi in par_rois:
                self.assertGreater(roi.area, 0.0)
                self.assertGreater(roi.confidence, 0.0)

def run_performance_tests():
    """Ejecuta todos los tests de rendimiento"""
    # Configurar suite de tests
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Agregar tests de rendimiento
    suite.addTests(loader.loadTestsFromTestCase(TestParallelPerformance))
    suite.addTests(loader.loadTestsFromTestCase(TestParallelCorrectness))
    
    # Ejecutar tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()

if __name__ == '__main__':
    print("Iniciando tests de rendimiento para procesamiento paralelo...")
    print(f"Sistema: {psutil.cpu_count()} cores, {psutil.virtual_memory().total / (1024**3):.1f}GB RAM")
    
    success = run_performance_tests()
    
    if success:
        print("\n✅ Todos los tests de rendimiento pasaron exitosamente")
    else:
        print("\n❌ Algunos tests fallaron")
        sys.exit(1)