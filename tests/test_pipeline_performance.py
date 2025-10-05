#!/usr/bin/env python3
"""
Tests de Rendimiento para el Pipeline Científico
===============================================

Este módulo contiene tests específicos para evaluar el rendimiento
del pipeline científico bajo diferentes condiciones y cargas de trabajo.
"""

import unittest
import time
import psutil
import threading
import tempfile
import os
import sys
from pathlib import Path
from unittest.mock import Mock, patch
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Importar módulos del pipeline
try:
    from core.unified_pipeline import ScientificPipeline, PipelineResult
    from core.pipeline_config import create_pipeline_config, get_predefined_config
    PIPELINE_AVAILABLE = True
except ImportError as e:
    print(f"Error importando pipeline: {e}")
    PIPELINE_AVAILABLE = False

# Importar utilidades de test
try:
    from tests.test_utils import create_test_image, create_ballistic_image
    TEST_UTILS_AVAILABLE = True
except ImportError:
    TEST_UTILS_AVAILABLE = False
    
    # Crear funciones mock si no están disponibles
    def create_test_image(width=800, height=600, noise_level=0.1):
        """Crear imagen de test simple"""
        return np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    
    def create_ballistic_image(width=800, height=600, pattern_type="striation"):
        """Crear imagen balística de test"""
        return create_test_image(width, height)


class PerformanceMetrics:
    """Clase para recopilar métricas de rendimiento"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reiniciar métricas"""
        self.start_time = None
        self.end_time = None
        self.peak_memory = 0
        self.cpu_usage = []
        self.processing_times = {}
        self.thread_count = 0
    
    def start_monitoring(self):
        """Iniciar monitoreo de rendimiento"""
        self.start_time = time.time()
        self.peak_memory = psutil.virtual_memory().used
        self.cpu_usage = []
        
        # Iniciar monitoreo de CPU en hilo separado
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_resources)
        self._monitor_thread.daemon = True
        self._monitor_thread.start()
    
    def stop_monitoring(self):
        """Detener monitoreo de rendimiento"""
        self.end_time = time.time()
        self._monitoring = False
        if hasattr(self, '_monitor_thread'):
            self._monitor_thread.join(timeout=1.0)
    
    def _monitor_resources(self):
        """Monitorear recursos del sistema"""
        while getattr(self, '_monitoring', False):
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=0.1)
                self.cpu_usage.append(cpu_percent)
                
                # Memory usage
                current_memory = psutil.virtual_memory().used
                self.peak_memory = max(self.peak_memory, current_memory)
                
                # Thread count
                self.thread_count = max(self.thread_count, threading.active_count())
                
                time.sleep(0.1)
            except Exception:
                break
    
    def get_execution_time(self):
        """Obtener tiempo total de ejecución"""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0
    
    def get_average_cpu_usage(self):
        """Obtener uso promedio de CPU"""
        return np.mean(self.cpu_usage) if self.cpu_usage else 0
    
    def get_peak_memory_mb(self):
        """Obtener pico de memoria en MB"""
        return self.peak_memory / (1024 * 1024)
    
    def add_processing_time(self, step_name, duration):
        """Agregar tiempo de procesamiento para un paso"""
        self.processing_times[step_name] = duration
    
    def get_summary(self):
        """Obtener resumen de métricas"""
        return {
            'execution_time': self.get_execution_time(),
            'average_cpu_usage': self.get_average_cpu_usage(),
            'peak_memory_mb': self.get_peak_memory_mb(),
            'max_threads': self.thread_count,
            'processing_times': self.processing_times.copy()
        }


@unittest.skipUnless(PIPELINE_AVAILABLE, "Pipeline no disponible")
class TestPipelinePerformance(unittest.TestCase):
    """Tests de rendimiento del pipeline"""
    
    def setUp(self):
        """Configuración inicial para tests de rendimiento"""
        self.temp_dir = tempfile.mkdtemp()
        self.metrics = PerformanceMetrics()
        
        # Crear imágenes de test
        self.test_image1 = create_test_image(800, 600)
        self.test_image2 = create_ballistic_image(800, 600)
        
        # Guardar imágenes temporales
        import cv2
        self.image1_path = os.path.join(self.temp_dir, "test1.jpg")
        self.image2_path = os.path.join(self.temp_dir, "test2.jpg")
        cv2.imwrite(self.image1_path, self.test_image1)
        cv2.imwrite(self.image2_path, self.test_image2)
    
    def tearDown(self):
        """Limpieza después de los tests"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_basic_pipeline_performance(self):
        """Test de rendimiento del pipeline básico"""
        config = create_pipeline_config("basic")
        pipeline = ScientificPipeline(config)
        
        self.metrics.start_monitoring()
        
        start_time = time.time()
        result = pipeline.analyze_images(self.image1_path, self.image2_path)
        end_time = time.time()
        
        self.metrics.stop_monitoring()
        
        # Verificar que el análisis se completó
        self.assertIsInstance(result, PipelineResult)
        
        # Verificar métricas de rendimiento
        execution_time = end_time - start_time
        self.assertLess(execution_time, 30.0, "Pipeline básico demasiado lento")
        
        print(f"\nRendimiento Pipeline Básico:")
        print(f"  Tiempo de ejecución: {execution_time:.2f}s")
        print(f"  Uso promedio CPU: {self.metrics.get_average_cpu_usage():.1f}%")
        print(f"  Pico de memoria: {self.metrics.get_peak_memory_mb():.1f}MB")
    
    def test_standard_pipeline_performance(self):
        """Test de rendimiento del pipeline estándar"""
        config = create_pipeline_config("standard")
        pipeline = ScientificPipeline(config)
        
        self.metrics.start_monitoring()
        
        start_time = time.time()
        result = pipeline.analyze_images(self.image1_path, self.image2_path)
        end_time = time.time()
        
        self.metrics.stop_monitoring()
        
        # Verificar que el análisis se completó
        self.assertIsInstance(result, PipelineResult)
        
        # Verificar métricas de rendimiento
        execution_time = end_time - start_time
        self.assertLess(execution_time, 60.0, "Pipeline estándar demasiado lento")
        
        print(f"\nRendimiento Pipeline Estándar:")
        print(f"  Tiempo de ejecución: {execution_time:.2f}s")
        print(f"  Uso promedio CPU: {self.metrics.get_average_cpu_usage():.1f}%")
        print(f"  Pico de memoria: {self.metrics.get_peak_memory_mb():.1f}MB")
    
    def test_advanced_pipeline_performance(self):
        """Test de rendimiento del pipeline avanzado"""
        config = create_pipeline_config("advanced")
        pipeline = ScientificPipeline(config)
        
        self.metrics.start_monitoring()
        
        start_time = time.time()
        result = pipeline.analyze_images(self.image1_path, self.image2_path)
        end_time = time.time()
        
        self.metrics.stop_monitoring()
        
        # Verificar que el análisis se completó
        self.assertIsInstance(result, PipelineResult)
        
        # Verificar métricas de rendimiento
        execution_time = end_time - start_time
        self.assertLess(execution_time, 120.0, "Pipeline avanzado demasiado lento")
        
        print(f"\nRendimiento Pipeline Avanzado:")
        print(f"  Tiempo de ejecución: {execution_time:.2f}s")
        print(f"  Uso promedio CPU: {self.metrics.get_average_cpu_usage():.1f}%")
        print(f"  Pico de memoria: {self.metrics.get_peak_memory_mb():.1f}MB")
    
    def test_forensic_pipeline_performance(self):
        """Test de rendimiento del pipeline forense"""
        config = create_pipeline_config("forensic")
        pipeline = ScientificPipeline(config)
        
        self.metrics.start_monitoring()
        
        start_time = time.time()
        result = pipeline.analyze_images(self.image1_path, self.image2_path)
        end_time = time.time()
        
        self.metrics.stop_monitoring()
        
        # Verificar que el análisis se completó
        self.assertIsInstance(result, PipelineResult)
        
        # Verificar métricas de rendimiento
        execution_time = end_time - start_time
        self.assertLess(execution_time, 300.0, "Pipeline forense demasiado lento")
        
        print(f"\nRendimiento Pipeline Forense:")
        print(f"  Tiempo de ejecución: {execution_time:.2f}s")
        print(f"  Uso promedio CPU: {self.metrics.get_average_cpu_usage():.1f}%")
        print(f"  Pico de memoria: {self.metrics.get_peak_memory_mb():.1f}MB")
    
    def test_memory_usage_scaling(self):
        """Test de escalabilidad del uso de memoria"""
        config = create_pipeline_config("standard")
        pipeline = ScientificPipeline(config)
        
        # Test con diferentes tamaños de imagen
        image_sizes = [(400, 300), (800, 600), (1600, 1200)]
        memory_usage = []
        
        for width, height in image_sizes:
            # Crear imagen de test
            test_image = create_test_image(width, height)
            image_path = os.path.join(self.temp_dir, f"test_{width}x{height}.jpg")
            
            import cv2
            cv2.imwrite(image_path, test_image)
            
            # Medir uso de memoria
            self.metrics.reset()
            self.metrics.start_monitoring()
            
            try:
                result = pipeline.analyze_images(image_path, self.image2_path)
                self.assertIsInstance(result, PipelineResult)
            except Exception as e:
                print(f"Error con imagen {width}x{height}: {e}")
                continue
            
            self.metrics.stop_monitoring()
            memory_usage.append(self.metrics.get_peak_memory_mb())
            
            print(f"Memoria para {width}x{height}: {self.metrics.get_peak_memory_mb():.1f}MB")
        
        # Verificar que el uso de memoria no crece exponencialmente
        if len(memory_usage) >= 2:
            growth_ratio = memory_usage[-1] / memory_usage[0]
            self.assertLess(growth_ratio, 10.0, "Uso de memoria crece demasiado rápido")
    
    def test_concurrent_pipeline_execution(self):
        """Test de ejecución concurrente del pipeline"""
        config = create_pipeline_config("basic")
        
        # Crear múltiples pares de imágenes
        image_pairs = []
        for i in range(3):
            img1 = create_test_image(400, 300)
            img2 = create_test_image(400, 300)
            
            path1 = os.path.join(self.temp_dir, f"concurrent1_{i}.jpg")
            path2 = os.path.join(self.temp_dir, f"concurrent2_{i}.jpg")
            
            import cv2
            cv2.imwrite(path1, img1)
            cv2.imwrite(path2, img2)
            
            image_pairs.append((path1, path2))
        
        self.metrics.start_monitoring()
        start_time = time.time()
        
        # Ejecutar pipelines concurrentemente
        results = []
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = []
            
            for path1, path2 in image_pairs:
                pipeline = ScientificPipeline(config)
                future = executor.submit(pipeline.analyze_images, path1, path2)
                futures.append(future)
            
            # Recopilar resultados
            for future in as_completed(futures):
                try:
                    result = future.result(timeout=60)
                    results.append(result)
                except Exception as e:
                    print(f"Error en ejecución concurrente: {e}")
        
        end_time = time.time()
        self.metrics.stop_monitoring()
        
        # Verificar resultados
        self.assertEqual(len(results), len(image_pairs))
        
        execution_time = end_time - start_time
        print(f"\nEjecución Concurrente:")
        print(f"  Tiempo total: {execution_time:.2f}s")
        print(f"  Hilos máximos: {self.metrics.thread_count}")
        print(f"  Uso promedio CPU: {self.metrics.get_average_cpu_usage():.1f}%")
    
    def test_pipeline_step_profiling(self):
        """Test de profiling de pasos individuales del pipeline"""
        config = create_pipeline_config("standard")
        pipeline = ScientificPipeline(config)
        
        # Instrumentar pipeline para medir tiempos
        step_times = {}
        
        # Mock de métodos para medir tiempos
        original_methods = {}
        
        def time_method(method_name, original_method):
            def timed_method(*args, **kwargs):
                start = time.time()
                result = original_method(*args, **kwargs)
                end = time.time()
                step_times[method_name] = end - start
                return result
            return timed_method
        
        # Instrumentar métodos principales
        methods_to_time = [
            'load_images', 'assess_quality', 'preprocess_images',
            'detect_roi', 'extract_and_match_features', 'perform_cmc_analysis'
        ]
        
        for method_name in methods_to_time:
            if hasattr(pipeline, method_name):
                original_method = getattr(pipeline, method_name)
                setattr(pipeline, method_name, 
                       time_method(method_name, original_method))
        
        # Ejecutar pipeline
        result = pipeline.analyze_images(self.image1_path, self.image2_path)
        self.assertIsInstance(result, PipelineResult)
        
        # Mostrar tiempos por paso
        print(f"\nProfiling de Pasos del Pipeline:")
        total_time = sum(step_times.values())
        
        for step, duration in sorted(step_times.items(), key=lambda x: x[1], reverse=True):
            percentage = (duration / total_time) * 100 if total_time > 0 else 0
            print(f"  {step}: {duration:.3f}s ({percentage:.1f}%)")
        
        print(f"  Total medido: {total_time:.3f}s")
        
        # Verificar que ningún paso tome más del 50% del tiempo total
        for step, duration in step_times.items():
            percentage = (duration / total_time) * 100 if total_time > 0 else 0
            self.assertLess(percentage, 50.0, 
                          f"Paso '{step}' toma demasiado tiempo: {percentage:.1f}%")
    
    def test_cache_performance_impact(self):
        """Test del impacto del cache en el rendimiento"""
        # Configuración con cache habilitado
        config_with_cache = create_pipeline_config("standard")
        config_with_cache.enable_caching = True
        
        # Configuración sin cache
        config_without_cache = create_pipeline_config("standard")
        config_without_cache.enable_caching = False
        
        # Primera ejecución sin cache
        pipeline_no_cache = ScientificPipeline(config_without_cache)
        
        start_time = time.time()
        result1 = pipeline_no_cache.analyze_images(self.image1_path, self.image2_path)
        time_without_cache = time.time() - start_time
        
        # Primera ejecución con cache (debería ser similar)
        pipeline_with_cache = ScientificPipeline(config_with_cache)
        
        start_time = time.time()
        result2 = pipeline_with_cache.analyze_images(self.image1_path, self.image2_path)
        time_first_with_cache = time.time() - start_time
        
        # Segunda ejecución con cache (debería ser más rápida)
        start_time = time.time()
        result3 = pipeline_with_cache.analyze_images(self.image1_path, self.image2_path)
        time_second_with_cache = time.time() - start_time
        
        print(f"\nImpacto del Cache:")
        print(f"  Sin cache: {time_without_cache:.2f}s")
        print(f"  Con cache (1ra vez): {time_first_with_cache:.2f}s")
        print(f"  Con cache (2da vez): {time_second_with_cache:.2f}s")
        
        # La segunda ejecución con cache debería ser más rápida
        # (aunque puede no ser significativo en tests pequeños)
        self.assertLessEqual(time_second_with_cache, time_first_with_cache * 1.1)
    
    def test_parallel_processing_performance(self):
        """Test del impacto del procesamiento paralelo"""
        # Configuración con procesamiento paralelo
        config_parallel = create_pipeline_config("standard")
        config_parallel.enable_parallel_processing = True
        config_parallel.max_processing_threads = 4
        
        # Configuración secuencial
        config_sequential = create_pipeline_config("standard")
        config_sequential.enable_parallel_processing = False
        config_sequential.max_processing_threads = 1
        
        # Ejecución secuencial
        pipeline_sequential = ScientificPipeline(config_sequential)
        
        start_time = time.time()
        result1 = pipeline_sequential.analyze_images(self.image1_path, self.image2_path)
        time_sequential = time.time() - start_time
        
        # Ejecución paralela
        pipeline_parallel = ScientificPipeline(config_parallel)
        
        start_time = time.time()
        result2 = pipeline_parallel.analyze_images(self.image1_path, self.image2_path)
        time_parallel = time.time() - start_time
        
        print(f"\nImpacto del Procesamiento Paralelo:")
        print(f"  Secuencial: {time_sequential:.2f}s")
        print(f"  Paralelo: {time_parallel:.2f}s")
        
        if time_sequential > 0:
            speedup = time_sequential / time_parallel
            print(f"  Aceleración: {speedup:.2f}x")
            
            # El procesamiento paralelo debería ser al menos tan rápido
            self.assertLessEqual(time_parallel, time_sequential * 1.2)


@unittest.skipUnless(PIPELINE_AVAILABLE, "Pipeline no disponible")
class TestPipelineBenchmarks(unittest.TestCase):
    """Benchmarks específicos del pipeline"""
    
    def setUp(self):
        """Configuración para benchmarks"""
        self.temp_dir = tempfile.mkdtemp()
        
        # Crear conjunto de imágenes de diferentes tamaños
        self.benchmark_images = {}
        sizes = [
            ("small", 400, 300),
            ("medium", 800, 600),
            ("large", 1600, 1200)
        ]
        
        import cv2
        for size_name, width, height in sizes:
            img1 = create_ballistic_image(width, height, "striation")
            img2 = create_ballistic_image(width, height, "impression")
            
            path1 = os.path.join(self.temp_dir, f"{size_name}_1.jpg")
            path2 = os.path.join(self.temp_dir, f"{size_name}_2.jpg")
            
            cv2.imwrite(path1, img1)
            cv2.imwrite(path2, img2)
            
            self.benchmark_images[size_name] = (path1, path2)
    
    def tearDown(self):
        """Limpieza después de benchmarks"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_benchmark_all_levels(self):
        """Benchmark de todos los niveles de pipeline"""
        levels = ["basic", "standard", "advanced", "forensic"]
        results = {}
        
        print(f"\n{'='*60}")
        print("BENCHMARK COMPLETO DEL PIPELINE CIENTÍFICO")
        print(f"{'='*60}")
        
        for level in levels:
            print(f"\nNivel: {level.upper()}")
            print("-" * 40)
            
            config = create_pipeline_config(level)
            pipeline = ScientificPipeline(config)
            
            level_results = {}
            
            for size_name, (path1, path2) in self.benchmark_images.items():
                metrics = PerformanceMetrics()
                metrics.start_monitoring()
                
                start_time = time.time()
                try:
                    result = pipeline.analyze_images(path1, path2)
                    execution_time = time.time() - start_time
                    success = True
                except Exception as e:
                    execution_time = time.time() - start_time
                    success = False
                    print(f"  Error en {size_name}: {e}")
                
                metrics.stop_monitoring()
                
                level_results[size_name] = {
                    'time': execution_time,
                    'success': success,
                    'memory_mb': metrics.get_peak_memory_mb(),
                    'cpu_avg': metrics.get_average_cpu_usage()
                }
                
                status = "✓" if success else "✗"
                print(f"  {size_name:8} {status} {execution_time:6.2f}s "
                      f"{metrics.get_peak_memory_mb():6.1f}MB "
                      f"{metrics.get_average_cpu_usage():5.1f}%")
            
            results[level] = level_results
        
        # Resumen comparativo
        print(f"\n{'='*60}")
        print("RESUMEN COMPARATIVO")
        print(f"{'='*60}")
        
        print(f"{'Nivel':<10} {'Tamaño':<8} {'Tiempo':<8} {'Memoria':<8} {'CPU':<6}")
        print("-" * 50)
        
        for level in levels:
            for size_name in ["small", "medium", "large"]:
                if size_name in results[level] and results[level][size_name]['success']:
                    r = results[level][size_name]
                    print(f"{level:<10} {size_name:<8} {r['time']:6.2f}s "
                          f"{r['memory_mb']:6.1f}MB {r['cpu_avg']:5.1f}%")
        
        return results
    
    def test_benchmark_predefined_configs(self):
        """Benchmark de configuraciones predefinidas"""
        from core.pipeline_config import list_predefined_configs, get_predefined_config
        
        predefined_names = list_predefined_configs()
        
        print(f"\n{'='*60}")
        print("BENCHMARK CONFIGURACIONES PREDEFINIDAS")
        print(f"{'='*60}")
        
        for config_name in predefined_names:
            print(f"\nConfiguración: {config_name}")
            print("-" * 40)
            
            try:
                config = get_predefined_config(config_name)
                pipeline = ScientificPipeline(config)
                
                # Test con imagen mediana
                path1, path2 = self.benchmark_images["medium"]
                
                metrics = PerformanceMetrics()
                metrics.start_monitoring()
                
                start_time = time.time()
                result = pipeline.analyze_images(path1, path2)
                execution_time = time.time() - start_time
                
                metrics.stop_monitoring()
                
                print(f"  Tiempo: {execution_time:.2f}s")
                print(f"  Memoria: {metrics.get_peak_memory_mb():.1f}MB")
                print(f"  CPU: {metrics.get_average_cpu_usage():.1f}%")
                print(f"  Conclusión: {result.afte_conclusion.value}")
                
            except Exception as e:
                print(f"  Error: {e}")
    
    def test_stress_test(self):
        """Test de estrés del pipeline"""
        print(f"\n{'='*60}")
        print("TEST DE ESTRÉS")
        print(f"{'='*60}")
        
        config = create_pipeline_config("basic")  # Usar configuración rápida
        
        # Test de múltiples ejecuciones consecutivas
        num_iterations = 5
        execution_times = []
        memory_usage = []
        
        print(f"\nEjecuciones consecutivas ({num_iterations} iteraciones):")
        
        for i in range(num_iterations):
            pipeline = ScientificPipeline(config)
            
            metrics = PerformanceMetrics()
            metrics.start_monitoring()
            
            start_time = time.time()
            try:
                result = pipeline.analyze_images(
                    self.benchmark_images["small"][0],
                    self.benchmark_images["small"][1]
                )
                execution_time = time.time() - start_time
                success = True
            except Exception as e:
                execution_time = time.time() - start_time
                success = False
                print(f"  Iteración {i+1}: Error - {e}")
                continue
            
            metrics.stop_monitoring()
            
            execution_times.append(execution_time)
            memory_usage.append(metrics.get_peak_memory_mb())
            
            print(f"  Iteración {i+1}: {execution_time:.2f}s, "
                  f"{metrics.get_peak_memory_mb():.1f}MB")
        
        if execution_times:
            avg_time = np.mean(execution_times)
            std_time = np.std(execution_times)
            avg_memory = np.mean(memory_usage)
            
            print(f"\nEstadísticas:")
            print(f"  Tiempo promedio: {avg_time:.2f}s ± {std_time:.2f}s")
            print(f"  Memoria promedio: {avg_memory:.1f}MB")
            print(f"  Variabilidad tiempo: {(std_time/avg_time)*100:.1f}%")
            
            # Verificar estabilidad (variabilidad < 20%)
            variability = (std_time / avg_time) * 100 if avg_time > 0 else 0
            self.assertLess(variability, 20.0, 
                          f"Pipeline inestable: variabilidad {variability:.1f}%")


if __name__ == '__main__':
    # Configurar logging
    import logging
    logging.basicConfig(
        level=logging.WARNING,  # Reducir ruido en benchmarks
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Ejecutar tests con verbosidad alta para ver benchmarks
    unittest.main(verbosity=2)