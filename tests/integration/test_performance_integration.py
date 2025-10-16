#!/usr/bin/env python3
"""
Tests de Integración de Performance para el Sistema SIGeC-Balistica
===================================================================

Este módulo contiene tests específicos para validar el rendimiento
y la eficiencia del sistema completo bajo diferentes cargas y condiciones.

Autor: SIGeC-Balistica Team
Fecha: 2024
"""

import pytest
import time
import threading
import multiprocessing
import psutil
import gc
import tempfile
import shutil
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from unittest.mock import Mock, patch
from typing import Dict, Any, List, Tuple
import json
import statistics

# Imports del sistema
try:
    from core.unified_pipeline import UnifiedPipeline
    from performance.enhanced_monitoring_system import PerformanceMonitor
    from utils.dependency_manager import DependencyManager
    from common.test_helpers import TestImageGenerator
    from database.database_manager import DatabaseManager
    from config.unified_config import UnifiedConfig
except ImportError as e:
    pytest.skip(f"Dependencias no disponibles: {e}", allow_module_level=True)

class TestPerformanceIntegration:
    """Tests de integración de performance"""
    
    @pytest.fixture(autouse=True)
    def setup_performance_test_environment(self):
        """Configura entorno para tests de performance"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.perf_test_dir = self.temp_dir / "performance_tests"
        self.perf_test_dir.mkdir(exist_ok=True)
        
        # Crear imágenes de prueba de diferentes tamaños
        self.image_generator = TestImageGenerator()
        self._create_performance_test_images()
        
        # Configuración optimizada para performance
        self.performance_config = self._create_performance_config()
        
        # Métricas de baseline
        self.baseline_metrics = {}
        
        yield
        
        # Limpieza
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_performance_test_images(self):
        """Crea imágenes de diferentes tamaños para tests de performance"""
        self.test_images = {
            'small': [],      # 256x192
            'medium': [],     # 512x384
            'large': [],      # 1024x768
            'xlarge': []      # 2048x1536
        }
        
        sizes = {
            'small': (256, 192),
            'medium': (512, 384),
            'large': (1024, 768),
            'xlarge': (2048, 1536)
        }
        
        features = ['striations', 'firing_pin', 'breech_face']
        
        for size_name, (width, height) in sizes.items():
            for i in range(5):  # 5 imágenes por tamaño
                img_path = self.perf_test_dir / f"{size_name}_{i}.jpg"
                image = self.image_generator.create_ballistic_image(
                    width=width,
                    height=height,
                    features=features,
                    noise_level=0.1
                )
                self.image_generator.save_image(image, img_path)
                self.test_images[size_name].append(str(img_path))
    
    def _create_performance_config(self) -> Dict[str, Any]:
        """Crea configuración optimizada para performance"""
        return {
            'pipeline': {
                'enable_all': True,
                'optimization_level': 'high'
            },
            'performance': {
                'enable_monitoring': True,
                'enable_profiling': True,
                'enable_caching': True,
                'cache_size': 1000,
                'enable_gpu_acceleration': True,
                'parallel_processing': True,
                'memory_optimization': True,
                'batch_processing': True
            },
            'preprocessing': {
                'optimization_level': 'high',
                'parallel_filters': True,
                'memory_efficient': True
            },
            'matching': {
                'optimization_level': 'high',
                'parallel_matching': True,
                'gpu_acceleration': True
            },
            'deep_learning': {
                'batch_inference': True,
                'model_optimization': True,
                'memory_efficient': True
            },
            'database': {
                'connection_pooling': True,
                'query_optimization': True,
                'indexing': True
            }
        }
    
    def test_processing_time_scalability(self):
        """Test de escalabilidad del tiempo de procesamiento"""
        pipeline = UnifiedPipeline(self.performance_config)
        
        processing_times = {}
        
        # Medir tiempo para diferentes tamaños de imagen
        for size_name, images in self.test_images.items():
            if len(images) >= 2:
                times = []
                
                # Ejecutar múltiples veces para obtener promedio
                for i in range(3):
                    start_time = time.time()
                    result = pipeline.process_images(images[0], images[1])
                    end_time = time.time()
                    
                    assert result.success, f"Procesamiento falló para tamaño {size_name}"
                    times.append(end_time - start_time)
                
                processing_times[size_name] = {
                    'mean': statistics.mean(times),
                    'std': statistics.stdev(times) if len(times) > 1 else 0,
                    'min': min(times),
                    'max': max(times)
                }
        
        # Validar escalabilidad
        size_order = ['small', 'medium', 'large', 'xlarge']
        mean_times = [processing_times[size]['mean'] for size in size_order if size in processing_times]
        
        # El tiempo debería aumentar con el tamaño, pero no linealmente
        for i in range(1, len(mean_times)):
            ratio = mean_times[i] / mean_times[i-1]
            assert ratio < 4.0, f"Escalabilidad pobre: ratio {ratio:.2f} entre {size_order[i]} y {size_order[i-1]}"
        
        print("✅ Escalabilidad del tiempo de procesamiento validada")
        for size, metrics in processing_times.items():
            print(f"   {size}: {metrics['mean']:.3f}s ± {metrics['std']:.3f}s")
    
    def test_memory_usage_efficiency(self):
        """Test de eficiencia en el uso de memoria"""
        process = psutil.Process()
        
        # Memoria inicial
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        pipeline = UnifiedPipeline(self.performance_config)
        
        memory_usage = {}
        
        # Procesar diferentes tamaños y medir memoria
        for size_name, images in self.test_images.items():
            if len(images) >= 2:
                # Limpiar memoria antes de cada test
                gc.collect()
                
                pre_memory = process.memory_info().rss / 1024 / 1024
                
                result = pipeline.process_images(images[0], images[1])
                
                post_memory = process.memory_info().rss / 1024 / 1024
                memory_increase = post_memory - pre_memory
                
                memory_usage[size_name] = {
                    'increase': memory_increase,
                    'peak': post_memory,
                    'success': result.success
                }
                
                assert result.success, f"Procesamiento falló para {size_name}"
        
        # Validar eficiencia de memoria
        for size_name, usage in memory_usage.items():
            # La memoria no debería aumentar excesivamente
            if size_name == 'small':
                assert usage['increase'] < 100, f"Uso excesivo de memoria para {size_name}: {usage['increase']:.1f}MB"
            elif size_name == 'medium':
                assert usage['increase'] < 200, f"Uso excesivo de memoria para {size_name}: {usage['increase']:.1f}MB"
            elif size_name == 'large':
                assert usage['increase'] < 400, f"Uso excesivo de memoria para {size_name}: {usage['increase']:.1f}MB"
            elif size_name == 'xlarge':
                assert usage['increase'] < 800, f"Uso excesivo de memoria para {size_name}: {usage['increase']:.1f}MB"
        
        # Limpiar y verificar liberación de memoria
        del pipeline
        gc.collect()
        
        final_memory = process.memory_info().rss / 1024 / 1024
        memory_cleanup = max(memory_usage.values(), key=lambda x: x['peak'])['peak'] - final_memory
        
        assert memory_cleanup > 0, "No se liberó memoria tras limpieza"
        
        print("✅ Eficiencia en el uso de memoria validada")
        for size, usage in memory_usage.items():
            print(f"   {size}: +{usage['increase']:.1f}MB")
        print(f"   Limpieza: -{memory_cleanup:.1f}MB")
    
    def test_concurrent_processing_performance(self):
        """Test de performance con procesamiento concurrente"""
        # Crear múltiples pares de imágenes
        image_pairs = []
        for size_name, images in self.test_images.items():
            if len(images) >= 4:  # Necesitamos al menos 4 para crear 2 pares
                image_pairs.extend([
                    (images[0], images[1]),
                    (images[2], images[3])
                ])
        
        # Limitar a 10 pares para el test
        image_pairs = image_pairs[:10]
        
        config = self.performance_config.copy()
        config['performance']['parallel_processing'] = True
        
        # Test secuencial
        start_time = time.time()
        sequential_results = []
        
        for img1, img2 in image_pairs:
            pipeline = UnifiedPipeline(config)
            result = pipeline.process_images(img1, img2)
            sequential_results.append(result)
        
        sequential_time = time.time() - start_time
        
        # Test concurrente
        def process_pair(pair):
            pipeline = UnifiedPipeline(config)
            return pipeline.process_images(pair[0], pair[1])
        
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            concurrent_results = list(executor.map(process_pair, image_pairs))
        
        concurrent_time = time.time() - start_time
        
        # Validar resultados
        assert len(sequential_results) == len(image_pairs), "Resultados secuenciales incompletos"
        assert len(concurrent_results) == len(image_pairs), "Resultados concurrentes incompletos"
        
        # Todos los procesamientos deberían ser exitosos
        sequential_success = sum(1 for r in sequential_results if r.success)
        concurrent_success = sum(1 for r in concurrent_results if r.success)
        
        assert sequential_success == len(image_pairs), "Fallos en procesamiento secuencial"
        assert concurrent_success == len(image_pairs), "Fallos en procesamiento concurrente"
        
        # El procesamiento concurrente debería ser más rápido
        speedup = sequential_time / concurrent_time
        assert speedup > 1.5, f"Speedup insuficiente: {speedup:.2f}x"
        
        print("✅ Performance de procesamiento concurrente validada")
        print(f"   Tiempo secuencial: {sequential_time:.2f}s")
        print(f"   Tiempo concurrente: {concurrent_time:.2f}s")
        print(f"   Speedup: {speedup:.2f}x")
    
    def test_caching_performance_impact(self):
        """Test del impacto de caching en performance"""
        # Configuración sin cache
        config_no_cache = self.performance_config.copy()
        config_no_cache['performance']['enable_caching'] = False
        
        # Configuración con cache
        config_with_cache = self.performance_config.copy()
        config_with_cache['performance']['enable_caching'] = True
        config_with_cache['performance']['cache_size'] = 100
        
        # Usar las mismas imágenes múltiples veces
        test_image1 = self.test_images['medium'][0]
        test_image2 = self.test_images['medium'][1]
        
        # Test sin cache
        pipeline_no_cache = UnifiedPipeline(config_no_cache)
        
        no_cache_times = []
        for i in range(5):
            start_time = time.time()
            result = pipeline_no_cache.process_images(test_image1, test_image2)
            end_time = time.time()
            
            assert result.success, f"Procesamiento sin cache falló en iteración {i}"
            no_cache_times.append(end_time - start_time)
        
        # Test con cache
        pipeline_with_cache = UnifiedPipeline(config_with_cache)
        
        with_cache_times = []
        for i in range(5):
            start_time = time.time()
            result = pipeline_with_cache.process_images(test_image1, test_image2)
            end_time = time.time()
            
            assert result.success, f"Procesamiento con cache falló en iteración {i}"
            with_cache_times.append(end_time - start_time)
        
        # Analizar impacto del cache
        avg_no_cache = statistics.mean(no_cache_times)
        avg_with_cache = statistics.mean(with_cache_times)
        
        # El cache debería mejorar el rendimiento en ejecuciones posteriores
        cache_improvement = (avg_no_cache - avg_with_cache) / avg_no_cache
        
        # Al menos 10% de mejora esperada
        assert cache_improvement > 0.1, f"Mejora de cache insuficiente: {cache_improvement:.2%}"
        
        print("✅ Impacto de caching en performance validado")
        print(f"   Sin cache: {avg_no_cache:.3f}s")
        print(f"   Con cache: {avg_with_cache:.3f}s")
        print(f"   Mejora: {cache_improvement:.2%}")
    
    def test_gpu_acceleration_performance(self):
        """Test de performance con aceleración GPU"""
        if not DependencyManager.is_available('torch') or not DependencyManager.has_gpu():
            pytest.skip("GPU o PyTorch no disponible")
        
        # Configuración CPU
        config_cpu = self.performance_config.copy()
        config_cpu['performance']['enable_gpu_acceleration'] = False
        config_cpu['deep_learning']['device'] = 'cpu'
        config_cpu['matching']['gpu_acceleration'] = False
        
        # Configuración GPU
        config_gpu = self.performance_config.copy()
        config_gpu['performance']['enable_gpu_acceleration'] = True
        config_gpu['deep_learning']['device'] = 'cuda'
        config_gpu['matching']['gpu_acceleration'] = True
        
        # Usar imágenes grandes para maximizar beneficio de GPU
        test_image1 = self.test_images['large'][0]
        test_image2 = self.test_images['large'][1]
        
        # Test CPU
        pipeline_cpu = UnifiedPipeline(config_cpu)
        
        cpu_times = []
        for i in range(3):
            start_time = time.time()
            result = pipeline_cpu.process_images(test_image1, test_image2)
            end_time = time.time()
            
            assert result.success, f"Procesamiento CPU falló en iteración {i}"
            cpu_times.append(end_time - start_time)
        
        # Test GPU
        pipeline_gpu = UnifiedPipeline(config_gpu)
        
        gpu_times = []
        for i in range(3):
            start_time = time.time()
            result = pipeline_gpu.process_images(test_image1, test_image2)
            end_time = time.time()
            
            assert result.success, f"Procesamiento GPU falló en iteración {i}"
            gpu_times.append(end_time - start_time)
        
        # Analizar speedup de GPU
        avg_cpu_time = statistics.mean(cpu_times)
        avg_gpu_time = statistics.mean(gpu_times)
        
        gpu_speedup = avg_cpu_time / avg_gpu_time
        
        # GPU debería ser al menos 20% más rápida
        assert gpu_speedup > 1.2, f"Speedup GPU insuficiente: {gpu_speedup:.2f}x"
        
        print("✅ Performance con aceleración GPU validada")
        print(f"   Tiempo CPU: {avg_cpu_time:.3f}s")
        print(f"   Tiempo GPU: {avg_gpu_time:.3f}s")
        print(f"   Speedup GPU: {gpu_speedup:.2f}x")
    
    def test_database_performance_integration(self):
        """Test de performance de integración con base de datos"""
        config = self.performance_config.copy()
        config['database'] = {
            'enable': True,
            'connection_string': f'sqlite:///{self.temp_dir}/perf_test.db',
            'connection_pooling': True,
            'batch_operations': True,
            'indexing': True
        }
        
        pipeline = UnifiedPipeline(config)
        db_manager = DatabaseManager(config['database'])
        
        # Procesar múltiples pares y almacenar
        image_pairs = [
            (self.test_images['small'][0], self.test_images['small'][1]),
            (self.test_images['medium'][0], self.test_images['medium'][1]),
            (self.test_images['large'][0], self.test_images['large'][1])
        ]
        
        # Medir tiempo de procesamiento + almacenamiento
        start_time = time.time()
        
        case_ids = []
        for img1, img2 in image_pairs:
            result = pipeline.process_images(img1, img2)
            assert result.success, "Procesamiento falló"
            
            # Almacenar en BD
            case_id = db_manager.store_case_result(result)
            case_ids.append(case_id)
        
        storage_time = time.time() - start_time
        
        # Medir tiempo de consultas
        start_time = time.time()
        
        for case_id in case_ids:
            retrieved_case = db_manager.get_case(case_id)
            assert retrieved_case is not None, f"Caso {case_id} no recuperado"
        
        retrieval_time = time.time() - start_time
        
        # Medir tiempo de búsqueda por similitud
        start_time = time.time()
        
        similar_cases = db_manager.find_similar_cases(
            similarity_threshold=0.5,
            limit=10
        )
        
        search_time = time.time() - start_time
        
        # Validar performance de BD
        assert storage_time < 10.0, f"Almacenamiento muy lento: {storage_time:.2f}s"
        assert retrieval_time < 2.0, f"Recuperación muy lenta: {retrieval_time:.2f}s"
        assert search_time < 5.0, f"Búsqueda muy lenta: {search_time:.2f}s"
        
        print("✅ Performance de integración con BD validada")
        print(f"   Almacenamiento: {storage_time:.3f}s")
        print(f"   Recuperación: {retrieval_time:.3f}s")
        print(f"   Búsqueda: {search_time:.3f}s")
    
    def test_monitoring_system_overhead(self):
        """Test del overhead del sistema de monitoreo"""
        # Configuración sin monitoreo
        config_no_monitor = self.performance_config.copy()
        config_no_monitor['performance']['enable_monitoring'] = False
        config_no_monitor['performance']['enable_profiling'] = False
        
        # Configuración con monitoreo
        config_with_monitor = self.performance_config.copy()
        config_with_monitor['performance']['enable_monitoring'] = True
        config_with_monitor['performance']['enable_profiling'] = True
        
        test_image1 = self.test_images['medium'][0]
        test_image2 = self.test_images['medium'][1]
        
        # Test sin monitoreo
        pipeline_no_monitor = UnifiedPipeline(config_no_monitor)
        
        no_monitor_times = []
        for i in range(5):
            start_time = time.time()
            result = pipeline_no_monitor.process_images(test_image1, test_image2)
            end_time = time.time()
            
            assert result.success, f"Procesamiento sin monitoreo falló en iteración {i}"
            no_monitor_times.append(end_time - start_time)
        
        # Test con monitoreo
        with PerformanceMonitor() as monitor:
            pipeline_with_monitor = UnifiedPipeline(config_with_monitor)
            
            with_monitor_times = []
            for i in range(5):
                start_time = time.time()
                result = pipeline_with_monitor.process_images(test_image1, test_image2)
                end_time = time.time()
                
                assert result.success, f"Procesamiento con monitoreo falló en iteración {i}"
                with_monitor_times.append(end_time - start_time)
        
        # Analizar overhead
        avg_no_monitor = statistics.mean(no_monitor_times)
        avg_with_monitor = statistics.mean(with_monitor_times)
        
        overhead = (avg_with_monitor - avg_no_monitor) / avg_no_monitor
        
        # El overhead debería ser mínimo (< 10%)
        assert overhead < 0.1, f"Overhead de monitoreo muy alto: {overhead:.2%}"
        
        # Verificar que se recolectaron métricas
        metrics = monitor.get_metrics()
        assert 'total_processing_time' in metrics, "Métricas no recolectadas"
        assert 'memory_usage' in metrics, "Uso de memoria no monitoreado"
        
        print("✅ Overhead del sistema de monitoreo validado")
        print(f"   Sin monitoreo: {avg_no_monitor:.3f}s")
        print(f"   Con monitoreo: {avg_with_monitor:.3f}s")
        print(f"   Overhead: {overhead:.2%}")
    
    def test_stress_testing(self):
        """Test de estrés del sistema"""
        config = self.performance_config.copy()
        config['performance']['stress_test_mode'] = True
        
        # Crear carga de trabajo intensa
        stress_pairs = []
        for _ in range(20):  # 20 pares para stress test
            # Usar imágenes de diferentes tamaños aleatoriamente
            size1 = np.random.choice(list(self.test_images.keys()))
            size2 = np.random.choice(list(self.test_images.keys()))
            
            img1 = np.random.choice(self.test_images[size1])
            img2 = np.random.choice(self.test_images[size2])
            
            stress_pairs.append((img1, img2))
        
        # Monitorear recursos durante stress test
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024
        
        start_time = time.time()
        successful_processes = 0
        failed_processes = 0
        
        # Ejecutar stress test con múltiples threads
        def stress_worker(pair):
            try:
                pipeline = UnifiedPipeline(config)
                result = pipeline.process_images(pair[0], pair[1])
                return result.success
            except Exception as e:
                print(f"Error en stress worker: {e}")
                return False
        
        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = [executor.submit(stress_worker, pair) for pair in stress_pairs]
            
            for future in futures:
                try:
                    success = future.result(timeout=60)
                    if success:
                        successful_processes += 1
                    else:
                        failed_processes += 1
                except Exception:
                    failed_processes += 1
        
        total_time = time.time() - start_time
        peak_memory = process.memory_info().rss / 1024 / 1024
        memory_increase = peak_memory - initial_memory
        
        # Validar resultados del stress test
        success_rate = successful_processes / len(stress_pairs)
        
        assert success_rate > 0.8, f"Tasa de éxito muy baja en stress test: {success_rate:.2%}"
        assert total_time < 300, f"Stress test muy lento: {total_time:.1f}s"  # 5 minutos máximo
        assert memory_increase < 2000, f"Uso excesivo de memoria: {memory_increase:.1f}MB"
        
        print("✅ Stress testing validado")
        print(f"   Pares procesados: {len(stress_pairs)}")
        print(f"   Tasa de éxito: {success_rate:.2%}")
        print(f"   Tiempo total: {total_time:.1f}s")
        print(f"   Incremento memoria: {memory_increase:.1f}MB")
        print(f"   Throughput: {len(stress_pairs)/total_time:.2f} pares/s")

if __name__ == "__main__":
    # Ejecutar tests si se llama directamente
    pytest.main([__file__, "-v", "--tb=short"])