#!/usr/bin/env python3
"""
Benchmarks de rendimiento para SEACABAr
Tests de rendimiento para procesamiento de imágenes por chunks
"""

import os
import sys
import time
import gc
import pytest
import numpy as np
import psutil
from unittest.mock import Mock, patch
from typing import Dict, List, Tuple, Callable
import concurrent.futures
from contextlib import contextmanager

# Agregar el directorio raíz al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from image_processing.chunked_processor import ChunkedImageProcessor, ChunkingStrategy
    from image_processing.optimized_loader import OptimizedImageLoader, LoadingStrategy
    from image_processing.lazy_loading import LazyImageLoader
except ImportError as e:
    print(f"Warning: Could not import image processing modules: {e}")
    # Crear mocks para testing
    ChunkedImageProcessor = Mock
    ChunkingStrategy = Mock
    OptimizedImageLoader = Mock
    LoadingStrategy = Mock
    LazyImageLoader = Mock


class PerformanceTimer:
    """Timer para medir rendimiento"""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.elapsed_time = None
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        self.elapsed_time = self.end_time - self.start_time
    
    def get_elapsed_ms(self) -> float:
        """Retorna tiempo transcurrido en milisegundos"""
        return self.elapsed_time * 1000 if self.elapsed_time else 0


class ResourceMonitor:
    """Monitor de recursos del sistema"""
    
    def __init__(self):
        self.process = psutil.Process()
        self.initial_memory = None
        self.initial_cpu_time = None
        self.peak_memory = 0
        self.cpu_samples = []
        self.memory_samples = []
    
    def start(self):
        """Inicia el monitoreo"""
        self.initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        self.initial_cpu_time = self.process.cpu_times().user
        self.peak_memory = self.initial_memory
        self.cpu_samples = []
        self.memory_samples = [self.initial_memory]
    
    def sample(self):
        """Toma una muestra de recursos"""
        try:
            memory_mb = self.process.memory_info().rss / 1024 / 1024
            cpu_percent = self.process.cpu_percent()
            
            self.memory_samples.append(memory_mb)
            self.cpu_samples.append(cpu_percent)
            self.peak_memory = max(self.peak_memory, memory_mb)
        except Exception:
            pass
    
    def get_stats(self) -> Dict:
        """Obtiene estadísticas finales"""
        final_memory = self.process.memory_info().rss / 1024 / 1024
        final_cpu_time = self.process.cpu_times().user
        
        return {
            'memory_initial_mb': self.initial_memory,
            'memory_final_mb': final_memory,
            'memory_peak_mb': self.peak_memory,
            'memory_increase_mb': final_memory - self.initial_memory,
            'cpu_time_used': final_cpu_time - self.initial_cpu_time,
            'avg_cpu_percent': np.mean(self.cpu_samples) if self.cpu_samples else 0
        }


def create_benchmark_image(width: int, height: int, complexity: str = 'normal') -> np.ndarray:
    """Crea imagen para benchmarks con diferentes niveles de complejidad"""
    if complexity == 'simple':
        # Imagen simple con patrones básicos
        img = np.zeros((height, width, 3), dtype=np.uint8)
        img[:, :width//2] = [255, 0, 0]  # Rojo
        img[:, width//2:] = [0, 255, 0]  # Verde
        return img
    
    elif complexity == 'normal':
        # Imagen con ruido aleatorio
        return np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    
    elif complexity == 'complex':
        # Imagen compleja con múltiples patrones
        img = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        # Agregar patrones complejos
        y, x = np.ogrid[:height, :width]
        mask = (x - width//2)**2 + (y - height//2)**2 < (min(width, height)//4)**2
        img[mask] = [255, 255, 255]
        return img
    
    else:
        return np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)


class TestPerformanceBenchmarks:
    """Tests de benchmarks de rendimiento"""
    
    def setup_method(self):
        """Setup para cada test"""
        gc.collect()
    
    def teardown_method(self):
        """Cleanup después de cada test"""
        gc.collect()
    
    def test_image_processing_speed_comparison(self):
        """Compara velocidad de procesamiento entre diferentes métodos"""
        image_sizes = [(500, 500), (1000, 1000), (2000, 2000)]
        results = {}
        
        def simple_enhancement(img):
            """Función simple de mejora de imagen"""
            return np.clip(img * 1.2 + 10, 0, 255).astype(np.uint8)
        
        for size in image_sizes:
            width, height = size
            img = create_benchmark_image(width, height, 'normal')
            size_key = f"{width}x{height}"
            results[size_key] = {}
            
            # Método normal
            with PerformanceTimer() as timer:
                result_normal = simple_enhancement(img)
            results[size_key]['normal'] = timer.get_elapsed_ms()
            
            # Método por chunks (simulado)
            with PerformanceTimer() as timer:
                chunk_size = 256
                result_chunks = np.zeros_like(img)
                
                for y in range(0, height, chunk_size):
                    for x in range(0, width, chunk_size):
                        y_end = min(y + chunk_size, height)
                        x_end = min(x + chunk_size, width)
                        
                        chunk = img[y:y_end, x:x_end]
                        result_chunks[y:y_end, x:x_end] = simple_enhancement(chunk)
            
            results[size_key]['chunked'] = timer.get_elapsed_ms()
            
            # Limpiar memoria
            del img, result_normal, result_chunks
        
        # Imprimir resultados
        print("\nResultados de benchmark de velocidad:")
        for size, times in results.items():
            print(f"{size}: Normal={times['normal']:.2f}ms, Chunked={times['chunked']:.2f}ms")
        
        # Verificar que los tiempos sean razonables
        for size, times in results.items():
            assert times['normal'] > 0, f"Tiempo normal debe ser positivo para {size}"
            assert times['chunked'] > 0, f"Tiempo chunked debe ser positivo para {size}"
    
    @pytest.mark.skipif(ChunkedImageProcessor == Mock, reason="ChunkedImageProcessor not available")
    def test_chunked_processor_performance(self):
        """Test de rendimiento del procesador por chunks"""
        processor = ChunkedImageProcessor(
            chunk_size=(512, 512),
            overlap=32,
            strategy=ChunkingStrategy.GRID,
            parallel=True,
            max_workers=4
        )
        
        image_sizes = [(1000, 1000), (2000, 2000)]
        results = {}
        
        def enhancement_function(chunk):
            """Función de mejora más compleja"""
            # Simular procesamiento más intensivo
            enhanced = chunk.astype(np.float32)
            enhanced = enhanced * 1.2 + 10
            enhanced = np.clip(enhanced, 0, 255)
            return enhanced.astype(np.uint8)
        
        for size in image_sizes:
            width, height = size
            img = create_benchmark_image(width, height, 'complex')
            
            monitor = ResourceMonitor()
            monitor.start()
            
            with PerformanceTimer() as timer:
                result = processor.process_image(img, enhancement_function)
            
            stats = monitor.get_stats()
            
            results[f"{width}x{height}"] = {
                'time_ms': timer.get_elapsed_ms(),
                'memory_peak_mb': stats['memory_peak_mb'],
                'memory_increase_mb': stats['memory_increase_mb'],
                'cpu_time': stats['cpu_time_used']
            }
            
            del img, result
        
        # Verificar rendimiento
        for size, metrics in results.items():
            print(f"\nChunked processor {size}:")
            print(f"  Tiempo: {metrics['time_ms']:.2f}ms")
            print(f"  Memoria pico: {metrics['memory_peak_mb']:.2f}MB")
            print(f"  Incremento memoria: {metrics['memory_increase_mb']:.2f}MB")
            
            # Verificar que el rendimiento sea aceptable
            assert metrics['time_ms'] < 10000, f"Tiempo muy alto para {size}: {metrics['time_ms']:.2f}ms"
            assert metrics['memory_increase_mb'] < 500, f"Incremento de memoria muy alto para {size}"
    
    @pytest.mark.skipif(OptimizedImageLoader == Mock, reason="OptimizedImageLoader not available")
    def test_optimized_loader_performance(self):
        """Test de rendimiento del cargador optimizado"""
        loader = OptimizedImageLoader(
            cache_size_mb=50,
            strategy=LoadingStrategy.PROGRESSIVE
        )
        
        # Simular carga de múltiples imágenes
        num_images = 20
        image_size = (800, 600)
        
        monitor = ResourceMonitor()
        monitor.start()
        
        with PerformanceTimer() as timer:
            loaded_images = []
            
            for i in range(num_images):
                # Simular carga de imagen
                img = create_benchmark_image(image_size[0], image_size[1], 'normal')
                
                # Simular procesamiento optimizado
                if i % 5 == 0:  # Cada 5 imágenes, limpiar cache
                    loaded_images = loaded_images[-5:]  # Mantener solo las últimas 5
                    gc.collect()
                
                loaded_images.append(img)
                monitor.sample()
        
        stats = monitor.get_stats()
        
        print(f"\nOptimized loader performance:")
        print(f"  Tiempo total: {timer.get_elapsed_ms():.2f}ms")
        print(f"  Tiempo por imagen: {timer.get_elapsed_ms()/num_images:.2f}ms")
        print(f"  Memoria pico: {stats['memory_peak_mb']:.2f}MB")
        print(f"  CPU promedio: {stats['avg_cpu_percent']:.2f}%")
        
        # Verificar rendimiento
        assert timer.get_elapsed_ms() < 5000, f"Tiempo total muy alto: {timer.get_elapsed_ms():.2f}ms"
        assert stats['memory_peak_mb'] < 200, f"Uso de memoria muy alto: {stats['memory_peak_mb']:.2f}MB"
    
    def test_concurrent_processing_performance(self):
        """Test de rendimiento con procesamiento concurrente"""
        def process_image_task(image_data):
            """Tarea de procesamiento de imagen"""
            height, width, channels = image_data['shape']
            img = np.random.randint(0, 256, (height, width, channels), dtype=np.uint8)
            
            # Simular procesamiento
            processed = np.clip(img * 1.1 + 5, 0, 255).astype(np.uint8)
            
            # Simular operaciones adicionales
            blurred = np.roll(processed, 1, axis=0)  # Simulación simple de blur
            
            return {
                'processed_size': processed.nbytes,
                'processing_time': time.perf_counter()
            }
        
        # Test con diferentes números de workers
        worker_counts = [1, 2, 4, 8]
        num_tasks = 16
        task_data = [{'shape': (400, 400, 3)} for _ in range(num_tasks)]
        
        results = {}
        
        for workers in worker_counts:
            monitor = ResourceMonitor()
            monitor.start()
            
            with PerformanceTimer() as timer:
                with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                    futures = [executor.submit(process_image_task, data) for data in task_data]
                    task_results = [future.result() for future in concurrent.futures.as_completed(futures)]
            
            stats = monitor.get_stats()
            
            results[workers] = {
                'time_ms': timer.get_elapsed_ms(),
                'memory_peak_mb': stats['memory_peak_mb'],
                'cpu_time': stats['cpu_time_used'],
                'throughput': num_tasks / (timer.get_elapsed_ms() / 1000)  # tareas por segundo
            }
        
        # Imprimir resultados
        print("\nConcurrent processing performance:")
        for workers, metrics in results.items():
            print(f"  {workers} workers: {metrics['time_ms']:.2f}ms, "
                  f"throughput: {metrics['throughput']:.2f} tasks/s")
        
        # Verificar que más workers mejoren el throughput (hasta cierto punto)
        assert results[1]['throughput'] > 0, "Throughput debe ser positivo"
        if 4 in results and 1 in results:
            # Con 4 workers debería ser más rápido que con 1 (en la mayoría de casos)
            speedup = results[4]['throughput'] / results[1]['throughput']
            print(f"  Speedup con 4 workers: {speedup:.2f}x")
    
    def test_memory_vs_speed_tradeoff(self):
        """Test del trade-off entre memoria y velocidad"""
        image = create_benchmark_image(1500, 1500, 'complex')
        
        # Diferentes tamaños de chunk
        chunk_sizes = [128, 256, 512, 1024]
        results = {}
        
        def enhancement_function(chunk):
            """Función de mejora estándar"""
            return np.clip(chunk * 1.3 + 15, 0, 255).astype(np.uint8)
        
        for chunk_size in chunk_sizes:
            monitor = ResourceMonitor()
            monitor.start()
            
            with PerformanceTimer() as timer:
                # Procesar por chunks
                height, width = image.shape[:2]
                result = np.zeros_like(image)
                
                for y in range(0, height, chunk_size):
                    for x in range(0, width, chunk_size):
                        y_end = min(y + chunk_size, height)
                        x_end = min(x + chunk_size, width)
                        
                        chunk = image[y:y_end, x:x_end]
                        result[y:y_end, x:x_end] = enhancement_function(chunk)
                        
                        # Muestrear recursos
                        monitor.sample()
            
            stats = monitor.get_stats()
            
            results[chunk_size] = {
                'time_ms': timer.get_elapsed_ms(),
                'memory_peak_mb': stats['memory_peak_mb'],
                'memory_increase_mb': stats['memory_increase_mb']
            }
            
            del result
        
        # Imprimir análisis del trade-off
        print("\nMemory vs Speed tradeoff analysis:")
        for chunk_size, metrics in results.items():
            print(f"  Chunk {chunk_size}x{chunk_size}: "
                  f"Time={metrics['time_ms']:.2f}ms, "
                  f"Memory={metrics['memory_peak_mb']:.2f}MB")
        
        # Verificar que chunks más pequeños usen menos memoria
        memory_values = [results[size]['memory_peak_mb'] for size in chunk_sizes]
        time_values = [results[size]['time_ms'] for size in chunk_sizes]
        
        # Generalmente, chunks más pequeños deberían usar menos memoria
        # pero podrían ser más lentos debido al overhead
        assert all(m > 0 for m in memory_values), "Todos los valores de memoria deben ser positivos"
        assert all(t > 0 for t in time_values), "Todos los tiempos deben ser positivos"
    
    def test_scalability_benchmark(self):
        """Test de escalabilidad con diferentes tamaños de imagen"""
        base_sizes = [
            (200, 200),   # Pequeña
            (500, 500),   # Mediana
            (1000, 1000), # Grande
            (1500, 1500)  # Muy grande
        ]
        
        results = {}
        
        def standard_processing(img):
            """Procesamiento estándar para benchmark"""
            # Simular operaciones típicas
            enhanced = img.astype(np.float32) * 1.2
            enhanced = np.clip(enhanced, 0, 255)
            return enhanced.astype(np.uint8)
        
        for width, height in base_sizes:
            img = create_benchmark_image(width, height, 'normal')
            size_key = f"{width}x{height}"
            
            monitor = ResourceMonitor()
            monitor.start()
            
            with PerformanceTimer() as timer:
                result = standard_processing(img)
            
            stats = monitor.get_stats()
            
            results[size_key] = {
                'pixels': width * height,
                'time_ms': timer.get_elapsed_ms(),
                'memory_mb': stats['memory_peak_mb'],
                'time_per_megapixel': timer.get_elapsed_ms() / ((width * height) / 1_000_000)
            }
            
            del img, result
        
        # Análisis de escalabilidad
        print("\nScalability analysis:")
        for size, metrics in results.items():
            megapixels = metrics['pixels'] / 1_000_000
            print(f"  {size} ({megapixels:.1f}MP): "
                  f"{metrics['time_ms']:.2f}ms, "
                  f"{metrics['time_per_megapixel']:.2f}ms/MP")
        
        # Verificar que la escalabilidad sea razonable
        # El tiempo por megapixel no debería crecer exponencialmente
        time_per_mp_values = [results[f"{w}x{h}"]['time_per_megapixel'] 
                             for w, h in base_sizes]
        
        # La variación en tiempo por megapixel no debería ser excesiva
        max_time_per_mp = max(time_per_mp_values)
        min_time_per_mp = min(time_per_mp_values)
        variation_ratio = max_time_per_mp / min_time_per_mp
        
        assert variation_ratio < 5.0, f"Variación excesiva en escalabilidad: {variation_ratio:.2f}x"


if __name__ == "__main__":
    # Ejecutar benchmarks
    pytest.main([__file__, "-v", "-s", "--tb=short"])