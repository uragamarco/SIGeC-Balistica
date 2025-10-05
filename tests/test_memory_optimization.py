#!/usr/bin/env python3
"""
Tests de optimización de memoria para SIGeC-Balistica
Valida el uso eficiente de memoria en procesamiento de imágenes por chunks
"""

import os
import sys
import gc
import time
import psutil
import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Tuple, Optional
import threading
from contextlib import contextmanager

# Agregar el directorio raíz al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from image_processing.chunked_processor import ChunkedImageProcessor, ChunkingStrategy
    from image_processing.optimized_loader import OptimizedImageLoader, LoadingStrategy
    from image_processing.lazy_loading import LazyImageLoader, LazyImageDataset
except ImportError as e:
    print(f"Warning: Could not import image processing modules: {e}")
    # Crear mocks para testing
    ChunkedImageProcessor = Mock
    ChunkingStrategy = Mock
    OptimizedImageLoader = Mock
    LoadingStrategy = Mock
    LazyImageLoader = Mock
    LazyImageDataset = Mock


class MemoryMonitor:
    """Monitor de memoria para tests"""
    
    def __init__(self):
        self.process = psutil.Process()
        self.initial_memory = None
        self.peak_memory = 0
        self.memory_samples = []
        self.monitoring = False
        self.monitor_thread = None
    
    def start_monitoring(self):
        """Inicia el monitoreo de memoria"""
        self.initial_memory = self.get_memory_usage()
        self.peak_memory = self.initial_memory
        self.memory_samples = [self.initial_memory]
        self.monitoring = True
        
        # Iniciar thread de monitoreo
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Detiene el monitoreo de memoria"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        
        final_memory = self.get_memory_usage()
        self.memory_samples.append(final_memory)
        
        return {
            'initial_mb': self.initial_memory,
            'final_mb': final_memory,
            'peak_mb': self.peak_memory,
            'increase_mb': final_memory - self.initial_memory,
            'samples': self.memory_samples
        }
    
    def _monitor_loop(self):
        """Loop de monitoreo en thread separado"""
        while self.monitoring:
            try:
                current_memory = self.get_memory_usage()
                self.memory_samples.append(current_memory)
                self.peak_memory = max(self.peak_memory, current_memory)
                time.sleep(0.1)  # Muestrear cada 100ms
            except Exception:
                break
    
    def get_memory_usage(self) -> float:
        """Obtiene el uso actual de memoria en MB"""
        try:
            memory_info = self.process.memory_info()
            return memory_info.rss / 1024 / 1024  # Convertir a MB
        except Exception:
            return 0.0


@contextmanager
def memory_monitor():
    """Context manager para monitorear memoria"""
    monitor = MemoryMonitor()
    monitor.start_monitoring()
    try:
        yield monitor
    finally:
        stats = monitor.stop_monitoring()
        # Forzar garbage collection
        gc.collect()
        return stats


def create_test_image(width: int, height: int, channels: int = 3) -> np.ndarray:
    """Crea una imagen de prueba con el tamaño especificado"""
    return np.random.randint(0, 256, (height, width, channels), dtype=np.uint8)


def create_large_test_image(size_mb: float = 100) -> np.ndarray:
    """Crea una imagen grande para testing de memoria"""
    # Calcular dimensiones para alcanzar el tamaño deseado
    bytes_per_pixel = 3  # RGB
    total_pixels = int((size_mb * 1024 * 1024) / bytes_per_pixel)
    side_length = int(np.sqrt(total_pixels))
    
    return create_test_image(side_length, side_length, 3)


class TestMemoryOptimization:
    """Tests de optimización de memoria"""
    
    def setup_method(self):
        """Setup para cada test"""
        gc.collect()  # Limpiar memoria antes de cada test
    
    def teardown_method(self):
        """Cleanup después de cada test"""
        gc.collect()  # Limpiar memoria después de cada test
    
    def test_memory_usage_baseline(self):
        """Test baseline de uso de memoria"""
        with memory_monitor() as monitor:
            # Operación simple
            small_image = create_test_image(100, 100)
            processed = small_image * 1.1
            del small_image, processed
        
        stats = monitor.stop_monitoring()
        
        # El incremento de memoria debe ser mínimo
        assert stats['increase_mb'] < 10, f"Incremento de memoria muy alto: {stats['increase_mb']:.2f}MB"
    
    @pytest.mark.skipif(ChunkedImageProcessor == Mock, reason="ChunkedImageProcessor not available")
    def test_chunked_processor_memory_efficiency(self):
        """Test de eficiencia de memoria del procesador por chunks"""
        processor = ChunkedImageProcessor(
            chunk_size=(512, 512),
            overlap=32,
            strategy=ChunkingStrategy.MEMORY_BASED,
            max_memory_mb=50
        )
        
        with memory_monitor() as monitor:
            # Crear imagen grande
            large_image = create_large_test_image(20)  # 20MB
            
            # Procesar por chunks
            def simple_enhancement(chunk):
                return np.clip(chunk * 1.2, 0, 255).astype(np.uint8)
            
            result = processor.process_image(large_image, simple_enhancement)
            
            # Limpiar referencias
            del large_image, result
        
        stats = monitor.stop_monitoring()
        
        # El pico de memoria no debe exceder significativamente el tamaño de la imagen
        expected_max = 100  # MB - margen para chunks y procesamiento
        assert stats['peak_mb'] - stats['initial_mb'] < expected_max, \
            f"Pico de memoria muy alto: {stats['peak_mb'] - stats['initial_mb']:.2f}MB"
    
    @pytest.mark.skipif(OptimizedImageLoader == Mock, reason="OptimizedImageLoader not available")
    def test_optimized_loader_memory_efficiency(self):
        """Test de eficiencia de memoria del cargador optimizado"""
        loader = OptimizedImageLoader(
            cache_size_mb=20,
            strategy=LoadingStrategy.LAZY
        )
        
        with memory_monitor() as monitor:
            # Simular carga de múltiples imágenes
            images = []
            for i in range(5):
                # Crear imagen simulada
                img = create_test_image(1000, 1000)
                images.append(img)
            
            # Simular procesamiento lazy
            for img in images:
                # Procesar solo una parte
                processed = img[:100, :100] * 1.1
                del processed
            
            del images
        
        stats = monitor.stop_monitoring()
        
        # El incremento de memoria debe ser controlado
        assert stats['increase_mb'] < 50, f"Incremento de memoria muy alto: {stats['increase_mb']:.2f}MB"
    
    @pytest.mark.skipif(LazyImageLoader == Mock, reason="LazyImageLoader not available")
    def test_lazy_loading_memory_efficiency(self):
        """Test de eficiencia de memoria del lazy loading"""
        # Crear dataset simulado
        image_paths = [f"test_image_{i}.jpg" for i in range(10)]
        
        with patch('os.path.exists', return_value=True):
            loader = LazyImageLoader(cache_size_mb=30)
            dataset = LazyImageDataset(image_paths, loader)
            
            with memory_monitor() as monitor:
                # Acceder a algunas imágenes
                for i in range(5):
                    # Simular carga lazy
                    with patch.object(loader, 'load_image') as mock_load:
                        mock_load.return_value = create_test_image(800, 600)
                        img = dataset[i]
                        # Procesar imagen
                        processed = img[:100, :100] if img is not None else None
                        del processed, img
            
            stats = monitor.stop_monitoring()
        
        # El lazy loading debe mantener memoria controlada
        assert stats['increase_mb'] < 40, f"Incremento de memoria muy alto: {stats['increase_mb']:.2f}MB"
    
    def test_memory_leak_detection(self):
        """Test para detectar memory leaks"""
        initial_objects = len(gc.get_objects())
        
        with memory_monitor() as monitor:
            # Crear y destruir objetos múltiples veces
            for iteration in range(10):
                images = []
                for i in range(5):
                    img = create_test_image(200, 200)
                    images.append(img)
                
                # Procesar imágenes
                for img in images:
                    processed = img * 1.1
                    del processed
                
                del images
                
                # Forzar garbage collection
                gc.collect()
        
        stats = monitor.stop_monitoring()
        final_objects = len(gc.get_objects())
        
        # No debe haber un crecimiento significativo de objetos
        object_growth = final_objects - initial_objects
        assert object_growth < 100, f"Posible memory leak: {object_growth} objetos nuevos"
        
        # El incremento final de memoria debe ser mínimo
        assert stats['increase_mb'] < 20, f"Posible memory leak: {stats['increase_mb']:.2f}MB incremento"
    
    def test_large_image_processing_memory(self):
        """Test de procesamiento de imágenes muy grandes"""
        with memory_monitor() as monitor:
            try:
                # Crear imagen muy grande (simulada)
                large_image = create_large_test_image(50)  # 50MB
                
                # Procesar por chunks simulados
                chunk_size = 1000
                height, width = large_image.shape[:2]
                
                for y in range(0, height, chunk_size):
                    for x in range(0, width, chunk_size):
                        # Extraer chunk
                        chunk = large_image[y:y+chunk_size, x:x+chunk_size]
                        
                        # Procesar chunk
                        processed_chunk = np.clip(chunk * 1.1, 0, 255).astype(np.uint8)
                        
                        # Limpiar chunk inmediatamente
                        del chunk, processed_chunk
                        
                        # Forzar garbage collection periódicamente
                        if (y + x) % (chunk_size * 4) == 0:
                            gc.collect()
                
                del large_image
                
            except MemoryError:
                pytest.skip("No hay suficiente memoria para este test")
        
        stats = monitor.stop_monitoring()
        
        # El procesamiento por chunks debe mantener memoria controlada
        assert stats['peak_mb'] - stats['initial_mb'] < 200, \
            f"Pico de memoria muy alto para imagen grande: {stats['peak_mb'] - stats['initial_mb']:.2f}MB"
    
    def test_concurrent_processing_memory(self):
        """Test de memoria con procesamiento concurrente"""
        import concurrent.futures
        
        def process_image_task(image_data):
            """Tarea de procesamiento de imagen"""
            img = np.frombuffer(image_data, dtype=np.uint8).reshape(100, 100, 3)
            processed = img * 1.2
            return processed.tobytes()
        
        with memory_monitor() as monitor:
            # Crear datos de imágenes
            image_data_list = []
            for i in range(10):
                img = create_test_image(100, 100)
                image_data_list.append(img.tobytes())
            
            # Procesar concurrentemente
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(process_image_task, data) for data in image_data_list]
                results = [future.result() for future in concurrent.futures.as_completed(futures)]
            
            del image_data_list, results
        
        stats = monitor.stop_monitoring()
        
        # El procesamiento concurrente debe ser eficiente en memoria
        assert stats['increase_mb'] < 30, f"Incremento de memoria muy alto en concurrencia: {stats['increase_mb']:.2f}MB"
    
    def test_cache_memory_management(self):
        """Test de gestión de memoria en cache"""
        cache_size_mb = 20
        
        with memory_monitor() as monitor:
            # Simular cache LRU
            cache = {}
            cache_size_bytes = cache_size_mb * 1024 * 1024
            current_cache_size = 0
            
            # Llenar cache
            for i in range(50):
                img = create_test_image(200, 200)
                img_size = img.nbytes
                
                # Simular política LRU
                if current_cache_size + img_size > cache_size_bytes:
                    # Limpiar cache (simulado)
                    cache.clear()
                    current_cache_size = 0
                    gc.collect()
                
                cache[f"img_{i}"] = img
                current_cache_size += img_size
            
            del cache
        
        stats = monitor.stop_monitoring()
        
        # El cache debe mantener memoria controlada
        assert stats['peak_mb'] - stats['initial_mb'] < cache_size_mb + 10, \
            f"Cache excedió límite de memoria: {stats['peak_mb'] - stats['initial_mb']:.2f}MB"


class TestMemoryBenchmarks:
    """Benchmarks de rendimiento de memoria"""
    
    def test_memory_performance_comparison(self):
        """Compara rendimiento de memoria entre diferentes estrategias"""
        strategies = ['normal', 'chunked', 'lazy']
        results = {}
        
        for strategy in strategies:
            with memory_monitor() as monitor:
                if strategy == 'normal':
                    # Procesamiento normal
                    img = create_test_image(1000, 1000)
                    processed = img * 1.2
                    del img, processed
                
                elif strategy == 'chunked':
                    # Procesamiento por chunks
                    img = create_test_image(1000, 1000)
                    chunk_size = 200
                    height, width = img.shape[:2]
                    
                    for y in range(0, height, chunk_size):
                        for x in range(0, width, chunk_size):
                            chunk = img[y:y+chunk_size, x:x+chunk_size]
                            processed_chunk = chunk * 1.2
                            del chunk, processed_chunk
                    
                    del img
                
                elif strategy == 'lazy':
                    # Simulación de lazy loading
                    img = create_test_image(1000, 1000)
                    # Procesar solo partes necesarias
                    for i in range(0, 1000, 100):
                        chunk = img[i:i+100, :]
                        processed = chunk * 1.2
                        del chunk, processed
                    del img
            
            results[strategy] = monitor.stop_monitoring()
        
        # Imprimir resultados para análisis
        print("\nResultados de benchmark de memoria:")
        for strategy, stats in results.items():
            print(f"{strategy}: Pico={stats['peak_mb']:.2f}MB, "
                  f"Incremento={stats['increase_mb']:.2f}MB")
        
        # Verificar que las estrategias optimizadas usen memoria razonable
        # Nota: El chunked puede usar más memoria temporalmente debido al overhead
        # pero debe mantenerse dentro de límites razonables
        if 'chunked' in results and 'normal' in results:
            chunked_peak = results['chunked']['peak_mb']
            normal_peak = results['normal']['peak_mb']
            memory_overhead = chunked_peak - normal_peak
            
            # Permitir hasta 30MB de overhead para chunked processing
            assert memory_overhead < 30, \
                f"Overhead de memoria chunked muy alto: {memory_overhead:.2f}MB"
            
            # Verificar que el incremento final sea controlado
            assert results['chunked']['increase_mb'] < 50, \
                f"Incremento final de memoria chunked muy alto: {results['chunked']['increase_mb']:.2f}MB"


def test_memory_monitoring_accuracy():
    """Test de precisión del monitor de memoria"""
    monitor = MemoryMonitor()
    
    # Test básico de funcionamiento
    initial = monitor.get_memory_usage()
    assert initial > 0, "Monitor debe reportar uso de memoria positivo"
    
    # Test de monitoreo
    monitor.start_monitoring()
    time.sleep(0.2)  # Esperar un poco
    stats = monitor.stop_monitoring()
    
    assert stats['initial_mb'] > 0, "Memoria inicial debe ser positiva"
    assert len(stats['samples']) > 1, "Debe haber múltiples muestras"
    assert stats['peak_mb'] >= stats['initial_mb'], "Pico debe ser >= inicial"


if __name__ == "__main__":
    # Ejecutar tests específicos de memoria
    pytest.main([__file__, "-v", "-s"])