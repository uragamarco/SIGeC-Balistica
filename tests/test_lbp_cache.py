#!/usr/bin/env python3
"""
Tests para el Sistema de Cache LBP
Sistema Balístico Forense MVP

Tests para validar:
- Correctitud del cache (resultados idénticos)
- Rendimiento del cache (mejora de velocidad)
- Gestión de memoria
- Políticas de evicción
- Estadísticas del cache
"""

import unittest
import numpy as np
import time
import sys
from pathlib import Path

# Agregar el directorio raíz al path
sys.path.append(str(Path(__file__).parent.parent))

from image_processing.lbp_cache import (
    LBPPatternCache, 
    cached_local_binary_pattern,
    get_lbp_cache_stats,
    clear_lbp_cache,
    set_lbp_cache_config
)
from skimage.feature import local_binary_pattern


class TestLBPCache(unittest.TestCase):
    """Tests para el sistema de cache LBP"""
    
    def setUp(self):
        """Configuración inicial para cada test"""
        # Limpiar cache antes de cada test
        clear_lbp_cache()
        
        # Configurar cache para tests
        set_lbp_cache_config(
            max_size=100,
            max_memory_mb=50,
            eviction_policy='lru'
        )
        
        # Crear imágenes de test
        self.test_image_small = np.random.randint(0, 256, (50, 50), dtype=np.uint8)
        self.test_image_medium = np.random.randint(0, 256, (200, 200), dtype=np.uint8)
        self.test_image_large = np.random.randint(0, 256, (500, 500), dtype=np.uint8)
        
        # Parámetros LBP comunes
        self.lbp_params = [
            (8, 1, 'uniform'),
            (16, 2, 'uniform'),
            (24, 3, 'uniform'),
            (8, 1, 'default'),
            (16, 2, 'default')
        ]
    
    def tearDown(self):
        """Limpieza después de cada test"""
        clear_lbp_cache()
    
    def test_cache_correctness(self):
        """Test que verifica que el cache devuelve resultados correctos"""
        print("\n🧪 Test: Correctitud del cache")
        
        for n_points, radius, method in self.lbp_params:
            with self.subTest(n_points=n_points, radius=radius, method=method):
                # Calcular LBP original
                original_result = local_binary_pattern(
                    self.test_image_medium, n_points, radius, method=method
                )
                
                # Calcular LBP con cache (primera vez)
                cached_result_1, _ = cached_local_binary_pattern(
                    self.test_image_medium, n_points, radius, method=method
                )
                
                # Calcular LBP con cache (segunda vez, debería usar cache)
                cached_result_2, _ = cached_local_binary_pattern(
                    self.test_image_medium, n_points, radius, method=method
                )
                
                # Verificar que todos los resultados son idénticos
                np.testing.assert_array_equal(
                    original_result, cached_result_1,
                    "El resultado con cache no coincide con el original"
                )
                
                np.testing.assert_array_equal(
                    cached_result_1, cached_result_2,
                    "Los resultados del cache no son consistentes"
                )
        
        print("✅ Cache devuelve resultados correctos")
    
    def test_cache_performance(self):
        """Test que verifica mejora de rendimiento del cache"""
        print("\n🚀 Test: Rendimiento del cache")
        
        # Usar imagen grande para mejor medición
        test_image = self.test_image_large
        n_points, radius, method = 24, 3, 'uniform'
        
        # Medir tiempo sin cache (múltiples ejecuciones)
        times_no_cache = []
        for _ in range(5):
            start_time = time.time()
            local_binary_pattern(test_image, n_points, radius, method=method)
            times_no_cache.append(time.time() - start_time)
        
        avg_time_no_cache = np.mean(times_no_cache)
        
        # Primera ejecución con cache (debería ser similar al tiempo sin cache)
        start_time = time.time()
        cached_result, _ = cached_local_binary_pattern(test_image, n_points, radius, method=method)
        first_cache_time = time.time() - start_time
        
        # Múltiples ejecuciones con cache (deberían ser más rápidas)
        times_with_cache = []
        for _ in range(10):
            start_time = time.time()
            cached_local_binary_pattern(test_image, n_points, radius, method=method)
            times_with_cache.append(time.time() - start_time)
        
        avg_time_with_cache = np.mean(times_with_cache)
        
        # Verificar mejora de rendimiento
        speedup = avg_time_no_cache / avg_time_with_cache
        
        print(f"Tiempo promedio sin cache: {avg_time_no_cache:.4f}s")
        print(f"Tiempo promedio con cache: {avg_time_with_cache:.4f}s")
        print(f"Aceleración: {speedup:.2f}x")
        
        # El cache debería ser al menos 2x más rápido
        self.assertGreater(speedup, 2.0, 
                          f"Cache no es suficientemente rápido (speedup: {speedup:.2f}x)")
        
        print("✅ Cache mejora significativamente el rendimiento")
    
    def test_cache_statistics(self):
        """Test que verifica las estadísticas del cache"""
        print("\n📊 Test: Estadísticas del cache")
        
        # Ejecutar algunas operaciones
        for i, (n_points, radius, method) in enumerate(self.lbp_params):
            # Primera ejecución (miss)
            cached_local_binary_pattern(
                self.test_image_small, n_points, radius, method=method
            )
            
            # Segunda ejecución (hit)
            cached_local_binary_pattern(
                self.test_image_small, n_points, radius, method=method
            )
        
        # Obtener estadísticas
        stats = get_lbp_cache_stats()
        
        # Verificar estadísticas
        self.assertGreater(stats['total_hits'] + stats['total_misses'], 0, "No se registraron requests")
        self.assertGreater(stats['total_hits'], 0, "No se registraron hits")
        self.assertGreater(stats['total_misses'], 0, "No se registraron misses")
        self.assertGreater(stats['hit_rate'], 0.0, "Hit rate es 0")
        self.assertLess(stats['hit_rate'], 1.0, "Hit rate es 100% (sospechoso)")
        
        print(f"Total requests: {stats['total_hits'] + stats['total_misses']}")
        print(f"Cache hits: {stats['total_hits']}")
        print(f"Cache misses: {stats['total_misses']}")
        print(f"Hit rate: {stats['hit_rate']:.2%}")
        print(f"Memory usage: {stats['memory_used_mb']:.2f} MB")
        
        print("✅ Estadísticas del cache funcionan correctamente")
    
    def test_memory_management(self):
        """Test que verifica la gestión de memoria del cache"""
        print("\n💾 Test: Gestión de memoria")
        
        # Configurar cache con límite de memoria pequeño
        set_lbp_cache_config(max_memory_mb=10)
        
        # Llenar cache con imágenes grandes
        large_images = []
        for i in range(20):
            # Crear imagen única
            image = np.random.randint(0, 256, (300, 300), dtype=np.uint8)
            large_images.append(image)
            
            # Procesar con cache
            cached_local_binary_pattern(image, 24, 3, method='uniform')
        
        # Verificar que el cache no excede el límite de memoria
        stats = get_lbp_cache_stats()
        self.assertLessEqual(stats['memory_used_mb'], 15.0,  # Margen de tolerancia
                            f"Cache excede límite de memoria: {stats['memory_used_mb']:.2f} MB")
        
        print(f"Memoria utilizada: {stats['memory_used_mb']:.2f} MB")
        print("✅ Gestión de memoria funciona correctamente")
    
    def test_eviction_policies(self):
        """Test que verifica las políticas de evicción"""
        print("\n🔄 Test: Políticas de evicción")
        
        # Test LRU con tamaño muy pequeño para forzar evicción
        set_lbp_cache_config(max_size=2, eviction_policy='lru')
        clear_lbp_cache()
        
        # Crear 3 imágenes diferentes y únicas
        images = []
        for i in range(3):
            # Crear imágenes muy diferentes para asegurar hashes únicos
            image = np.zeros((50, 50), dtype=np.uint8)
            image[i*10:(i+1)*10, i*10:(i+1)*10] = 255  # Patrón único por imagen
            images.append(image)
        
        # Llenar cache con 2 imágenes
        cached_local_binary_pattern(images[0], 8, 1, method='uniform')
        cached_local_binary_pattern(images[1], 8, 1, method='uniform')
        
        stats = get_lbp_cache_stats()
        self.assertEqual(stats['cache_size'], 2, f"Cache debería tener 2 entradas, tiene {stats['cache_size']}")
        
        # Agregar tercera imagen (debería evictar una)
        cached_local_binary_pattern(images[2], 8, 1, method='uniform')
        
        stats = get_lbp_cache_stats()
        self.assertLessEqual(stats['cache_size'], 2, f"Cache excede tamaño máximo: {stats['cache_size']} > 2")
        
        print(f"Cache size después de evicción: {stats['cache_size']}")
        print("✅ Políticas de evicción funcionan correctamente")
    
    def test_different_image_types(self):
        """Test que verifica compatibilidad con diferentes tipos de imagen"""
        print("\n🖼️ Test: Tipos de imagen")
        
        # Test con diferentes tipos de datos
        image_types = [
            np.random.randint(0, 256, (100, 100), dtype=np.uint8),
            np.random.randint(0, 65536, (100, 100), dtype=np.uint16),
            np.random.random((100, 100)).astype(np.float32),
            np.random.random((100, 100)).astype(np.float64)
        ]
        
        for i, image_type in enumerate(image_types):
            with self.subTest(dtype=image_type.dtype):
                try:
                    result, _ = cached_local_binary_pattern(image_type, 8, 1, method='uniform')
                    self.assertIsNotNone(result, f"Resultado None para tipo {image_type.dtype}")
                except Exception as e:
                    self.fail(f"Error con tipo {image_type.dtype}: {e}")
        
        print("✅ Cache funciona con diferentes tipos de imagen")
    
    def test_concurrent_access(self):
        """Test básico de acceso concurrente"""
        print("\n🔀 Test: Acceso concurrente")
        
        import threading
        
        results = []
        errors = []
        
        def worker():
            try:
                for _ in range(10):
                    result, _ = cached_local_binary_pattern(
                        self.test_image_medium, 16, 2, method='uniform'
                    )
                    results.append(result)
            except Exception as e:
                errors.append(e)
        
        # Crear múltiples threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()
        
        # Esperar a que terminen
        for thread in threads:
            thread.join()
        
        # Verificar que no hubo errores
        self.assertEqual(len(errors), 0, f"Errores en acceso concurrente: {errors}")
        self.assertGreater(len(results), 0, "No se obtuvieron resultados")
        
        # Verificar que todos los resultados son iguales
        first_result = results[0]
        for result in results[1:]:
            np.testing.assert_array_equal(first_result, result,
                                        "Resultados inconsistentes en acceso concurrente")
        
        print("✅ Acceso concurrente funciona correctamente")


class TestLBPCacheIntegration(unittest.TestCase):
    """Tests de integración del cache LBP con el sistema completo"""
    
    def setUp(self):
        """Configuración inicial"""
        clear_lbp_cache()
        
        # Crear imagen de test realista
        self.test_image = self._create_realistic_ballistic_image()
    
    def _create_realistic_ballistic_image(self) -> np.ndarray:
        """Crea una imagen sintética que simula una imagen balística"""
        # Crear imagen base
        image = np.zeros((400, 400), dtype=np.uint8)
        
        # Agregar ruido de fondo
        noise = np.random.normal(128, 30, (400, 400))
        image = np.clip(noise, 0, 255).astype(np.uint8)
        
        # Agregar círculo central (simula firing pin)
        center = (200, 200)
        radius = 50
        y, x = np.ogrid[:400, :400]
        mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2
        image[mask] = np.clip(image[mask] + 50, 0, 255)
        
        # Agregar textura
        for i in range(0, 400, 20):
            for j in range(0, 400, 20):
                if np.random.random() > 0.7:
                    # Usar int32 para evitar overflow, luego convertir a uint8
                    texture_patch = image[i:i+10, j:j+10].astype(np.int32)
                    texture_patch += np.random.randint(-20, 20)
                    image[i:i+10, j:j+10] = np.clip(texture_patch, 0, 255).astype(np.uint8)
        
        return np.clip(image, 0, 255).astype(np.uint8)
    
    def test_integration_with_feature_extractor(self):
        """Test de integración con el extractor de características"""
        print("\n🔗 Test: Integración con extractor de características")
        
        try:
            from image_processing.ballistic_features import BallisticFeatureExtractor
            
            # Crear extractor
            extractor = BallisticFeatureExtractor()
            
            # Extraer características (debería usar cache internamente)
            start_time = time.time()
            features_1 = extractor.extract_all_features(self.test_image)
            time_1 = time.time() - start_time
            
            # Segunda extracción (debería ser más rápida por cache)
            start_time = time.time()
            features_2 = extractor.extract_all_features(self.test_image)
            time_2 = time.time() - start_time
            
            # Verificar que los resultados son consistentes
            self.assertIsNotNone(features_1, "Primera extracción falló")
            self.assertIsNotNone(features_2, "Segunda extracción falló")
            
            # Verificar mejora de rendimiento
            if time_1 > 0.1:  # Solo si la primera toma tiempo significativo
                speedup = time_1 / time_2
                print(f"Primera extracción: {time_1:.3f}s")
                print(f"Segunda extracción: {time_2:.3f}s")
                print(f"Aceleración: {speedup:.2f}x")
                
                self.assertGreater(speedup, 1.2, "Cache no mejora rendimiento en integración")
            
            print("✅ Integración con extractor funciona correctamente")
            
        except ImportError:
            self.skipTest("BallisticFeatureExtractor no disponible")
    
    def test_cache_persistence_across_operations(self):
        """Test que verifica persistencia del cache entre operaciones"""
        print("\n💾 Test: Persistencia del cache")
        
        # Realizar múltiples operaciones diferentes
        operations = [
            (self.test_image, 8, 1, 'uniform'),
            (self.test_image, 16, 2, 'uniform'),
            (self.test_image, 24, 3, 'uniform'),
            (self.test_image, 8, 1, 'default')
        ]
        
        # Primera ronda (llenar cache)
        for image, n_points, radius, method in operations:
            cached_local_binary_pattern(image, n_points, radius, method=method)
        
        initial_stats = get_lbp_cache_stats()
        
        # Segunda ronda (debería usar cache)
        for image, n_points, radius, method in operations:
            cached_local_binary_pattern(image, n_points, radius, method=method)
        
        final_stats = get_lbp_cache_stats()
        
        # Verificar que el cache se utilizó
        self.assertGreater(final_stats['total_hits'], initial_stats['total_hits'],
                          "Cache no se utilizó en segunda ronda")
        
        print(f"Cache hits incrementaron de {initial_stats['total_hits']} a {final_stats['total_hits']}")
        print("✅ Cache persiste correctamente entre operaciones")


def run_performance_benchmark():
    """Ejecuta benchmark de rendimiento del cache"""
    print("\n" + "="*60)
    print("🏁 BENCHMARK DE RENDIMIENTO DEL CACHE LBP")
    print("="*60)
    
    # Configurar cache
    clear_lbp_cache()
    set_lbp_cache_config(max_size=1000, max_memory_mb=100)
    
    # Crear conjunto de imágenes de test
    test_images = []
    image_sizes = [(100, 100), (200, 200), (400, 400), (800, 800)]
    
    for size in image_sizes:
        for i in range(3):
            image = np.random.randint(0, 256, size, dtype=np.uint8)
            test_images.append((image, f"{size[0]}x{size[1]}_{i}"))
    
    # Parámetros LBP para test
    lbp_configs = [
        (8, 1, 'uniform'),
        (16, 2, 'uniform'),
        (24, 3, 'uniform')
    ]
    
    results = {}
    
    for n_points, radius, method in lbp_configs:
        config_name = f"LBP_{n_points}_{radius}_{method}"
        print(f"\n📊 Benchmark para {config_name}")
        
        # Test sin cache
        times_no_cache = []
        for image, name in test_images:
            start_time = time.time()
            local_binary_pattern(image, n_points, radius, method=method)
            times_no_cache.append(time.time() - start_time)
        
        avg_time_no_cache = np.mean(times_no_cache)
        
        # Test con cache (primera pasada)
        clear_lbp_cache()
        times_cache_first = []
        for image, name in test_images:
            start_time = time.time()
            cached_local_binary_pattern(image, n_points, radius, method=method)
            times_cache_first.append(time.time() - start_time)
        
        avg_time_cache_first = np.mean(times_cache_first)
        
        # Test con cache (segunda pasada - hits)
        times_cache_hits = []
        for image, name in test_images:
            start_time = time.time()
            cached_local_binary_pattern(image, n_points, radius, method=method)
            times_cache_hits.append(time.time() - start_time)
        
        avg_time_cache_hits = np.mean(times_cache_hits)
        
        # Calcular métricas
        speedup_hits = avg_time_no_cache / avg_time_cache_hits if avg_time_cache_hits > 0 else 0
        overhead = (avg_time_cache_first / avg_time_no_cache - 1) * 100 if avg_time_no_cache > 0 else 0
        
        results[config_name] = {
            'time_no_cache': avg_time_no_cache,
            'time_cache_first': avg_time_cache_first,
            'time_cache_hits': avg_time_cache_hits,
            'speedup': speedup_hits,
            'overhead_percent': overhead
        }
        
        print(f"  Sin cache: {avg_time_no_cache:.4f}s")
        print(f"  Con cache (primera vez): {avg_time_cache_first:.4f}s")
        print(f"  Con cache (hits): {avg_time_cache_hits:.4f}s")
        print(f"  Aceleración: {speedup_hits:.2f}x")
        print(f"  Overhead inicial: {overhead:.1f}%")
    
    # Estadísticas finales del cache
    final_stats = get_lbp_cache_stats()
    print(f"\n📈 Estadísticas finales del cache:")
    print(f"  Total hits: {final_stats['total_hits']}")
    print(f"  Total misses: {final_stats['total_misses']}")
    print(f"  Hit rate: {final_stats['hit_rate']:.2%}")
    print(f"  Memory usage: {final_stats['memory_used_mb']:.2f} MB")
    print(f"  Cache size: {final_stats['cache_size']} entries")
    
    return results


if __name__ == '__main__':
    print("🧪 Ejecutando tests del sistema de cache LBP...")
    
    # Ejecutar tests unitarios
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Ejecutar benchmark
    benchmark_results = run_performance_benchmark()
    
    print("\n" + "="*60)
    print("✅ TESTS COMPLETADOS")
    print("="*60)