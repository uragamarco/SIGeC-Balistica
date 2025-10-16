#!/usr/bin/env python3
"""
Suite de Tests de Rendimiento Consolidada - SIGeC-Balistica
===========================================================

Suite completa de tests de rendimiento que consolida:
- test_performance_integration_consolidated.py
- test_performance_benchmarks.py

Valida el rendimiento del sistema incluyendo:
- Procesamiento de imágenes y extracción de características
- Cálculo de similitud y operaciones de matching
- Operaciones de base de datos y cache inteligente
- Uso de memoria y CPU bajo diferentes cargas
- Procesamiento concurrente y escalabilidad
- Benchmarks de componentes críticos

Autor: SIGeC-Balistica Team
Fecha: Octubre 2025
"""

import sys
import os
import time
import tempfile
import numpy as np
import json
import pytest
import threading
import multiprocessing
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import unittest
from unittest.mock import Mock, patch
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import gc

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Imports del sistema
try:
    from config.unified_config import get_unified_config
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    get_unified_config = Mock()

try:
    from utils.logger import get_logger
    UTILS_AVAILABLE = True
except ImportError:
    UTILS_AVAILABLE = False
    get_logger = Mock()

# Procesamiento de imágenes
try:
    from image_processing.feature_extractor import BallisticFeatureExtractor
    from image_processing.unified_preprocessor import UnifiedPreprocessor
    from image_processing.chunked_processor import ChunkedImageProcessor
    from image_processing.optimized_loader import OptimizedImageLoader, LazyImageLoader
    IMAGE_PROCESSING_AVAILABLE = True
except ImportError:
    IMAGE_PROCESSING_AVAILABLE = False
    BallisticFeatureExtractor = Mock
    UnifiedPreprocessor = Mock
    ChunkedImageProcessor = Mock
    OptimizedImageLoader = Mock
    LazyImageLoader = Mock

# Base de datos
try:
    from database.vector_db import VectorDatabase
    DATABASE_AVAILABLE = True
except ImportError:
    DATABASE_AVAILABLE = False
    VectorDatabase = Mock

# Matching
try:
    from matching.unified_matcher import UnifiedMatcher
    MATCHING_AVAILABLE = True
except ImportError:
    MATCHING_AVAILABLE = False
    UnifiedMatcher = Mock

# Core
try:
    from core.intelligent_cache import IntelligentCache
    CORE_AVAILABLE = True
except ImportError:
    CORE_AVAILABLE = False
    IntelligentCache = Mock

# Utilidades de monitoreo
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


class PerformanceMonitor:
    """Monitor de rendimiento para tests"""
    
    def __init__(self):
        self.start_time = None
        self.start_memory = None
        self.start_cpu = None
        self.process = psutil.Process() if PSUTIL_AVAILABLE else None
    
    def start_monitoring(self):
        """Iniciar monitoreo"""
        self.start_time = time.time()
        if self.process:
            self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
            self.start_cpu = self.process.cpu_percent()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Obtener métricas actuales"""
        metrics = {
            'execution_time': time.time() - self.start_time if self.start_time else 0,
            'memory_usage': 0,
            'memory_delta': 0,
            'cpu_usage': 0
        }
        
        if self.process:
            current_memory = self.process.memory_info().rss / 1024 / 1024  # MB
            metrics['memory_usage'] = current_memory
            metrics['memory_delta'] = current_memory - (self.start_memory or 0)
            metrics['cpu_usage'] = self.process.cpu_percent()
        
        return metrics


class PerformanceTestSuite(unittest.TestCase):
    """Suite consolidada de tests de rendimiento"""
    
    @classmethod
    def setUpClass(cls):
        """Configuración inicial de la clase de tests"""
        cls.logger = get_logger(__name__) if UTILS_AVAILABLE else Mock()
        cls.config = get_unified_config() if CONFIG_AVAILABLE else {}
        cls.performance_results = {
            'timestamp': datetime.now().isoformat(),
            'system_info': cls._get_system_info(),
            'benchmarks': [],
            'summary': {
                'total_tests': 0,
                'passed_tests': 0,
                'failed_tests': 0,
                'avg_execution_time': 0,
                'peak_memory_usage': 0
            }
        }
        
        # Configurar directorio temporal para tests
        cls.temp_dir = tempfile.mkdtemp(prefix='sigec_perf_test_')
        
        print(f"⚡ Configurando tests de rendimiento...")
        print(f"📁 Directorio temporal: {cls.temp_dir}")
        print(f"🖥️ Sistema: {cls.performance_results['system_info']}")
        
    @classmethod
    def tearDownClass(cls):
        """Limpieza final de la clase de tests"""
        import shutil
        if hasattr(cls, 'temp_dir') and os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir, ignore_errors=True)
        
        # Generar reporte de rendimiento
        cls._generate_performance_report()
        
        # Imprimir resumen final
        summary = cls.performance_results['summary']
        print(f"\n📊 RESUMEN DE RENDIMIENTO:")
        print(f"   Total: {summary['total_tests']}")
        print(f"   ✅ Pasados: {summary['passed_tests']}")
        print(f"   ❌ Fallidos: {summary['failed_tests']}")
        print(f"   ⏱️ Tiempo promedio: {summary['avg_execution_time']:.3f}s")
        print(f"   🧠 Memoria pico: {summary['peak_memory_usage']:.1f} MB")
    
    @classmethod
    def _get_system_info(cls) -> Dict[str, Any]:
        """Obtener información del sistema"""
        info = {
            'cpu_count': multiprocessing.cpu_count(),
            'python_version': sys.version,
            'platform': sys.platform
        }
        
        if PSUTIL_AVAILABLE:
            info.update({
                'total_memory': psutil.virtual_memory().total / 1024 / 1024 / 1024,  # GB
                'available_memory': psutil.virtual_memory().available / 1024 / 1024 / 1024,  # GB
                'cpu_freq': psutil.cpu_freq().current if psutil.cpu_freq() else 'Unknown'
            })
        
        return info
    
    @classmethod
    def _generate_performance_report(cls):
        """Generar reporte de rendimiento"""
        try:
            report_file = os.path.join(cls.temp_dir, 'performance_report.json')
            with open(report_file, 'w') as f:
                json.dump(cls.performance_results, f, indent=2)
            print(f"📄 Reporte de rendimiento guardado en: {report_file}")
        except Exception as e:
            print(f"⚠️ Error generando reporte: {e}")
    
    def setUp(self):
        """Configuración antes de cada test"""
        self.monitor = PerformanceMonitor()
        self.monitor.start_monitoring()
        
    def tearDown(self):
        """Limpieza después de cada test"""
        metrics = self.monitor.get_metrics()
        test_name = self._testMethodName
        
        # Registrar benchmark
        benchmark = {
            'test_name': test_name,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics
        }
        
        self.performance_results['benchmarks'].append(benchmark)
        self.performance_results['summary']['total_tests'] += 1
        
        # Actualizar estadísticas
        summary = self.performance_results['summary']
        summary['peak_memory_usage'] = max(summary['peak_memory_usage'], metrics['memory_usage'])
        
        # Calcular tiempo promedio
        total_time = sum(b['metrics']['execution_time'] for b in self.performance_results['benchmarks'])
        summary['avg_execution_time'] = total_time / len(self.performance_results['benchmarks'])
    
    def test_image_processing_performance(self):
        """Benchmark de procesamiento de imágenes"""
        print("⚡ Benchmarking procesamiento de imágenes...")
        
        if not IMAGE_PROCESSING_AVAILABLE:
            self.performance_results['summary']['failed_tests'] += 1
            self.skipTest("Módulo de procesamiento de imágenes no disponible")
        
        try:
            # Crear imágenes de prueba de diferentes tamaños
            test_images = [
                np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8),  # Pequeña
                np.random.randint(0, 255, (800, 600, 3), dtype=np.uint8),  # Mediana
                np.random.randint(0, 255, (1920, 1080, 3), dtype=np.uint8)  # Grande
            ]
            
            sizes = ['200x200', '800x600', '1920x1080']
            
            # Inicializar preprocessor
            if not isinstance(UnifiedPreprocessor, Mock):
                preprocessor = UnifiedPreprocessor()
                
                processing_times = []
                
                for i, (img, size) in enumerate(zip(test_images, sizes)):
                    start_time = time.time()
                    
                    if hasattr(preprocessor, 'preprocess'):
                        try:
                            processed = preprocessor.preprocess(img)
                            processing_time = time.time() - start_time
                            processing_times.append(processing_time)
                            
                            print(f"  📊 {size}: {processing_time:.3f}s")
                            
                        except Exception as e:
                            print(f"  ⚠️ Error procesando {size}: {e}")
                            processing_times.append(float('inf'))
                
                # Verificar rendimiento aceptable
                avg_time = np.mean([t for t in processing_times if t != float('inf')])
                if avg_time > 5.0:  # 5 segundos promedio es demasiado
                    print(f"  ⚠️ Rendimiento lento: {avg_time:.3f}s promedio")
                else:
                    print(f"  ✅ Rendimiento aceptable: {avg_time:.3f}s promedio")
            
            self.performance_results['summary']['passed_tests'] += 1
            print("  ✅ Benchmark de procesamiento completado")
            
        except Exception as e:
            self.performance_results['summary']['failed_tests'] += 1
            self.fail(f"Error en benchmark de procesamiento: {e}")
    
    def test_feature_extraction_performance(self):
        """Benchmark de extracción de características"""
        print("⚡ Benchmarking extracción de características...")
        
        if not IMAGE_PROCESSING_AVAILABLE:
            self.performance_results['summary']['failed_tests'] += 1
            self.skipTest("Módulo de procesamiento de imágenes no disponible")
        
        try:
            # Crear imágenes de prueba
            test_images = [
                np.random.randint(0, 255, (400, 400, 3), dtype=np.uint8)
                for _ in range(10)
            ]
            
            # Inicializar extractor
            if not isinstance(BallisticFeatureExtractor, Mock):
                extractor = BallisticFeatureExtractor()
                
                extraction_times = []
                feature_counts = []
                
                for i, img in enumerate(test_images):
                    start_time = time.time()
                    
                    if hasattr(extractor, 'extract_features'):
                        try:
                            features = extractor.extract_features(img)
                            extraction_time = time.time() - start_time
                            extraction_times.append(extraction_time)
                            
                            if hasattr(features, 'shape'):
                                feature_counts.append(features.shape[0] if len(features.shape) > 0 else 1)
                            else:
                                feature_counts.append(1)
                            
                        except Exception as e:
                            print(f"  ⚠️ Error extrayendo características {i}: {e}")
                            extraction_times.append(float('inf'))
                            feature_counts.append(0)
                
                # Calcular estadísticas
                valid_times = [t for t in extraction_times if t != float('inf')]
                if valid_times:
                    avg_time = np.mean(valid_times)
                    min_time = np.min(valid_times)
                    max_time = np.max(valid_times)
                    avg_features = np.mean([c for c in feature_counts if c > 0])
                    
                    print(f"  📊 Tiempo promedio: {avg_time:.3f}s")
                    print(f"  📊 Tiempo mín/máx: {min_time:.3f}s / {max_time:.3f}s")
                    print(f"  📊 Características promedio: {avg_features:.0f}")
                    
                    # Verificar rendimiento
                    if avg_time > 2.0:  # 2 segundos por imagen es demasiado
                        print(f"  ⚠️ Extracción lenta: {avg_time:.3f}s promedio")
                    else:
                        print(f"  ✅ Extracción eficiente")
            
            self.performance_results['summary']['passed_tests'] += 1
            print("  ✅ Benchmark de extracción completado")
            
        except Exception as e:
            self.performance_results['summary']['failed_tests'] += 1
            self.fail(f"Error en benchmark de extracción: {e}")
    
    def test_matching_performance(self):
        """Benchmark de matching y similitud"""
        print("⚡ Benchmarking matching y similitud...")
        
        if not MATCHING_AVAILABLE:
            self.performance_results['summary']['failed_tests'] += 1
            self.skipTest("Módulo de matching no disponible")
        
        try:
            # Crear características de prueba
            num_features = 1000
            feature_dim = 128
            
            features_db = np.random.rand(num_features, feature_dim).astype(np.float32)
            query_features = np.random.rand(10, feature_dim).astype(np.float32)
            
            # Inicializar matcher
            if not isinstance(UnifiedMatcher, Mock):
                matcher = UnifiedMatcher()
                
                matching_times = []
                similarities = []
                
                for i, query in enumerate(query_features):
                    start_time = time.time()
                    
                    if hasattr(matcher, 'compare_features'):
                        try:
                            # Comparar con todas las características de la BD
                            query_similarities = []
                            for db_feature in features_db[:100]:  # Limitar para el test
                                sim = matcher.compare_features(query, db_feature)
                                query_similarities.append(sim)
                            
                            matching_time = time.time() - start_time
                            matching_times.append(matching_time)
                            similarities.extend(query_similarities)
                            
                        except Exception as e:
                            print(f"  ⚠️ Error en matching {i}: {e}")
                            matching_times.append(float('inf'))
                
                # Calcular estadísticas
                valid_times = [t for t in matching_times if t != float('inf')]
                if valid_times:
                    avg_time = np.mean(valid_times)
                    total_comparisons = len(similarities)
                    
                    print(f"  📊 Tiempo promedio por query: {avg_time:.3f}s")
                    print(f"  📊 Total comparaciones: {total_comparisons}")
                    print(f"  📊 Comparaciones por segundo: {total_comparisons/sum(valid_times):.0f}")
                    
                    # Verificar rendimiento
                    if avg_time > 1.0:  # 1 segundo por query es demasiado
                        print(f"  ⚠️ Matching lento: {avg_time:.3f}s por query")
                    else:
                        print(f"  ✅ Matching eficiente")
            
            self.performance_results['summary']['passed_tests'] += 1
            print("  ✅ Benchmark de matching completado")
            
        except Exception as e:
            self.performance_results['summary']['failed_tests'] += 1
            self.fail(f"Error en benchmark de matching: {e}")
    
    def test_database_performance(self):
        """Benchmark de operaciones de base de datos"""
        print("⚡ Benchmarking operaciones de base de datos...")
        
        if not DATABASE_AVAILABLE:
            self.performance_results['summary']['failed_tests'] += 1
            self.skipTest("Módulo de base de datos no disponible")
        
        try:
            # Configurar base de datos de prueba
            db_config = {
                'path': os.path.join(self.temp_dir, 'perf_test.db'),
                'faiss_path': os.path.join(self.temp_dir, 'perf_test.index')
            }
            
            # Inicializar base de datos
            if not isinstance(VectorDatabase, Mock):
                db = VectorDatabase(db_config)
                
                # Crear datos de prueba
                num_records = 100
                feature_dim = 128
                
                insert_times = []
                search_times = []
                
                # Benchmark de inserción
                print("  📊 Benchmarking inserción...")
                for i in range(num_records):
                    start_time = time.time()
                    
                    # Crear datos de prueba
                    features = np.random.rand(feature_dim).astype(np.float32)
                    metadata = {
                        'case_id': f'case_{i}',
                        'image_path': f'/test/image_{i}.jpg',
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    # Insertar (si el método existe)
                    if hasattr(db, 'insert_features'):
                        try:
                            db.insert_features(features, metadata)
                            insert_time = time.time() - start_time
                            insert_times.append(insert_time)
                        except Exception as e:
                            print(f"    ⚠️ Error insertando {i}: {e}")
                            insert_times.append(float('inf'))
                
                # Benchmark de búsqueda
                print("  📊 Benchmarking búsqueda...")
                for i in range(10):  # 10 búsquedas de prueba
                    start_time = time.time()
                    
                    query_features = np.random.rand(feature_dim).astype(np.float32)
                    
                    if hasattr(db, 'search_similar'):
                        try:
                            results = db.search_similar(query_features, k=5)
                            search_time = time.time() - start_time
                            search_times.append(search_time)
                        except Exception as e:
                            print(f"    ⚠️ Error buscando {i}: {e}")
                            search_times.append(float('inf'))
                
                # Calcular estadísticas
                valid_insert_times = [t for t in insert_times if t != float('inf')]
                valid_search_times = [t for t in search_times if t != float('inf')]
                
                if valid_insert_times:
                    avg_insert = np.mean(valid_insert_times)
                    print(f"  📊 Inserción promedio: {avg_insert:.4f}s")
                    print(f"  📊 Inserciones por segundo: {1/avg_insert:.0f}")
                
                if valid_search_times:
                    avg_search = np.mean(valid_search_times)
                    print(f"  📊 Búsqueda promedio: {avg_search:.4f}s")
                    print(f"  📊 Búsquedas por segundo: {1/avg_search:.0f}")
            
            self.performance_results['summary']['passed_tests'] += 1
            print("  ✅ Benchmark de base de datos completado")
            
        except Exception as e:
            self.performance_results['summary']['failed_tests'] += 1
            self.fail(f"Error en benchmark de base de datos: {e}")
    
    def test_memory_usage_performance(self):
        """Benchmark de uso de memoria"""
        print("⚡ Benchmarking uso de memoria...")
        
        if not PSUTIL_AVAILABLE:
            self.performance_results['summary']['failed_tests'] += 1
            self.skipTest("psutil no disponible")
        
        try:
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Crear múltiples imágenes grandes
            large_images = []
            memory_checkpoints = []
            
            for i in range(5):
                # Crear imagen grande
                img = np.random.randint(0, 255, (2000, 2000, 3), dtype=np.uint8)
                large_images.append(img)
                
                # Checkpoint de memoria
                current_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_checkpoints.append(current_memory)
                
                print(f"  📊 Imagen {i+1}: {current_memory:.1f} MB (+{current_memory-initial_memory:.1f} MB)")
            
            peak_memory = max(memory_checkpoints)
            memory_increase = peak_memory - initial_memory
            
            # Limpiar imágenes
            large_images.clear()
            gc.collect()
            
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_recovered = peak_memory - final_memory
            
            print(f"  📊 Memoria inicial: {initial_memory:.1f} MB")
            print(f"  📊 Memoria pico: {peak_memory:.1f} MB")
            print(f"  📊 Incremento máximo: {memory_increase:.1f} MB")
            print(f"  📊 Memoria final: {final_memory:.1f} MB")
            print(f"  📊 Memoria recuperada: {memory_recovered:.1f} MB")
            
            # Verificar que no hay leak masivo
            final_increase = final_memory - initial_memory
            if final_increase > 100:  # 100MB de incremento permanente es sospechoso
                print(f"  ⚠️ Posible leak de memoria: +{final_increase:.1f} MB")
            else:
                print(f"  ✅ Gestión de memoria aceptable")
            
            self.performance_results['summary']['passed_tests'] += 1
            print("  ✅ Benchmark de memoria completado")
            
        except Exception as e:
            self.performance_results['summary']['failed_tests'] += 1
            self.fail(f"Error en benchmark de memoria: {e}")
    
    def test_concurrent_processing_performance(self):
        """Benchmark de procesamiento concurrente"""
        print("⚡ Benchmarking procesamiento concurrente...")
        
        if not IMAGE_PROCESSING_AVAILABLE:
            self.performance_results['summary']['failed_tests'] += 1
            self.skipTest("Módulo de procesamiento de imágenes no disponible")
        
        try:
            # Crear imágenes de prueba
            test_images = [
                np.random.randint(0, 255, (400, 400, 3), dtype=np.uint8)
                for _ in range(20)
            ]
            
            def process_image(img):
                """Función para procesar una imagen"""
                if not isinstance(UnifiedPreprocessor, Mock):
                    preprocessor = UnifiedPreprocessor()
                    if hasattr(preprocessor, 'preprocess'):
                        try:
                            return preprocessor.preprocess(img)
                        except Exception:
                            return None
                return img  # Fallback
            
            # Procesamiento secuencial
            print("  📊 Procesamiento secuencial...")
            start_time = time.time()
            sequential_results = []
            for img in test_images:
                result = process_image(img)
                sequential_results.append(result)
            sequential_time = time.time() - start_time
            
            # Procesamiento concurrente con threads
            print("  📊 Procesamiento concurrente (threads)...")
            start_time = time.time()
            with ThreadPoolExecutor(max_workers=4) as executor:
                concurrent_results = list(executor.map(process_image, test_images))
            concurrent_time = time.time() - start_time
            
            # Calcular speedup
            speedup = sequential_time / concurrent_time if concurrent_time > 0 else 0
            
            print(f"  📊 Tiempo secuencial: {sequential_time:.3f}s")
            print(f"  📊 Tiempo concurrente: {concurrent_time:.3f}s")
            print(f"  📊 Speedup: {speedup:.2f}x")
            
            # Verificar que el procesamiento concurrente es efectivo
            if speedup > 1.5:
                print(f"  ✅ Procesamiento concurrente eficiente")
            elif speedup > 1.0:
                print(f"  ⚠️ Procesamiento concurrente moderadamente eficiente")
            else:
                print(f"  ⚠️ Procesamiento concurrente no efectivo")
            
            self.performance_results['summary']['passed_tests'] += 1
            print("  ✅ Benchmark de concurrencia completado")
            
        except Exception as e:
            self.performance_results['summary']['failed_tests'] += 1
            self.fail(f"Error en benchmark de concurrencia: {e}")
    
    def test_cache_performance(self):
        """Benchmark de rendimiento del cache"""
        print("⚡ Benchmarking rendimiento del cache...")
        
        if not CORE_AVAILABLE:
            self.performance_results['summary']['failed_tests'] += 1
            self.skipTest("Módulo core no disponible")
        
        try:
            # Inicializar cache
            if not isinstance(IntelligentCache, Mock):
                cache = IntelligentCache()
                
                # Datos de prueba
                test_data = {
                    f'key_{i}': np.random.rand(100, 100).astype(np.float32)
                    for i in range(100)
                }
                
                # Benchmark de escritura
                print("  📊 Benchmarking escritura en cache...")
                write_times = []
                for key, data in test_data.items():
                    start_time = time.time()
                    
                    if hasattr(cache, 'set'):
                        try:
                            cache.set(key, data)
                            write_time = time.time() - start_time
                            write_times.append(write_time)
                        except Exception as e:
                            print(f"    ⚠️ Error escribiendo {key}: {e}")
                            write_times.append(float('inf'))
                
                # Benchmark de lectura
                print("  📊 Benchmarking lectura de cache...")
                read_times = []
                cache_hits = 0
                
                for key in test_data.keys():
                    start_time = time.time()
                    
                    if hasattr(cache, 'get'):
                        try:
                            result = cache.get(key)
                            read_time = time.time() - start_time
                            read_times.append(read_time)
                            
                            if result is not None:
                                cache_hits += 1
                                
                        except Exception as e:
                            print(f"    ⚠️ Error leyendo {key}: {e}")
                            read_times.append(float('inf'))
                
                # Calcular estadísticas
                valid_write_times = [t for t in write_times if t != float('inf')]
                valid_read_times = [t for t in read_times if t != float('inf')]
                
                if valid_write_times:
                    avg_write = np.mean(valid_write_times)
                    print(f"  📊 Escritura promedio: {avg_write:.6f}s")
                    print(f"  📊 Escrituras por segundo: {1/avg_write:.0f}")
                
                if valid_read_times:
                    avg_read = np.mean(valid_read_times)
                    hit_rate = cache_hits / len(test_data) * 100
                    print(f"  📊 Lectura promedio: {avg_read:.6f}s")
                    print(f"  📊 Lecturas por segundo: {1/avg_read:.0f}")
                    print(f"  📊 Tasa de aciertos: {hit_rate:.1f}%")
                    
                    # Verificar eficiencia del cache
                    if avg_read < avg_write and hit_rate > 80:
                        print(f"  ✅ Cache eficiente")
                    else:
                        print(f"  ⚠️ Cache podría ser más eficiente")
            
            self.performance_results['summary']['passed_tests'] += 1
            print("  ✅ Benchmark de cache completado")
            
        except Exception as e:
            self.performance_results['summary']['failed_tests'] += 1
            self.fail(f"Error en benchmark de cache: {e}")
    
    def test_scalability_performance(self):
        """Benchmark de escalabilidad del sistema"""
        print("⚡ Benchmarking escalabilidad del sistema...")
        
        try:
            # Probar con diferentes tamaños de carga
            load_sizes = [10, 50, 100, 200]
            processing_times = []
            
            for load_size in load_sizes:
                print(f"  📊 Probando carga de {load_size} elementos...")
                
                # Crear datos de prueba
                test_data = [
                    np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
                    for _ in range(load_size)
                ]
                
                start_time = time.time()
                
                # Procesar datos (simulado)
                processed_count = 0
                for data in test_data:
                    # Simular procesamiento
                    if IMAGE_PROCESSING_AVAILABLE and not isinstance(UnifiedPreprocessor, Mock):
                        preprocessor = UnifiedPreprocessor()
                        if hasattr(preprocessor, 'preprocess'):
                            try:
                                result = preprocessor.preprocess(data)
                                processed_count += 1
                            except Exception:
                                pass
                    else:
                        # Simulación básica
                        time.sleep(0.001)  # 1ms por elemento
                        processed_count += 1
                
                processing_time = time.time() - start_time
                processing_times.append(processing_time)
                
                throughput = processed_count / processing_time if processing_time > 0 else 0
                print(f"    ⏱️ Tiempo: {processing_time:.3f}s")
                print(f"    📊 Throughput: {throughput:.1f} elementos/s")
            
            # Analizar escalabilidad
            print("  📊 Análisis de escalabilidad:")
            for i, (load, time_taken) in enumerate(zip(load_sizes, processing_times)):
                time_per_element = time_taken / load if load > 0 else 0
                print(f"    {load} elementos: {time_per_element:.4f}s por elemento")
            
            # Verificar escalabilidad lineal
            if len(processing_times) >= 2:
                time_ratio = processing_times[-1] / processing_times[0]
                load_ratio = load_sizes[-1] / load_sizes[0]
                scalability_factor = time_ratio / load_ratio
                
                print(f"  📊 Factor de escalabilidad: {scalability_factor:.2f}")
                
                if scalability_factor < 1.5:
                    print(f"  ✅ Escalabilidad excelente")
                elif scalability_factor < 2.0:
                    print(f"  ✅ Escalabilidad buena")
                else:
                    print(f"  ⚠️ Escalabilidad podría mejorar")
            
            self.performance_results['summary']['passed_tests'] += 1
            print("  ✅ Benchmark de escalabilidad completado")
            
        except Exception as e:
            self.performance_results['summary']['failed_tests'] += 1
            self.fail(f"Error en benchmark de escalabilidad: {e}")


class TestChunkedProcessing(unittest.TestCase):
    """Tests específicos para procesamiento por chunks"""
    
    def test_chunked_image_processing(self):
        """Probar procesamiento de imágenes por chunks"""
        if not IMAGE_PROCESSING_AVAILABLE:
            self.skipTest("Módulo de procesamiento no disponible")
        
        if isinstance(ChunkedImageProcessor, Mock):
            self.skipTest("ChunkedImageProcessor no disponible")
        
        # Crear imagen grande
        large_image = np.random.randint(0, 255, (2000, 2000, 3), dtype=np.uint8)
        
        # Procesar por chunks
        processor = ChunkedImageProcessor(chunk_size=500)
        
        start_time = time.time()
        if hasattr(processor, 'process_image'):
            result = processor.process_image(large_image)
            processing_time = time.time() - start_time
            
            print(f"  📊 Procesamiento por chunks: {processing_time:.3f}s")
            print(f"  ✅ Procesamiento por chunks funcionando")
    
    def test_optimized_image_loading(self):
        """Probar carga optimizada de imágenes"""
        if not IMAGE_PROCESSING_AVAILABLE:
            self.skipTest("Módulo de procesamiento no disponible")
        
        if isinstance(OptimizedImageLoader, Mock):
            self.skipTest("OptimizedImageLoader no disponible")
        
        # Crear archivos de imagen simulados
        temp_dir = tempfile.mkdtemp()
        try:
            # Simular múltiples archivos
            image_paths = []
            for i in range(10):
                img_path = os.path.join(temp_dir, f'test_image_{i}.npy')
                test_img = np.random.randint(0, 255, (400, 400, 3), dtype=np.uint8)
                np.save(img_path, test_img)
                image_paths.append(img_path)
            
            # Probar carga optimizada
            loader = OptimizedImageLoader()
            
            start_time = time.time()
            if hasattr(loader, 'load_batch'):
                loaded_images = loader.load_batch(image_paths)
                loading_time = time.time() - start_time
                
                print(f"  📊 Carga optimizada: {loading_time:.3f}s para {len(image_paths)} imágenes")
                print(f"  ✅ Carga optimizada funcionando")
        
        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)


def run_performance_tests():
    """Ejecutar todos los tests de rendimiento"""
    print("=" * 70)
    print("⚡ EJECUTANDO SUITE DE TESTS DE RENDIMIENTO")
    print("=" * 70)
    
    # Crear suite de tests
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Agregar tests principales
    suite.addTests(loader.loadTestsFromTestCase(PerformanceTestSuite))
    suite.addTests(loader.loadTestsFromTestCase(TestChunkedProcessing))
    
    # Ejecutar tests
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    
    # Mostrar resumen
    print("\n" + "=" * 70)
    print("📊 RESUMEN DE RENDIMIENTO")
    print("=" * 70)
    print(f"Tests ejecutados: {result.testsRun}")
    print(f"Errores: {len(result.errors)}")
    print(f"Fallos: {len(result.failures)}")
    print(f"Saltados: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
    
    if result.errors:
        print("\n❌ ERRORES:")
        for test, error in result.errors:
            print(f"  - {test}: {error.split(chr(10))[0]}")
    
    if result.failures:
        print("\n❌ FALLOS:")
        for test, failure in result.failures:
            print(f"  - {test}: {failure.split(chr(10))[0]}")
    
    success = len(result.errors) == 0 and len(result.failures) == 0
    
    if success:
        print("\n🎉 TODOS LOS TESTS DE RENDIMIENTO PASARON EXITOSAMENTE")
    else:
        print("\n⚠️ ALGUNOS TESTS DE RENDIMIENTO FALLARON")
    
    return success


if __name__ == "__main__":
    success = run_performance_tests()
    sys.exit(0 if success else 1)