"""
GPU Performance Benchmark
Sistema Balístico Forense MVP

Módulo para benchmarking de rendimiento GPU vs CPU
para operaciones de procesamiento de imágenes y matching
"""

import cv2
import numpy as np
import time
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import json
from dataclasses import dataclass, field

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from image_processing.gpu_accelerator import GPUAccelerator
except ImportError:
    GPUAccelerator = None

try:
    from image_processing.unified_preprocessor import UnifiedPreprocessor, PreprocessingConfig
except ImportError:
    print("Warning: UnifiedPreprocessor not available")
    UnifiedPreprocessor = None
    PreprocessingConfig = None

try:
    from matching.unified_matcher import UnifiedMatcher, MatchingConfig, AlgorithmType
except ImportError:
    print("Warning: UnifiedMatcher not available")
    UnifiedMatcher = None
    MatchingConfig = None
    AlgorithmType = None

@dataclass
class BenchmarkResult:
    """Resultado de benchmark"""
    operation: str
    cpu_time: float
    gpu_time: float
    speedup: float
    cpu_memory: float = 0.0
    gpu_memory: float = 0.0
    image_size: Tuple[int, int] = (0, 0)
    success: bool = True
    error_message: str = ""

@dataclass
class BenchmarkConfig:
    """Configuración de benchmark"""
    test_image_sizes: List[Tuple[int, int]] = field(default_factory=lambda: [
        (640, 480), (1280, 720), (1920, 1080), (2560, 1440), (3840, 2160)
    ])
    iterations: int = 5
    warmup_iterations: int = 2
    test_preprocessing: bool = True
    test_matching: bool = True
    test_feature_extraction: bool = True
    algorithms_to_test: List[AlgorithmType] = field(default_factory=lambda: [
        AlgorithmType.ORB, AlgorithmType.SIFT, AlgorithmType.AKAZE
    ])

class GPUBenchmark:
    """Benchmark de rendimiento GPU vs CPU"""
    
    def __init__(self, config: Optional[BenchmarkConfig] = None):
        """
        Inicializa el benchmark
        
        Args:
            config: Configuración del benchmark
        """
        self.config = config or BenchmarkConfig()
        self.logger = logging.getLogger(__name__)
        
        # Inicializar GPU accelerator
        self.gpu_accelerator = None
        if GPUAccelerator:
            try:
                self.gpu_accelerator = GPUAccelerator()
                if self.gpu_accelerator.is_gpu_enabled():
                    self.logger.info(f"GPU disponible: {self.gpu_accelerator.get_gpu_info()}")
                else:
                    self.logger.warning("GPU no disponible para benchmark")
            except Exception as e:
                self.logger.error(f"Error inicializando GPU: {e}")
        
        # Inicializar procesadores
        self.cpu_preprocessor = UnifiedPreprocessor(
            PreprocessingConfig(enable_gpu_acceleration=False)
        )
        
        if self.gpu_accelerator and self.gpu_accelerator.is_gpu_enabled():
            self.gpu_preprocessor = UnifiedPreprocessor(
                PreprocessingConfig(enable_gpu_acceleration=True)
            )
        else:
            self.gpu_preprocessor = None
        
        # Inicializar matchers
        self.cpu_matcher = UnifiedMatcher(
            MatchingConfig(enable_gpu_acceleration=False)
        )
        
        if self.gpu_accelerator and self.gpu_accelerator.is_gpu_enabled():
            self.gpu_matcher = UnifiedMatcher(
                MatchingConfig(enable_gpu_acceleration=True)
            )
        else:
            self.gpu_matcher = None
    
    def create_test_image(self, size: Tuple[int, int]) -> np.ndarray:
        """
        Crea una imagen de prueba sintética
        
        Args:
            size: Tamaño de la imagen (width, height)
            
        Returns:
            Imagen de prueba
        """
        width, height = size
        
        # Crear imagen con patrones complejos para simular características balísticas
        image = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Agregar círculos concéntricos
        center = (width // 2, height // 2)
        for i in range(5, min(width, height) // 4, 10):
            cv2.circle(image, center, i, (50 + i, 100 + i, 150 + i), 2)
        
        # Agregar líneas radiales
        for angle in range(0, 360, 15):
            x = int(center[0] + (width // 4) * np.cos(np.radians(angle)))
            y = int(center[1] + (height // 4) * np.sin(np.radians(angle)))
            cv2.line(image, center, (x, y), (200, 150, 100), 1)
        
        # Agregar ruido
        noise = np.random.randint(0, 50, (height, width, 3), dtype=np.uint8)
        image = cv2.add(image, noise)
        
        return image
    
    def benchmark_preprocessing(self, image: np.ndarray) -> Tuple[BenchmarkResult, BenchmarkResult]:
        """
        Benchmark de preprocesamiento
        
        Args:
            image: Imagen de prueba
            
        Returns:
            Tupla con resultados CPU y GPU
        """
        height, width = image.shape[:2]
        
        # Benchmark CPU
        cpu_times = []
        for _ in range(self.config.warmup_iterations + self.config.iterations):
            start_time = time.perf_counter()
            try:
                _ = self.cpu_preprocessor.preprocess_image(image.copy())
                end_time = time.perf_counter()
                cpu_times.append(end_time - start_time)
            except Exception as e:
                return BenchmarkResult(
                    operation="preprocessing",
                    cpu_time=0,
                    gpu_time=0,
                    speedup=0,
                    image_size=(width, height),
                    success=False,
                    error_message=f"CPU error: {e}"
                ), None
        
        cpu_time = np.mean(cpu_times[self.config.warmup_iterations:])
        
        # Benchmark GPU
        gpu_time = 0
        gpu_success = True
        gpu_error = ""
        
        if self.gpu_preprocessor:
            gpu_times = []
            for _ in range(self.config.warmup_iterations + self.config.iterations):
                start_time = time.perf_counter()
                try:
                    _ = self.gpu_preprocessor.preprocess_image(image.copy())
                    end_time = time.perf_counter()
                    gpu_times.append(end_time - start_time)
                except Exception as e:
                    gpu_success = False
                    gpu_error = f"GPU error: {e}"
                    break
            
            if gpu_success:
                gpu_time = np.mean(gpu_times[self.config.warmup_iterations:])
        
        speedup = cpu_time / gpu_time if gpu_time > 0 else 0
        
        cpu_result = BenchmarkResult(
            operation="preprocessing_cpu",
            cpu_time=cpu_time,
            gpu_time=0,
            speedup=1.0,
            image_size=(width, height),
            success=True
        )
        
        gpu_result = BenchmarkResult(
            operation="preprocessing_gpu",
            cpu_time=0,
            gpu_time=gpu_time,
            speedup=speedup,
            image_size=(width, height),
            success=gpu_success,
            error_message=gpu_error
        ) if self.gpu_preprocessor else None
        
        return cpu_result, gpu_result
    
    def benchmark_feature_extraction(self, image: np.ndarray, algorithm: AlgorithmType) -> Tuple[BenchmarkResult, BenchmarkResult]:
        """
        Benchmark de extracción de características
        
        Args:
            image: Imagen de prueba
            algorithm: Algoritmo a probar
            
        Returns:
            Tupla con resultados CPU y GPU
        """
        height, width = image.shape[:2]
        
        # Benchmark CPU
        cpu_times = []
        for _ in range(self.config.warmup_iterations + self.config.iterations):
            start_time = time.perf_counter()
            try:
                _ = self.cpu_matcher.extract_features(image.copy(), algorithm)
                end_time = time.perf_counter()
                cpu_times.append(end_time - start_time)
            except Exception as e:
                return BenchmarkResult(
                    operation=f"feature_extraction_{algorithm.value}",
                    cpu_time=0,
                    gpu_time=0,
                    speedup=0,
                    image_size=(width, height),
                    success=False,
                    error_message=f"CPU error: {e}"
                ), None
        
        cpu_time = np.mean(cpu_times[self.config.warmup_iterations:])
        
        # Benchmark GPU
        gpu_time = 0
        gpu_success = True
        gpu_error = ""
        
        if self.gpu_matcher:
            gpu_times = []
            for _ in range(self.config.warmup_iterations + self.config.iterations):
                start_time = time.perf_counter()
                try:
                    _ = self.gpu_matcher.extract_features(image.copy(), algorithm)
                    end_time = time.perf_counter()
                    gpu_times.append(end_time - start_time)
                except Exception as e:
                    gpu_success = False
                    gpu_error = f"GPU error: {e}"
                    break
            
            if gpu_success:
                gpu_time = np.mean(gpu_times[self.config.warmup_iterations:])
        
        speedup = cpu_time / gpu_time if gpu_time > 0 else 0
        
        cpu_result = BenchmarkResult(
            operation=f"feature_extraction_{algorithm.value}_cpu",
            cpu_time=cpu_time,
            gpu_time=0,
            speedup=1.0,
            image_size=(width, height),
            success=True
        )
        
        gpu_result = BenchmarkResult(
            operation=f"feature_extraction_{algorithm.value}_gpu",
            cpu_time=0,
            gpu_time=gpu_time,
            speedup=speedup,
            image_size=(width, height),
            success=gpu_success,
            error_message=gpu_error
        ) if self.gpu_matcher else None
        
        return cpu_result, gpu_result
    
    def benchmark_matching(self, image1: np.ndarray, image2: np.ndarray, algorithm: AlgorithmType) -> Tuple[BenchmarkResult, BenchmarkResult]:
        """
        Benchmark de matching
        
        Args:
            image1: Primera imagen
            image2: Segunda imagen
            algorithm: Algoritmo a probar
            
        Returns:
            Tupla con resultados CPU y GPU
        """
        height, width = image1.shape[:2]
        
        # Extraer características una vez
        try:
            cpu_features1 = self.cpu_matcher.extract_features(image1, algorithm)
            cpu_features2 = self.cpu_matcher.extract_features(image2, algorithm)
        except Exception as e:
            return BenchmarkResult(
                operation=f"matching_{algorithm.value}",
                cpu_time=0,
                gpu_time=0,
                speedup=0,
                image_size=(width, height),
                success=False,
                error_message=f"CPU feature extraction error: {e}"
            ), None
        
        # Benchmark CPU matching
        cpu_times = []
        for _ in range(self.config.warmup_iterations + self.config.iterations):
            start_time = time.perf_counter()
            try:
                _ = self.cpu_matcher.match_features(cpu_features1, cpu_features2, algorithm)
                end_time = time.perf_counter()
                cpu_times.append(end_time - start_time)
            except Exception as e:
                return BenchmarkResult(
                    operation=f"matching_{algorithm.value}",
                    cpu_time=0,
                    gpu_time=0,
                    speedup=0,
                    image_size=(width, height),
                    success=False,
                    error_message=f"CPU matching error: {e}"
                ), None
        
        cpu_time = np.mean(cpu_times[self.config.warmup_iterations:])
        
        # Benchmark GPU matching
        gpu_time = 0
        gpu_success = True
        gpu_error = ""
        
        if self.gpu_matcher:
            try:
                gpu_features1 = self.gpu_matcher.extract_features(image1, algorithm)
                gpu_features2 = self.gpu_matcher.extract_features(image2, algorithm)
                
                gpu_times = []
                for _ in range(self.config.warmup_iterations + self.config.iterations):
                    start_time = time.perf_counter()
                    try:
                        _ = self.gpu_matcher.match_features(gpu_features1, gpu_features2, algorithm)
                        end_time = time.perf_counter()
                        gpu_times.append(end_time - start_time)
                    except Exception as e:
                        gpu_success = False
                        gpu_error = f"GPU matching error: {e}"
                        break
                
                if gpu_success:
                    gpu_time = np.mean(gpu_times[self.config.warmup_iterations:])
                    
            except Exception as e:
                gpu_success = False
                gpu_error = f"GPU feature extraction error: {e}"
        
        speedup = cpu_time / gpu_time if gpu_time > 0 else 0
        
        cpu_result = BenchmarkResult(
            operation=f"matching_{algorithm.value}_cpu",
            cpu_time=cpu_time,
            gpu_time=0,
            speedup=1.0,
            image_size=(width, height),
            success=True
        )
        
        gpu_result = BenchmarkResult(
            operation=f"matching_{algorithm.value}_gpu",
            cpu_time=0,
            gpu_time=gpu_time,
            speedup=speedup,
            image_size=(width, height),
            success=gpu_success,
            error_message=gpu_error
        ) if self.gpu_matcher else None
        
        return cpu_result, gpu_result
    
    def run_full_benchmark(self) -> Dict[str, List[BenchmarkResult]]:
        """
        Ejecuta el benchmark completo
        
        Returns:
            Diccionario con todos los resultados
        """
        results = {
            'preprocessing': [],
            'feature_extraction': [],
            'matching': []
        }
        
        self.logger.info("Iniciando benchmark completo GPU vs CPU")
        
        for size in self.config.test_image_sizes:
            self.logger.info(f"Probando tamaño de imagen: {size}")
            
            # Crear imágenes de prueba
            image1 = self.create_test_image(size)
            image2 = self.create_test_image(size)
            
            # Benchmark preprocessing
            if self.config.test_preprocessing:
                cpu_result, gpu_result = self.benchmark_preprocessing(image1)
                results['preprocessing'].append(cpu_result)
                if gpu_result:
                    results['preprocessing'].append(gpu_result)
            
            # Benchmark feature extraction
            if self.config.test_feature_extraction:
                for algorithm in self.config.algorithms_to_test:
                    cpu_result, gpu_result = self.benchmark_feature_extraction(image1, algorithm)
                    results['feature_extraction'].append(cpu_result)
                    if gpu_result:
                        results['feature_extraction'].append(gpu_result)
            
            # Benchmark matching
            if self.config.test_matching:
                for algorithm in self.config.algorithms_to_test:
                    cpu_result, gpu_result = self.benchmark_matching(image1, image2, algorithm)
                    results['matching'].append(cpu_result)
                    if gpu_result:
                        results['matching'].append(gpu_result)
        
        return results
    
    def generate_report(self, results: Dict[str, List[BenchmarkResult]], output_path: str) -> bool:
        """
        Genera reporte de benchmark
        
        Args:
            results: Resultados del benchmark
            output_path: Ruta del archivo de salida
            
        Returns:
            True si se generó correctamente
        """
        try:
            report = {
                'system_info': {
                    'gpu_available': self.gpu_accelerator is not None and self.gpu_accelerator.is_gpu_enabled(),
                    'gpu_info': self.gpu_accelerator.get_gpu_info() if self.gpu_accelerator and self.gpu_accelerator.is_gpu_enabled() else None
                },
                'config': {
                    'test_image_sizes': self.config.test_image_sizes,
                    'iterations': self.config.iterations,
                    'warmup_iterations': self.config.warmup_iterations,
                    'algorithms_tested': [alg.value for alg in self.config.algorithms_to_test]
                },
                'results': {}
            }
            
            # Procesar resultados
            for category, benchmark_results in results.items():
                report['results'][category] = []
                
                for result in benchmark_results:
                    report['results'][category].append({
                        'operation': result.operation,
                        'cpu_time': result.cpu_time,
                        'gpu_time': result.gpu_time,
                        'speedup': result.speedup,
                        'image_size': result.image_size,
                        'success': result.success,
                        'error_message': result.error_message
                    })
            
            # Calcular estadísticas de resumen
            summary = {}
            for category in results.keys():
                gpu_results = [r for r in results[category] if 'gpu' in r.operation and r.success]
                if gpu_results:
                    speedups = [r.speedup for r in gpu_results if r.speedup > 0]
                    if speedups:
                        summary[category] = {
                            'avg_speedup': np.mean(speedups),
                            'max_speedup': np.max(speedups),
                            'min_speedup': np.min(speedups),
                            'successful_tests': len([r for r in gpu_results if r.success]),
                            'total_tests': len(gpu_results)
                        }
            
            report['summary'] = summary
            
            # Guardar reporte
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Reporte de benchmark guardado en: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error generando reporte: {e}")
            return False

def main():
    """Función principal para ejecutar benchmark"""
    logging.basicConfig(level=logging.INFO)
    
    # Configurar benchmark
    config = BenchmarkConfig(
        test_image_sizes=[(640, 480), (1280, 720), (1920, 1080)],
        iterations=3,
        warmup_iterations=1
    )
    
    # Ejecutar benchmark
    benchmark = GPUBenchmark(config)
    results = benchmark.run_full_benchmark()
    
    # Generar reporte
    output_path = "gpu_benchmark_report.json"
    benchmark.generate_report(results, output_path)
    
    # Mostrar resumen
    print("\n=== RESUMEN DE BENCHMARK GPU vs CPU ===")
    for category, benchmark_results in results.items():
        gpu_results = [r for r in benchmark_results if 'gpu' in r.operation and r.success]
        if gpu_results:
            speedups = [r.speedup for r in gpu_results if r.speedup > 0]
            if speedups:
                print(f"\n{category.upper()}:")
                print(f"  Aceleración promedio: {np.mean(speedups):.2f}x")
                print(f"  Aceleración máxima: {np.max(speedups):.2f}x")
                print(f"  Pruebas exitosas: {len([r for r in gpu_results if r.success])}/{len(gpu_results)}")

if __name__ == "__main__":
    main()