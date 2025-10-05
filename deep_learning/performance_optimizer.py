"""
Optimizador de rendimiento para SEACABAr Deep Learning
====================================================

Herramientas para optimización de rendimiento CPU/GPU, análisis de memoria
y profiling de modelos de deep learning balístico.
"""

import torch
import torch.nn as nn
import psutil
import time
import gc
from typing import Dict, List, Tuple, Optional, Any, Callable
import logging
from pathlib import Path
import json
import numpy as np
from dataclasses import dataclass, asdict
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Métricas de rendimiento del sistema."""
    
    # Tiempo
    inference_time_ms: float = 0.0
    training_time_ms: float = 0.0
    data_loading_time_ms: float = 0.0
    
    # Memoria
    peak_memory_mb: float = 0.0
    current_memory_mb: float = 0.0
    memory_efficiency: float = 0.0
    
    # GPU (si disponible)
    gpu_memory_mb: float = 0.0
    gpu_utilization: float = 0.0
    
    # CPU
    cpu_utilization: float = 0.0
    num_threads: int = 1
    
    # Throughput
    samples_per_second: float = 0.0
    batches_per_second: float = 0.0
    
    # Modelo
    model_parameters: int = 0
    model_size_mb: float = 0.0
    flops: int = 0


class SystemProfiler:
    """Profiler del sistema para monitoreo de recursos."""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.has_gpu = torch.cuda.is_available()
        
        logger.info(f"SystemProfiler inicializado - Device: {self.device}")
        if self.has_gpu:
            logger.info(f"GPU detectada: {torch.cuda.get_device_name()}")
    
    def get_system_info(self) -> Dict[str, Any]:
        """Obtener información del sistema."""
        info = {
            'cpu_count': psutil.cpu_count(),
            'cpu_freq': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
            'device': str(self.device),
            'has_gpu': self.has_gpu,
            'torch_version': torch.__version__,
            'num_threads': torch.get_num_threads()
        }
        
        if self.has_gpu:
            info.update({
                'gpu_name': torch.cuda.get_device_name(),
                'gpu_memory_gb': torch.cuda.get_device_properties(0).total_memory / (1024**3),
                'cuda_version': torch.version.cuda
            })
        
        return info
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Obtener uso actual de memoria."""
        memory = psutil.virtual_memory()
        usage = {
            'total_mb': memory.total / (1024**2),
            'available_mb': memory.available / (1024**2),
            'used_mb': memory.used / (1024**2),
            'percent': memory.percent
        }
        
        if self.has_gpu:
            gpu_memory = torch.cuda.memory_stats()
            usage.update({
                'gpu_allocated_mb': torch.cuda.memory_allocated() / (1024**2),
                'gpu_reserved_mb': torch.cuda.memory_reserved() / (1024**2),
                'gpu_max_allocated_mb': torch.cuda.max_memory_allocated() / (1024**2)
            })
        
        return usage
    
    def get_cpu_usage(self) -> Dict[str, float]:
        """Obtener uso de CPU."""
        return {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'cpu_per_core': psutil.cpu_percent(interval=1, percpu=True),
            'load_avg': psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None
        }


class ModelProfiler:
    """Profiler específico para modelos de deep learning."""
    
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.model.to(device)
        
        logger.info(f"ModelProfiler inicializado para modelo en {device}")
    
    def count_parameters(self) -> Dict[str, int]:
        """Contar parámetros del modelo."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'non_trainable_parameters': total_params - trainable_params
        }
    
    def estimate_model_size(self) -> float:
        """Estimar tamaño del modelo en MB."""
        param_size = 0
        buffer_size = 0
        
        for param in self.model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in self.model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        return (param_size + buffer_size) / (1024**2)
    
    @contextmanager
    def profile_inference(self, input_tensor: torch.Tensor):
        """Context manager para profiling de inferencia."""
        self.model.eval()
        
        # Limpiar memoria
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        gc.collect()
        
        # Mediciones iniciales
        start_memory = psutil.virtual_memory().used / (1024**2)
        start_time = time.perf_counter()
        
        try:
            with torch.no_grad():
                yield
        finally:
            # Mediciones finales
            end_time = time.perf_counter()
            end_memory = psutil.virtual_memory().used / (1024**2)
            
            inference_time = (end_time - start_time) * 1000  # ms
            memory_used = end_memory - start_memory
            
            logger.info(f"Inferencia completada: {inference_time:.2f}ms, {memory_used:.2f}MB")
    
    def benchmark_inference(self, input_shape: Tuple[int, ...], 
                          num_runs: int = 100, warmup_runs: int = 10) -> PerformanceMetrics:
        """Benchmark de inferencia del modelo."""
        logger.info(f"Iniciando benchmark de inferencia: {num_runs} runs, warmup {warmup_runs}")
        
        self.model.eval()
        input_tensor = torch.randn(input_shape).to(self.device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = self.model(input_tensor)
        
        # Limpiar memoria antes del benchmark
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        gc.collect()
        
        # Benchmark
        times = []
        start_memory = psutil.virtual_memory().used / (1024**2)
        
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = time.perf_counter()
                _ = self.model(input_tensor)
                end_time = time.perf_counter()
                times.append((end_time - start_time) * 1000)  # ms
        
        end_memory = psutil.virtual_memory().used / (1024**2)
        
        # Calcular métricas
        avg_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)
        
        batch_size = input_shape[0]
        samples_per_second = (batch_size * 1000) / avg_time
        
        metrics = PerformanceMetrics(
            inference_time_ms=avg_time,
            peak_memory_mb=end_memory - start_memory,
            current_memory_mb=psutil.virtual_memory().used / (1024**2),
            samples_per_second=samples_per_second,
            batches_per_second=1000 / avg_time,
            model_parameters=self.count_parameters()['total_parameters'],
            model_size_mb=self.estimate_model_size()
        )
        
        if torch.cuda.is_available():
            metrics.gpu_memory_mb = torch.cuda.max_memory_allocated() / (1024**2)
        
        logger.info(f"Benchmark completado: {avg_time:.2f}±{std_time:.2f}ms, {samples_per_second:.1f} samples/s")
        
        return metrics


class PerformanceOptimizer:
    """Optimizador de rendimiento para modelos balísticos."""
    
    def __init__(self):
        self.system_profiler = SystemProfiler()
        self.device = self.system_profiler.device
        
        logger.info("PerformanceOptimizer inicializado")
    
    def optimize_torch_settings(self) -> Dict[str, Any]:
        """Optimizar configuraciones de PyTorch."""
        optimizations = {}
        
        # Configurar número de threads
        if self.device.type == 'cpu':
            num_threads = min(psutil.cpu_count(), 8)  # Limitar para evitar overhead
            torch.set_num_threads(num_threads)
            optimizations['num_threads'] = num_threads
        
        # Configurar backend de cuDNN
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            optimizations['cudnn_benchmark'] = True
        
        # Configurar memoria
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            optimizations['gpu_memory_cleared'] = True
        
        logger.info(f"Optimizaciones aplicadas: {optimizations}")
        return optimizations
    
    def optimize_model_for_inference(self, model: nn.Module) -> nn.Module:
        """Optimizar modelo para inferencia."""
        logger.info("Optimizando modelo para inferencia...")
        
        # Modo evaluación
        model.eval()
        
        # Fusión de operaciones (si es posible)
        try:
            if hasattr(torch.jit, 'optimize_for_inference'):
                model = torch.jit.optimize_for_inference(torch.jit.script(model))
                logger.info("Modelo optimizado con TorchScript")
        except Exception as e:
            logger.warning(f"No se pudo aplicar TorchScript: {e}")
        
        return model
    
    def analyze_model_complexity(self, model: nn.Module, 
                                input_shape: Tuple[int, ...]) -> Dict[str, Any]:
        """Analizar complejidad computacional del modelo."""
        profiler = ModelProfiler(model, self.device)
        
        # Parámetros
        params_info = profiler.count_parameters()
        model_size = profiler.estimate_model_size()
        
        # Benchmark de inferencia
        metrics = profiler.benchmark_inference(input_shape)
        
        analysis = {
            'parameters': params_info,
            'model_size_mb': model_size,
            'performance_metrics': asdict(metrics),
            'complexity_score': self._calculate_complexity_score(params_info, metrics),
            'recommendations': self._generate_recommendations(params_info, metrics)
        }
        
        return analysis
    
    def _calculate_complexity_score(self, params_info: Dict, metrics: PerformanceMetrics) -> float:
        """Calcular score de complejidad (0-100, menor es mejor)."""
        # Normalizar parámetros (log scale)
        param_score = min(np.log10(params_info['total_parameters']) * 10, 50)
        
        # Normalizar tiempo de inferencia
        time_score = min(metrics.inference_time_ms / 10, 30)
        
        # Normalizar memoria
        memory_score = min(metrics.peak_memory_mb / 100, 20)
        
        return param_score + time_score + memory_score
    
    def _generate_recommendations(self, params_info: Dict, 
                                metrics: PerformanceMetrics) -> List[str]:
        """Generar recomendaciones de optimización."""
        recommendations = []
        
        # Recomendaciones basadas en parámetros
        if params_info['total_parameters'] > 50_000_000:
            recommendations.append("Considerar pruning o quantización para reducir parámetros")
        
        # Recomendaciones basadas en tiempo
        if metrics.inference_time_ms > 100:
            recommendations.append("Tiempo de inferencia alto - considerar optimización de arquitectura")
        
        # Recomendaciones basadas en memoria
        if metrics.peak_memory_mb > 1000:
            recommendations.append("Alto uso de memoria - considerar batch size menor")
        
        # Recomendaciones basadas en throughput
        if metrics.samples_per_second < 10:
            recommendations.append("Bajo throughput - revisar cuellos de botella")
        
        if not recommendations:
            recommendations.append("Rendimiento óptimo - no se requieren optimizaciones")
        
        return recommendations
    
    def generate_performance_report(self, model: nn.Module, 
                                  input_shape: Tuple[int, ...],
                                  output_path: Optional[str] = None) -> Dict[str, Any]:
        """Generar reporte completo de rendimiento."""
        logger.info("Generando reporte de rendimiento...")
        
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'system_info': self.system_profiler.get_system_info(),
            'model_analysis': self.analyze_model_complexity(model, input_shape),
            'memory_usage': self.system_profiler.get_memory_usage(),
            'cpu_usage': self.system_profiler.get_cpu_usage()
        }
        
        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"Reporte guardado en: {output_file}")
        
        return report


def quick_performance_test():
    """Prueba rápida del sistema de optimización."""
    print("=== PRUEBA DE OPTIMIZACIÓN DE RENDIMIENTO ===")
    
    try:
        # Inicializar optimizador
        optimizer = PerformanceOptimizer()
        
        # Información del sistema
        system_info = optimizer.system_profiler.get_system_info()
        print(f"\n📊 Sistema: {system_info['cpu_count']} CPUs, {system_info['memory_total_gb']:.1f}GB RAM")
        print(f"🔧 Device: {system_info['device']}")
        
        # Optimizar configuraciones
        optimizations = optimizer.optimize_torch_settings()
        print(f"⚡ Optimizaciones aplicadas: {len(optimizations)}")
        
        # Crear modelo de prueba
        from deep_learning.models.cnn_models import BallisticCNN
        model = BallisticCNN(num_classes=10, input_channels=3)
        
        # Analizar rendimiento
        input_shape = (4, 3, 224, 224)  # batch_size, channels, height, width
        analysis = optimizer.analyze_model_complexity(model, input_shape)
        
        print(f"\n🧠 Modelo: {analysis['parameters']['total_parameters']:,} parámetros")
        print(f"💾 Tamaño: {analysis['model_size_mb']:.1f}MB")
        print(f"⏱️  Inferencia: {analysis['performance_metrics']['inference_time_ms']:.2f}ms")
        print(f"🚀 Throughput: {analysis['performance_metrics']['samples_per_second']:.1f} samples/s")
        print(f"📈 Score complejidad: {analysis['complexity_score']:.1f}")
        
        print(f"\n💡 Recomendaciones:")
        for rec in analysis['recommendations']:
            print(f"  - {rec}")
        
        print("\n✅ Prueba de optimización completada")
        
    except Exception as e:
        print(f"❌ Error en prueba: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    quick_performance_test()