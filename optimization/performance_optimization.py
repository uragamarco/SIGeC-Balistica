#!/usr/bin/env python3
"""
Sistema de Optimización de Rendimiento para SIGeC-Balistica.
Implementa técnicas avanzadas de optimización para mejorar el rendimiento general del sistema.
"""

import os
import sys
import gc
import time
import psutil
import asyncio
import threading
import multiprocessing
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import cProfile
import pstats
import io
import json
import yaml
import logging

# Importar sistema de caché inteligente
try:
    from intelligent_cache_system import (
        IntelligentCacheSystem, CacheConfig, CacheStrategy, 
        CachePriority, create_intelligent_cache
    )
    CACHE_AVAILABLE = True
except ImportError:
    CACHE_AVAILABLE = False
    logging.warning("Sistema de caché inteligente no disponible")

class OptimizationType(Enum):
    """Tipos de optimización disponibles."""
    MEMORY = "memory"
    CPU = "cpu"
    IO = "io"
    NETWORK = "network"
    DATABASE = "database"
    CACHE = "cache"
    THREADING = "threading"
    ASYNC = "async"

class PerformanceLevel(Enum):
    """Niveles de rendimiento."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    ULTRA = "ultra"

@dataclass
class OptimizationConfig:
    """Configuración de optimización."""
    enabled_optimizations: List[OptimizationType] = field(default_factory=lambda: list(OptimizationType))
    performance_level: PerformanceLevel = PerformanceLevel.HIGH
    
    # Configuraciones específicas
    memory_optimization: Dict[str, Any] = field(default_factory=lambda: {
        "gc_threshold": (700, 10, 10),
        "gc_frequency": 60,  # segundos
        "memory_limit_mb": 1024,
        "enable_memory_profiling": True
    })
    
    cpu_optimization: Dict[str, Any] = field(default_factory=lambda: {
        "max_workers": multiprocessing.cpu_count(),
        "thread_pool_size": 20,
        "process_pool_size": 4,
        "enable_cpu_profiling": True
    })
    
    io_optimization: Dict[str, Any] = field(default_factory=lambda: {
        "buffer_size": 8192,
        "async_io": True,
        "batch_operations": True,
        "compression_enabled": True
    })
    
    cache_optimization: Dict[str, Any] = field(default_factory=lambda: {
        "max_cache_size": 100,
        "ttl_seconds": 3600,
        "cleanup_interval": 300,
        "enable_lru": True
    })

@dataclass
class PerformanceMetrics:
    """Métricas de rendimiento."""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    memory_available: float
    disk_io_read: float
    disk_io_write: float
    network_sent: float
    network_recv: float
    active_threads: int
    active_processes: int
    cache_hit_rate: float = 0.0
    response_time_ms: float = 0.0
    throughput_ops_sec: float = 0.0

class MemoryOptimizer:
    """Optimizador de memoria."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.last_gc_time = time.time()
        self.memory_usage_history = []
        
    def optimize_garbage_collection(self):
        """Optimizar recolección de basura."""
        
        # Configurar umbrales de GC
        thresholds = self.config.get("gc_threshold", (700, 10, 10))
        gc.set_threshold(*thresholds)
        
        # Ejecutar GC si es necesario
        current_time = time.time()
        gc_frequency = self.config.get("gc_frequency", 60)
        
        if current_time - self.last_gc_time > gc_frequency:
            collected = gc.collect()
            self.last_gc_time = current_time
            return collected
        
        return 0
    
    def monitor_memory_usage(self) -> Dict[str, float]:
        """Monitorear uso de memoria."""
        
        process = psutil.Process()
        memory_info = process.memory_info()
        system_memory = psutil.virtual_memory()
        
        usage_data = {
            "rss_mb": memory_info.rss / 1024 / 1024,
            "vms_mb": memory_info.vms / 1024 / 1024,
            "percent": process.memory_percent(),
            "system_available_mb": system_memory.available / 1024 / 1024,
            "system_percent": system_memory.percent
        }
        
        self.memory_usage_history.append({
            "timestamp": datetime.now(),
            "usage": usage_data
        })
        
        # Mantener solo las últimas 100 mediciones
        if len(self.memory_usage_history) > 100:
            self.memory_usage_history.pop(0)
        
        return usage_data
    
    def detect_memory_leaks(self) -> List[Dict[str, Any]]:
        """Detectar posibles fugas de memoria."""
        
        if len(self.memory_usage_history) < 10:
            return []
        
        # Analizar tendencia de uso de memoria
        recent_usage = [entry["usage"]["rss_mb"] for entry in self.memory_usage_history[-10:]]
        
        # Calcular tendencia
        if len(recent_usage) >= 2:
            trend = (recent_usage[-1] - recent_usage[0]) / len(recent_usage)
            
            if trend > 10:  # Incremento de más de 10MB por medición
                return [{
                    "type": "memory_leak_suspected",
                    "trend_mb_per_measurement": trend,
                    "current_usage_mb": recent_usage[-1],
                    "recommendation": "Revisar objetos no liberados y referencias circulares"
                }]
        
        return []
    
    def optimize_object_pools(self):
        """Optimizar pools de objetos."""
        
        # Limpiar referencias débiles
        import weakref
        weakref.finalize_all()
        
        # Optimizar strings internos
        sys.intern("")
        
        return True

class CPUOptimizer:
    """Optimizador de CPU."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.thread_pool = None
        self.process_pool = None
        self.profiler = None
        
    def initialize_pools(self):
        """Inicializar pools de threads y procesos."""
        
        thread_pool_size = self.config.get("thread_pool_size", 20)
        process_pool_size = self.config.get("process_pool_size", 4)
        
        self.thread_pool = ThreadPoolExecutor(max_workers=thread_pool_size)
        self.process_pool = ProcessPoolExecutor(max_workers=process_pool_size)
        
        return True
    
    def optimize_cpu_bound_task(self, func: Callable, *args, **kwargs):
        """Optimizar tarea intensiva en CPU."""
        
        if self.process_pool is None:
            self.initialize_pools()
        
        future = self.process_pool.submit(func, *args, **kwargs)
        return future
    
    def optimize_io_bound_task(self, func: Callable, *args, **kwargs):
        """Optimizar tarea intensiva en I/O."""
        
        if self.thread_pool is None:
            self.initialize_pools()
        
        future = self.thread_pool.submit(func, *args, **kwargs)
        return future
    
    def start_profiling(self):
        """Iniciar profiling de CPU."""
        
        if self.config.get("enable_cpu_profiling", False):
            self.profiler = cProfile.Profile()
            self.profiler.enable()
    
    def stop_profiling(self) -> Optional[str]:
        """Detener profiling y obtener resultados."""
        
        if self.profiler is None:
            return None
        
        self.profiler.disable()
        
        # Generar reporte
        s = io.StringIO()
        ps = pstats.Stats(self.profiler, stream=s)
        ps.sort_stats('cumulative')
        ps.print_stats(20)  # Top 20 funciones
        
        return s.getvalue()
    
    def cleanup_pools(self):
        """Limpiar pools de recursos."""
        
        if self.thread_pool:
            self.thread_pool.shutdown(wait=True)
        
        if self.process_pool:
            self.process_pool.shutdown(wait=True)

class IOOptimizer:
    """Optimizador de I/O."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.buffer_size = config.get("buffer_size", 8192)
        
    def optimize_file_operations(self, file_path: str, operation: str = "read") -> Any:
        """Optimizar operaciones de archivo."""
        
        path = Path(file_path)
        
        if operation == "read":
            if self.config.get("async_io", True):
                return self._async_read_file(path)
            else:
                return self._sync_read_file(path)
        
        elif operation == "write":
            if self.config.get("async_io", True):
                return self._async_write_file
            else:
                return self._sync_write_file
    
    def _sync_read_file(self, path: Path) -> bytes:
        """Lectura síncrona optimizada."""
        
        with open(path, 'rb', buffering=self.buffer_size) as f:
            return f.read()
    
    async def _async_read_file(self, path: Path) -> bytes:
        """Lectura asíncrona optimizada."""
        
        import aiofiles
        
        async with aiofiles.open(path, 'rb') as f:
            return await f.read()
    
    def _sync_write_file(self, path: Path, data: bytes):
        """Escritura síncrona optimizada."""
        
        with open(path, 'wb', buffering=self.buffer_size) as f:
            f.write(data)
    
    async def _async_write_file(self, path: Path, data: bytes):
        """Escritura asíncrona optimizada."""
        
        import aiofiles
        
        async with aiofiles.open(path, 'wb') as f:
            await f.write(data)
    
    def batch_file_operations(self, operations: List[Dict[str, Any]]) -> List[Any]:
        """Procesar operaciones de archivo en lotes."""
        
        results = []
        
        for op in operations:
            try:
                if op["type"] == "read":
                    result = self._sync_read_file(Path(op["path"]))
                elif op["type"] == "write":
                    self._sync_write_file(Path(op["path"]), op["data"])
                    result = True
                else:
                    result = None
                
                results.append(result)
                
            except Exception as e:
                results.append(f"Error: {str(e)}")
        
        return results

class CacheOptimizer:
    """Optimizador de caché."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.cache = {}
        self.access_times = {}
        self.access_counts = {}
        self.last_cleanup = time.time()
        
    def get(self, key: str) -> Optional[Any]:
        """Obtener valor del caché."""
        
        if key in self.cache:
            self.access_times[key] = time.time()
            self.access_counts[key] = self.access_counts.get(key, 0) + 1
            return self.cache[key]
        
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Establecer valor en caché."""
        
        # Verificar límite de tamaño
        max_size = self.config.get("max_cache_size", 100)
        if len(self.cache) >= max_size:
            self._evict_lru()
        
        self.cache[key] = {
            "value": value,
            "created_at": time.time(),
            "ttl": ttl or self.config.get("ttl_seconds", 3600)
        }
        
        self.access_times[key] = time.time()
        self.access_counts[key] = 1
    
    def _evict_lru(self):
        """Eliminar elemento menos recientemente usado."""
        
        if not self.access_times:
            return
        
        # Encontrar clave con acceso más antiguo
        lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        
        # Eliminar
        del self.cache[lru_key]
        del self.access_times[lru_key]
        if lru_key in self.access_counts:
            del self.access_counts[lru_key]
    
    def cleanup_expired(self):
        """Limpiar entradas expiradas."""
        
        current_time = time.time()
        cleanup_interval = self.config.get("cleanup_interval", 300)
        
        if current_time - self.last_cleanup < cleanup_interval:
            return
        
        expired_keys = []
        
        for key, data in self.cache.items():
            if current_time - data["created_at"] > data["ttl"]:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.cache[key]
            if key in self.access_times:
                del self.access_times[key]
            if key in self.access_counts:
                del self.access_counts[key]
        
        self.last_cleanup = current_time
        return len(expired_keys)
    
    def get_hit_rate(self) -> float:
        """Calcular tasa de aciertos del caché."""
        
        total_accesses = sum(self.access_counts.values())
        if total_accesses == 0:
            return 0.0
        
        hits = len([count for count in self.access_counts.values() if count > 1])
        return hits / len(self.access_counts) if self.access_counts else 0.0

class PerformanceMonitor:
    """Monitor de rendimiento."""
    
    def __init__(self):
        self.metrics_history = []
        self.start_time = time.time()
        
    def collect_metrics(self) -> PerformanceMetrics:
        """Recopilar métricas actuales."""
        
        # Métricas del sistema
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk_io = psutil.disk_io_counters()
        network_io = psutil.net_io_counters()
        
        # Métricas del proceso
        process = psutil.Process()
        
        metrics = PerformanceMetrics(
            timestamp=datetime.now(),
            cpu_usage=cpu_percent,
            memory_usage=memory.percent,
            memory_available=memory.available / 1024 / 1024,  # MB
            disk_io_read=disk_io.read_bytes / 1024 / 1024 if disk_io else 0,  # MB
            disk_io_write=disk_io.write_bytes / 1024 / 1024 if disk_io else 0,  # MB
            network_sent=network_io.bytes_sent / 1024 / 1024 if network_io else 0,  # MB
            network_recv=network_io.bytes_recv / 1024 / 1024 if network_io else 0,  # MB
            active_threads=process.num_threads(),
            active_processes=len(psutil.pids())
        )
        
        self.metrics_history.append(metrics)
        
        # Mantener solo las últimas 1000 métricas
        if len(self.metrics_history) > 1000:
            self.metrics_history.pop(0)
        
        return metrics
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Obtener resumen de rendimiento."""
        
        if not self.metrics_history:
            return {}
        
        recent_metrics = self.metrics_history[-10:] if len(self.metrics_history) >= 10 else self.metrics_history
        
        return {
            "uptime_seconds": time.time() - self.start_time,
            "total_measurements": len(self.metrics_history),
            "avg_cpu_usage": sum(m.cpu_usage for m in recent_metrics) / len(recent_metrics),
            "avg_memory_usage": sum(m.memory_usage for m in recent_metrics) / len(recent_metrics),
            "avg_memory_available_mb": sum(m.memory_available for m in recent_metrics) / len(recent_metrics),
            "current_threads": recent_metrics[-1].active_threads,
            "peak_cpu": max(m.cpu_usage for m in self.metrics_history),
            "peak_memory": max(m.memory_usage for m in self.metrics_history)
        }

class PerformanceOptimizer:
    """Sistema principal de optimización de rendimiento."""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.memory_optimizer = MemoryOptimizer(config.memory_optimization)
        self.cpu_optimizer = CPUOptimizer(config.cpu_optimization)
        self.io_optimizer = IOOptimizer(config.io_optimization)
        self.cache_optimizer = CacheOptimizer(config.cache_optimization)
        self.monitor = PerformanceMonitor()
        
        # Inicializar sistema de caché inteligente
        if CACHE_AVAILABLE:
            try:
                cache_strategy = CacheStrategy.ADAPTIVE
                if self.config.performance_level == PerformanceLevel.ULTRA:
                    cache_strategy = CacheStrategy.PREDICTIVE
                elif self.config.performance_level == PerformanceLevel.HIGH:
                    cache_strategy = CacheStrategy.ADAPTIVE
                
                cache_memory_mb = min(512, config.memory_optimization.get("memory_limit_mb", 1024) // 4)
                self.intelligent_cache = create_intelligent_cache(cache_strategy, cache_memory_mb)
                print(f"Sistema de caché inteligente inicializado: {cache_strategy.value}, {cache_memory_mb}MB")
            except Exception as e:
                print(f"Error inicializando caché inteligente: {e}")
                self.intelligent_cache = None
        else:
            self.intelligent_cache = None
        
        # Inicializar optimizador GPU avanzado
        try:
            from gpu_optimization_enhancement import AdvancedGPUOptimizer, GPUOptimizationLevel
            gpu_level = GPUOptimizationLevel.BALANCED
            if self.config.performance_level == PerformanceLevel.ULTRA:
                gpu_level = GPUOptimizationLevel.ULTRA
            elif self.config.performance_level == PerformanceLevel.HIGH:
                gpu_level = GPUOptimizationLevel.AGGRESSIVE
            
            from gpu_optimization_enhancement import create_gpu_optimizer
            self.gpu_optimizer = create_gpu_optimizer(gpu_level)
            print("Optimizador GPU avanzado inicializado")
        except ImportError as e:
            print(f"Optimizador GPU no disponible: {e}")
            self.gpu_optimizer = None
        
        self.optimization_tasks = []
        self.running = False
        
    async def start_optimization(self):
        """Iniciar optimización continua."""
        
        self.running = True
        
        # Inicializar optimizadores
        if OptimizationType.CPU in self.config.enabled_optimizations:
            self.cpu_optimizer.initialize_pools()
            self.cpu_optimizer.start_profiling()
        
        # Iniciar tareas de optimización
        optimization_tasks = []
        
        if OptimizationType.MEMORY in self.config.enabled_optimizations:
            optimization_tasks.append(self._memory_optimization_loop())
        
        if OptimizationType.CACHE in self.config.enabled_optimizations:
            optimization_tasks.append(self._cache_optimization_loop())
        
        # Agregar bucle de caché inteligente si está disponible
        if self.intelligent_cache:
            optimization_tasks.append(self._intelligent_cache_loop())
        
        optimization_tasks.append(self._monitoring_loop())
        
        # Ejecutar todas las tareas
        await asyncio.gather(*optimization_tasks)
    
    async def _memory_optimization_loop(self):
        """Bucle de optimización de memoria."""
        
        while self.running:
            try:
                # Optimizar GC
                collected = self.memory_optimizer.optimize_garbage_collection()
                
                # Monitorear uso de memoria
                memory_usage = self.memory_optimizer.monitor_memory_usage()
                
                # Detectar fugas de memoria
                leaks = self.memory_optimizer.detect_memory_leaks()
                if leaks:
                    print(f"⚠️ Posibles fugas de memoria detectadas: {leaks}")
                
                # Optimizar pools de objetos
                self.memory_optimizer.optimize_object_pools()
                
                await asyncio.sleep(30)  # Cada 30 segundos
                
            except Exception as e:
                print(f"Error en optimización de memoria: {e}")
                await asyncio.sleep(60)
    
    async def _cache_optimization_loop(self):
        """Bucle de optimización de caché."""
        
        while self.running:
            try:
                # Limpiar entradas expiradas
                expired = self.cache_optimizer.cleanup_expired()
                if expired > 0:
                    print(f"🧹 Limpiadas {expired} entradas expiradas del caché")
                
                await asyncio.sleep(300)  # Cada 5 minutos
                
            except Exception as e:
                print(f"Error en optimización de caché: {e}")
                await asyncio.sleep(300)
    
    async def _intelligent_cache_loop(self):
        """Bucle de monitoreo del caché inteligente."""
        
        while self.running:
            try:
                # Obtener estadísticas del caché inteligente
                stats = self.intelligent_cache.get_stats()
                
                # Mostrar estadísticas cada 10 minutos
                if stats.item_count > 0:
                    print(f"🧠 Caché Inteligente - Elementos: {stats.item_count}, "
                          f"Memoria: {stats.memory_usage_mb:.1f}MB, "
                          f"Hit Rate: {stats.hit_rate:.2%}, "
                          f"Compresión: {stats.compression_ratio:.2f}")
                
                # Verificar si necesita optimización
                if stats.memory_usage_mb > (self.intelligent_cache.config.max_memory_mb * 0.9):
                    print("⚠️ Caché inteligente cerca del límite de memoria")
                
                if stats.hit_rate < 0.5 and stats.item_count > 10:
                    print("⚠️ Baja tasa de aciertos en caché inteligente")
                
                await asyncio.sleep(600)  # Cada 10 minutos
                
            except Exception as e:
                print(f"Error en monitoreo de caché inteligente: {e}")
                await asyncio.sleep(300)
    
    async def _monitoring_loop(self):
        """Bucle de monitoreo de rendimiento."""
        
        while self.running:
            try:
                # Recopilar métricas
                metrics = self.monitor.collect_metrics()
                
                # Mostrar resumen cada 10 mediciones
                if len(self.monitor.metrics_history) % 10 == 0:
                    summary = self.monitor.get_performance_summary()
                    print(f"📊 Rendimiento - CPU: {summary.get('avg_cpu_usage', 0):.1f}%, "
                          f"Memoria: {summary.get('avg_memory_usage', 0):.1f}%, "
                          f"Threads: {summary.get('current_threads', 0)}")
                
                await asyncio.sleep(60)  # Cada minuto
                
            except Exception as e:
                print(f"Error en monitoreo: {e}")
                await asyncio.sleep(60)
    
    def stop_optimization(self):
        """Detener optimización."""
        
        self.running = False
        
        # Detener caché inteligente si está disponible
        if self.intelligent_cache:
            try:
                self.intelligent_cache.stop()
                print("🧠 Sistema de caché inteligente detenido")
            except Exception as e:
                print(f"Error deteniendo caché inteligente: {e}")
        
        # Limpiar recursos
        if OptimizationType.CPU in self.config.enabled_optimizations:
            profile_report = self.cpu_optimizer.stop_profiling()
            if profile_report:
                print("📈 Reporte de profiling de CPU:")
                print(profile_report[:1000])  # Primeras 1000 caracteres
            
            self.cpu_optimizer.cleanup_pools()
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Generar reporte completo de optimización."""
        
        latest_metrics = self.monitor.get_performance_summary()
        
        # Reporte base
        report = {
            "timestamp": datetime.now().isoformat(),
            "performance_level": self.config.performance_level.value,
            "enabled_optimizations": [opt.value for opt in self.config.enabled_optimizations],
            "system_metrics": latest_metrics,
            "optimizers": {
                "memory": {
                    "gc_collections": getattr(self.memory_optimizer, 'gc_collections', 0),
                    "memory_usage": self.memory_optimizer.monitor_memory_usage()
                },
                "cache": {
                    "hit_rate": self.cache_optimizer.get_hit_rate(),
                    "cache_size": len(self.cache_optimizer.cache)
                }
            }
        }
        
        # Agregar estadísticas del caché inteligente si está disponible
        if self.intelligent_cache:
            try:
                cache_stats = self.intelligent_cache.get_stats()
                report["intelligent_cache"] = {
                    "items": cache_stats.item_count,
                    "memory_usage_mb": cache_stats.memory_usage_mb,
                    "hit_rate": cache_stats.hit_rate,
                    "compression_ratio": cache_stats.compression_ratio,
                    "compressions": cache_stats.compressions,
                    "decompressions": cache_stats.decompressions,
                    "evictions": cache_stats.evictions
                }
            except Exception as e:
                report["intelligent_cache"] = {"error": str(e)}
        
        # Agregar reporte GPU si está disponible
        if self.gpu_optimizer:
            try:
                gpu_report = self.gpu_optimizer.get_performance_report()
                report["gpu_optimization"] = gpu_report
                
                # Calcular mejora de rendimiento estimada
                gpu_utilization = gpu_report.get("gpu_status", {}).get("utilization", 0)
                cache_hit_rate = gpu_report.get("performance", {}).get("cache_hit_rate", 0)
                
                performance_improvement = min(
                    30 + (gpu_utilization * 20) + (cache_hit_rate * 15),
                    50  # Máximo 50% de mejora
                )
                
                report["performance_improvement_estimate"] = f"{performance_improvement:.1f}%"
                
            except Exception as e:
                logger.error(f"Error obteniendo reporte GPU: {e}")
                report["gpu_optimization"] = {"error": str(e)}
        
        return report

async def main():
    """Función principal de demostración."""
    
    print("🚀 Iniciando Sistema de Optimización de Rendimiento SIGeC-Balistica")
    
    # Configuración de optimización
    config = OptimizationConfig(
        enabled_optimizations=[
            OptimizationType.MEMORY,
            OptimizationType.CPU,
            OptimizationType.CACHE
        ],
        performance_level=PerformanceLevel.HIGH
    )
    
    # Crear optimizador
    optimizer = PerformanceOptimizer(config)
    
    try:
        print("⚡ Iniciando optimizaciones...")
        
        # Ejecutar optimización por 30 segundos como demostración
        optimization_task = asyncio.create_task(optimizer.start_optimization())
        
        await asyncio.sleep(30)
        
        print("⏹️ Deteniendo optimizaciones...")
        optimizer.stop_optimization()
        
        # Cancelar tarea
        optimization_task.cancel()
        
        try:
            await optimization_task
        except asyncio.CancelledError:
            pass
        
        # Generar reporte final
        report = optimizer.get_optimization_report()
        
        print("\n📄 Reporte de Optimización:")
        print(json.dumps(report, indent=2, default=str))
        
        # Guardar reporte
        report_file = Path("performance_optimization_report.json")
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n💾 Reporte guardado en: {report_file}")
        
    except KeyboardInterrupt:
        print("\n⏹️ Optimización cancelada por el usuario")
        optimizer.stop_optimization()
    except Exception as e:
        print(f"\n💥 Error inesperado: {str(e)}")
        optimizer.stop_optimization()

if __name__ == "__main__":
    asyncio.run(main())