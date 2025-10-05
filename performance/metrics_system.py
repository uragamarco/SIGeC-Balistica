#!/usr/bin/env python3
"""
Sistema de métricas de rendimiento para SIGeC-Balisticar.
Monitorea y recopila métricas en tiempo real del sistema.
"""

import time
import threading
import psutil
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from collections import deque, defaultdict
import json
from pathlib import Path
import statistics

# Importar sistema de notificaciones
try:
    from core.notification_system import notify_warning, notify_critical, NotificationType
    NOTIFICATIONS_AVAILABLE = True
except ImportError:
    NOTIFICATIONS_AVAILABLE = False

@dataclass
class MetricValue:
    """Representa un valor de métrica con timestamp."""
    timestamp: datetime
    value: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MetricThreshold:
    """Define umbrales para alertas de métricas."""
    warning_threshold: float
    critical_threshold: float
    comparison: str = "greater"  # "greater", "less", "equal"
    enabled: bool = True

class MetricCollector:
    """Recopilador base para métricas."""
    
    def __init__(self, name: str, collection_interval: float = 1.0):
        self.name = name
        self.collection_interval = collection_interval
        self.enabled = True
        self.last_collection = 0
    
    def should_collect(self) -> bool:
        """Determina si es momento de recopilar la métrica."""
        current_time = time.time()
        return (current_time - self.last_collection) >= self.collection_interval
    
    def collect(self) -> Optional[float]:
        """Recopila el valor de la métrica. Debe ser implementado por subclases."""
        raise NotImplementedError
    
    def update_collection_time(self):
        """Actualiza el tiempo de la última recopilación."""
        self.last_collection = time.time()

class CPUCollector(MetricCollector):
    """Recopilador de métricas de CPU."""
    
    def collect(self) -> Optional[float]:
        try:
            return psutil.cpu_percent(interval=0.1)
        except Exception:
            return None

class MemoryCollector(MetricCollector):
    """Recopilador de métricas de memoria."""
    
    def collect(self) -> Optional[float]:
        try:
            return psutil.virtual_memory().percent
        except Exception:
            return None

class DiskCollector(MetricCollector):
    """Recopilador de métricas de disco."""
    
    def __init__(self, name: str, path: str = "/", collection_interval: float = 5.0):
        super().__init__(name, collection_interval)
        self.path = path
    
    def collect(self) -> Optional[float]:
        try:
            return psutil.disk_usage(self.path).percent
        except Exception:
            return None

class NetworkCollector(MetricCollector):
    """Recopilador de métricas de red."""
    
    def __init__(self, name: str, collection_interval: float = 2.0):
        super().__init__(name, collection_interval)
        self.last_bytes_sent = 0
        self.last_bytes_recv = 0
        self.last_time = time.time()
    
    def collect(self) -> Optional[Dict[str, float]]:
        try:
            net_io = psutil.net_io_counters()
            current_time = time.time()
            
            if self.last_bytes_sent > 0:  # No es la primera medición
                time_diff = current_time - self.last_time
                bytes_sent_diff = net_io.bytes_sent - self.last_bytes_sent
                bytes_recv_diff = net_io.bytes_recv - self.last_bytes_recv
                
                # Calcular velocidad en MB/s
                send_speed = (bytes_sent_diff / time_diff) / (1024 * 1024)
                recv_speed = (bytes_recv_diff / time_diff) / (1024 * 1024)
                
                result = {
                    'send_speed_mbps': send_speed,
                    'recv_speed_mbps': recv_speed,
                    'total_speed_mbps': send_speed + recv_speed
                }
            else:
                result = {
                    'send_speed_mbps': 0.0,
                    'recv_speed_mbps': 0.0,
                    'total_speed_mbps': 0.0
                }
            
            # Actualizar valores para la próxima medición
            self.last_bytes_sent = net_io.bytes_sent
            self.last_bytes_recv = net_io.bytes_recv
            self.last_time = current_time
            
            return result
        except Exception:
            return None

class GPUCollector(MetricCollector):
    """Recopilador de métricas de GPU."""
    
    def __init__(self, name: str, collection_interval: float = 2.0):
        super().__init__(name, collection_interval)
        self.gpu_available = self._check_gpu_availability()
    
    def _check_gpu_availability(self) -> bool:
        """Verifica si hay GPU disponible."""
        try:
            import cupy as cp
            return cp.cuda.is_available()
        except ImportError:
            return False
    
    def collect(self) -> Optional[Dict[str, float]]:
        if not self.gpu_available:
            return None
        
        try:
            import cupy as cp
            
            # Obtener información de memoria GPU
            mempool = cp.get_default_memory_pool()
            total_bytes = mempool.total_bytes()
            used_bytes = mempool.used_bytes()
            
            # Calcular porcentaje de uso
            if total_bytes > 0:
                usage_percent = (used_bytes / total_bytes) * 100
            else:
                usage_percent = 0.0
            
            return {
                'memory_usage_percent': usage_percent,
                'memory_used_mb': used_bytes / (1024 * 1024),
                'memory_total_mb': total_bytes / (1024 * 1024)
            }
        except Exception:
            return None

class PerformanceMetricsSystem:
    """Sistema principal de métricas de rendimiento."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.collectors: Dict[str, MetricCollector] = {}
        self.thresholds: Dict[str, MetricThreshold] = {}
        self.running = False
        self.thread = None
        self.lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
        
        # Configuración por defecto
        self.collection_interval = self.config.get('collection_interval', 1.0)
        self.max_history = self.config.get('max_history', 1000)
        self.enable_alerts = self.config.get('enable_alerts', True)
        
        # Configurar recopiladores por defecto
        self._setup_default_collectors()
        self._setup_default_thresholds()
    
    def _setup_default_collectors(self):
        """Configura los recopiladores de métricas por defecto."""
        
        self.collectors['cpu_usage'] = CPUCollector('cpu_usage', 1.0)
        self.collectors['memory_usage'] = MemoryCollector('memory_usage', 1.0)
        self.collectors['disk_usage'] = DiskCollector('disk_usage', '/', 5.0)
        self.collectors['network'] = NetworkCollector('network', 2.0)
        self.collectors['gpu'] = GPUCollector('gpu', 2.0)
    
    def _setup_default_thresholds(self):
        """Configura umbrales por defecto para alertas."""
        
        self.thresholds['cpu_usage'] = MetricThreshold(80.0, 95.0)
        self.thresholds['memory_usage'] = MetricThreshold(85.0, 95.0)
        self.thresholds['disk_usage'] = MetricThreshold(85.0, 95.0)
        self.thresholds['gpu.memory_usage_percent'] = MetricThreshold(90.0, 98.0)
    
    def start_monitoring(self):
        """Inicia el monitoreo de métricas."""
        
        if self.running:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.thread.start()
        
        self.logger.info("Sistema de métricas iniciado")
    
    def stop_monitoring(self):
        """Detiene el monitoreo de métricas."""
        
        self.running = False
        if self.thread:
            self.thread.join(timeout=5.0)
        
        self.logger.info("Sistema de métricas detenido")
    
    def _monitoring_loop(self):
        """Bucle principal de monitoreo."""
        
        while self.running:
            try:
                self._collect_all_metrics()
                time.sleep(self.collection_interval)
            except Exception as e:
                self.logger.error(f"Error en bucle de monitoreo: {e}")
                time.sleep(1.0)
    
    def _collect_all_metrics(self):
        """Recopila todas las métricas configuradas."""
        
        current_time = datetime.now()
        
        for name, collector in self.collectors.items():
            if not collector.enabled or not collector.should_collect():
                continue
            
            try:
                value = collector.collect()
                if value is not None:
                    self._store_metric(name, value, current_time)
                    self._check_thresholds(name, value)
                
                collector.update_collection_time()
                
            except Exception as e:
                self.logger.error(f"Error recopilando métrica {name}: {e}")
    
    def _store_metric(self, name: str, value: Any, timestamp: datetime):
        """Almacena un valor de métrica."""
        
        with self.lock:
            if isinstance(value, dict):
                # Métricas compuestas (ej: GPU, red)
                for sub_name, sub_value in value.items():
                    full_name = f"{name}.{sub_name}"
                    metric_value = MetricValue(timestamp, float(sub_value))
                    self.metrics[full_name].append(metric_value)
            else:
                # Métrica simple
                metric_value = MetricValue(timestamp, float(value))
                self.metrics[name].append(metric_value)
    
    def _check_thresholds(self, name: str, value: Any):
        """Verifica umbrales y envía alertas si es necesario."""
        
        if not self.enable_alerts or not NOTIFICATIONS_AVAILABLE:
            return
        
        # Verificar métricas simples
        if isinstance(value, (int, float)):
            self._check_single_threshold(name, float(value))
        elif isinstance(value, dict):
            # Verificar métricas compuestas
            for sub_name, sub_value in value.items():
                full_name = f"{name}.{sub_name}"
                self._check_single_threshold(full_name, float(sub_value))
    
    def _check_single_threshold(self, metric_name: str, value: float):
        """Verifica umbral para una métrica individual."""
        
        threshold = self.thresholds.get(metric_name)
        if not threshold or not threshold.enabled:
            return
        
        # Determinar si se excede el umbral
        exceeds_critical = False
        exceeds_warning = False
        
        if threshold.comparison == "greater":
            exceeds_critical = value > threshold.critical_threshold
            exceeds_warning = value > threshold.warning_threshold
        elif threshold.comparison == "less":
            exceeds_critical = value < threshold.critical_threshold
            exceeds_warning = value < threshold.warning_threshold
        
        # Enviar notificaciones
        if exceeds_critical:
            notify_critical(
                f"Umbral Crítico Excedido: {metric_name}",
                f"Valor actual: {value:.2f}, Umbral crítico: {threshold.critical_threshold}",
                "metrics_system",
                metadata={'metric': metric_name, 'value': value, 'threshold': threshold.critical_threshold}
            )
        elif exceeds_warning:
            notify_warning(
                f"Umbral de Advertencia Excedido: {metric_name}",
                f"Valor actual: {value:.2f}, Umbral advertencia: {threshold.warning_threshold}",
                "metrics_system",
                metadata={'metric': metric_name, 'value': value, 'threshold': threshold.warning_threshold}
            )
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Obtiene las métricas actuales."""
        
        current_metrics = {}
        
        with self.lock:
            for name, metric_history in self.metrics.items():
                if metric_history:
                    latest = metric_history[-1]
                    current_metrics[name] = {
                        'value': latest.value,
                        'timestamp': latest.timestamp.isoformat(),
                        'metadata': latest.metadata
                    }
        
        return current_metrics
    
    def get_metric_history(self, metric_name: str, 
                          duration: Optional[timedelta] = None,
                          limit: Optional[int] = None) -> List[MetricValue]:
        """Obtiene el historial de una métrica."""
        
        with self.lock:
            if metric_name not in self.metrics:
                return []
            
            history = list(self.metrics[metric_name])
        
        # Filtrar por duración si se especifica
        if duration:
            cutoff_time = datetime.now() - duration
            history = [m for m in history if m.timestamp >= cutoff_time]
        
        # Limitar cantidad si se especifica
        if limit:
            history = history[-limit:]
        
        return history
    
    def get_metric_statistics(self, metric_name: str, 
                            duration: Optional[timedelta] = None) -> Dict[str, float]:
        """Obtiene estadísticas de una métrica."""
        
        history = self.get_metric_history(metric_name, duration)
        
        if not history:
            return {}
        
        values = [m.value for m in history]
        
        return {
            'count': len(values),
            'min': min(values),
            'max': max(values),
            'mean': statistics.mean(values),
            'median': statistics.median(values),
            'stdev': statistics.stdev(values) if len(values) > 1 else 0.0,
            'latest': values[-1] if values else 0.0
        }
    
    def export_metrics(self, filepath: str, 
                      duration: Optional[timedelta] = None):
        """Exporta métricas a un archivo JSON."""
        
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'duration': duration.total_seconds() if duration else None,
            'metrics': {}
        }
        
        with self.lock:
            for name in self.metrics.keys():
                history = self.get_metric_history(name, duration)
                export_data['metrics'][name] = [
                    {
                        'timestamp': m.timestamp.isoformat(),
                        'value': m.value,
                        'metadata': m.metadata
                    }
                    for m in history
                ]
        
        # Guardar archivo
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Métricas exportadas a {filepath}")
    
    def add_custom_collector(self, name: str, collector: MetricCollector):
        """Agrega un recopilador personalizado."""
        self.collectors[name] = collector
    
    def set_threshold(self, metric_name: str, threshold: MetricThreshold):
        """Establece un umbral para una métrica."""
        self.thresholds[metric_name] = threshold
    
    def get_system_health_score(self) -> float:
        """Calcula un puntaje de salud del sistema (0-100)."""
        
        current_metrics = self.get_current_metrics()
        
        if not current_metrics:
            return 100.0  # Sin métricas, asumimos que está bien
        
        scores = []
        
        # CPU
        if 'cpu_usage' in current_metrics:
            cpu_usage = current_metrics['cpu_usage']['value']
            cpu_score = max(0, 100 - cpu_usage)
            scores.append(cpu_score)
        
        # Memoria
        if 'memory_usage' in current_metrics:
            mem_usage = current_metrics['memory_usage']['value']
            mem_score = max(0, 100 - mem_usage)
            scores.append(mem_score)
        
        # Disco
        if 'disk_usage' in current_metrics:
            disk_usage = current_metrics['disk_usage']['value']
            disk_score = max(0, 100 - disk_usage)
            scores.append(disk_score)
        
        # GPU (si está disponible)
        if 'gpu.memory_usage_percent' in current_metrics:
            gpu_usage = current_metrics['gpu.memory_usage_percent']['value']
            gpu_score = max(0, 100 - gpu_usage)
            scores.append(gpu_score)
        
        # Calcular promedio ponderado
        if scores:
            return sum(scores) / len(scores)
        else:
            return 100.0

# Instancia global del sistema de métricas
_metrics_system = None

def get_metrics_system() -> PerformanceMetricsSystem:
    """Obtiene la instancia global del sistema de métricas."""
    global _metrics_system
    if _metrics_system is None:
        _metrics_system = PerformanceMetricsSystem()
    return _metrics_system

# Funciones de conveniencia
def start_metrics_monitoring():
    """Inicia el monitoreo de métricas."""
    get_metrics_system().start_monitoring()

def stop_metrics_monitoring():
    """Detiene el monitoreo de métricas."""
    get_metrics_system().stop_monitoring()

def get_current_system_metrics() -> Dict[str, Any]:
    """Obtiene las métricas actuales del sistema."""
    return get_metrics_system().get_current_metrics()

def get_system_health_score() -> float:
    """Obtiene el puntaje de salud del sistema."""
    return get_metrics_system().get_system_health_score()