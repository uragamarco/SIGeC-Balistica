"""
Monitor de GPU para SEACABAr
Sistema de monitoreo continuo de memoria y rendimiento GPU

Este módulo implementa:
- Monitoreo en tiempo real de uso de memoria GPU
- Alertas automáticas de memoria baja
- Estadísticas de uso por operación
- Métricas de rendimiento GPU
- Dashboard de monitoreo

Autor: Sistema SEACABA
Fecha: 2024
"""

import logging
import threading
import time
import queue
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque, defaultdict
import json
import os

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None

logger = logging.getLogger(__name__)

@dataclass
class GPUMetrics:
    """Métricas de GPU en un momento específico"""
    timestamp: datetime
    memory_used_bytes: int = 0
    memory_total_bytes: int = 0
    memory_free_bytes: int = 0
    memory_usage_percent: float = 0.0
    gpu_utilization_percent: float = 0.0
    temperature_celsius: Optional[float] = None
    power_usage_watts: Optional[float] = None
    active_operations: int = 0
    operation_queue_size: int = 0

@dataclass
class OperationMetrics:
    """Métricas de una operación específica"""
    operation_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    memory_before_bytes: int = 0
    memory_after_bytes: int = 0
    memory_peak_bytes: int = 0
    duration_seconds: float = 0.0
    success: bool = True
    error_message: Optional[str] = None

@dataclass
class AlertConfig:
    """Configuración de alertas"""
    memory_threshold_percent: float = 85.0
    temperature_threshold_celsius: float = 80.0
    utilization_threshold_percent: float = 95.0
    consecutive_alerts_required: int = 3
    alert_cooldown_seconds: int = 300  # 5 minutos
    enable_email_alerts: bool = False
    email_recipients: List[str] = field(default_factory=list)

class GPUMonitor:
    """
    Monitor de GPU con alertas y métricas en tiempo real
    """
    
    def __init__(self, 
                 monitoring_interval: float = 1.0,
                 history_size: int = 3600,  # 1 hora a 1 segundo por muestra
                 alert_config: Optional[AlertConfig] = None):
        """
        Inicializar monitor de GPU
        
        Args:
            monitoring_interval: Intervalo de monitoreo en segundos
            history_size: Número de muestras a mantener en historial
            alert_config: Configuración de alertas
        """
        self.monitoring_interval = monitoring_interval
        self.history_size = history_size
        self.alert_config = alert_config or AlertConfig()
        
        # Estructuras de datos
        self.metrics_history: deque = deque(maxlen=history_size)
        self.operation_metrics: Dict[str, List[OperationMetrics]] = defaultdict(list)
        self.active_operations: Dict[str, OperationMetrics] = {}
        
        # Control de monitoreo
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        
        # Sistema de alertas
        self._alert_callbacks: List[Callable] = []
        self._alert_history: deque = deque(maxlen=100)
        self._last_alert_time: Dict[str, datetime] = {}
        self._consecutive_alerts: Dict[str, int] = defaultdict(int)
        
        # Cola de eventos
        self._event_queue: queue.Queue = queue.Queue()
        
        logger.info("GPUMonitor inicializado")
    
    def start_monitoring(self):
        """Iniciar monitoreo continuo"""
        if self._monitoring:
            logger.warning("El monitoreo ya está activo")
            return
            
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        
        logger.info("Monitoreo de GPU iniciado")
    
    def stop_monitoring(self):
        """Detener monitoreo"""
        self._monitoring = False
        
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5)
        
        logger.info("Monitoreo de GPU detenido")
    
    def _monitor_loop(self):
        """Bucle principal de monitoreo"""
        while self._monitoring:
            try:
                # Recopilar métricas
                metrics = self._collect_metrics()
                
                with self._lock:
                    self.metrics_history.append(metrics)
                
                # Verificar alertas
                self._check_alerts(metrics)
                
                # Procesar eventos pendientes
                self._process_events()
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error en bucle de monitoreo: {e}")
                time.sleep(self.monitoring_interval)
    
    def _collect_metrics(self) -> GPUMetrics:
        """Recopilar métricas actuales de GPU"""
        metrics = GPUMetrics(timestamp=datetime.now())
        
        if not CUPY_AVAILABLE:
            return metrics
        
        try:
            # Métricas de memoria CuPy
            mempool = cp.get_default_memory_pool()
            metrics.memory_used_bytes = mempool.used_bytes()
            
            # Intentar obtener memoria total del dispositivo
            try:
                device = cp.cuda.Device()
                total_memory = device.mem_info[1]  # Total memory
                free_memory = device.mem_info[0]   # Free memory
                
                metrics.memory_total_bytes = total_memory
                metrics.memory_free_bytes = free_memory
                metrics.memory_usage_percent = (metrics.memory_used_bytes / total_memory) * 100
                
            except Exception:
                # Fallback si no se puede obtener info del dispositivo
                if hasattr(cp.cuda, 'runtime'):
                    try:
                        free, total = cp.cuda.runtime.memGetInfo()
                        metrics.memory_total_bytes = total
                        metrics.memory_free_bytes = free
                        metrics.memory_usage_percent = ((total - free) / total) * 100
                    except Exception:
                        pass
            
            # Métricas de operaciones activas
            with self._lock:
                metrics.active_operations = len(self.active_operations)
                metrics.operation_queue_size = self._event_queue.qsize()
            
            # Intentar obtener métricas adicionales con nvidia-ml-py si está disponible
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                
                # Utilización GPU
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                metrics.gpu_utilization_percent = util.gpu
                
                # Temperatura
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                metrics.temperature_celsius = temp
                
                # Consumo de energía
                try:
                    power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW to W
                    metrics.power_usage_watts = power
                except Exception:
                    pass
                    
            except ImportError:
                pass
            except Exception as e:
                logger.debug(f"Error obteniendo métricas adicionales: {e}")
        
        except Exception as e:
            logger.warning(f"Error recopilando métricas GPU: {e}")
        
        return metrics
    
    def _check_alerts(self, metrics: GPUMetrics):
        """Verificar y generar alertas basadas en métricas"""
        alerts_to_send = []
        
        # Alerta de memoria
        if metrics.memory_usage_percent > self.alert_config.memory_threshold_percent:
            alert_type = "memory_high"
            self._consecutive_alerts[alert_type] += 1
            
            if (self._consecutive_alerts[alert_type] >= self.alert_config.consecutive_alerts_required and
                self._should_send_alert(alert_type)):
                
                alerts_to_send.append({
                    'type': alert_type,
                    'severity': 'warning',
                    'message': f"Uso de memoria GPU alto: {metrics.memory_usage_percent:.1f}%",
                    'metrics': metrics,
                    'timestamp': datetime.now()
                })
        else:
            self._consecutive_alerts["memory_high"] = 0
        
        # Alerta de temperatura
        if (metrics.temperature_celsius and 
            metrics.temperature_celsius > self.alert_config.temperature_threshold_celsius):
            
            alert_type = "temperature_high"
            self._consecutive_alerts[alert_type] += 1
            
            if (self._consecutive_alerts[alert_type] >= self.alert_config.consecutive_alerts_required and
                self._should_send_alert(alert_type)):
                
                alerts_to_send.append({
                    'type': alert_type,
                    'severity': 'critical',
                    'message': f"Temperatura GPU alta: {metrics.temperature_celsius:.1f}°C",
                    'metrics': metrics,
                    'timestamp': datetime.now()
                })
        else:
            self._consecutive_alerts["temperature_high"] = 0
        
        # Alerta de utilización
        if metrics.gpu_utilization_percent > self.alert_config.utilization_threshold_percent:
            alert_type = "utilization_high"
            self._consecutive_alerts[alert_type] += 1
            
            if (self._consecutive_alerts[alert_type] >= self.alert_config.consecutive_alerts_required and
                self._should_send_alert(alert_type)):
                
                alerts_to_send.append({
                    'type': alert_type,
                    'severity': 'info',
                    'message': f"Utilización GPU alta: {metrics.gpu_utilization_percent:.1f}%",
                    'metrics': metrics,
                    'timestamp': datetime.now()
                })
        else:
            self._consecutive_alerts["utilization_high"] = 0
        
        # Enviar alertas
        for alert in alerts_to_send:
            self._send_alert(alert)
    
    def _should_send_alert(self, alert_type: str) -> bool:
        """Verificar si se debe enviar una alerta (cooldown)"""
        now = datetime.now()
        last_alert = self._last_alert_time.get(alert_type)
        
        if not last_alert:
            return True
        
        cooldown = timedelta(seconds=self.alert_config.alert_cooldown_seconds)
        return now - last_alert > cooldown
    
    def _send_alert(self, alert: Dict[str, Any]):
        """Enviar alerta a todos los callbacks registrados"""
        with self._lock:
            self._alert_history.append(alert)
            self._last_alert_time[alert['type']] = alert['timestamp']
        
        # Llamar callbacks
        for callback in self._alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Error en callback de alerta: {e}")
        
        # Log de la alerta
        severity = alert['severity']
        message = alert['message']
        
        if severity == 'critical':
            logger.critical(f"ALERTA GPU: {message}")
        elif severity == 'warning':
            logger.warning(f"ALERTA GPU: {message}")
        else:
            logger.info(f"ALERTA GPU: {message}")
    
    def _process_events(self):
        """Procesar eventos pendientes en la cola"""
        while not self._event_queue.empty():
            try:
                event = self._event_queue.get_nowait()
                self._handle_event(event)
            except queue.Empty:
                break
            except Exception as e:
                logger.error(f"Error procesando evento: {e}")
    
    def _handle_event(self, event: Dict[str, Any]):
        """Manejar un evento específico"""
        event_type = event.get('type')
        
        if event_type == 'operation_start':
            self._handle_operation_start(event)
        elif event_type == 'operation_end':
            self._handle_operation_end(event)
    
    def _handle_operation_start(self, event: Dict[str, Any]):
        """Manejar inicio de operación"""
        operation_id = event['operation_id']
        operation_name = event['operation_name']
        
        metrics = OperationMetrics(
            operation_name=operation_name,
            start_time=datetime.now(),
            memory_before_bytes=self._get_current_memory_usage()
        )
        
        with self._lock:
            self.active_operations[operation_id] = metrics
    
    def _handle_operation_end(self, event: Dict[str, Any]):
        """Manejar fin de operación"""
        operation_id = event['operation_id']
        success = event.get('success', True)
        error_message = event.get('error_message')
        
        with self._lock:
            if operation_id in self.active_operations:
                metrics = self.active_operations.pop(operation_id)
                metrics.end_time = datetime.now()
                metrics.duration_seconds = (metrics.end_time - metrics.start_time).total_seconds()
                metrics.memory_after_bytes = self._get_current_memory_usage()
                metrics.success = success
                metrics.error_message = error_message
                
                # Calcular pico de memoria (aproximado)
                metrics.memory_peak_bytes = max(metrics.memory_before_bytes, metrics.memory_after_bytes)
                
                # Agregar a historial
                self.operation_metrics[metrics.operation_name].append(metrics)
                
                # Mantener solo las últimas 1000 operaciones por tipo
                if len(self.operation_metrics[metrics.operation_name]) > 1000:
                    self.operation_metrics[metrics.operation_name] = \
                        self.operation_metrics[metrics.operation_name][-1000:]
    
    def _get_current_memory_usage(self) -> int:
        """Obtener uso actual de memoria GPU"""
        if CUPY_AVAILABLE:
            try:
                mempool = cp.get_default_memory_pool()
                return mempool.used_bytes()
            except Exception:
                pass
        return 0
    
    def start_operation(self, operation_name: str) -> str:
        """
        Registrar inicio de operación
        
        Args:
            operation_name: Nombre de la operación
            
        Returns:
            ID único de la operación
        """
        operation_id = f"{operation_name}_{int(time.time() * 1000000)}"
        
        event = {
            'type': 'operation_start',
            'operation_id': operation_id,
            'operation_name': operation_name
        }
        
        self._event_queue.put(event)
        return operation_id
    
    def end_operation(self, operation_id: str, success: bool = True, error_message: Optional[str] = None):
        """
        Registrar fin de operación
        
        Args:
            operation_id: ID de la operación
            success: Si la operación fue exitosa
            error_message: Mensaje de error si falló
        """
        event = {
            'type': 'operation_end',
            'operation_id': operation_id,
            'success': success,
            'error_message': error_message
        }
        
        self._event_queue.put(event)
    
    def add_alert_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Agregar callback para alertas"""
        self._alert_callbacks.append(callback)
    
    def remove_alert_callback(self, callback: Callable):
        """Remover callback de alertas"""
        if callback in self._alert_callbacks:
            self._alert_callbacks.remove(callback)
    
    def get_current_metrics(self) -> Optional[GPUMetrics]:
        """Obtener métricas actuales"""
        with self._lock:
            if self.metrics_history:
                return self.metrics_history[-1]
        return None
    
    def get_metrics_history(self, minutes: int = 60) -> List[GPUMetrics]:
        """
        Obtener historial de métricas
        
        Args:
            minutes: Minutos de historial a obtener
            
        Returns:
            Lista de métricas
        """
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        
        with self._lock:
            return [m for m in self.metrics_history if m.timestamp >= cutoff_time]
    
    def get_operation_stats(self, operation_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Obtener estadísticas de operaciones
        
        Args:
            operation_name: Nombre específico de operación o None para todas
            
        Returns:
            Diccionario con estadísticas
        """
        with self._lock:
            if operation_name:
                operations = self.operation_metrics.get(operation_name, [])
                operation_names = [operation_name]
            else:
                operations = []
                for ops in self.operation_metrics.values():
                    operations.extend(ops)
                operation_names = list(self.operation_metrics.keys())
        
        if not operations:
            return {'operation_names': operation_names, 'total_operations': 0}
        
        # Calcular estadísticas
        durations = [op.duration_seconds for op in operations if op.end_time]
        memory_usage = [op.memory_after_bytes - op.memory_before_bytes for op in operations if op.end_time]
        success_count = sum(1 for op in operations if op.success)
        
        stats = {
            'operation_names': operation_names,
            'total_operations': len(operations),
            'successful_operations': success_count,
            'failed_operations': len(operations) - success_count,
            'success_rate': success_count / len(operations) if operations else 0,
        }
        
        if durations:
            stats.update({
                'avg_duration_seconds': sum(durations) / len(durations),
                'min_duration_seconds': min(durations),
                'max_duration_seconds': max(durations),
            })
        
        if memory_usage:
            stats.update({
                'avg_memory_usage_bytes': sum(memory_usage) / len(memory_usage),
                'min_memory_usage_bytes': min(memory_usage),
                'max_memory_usage_bytes': max(memory_usage),
            })
        
        return stats
    
    def get_alert_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Obtener historial de alertas"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with self._lock:
            return [alert for alert in self._alert_history 
                   if alert['timestamp'] >= cutoff_time]
    
    def export_metrics(self, filepath: str, format: str = 'json'):
        """
        Exportar métricas a archivo
        
        Args:
            filepath: Ruta del archivo
            format: Formato ('json' o 'csv')
        """
        with self._lock:
            data = {
                'metrics_history': [
                    {
                        'timestamp': m.timestamp.isoformat(),
                        'memory_used_bytes': m.memory_used_bytes,
                        'memory_total_bytes': m.memory_total_bytes,
                        'memory_usage_percent': m.memory_usage_percent,
                        'gpu_utilization_percent': m.gpu_utilization_percent,
                        'temperature_celsius': m.temperature_celsius,
                        'active_operations': m.active_operations
                    }
                    for m in self.metrics_history
                ],
                'operation_stats': self.get_operation_stats(),
                'alert_history': [
                    {
                        **alert,
                        'timestamp': alert['timestamp'].isoformat()
                    }
                    for alert in self._alert_history
                ]
            }
        
        if format.lower() == 'json':
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
        else:
            raise ValueError(f"Formato no soportado: {format}")
        
        logger.info(f"Métricas exportadas a {filepath}")

# Instancia global del monitor
_gpu_monitor = None

def get_gpu_monitor(**kwargs) -> GPUMonitor:
    """Obtener instancia global del monitor GPU"""
    global _gpu_monitor
    if _gpu_monitor is None:
        _gpu_monitor = GPUMonitor(**kwargs)
    return _gpu_monitor

def start_gpu_monitoring():
    """Iniciar monitoreo global de GPU"""
    monitor = get_gpu_monitor()
    monitor.start_monitoring()

def stop_gpu_monitoring():
    """Detener monitoreo global de GPU"""
    monitor = get_gpu_monitor()
    monitor.stop_monitoring()

def get_gpu_metrics() -> Optional[GPUMetrics]:
    """Obtener métricas actuales de GPU"""
    monitor = get_gpu_monitor()
    return monitor.get_current_metrics()

def monitor_operation(operation_name: str):
    """Decorator para monitorear operaciones automáticamente"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            monitor = get_gpu_monitor()
            operation_id = monitor.start_operation(operation_name)
            
            try:
                result = func(*args, **kwargs)
                monitor.end_operation(operation_id, success=True)
                return result
            except Exception as e:
                monitor.end_operation(operation_id, success=False, error_message=str(e))
                raise
        
        return wrapper
    return decorator