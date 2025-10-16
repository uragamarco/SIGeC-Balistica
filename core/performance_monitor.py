#!/usr/bin/env python3
"""
Sistema de Monitoreo de Rendimiento Automático para SIGeC-Balistica
Integra telemetría y monitoreo avanzado para operaciones críticas del sistema.

Este módulo proporciona:
- Decoradores inteligentes para monitoreo automático de rendimiento
- Umbrales configurables para diferentes tipos de operaciones
- Alertas automáticas cuando se exceden los umbrales
- Integración con sistemas de telemetría y monitoreo avanzado
- Recomendaciones automáticas de optimización

Autor: Sistema SIGeC-Balistica
Fecha: 2024
"""

import time
import threading
import functools
import logging
import psutil
from datetime import datetime
from typing import Dict, Any, Optional, Callable, Union, List
from dataclasses import dataclass, field
from enum import Enum
import traceback

# Importaciones del sistema
try:
    from core.telemetry_system import (
        get_telemetry_system, record_performance_event, 
        record_error_event, record_user_action
    )
    TELEMETRY_AVAILABLE = True
except ImportError:
    TELEMETRY_AVAILABLE = False

try:
    from performance.enhanced_monitoring_system import (
        get_enhanced_monitoring_system, AlertSeverity
    )
    ENHANCED_MONITORING_AVAILABLE = True
except ImportError:
    ENHANCED_MONITORING_AVAILABLE = False
    # Definir AlertSeverity como fallback
    class AlertSeverity(Enum):
        LOW = "low"
        MEDIUM = "medium"
        HIGH = "high"
        CRITICAL = "critical"

try:
    from core.notification_system import notify_warning, notify_error, notify_critical
    NOTIFICATIONS_AVAILABLE = True
except ImportError:
    NOTIFICATIONS_AVAILABLE = False

logger = logging.getLogger(__name__)

class OperationType(Enum):
    """Tipos de operaciones críticas."""
    IMAGE_ANALYSIS = "image_analysis"
    IMAGE_PROCESSING = "image_processing"
    DATABASE_OPERATION = "database_operation"
    IMAGE_COMPARISON = "image_comparison"
    NIST_VALIDATION = "nist_validation"
    PIPELINE_EXECUTION = "pipeline_execution"
    FEATURE_EXTRACTION = "feature_extraction"
    REPORT_GENERATION = "report_generation"
    DATA_VALIDATION = "data_validation"

@dataclass
class PerformanceThreshold:
    """Umbrales de rendimiento para operaciones."""
    warning_duration_ms: float
    critical_duration_ms: float
    warning_memory_mb: float = 500.0
    critical_memory_mb: float = 1000.0
    warning_cpu_percent: float = 80.0
    critical_cpu_percent: float = 95.0
    max_retries: int = 3
    timeout_ms: Optional[float] = None

@dataclass
class PerformanceMetrics:
    """Métricas de rendimiento de una operación."""
    operation_name: str
    operation_type: OperationType
    component: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: float = 0.0
    memory_before_mb: float = 0.0
    memory_after_mb: float = 0.0
    memory_peak_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    success: bool = True
    error_message: Optional[str] = None
    retry_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

class PerformanceMonitor:
    """Monitor de rendimiento automático."""
    
    def __init__(self):
        self.thresholds = self._setup_default_thresholds()
        self.active_operations: Dict[str, PerformanceMetrics] = {}
        self.lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
        
        # Configurar sistemas de monitoreo
        self.telemetry_system = None
        self.enhanced_monitoring = None
        
        if TELEMETRY_AVAILABLE:
            try:
                self.telemetry_system = get_telemetry_system()
            except Exception as e:
                self.logger.warning(f"No se pudo inicializar telemetría: {e}")
        
        if ENHANCED_MONITORING_AVAILABLE:
            try:
                self.enhanced_monitoring = get_enhanced_monitoring_system()
            except Exception as e:
                self.logger.warning(f"No se pudo inicializar monitoreo avanzado: {e}")
    
    def _setup_default_thresholds(self) -> Dict[OperationType, PerformanceThreshold]:
        """Configura umbrales por defecto para cada tipo de operación."""
        return {
            OperationType.IMAGE_ANALYSIS: PerformanceThreshold(
                warning_duration_ms=30000,  # 30 segundos
                critical_duration_ms=60000,  # 60 segundos
                warning_memory_mb=800,
                critical_memory_mb=1500,
                timeout_ms=120000  # 2 minutos
            ),
            OperationType.IMAGE_PROCESSING: PerformanceThreshold(
                warning_duration_ms=10000,  # 10 segundos
                critical_duration_ms=20000,  # 20 segundos
                warning_memory_mb=400,
                critical_memory_mb=800,
                timeout_ms=45000  # 45 segundos
            ),
            OperationType.DATABASE_OPERATION: PerformanceThreshold(
                warning_duration_ms=5000,   # 5 segundos
                critical_duration_ms=15000, # 15 segundos
                warning_memory_mb=200,
                critical_memory_mb=500,
                timeout_ms=30000  # 30 segundos
            ),
            OperationType.IMAGE_COMPARISON: PerformanceThreshold(
                warning_duration_ms=20000,  # 20 segundos
                critical_duration_ms=45000, # 45 segundos
                warning_memory_mb=600,
                critical_memory_mb=1200,
                timeout_ms=90000  # 90 segundos
            ),
            OperationType.NIST_VALIDATION: PerformanceThreshold(
                warning_duration_ms=8000,   # 8 segundos
                critical_duration_ms=15000, # 15 segundos
                warning_memory_mb=300,
                critical_memory_mb=600,
                timeout_ms=30000  # 30 segundos
            ),
            OperationType.PIPELINE_EXECUTION: PerformanceThreshold(
                warning_duration_ms=45000,  # 45 segundos
                critical_duration_ms=90000, # 90 segundos
                warning_memory_mb=1000,
                critical_memory_mb=2000,
                timeout_ms=180000  # 3 minutos
            ),
            OperationType.FEATURE_EXTRACTION: PerformanceThreshold(
                warning_duration_ms=15000,  # 15 segundos
                critical_duration_ms=30000, # 30 segundos
                warning_memory_mb=500,
                critical_memory_mb=1000,
                timeout_ms=60000  # 60 segundos
            ),
            OperationType.REPORT_GENERATION: PerformanceThreshold(
                warning_duration_ms=5000,   # 5 segundos
                critical_duration_ms=12000, # 12 segundos
                warning_memory_mb=200,
                critical_memory_mb=400,
                timeout_ms=20000  # 20 segundos
            )
        }
    
    def start_operation_monitoring(self, operation_name: str, operation_type: OperationType, 
                                 component: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Inicia el monitoreo de una operación."""
        operation_id = f"{component}_{operation_name}_{int(time.time() * 1000)}"
        
        # Obtener métricas iniciales
        process = psutil.Process()
        memory_info = process.memory_info()
        
        metrics = PerformanceMetrics(
            operation_name=operation_name,
            operation_type=operation_type,
            component=component,
            start_time=datetime.now(),
            memory_before_mb=memory_info.rss / (1024 * 1024),
            metadata=metadata or {}
        )
        
        with self.lock:
            self.active_operations[operation_id] = metrics
        
        # Registrar inicio en telemetría
        if self.telemetry_system:
            try:
                record_user_action(
                    f"{operation_name}_started",
                    component,
                    data={
                        'operation_type': operation_type.value,
                        'operation_id': operation_id,
                        'memory_before_mb': metrics.memory_before_mb,
                        **metadata
                    } if metadata else {
                        'operation_type': operation_type.value,
                        'operation_id': operation_id,
                        'memory_before_mb': metrics.memory_before_mb
                    }
                )
            except Exception as e:
                self.logger.warning(f"Error registrando inicio de operación en telemetría: {e}")
        
        return operation_id
    
    def end_operation_monitoring(self, operation_id: str, success: bool = True, 
                               error: Optional[Exception] = None) -> PerformanceMetrics:
        """Finaliza el monitoreo de una operación."""
        with self.lock:
            if operation_id not in self.active_operations:
                self.logger.warning(f"Operación {operation_id} no encontrada")
                return None
            
            metrics = self.active_operations[operation_id]
            del self.active_operations[operation_id]
        
        # Calcular métricas finales
        metrics.end_time = datetime.now()
        metrics.duration_ms = (metrics.end_time - metrics.start_time).total_seconds() * 1000
        metrics.success = success
        
        if error:
            metrics.error_message = str(error)
        
        # Obtener métricas finales del sistema
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            metrics.memory_after_mb = memory_info.rss / (1024 * 1024)
            metrics.cpu_usage_percent = process.cpu_percent()
        except Exception as e:
            self.logger.warning(f"Error obteniendo métricas del sistema: {e}")
        
        # Analizar rendimiento y generar alertas
        self._analyze_performance(metrics)
        
        # Registrar en telemetría
        self._record_telemetry(metrics)
        
        return metrics
    
    def _analyze_performance(self, metrics: PerformanceMetrics):
        """Analiza el rendimiento y genera alertas si es necesario."""
        threshold = self.thresholds.get(metrics.operation_type)
        if not threshold:
            return
        
        alerts = []
        severity = None
        
        # Verificar duración
        if metrics.duration_ms >= threshold.critical_duration_ms:
            severity = AlertSeverity.CRITICAL
            alerts.append(f"Operación {metrics.operation_name} excedió tiempo crítico: {metrics.duration_ms:.0f}ms")
        elif metrics.duration_ms >= threshold.warning_duration_ms:
            severity = AlertSeverity.HIGH if not severity else severity
            alerts.append(f"Operación {metrics.operation_name} excedió tiempo de advertencia: {metrics.duration_ms:.0f}ms")
        
        # Verificar memoria
        memory_used = metrics.memory_after_mb - metrics.memory_before_mb
        if memory_used >= threshold.critical_memory_mb:
            severity = AlertSeverity.CRITICAL
            alerts.append(f"Operación {metrics.operation_name} usó memoria crítica: {memory_used:.1f}MB")
        elif memory_used >= threshold.warning_memory_mb:
            severity = AlertSeverity.HIGH if not severity else severity
            alerts.append(f"Operación {metrics.operation_name} usó memoria alta: {memory_used:.1f}MB")
        
        # Verificar CPU
        if metrics.cpu_usage_percent >= threshold.critical_cpu_percent:
            severity = AlertSeverity.CRITICAL
            alerts.append(f"Operación {metrics.operation_name} usó CPU crítico: {metrics.cpu_usage_percent:.1f}%")
        elif metrics.cpu_usage_percent >= threshold.warning_cpu_percent:
            severity = AlertSeverity.HIGH if not severity else severity
            alerts.append(f"Operación {metrics.operation_name} usó CPU alto: {metrics.cpu_usage_percent:.1f}%")
        
        # Enviar alertas si es necesario
        if alerts and severity:
            self._send_performance_alert(metrics, severity, alerts)
    
    def _send_performance_alert(self, metrics: PerformanceMetrics, severity: AlertSeverity, alerts: List[str]):
        """Envía alertas de rendimiento."""
        alert_message = f"Alerta de rendimiento en {metrics.component}:\n" + "\n".join(alerts)
        
        # Generar recomendaciones
        recommendations = self._generate_recommendations(metrics)
        if recommendations:
            alert_message += f"\n\nRecomendaciones:\n" + "\n".join(recommendations)
        
        # Enviar notificación
        if NOTIFICATIONS_AVAILABLE:
            try:
                if severity == AlertSeverity.CRITICAL:
                    notify_critical("Rendimiento Crítico", alert_message)
                elif severity == AlertSeverity.HIGH:
                    notify_error("Rendimiento Alto", alert_message)
                else:
                    notify_warning("Rendimiento", alert_message)
            except Exception as e:
                self.logger.error(f"Error enviando notificación: {e}")
        
        # Log de la alerta
        self.logger.warning(f"Alerta de rendimiento: {alert_message}")
    
    def _generate_recommendations(self, metrics: PerformanceMetrics) -> List[str]:
        """Genera recomendaciones de optimización basadas en las métricas."""
        recommendations = []
        threshold = self.thresholds.get(metrics.operation_type)
        
        if not threshold:
            return recommendations
        
        # Recomendaciones por duración
        if metrics.duration_ms >= threshold.warning_duration_ms:
            if metrics.operation_type == OperationType.IMAGE_PROCESSING:
                recommendations.extend([
                    "Considere reducir la resolución de las imágenes",
                    "Verifique si hay algoritmos más eficientes disponibles",
                    "Considere procesamiento en paralelo"
                ])
            elif metrics.operation_type == OperationType.DATABASE_OPERATION:
                recommendations.extend([
                    "Revise los índices de la base de datos",
                    "Considere optimizar las consultas SQL",
                    "Verifique la conexión a la base de datos"
                ])
            elif metrics.operation_type == OperationType.IMAGE_ANALYSIS:
                recommendations.extend([
                    "Considere usar un nivel de análisis más básico",
                    "Verifique la calidad de las imágenes de entrada",
                    "Considere procesamiento por lotes"
                ])
        
        # Recomendaciones por memoria
        memory_used = metrics.memory_after_mb - metrics.memory_before_mb
        if memory_used >= threshold.warning_memory_mb:
            recommendations.extend([
                "Libere memoria no utilizada",
                "Considere procesamiento por chunks",
                "Verifique posibles memory leaks"
            ])
        
        # Recomendaciones por CPU
        if metrics.cpu_usage_percent >= threshold.warning_cpu_percent:
            recommendations.extend([
                "Considere reducir la carga de trabajo simultánea",
                "Verifique procesos en segundo plano",
                "Considere optimizaciones de algoritmos"
            ])
        
        return recommendations
    
    def _record_telemetry(self, metrics: PerformanceMetrics):
        """Registra las métricas en el sistema de telemetría."""
        if not self.telemetry_system:
            return
        
        try:
            # Registrar evento de rendimiento
            record_performance_event(
                metrics.operation_name,
                metrics.component,
                metrics.duration_ms,
                success=metrics.success,
                metadata={
                    'operation_type': metrics.operation_type.value,
                    'memory_used_mb': metrics.memory_after_mb - metrics.memory_before_mb,
                    'cpu_usage_percent': metrics.cpu_usage_percent,
                    'retry_count': metrics.retry_count,
                    **metrics.metadata
                }
            )
            
            # Registrar error si hubo uno
            if not metrics.success and metrics.error_message:
                record_error_event(
                    Exception(metrics.error_message),
                    metrics.component,
                    metrics.operation_name
                )
        
        except Exception as e:
            self.logger.warning(f"Error registrando telemetría: {e}")
    
    def get_operation_statistics(self) -> Dict[str, Any]:
        """Obtiene estadísticas de operaciones activas."""
        with self.lock:
            active_count = len(self.active_operations)
            operations_by_type = {}
            
            for metrics in self.active_operations.values():
                op_type = metrics.operation_type.value
                if op_type not in operations_by_type:
                    operations_by_type[op_type] = 0
                operations_by_type[op_type] += 1
        
        return {
            'active_operations': active_count,
            'operations_by_type': operations_by_type,
            'configured_thresholds': {
                op_type.value: {
                    'warning_duration_ms': threshold.warning_duration_ms,
                    'critical_duration_ms': threshold.critical_duration_ms
                }
                for op_type, threshold in self.thresholds.items()
            }
        }

# Instancia global del monitor
_performance_monitor = None

def get_performance_monitor() -> PerformanceMonitor:
    """Obtiene la instancia global del monitor de rendimiento."""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
    return _performance_monitor

# Decoradores para monitoreo automático
def monitor_performance(operation_type: OperationType, operation_name: str = None):
    """
    Decorador para monitoreo automático de rendimiento.
    
    Args:
        operation_type: Tipo de operación a monitorear
        operation_name: Nombre específico de la operación (opcional)
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            monitor = get_performance_monitor()
            op_name = operation_name or func.__name__
            component = func.__module__.split('.')[-1] if hasattr(func, '__module__') else 'unknown'
            
            # Extraer metadatos de los argumentos si es posible
            metadata = {}
            if args:
                if hasattr(args[0], '__class__'):
                    metadata['class'] = args[0].__class__.__name__
            
            operation_id = monitor.start_operation_monitoring(
                op_name, operation_type, component, metadata
            )
            
            try:
                result = func(*args, **kwargs)
                monitor.end_operation_monitoring(operation_id, success=True)
                return result
            
            except Exception as e:
                monitor.end_operation_monitoring(operation_id, success=False, error=e)
                raise
        
        return wrapper
    return decorator

# Decoradores específicos para operaciones comunes
def monitor_image_analysis(operation_name: str = None):
    """Decorador para monitoreo de análisis de imágenes."""
    return monitor_performance(OperationType.IMAGE_ANALYSIS, operation_name)

def monitor_image_processing(operation_name: str = None):
    """Decorador para monitoreo de procesamiento de imágenes."""
    return monitor_performance(OperationType.IMAGE_PROCESSING, operation_name)

def monitor_database_operation(operation_name: str = None):
    """Decorador para monitoreo de operaciones de base de datos."""
    return monitor_performance(OperationType.DATABASE_OPERATION, operation_name)

def monitor_image_comparison(operation_name: str = None):
    """Decorador para monitoreo de comparación de imágenes."""
    return monitor_performance(OperationType.IMAGE_COMPARISON, operation_name)

def monitor_nist_validation(operation_name: str = None):
    """Decorador para monitoreo de validación NIST."""
    return monitor_performance(OperationType.NIST_VALIDATION, operation_name)

def monitor_pipeline_execution(operation_name: str = None):
    """Decorador para monitoreo de ejecución de pipeline."""
    return monitor_performance(OperationType.PIPELINE_EXECUTION, operation_name)

# Función de contexto para monitoreo manual
class PerformanceContext:
    """Contexto para monitoreo manual de rendimiento."""
    
    def __init__(self, operation_type: OperationType, operation_name: str, 
                 component: str, metadata: Optional[Dict[str, Any]] = None):
        self.monitor = get_performance_monitor()
        self.operation_type = operation_type
        self.operation_name = operation_name
        self.component = component
        self.metadata = metadata
        self.operation_id = None
    
    def __enter__(self):
        self.operation_id = self.monitor.start_operation_monitoring(
            self.operation_name, self.operation_type, self.component, self.metadata
        )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        success = exc_type is None
        error = exc_val if exc_type else None
        self.monitor.end_operation_monitoring(self.operation_id, success, error)