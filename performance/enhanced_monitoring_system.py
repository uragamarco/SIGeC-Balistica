#!/usr/bin/env python3
"""
Sistema de Monitoreo Expandido para SIGeC-Balistica
Sistema avanzado de monitoreo con métricas expandidas, alertas inteligentes y dashboard en tiempo real

Este módulo implementa:
- Métricas expandidas del sistema (CPU, memoria, disco, red, GPU, procesos)
- Alertas inteligentes con machine learning para detección de anomalías
- Dashboard en tiempo real con visualizaciones interactivas
- Sistema de predicción de problemas
- Análisis de tendencias y patrones
- Integración completa con sistema de notificaciones

Autor: Sistema SIGeC-Balistica
Fecha: 2024
"""

import asyncio
import logging
import threading
import time
import queue
import json
import pickle
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field, asdict
from collections import deque, defaultdict
from pathlib import Path
import statistics
import numpy as np
from enum import Enum

# Importaciones del sistema
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

# Importaciones internas
try:
    from core.notification_system import (
        get_notification_manager, NotificationType, NotificationChannel,
        notify_info, notify_warning, notify_error, notify_critical
    )
    NOTIFICATIONS_AVAILABLE = True
except ImportError:
    NOTIFICATIONS_AVAILABLE = False

try:
    from performance.metrics_system import PerformanceMetricsSystem, MetricCollector, MetricThreshold
    from performance.gpu_monitor import GPUMonitor
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False

logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    """Niveles de severidad de alertas."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class MetricCategory(Enum):
    """Categorías de métricas."""
    SYSTEM = "system"
    PERFORMANCE = "performance"
    RESOURCE = "resource"
    APPLICATION = "application"
    SECURITY = "security"
    CUSTOM = "custom"

@dataclass
class AdvancedMetric:
    """Métrica avanzada con metadatos expandidos."""
    name: str
    value: Union[float, int, str, Dict, List]
    timestamp: datetime
    category: MetricCategory
    unit: str = ""
    description: str = ""
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte la métrica a diccionario."""
        return {
            'name': self.name,
            'value': self.value,
            'timestamp': self.timestamp.isoformat(),
            'category': self.category.value,
            'unit': self.unit,
            'description': self.description,
            'tags': self.tags,
            'metadata': self.metadata
        }

@dataclass
class SmartAlert:
    """Alerta inteligente con contexto expandido."""
    id: str
    timestamp: datetime
    severity: AlertSeverity
    title: str
    message: str
    component: str
    metric_name: str
    current_value: Any
    threshold_value: Any
    trend: str = ""  # "increasing", "decreasing", "stable", "volatile"
    prediction: Optional[str] = None
    recommendations: List[str] = field(default_factory=list)
    related_metrics: List[str] = field(default_factory=list)
    auto_resolve: bool = False
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte la alerta a diccionario."""
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'severity': self.severity.value,
            'title': self.title,
            'message': self.message,
            'component': self.component,
            'metric_name': self.metric_name,
            'current_value': self.current_value,
            'threshold_value': self.threshold_value,
            'trend': self.trend,
            'prediction': self.prediction,
            'recommendations': self.recommendations,
            'related_metrics': self.related_metrics,
            'auto_resolve': self.auto_resolve,
            'resolved': self.resolved,
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None
        }

@dataclass
class MonitoringConfig:
    """Configuración del sistema de monitoreo expandido."""
    collection_interval: float = 1.0
    history_retention_hours: int = 24
    max_metrics_in_memory: int = 10000
    enable_ml_anomaly_detection: bool = True
    enable_trend_analysis: bool = True
    enable_predictive_alerts: bool = True
    dashboard_update_interval: float = 5.0
    alert_cooldown_minutes: int = 5
    auto_cleanup_interval_hours: int = 6
    export_metrics_interval_hours: int = 1
    
    # Configuración de alertas
    cpu_warning_threshold: float = 80.0
    cpu_critical_threshold: float = 95.0
    memory_warning_threshold: float = 85.0
    memory_critical_threshold: float = 95.0
    disk_warning_threshold: float = 90.0
    disk_critical_threshold: float = 98.0
    gpu_memory_warning_threshold: float = 90.0
    gpu_memory_critical_threshold: float = 98.0
    
    # Configuración ML
    anomaly_detection_window: int = 100
    anomaly_contamination: float = 0.1
    trend_analysis_window: int = 50

class AdvancedMetricCollector:
    """Recopilador avanzado de métricas del sistema."""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.AdvancedMetricCollector")
        
    def collect_system_metrics(self) -> List[AdvancedMetric]:
        """Recopila métricas avanzadas del sistema."""
        metrics = []
        timestamp = datetime.now()
        
        if not PSUTIL_AVAILABLE:
            return metrics
        
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_count = psutil.cpu_count()
            cpu_freq = psutil.cpu_freq()
            load_avg = psutil.getloadavg() if hasattr(psutil, 'getloadavg') else (0, 0, 0)
            
            metrics.extend([
                AdvancedMetric(
                    name="cpu_usage_percent",
                    value=cpu_percent,
                    timestamp=timestamp,
                    category=MetricCategory.SYSTEM,
                    unit="%",
                    description="Porcentaje de uso de CPU",
                    tags={"type": "cpu", "resource": "processor"}
                ),
                AdvancedMetric(
                    name="cpu_count",
                    value=cpu_count,
                    timestamp=timestamp,
                    category=MetricCategory.SYSTEM,
                    unit="cores",
                    description="Número de núcleos de CPU",
                    tags={"type": "cpu", "resource": "processor"}
                ),
                AdvancedMetric(
                    name="cpu_frequency_mhz",
                    value=cpu_freq.current if cpu_freq else 0,
                    timestamp=timestamp,
                    category=MetricCategory.SYSTEM,
                    unit="MHz",
                    description="Frecuencia actual de CPU",
                    tags={"type": "cpu", "resource": "processor"}
                ),
                AdvancedMetric(
                    name="load_average_1m",
                    value=load_avg[0],
                    timestamp=timestamp,
                    category=MetricCategory.SYSTEM,
                    unit="",
                    description="Promedio de carga del sistema (1 minuto)",
                    tags={"type": "load", "resource": "system"}
                )
            ])
            
            # Memory metrics
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            metrics.extend([
                AdvancedMetric(
                    name="memory_usage_percent",
                    value=memory.percent,
                    timestamp=timestamp,
                    category=MetricCategory.RESOURCE,
                    unit="%",
                    description="Porcentaje de uso de memoria",
                    tags={"type": "memory", "resource": "ram"}
                ),
                AdvancedMetric(
                    name="memory_available_gb",
                    value=memory.available / (1024**3),
                    timestamp=timestamp,
                    category=MetricCategory.RESOURCE,
                    unit="GB",
                    description="Memoria disponible",
                    tags={"type": "memory", "resource": "ram"}
                ),
                AdvancedMetric(
                    name="memory_used_gb",
                    value=memory.used / (1024**3),
                    timestamp=timestamp,
                    category=MetricCategory.RESOURCE,
                    unit="GB",
                    description="Memoria utilizada",
                    tags={"type": "memory", "resource": "ram"}
                ),
                AdvancedMetric(
                    name="swap_usage_percent",
                    value=swap.percent,
                    timestamp=timestamp,
                    category=MetricCategory.RESOURCE,
                    unit="%",
                    description="Porcentaje de uso de swap",
                    tags={"type": "memory", "resource": "swap"}
                )
            ])
            
            # Disk metrics
            disk_usage = psutil.disk_usage('/')
            disk_io = psutil.disk_io_counters()
            
            metrics.extend([
                AdvancedMetric(
                    name="disk_usage_percent",
                    value=disk_usage.percent,
                    timestamp=timestamp,
                    category=MetricCategory.RESOURCE,
                    unit="%",
                    description="Porcentaje de uso de disco",
                    tags={"type": "disk", "resource": "storage"}
                ),
                AdvancedMetric(
                    name="disk_free_gb",
                    value=disk_usage.free / (1024**3),
                    timestamp=timestamp,
                    category=MetricCategory.RESOURCE,
                    unit="GB",
                    description="Espacio libre en disco",
                    tags={"type": "disk", "resource": "storage"}
                )
            ])
            
            if disk_io:
                metrics.extend([
                    AdvancedMetric(
                        name="disk_read_mb_per_sec",
                        value=disk_io.read_bytes / (1024**2),
                        timestamp=timestamp,
                        category=MetricCategory.PERFORMANCE,
                        unit="MB/s",
                        description="Velocidad de lectura de disco",
                        tags={"type": "disk", "resource": "io"}
                    ),
                    AdvancedMetric(
                        name="disk_write_mb_per_sec",
                        value=disk_io.write_bytes / (1024**2),
                        timestamp=timestamp,
                        category=MetricCategory.PERFORMANCE,
                        unit="MB/s",
                        description="Velocidad de escritura de disco",
                        tags={"type": "disk", "resource": "io"}
                    )
                ])
            
            # Network metrics
            network_io = psutil.net_io_counters()
            if network_io:
                metrics.extend([
                    AdvancedMetric(
                        name="network_sent_mb_per_sec",
                        value=network_io.bytes_sent / (1024**2),
                        timestamp=timestamp,
                        category=MetricCategory.PERFORMANCE,
                        unit="MB/s",
                        description="Datos enviados por red",
                        tags={"type": "network", "resource": "io"}
                    ),
                    AdvancedMetric(
                        name="network_recv_mb_per_sec",
                        value=network_io.bytes_recv / (1024**2),
                        timestamp=timestamp,
                        category=MetricCategory.PERFORMANCE,
                        unit="MB/s",
                        description="Datos recibidos por red",
                        tags={"type": "network", "resource": "io"}
                    )
                ])
            
            # Process metrics
            process_count = len(psutil.pids())
            current_process = psutil.Process()
            
            metrics.extend([
                AdvancedMetric(
                    name="process_count",
                    value=process_count,
                    timestamp=timestamp,
                    category=MetricCategory.SYSTEM,
                    unit="processes",
                    description="Número total de procesos",
                    tags={"type": "process", "resource": "system"}
                ),
                AdvancedMetric(
                    name="app_memory_usage_mb",
                    value=current_process.memory_info().rss / (1024**2),
                    timestamp=timestamp,
                    category=MetricCategory.APPLICATION,
                    unit="MB",
                    description="Uso de memoria de la aplicación",
                    tags={"type": "memory", "resource": "application"}
                ),
                AdvancedMetric(
                    name="app_cpu_usage_percent",
                    value=current_process.cpu_percent(),
                    timestamp=timestamp,
                    category=MetricCategory.APPLICATION,
                    unit="%",
                    description="Uso de CPU de la aplicación",
                    tags={"type": "cpu", "resource": "application"}
                )
            ])
            
        except Exception as e:
            self.logger.error(f"Error recopilando métricas del sistema: {e}")
        
        return metrics
    
    def collect_gpu_metrics(self) -> List[AdvancedMetric]:
        """Recopila métricas de GPU."""
        metrics = []
        timestamp = datetime.now()
        
        if not CUPY_AVAILABLE:
            return metrics
        
        try:
            # Obtener información de GPU usando CuPy
            mempool = cp.get_default_memory_pool()
            used_bytes = mempool.used_bytes()
            total_bytes = mempool.total_bytes()
            
            if total_bytes > 0:
                usage_percent = (used_bytes / total_bytes) * 100
            else:
                usage_percent = 0
            
            metrics.extend([
                AdvancedMetric(
                    name="gpu_memory_usage_percent",
                    value=usage_percent,
                    timestamp=timestamp,
                    category=MetricCategory.RESOURCE,
                    unit="%",
                    description="Porcentaje de uso de memoria GPU",
                    tags={"type": "gpu", "resource": "memory"}
                ),
                AdvancedMetric(
                    name="gpu_memory_used_gb",
                    value=used_bytes / (1024**3),
                    timestamp=timestamp,
                    category=MetricCategory.RESOURCE,
                    unit="GB",
                    description="Memoria GPU utilizada",
                    tags={"type": "gpu", "resource": "memory"}
                ),
                AdvancedMetric(
                    name="gpu_memory_total_gb",
                    value=total_bytes / (1024**3),
                    timestamp=timestamp,
                    category=MetricCategory.RESOURCE,
                    unit="GB",
                    description="Memoria GPU total",
                    tags={"type": "gpu", "resource": "memory"}
                )
            ])
            
        except Exception as e:
            self.logger.error(f"Error recopilando métricas de GPU: {e}")
        
        return metrics

class AnomalyDetector:
    """Detector de anomalías usando machine learning."""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.models: Dict[str, IsolationForest] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.training_data: Dict[str, deque] = defaultdict(lambda: deque(maxlen=config.anomaly_detection_window))
        self.logger = logging.getLogger(f"{__name__}.AnomalyDetector")
        
    def add_metric_value(self, metric_name: str, value: float):
        """Añade un valor de métrica para análisis de anomalías."""
        if not ML_AVAILABLE:
            return
        
        self.training_data[metric_name].append(value)
        
        # Entrenar modelo si tenemos suficientes datos
        if len(self.training_data[metric_name]) >= self.config.anomaly_detection_window:
            self._train_model(metric_name)
    
    def _train_model(self, metric_name: str):
        """Entrena el modelo de detección de anomalías para una métrica."""
        try:
            data = np.array(list(self.training_data[metric_name])).reshape(-1, 1)
            
            # Escalar datos
            if metric_name not in self.scalers:
                self.scalers[metric_name] = StandardScaler()
            
            scaled_data = self.scalers[metric_name].fit_transform(data)
            
            # Entrenar modelo
            if metric_name not in self.models:
                self.models[metric_name] = IsolationForest(
                    contamination=self.config.anomaly_contamination,
                    random_state=42
                )
            
            self.models[metric_name].fit(scaled_data)
            
        except Exception as e:
            self.logger.error(f"Error entrenando modelo para {metric_name}: {e}")
    
    def detect_anomaly(self, metric_name: str, value: float) -> Tuple[bool, float]:
        """Detecta si un valor es anómalo."""
        if not ML_AVAILABLE or metric_name not in self.models:
            return False, 0.0
        
        try:
            scaled_value = self.scalers[metric_name].transform([[value]])
            anomaly_score = self.models[metric_name].decision_function(scaled_value)[0]
            is_anomaly = self.models[metric_name].predict(scaled_value)[0] == -1
            
            return is_anomaly, anomaly_score
            
        except Exception as e:
            self.logger.error(f"Error detectando anomalía para {metric_name}: {e}")
            return False, 0.0

class TrendAnalyzer:
    """Analizador de tendencias de métricas."""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.metric_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=config.trend_analysis_window))
        self.logger = logging.getLogger(f"{__name__}.TrendAnalyzer")
    
    def add_metric_value(self, metric_name: str, value: float, timestamp: datetime):
        """Añade un valor de métrica para análisis de tendencias."""
        self.metric_history[metric_name].append((timestamp, value))
    
    def analyze_trend(self, metric_name: str) -> Dict[str, Any]:
        """Analiza la tendencia de una métrica."""
        if metric_name not in self.metric_history or len(self.metric_history[metric_name]) < 10:
            return {"trend": "insufficient_data", "confidence": 0.0}
        
        try:
            values = [v for _, v in self.metric_history[metric_name]]
            
            # Calcular tendencia usando regresión lineal simple
            n = len(values)
            x = np.arange(n)
            y = np.array(values)
            
            # Coeficientes de regresión lineal
            slope = np.polyfit(x, y, 1)[0]
            
            # Calcular volatilidad
            volatility = np.std(values) / np.mean(values) if np.mean(values) != 0 else 0
            
            # Determinar tendencia
            if abs(slope) < 0.01:
                trend = "stable"
            elif slope > 0:
                trend = "increasing"
            else:
                trend = "decreasing"
            
            # Determinar si es volátil
            if volatility > 0.2:
                trend = "volatile"
            
            # Calcular confianza basada en R²
            correlation = np.corrcoef(x, y)[0, 1] if len(x) > 1 else 0
            confidence = abs(correlation) ** 2
            
            return {
                "trend": trend,
                "slope": slope,
                "volatility": volatility,
                "confidence": confidence,
                "recent_avg": np.mean(values[-5:]),
                "overall_avg": np.mean(values)
            }
            
        except Exception as e:
            self.logger.error(f"Error analizando tendencia para {metric_name}: {e}")
            return {"trend": "error", "confidence": 0.0}

class SmartAlertManager:
    """Gestor inteligente de alertas."""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.active_alerts: Dict[str, SmartAlert] = {}
        self.alert_history: deque = deque(maxlen=1000)
        self.alert_cooldowns: Dict[str, datetime] = {}
        self.logger = logging.getLogger(f"{__name__}.SmartAlertManager")
    
    def check_metric_alert(self, metric: AdvancedMetric, trend_info: Dict[str, Any], 
                          anomaly_info: Tuple[bool, float]) -> Optional[SmartAlert]:
        """Verifica si una métrica debe generar una alerta."""
        
        alert_key = f"{metric.name}_{metric.category.value}"
        
        # Verificar cooldown
        if alert_key in self.alert_cooldowns:
            cooldown_end = self.alert_cooldowns[alert_key] + timedelta(minutes=self.config.alert_cooldown_minutes)
            if datetime.now() < cooldown_end:
                return None
        
        # Determinar si debe alertar
        should_alert, severity, threshold = self._should_alert(metric, trend_info, anomaly_info)
        
        if not should_alert:
            # Resolver alerta existente si aplica
            if alert_key in self.active_alerts:
                self._resolve_alert(alert_key)
            return None
        
        # Crear nueva alerta o actualizar existente
        alert = self._create_or_update_alert(metric, severity, threshold, trend_info, anomaly_info)
        
        # Establecer cooldown
        self.alert_cooldowns[alert_key] = datetime.now()
        
        return alert
    
    def _should_alert(self, metric: AdvancedMetric, trend_info: Dict[str, Any], 
                     anomaly_info: Tuple[bool, float]) -> Tuple[bool, AlertSeverity, float]:
        """Determina si debe generar una alerta."""
        
        is_anomaly, anomaly_score = anomaly_info
        
        # Alertas basadas en umbrales específicos
        if metric.name == "cpu_usage_percent":
            if metric.value >= self.config.cpu_critical_threshold:
                return True, AlertSeverity.CRITICAL, self.config.cpu_critical_threshold
            elif metric.value >= self.config.cpu_warning_threshold:
                return True, AlertSeverity.HIGH, self.config.cpu_warning_threshold
        
        elif metric.name == "memory_usage_percent":
            if metric.value >= self.config.memory_critical_threshold:
                return True, AlertSeverity.CRITICAL, self.config.memory_critical_threshold
            elif metric.value >= self.config.memory_warning_threshold:
                return True, AlertSeverity.HIGH, self.config.memory_warning_threshold
        
        elif metric.name == "disk_usage_percent":
            if metric.value >= self.config.disk_critical_threshold:
                return True, AlertSeverity.CRITICAL, self.config.disk_critical_threshold
            elif metric.value >= self.config.disk_warning_threshold:
                return True, AlertSeverity.HIGH, self.config.disk_warning_threshold
        
        elif metric.name == "gpu_memory_usage_percent":
            if metric.value >= self.config.gpu_memory_critical_threshold:
                return True, AlertSeverity.CRITICAL, self.config.gpu_memory_critical_threshold
            elif metric.value >= self.config.gpu_memory_warning_threshold:
                return True, AlertSeverity.HIGH, self.config.gpu_memory_warning_threshold
        
        # Alertas basadas en anomalías
        if is_anomaly and anomaly_score < -0.5:
            return True, AlertSeverity.MEDIUM, anomaly_score
        
        # Alertas basadas en tendencias
        if trend_info.get("trend") == "volatile" and trend_info.get("confidence", 0) > 0.7:
            return True, AlertSeverity.LOW, trend_info.get("volatility", 0)
        
        return False, AlertSeverity.LOW, 0
    
    def _create_or_update_alert(self, metric: AdvancedMetric, severity: AlertSeverity, 
                               threshold: float, trend_info: Dict[str, Any], 
                               anomaly_info: Tuple[bool, float]) -> SmartAlert:
        """Crea o actualiza una alerta."""
        
        alert_id = f"{metric.name}_{int(time.time())}"
        
        # Generar recomendaciones
        recommendations = self._generate_recommendations(metric, severity, trend_info)
        
        # Generar predicción
        prediction = self._generate_prediction(metric, trend_info)
        
        alert = SmartAlert(
            id=alert_id,
            timestamp=datetime.now(),
            severity=severity,
            title=f"Alerta de {metric.name}",
            message=f"{metric.description}: {metric.value} {metric.unit}",
            component=metric.tags.get("resource", "system"),
            metric_name=metric.name,
            current_value=metric.value,
            threshold_value=threshold,
            trend=trend_info.get("trend", "unknown"),
            prediction=prediction,
            recommendations=recommendations,
            related_metrics=self._find_related_metrics(metric),
            auto_resolve=severity in [AlertSeverity.LOW, AlertSeverity.MEDIUM]
        )
        
        # Almacenar alerta
        alert_key = f"{metric.name}_{metric.category.value}"
        self.active_alerts[alert_key] = alert
        self.alert_history.append(alert)
        
        # Enviar notificación
        self._send_alert_notification(alert)
        
        return alert
    
    def _generate_recommendations(self, metric: AdvancedMetric, severity: AlertSeverity, 
                                 trend_info: Dict[str, Any]) -> List[str]:
        """Genera recomendaciones para resolver la alerta."""
        
        recommendations = []
        
        if metric.name == "cpu_usage_percent":
            recommendations.extend([
                "Verificar procesos con alto uso de CPU",
                "Considerar optimizar algoritmos de procesamiento",
                "Evaluar la necesidad de más recursos de CPU"
            ])
        
        elif metric.name == "memory_usage_percent":
            recommendations.extend([
                "Revisar uso de memoria de la aplicación",
                "Implementar liberación de memoria no utilizada",
                "Considerar aumentar la memoria RAM disponible"
            ])
        
        elif metric.name == "disk_usage_percent":
            recommendations.extend([
                "Limpiar archivos temporales y logs antiguos",
                "Implementar rotación automática de logs",
                "Considerar expandir el almacenamiento"
            ])
        
        elif metric.name == "gpu_memory_usage_percent":
            recommendations.extend([
                "Optimizar el uso de memoria GPU",
                "Implementar liberación de memoria GPU",
                "Reducir el tamaño de batch en procesamiento"
            ])
        
        # Recomendaciones basadas en tendencias
        if trend_info.get("trend") == "increasing":
            recommendations.append("Monitorear de cerca - tendencia creciente detectada")
        elif trend_info.get("trend") == "volatile":
            recommendations.append("Investigar causa de la volatilidad en la métrica")
        
        return recommendations
    
    def _generate_prediction(self, metric: AdvancedMetric, trend_info: Dict[str, Any]) -> Optional[str]:
        """Genera una predicción basada en la tendencia."""
        
        trend = trend_info.get("trend", "unknown")
        confidence = trend_info.get("confidence", 0)
        
        if confidence < 0.5:
            return None
        
        if trend == "increasing":
            return f"Se predice un aumento continuo en {metric.name}"
        elif trend == "decreasing":
            return f"Se predice una disminución continua en {metric.name}"
        elif trend == "volatile":
            return f"Se predice comportamiento inestable en {metric.name}"
        
        return None
    
    def _find_related_metrics(self, metric: AdvancedMetric) -> List[str]:
        """Encuentra métricas relacionadas."""
        
        related = []
        
        if metric.name.startswith("cpu"):
            related.extend(["memory_usage_percent", "load_average_1m", "process_count"])
        elif metric.name.startswith("memory"):
            related.extend(["cpu_usage_percent", "swap_usage_percent", "app_memory_usage_mb"])
        elif metric.name.startswith("disk"):
            related.extend(["memory_usage_percent", "disk_read_mb_per_sec", "disk_write_mb_per_sec"])
        elif metric.name.startswith("gpu"):
            related.extend(["cpu_usage_percent", "memory_usage_percent"])
        
        return related
    
    def _send_alert_notification(self, alert: SmartAlert):
        """Envía notificación de la alerta."""
        
        if not NOTIFICATIONS_AVAILABLE:
            return
        
        # Mapear severidad a tipo de notificación
        notification_type_map = {
            AlertSeverity.LOW: NotificationType.INFO,
            AlertSeverity.MEDIUM: NotificationType.WARNING,
            AlertSeverity.HIGH: NotificationType.ERROR,
            AlertSeverity.CRITICAL: NotificationType.CRITICAL
        }
        
        notification_type = notification_type_map.get(alert.severity, NotificationType.WARNING)
        
        # Crear mensaje detallado
        message = f"{alert.message}\n"
        message += f"Tendencia: {alert.trend}\n"
        if alert.prediction:
            message += f"Predicción: {alert.prediction}\n"
        if alert.recommendations:
            message += f"Recomendaciones: {', '.join(alert.recommendations[:2])}"
        
        # Enviar notificación
        if notification_type == NotificationType.CRITICAL:
            notify_critical(alert.title, message, alert.component, metadata=alert.to_dict())
        elif notification_type == NotificationType.ERROR:
            notify_error(alert.title, message, alert.component, metadata=alert.to_dict())
        elif notification_type == NotificationType.WARNING:
            notify_warning(alert.title, message, alert.component, metadata=alert.to_dict())
        else:
            notify_info(alert.title, message, alert.component, metadata=alert.to_dict())
    
    def _resolve_alert(self, alert_key: str):
        """Resuelve una alerta activa."""
        
        if alert_key in self.active_alerts:
            alert = self.active_alerts[alert_key]
            alert.resolved = True
            alert.resolved_at = datetime.now()
            
            # Enviar notificación de resolución
            if NOTIFICATIONS_AVAILABLE:
                notify_info(
                    f"Alerta Resuelta: {alert.title}",
                    f"La alerta {alert.title} se ha resuelto automáticamente",
                    alert.component
                )
            
            del self.active_alerts[alert_key]
    
    def get_active_alerts(self) -> List[SmartAlert]:
        """Obtiene todas las alertas activas."""
        return list(self.active_alerts.values())
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """Obtiene estadísticas de alertas."""
        
        total_alerts = len(self.alert_history)
        active_alerts = len(self.active_alerts)
        
        # Contar por severidad
        severity_counts = defaultdict(int)
        for alert in self.alert_history:
            severity_counts[alert.severity.value] += 1
        
        # Contar por componente
        component_counts = defaultdict(int)
        for alert in self.alert_history:
            component_counts[alert.component] += 1
        
        return {
            "total_alerts": total_alerts,
            "active_alerts": active_alerts,
            "resolved_alerts": total_alerts - active_alerts,
            "by_severity": dict(severity_counts),
            "by_component": dict(component_counts),
            "last_24h": len([
                a for a in self.alert_history 
                if a.timestamp > datetime.now() - timedelta(days=1)
            ])
        }

class EnhancedMonitoringSystem:
    """Sistema de monitoreo expandido principal."""
    
    def __init__(self, config: Optional[MonitoringConfig] = None):
        self.config = config or MonitoringConfig()
        self.logger = logging.getLogger(__name__)
        
        # Componentes del sistema
        self.metric_collector = AdvancedMetricCollector(self.config)
        self.anomaly_detector = AnomalyDetector(self.config)
        self.trend_analyzer = TrendAnalyzer(self.config)
        self.alert_manager = SmartAlertManager(self.config)
        
        # Almacenamiento de métricas
        self.metrics_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self.config.max_metrics_in_memory)
        )
        
        # Control de ejecución
        self.running = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.dashboard_thread: Optional[threading.Thread] = None
        
        # Eventos y callbacks
        self.metric_callbacks: List[Callable] = []
        self.alert_callbacks: List[Callable] = []
        
        # Estadísticas
        self.start_time = datetime.now()
        self.collection_count = 0
        self.last_export = datetime.now()
        
        self.logger.info("Sistema de monitoreo expandido inicializado")
    
    def start_monitoring(self):
        """Inicia el sistema de monitoreo."""
        
        if self.running:
            self.logger.warning("El sistema de monitoreo ya está ejecutándose")
            return
        
        self.running = True
        self.start_time = datetime.now()
        
        # Iniciar hilo de monitoreo
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        # Iniciar hilo de dashboard
        self.dashboard_thread = threading.Thread(target=self._dashboard_loop, daemon=True)
        self.dashboard_thread.start()
        
        self.logger.info("Sistema de monitoreo expandido iniciado")
    
    def stop_monitoring(self):
        """Detiene el sistema de monitoreo."""
        
        self.running = False
        
        # Esperar a que terminen los hilos
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
        
        if self.dashboard_thread and self.dashboard_thread.is_alive():
            self.dashboard_thread.join(timeout=5)
        
        self.logger.info("Sistema de monitoreo expandido detenido")
    
    def _monitoring_loop(self):
        """Bucle principal de monitoreo."""
        
        while self.running:
            try:
                start_time = time.time()
                
                # Recopilar métricas
                system_metrics = self.metric_collector.collect_system_metrics()
                gpu_metrics = self.metric_collector.collect_gpu_metrics()
                
                all_metrics = system_metrics + gpu_metrics
                
                # Procesar cada métrica
                for metric in all_metrics:
                    self._process_metric(metric)
                
                self.collection_count += 1
                
                # Exportar métricas periódicamente
                if (datetime.now() - self.last_export).total_seconds() > (self.config.export_metrics_interval_hours * 3600):
                    self._export_metrics()
                
                # Limpiar datos antiguos
                self._cleanup_old_data()
                
                # Calcular tiempo de espera
                processing_time = time.time() - start_time
                sleep_time = max(0, self.config.collection_interval - processing_time)
                
                time.sleep(sleep_time)
                
            except Exception as e:
                self.logger.error(f"Error en bucle de monitoreo: {e}")
                time.sleep(self.config.collection_interval)
    
    def _process_metric(self, metric: AdvancedMetric):
        """Procesa una métrica individual."""
        
        # Almacenar métrica
        self.metrics_history[metric.name].append(metric)
        
        # Análisis de anomalías (solo para métricas numéricas)
        anomaly_info = (False, 0.0)
        if isinstance(metric.value, (int, float)):
            self.anomaly_detector.add_metric_value(metric.name, float(metric.value))
            anomaly_info = self.anomaly_detector.detect_anomaly(metric.name, float(metric.value))
            
            # Análisis de tendencias
            self.trend_analyzer.add_metric_value(metric.name, float(metric.value), metric.timestamp)
        
        trend_info = self.trend_analyzer.analyze_trend(metric.name)
        
        # Verificar alertas
        alert = self.alert_manager.check_metric_alert(metric, trend_info, anomaly_info)
        
        # Ejecutar callbacks
        for callback in self.metric_callbacks:
            try:
                callback(metric, trend_info, anomaly_info, alert)
            except Exception as e:
                self.logger.error(f"Error ejecutando callback de métrica: {e}")
    
    def _dashboard_loop(self):
        """Bucle del dashboard en tiempo real."""
        
        while self.running:
            try:
                # Generar datos del dashboard
                dashboard_data = self.get_dashboard_data()
                
                # Ejecutar callbacks del dashboard
                for callback in self.alert_callbacks:
                    try:
                        callback(dashboard_data)
                    except Exception as e:
                        self.logger.error(f"Error ejecutando callback de dashboard: {e}")
                
                time.sleep(self.config.dashboard_update_interval)
                
            except Exception as e:
                self.logger.error(f"Error en bucle de dashboard: {e}")
                time.sleep(self.config.dashboard_update_interval)
    
    def _export_metrics(self):
        """Exporta métricas a archivo."""
        
        try:
            export_dir = Path("logs/metrics")
            export_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_file = export_dir / f"metrics_export_{timestamp}.json"
            
            # Preparar datos para exportación
            export_data = {
                "export_timestamp": datetime.now().isoformat(),
                "system_info": self.get_system_info(),
                "metrics": {},
                "alerts": [alert.to_dict() for alert in self.alert_manager.get_active_alerts()],
                "statistics": self.get_monitoring_statistics()
            }
            
            # Exportar métricas recientes
            for metric_name, metric_history in self.metrics_history.items():
                recent_metrics = list(metric_history)[-100:]  # Últimas 100 muestras
                export_data["metrics"][metric_name] = [
                    metric.to_dict() for metric in recent_metrics
                ]
            
            # Guardar archivo
            with open(export_file, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            self.last_export = datetime.now()
            self.logger.info(f"Métricas exportadas a {export_file}")
            
        except Exception as e:
            self.logger.error(f"Error exportando métricas: {e}")
    
    def _cleanup_old_data(self):
        """Limpia datos antiguos del sistema."""
        
        try:
            cutoff_time = datetime.now() - timedelta(hours=self.config.history_retention_hours)
            
            # Limpiar historial de métricas
            for metric_name in list(self.metrics_history.keys()):
                metric_history = self.metrics_history[metric_name]
                
                # Filtrar métricas antiguas
                filtered_metrics = deque(
                    [m for m in metric_history if m.timestamp > cutoff_time],
                    maxlen=self.config.max_metrics_in_memory
                )
                
                if len(filtered_metrics) == 0:
                    del self.metrics_history[metric_name]
                else:
                    self.metrics_history[metric_name] = filtered_metrics
            
        except Exception as e:
            self.logger.error(f"Error limpiando datos antiguos: {e}")
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Obtiene datos para el dashboard en tiempo real."""
        
        current_metrics = {}
        
        # Obtener métricas actuales
        for metric_name, metric_history in self.metrics_history.items():
            if metric_history:
                latest_metric = metric_history[-1]
                current_metrics[metric_name] = {
                    "value": latest_metric.value,
                    "unit": latest_metric.unit,
                    "timestamp": latest_metric.timestamp.isoformat(),
                    "category": latest_metric.category.value,
                    "trend": self.trend_analyzer.analyze_trend(metric_name)
                }
        
        return {
            "timestamp": datetime.now().isoformat(),
            "system_status": self._get_system_status(),
            "current_metrics": current_metrics,
            "active_alerts": [alert.to_dict() for alert in self.alert_manager.get_active_alerts()],
            "alert_statistics": self.alert_manager.get_alert_statistics(),
            "monitoring_statistics": self.get_monitoring_statistics()
        }
    
    def _get_system_status(self) -> str:
        """Determina el estado general del sistema."""
        
        active_alerts = self.alert_manager.get_active_alerts()
        
        # Verificar alertas críticas
        critical_alerts = [a for a in active_alerts if a.severity == AlertSeverity.CRITICAL]
        if critical_alerts:
            return "critical"
        
        # Verificar alertas altas
        high_alerts = [a for a in active_alerts if a.severity == AlertSeverity.HIGH]
        if high_alerts:
            return "warning"
        
        # Verificar alertas medias
        medium_alerts = [a for a in active_alerts if a.severity == AlertSeverity.MEDIUM]
        if medium_alerts:
            return "caution"
        
        return "healthy"
    
    def get_system_info(self) -> Dict[str, Any]:
        """Obtiene información del sistema."""
        
        info = {
            "monitoring_start_time": self.start_time.isoformat(),
            "uptime_hours": (datetime.now() - self.start_time).total_seconds() / 3600,
            "collection_count": self.collection_count,
            "config": asdict(self.config)
        }
        
        if PSUTIL_AVAILABLE:
            info.update({
                "cpu_count": psutil.cpu_count(),
                "memory_total_gb": psutil.virtual_memory().total / (1024**3),
                "platform": psutil.os.name if hasattr(psutil, 'os') else "unknown"
            })
        
        return info
    
    def get_monitoring_statistics(self) -> Dict[str, Any]:
        """Obtiene estadísticas del monitoreo."""
        
        return {
            "total_metrics_collected": sum(len(history) for history in self.metrics_history.values()),
            "unique_metrics": len(self.metrics_history),
            "collection_rate_per_minute": self.collection_count / max(1, (datetime.now() - self.start_time).total_seconds() / 60),
            "memory_usage_mb": sum(len(history) for history in self.metrics_history.values()) * 0.001,  # Estimación
            "alert_statistics": self.alert_manager.get_alert_statistics()
        }
    
    def add_metric_callback(self, callback: Callable):
        """Añade un callback para procesar métricas."""
        self.metric_callbacks.append(callback)
    
    def add_dashboard_callback(self, callback: Callable):
        """Añade un callback para el dashboard."""
        self.alert_callbacks.append(callback)
    
    def get_metric_history(self, metric_name: str, hours: int = 1) -> List[AdvancedMetric]:
        """Obtiene el historial de una métrica."""
        
        if metric_name not in self.metrics_history:
            return []
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [
            metric for metric in self.metrics_history[metric_name]
            if metric.timestamp > cutoff_time
        ]
    
    def get_current_metrics(self) -> Dict[str, AdvancedMetric]:
        """Obtiene las métricas actuales."""
        
        current = {}
        for metric_name, metric_history in self.metrics_history.items():
            if metric_history:
                current[metric_name] = metric_history[-1]
        
        return current

# Instancia global del sistema de monitoreo
_enhanced_monitoring_system = None

def get_enhanced_monitoring_system(config: Optional[MonitoringConfig] = None) -> EnhancedMonitoringSystem:
    """Obtiene la instancia global del sistema de monitoreo expandido."""
    global _enhanced_monitoring_system
    if _enhanced_monitoring_system is None:
        _enhanced_monitoring_system = EnhancedMonitoringSystem(config)
    return _enhanced_monitoring_system

# Funciones de conveniencia
def start_enhanced_monitoring(config: Optional[MonitoringConfig] = None):
    """Inicia el sistema de monitoreo expandido."""
    system = get_enhanced_monitoring_system(config)
    system.start_monitoring()

def stop_enhanced_monitoring():
    """Detiene el sistema de monitoreo expandido."""
    system = get_enhanced_monitoring_system()
    system.stop_monitoring()

def get_dashboard_data() -> Dict[str, Any]:
    """Obtiene datos del dashboard en tiempo real."""
    system = get_enhanced_monitoring_system()
    return system.get_dashboard_data()

def get_active_alerts() -> List[SmartAlert]:
    """Obtiene las alertas activas."""
    system = get_enhanced_monitoring_system()
    return system.alert_manager.get_active_alerts()

def get_monitoring_statistics() -> Dict[str, Any]:
    """Obtiene estadísticas del monitoreo."""
    system = get_enhanced_monitoring_system()
    return system.get_monitoring_statistics()