#!/usr/bin/env python3
"""
Sistema de telemetría para SEACABAr.
Recopila datos de uso, rendimiento y eventos para análisis y mejora del sistema.
"""

import time
import threading
import uuid
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import hashlib
import platform

# Importar sistema de métricas
try:
    from performance.metrics_system import get_metrics_system
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False

class EventType(Enum):
    """Tipos de eventos de telemetría."""
    USER_ACTION = "user_action"
    SYSTEM_EVENT = "system_event"
    PERFORMANCE = "performance"
    ERROR = "error"
    FEATURE_USAGE = "feature_usage"
    SESSION = "session"

class EventSeverity(Enum):
    """Severidad de eventos."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class TelemetryEvent:
    """Representa un evento de telemetría."""
    id: str
    timestamp: datetime
    event_type: EventType
    severity: EventSeverity
    component: str
    action: str
    data: Dict[str, Any] = field(default_factory=dict)
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    duration_ms: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SessionInfo:
    """Información de sesión de usuario."""
    session_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    user_id: Optional[str] = None
    platform: str = ""
    version: str = ""
    events_count: int = 0
    duration_seconds: float = 0.0

class TelemetryCollector:
    """Recopilador base para datos de telemetría."""
    
    def __init__(self, name: str):
        self.name = name
        self.enabled = True
    
    def collect(self) -> Dict[str, Any]:
        """Recopila datos de telemetría. Debe ser implementado por subclases."""
        raise NotImplementedError

class SystemInfoCollector(TelemetryCollector):
    """Recopila información del sistema."""
    
    def collect(self) -> Dict[str, Any]:
        return {
            'platform': platform.system(),
            'platform_version': platform.version(),
            'architecture': platform.architecture()[0],
            'processor': platform.processor(),
            'python_version': platform.python_version(),
            'hostname_hash': hashlib.sha256(platform.node().encode()).hexdigest()[:16]
        }

class PerformanceCollector(TelemetryCollector):
    """Recopila métricas de rendimiento."""
    
    def collect(self) -> Dict[str, Any]:
        if not METRICS_AVAILABLE:
            return {}
        
        try:
            metrics_system = get_metrics_system()
            current_metrics = metrics_system.get_current_metrics()
            health_score = metrics_system.get_system_health_score()
            
            return {
                'health_score': health_score,
                'cpu_usage': current_metrics.get('cpu_usage', {}).get('value', 0),
                'memory_usage': current_metrics.get('memory_usage', {}).get('value', 0),
                'disk_usage': current_metrics.get('disk_usage', {}).get('value', 0),
                'gpu_available': 'gpu.memory_usage_percent' in current_metrics,
                'gpu_usage': current_metrics.get('gpu.memory_usage_percent', {}).get('value', 0)
            }
        except Exception:
            return {}

class FeatureUsageCollector(TelemetryCollector):
    """Recopila datos de uso de características."""
    
    def __init__(self, name: str):
        super().__init__(name)
        self.feature_counts = {}
        self.lock = threading.Lock()
    
    def record_feature_usage(self, feature: str, metadata: Optional[Dict[str, Any]] = None):
        """Registra el uso de una característica."""
        with self.lock:
            if feature not in self.feature_counts:
                self.feature_counts[feature] = {
                    'count': 0,
                    'last_used': None,
                    'metadata': {}
                }
            
            self.feature_counts[feature]['count'] += 1
            self.feature_counts[feature]['last_used'] = datetime.now().isoformat()
            
            if metadata:
                self.feature_counts[feature]['metadata'].update(metadata)
    
    def collect(self) -> Dict[str, Any]:
        with self.lock:
            return self.feature_counts.copy()

class TelemetrySystem:
    """Sistema principal de telemetría."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.events: List[TelemetryEvent] = []
        self.sessions: Dict[str, SessionInfo] = {}
        self.collectors: Dict[str, TelemetryCollector] = {}
        self.current_session: Optional[SessionInfo] = None
        self.lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
        
        # Configuración
        self.enabled = self.config.get('enabled', True)
        self.max_events = self.config.get('max_events', 10000)
        self.auto_flush_interval = self.config.get('auto_flush_interval', 300)  # 5 minutos
        self.data_retention_days = self.config.get('data_retention_days', 30)
        self.privacy_mode = self.config.get('privacy_mode', True)
        
        # Archivos de datos
        self.data_dir = Path(self.config.get('data_dir', 'telemetry'))
        self.data_dir.mkdir(exist_ok=True)
        
        # Configurar recopiladores por defecto
        self._setup_default_collectors()
        
        # Iniciar sesión automáticamente
        if self.enabled:
            self.start_session()
            
        # Hilo de limpieza automática
        self._start_cleanup_thread()
    
    def _setup_default_collectors(self):
        """Configura recopiladores por defecto."""
        self.collectors['system_info'] = SystemInfoCollector('system_info')
        self.collectors['performance'] = PerformanceCollector('performance')
        self.collectors['feature_usage'] = FeatureUsageCollector('feature_usage')
    
    def _start_cleanup_thread(self):
        """Inicia hilo de limpieza automática."""
        def cleanup_loop():
            while True:
                try:
                    time.sleep(self.auto_flush_interval)
                    self._cleanup_old_data()
                    self._flush_to_disk()
                except Exception as e:
                    self.logger.error(f"Error en limpieza automática: {e}")
        
        cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
        cleanup_thread.start()
    
    def start_session(self, user_id: Optional[str] = None) -> str:
        """Inicia una nueva sesión de telemetría."""
        
        if not self.enabled:
            return ""
        
        # Finalizar sesión anterior si existe
        if self.current_session:
            self.end_session()
        
        # Crear nueva sesión
        session_id = str(uuid.uuid4())
        
        # Recopilar información del sistema
        system_info = self.collectors['system_info'].collect()
        
        session = SessionInfo(
            session_id=session_id,
            start_time=datetime.now(),
            user_id=self._anonymize_user_id(user_id) if user_id else None,
            platform=system_info.get('platform', 'unknown'),
            version=self.config.get('app_version', '1.0.0')
        )
        
        with self.lock:
            self.sessions[session_id] = session
            self.current_session = session
        
        # Registrar evento de inicio de sesión
        self.record_event(
            EventType.SESSION,
            EventSeverity.INFO,
            "telemetry",
            "session_start",
            data={
                'session_id': session_id,
                'system_info': system_info
            }
        )
        
        self.logger.info(f"Sesión de telemetría iniciada: {session_id}")
        return session_id
    
    def end_session(self):
        """Finaliza la sesión actual."""
        
        if not self.current_session:
            return
        
        session = self.current_session
        session.end_time = datetime.now()
        session.duration_seconds = (session.end_time - session.start_time).total_seconds()
        
        # Registrar evento de fin de sesión
        self.record_event(
            EventType.SESSION,
            EventSeverity.INFO,
            "telemetry",
            "session_end",
            data={
                'session_id': session.session_id,
                'duration_seconds': session.duration_seconds,
                'events_count': session.events_count
            }
        )
        
        self.current_session = None
        self.logger.info(f"Sesión de telemetría finalizada: {session.session_id}")
    
    def record_event(self, 
                    event_type: EventType,
                    severity: EventSeverity,
                    component: str,
                    action: str,
                    data: Optional[Dict[str, Any]] = None,
                    duration_ms: Optional[float] = None,
                    metadata: Optional[Dict[str, Any]] = None) -> str:
        """Registra un evento de telemetría."""
        
        if not self.enabled:
            return ""
        
        event_id = str(uuid.uuid4())
        
        event = TelemetryEvent(
            id=event_id,
            timestamp=datetime.now(),
            event_type=event_type,
            severity=severity,
            component=component,
            action=action,
            data=self._sanitize_data(data or {}),
            user_id=self.current_session.user_id if self.current_session else None,
            session_id=self.current_session.session_id if self.current_session else None,
            duration_ms=duration_ms,
            metadata=metadata or {}
        )
        
        with self.lock:
            self.events.append(event)
            
            # Actualizar contador de eventos de la sesión
            if self.current_session:
                self.current_session.events_count += 1
            
            # Limpiar eventos antiguos si hay demasiados
            if len(self.events) > self.max_events:
                self.events = self.events[-self.max_events:]
        
        return event_id
    
    def record_user_action(self, action: str, component: str, 
                          data: Optional[Dict[str, Any]] = None,
                          duration_ms: Optional[float] = None) -> str:
        """Registra una acción del usuario."""
        return self.record_event(
            EventType.USER_ACTION,
            EventSeverity.INFO,
            component,
            action,
            data,
            duration_ms
        )
    
    def record_feature_usage(self, feature: str, component: str,
                           metadata: Optional[Dict[str, Any]] = None) -> str:
        """Registra el uso de una característica."""
        
        # Actualizar contador en el recopilador
        feature_collector = self.collectors.get('feature_usage')
        if isinstance(feature_collector, FeatureUsageCollector):
            feature_collector.record_feature_usage(feature, metadata)
        
        return self.record_event(
            EventType.FEATURE_USAGE,
            EventSeverity.INFO,
            component,
            f"feature_used_{feature}",
            data={'feature': feature},
            metadata=metadata
        )
    
    def record_performance_event(self, operation: str, component: str,
                               duration_ms: float,
                               success: bool = True,
                               metadata: Optional[Dict[str, Any]] = None) -> str:
        """Registra un evento de rendimiento."""
        
        # Recopilar métricas de rendimiento actuales
        performance_data = self.collectors['performance'].collect()
        performance_data.update({
            'operation': operation,
            'duration_ms': duration_ms,
            'success': success
        })
        
        severity = EventSeverity.INFO if success else EventSeverity.WARNING
        
        return self.record_event(
            EventType.PERFORMANCE,
            severity,
            component,
            f"performance_{operation}",
            data=performance_data,
            duration_ms=duration_ms,
            metadata=metadata
        )
    
    def record_error_event(self, error: Exception, component: str,
                          operation: str,
                          metadata: Optional[Dict[str, Any]] = None) -> str:
        """Registra un evento de error."""
        
        error_data = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'operation': operation
        }
        
        return self.record_event(
            EventType.ERROR,
            EventSeverity.ERROR,
            component,
            f"error_{operation}",
            data=error_data,
            metadata=metadata
        )
    
    def _sanitize_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitiza datos sensibles según el modo de privacidad."""
        
        if not self.privacy_mode:
            return data
        
        sanitized = {}
        sensitive_keys = {'password', 'token', 'key', 'secret', 'auth', 'credential'}
        
        for key, value in data.items():
            key_lower = key.lower()
            
            # Verificar si la clave contiene información sensible
            if any(sensitive in key_lower for sensitive in sensitive_keys):
                sanitized[key] = "[REDACTED]"
            elif isinstance(value, str) and len(value) > 100:
                # Truncar strings muy largos
                sanitized[key] = value[:100] + "..."
            elif isinstance(value, dict):
                # Recursivamente sanitizar diccionarios anidados
                sanitized[key] = self._sanitize_data(value)
            else:
                sanitized[key] = value
        
        return sanitized
    
    def _anonymize_user_id(self, user_id: str) -> str:
        """Anonimiza el ID de usuario."""
        return hashlib.sha256(user_id.encode()).hexdigest()[:16]
    
    def get_events(self, 
                  event_type: Optional[EventType] = None,
                  component: Optional[str] = None,
                  session_id: Optional[str] = None,
                  limit: int = 100) -> List[TelemetryEvent]:
        """Obtiene eventos filtrados."""
        
        with self.lock:
            filtered_events = self.events.copy()
        
        # Aplicar filtros
        if event_type:
            filtered_events = [e for e in filtered_events if e.event_type == event_type]
        
        if component:
            filtered_events = [e for e in filtered_events if e.component == component]
        
        if session_id:
            filtered_events = [e for e in filtered_events if e.session_id == session_id]
        
        # Ordenar por timestamp descendente y limitar
        filtered_events.sort(key=lambda x: x.timestamp, reverse=True)
        return filtered_events[:limit]
    
    def get_session_analytics(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Obtiene analíticas de una sesión."""
        
        target_session = None
        
        if session_id:
            target_session = self.sessions.get(session_id)
        else:
            target_session = self.current_session
        
        if not target_session:
            return {}
        
        # Obtener eventos de la sesión
        session_events = self.get_events(session_id=target_session.session_id, limit=10000)
        
        # Calcular estadísticas
        event_types = {}
        components = {}
        actions = {}
        
        for event in session_events:
            # Por tipo
            event_types[event.event_type.value] = event_types.get(event.event_type.value, 0) + 1
            
            # Por componente
            components[event.component] = components.get(event.component, 0) + 1
            
            # Por acción
            actions[event.action] = actions.get(event.action, 0) + 1
        
        return {
            'session_info': asdict(target_session),
            'total_events': len(session_events),
            'event_types': event_types,
            'components': components,
            'top_actions': dict(sorted(actions.items(), key=lambda x: x[1], reverse=True)[:10]),
            'feature_usage': self.collectors['feature_usage'].collect()
        }
    
    def export_data(self, filepath: str, 
                   start_date: Optional[datetime] = None,
                   end_date: Optional[datetime] = None):
        """Exporta datos de telemetría a un archivo."""
        
        # Filtrar eventos por fecha si se especifica
        events_to_export = []
        
        with self.lock:
            for event in self.events:
                if start_date and event.timestamp < start_date:
                    continue
                if end_date and event.timestamp > end_date:
                    continue
                events_to_export.append(event)
        
        # Preparar datos para exportación
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'start_date': start_date.isoformat() if start_date else None,
            'end_date': end_date.isoformat() if end_date else None,
            'total_events': len(events_to_export),
            'sessions': [asdict(session) for session in self.sessions.values()],
            'events': [asdict(event) for event in events_to_export],
            'collectors_data': {
                name: collector.collect() 
                for name, collector in self.collectors.items()
            }
        }
        
        # Guardar archivo
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)
        
        self.logger.info(f"Datos de telemetría exportados a {filepath}")
    
    def _flush_to_disk(self):
        """Guarda datos de telemetría en disco."""
        
        if not self.enabled:
            return
        
        try:
            # Guardar eventos recientes
            recent_events_file = self.data_dir / f"events_{datetime.now().strftime('%Y%m%d')}.json"
            
            # Obtener eventos del día actual
            today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            today_events = [
                asdict(event) for event in self.events 
                if event.timestamp >= today
            ]
            
            if today_events:
                with open(recent_events_file, 'w', encoding='utf-8') as f:
                    json.dump(today_events, f, indent=2, ensure_ascii=False, default=str)
            
            # Guardar sesiones
            sessions_file = self.data_dir / "sessions.json"
            sessions_data = [asdict(session) for session in self.sessions.values()]
            
            with open(sessions_file, 'w', encoding='utf-8') as f:
                json.dump(sessions_data, f, indent=2, ensure_ascii=False, default=str)
                
        except Exception as e:
            self.logger.error(f"Error guardando datos de telemetría: {e}")
    
    def _cleanup_old_data(self):
        """Limpia datos antiguos según la política de retención."""
        
        cutoff_date = datetime.now() - timedelta(days=self.data_retention_days)
        
        with self.lock:
            # Limpiar eventos antiguos
            self.events = [e for e in self.events if e.timestamp >= cutoff_date]
            
            # Limpiar sesiones antiguas
            old_session_ids = [
                sid for sid, session in self.sessions.items()
                if session.start_time < cutoff_date
            ]
            
            for sid in old_session_ids:
                del self.sessions[sid]
        
        # Limpiar archivos antiguos
        try:
            for file_path in self.data_dir.glob("events_*.json"):
                if file_path.stat().st_mtime < cutoff_date.timestamp():
                    file_path.unlink()
        except Exception as e:
            self.logger.error(f"Error limpiando archivos antiguos: {e}")

# Instancia global del sistema de telemetría
_telemetry_system = None

def get_telemetry_system() -> TelemetrySystem:
    """Obtiene la instancia global del sistema de telemetría."""
    global _telemetry_system
    if _telemetry_system is None:
        _telemetry_system = TelemetrySystem()
    return _telemetry_system

# Funciones de conveniencia
def record_user_action(action: str, component: str, **kwargs) -> str:
    """Registra una acción del usuario."""
    return get_telemetry_system().record_user_action(action, component, **kwargs)

def record_feature_usage(feature: str, component: str, **kwargs) -> str:
    """Registra el uso de una característica."""
    return get_telemetry_system().record_feature_usage(feature, component, **kwargs)

def record_performance_event(operation: str, component: str, duration_ms: float, **kwargs) -> str:
    """Registra un evento de rendimiento."""
    return get_telemetry_system().record_performance_event(operation, component, duration_ms, **kwargs)

def record_error_event(error: Exception, component: str, operation: str, **kwargs) -> str:
    """Registra un evento de error."""
    return get_telemetry_system().record_error_event(error, component, operation, **kwargs)

# Decorador para medir rendimiento automáticamente
def measure_performance(component: str, operation: str = None):
    """Decorador para medir automáticamente el rendimiento de funciones."""
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            op_name = operation or func.__name__
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000
                
                record_performance_event(
                    op_name, 
                    component, 
                    duration_ms, 
                    success=True
                )
                
                return result
                
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                
                record_performance_event(
                    op_name, 
                    component, 
                    duration_ms, 
                    success=False
                )
                
                record_error_event(e, component, op_name)
                raise
        
        return wrapper
    return decorator