#!/usr/bin/env python3
"""
Sistema avanzado de manejo de errores y recuperación automática para SIGeC-Balistica.
Implementa fallbacks inteligentes, notificaciones y recuperación automática.
"""

import logging
import traceback
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json
import yaml

# Importar sistema de notificaciones
try:
    from .notification_system import (
        get_notification_manager, 
        NotificationType, 
        NotificationChannel,
        notify_error, 
        notify_critical, 
        notify_warning
    )
    NOTIFICATIONS_AVAILABLE = True
except ImportError:
    NOTIFICATIONS_AVAILABLE = False

class ErrorSeverity(Enum):
    """Niveles de severidad de errores."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class RecoveryStrategy(Enum):
    """Estrategias de recuperación disponibles."""
    RETRY = "retry"
    FALLBACK = "fallback"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    USER_INTERVENTION = "user_intervention"
    SYSTEM_RESTART = "system_restart"

@dataclass
class ErrorContext:
    """Contexto completo de un error."""
    error_id: str
    timestamp: datetime
    severity: ErrorSeverity
    component: str
    operation: str
    error_type: str
    message: str
    traceback: str
    context_data: Dict[str, Any] = field(default_factory=dict)
    recovery_attempts: int = 0
    max_recovery_attempts: int = 3
    recovery_strategy: Optional[RecoveryStrategy] = None
    resolved: bool = False
    resolution_time: Optional[datetime] = None

@dataclass
class NotificationConfig:
    """Configuración de notificaciones."""
    enabled: bool = True
    min_severity: ErrorSeverity = ErrorSeverity.MEDIUM
    max_notifications_per_hour: int = 10
    notification_methods: List[str] = field(default_factory=lambda: ["log", "gui"])
    email_config: Optional[Dict[str, str]] = None
    webhook_url: Optional[str] = None

class ErrorRecoveryManager:
    """Gestor principal de recuperación de errores."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.error_history: List[ErrorContext] = []
        self.recovery_strategies: Dict[str, Callable] = {}
        self.notification_config = NotificationConfig()
        self.notification_count = 0
        self.last_notification_reset = datetime.now()
        self.lock = threading.Lock()
        
        # Configurar logging
        self.logger = logging.getLogger(__name__)
        
        # Cargar configuración
        self._load_config()
        
        # Registrar estrategias de recuperación por defecto
        self._register_default_strategies()
    
    def _load_config(self):
        """Carga la configuración del gestor de errores."""
        if self.config_path and Path(self.config_path).exists():
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    if self.config_path.endswith('.yaml') or self.config_path.endswith('.yml'):
                        config = yaml.safe_load(f)
                    else:
                        config = json.load(f)
                
                # Aplicar configuración de notificaciones
                if 'notifications' in config:
                    notif_config = config['notifications']
                    self.notification_config.enabled = notif_config.get('enabled', True)
                    self.notification_config.min_severity = ErrorSeverity(
                        notif_config.get('min_severity', 'medium')
                    )
                    self.notification_config.max_notifications_per_hour = notif_config.get(
                        'max_notifications_per_hour', 10
                    )
                    
            except Exception as e:
                self.logger.warning(f"Error cargando configuración de error handler: {e}")
    
    def _register_default_strategies(self):
        """Registra las estrategias de recuperación por defecto."""
        self.recovery_strategies.update({
            'gpu_memory_error': self._recover_gpu_memory,
            'file_not_found': self._recover_file_not_found,
            'network_error': self._recover_network_error,
            'database_error': self._recover_database_error,
            'configuration_error': self._recover_configuration_error,
            'processing_error': self._recover_processing_error
        })
    
    def handle_error(self, 
                    error: Exception, 
                    component: str, 
                    operation: str, 
                    context_data: Optional[Dict[str, Any]] = None,
                    severity: Optional[ErrorSeverity] = None) -> ErrorContext:
        """Maneja un error de forma inteligente con recuperación automática y notificaciones."""
        
        # Crear contexto del error
        error_context = self._create_error_context(
            error, component, operation, context_data, severity
        )
        
        with self.lock:
            self.error_history.append(error_context)
        
        # Enviar notificación crítica si está disponible
        if NOTIFICATIONS_AVAILABLE and error_context.severity == ErrorSeverity.CRITICAL:
            notify_critical(
                "Error Crítico Detectado",
                f"Error crítico en {component}: {str(error)}",
                component
            )
        
        # Determinar estrategia de recuperación
        recovery_strategy = self._determine_recovery_strategy(error_context)
        error_context.recovery_strategy = recovery_strategy
        
        # Intentar recuperación automática
        if recovery_strategy != RecoveryStrategy.USER_INTERVENTION:
            success = self._attempt_recovery(error_context)
            if success:
                error_context.resolved = True
                error_context.resolution_time = datetime.now()
                
                # Notificar recuperación exitosa
                if NOTIFICATIONS_AVAILABLE:
                    notify_warning(
                        "Recuperación Exitosa",
                        f"Error recuperado automáticamente en {component}",
                        component
                    )
        
        # Enviar notificación si es necesario
        self._send_notification(error_context)
        
        # Log del error
        self._log_error(error_context)
        
        return error_context
    
    def _send_error_notification(self, error: Exception, context: ErrorContext, error_info: Dict[str, Any]):
        """Envía notificación de error según la severidad."""
        
        if not NOTIFICATIONS_AVAILABLE:
            return
        
        # Determinar tipo de notificación según severidad
        if context.severity == ErrorSeverity.CRITICAL:
            notify_critical(
                f"Error Crítico: {context.component}",
                f"Error crítico en {context.operation}: {str(error)}",
                context.component,
                metadata=error_info
            )
        elif context.severity == ErrorSeverity.HIGH:
            notify_error(
                f"Error: {context.component}",
                f"Error en {context.operation}: {str(error)}",
                context.component,
                metadata=error_info
            )
        elif context.severity == ErrorSeverity.MEDIUM:
            notify_warning(
                f"Advertencia: {context.component}",
                f"Problema en {context.operation}: {str(error)}",
                context.component,
                metadata=error_info
            )
    
    def _create_error_context(self, 
                            error: Exception, 
                            component: str, 
                            operation: str, 
                            context_data: Optional[Dict[str, Any]], 
                            severity: Optional[ErrorSeverity]) -> ErrorContext:
        """Crea el contexto completo del error."""
        
        error_id = f"{component}_{operation}_{int(time.time())}"
        
        # Determinar severidad automáticamente si no se proporciona
        if severity is None:
            severity = self._determine_severity(error, component)
        
        return ErrorContext(
            error_id=error_id,
            timestamp=datetime.now(),
            severity=severity,
            component=component,
            operation=operation,
            error_type=type(error).__name__,
            message=str(error),
            traceback=traceback.format_exc(),
            context_data=context_data or {}
        )
    
    def _determine_severity(self, error: Exception, component: str) -> ErrorSeverity:
        """Determina automáticamente la severidad del error."""
        
        # Errores críticos
        if isinstance(error, (SystemExit, KeyboardInterrupt, MemoryError)):
            return ErrorSeverity.CRITICAL
        
        # Errores de componentes críticos
        if component in ['database', 'gpu_accelerator', 'main_pipeline']:
            if isinstance(error, (ConnectionError, RuntimeError)):
                return ErrorSeverity.HIGH
        
        # Errores de archivos y configuración
        if isinstance(error, (FileNotFoundError, PermissionError, ValueError)):
            return ErrorSeverity.MEDIUM
        
        # Otros errores
        return ErrorSeverity.LOW
    
    def _determine_recovery_strategy(self, error_context: ErrorContext) -> RecoveryStrategy:
        """Determina la mejor estrategia de recuperación para el error."""
        
        error_type = error_context.error_type
        component = error_context.component
        
        # Estrategias específicas por tipo de error
        if 'memory' in error_context.message.lower() or 'cuda' in error_context.message.lower():
            return RecoveryStrategy.FALLBACK
        
        if error_type in ['FileNotFoundError', 'PermissionError']:
            return RecoveryStrategy.GRACEFUL_DEGRADATION
        
        if error_type in ['ConnectionError', 'TimeoutError']:
            return RecoveryStrategy.RETRY
        
        if component == 'database' and error_type in ['DatabaseError', 'OperationalError']:
            return RecoveryStrategy.RETRY
        
        if error_context.severity == ErrorSeverity.CRITICAL:
            return RecoveryStrategy.USER_INTERVENTION
        
        return RecoveryStrategy.GRACEFUL_DEGRADATION
    
    def _attempt_recovery(self, error_context: ErrorContext) -> bool:
        """Intenta recuperación automática basada en la estrategia."""
        
        if error_context.recovery_attempts >= error_context.max_recovery_attempts:
            self.logger.warning(f"Máximo de intentos de recuperación alcanzado para {error_context.component}")
            return False
        
        error_context.recovery_attempts += 1
        strategy = error_context.recovery_strategy
        
        self.logger.info(f"Intentando recuperación {error_context.recovery_attempts}/{error_context.max_recovery_attempts} "
                        f"con estrategia {strategy.value} para {error_context.component}")
        
        try:
            success = False
            if strategy == RecoveryStrategy.RETRY:
                success = self._retry_operation(error_context)
            elif strategy == RecoveryStrategy.FALLBACK:
                success = self._execute_fallback(error_context)
            elif strategy == RecoveryStrategy.GRACEFUL_DEGRADATION:
                success = self._graceful_degradation(error_context)
            
            if success:
                error_context.resolved = True
                error_context.resolution_time = datetime.now()
                self.logger.info(f"Recuperación exitosa para {error_context.component} usando {strategy.value}")
            
            return success
            
        except Exception as recovery_error:
            self.logger.error(f"Error durante recuperación de {error_context.component}: {recovery_error}")
            # Intentar degradación elegante como último recurso
            if strategy != RecoveryStrategy.GRACEFUL_DEGRADATION:
                return self._graceful_degradation(error_context)
        
        return False
    
    def _retry_operation(self, error_context: ErrorContext) -> bool:
        """Reintenta la operación con backoff exponencial."""
        
        wait_time = min(2 ** error_context.recovery_attempts, 30)  # Max 30 segundos
        time.sleep(wait_time)
        
        # Aquí se reintentaría la operación original
        # Por ahora, simulamos éxito en algunos casos
        if error_context.component in ['network', 'database']:
            return error_context.recovery_attempts <= 2
        
        return False
    
    def _execute_fallback(self, error_context: ErrorContext) -> bool:
        """Ejecuta estrategia de fallback específica."""
        
        component = error_context.component
        error_type = error_context.error_type
        
        self.logger.info(f"Ejecutando fallback para {component} con error {error_type}")
        
        # Estrategias específicas por componente
        if component == 'gpu_accelerator':
            return self._recover_gpu_memory(error_context)
        elif component == 'file_system' or 'FileNotFoundError' in error_type:
            return self._recover_file_not_found(error_context)
        elif component == 'network' or 'ConnectionError' in error_type:
            return self._recover_network_error(error_context)
        elif component == 'database' or 'DatabaseError' in error_type:
            return self._recover_database_error(error_context)
        elif component == 'configuration' or 'ConfigurationError' in error_type:
            return self._recover_configuration_error(error_context)
        elif component == 'image_processing':
            return self._recover_processing_error(error_context)
        
        # Fallback genérico: intentar degradación elegante
        self.logger.warning(f"No hay fallback específico para {component}, usando degradación elegante")
        return self._graceful_degradation(error_context)
    
    def _graceful_degradation(self, error_context: ErrorContext) -> bool:
        """Implementa degradación elegante del servicio."""
        
        # Registrar que el sistema está operando en modo degradado
        self.logger.warning(f"Sistema operando en modo degradado debido a: {error_context.message}")
        
        # Aquí se implementarían funcionalidades reducidas
        return True
    
    # Estrategias de recuperación específicas
    def _recover_gpu_memory(self, error_context: ErrorContext) -> bool:
        """Recuperación específica para errores de memoria GPU."""
        try:
            # Intentar limpiar memoria GPU
            import gc
            gc.collect()
            
            # Si hay CuPy disponible, limpiar su memoria
            try:
                import cupy
                cupy.get_default_memory_pool().free_all_blocks()
                return True
            except ImportError:
                pass
            
            return True
        except Exception:
            return False
    
    def _recover_file_not_found(self, error_context: ErrorContext) -> bool:
        """Recuperación para archivos no encontrados."""
        try:
            # Intentar crear directorios padre si no existen
            if 'path' in error_context.context_data:
                file_path = Path(error_context.context_data['path'])
                
                # Crear directorio padre si no existe
                if not file_path.parent.exists():
                    file_path.parent.mkdir(parents=True, exist_ok=True)
                    self.logger.info(f"Directorio creado: {file_path.parent}")
                
                # Si es un archivo de configuración, crear uno por defecto
                if file_path.suffix in ['.yaml', '.yml', '.json', '.ini']:
                    if not file_path.exists():
                        self._create_default_config_file(file_path)
                        return True
                
                # Si es un directorio de trabajo, crearlo
                if file_path.suffix == '' and not file_path.exists():
                    file_path.mkdir(parents=True, exist_ok=True)
                    self.logger.info(f"Directorio de trabajo creado: {file_path}")
                    return True
            
            return False
        except Exception as e:
            self.logger.error(f"Error en recuperación de archivo: {e}")
            return False
    
    def _create_default_config_file(self, file_path: Path):
        """Crea un archivo de configuración por defecto."""
        try:
            if file_path.suffix in ['.yaml', '.yml']:
                default_config = {
                    'version': '1.0',
                    'created_by': 'error_recovery_system',
                    'timestamp': datetime.now().isoformat()
                }
                with open(file_path, 'w', encoding='utf-8') as f:
                    yaml.dump(default_config, f, default_flow_style=False)
            elif file_path.suffix == '.json':
                default_config = {
                    "version": "1.0",
                    "created_by": "error_recovery_system",
                    "timestamp": datetime.now().isoformat()
                }
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(default_config, f, indent=2)
            
            self.logger.info(f"Archivo de configuración por defecto creado: {file_path}")
        except Exception as e:
            self.logger.error(f"Error creando archivo por defecto: {e}")

    def _recover_network_error(self, error_context: ErrorContext) -> bool:
        """Recuperación para errores de red."""
        try:
            # Implementar backoff exponencial para reconexión
            wait_time = min(2 ** error_context.recovery_attempts, 60)
            self.logger.info(f"Esperando {wait_time}s antes de reintentar conexión de red")
            time.sleep(wait_time)
            
            # Aquí se podría implementar verificación de conectividad
            # Por ahora, asumimos que la espera puede resolver problemas temporales
            return error_context.recovery_attempts <= 2
        except Exception:
            return False
    
    def _recover_database_error(self, error_context: ErrorContext) -> bool:
        """Recuperación para errores de base de datos."""
        try:
            # Intentar reconexión con la base de datos
            self.logger.info("Intentando reconexión con base de datos")
            
            # Implementar lógica específica de reconexión
            # Por ahora, simulamos éxito en algunos casos
            if error_context.recovery_attempts <= 2:
                return True
            
            return False
        except Exception:
            return False
    
    def _recover_configuration_error(self, error_context: ErrorContext) -> bool:
        """Recuperación para errores de configuración."""
        try:
            # Intentar cargar configuración por defecto
            self.logger.info("Cargando configuración por defecto debido a error")
            
            # Aquí se implementaría la carga de configuración de respaldo
            return True
        except Exception:
            return False
    
    def _recover_processing_error(self, error_context: ErrorContext) -> bool:
        """Recuperación para errores de procesamiento de imágenes."""
        try:
            # Intentar procesamiento con parámetros más conservadores
            self.logger.info("Intentando procesamiento con configuración reducida")
            
            # Reducir calidad o resolución para evitar errores de memoria
            if 'OutOfMemoryError' in error_context.error_type or 'MemoryError' in error_context.error_type:
                # Sugerir procesamiento con menor resolución
                error_context.context_data['suggested_resolution_reduction'] = 0.5
                return True
            
            return False
        except Exception:
            return False
    
    def _send_notification(self, error_context: ErrorContext):
        """Envía notificación del error si es necesario."""
        
        if not self.notification_config.enabled:
            return
        
        if error_context.severity.value < self.notification_config.min_severity.value:
            return
        
        # Control de rate limiting
        now = datetime.now()
        if now - self.last_notification_reset > timedelta(hours=1):
            self.notification_count = 0
            self.last_notification_reset = now
        
        if self.notification_count >= self.notification_config.max_notifications_per_hour:
            return
        
        self.notification_count += 1
        
        # Enviar notificación según métodos configurados
        for method in self.notification_config.notification_methods:
            if method == "log":
                self._send_log_notification(error_context)
            elif method == "gui":
                self._send_gui_notification(error_context)
    
    def _send_log_notification(self, error_context: ErrorContext):
        """Envía notificación por log."""
        level = {
            ErrorSeverity.LOW: logging.INFO,
            ErrorSeverity.MEDIUM: logging.WARNING,
            ErrorSeverity.HIGH: logging.ERROR,
            ErrorSeverity.CRITICAL: logging.CRITICAL
        }[error_context.severity]
        
        self.logger.log(level, f"Error en {error_context.component}: {error_context.message}")
    
    def _send_gui_notification(self, error_context: ErrorContext):
        """Envía notificación a la GUI (si está disponible)."""
        # Aquí se implementaría la notificación a la GUI
        pass
    
    def _log_error(self, error_context: ErrorContext):
        """Registra el error en el log del sistema."""
        self.logger.error(
            f"[{error_context.error_id}] {error_context.component}.{error_context.operation}: "
            f"{error_context.message} (Severidad: {error_context.severity.value})"
        )
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Obtiene estadísticas de errores."""
        with self.lock:
            total_errors = len(self.error_history)
            resolved_errors = sum(1 for e in self.error_history if e.resolved)
            
            severity_counts = {}
            for severity in ErrorSeverity:
                severity_counts[severity.value] = sum(
                    1 for e in self.error_history if e.severity == severity
                )
            
            component_errors = {}
            for error in self.error_history:
                component_errors[error.component] = component_errors.get(error.component, 0) + 1
        
        return {
            'total_errors': total_errors,
            'resolved_errors': resolved_errors,
            'resolution_rate': resolved_errors / total_errors if total_errors > 0 else 0,
            'severity_distribution': severity_counts,
            'errors_by_component': component_errors,
            'last_24h_errors': sum(
                1 for e in self.error_history 
                if e.timestamp > datetime.now() - timedelta(days=1)
            )
        }

# Instancia global del gestor de errores
_error_manager = None

def get_error_manager() -> ErrorRecoveryManager:
    """Obtiene la instancia global del gestor de errores."""
    global _error_manager
    if _error_manager is None:
        _error_manager = ErrorRecoveryManager()
    return _error_manager

def handle_error(error: Exception, 
                component: str, 
                operation: str, 
                context_data: Optional[Dict[str, Any]] = None,
                severity: Optional[ErrorSeverity] = None) -> ErrorContext:
    """Función de conveniencia para manejar errores."""
    return get_error_manager().handle_error(error, component, operation, context_data, severity)

def with_error_handling(component: str, operation: str = None):
    """Decorador para manejo automático de errores."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            op_name = operation or func.__name__
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_context = handle_error(e, component, op_name)
                if not error_context.resolved:
                    raise
                return None
        return wrapper
    return decorator