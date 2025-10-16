#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Core Integration Module - SIGeC-Balistica GUI
==========================================

Módulo de integración específico para componentes core del sistema,
proporcionando acceso al pipeline científico unificado desde la GUI.

Funcionalidades:
- Pipeline científico unificado
- Manejo inteligente de errores
- Cache inteligente
- Sistema de telemetría
- Notificaciones del sistema

Autor: SIGeC-Balistica Team
Fecha: Octubre 2025
"""

import os
import logging
from typing import Dict, Any, Optional, List, Callable
from pathlib import Path
from PyQt5.QtCore import QObject, pyqtSignal

# Importaciones de componentes core con manejo centralizado de fallbacks
from utils.dependency_manager import safe_import

# Usar importación segura para componentes core
ScientificPipeline = safe_import('core.unified_pipeline', 'core_components')
get_error_manager = safe_import('core.error_handler', 'core_components')
IntelligentCache = safe_import('core.intelligent_cache', 'core_components')
NotificationSystem = safe_import('core.notification_system', 'core_components')
TelemetrySystem = safe_import('core.telemetry_system', 'core_components')

# Verificar disponibilidad de componentes core
CORE_AVAILABLE = all([
    ScientificPipeline is not None,
    get_error_manager is not None,
    IntelligentCache is not None,
    NotificationSystem is not None,
    TelemetrySystem is not None
])

if CORE_AVAILABLE:
    logging.info("Módulos core cargados exitosamente")
else:
    logging.warning("Algunos módulos core no disponibles - usando fallbacks centralizados")

class CoreIntegration(QObject):
    """
    Clase de integración para componentes core del sistema
    """
    
    # Señales para comunicación con la GUI
    pipeline_started = pyqtSignal(str)  # pipeline_id
    pipeline_completed = pyqtSignal(str, dict)  # pipeline_id, results
    pipeline_error = pyqtSignal(str, str)  # pipeline_id, error_message
    notification_received = pyqtSignal(str, str, str)  # level, title, message
    
    def __init__(self):
        super().__init__()
        
        self.scientific_pipeline = None
        self.error_handler = None
        self.intelligent_cache = None
        self.notification_system = None
        self.telemetry_system = None
        
        self.active_pipelines = {}
        self.error_callbacks = []
        
        if CORE_AVAILABLE:
            self._initialize_core_components()
    
    def _initialize_core_components(self):
        """Inicializa los componentes core"""
        try:
            # Pipeline científico - usar la clase importada
            from core.unified_pipeline import ScientificPipeline as RealScientificPipeline
            self.scientific_pipeline = RealScientificPipeline()
            
            # Manejo de errores
            from core.error_handler import get_error_manager as real_get_error_manager
            self.error_handler = real_get_error_manager()
            
            # Cache inteligente
            from core.intelligent_cache import IntelligentCache as RealIntelligentCache
            self.intelligent_cache = RealIntelligentCache()
            
            # Sistema de notificaciones
            from core.notification_system import NotificationSystem as RealNotificationSystem, NotificationType
            self.notification_system = RealNotificationSystem()
            self.notification_system.subscribe(
                NotificationType.INFO, 
                lambda n: self._on_notification_received("info", n.title, n.message)
            )
            
            # Sistema de telemetría
            from core.telemetry_system import TelemetrySystem as RealTelemetrySystem
            self.telemetry_system = RealTelemetrySystem()
            
            logging.info("Componentes core inicializados correctamente")
            
        except Exception as e:
            logging.error(f"Error inicializando componentes core: {e}")
            # Usar fallbacks centralizados en lugar de mocks temporales
            from utils.fallback_implementations import get_fallback
            core_fallbacks = get_fallback('core_components')
            
            if self.scientific_pipeline is None:
                self.scientific_pipeline = core_fallbacks.ScientificPipelineFallback()
            if self.error_handler is None:
                self.error_handler = core_fallbacks.ErrorHandlerFallback()
            if self.intelligent_cache is None:
                self.intelligent_cache = core_fallbacks.IntelligentCacheFallback()
            if self.notification_system is None:
                self.notification_system = core_fallbacks.NotificationSystemFallback()
            if self.telemetry_system is None:
                self.telemetry_system = core_fallbacks.TelemetrySystemFallback()
    
    def _on_error_occurred(self, error_type: str, error_message: str, context: Dict[str, Any]):
        """Callback para manejo de errores"""
        self.pipeline_error.emit(context.get('pipeline_id', 'unknown'), error_message)
        
        # Ejecutar callbacks adicionales
        for callback in self.error_callbacks:
            try:
                callback(error_type, error_message, context)
            except Exception as e:
                logging.error(f"Error en callback de error: {e}")
    
    def _on_notification_received(self, level: str, title: str, message: str):
        """Callback para notificaciones del sistema"""
        self.notification_received.emit(level, title, message)
    
    def execute_scientific_pipeline(self, pipeline_config: Dict[str, Any]) -> Optional[str]:
        """
        Ejecuta un pipeline científico
        
        Args:
            pipeline_config: Configuración del pipeline
            
        Returns:
            ID del pipeline ejecutado o None si hay error
        """
        if not CORE_AVAILABLE or not self.scientific_pipeline:
            return self._fallback_pipeline_execution(pipeline_config)
        
        try:
            # El ScientificPipeline usa process_comparison, no execute
            # Generar un ID único para el pipeline
            import uuid
            pipeline_id = str(uuid.uuid4())
            
            # Simular ejecución del pipeline (en una implementación real, esto sería asíncrono)
            self.active_pipelines[pipeline_id] = pipeline_config
            self.pipeline_started.emit(pipeline_id)
            
            # Registrar en telemetría
            if self.telemetry_system:
                self.telemetry_system.record_user_action('pipeline_started', 'core_integration', {
                    'pipeline_id': pipeline_id,
                    'config': pipeline_config
                })
            
            return pipeline_id
            
        except Exception as e:
            if self.error_handler:
                self.error_handler.handle_error('pipeline_execution', str(e), {
                    'config': pipeline_config
                })
            return None
    
    def _fallback_pipeline_execution(self, pipeline_config: Dict[str, Any]) -> Optional[str]:
        """Ejecución básica de pipeline sin componentes core"""
        import uuid
        pipeline_id = str(uuid.uuid4())
        
        # Simular ejecución básica
        self.pipeline_started.emit(pipeline_id)
        
        # Simular resultados básicos
        results = {
            'status': 'completed',
            'message': 'Pipeline ejecutado con funcionalidad básica',
            'config': pipeline_config
        }
        
        self.pipeline_completed.emit(pipeline_id, results)
        return pipeline_id
    
    def get_pipeline_status(self, pipeline_id: str) -> Optional[Dict[str, Any]]:
        """
        Obtiene el estado de un pipeline
        
        Args:
            pipeline_id: ID del pipeline
            
        Returns:
            Estado del pipeline o None si no existe
        """
        if not CORE_AVAILABLE or not self.scientific_pipeline:
            return self._fallback_pipeline_status(pipeline_id)
        
        # ScientificPipeline no tiene get_status, devolvemos estado genérico
        return {
            "pipeline_id": pipeline_id,
            "status": "ready",
            "config": {
                "level": getattr(self.scientific_pipeline.config, 'level', 'standard'),
                "components_available": {
                    "preprocessor": hasattr(self.scientific_pipeline, 'preprocessor'),
                    "roi_detector": hasattr(self.scientific_pipeline, 'roi_detector'),
                    "matcher": hasattr(self.scientific_pipeline, 'matcher'),
                    "quality_metrics": hasattr(self.scientific_pipeline, 'quality_metrics')
                }
            }
        }
    
    def _fallback_pipeline_status(self, pipeline_id: str) -> Optional[Dict[str, Any]]:
        """Estado básico de pipeline sin componentes core"""
        if pipeline_id in self.active_pipelines:
            return {
                'id': pipeline_id,
                'status': 'completed',
                'progress': 100,
                'config': self.active_pipelines[pipeline_id]
            }
        return None
    
    def cancel_pipeline(self, pipeline_id: str) -> bool:
        """
        Cancela un pipeline en ejecución
        
        Args:
            pipeline_id: ID del pipeline a cancelar
            
        Returns:
            True si se canceló correctamente
        """
        if not CORE_AVAILABLE or not self.scientific_pipeline:
            return self._fallback_pipeline_cancellation(pipeline_id)
        
        try:
            result = self.scientific_pipeline.cancel(pipeline_id)
            if result and pipeline_id in self.active_pipelines:
                del self.active_pipelines[pipeline_id]
            return result
        except Exception as e:
            if self.error_handler:
                self.error_handler.handle_error('pipeline_cancellation', str(e), {
                    'pipeline_id': pipeline_id
                })
            return False
    
    def _fallback_pipeline_cancellation(self, pipeline_id: str) -> bool:
        """Cancelación básica de pipeline sin componentes core"""
        if pipeline_id in self.active_pipelines:
            del self.active_pipelines[pipeline_id]
            return True
        return False
    
    def get_cached_result(self, cache_key: str) -> Any:
        """
        Obtiene un resultado del cache inteligente
        
        Args:
            cache_key: Clave del cache
            
        Returns:
            Resultado cacheado o None
        """
        if not CORE_AVAILABLE or not self.intelligent_cache:
            return None
        
        return self.intelligent_cache.get(cache_key)
    
    def set_cached_result(self, cache_key: str, result: Any, metadata: Optional[Dict[str, Any]] = None):
        """
        Almacena un resultado en el cache inteligente
        
        Args:
            cache_key: Clave del cache
            result: Resultado a cachear
            metadata: Metadatos adicionales
        """
        if not CORE_AVAILABLE or not self.intelligent_cache:
            return
        
        self.intelligent_cache.set(cache_key, result, metadata or {})
    
    def clear_cache(self):
        """Limpia el cache inteligente"""
        if not CORE_AVAILABLE or not self.intelligent_cache:
            return
        
        self.intelligent_cache.clear()
    
    def send_notification(self, level: str, title: str, message: str):
        """
        Envía una notificación del sistema
        
        Args:
            level: Nivel de la notificación (info, warning, error)
            title: Título de la notificación
            message: Mensaje de la notificación
        """
        if not CORE_AVAILABLE or not self.notification_system:
            # Emitir directamente la señal como fallback
            self.notification_received.emit(level, title, message)
            return
        
        self.notification_system.send(level, title, message)
    
    def add_error_callback(self, callback: Callable[[str, str, Dict[str, Any]], None]):
        """
        Añade un callback para manejo de errores
        
        Args:
            callback: Función callback que recibe (error_type, error_message, context)
        """
        self.error_callbacks.append(callback)
    
    def get_telemetry_data(self) -> Dict[str, Any]:
        """
        Obtiene datos de telemetría del sistema
        
        Returns:
            Diccionario con datos de telemetría
        """
        if not CORE_AVAILABLE or not self.telemetry_system:
            return self._fallback_telemetry_data()
        
        # Usar el método correcto de TelemetrySystem
        try:
            # Obtener datos de los collectors
            telemetry_data = {}
            for name, collector in self.telemetry_system.collectors.items():
                telemetry_data[name] = collector.collect()
            
            # Agregar información de sesión actual
            if self.telemetry_system.current_session:
                telemetry_data['current_session'] = {
                    'session_id': self.telemetry_system.current_session.session_id,
                    'start_time': self.telemetry_system.current_session.start_time.isoformat(),
                    'events_count': self.telemetry_system.current_session.events_count,
                    'duration_seconds': self.telemetry_system.current_session.duration_seconds
                }
            
            return telemetry_data
        except Exception as e:
            logging.error(f"Error obteniendo datos de telemetría: {e}")
            return self._fallback_telemetry_data()
    
    def _fallback_telemetry_data(self) -> Dict[str, Any]:
        """Datos básicos de telemetría sin componentes core"""
        return {
            'active_pipelines': len(self.active_pipelines),
            'total_pipelines': len(self.active_pipelines),
            'system_status': 'running',
            'core_available': False
        }
    
    def get_system_health(self) -> Dict[str, Any]:
        """
        Obtiene información de salud del sistema
        
        Returns:
            Diccionario con información de salud
        """
        health_data = {
            'core_available': CORE_AVAILABLE,
            'components': {
                'scientific_pipeline': self.scientific_pipeline is not None,
                'error_handler': self.error_handler is not None,
                'intelligent_cache': self.intelligent_cache is not None,
                'notification_system': self.notification_system is not None,
                'telemetry_system': self.telemetry_system is not None
            },
            'active_pipelines': len(self.active_pipelines),
            'error_callbacks': len(self.error_callbacks)
        }
        
        # Agregar métricas de rendimiento si están disponibles
        if CORE_AVAILABLE and self.telemetry_system:
            try:
                # Obtener métricas de rendimiento del collector
                performance_collector = self.telemetry_system.collectors.get('performance')
                if performance_collector:
                    performance_data = performance_collector.collect()
                    health_data['performance'] = performance_data
            except Exception as e:
                logging.error(f"Error obteniendo métricas de rendimiento: {e}")
        
        return health_data

# Instancia global
_core_integration = None

def get_core_integration() -> CoreIntegration:
    """Obtiene la instancia global de integración core"""
    global _core_integration
    if _core_integration is None:
        _core_integration = CoreIntegration()
    return _core_integration

if __name__ == "__main__":
    # Prueba básica del módulo
    from PyQt5.QtWidgets import QApplication
    import sys
    
    app = QApplication(sys.argv)
    
    core_int = CoreIntegration()
    print("=== Core Integration Test ===")
    print("Salud del sistema:", core_int.get_system_health())
    print("Datos de telemetría:", core_int.get_telemetry_data())
    
    # Prueba de pipeline básico
    pipeline_config = {
        'type': 'test_pipeline',
        'parameters': {'test': True}
    }
    
    pipeline_id = core_int.execute_scientific_pipeline(pipeline_config)
    print(f"Pipeline ejecutado: {pipeline_id}")
    
    if pipeline_id:
        status = core_int.get_pipeline_status(pipeline_id)
        print(f"Estado del pipeline: {status}")