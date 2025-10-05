#!/usr/bin/env python3
"""
Pruebas de integración para todos los sistemas de SEACABAr.
Valida que los componentes funcionen correctamente en conjunto.
"""

import unittest
import time
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import threading
from datetime import datetime, timedelta

# Importar sistemas a probar
try:
    from core.error_handler import ErrorRecoveryManager, ErrorSeverity, ErrorContext
    from core.fallback_system import FallbackManager, FallbackPriority
    from core.notification_system import (
        NotificationManager, NotificationType, NotificationChannel,
        get_notification_manager, notify_error, notify_critical
    )
    from performance.metrics_system import (
        PerformanceMetricsSystem, MetricThreshold,
        get_metrics_system, start_metrics_monitoring
    )
    from core.telemetry_system import (
        TelemetrySystem, EventType, EventSeverity,
        get_telemetry_system, record_user_action, measure_performance
    )
    SYSTEMS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: No se pudieron importar algunos sistemas: {e}")
    SYSTEMS_AVAILABLE = False

class TestSystemsIntegration(unittest.TestCase):
    """Pruebas de integración entre sistemas."""
    
    def setUp(self):
        """Configuración inicial para cada prueba."""
        if not SYSTEMS_AVAILABLE:
            self.skipTest("Sistemas no disponibles para pruebas")
        
        # Crear directorio temporal para pruebas
        self.test_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.test_dir)
        
        # Configurar sistemas con datos de prueba
        self.setup_test_systems()
    
    def setup_test_systems(self):
        """Configura sistemas para pruebas."""
        
        # Sistema de notificaciones
        self.notification_manager = NotificationManager()
        
        # Sistema de manejo de errores
        self.error_manager = ErrorRecoveryManager()
        
        # Sistema de fallbacks
        self.fallback_manager = FallbackManager()
        
        # Sistema de métricas
        metrics_config = {
            'collection_interval': 0.1,  # Más rápido para pruebas
            'max_history': 100,
            'enable_alerts': True
        }
        self.metrics_system = PerformanceMetricsSystem(metrics_config)
        
        # Sistema de telemetría
        telemetry_config = {
            'enabled': True,
            'data_dir': str(Path(self.test_dir) / 'telemetry'),
            'auto_flush_interval': 1,  # Más rápido para pruebas
            'privacy_mode': False  # Para pruebas más fáciles
        }
        self.telemetry_system = TelemetrySystem(telemetry_config)
    
    def test_error_handling_with_notifications(self):
        """Prueba integración entre manejo de errores y notificaciones."""
        
        # Configurar captura de notificaciones
        notifications_received = []
        
        def capture_notification(notification):
            notifications_received.append(notification)
        
        self.notification_manager.subscribe(NotificationType.ERROR, capture_notification)
        self.notification_manager.subscribe(NotificationType.CRITICAL, capture_notification)
        
        # Simular error crítico
        test_error = ValueError("Error de prueba crítico")
        context = ErrorContext(
            component="test_component",
            operation="test_operation",
            severity=ErrorSeverity.CRITICAL,
            context_data={"test": "data"}
        )
        
        # Manejar error (debería generar notificación)
        with patch('core.error_handler.NOTIFICATIONS_AVAILABLE', True):
            with patch('core.error_handler.notify_critical') as mock_notify:
                self.error_manager.handle_error(test_error, context)
                
                # Verificar que se llamó la notificación
                mock_notify.assert_called_once()
                call_args = mock_notify.call_args[0]
                self.assertIn("Error Crítico", call_args[0])
                self.assertIn("test_component", call_args[1])
    
    def test_fallback_system_with_telemetry(self):
        """Prueba integración entre sistema de fallbacks y telemetría."""
        
        # Registrar fallback de prueba
        def test_fallback():
            return "fallback_result"
        
        self.fallback_manager.register_fallback(
            "test_operation",
            test_fallback,
            FallbackPriority.HIGH,
            description="Fallback de prueba"
        )
        
        # Ejecutar fallback y verificar telemetría
        result = self.fallback_manager.execute_fallback("test_operation")
        
        self.assertEqual(result.success, True)
        self.assertEqual(result.result, "fallback_result")
        
        # Verificar que se registró en telemetría
        events = self.telemetry_system.get_events(
            event_type=EventType.SYSTEM_EVENT,
            component="fallback_system"
        )
        
        # Debería haber al menos un evento relacionado con fallbacks
        self.assertGreater(len(events), 0)
    
    def test_metrics_system_with_notifications(self):
        """Prueba integración entre sistema de métricas y notificaciones."""
        
        # Configurar umbral de prueba muy bajo
        threshold = MetricThreshold(
            warning_threshold=1.0,
            critical_threshold=2.0,
            comparison="greater"
        )
        self.metrics_system.set_threshold("test_metric", threshold)
        
        # Capturar notificaciones
        notifications_received = []
        
        def capture_notification(notification):
            notifications_received.append(notification)
        
        self.notification_manager.subscribe(NotificationType.WARNING, capture_notification)
        self.notification_manager.subscribe(NotificationType.CRITICAL, capture_notification)
        
        # Simular métrica que excede umbral
        with patch('performance.metrics_system.NOTIFICATIONS_AVAILABLE', True):
            with patch('performance.metrics_system.notify_warning') as mock_warn:
                with patch('performance.metrics_system.notify_critical') as mock_crit:
                    
                    # Simular valor que excede umbral crítico
                    self.metrics_system._check_single_threshold("test_metric", 5.0)
                    
                    # Verificar que se llamó notificación crítica
                    mock_crit.assert_called_once()
    
    def test_telemetry_performance_measurement(self):
        """Prueba medición automática de rendimiento con telemetría."""
        
        # Función de prueba con decorador de medición
        @measure_performance("test_component", "test_operation")
        def test_function():
            time.sleep(0.1)  # Simular trabajo
            return "test_result"
        
        # Ejecutar función
        result = test_function()
        self.assertEqual(result, "test_result")
        
        # Verificar que se registró evento de rendimiento
        events = self.telemetry_system.get_events(
            event_type=EventType.PERFORMANCE,
            component="test_component"
        )
        
        self.assertGreater(len(events), 0)
        
        # Verificar datos del evento
        perf_event = events[0]
        self.assertEqual(perf_event.action, "performance_test_operation")
        self.assertIsNotNone(perf_event.duration_ms)
        self.assertGreater(perf_event.duration_ms, 90)  # Al menos 90ms
    
    def test_error_recovery_with_fallbacks(self):
        """Prueba recuperación de errores usando sistema de fallbacks."""
        
        # Registrar fallback para operación específica
        def recovery_fallback():
            return "recovered_successfully"
        
        self.fallback_manager.register_fallback(
            "image_processing",
            recovery_fallback,
            FallbackPriority.HIGH
        )
        
        # Configurar estrategia de recuperación en error manager
        def custom_recovery_strategy(error, context):
            if context.component == "image_processing":
                # Usar fallback
                result = self.fallback_manager.execute_fallback("image_processing")
                return result.success
            return False
        
        # Simular error en procesamiento de imagen
        test_error = RuntimeError("Error en procesamiento")
        context = ErrorContext(
            component="image_processing",
            operation="process_image",
            severity=ErrorSeverity.HIGH
        )
        
        # Intentar recuperación
        with patch.object(self.error_manager, '_attempt_recovery', custom_recovery_strategy):
            recovered = self.error_manager.handle_error(test_error, context)
            
            # Verificar que se recuperó exitosamente
            self.assertTrue(recovered)
    
    def test_comprehensive_system_workflow(self):
        """Prueba flujo completo integrando todos los sistemas."""
        
        # 1. Iniciar monitoreo de métricas
        self.metrics_system.start_monitoring()
        
        # 2. Iniciar sesión de telemetría
        session_id = self.telemetry_system.start_session("test_user")
        self.assertIsNotNone(session_id)
        
        # 3. Registrar uso de característica
        feature_event_id = self.telemetry_system.record_feature_usage(
            "image_matching", 
            "matching_engine",
            metadata={"algorithm": "ORB", "threshold": 0.8}
        )
        self.assertIsNotNone(feature_event_id)
        
        # 4. Simular operación con posible error
        def risky_operation():
            # Simular fallo ocasional
            import random
            if random.random() < 0.3:  # 30% de probabilidad de fallo
                raise ConnectionError("Fallo de conexión simulado")
            return "operación_exitosa"
        
        # Registrar fallback para la operación
        def connection_fallback():
            return "conexión_recuperada_via_fallback"
        
        self.fallback_manager.register_fallback(
            "network_operation",
            connection_fallback,
            FallbackPriority.MEDIUM
        )
        
        # 5. Ejecutar operación múltiples veces
        results = []
        for i in range(10):
            try:
                result = risky_operation()
                results.append(result)
                
                # Registrar éxito en telemetría
                self.telemetry_system.record_performance_event(
                    "network_operation",
                    "network_component",
                    50.0,  # 50ms simulados
                    success=True
                )
                
            except Exception as e:
                # Manejar error
                context = ErrorContext(
                    component="network_component",
                    operation="network_operation",
                    severity=ErrorSeverity.MEDIUM
                )
                
                recovered = self.error_manager.handle_error(e, context)
                
                if not recovered:
                    # Usar fallback manualmente
                    fallback_result = self.fallback_manager.execute_fallback("network_operation")
                    if fallback_result.success:
                        results.append(fallback_result.result)
                
                # Registrar error en telemetría
                self.telemetry_system.record_error_event(
                    e, "network_component", "network_operation"
                )
        
        # 6. Verificar que se obtuvieron resultados
        self.assertGreater(len(results), 0)
        
        # 7. Obtener métricas actuales
        current_metrics = self.metrics_system.get_current_metrics()
        self.assertIsInstance(current_metrics, dict)
        
        # 8. Obtener analíticas de sesión
        session_analytics = self.telemetry_system.get_session_analytics()
        self.assertIn('session_info', session_analytics)
        self.assertGreater(session_analytics['total_events'], 0)
        
        # 9. Finalizar sesión
        self.telemetry_system.end_session()
        
        # 10. Detener monitoreo
        self.metrics_system.stop_monitoring()
        
        # Verificar que todo funcionó correctamente
        self.assertTrue(True)  # Si llegamos aquí, todo está bien
    
    def test_notification_system_channels(self):
        """Prueba diferentes canales de notificación."""
        
        # Configurar archivo temporal para notificaciones
        notifications_file = Path(self.test_dir) / "logs" / "notifications.json"
        
        # Enviar notificación que debería ir a múltiples canales
        notification_id = self.notification_manager.send_notification(
            NotificationType.CRITICAL,
            "Prueba de Canales",
            "Mensaje de prueba para múltiples canales",
            "test_component",
            channels=[NotificationChannel.LOG, NotificationChannel.FILE],
            metadata={"test": True}
        )
        
        self.assertIsNotNone(notification_id)
        
        # Verificar que se creó el archivo de notificaciones
        # (Puede tomar un momento)
        time.sleep(0.1)
        
        # Verificar que la notificación se almacenó
        notifications = self.notification_manager.get_notifications(
            component="test_component",
            limit=10
        )
        
        self.assertGreater(len(notifications), 0)
        self.assertEqual(notifications[0].title, "Prueba de Canales")
    
    def test_system_health_monitoring(self):
        """Prueba monitoreo integral de salud del sistema."""
        
        # Iniciar monitoreo
        self.metrics_system.start_monitoring()
        
        # Esperar a que se recopilen algunas métricas
        time.sleep(0.5)
        
        # Obtener puntaje de salud
        health_score = self.metrics_system.get_system_health_score()
        
        # Verificar que el puntaje está en rango válido
        self.assertGreaterEqual(health_score, 0.0)
        self.assertLessEqual(health_score, 100.0)
        
        # Obtener estadísticas de métricas
        cpu_stats = self.metrics_system.get_metric_statistics('cpu_usage')
        
        if cpu_stats:  # Si hay datos de CPU
            self.assertIn('mean', cpu_stats)
            self.assertIn('max', cpu_stats)
            self.assertIn('min', cpu_stats)
        
        # Detener monitoreo
        self.metrics_system.stop_monitoring()
    
    def tearDown(self):
        """Limpieza después de cada prueba."""
        
        if hasattr(self, 'metrics_system'):
            self.metrics_system.stop_monitoring()
        
        if hasattr(self, 'telemetry_system'):
            self.telemetry_system.end_session()

class TestSystemsPerformance(unittest.TestCase):
    """Pruebas de rendimiento de los sistemas."""
    
    def setUp(self):
        """Configuración para pruebas de rendimiento."""
        if not SYSTEMS_AVAILABLE:
            self.skipTest("Sistemas no disponibles para pruebas")
    
    def test_notification_system_performance(self):
        """Prueba rendimiento del sistema de notificaciones."""
        
        notification_manager = NotificationManager()
        
        # Medir tiempo de envío de múltiples notificaciones
        start_time = time.time()
        
        for i in range(100):
            notification_manager.send_notification(
                NotificationType.INFO,
                f"Notificación {i}",
                f"Mensaje de prueba {i}",
                "performance_test"
            )
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Verificar que se completó en tiempo razonable (< 1 segundo)
        self.assertLess(duration, 1.0)
        
        # Verificar que todas las notificaciones se almacenaron
        notifications = notification_manager.get_notifications(
            component="performance_test",
            limit=200
        )
        
        self.assertEqual(len(notifications), 100)
    
    def test_telemetry_system_performance(self):
        """Prueba rendimiento del sistema de telemetría."""
        
        telemetry_config = {
            'enabled': True,
            'data_dir': tempfile.mkdtemp(),
            'privacy_mode': False
        }
        
        telemetry_system = TelemetrySystem(telemetry_config)
        
        # Medir tiempo de registro de múltiples eventos
        start_time = time.time()
        
        for i in range(1000):
            telemetry_system.record_user_action(
                f"action_{i}",
                "performance_test",
                data={"iteration": i, "test": True}
            )
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Verificar que se completó en tiempo razonable (< 2 segundos)
        self.assertLess(duration, 2.0)
        
        # Verificar que todos los eventos se registraron
        events = telemetry_system.get_events(
            component="performance_test",
            limit=2000
        )
        
        self.assertEqual(len(events), 1000)
        
        # Limpiar
        shutil.rmtree(telemetry_config['data_dir'])

class TestSystemsReliability(unittest.TestCase):
    """Pruebas de confiabilidad y robustez de los sistemas."""
    
    def setUp(self):
        """Configuración para pruebas de confiabilidad."""
        if not SYSTEMS_AVAILABLE:
            self.skipTest("Sistemas no disponibles para pruebas")
    
    def test_concurrent_access(self):
        """Prueba acceso concurrente a los sistemas."""
        
        notification_manager = NotificationManager()
        results = []
        errors = []
        
        def worker_thread(thread_id):
            try:
                for i in range(50):
                    notification_id = notification_manager.send_notification(
                        NotificationType.INFO,
                        f"Thread {thread_id} - Notificación {i}",
                        f"Mensaje desde hilo {thread_id}",
                        f"thread_{thread_id}"
                    )
                    results.append(notification_id)
            except Exception as e:
                errors.append(e)
        
        # Crear múltiples hilos
        threads = []
        for i in range(10):
            thread = threading.Thread(target=worker_thread, args=(i,))
            threads.append(thread)
        
        # Iniciar todos los hilos
        for thread in threads:
            thread.start()
        
        # Esperar a que terminen
        for thread in threads:
            thread.join()
        
        # Verificar resultados
        self.assertEqual(len(errors), 0, f"Errores en hilos concurrentes: {errors}")
        self.assertEqual(len(results), 500)  # 10 hilos * 50 notificaciones
        
        # Verificar que todas las notificaciones se almacenaron
        all_notifications = notification_manager.get_notifications(limit=1000)
        self.assertGreaterEqual(len(all_notifications), 500)
    
    def test_system_recovery_after_failure(self):
        """Prueba recuperación de sistemas después de fallos."""
        
        # Simular fallo en sistema de métricas
        metrics_system = PerformanceMetricsSystem()
        
        # Iniciar monitoreo
        metrics_system.start_monitoring()
        self.assertTrue(metrics_system.running)
        
        # Simular fallo forzando detención
        metrics_system.stop_monitoring()
        self.assertFalse(metrics_system.running)
        
        # Verificar que se puede reiniciar
        metrics_system.start_monitoring()
        self.assertTrue(metrics_system.running)
        
        # Limpiar
        metrics_system.stop_monitoring()

if __name__ == '__main__':
    # Configurar logging para pruebas
    import logging
    logging.basicConfig(
        level=logging.WARNING,  # Reducir ruido en pruebas
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Ejecutar pruebas
    unittest.main(verbosity=2)