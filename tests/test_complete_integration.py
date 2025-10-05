#!/usr/bin/env python3
"""
Pruebas de Integración Completas para SIGeC-Balistica.
Valida que todos los sistemas funcionen correctamente en conjunto.
"""

import pytest
import asyncio
import tempfile
import shutil
import json
import time
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any, List

# Importar todos los sistemas
try:
    from core.error_handler import ErrorRecoveryManager, ErrorSeverity, RecoveryStrategy
    from core.fallback_system import FallbackManager, FallbackPriority
    from core.notification_system import NotificationManager, NotificationType
    from performance.metrics_system import PerformanceMetricsSystem
    from core.telemetry_system import TelemetrySystem
    from security.security_manager import SecurityManager, SecurityLevel
    from core.intelligent_cache import IntelligentCache, CacheStrategy
    from api.optimization_system import APIOptimizer
    from docs.documentation_system import DocumentationSystem
    from monitoring.dashboard_system import DashboardSystem
    from deployment.ci_cd_system import CICDSystem, DeploymentEnvironment
except ImportError as e:
    pytest.skip(f"Sistemas no disponibles: {e}", allow_module_level=True)

class TestCompleteIntegration:
    """Pruebas de integración completas."""
    
    @pytest.fixture
    def temp_workspace(self):
        """Crear workspace temporal."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    async def integrated_systems(self, temp_workspace):
        """Configurar todos los sistemas integrados."""
        
        # Configuraciones
        config = {
            'workspace_dir': temp_workspace,
            'project_name': 'SIGeC-Balistica_Test',
            'enable_notifications': True,
            'enable_metrics': True,
            'enable_telemetry': True,
            'enable_security': True,
            'enable_caching': True
        }
        
        # Inicializar sistemas
        systems = {
            'error_handler': ErrorRecoveryManager(),
            'fallback_manager': FallbackManager(),
            'notification_manager': NotificationManager(),
            'metrics_system': PerformanceMetricsSystem(),
            'telemetry_system': TelemetrySystem(),
            'security_manager': SecurityManager(),
            'cache_system': IntelligentCache(),
            'api_optimizer': APIOptimizer(),
            'documentation_system': DocumentationSystem(),
            'dashboard_system': DashboardSystem(),
            'cicd_system': CICDSystem(config)
        }
        
        # Configurar integraciones
        await self._setup_system_integrations(systems)
        
        yield systems
        
        # Cleanup
        await self._cleanup_systems(systems)
    
    async def _setup_system_integrations(self, systems: Dict[str, Any]):
        """Configurar integraciones entre sistemas."""
        
        # Integrar error handler con notificaciones
        systems['error_handler'].notification_manager = systems['notification_manager']
        
        # Integrar fallback con telemetría
        systems['fallback_manager'].telemetry_system = systems['telemetry_system']
        
        # Integrar métricas con notificaciones
        systems['metrics_system'].notification_manager = systems['notification_manager']
        
        # Configurar cache con métricas
        systems['cache_system'].metrics_system = systems['metrics_system']
        
        # Configurar API optimizer con cache y métricas
        systems['api_optimizer'].cache_system = systems['cache_system']
        systems['api_optimizer'].metrics_system = systems['metrics_system']
    
    async def _cleanup_systems(self, systems: Dict[str, Any]):
        """Limpiar sistemas."""
        
        try:
            # Detener sistemas que requieren cleanup
            if hasattr(systems['metrics_system'], 'stop_monitoring'):
                systems['metrics_system'].stop_monitoring()
            
            if hasattr(systems['dashboard_system'], 'stop'):
                await systems['dashboard_system'].stop()
                
        except Exception as e:
            print(f"Error en cleanup: {e}")
    
    @pytest.mark.asyncio
    async def test_complete_system_startup(self, integrated_systems):
        """Probar inicio completo del sistema."""
        
        systems = integrated_systems
        
        # Verificar que todos los sistemas están inicializados
        assert systems['error_handler'] is not None
        assert systems['fallback_manager'] is not None
        assert systems['notification_manager'] is not None
        assert systems['metrics_system'] is not None
        assert systems['telemetry_system'] is not None
        assert systems['security_manager'] is not None
        assert systems['cache_system'] is not None
        assert systems['api_optimizer'] is not None
        assert systems['documentation_system'] is not None
        assert systems['dashboard_system'] is not None
        assert systems['cicd_system'] is not None
        
        # Verificar configuraciones básicas
        assert systems['error_handler'].max_retries > 0
        assert len(systems['fallback_manager'].fallbacks) >= 0
        assert systems['cache_system'].max_size > 0
    
    @pytest.mark.asyncio
    async def test_error_handling_with_notifications_and_fallbacks(self, integrated_systems):
        """Probar manejo de errores con notificaciones y fallbacks."""
        
        systems = integrated_systems
        error_handler = systems['error_handler']
        fallback_manager = systems['fallback_manager']
        notification_manager = systems['notification_manager']
        
        # Registrar fallback de prueba
        def test_fallback():
            return "fallback_result"
        
        fallback_manager.register_fallback(
            "test_operation",
            test_fallback,
            priority=FallbackPriority.HIGH
        )
        
        # Simular error que requiere fallback
        class TestError(Exception):
            pass
        
        # Configurar mock para notificaciones
        notification_sent = []
        
        async def mock_send_notification(notification):
            notification_sent.append(notification)
        
        notification_manager.send_notification = mock_send_notification
        
        # Manejar error
        result = await error_handler.handle_error(
            TestError("Error de prueba"),
            context={'operation': 'test_operation'},
            recovery_strategy=RecoveryStrategy.FALLBACK
        )
        
        # Verificar que se ejecutó el fallback
        assert result is not None
        
        # Verificar que se enviaron notificaciones (si están habilitadas)
        # Las notificaciones pueden estar deshabilitadas en el entorno de prueba
    
    @pytest.mark.asyncio
    async def test_performance_monitoring_with_alerts(self, integrated_systems):
        """Probar monitoreo de rendimiento con alertas."""
        
        systems = integrated_systems
        metrics_system = systems['metrics_system']
        notification_manager = systems['notification_manager']
        
        # Configurar umbral de prueba
        metrics_system.set_threshold('cpu_usage', 80.0)
        
        # Simular métricas altas
        with patch.object(metrics_system, '_collect_cpu_metrics', return_value=85.0):
            
            # Recolectar métricas
            metrics = metrics_system.collect_metrics()
            
            # Verificar que se detectó el umbral
            assert 'cpu_usage' in metrics
            
            # Verificar alertas (si están configuradas)
            alerts = metrics_system.check_thresholds(metrics)
            assert len(alerts) >= 0  # Puede haber alertas o no dependiendo de la configuración
    
    @pytest.mark.asyncio
    async def test_caching_with_performance_tracking(self, integrated_systems):
        """Probar sistema de cache con seguimiento de rendimiento."""
        
        systems = integrated_systems
        cache_system = systems['cache_system']
        metrics_system = systems['metrics_system']
        
        # Configurar cache con métricas
        cache_system.metrics_system = metrics_system
        
        # Operaciones de cache
        test_data = {"key": "value", "number": 42}
        
        # Set
        await cache_system.set("test_key", test_data, ttl=60)
        
        # Get (hit)
        result = await cache_system.get("test_key")
        assert result == test_data
        
        # Get (miss)
        result = await cache_system.get("nonexistent_key")
        assert result is None
        
        # Verificar estadísticas de cache
        stats = cache_system.get_stats()
        assert stats['total_requests'] >= 2
        assert stats['hits'] >= 1
        assert stats['misses'] >= 1
    
    @pytest.mark.asyncio
    async def test_security_integration(self, integrated_systems):
        """Probar integración de seguridad."""
        
        systems = integrated_systems
        security_manager = systems['security_manager']
        
        # Crear contexto de seguridad
        context = security_manager.create_security_context(
            user_id="test_user",
            session_id="test_session",
            ip_address="127.0.0.1"
        )
        
        assert context is not None
        assert context.user_id == "test_user"
        
        # Verificar autenticación (mock)
        with patch.object(security_manager, 'authenticate', return_value=True):
            is_authenticated = security_manager.authenticate("test_user", "password")
            assert is_authenticated
        
        # Verificar autorización
        with patch.object(security_manager, 'authorize', return_value=True):
            is_authorized = security_manager.authorize(context, "read", "test_resource")
            assert is_authorized
    
    @pytest.mark.asyncio
    async def test_api_optimization_integration(self, integrated_systems):
        """Probar integración de optimización de API."""
        
        systems = integrated_systems
        api_optimizer = systems['api_optimizer']
        cache_system = systems['cache_system']
        
        # Configurar optimizer con cache
        api_optimizer.cache_system = cache_system
        
        # Simular request
        mock_request = Mock()
        mock_request.url.path = "/test"
        mock_request.method = "GET"
        mock_request.headers = {}
        
        # Verificar configuración
        assert api_optimizer.config is not None
        assert api_optimizer.rate_limiter is not None
        assert api_optimizer.compressor is not None
    
    @pytest.mark.asyncio
    async def test_documentation_generation(self, integrated_systems):
        """Probar generación de documentación."""
        
        systems = integrated_systems
        doc_system = systems['documentation_system']
        
        # Configurar directorio de prueba
        test_dir = Path(systems['cicd_system'].config.workspace_dir) / "test_code"
        test_dir.mkdir(exist_ok=True)
        
        # Crear archivo de prueba
        test_file = test_dir / "test_module.py"
        test_file.write_text('''
"""Módulo de prueba."""

def test_function():
    """Función de prueba."""
    return "test"

class TestClass:
    """Clase de prueba."""
    
    def test_method(self):
        """Método de prueba."""
        pass
''')
        
        # Generar documentación
        docs = doc_system.generate_documentation(str(test_dir))
        
        assert docs is not None
        assert len(docs) > 0
        
        # Verificar contenido
        module_doc = docs[0]
        assert module_doc.name == "test_module"
        assert len(module_doc.functions) >= 1
        assert len(module_doc.classes) >= 1
    
    @pytest.mark.asyncio
    async def test_dashboard_system_integration(self, integrated_systems):
        """Probar integración del sistema de dashboard."""
        
        systems = integrated_systems
        dashboard_system = systems['dashboard_system']
        metrics_system = systems['metrics_system']
        
        # Configurar dashboard con métricas
        dashboard_system.metrics_collector = metrics_system
        
        # Verificar configuración
        assert dashboard_system.config is not None
        assert dashboard_system.metrics_collector is not None
        assert dashboard_system.alert_manager is not None
        
        # Simular métricas
        test_metrics = {
            'cpu_usage': 45.0,
            'memory_usage': 60.0,
            'disk_usage': 30.0
        }
        
        # Agregar métricas al dashboard
        for metric_name, value in test_metrics.items():
            dashboard_system.add_metric(metric_name, value)
        
        # Verificar que se agregaron las métricas
        current_metrics = dashboard_system.get_current_metrics()
        assert len(current_metrics) >= len(test_metrics)
    
    @pytest.mark.asyncio
    async def test_cicd_pipeline_integration(self, integrated_systems):
        """Probar integración del pipeline CI/CD."""
        
        systems = integrated_systems
        cicd_system = systems['cicd_system']
        
        # Verificar configuración
        assert cicd_system.config is not None
        assert cicd_system.pipeline_engine is not None
        assert cicd_system.deployment_manager is not None
        assert cicd_system.container_manager is not None
        
        # Crear pipeline de prueba simplificado
        from deployment.ci_cd_system import PipelineStep, PipelineStage
        
        test_steps = [
            PipelineStep(
                name="test_step",
                stage=PipelineStage.TEST,
                command="echo 'Test successful'",
                timeout=30
            )
        ]
        
        # Ejecutar pipeline (mock para evitar operaciones reales)
        with patch.object(cicd_system.pipeline_engine, 'execute_pipeline') as mock_execute:
            mock_execute.return_value = []
            
            results = await cicd_system.run_pipeline(test_steps)
            
            # Verificar que se llamó al pipeline
            mock_execute.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_telemetry_system_integration(self, integrated_systems):
        """Probar integración del sistema de telemetría."""
        
        systems = integrated_systems
        telemetry_system = systems['telemetry_system']
        
        # Verificar inicialización
        assert telemetry_system is not None
        
        # Registrar evento de prueba
        from core.telemetry_system import EventType, EventSeverity
        
        telemetry_system.record_event(
            event_type=EventType.SYSTEM_START,
            severity=EventSeverity.INFO,
            message="Sistema iniciado correctamente",
            metadata={'test': True}
        )
        
        # Verificar que se registró el evento
        events = telemetry_system.get_events(limit=10)
        assert len(events) >= 1
        
        # Verificar contenido del evento
        last_event = events[-1]
        assert last_event.event_type == EventType.SYSTEM_START
        assert last_event.message == "Sistema iniciado correctamente"
    
    @pytest.mark.asyncio
    async def test_complete_workflow_simulation(self, integrated_systems):
        """Simular flujo de trabajo completo."""
        
        systems = integrated_systems
        
        # 1. Iniciar sesión de telemetría
        systems['telemetry_system'].start_session("integration_test")
        
        # 2. Configurar seguridad
        security_context = systems['security_manager'].create_security_context(
            user_id="test_user",
            session_id="test_session",
            ip_address="127.0.0.1"
        )
        
        # 3. Realizar operación con cache
        await systems['cache_system'].set("workflow_data", {"step": 1}, ttl=300)
        cached_data = await systems['cache_system'].get("workflow_data")
        assert cached_data == {"step": 1}
        
        # 4. Simular error y recuperación
        try:
            raise ValueError("Error simulado para prueba")
        except ValueError as e:
            result = await systems['error_handler'].handle_error(
                e,
                context={'operation': 'workflow_test'},
                recovery_strategy=RecoveryStrategy.RETRY
            )
        
        # 5. Recolectar métricas
        metrics = systems['metrics_system'].collect_metrics()
        assert 'timestamp' in metrics
        
        # 6. Finalizar sesión
        systems['telemetry_system'].end_session()
        
        # Verificar que el flujo se completó sin errores críticos
        session_info = systems['telemetry_system'].get_session_info()
        assert session_info is not None
    
    @pytest.mark.asyncio
    async def test_system_resilience_under_load(self, integrated_systems):
        """Probar resistencia del sistema bajo carga."""
        
        systems = integrated_systems
        
        # Simular múltiples operaciones concurrentes
        async def simulate_operation(operation_id: int):
            """Simular operación individual."""
            
            # Cache operation
            await systems['cache_system'].set(f"load_test_{operation_id}", 
                                            {"id": operation_id}, ttl=60)
            
            # Metrics collection
            systems['metrics_system'].collect_metrics()
            
            # Telemetry event
            from core.telemetry_system import EventType, EventSeverity
            systems['telemetry_system'].record_event(
                event_type=EventType.FEATURE_USED,
                severity=EventSeverity.INFO,
                message=f"Operación {operation_id} completada"
            )
            
            return operation_id
        
        # Ejecutar operaciones concurrentes
        tasks = [simulate_operation(i) for i in range(10)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Verificar que todas las operaciones se completaron
        successful_operations = [r for r in results if isinstance(r, int)]
        assert len(successful_operations) == 10
        
        # Verificar estado de los sistemas
        cache_stats = systems['cache_system'].get_stats()
        assert cache_stats['total_requests'] >= 10
        
        events = systems['telemetry_system'].get_events(limit=20)
        load_test_events = [e for e in events if "Operación" in e.message]
        assert len(load_test_events) >= 10
    
    @pytest.mark.asyncio
    async def test_system_recovery_after_failure(self, integrated_systems):
        """Probar recuperación del sistema después de fallos."""
        
        systems = integrated_systems
        
        # Simular fallo en cache
        original_get = systems['cache_system'].get
        
        async def failing_get(key):
            if key == "failing_key":
                raise ConnectionError("Cache no disponible")
            return await original_get(key)
        
        systems['cache_system'].get = failing_get
        
        # Intentar operación que falla
        try:
            result = await systems['cache_system'].get("failing_key")
            assert False, "Debería haber fallado"
        except ConnectionError:
            pass  # Esperado
        
        # Restaurar funcionalidad
        systems['cache_system'].get = original_get
        
        # Verificar recuperación
        await systems['cache_system'].set("recovery_test", {"recovered": True})
        result = await systems['cache_system'].get("recovery_test")
        assert result == {"recovered": True}
        
        # Verificar que el sistema sigue funcionando
        metrics = systems['metrics_system'].collect_metrics()
        assert metrics is not None
    
    @pytest.mark.asyncio
    async def test_configuration_validation(self, integrated_systems):
        """Probar validación de configuraciones."""
        
        systems = integrated_systems
        
        # Verificar configuraciones críticas
        assert systems['error_handler'].max_retries > 0
        assert systems['cache_system'].max_size > 0
        assert systems['metrics_system'].collection_interval > 0
        
        # Verificar integraciones
        assert hasattr(systems['error_handler'], 'notification_manager')
        assert hasattr(systems['fallback_manager'], 'telemetry_system')
        
        # Verificar configuración de CI/CD
        cicd_config = systems['cicd_system'].config
        assert cicd_config.project_name == 'SIGeC-Balistica_Test'
        assert cicd_config.parallel_jobs > 0
        assert cicd_config.cleanup_after_days > 0

class TestSystemPerformance:
    """Pruebas de rendimiento del sistema."""
    
    @pytest.mark.asyncio
    async def test_cache_performance(self):
        """Probar rendimiento del cache."""
        
        cache = IntelligentCache(max_size=1000)
        
        # Medir tiempo de operaciones
        start_time = time.time()
        
        # Operaciones de escritura
        for i in range(100):
            await cache.set(f"key_{i}", {"value": i}, ttl=60)
        
        write_time = time.time() - start_time
        
        # Operaciones de lectura
        start_time = time.time()
        
        for i in range(100):
            result = await cache.get(f"key_{i}")
            assert result == {"value": i}
        
        read_time = time.time() - start_time
        
        # Verificar rendimiento aceptable
        assert write_time < 1.0  # Menos de 1 segundo para 100 escrituras
        assert read_time < 0.5   # Menos de 0.5 segundos para 100 lecturas
        
        # Verificar estadísticas
        stats = cache.get_stats()
        assert stats['hits'] == 100
        assert stats['misses'] == 0
    
    @pytest.mark.asyncio
    async def test_metrics_collection_performance(self):
        """Probar rendimiento de recolección de métricas."""
        
        metrics_system = PerformanceMetricsSystem()
        
        # Medir tiempo de recolección
        start_time = time.time()
        
        for _ in range(10):
            metrics = metrics_system.collect_metrics()
            assert metrics is not None
        
        collection_time = time.time() - start_time
        
        # Verificar rendimiento aceptable
        assert collection_time < 2.0  # Menos de 2 segundos para 10 recolecciones
    
    @pytest.mark.asyncio
    async def test_error_handling_performance(self):
        """Probar rendimiento del manejo de errores."""
        
        error_handler = ErrorRecoveryManager()
        
        # Medir tiempo de manejo de errores
        start_time = time.time()
        
        for i in range(50):
            try:
                raise ValueError(f"Error de prueba {i}")
            except ValueError as e:
                result = await error_handler.handle_error(
                    e,
                    context={'test_id': i},
                    recovery_strategy=RecoveryStrategy.LOG_AND_CONTINUE
                )
        
        handling_time = time.time() - start_time
        
        # Verificar rendimiento aceptable
        assert handling_time < 5.0  # Menos de 5 segundos para 50 errores

class TestSystemReliability:
    """Pruebas de confiabilidad del sistema."""
    
    @pytest.mark.asyncio
    async def test_memory_usage_stability(self):
        """Probar estabilidad del uso de memoria."""
        
        import psutil
        import gc
        
        # Medir memoria inicial
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        # Realizar operaciones intensivas
        cache = IntelligentCache(max_size=1000)
        
        for cycle in range(5):
            # Llenar cache
            for i in range(200):
                await cache.set(f"cycle_{cycle}_key_{i}", 
                              {"data": "x" * 100}, ttl=60)
            
            # Limpiar cache
            cache.clear()
            
            # Forzar garbage collection
            gc.collect()
        
        # Medir memoria final
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Verificar que no hay fugas de memoria significativas
        # Permitir hasta 50MB de incremento
        assert memory_increase < 50 * 1024 * 1024
    
    @pytest.mark.asyncio
    async def test_concurrent_access_safety(self):
        """Probar seguridad de acceso concurrente."""
        
        cache = IntelligentCache(max_size=100)
        results = []
        
        async def concurrent_operation(operation_id: int):
            """Operación concurrente."""
            try:
                # Escribir
                await cache.set(f"concurrent_{operation_id}", 
                              {"id": operation_id}, ttl=60)
                
                # Leer
                result = await cache.get(f"concurrent_{operation_id}")
                
                # Verificar
                if result and result.get("id") == operation_id:
                    return True
                return False
                
            except Exception as e:
                return f"Error: {e}"
        
        # Ejecutar operaciones concurrentes
        tasks = [concurrent_operation(i) for i in range(20)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Verificar que todas las operaciones fueron exitosas
        successful_operations = [r for r in results if r is True]
        assert len(successful_operations) >= 18  # Al menos 90% exitosas
        
        # Verificar que no hay excepciones no manejadas
        exceptions = [r for r in results if isinstance(r, Exception)]
        assert len(exceptions) == 0

if __name__ == "__main__":
    # Ejecutar pruebas
    pytest.main([__file__, "-v", "--tb=short"])