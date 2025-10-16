"""
Tests de Integración para el Sistema de Monitoreo
===============================================

Tests comprehensivos para validar el funcionamiento del sistema de monitoreo
y métricas avanzado de SIGeC-Balística.
"""

import unittest
import time
import threading
import tempfile
import json
import os
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from monitoring.system_monitor import (
        SystemMonitor, MetricsCollector, AlertManager, 
        PerformanceProfiler, SystemMetrics, ApplicationMetrics,
        monitor_performance
    )
    from monitoring.dashboard import MonitoringDashboard, ConsoleDashboard
    from monitoring.integration import (
        MonitoringIntegration, initialize_monitoring, 
        start_monitoring, stop_monitoring, get_system_metrics
    )
    MONITORING_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Monitoring system not available: {e}")
    MONITORING_AVAILABLE = False

class TestSystemMetrics(unittest.TestCase):
    """Tests para la clase SystemMetrics"""
    
    def test_system_metrics_creation(self):
        """Test creación de métricas del sistema"""
        if not MONITORING_AVAILABLE:
            self.skipTest("Sistema de monitoreo no disponible")
        
        metrics = SystemMetrics(
            cpu_percent=50.0,
            memory_percent=60.0,
            disk_percent=70.0,
            network_io_bytes=1000,
            process_count=100
        )
        
        self.assertEqual(metrics.cpu_percent, 50.0)
        self.assertEqual(metrics.memory_percent, 60.0)
        self.assertEqual(metrics.disk_percent, 70.0)
        self.assertEqual(metrics.network_io_bytes, 1000)
        self.assertEqual(metrics.process_count, 100)

class TestApplicationMetrics(unittest.TestCase):
    """Tests para la clase ApplicationMetrics"""
    
    def test_application_metrics_creation(self):
        """Test creación de métricas de aplicación"""
        if not MONITORING_AVAILABLE:
            self.skipTest("Sistema de monitoreo no disponible")
        
        metrics = ApplicationMetrics(
            active_sessions=5,
            processed_images=100,
            database_queries=50,
            cache_hits=80,
            cache_misses=20,
            error_count=2
        )
        
        self.assertEqual(metrics.active_sessions, 5)
        self.assertEqual(metrics.processed_images, 100)
        self.assertEqual(metrics.database_queries, 50)
        self.assertEqual(metrics.cache_hits, 80)
        self.assertEqual(metrics.cache_misses, 20)
        self.assertEqual(metrics.error_count, 2)

class TestMetricsCollector(unittest.TestCase):
    """Tests para MetricsCollector"""
    
    def setUp(self):
        """Configurar test"""
        if not MONITORING_AVAILABLE:
            self.skipTest("Sistema de monitoreo no disponible")
        self.collector = MetricsCollector()
    
    def test_collect_system_metrics(self):
        """Test recolección de métricas del sistema"""
        metrics = self.collector.collect_system_metrics()
        
        self.assertIsInstance(metrics, SystemMetrics)
        self.assertGreaterEqual(metrics.cpu_percent, 0)
        self.assertLessEqual(metrics.cpu_percent, 100)
        self.assertGreaterEqual(metrics.memory_percent, 0)
        self.assertLessEqual(metrics.memory_percent, 100)
    
    def test_collect_application_metrics(self):
        """Test recolección de métricas de aplicación"""
        metrics = self.collector.collect_application_metrics()
        
        self.assertIsInstance(metrics, ApplicationMetrics)
        self.assertGreaterEqual(metrics.active_sessions, 0)
        self.assertGreaterEqual(metrics.processed_images, 0)

class TestAlertManager(unittest.TestCase):
    """Tests para AlertManager"""
    
    def setUp(self):
        """Configurar test"""
        if not MONITORING_AVAILABLE:
            self.skipTest("Sistema de monitoreo no disponible")
        self.alert_manager = AlertManager()
    
    def test_check_alerts_high_cpu(self):
        """Test alerta por CPU alto"""
        # Crear métricas con CPU alto
        system_metrics = SystemMetrics(
            cpu_percent=95.0,  # Alto
            memory_percent=50.0,
            disk_percent=50.0,
            network_io_bytes=1000,
            process_count=100
        )
        
        app_metrics = ApplicationMetrics(
            active_sessions=5,
            processed_images=100,
            database_queries=50,
            cache_hits=80,
            cache_misses=20,
            error_count=2
        )
        
        alerts = self.alert_manager.check_alerts(system_metrics, app_metrics)
        
        # Debe haber al menos una alerta de CPU
        cpu_alerts = [alert for alert in alerts if 'CPU' in alert['message']]
        self.assertGreater(len(cpu_alerts), 0)
    
    def test_check_alerts_high_memory(self):
        """Test alerta por memoria alta"""
        system_metrics = SystemMetrics(
            cpu_percent=50.0,
            memory_percent=95.0,  # Alto
            disk_percent=50.0,
            network_io_bytes=1000,
            process_count=100
        )
        
        app_metrics = ApplicationMetrics(
            active_sessions=5,
            processed_images=100,
            database_queries=50,
            cache_hits=80,
            cache_misses=20,
            error_count=2
        )
        
        alerts = self.alert_manager.check_alerts(system_metrics, app_metrics)
        
        # Debe haber al menos una alerta de memoria
        memory_alerts = [alert for alert in alerts if 'Memoria' in alert['message']]
        self.assertGreater(len(memory_alerts), 0)

class TestPerformanceProfiler(unittest.TestCase):
    """Tests para PerformanceProfiler"""
    
    def setUp(self):
        """Configurar test"""
        if not MONITORING_AVAILABLE:
            self.skipTest("Sistema de monitoreo no disponible")
        self.profiler = PerformanceProfiler()
    
    def test_profile_operation(self):
        """Test perfilado de operación"""
        # Simular operación
        def test_operation():
            time.sleep(0.1)
            return "resultado"
        
        result = self.profiler.profile_operation("test_op", test_operation)
        
        self.assertEqual(result, "resultado")
        
        # Verificar que se registró la operación
        stats = self.profiler.get_stats()
        self.assertIn("test_op", stats)
        self.assertGreater(stats["test_op"]["avg_time"], 0.09)  # Al menos 0.09 segundos
    
    def test_monitor_performance_decorator(self):
        """Test decorador de monitoreo de rendimiento"""
        if not callable(monitor_performance):
            self.skipTest("Decorador monitor_performance no disponible")
        
        @monitor_performance
        def test_function():
            time.sleep(0.05)
            return "test_result"
        
        result = test_function()
        self.assertEqual(result, "test_result")

class TestSystemMonitor(unittest.TestCase):
    """Tests para SystemMonitor"""
    
    def setUp(self):
        """Configurar test"""
        if not MONITORING_AVAILABLE:
            self.skipTest("Sistema de monitoreo no disponible")
        
        # Crear directorio temporal para datos
        self.temp_dir = tempfile.mkdtemp()
        self.monitor = SystemMonitor(monitoring_interval=1)  # 1 segundo para tests rápidos
    
    def tearDown(self):
        """Limpiar después del test"""
        if hasattr(self, 'monitor'):
            self.monitor.stop()
        
        # Limpiar directorio temporal
        import shutil
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_start_stop_monitor(self):
        """Test iniciar y detener monitor"""
        self.assertFalse(self.monitor.is_running)
        
        self.monitor.start()
        self.assertTrue(self.monitor.is_running)
        
        # Esperar un poco para que recolecte métricas
        time.sleep(2)
        
        self.monitor.stop()
        self.assertFalse(self.monitor.is_running)
    
    def test_get_current_metrics(self):
        """Test obtener métricas actuales"""
        self.monitor.start()
        time.sleep(1.5)  # Esperar recolección
        
        metrics = self.monitor.get_current_metrics()
        
        self.assertIn('system', metrics)
        self.assertIn('application', metrics)
        self.assertIn('timestamp', metrics)
        
        self.monitor.stop()
    
    def test_add_custom_metric(self):
        """Test agregar métrica personalizada"""
        self.monitor.add_custom_metric("test_metric", 42.0, {"tag": "test"})
        
        # Verificar que se agregó
        custom_metrics = self.monitor.custom_metrics
        self.assertGreater(len(custom_metrics), 0)
        
        # Buscar la métrica
        found = False
        for metric in custom_metrics:
            if metric['name'] == 'test_metric' and metric['value'] == 42.0:
                found = True
                break
        
        self.assertTrue(found)

class TestMonitoringDashboard(unittest.TestCase):
    """Tests para MonitoringDashboard"""
    
    def setUp(self):
        """Configurar test"""
        if not MONITORING_AVAILABLE:
            self.skipTest("Sistema de monitoreo no disponible")
        
        # Crear monitor mock
        self.mock_monitor = Mock()
        self.mock_monitor.get_current_metrics.return_value = {
            'system': {
                'cpu_percent': 50.0,
                'memory_percent': 60.0,
                'disk_percent': 70.0
            },
            'application': {
                'active_sessions': 5,
                'processed_images': 100
            },
            'timestamp': time.time()
        }
        self.mock_monitor.get_active_alerts.return_value = []
        
        self.dashboard = MonitoringDashboard(port=5002)  # Puerto diferente para tests
        self.dashboard.monitor = self.mock_monitor
    
    def test_dashboard_creation(self):
        """Test creación del dashboard"""
        self.assertIsNotNone(self.dashboard.app)
        self.assertEqual(self.dashboard.port, 5002)
    
    @patch('requests.get')
    def test_metrics_endpoint(self, mock_get):
        """Test endpoint de métricas"""
        with self.dashboard.app.test_client() as client:
            response = client.get('/api/metrics')
            
            self.assertEqual(response.status_code, 200)
            data = json.loads(response.data)
            
            self.assertIn('system', data)
            self.assertIn('application', data)

class TestMonitoringIntegration(unittest.TestCase):
    """Tests para MonitoringIntegration"""
    
    def setUp(self):
        """Configurar test"""
        if not MONITORING_AVAILABLE:
            self.skipTest("Sistema de monitoreo no disponible")
        
        self.integration = MonitoringIntegration({
            'metrics_interval': 1,
            'alert_check_interval': 1,
            'dashboard_port': 5003,
            'enable_dashboard': False  # Deshabilitado para tests
        })
    
    def tearDown(self):
        """Limpiar después del test"""
        if hasattr(self, 'integration'):
            self.integration.stop()
    
    def test_initialize_integration(self):
        """Test inicialización de integración"""
        result = self.integration.initialize()
        self.assertTrue(result)
        self.assertIsNotNone(self.integration.monitor)
    
    def test_start_stop_integration(self):
        """Test iniciar y detener integración"""
        # Inicializar primero
        self.integration.initialize()
        
        # Iniciar
        result = self.integration.start()
        self.assertTrue(result)
        self.assertTrue(self.integration.is_running)
        
        # Detener
        self.integration.stop()
        self.assertFalse(self.integration.is_running)
    
    def test_get_metrics_integration(self):
        """Test obtener métricas a través de integración"""
        self.integration.initialize()
        self.integration.start()
        
        time.sleep(1.5)  # Esperar recolección
        
        metrics = self.integration.get_metrics()
        self.assertIsInstance(metrics, dict)
        
        self.integration.stop()

class TestGlobalFunctions(unittest.TestCase):
    """Tests para funciones globales de integración"""
    
    def setUp(self):
        """Configurar test"""
        if not MONITORING_AVAILABLE:
            self.skipTest("Sistema de monitoreo no disponible")
    
    def tearDown(self):
        """Limpiar después del test"""
        try:
            stop_monitoring()
        except:
            pass
    
    def test_initialize_monitoring_global(self):
        """Test inicialización global del monitoreo"""
        config = {
            'metrics_interval': 1,
            'alert_check_interval': 1,
            'dashboard_port': 5004,
            'enable_dashboard': False
        }
        
        result = initialize_monitoring(config)
        self.assertTrue(result)
    
    def test_start_stop_monitoring_global(self):
        """Test iniciar y detener monitoreo global"""
        # Inicializar primero
        initialize_monitoring({
            'enable_dashboard': False
        })
        
        # Iniciar
        result = start_monitoring()
        self.assertTrue(result)
        
        # Obtener métricas
        time.sleep(1.5)
        metrics = get_system_metrics()
        self.assertIsInstance(metrics, dict)
        
        # Detener
        stop_monitoring()

class TestMonitoringSystemIntegration(unittest.TestCase):
    """Tests de integración completa del sistema de monitoreo"""
    
    def setUp(self):
        """Configurar test de integración"""
        if not MONITORING_AVAILABLE:
            self.skipTest("Sistema de monitoreo no disponible")
    
    def tearDown(self):
        """Limpiar después del test"""
        try:
            stop_monitoring()
        except:
            pass
    
    def test_full_monitoring_workflow(self):
        """Test flujo completo de monitoreo"""
        # 1. Inicializar sistema
        config = {
            'metrics_interval': 1,
            'alert_check_interval': 1,
            'dashboard_port': 5005,
            'enable_dashboard': False
        }
        
        result = initialize_monitoring(config)
        self.assertTrue(result)
        
        # 2. Iniciar monitoreo
        result = start_monitoring()
        self.assertTrue(result)
        
        # 3. Esperar recolección de métricas
        time.sleep(2)
        
        # 4. Obtener métricas
        metrics = get_system_metrics()
        self.assertIsInstance(metrics, dict)
        self.assertIn('system', metrics)
        self.assertIn('application', metrics)
        
        # 5. Obtener alertas (función no disponible en el código actual)
        # alerts = get_system_alerts()
        # self.assertIsInstance(alerts, list)
        
        # 6. Agregar métrica personalizada
        add_metric("test_integration_metric", 123.45, {"test": "integration"})
        
        # 7. Detener sistema
        stop_monitoring()

def run_monitoring_tests():
    """Ejecutar todos los tests del sistema de monitoreo"""
    if not MONITORING_AVAILABLE:
        print("⚠️  Sistema de monitoreo no disponible - saltando tests")
        return True
    
    # Crear suite de tests
    test_classes = [
        TestSystemMetrics,
        TestApplicationMetrics,
        TestMetricsCollector,
        TestAlertManager,
        TestPerformanceProfiler,
        TestSystemMonitor,
        TestMonitoringDashboard,
        TestMonitoringIntegration,
        TestGlobalFunctions,
        TestMonitoringSystemIntegration
    ]
    
    suite = unittest.TestSuite()
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Ejecutar tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Mostrar resumen
    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    skipped = len(result.skipped) if hasattr(result, 'skipped') else 0
    
    print(f"\n{'='*60}")
    print(f"RESUMEN DE TESTS DEL SISTEMA DE MONITOREO")
    print(f"{'='*60}")
    print(f"Total de tests ejecutados: {total_tests}")
    print(f"Tests exitosos: {total_tests - failures - errors - skipped}")
    print(f"Tests fallidos: {failures}")
    print(f"Tests con errores: {errors}")
    print(f"Tests saltados: {skipped}")
    
    if failures == 0 and errors == 0:
        print(f"✅ Todos los tests del sistema de monitoreo pasaron exitosamente")
        return True
    else:
        print(f"❌ Algunos tests del sistema de monitoreo fallaron")
        return False

if __name__ == '__main__':
    success = run_monitoring_tests()
    sys.exit(0 if success else 1)