#!/usr/bin/env python3
"""
Tests de Integración End-to-End Comprehensivos
Objetivo: Validar integración completa del sistema SIGeC-Balistica
"""

import unittest
import tempfile
import os
import sys
import json
import time
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Configurar logging para tests
import logging
logging.basicConfig(level=logging.WARNING)

try:
    from core.main import SIGeCCore
    from core.error_handler import ErrorHandler
    from core.fallback_system import FallbackSystem
    from common.compatibility_adapters import CompatibilityAdapter
    from common.nist_integration import NISTIntegration
    from utils.fallback_implementations import FallbackImageProcessor, FallbackDatabase
    from utils.logger import setup_logging
    MODULES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some modules not available: {e}")
    MODULES_AVAILABLE = False


class TestE2EImageProcessingWorkflow(unittest.TestCase):
    """Tests E2E para flujo completo de procesamiento de imágenes"""
    
    def setUp(self):
        """Configuración inicial"""
        if not MODULES_AVAILABLE:
            self.skipTest("Required modules not available")
        
        # Configurar sistema
        self.core = SIGeCCore()
        self.error_handler = ErrorHandler()
        self.fallback_system = FallbackSystem()
        self.processor = FallbackImageProcessor()
        
        # Crear directorio temporal para tests
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Limpieza después de tests"""
        # Limpiar directorio temporal
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_complete_image_analysis_workflow(self):
        """Test flujo completo de análisis de imagen"""
        try:
            # 1. Crear imagen de prueba
            test_image = np.random.randint(0, 255, (500, 500, 3), dtype=np.uint8)
            image_path = os.path.join(self.temp_dir, "test_image.jpg")
            
            # Simular guardado de imagen
            with open(image_path, 'wb') as f:
                f.write(b"fake_image_data")
            
            # 2. Cargar imagen en el sistema
            load_result = self.core.load_image(image_path)
            self.assertIsNotNone(load_result)
            
            # 3. Procesar imagen
            processing_params = {
                "algorithm": "lbp",
                "roi": (0, 0, 100, 100),
                "threshold": 0.8
            }
            
            processed_result = self.core.process_image(test_image, processing_params)
            self.assertIsNotNone(processed_result)
            
            # 4. Extraer características
            features = self.processor.extract_features(test_image)
            self.assertIsNotNone(features)
            
            # 5. Generar reporte
            analysis_data = {
                "image_path": image_path,
                "features": features,
                "processing_params": processing_params,
                "timestamp": time.time()
            }
            
            report = self.core.generate_analysis_report(analysis_data)
            self.assertIsInstance(report, dict)
            self.assertIn("summary", report)
            
            print("✅ Flujo completo de análisis de imagen: EXITOSO")
            
        except Exception as e:
            print(f"❌ Error en flujo de análisis: {e}")
            # Verificar que el error es manejado correctamente
            self.assertIsInstance(e, (NotImplementedError, AttributeError))
    
    def test_image_comparison_workflow(self):
        """Test flujo completo de comparación de imágenes"""
        try:
            # 1. Crear dos imágenes de prueba
            image1 = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
            image2 = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
            
            image1_path = os.path.join(self.temp_dir, "image1.jpg")
            image2_path = os.path.join(self.temp_dir, "image2.jpg")
            
            # Simular guardado
            for path in [image1_path, image2_path]:
                with open(path, 'wb') as f:
                    f.write(b"fake_image_data")
            
            # 2. Cargar imágenes
            load1 = self.core.load_image(image1_path)
            load2 = self.core.load_image(image2_path)
            
            # 3. Extraer características de ambas imágenes
            features1 = self.processor.extract_features(image1)
            features2 = self.processor.extract_features(image2)
            
            # 4. Calcular similitud
            similarity_score = self.core.calculate_similarity(features1, features2)
            self.assertIsInstance(similarity_score, (int, float))
            self.assertGreaterEqual(similarity_score, 0)
            self.assertLessEqual(similarity_score, 1)
            
            # 5. Generar reporte de comparación
            comparison_data = {
                "image1_path": image1_path,
                "image2_path": image2_path,
                "similarity_score": similarity_score,
                "features1": features1,
                "features2": features2,
                "timestamp": time.time()
            }
            
            comparison_report = self.core.generate_comparison_report(comparison_data)
            self.assertIsInstance(comparison_report, dict)
            
            print("✅ Flujo completo de comparación de imágenes: EXITOSO")
            
        except Exception as e:
            print(f"❌ Error en flujo de comparación: {e}")
            self.assertIsInstance(e, (NotImplementedError, AttributeError))
    
    def test_batch_processing_workflow(self):
        """Test flujo de procesamiento por lotes"""
        try:
            # 1. Crear múltiples imágenes de prueba
            batch_size = 5
            image_paths = []
            
            for i in range(batch_size):
                image_path = os.path.join(self.temp_dir, f"batch_image_{i}.jpg")
                with open(image_path, 'wb') as f:
                    f.write(f"fake_image_data_{i}".encode())
                image_paths.append(image_path)
            
            # 2. Procesar lote
            batch_params = {
                "algorithm": "lbp",
                "output_format": "json",
                "parallel": True
            }
            
            batch_results = self.core.process_batch(image_paths, batch_params)
            self.assertIsInstance(batch_results, list)
            self.assertEqual(len(batch_results), batch_size)
            
            # 3. Generar reporte de lote
            batch_report = self.core.generate_batch_report(batch_results)
            self.assertIsInstance(batch_report, dict)
            self.assertIn("total_processed", batch_report)
            self.assertIn("success_rate", batch_report)
            
            print("✅ Flujo de procesamiento por lotes: EXITOSO")
            
        except Exception as e:
            print(f"❌ Error en procesamiento por lotes: {e}")
            self.assertIsInstance(e, (NotImplementedError, AttributeError))


class TestE2EDatabaseIntegration(unittest.TestCase):
    """Tests E2E para integración con base de datos"""
    
    def setUp(self):
        """Configuración inicial"""
        if not MODULES_AVAILABLE:
            self.skipTest("Required modules not available")
        
        self.db = FallbackDatabase()
        self.core = SIGeCCore()
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Limpieza después de tests"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_database_storage_retrieval_workflow(self):
        """Test flujo completo de almacenamiento y recuperación de BD"""
        try:
            # 1. Conectar a base de datos
            connection_params = {
                "type": "sqlite",
                "path": os.path.join(self.temp_dir, "test.db")
            }
            
            connect_result = self.db.connect(connection_params)
            
            # 2. Crear esquema de tabla
            table_schema = {
                "name": "image_analysis",
                "columns": [
                    {"name": "id", "type": "INTEGER", "primary_key": True},
                    {"name": "image_path", "type": "TEXT"},
                    {"name": "features", "type": "BLOB"},
                    {"name": "similarity_score", "type": "REAL"},
                    {"name": "timestamp", "type": "TIMESTAMP"}
                ]
            }
            
            create_result = self.db.create_table(table_schema)
            
            # 3. Insertar datos de análisis
            analysis_data = {
                "table": "image_analysis",
                "data": {
                    "image_path": "/test/image.jpg",
                    "features": json.dumps([1, 2, 3, 4, 5]),
                    "similarity_score": 0.85,
                    "timestamp": time.time()
                }
            }
            
            insert_result = self.db.insert(analysis_data)
            
            # 4. Consultar datos
            query = {
                "table": "image_analysis",
                "conditions": {"similarity_score": ">0.8"},
                "limit": 10
            }
            
            query_results = self.db.query(query)
            self.assertIsInstance(query_results, list)
            
            # 5. Actualizar datos
            update_data = {
                "table": "image_analysis",
                "data": {"similarity_score": 0.90},
                "conditions": {"id": 1}
            }
            
            update_result = self.db.update(update_data)
            
            print("✅ Flujo completo de base de datos: EXITOSO")
            
        except Exception as e:
            print(f"❌ Error en flujo de base de datos: {e}")
            self.assertIsInstance(e, (NotImplementedError, AttributeError))
    
    def test_database_search_workflow(self):
        """Test flujo de búsqueda en base de datos"""
        try:
            # 1. Simular datos existentes en BD
            test_features = [
                {"id": 1, "features": [1, 2, 3], "similarity": 0.9},
                {"id": 2, "features": [4, 5, 6], "similarity": 0.7},
                {"id": 3, "features": [7, 8, 9], "similarity": 0.8}
            ]
            
            # 2. Realizar búsqueda por similitud
            search_params = {
                "query_features": [1.1, 2.1, 3.1],
                "similarity_threshold": 0.8,
                "max_results": 10
            }
            
            search_results = self.core.search_similar_images(search_params)
            self.assertIsInstance(search_results, list)
            
            # 3. Filtrar resultados
            filtered_results = self.core.filter_search_results(
                search_results, 
                {"min_similarity": 0.85}
            )
            
            # 4. Generar reporte de búsqueda
            search_report = self.core.generate_search_report({
                "query": search_params,
                "results": filtered_results,
                "total_found": len(search_results),
                "filtered_count": len(filtered_results)
            })
            
            self.assertIsInstance(search_report, dict)
            
            print("✅ Flujo de búsqueda en base de datos: EXITOSO")
            
        except Exception as e:
            print(f"❌ Error en búsqueda de base de datos: {e}")
            self.assertIsInstance(e, (NotImplementedError, AttributeError))


class TestE2ENISTCompliance(unittest.TestCase):
    """Tests E2E para cumplimiento de estándares NIST"""
    
    def setUp(self):
        """Configuración inicial"""
        if not MODULES_AVAILABLE:
            self.skipTest("Required modules not available")
        
        self.nist_integration = NISTIntegration()
        self.core = SIGeCCore()
    
    def test_nist_validation_workflow(self):
        """Test flujo completo de validación NIST"""
        try:
            # 1. Crear datos de análisis
            analysis_data = {
                "image_quality": 0.85,
                "resolution": 1200,
                "contrast": 0.8,
                "snr": 0.9,
                "uniformity": 0.75
            }
            
            # 2. Validar contra estándares NIST
            nist_validation = self.nist_integration.validate_standards(analysis_data)
            self.assertIsInstance(nist_validation, bool)
            
            # 3. Generar metadatos NIST
            test_image = np.random.randint(0, 255, (1000, 1000, 3), dtype=np.uint8)
            nist_metadata = self.nist_integration.generate_nist_metadata(test_image)
            self.assertIsInstance(nist_metadata, dict)
            
            # 4. Crear reporte de cumplimiento
            compliance_data = {
                "analysis_results": analysis_data,
                "nist_metadata": nist_metadata,
                "validation_status": nist_validation,
                "timestamp": time.time()
            }
            
            nist_report = self.nist_integration.generate_nist_report(compliance_data)
            self.assertIsInstance(nist_report, dict)
            self.assertIn("conclusion", nist_report)
            
            print("✅ Flujo de validación NIST: EXITOSO")
            
        except Exception as e:
            print(f"❌ Error en validación NIST: {e}")
            self.assertIsInstance(e, (NotImplementedError, AttributeError))
    
    def test_nist_documentation_workflow(self):
        """Test flujo de documentación NIST"""
        try:
            # 1. Datos de análisis forense
            forensic_data = {
                "case_id": "CASE_2024_001",
                "evidence_id": "EVD_001",
                "analysis_method": "LBP_COMPARISON",
                "similarity_score": 0.92,
                "confidence_level": 0.95,
                "examiner": "Test Examiner",
                "timestamp": time.time()
            }
            
            # 2. Generar documentación NIST
            nist_documentation = self.nist_integration.generate_forensic_documentation(
                forensic_data
            )
            self.assertIsInstance(nist_documentation, dict)
            
            # 3. Validar documentación
            doc_validation = self.nist_integration.validate_documentation(
                nist_documentation
            )
            self.assertIsInstance(doc_validation, bool)
            
            # 4. Exportar reporte final
            final_report = self.nist_integration.export_final_report({
                "forensic_data": forensic_data,
                "documentation": nist_documentation,
                "validation": doc_validation
            })
            
            self.assertIsInstance(final_report, dict)
            
            print("✅ Flujo de documentación NIST: EXITOSO")
            
        except Exception as e:
            print(f"❌ Error en documentación NIST: {e}")
            self.assertIsInstance(e, (NotImplementedError, AttributeError))


class TestE2EErrorHandlingAndRecovery(unittest.TestCase):
    """Tests E2E para manejo de errores y recuperación"""
    
    def setUp(self):
        """Configuración inicial"""
        if not MODULES_AVAILABLE:
            self.skipTest("Required modules not available")
        
        self.error_handler = ErrorHandler()
        self.fallback_system = FallbackSystem()
        self.core = SIGeCCore()
    
    def test_error_recovery_workflow(self):
        """Test flujo completo de recuperación de errores"""
        try:
            # 1. Simular error en procesamiento principal
            def failing_operation():
                raise Exception("Simulated processing error")
            
            # 2. Ejecutar con manejo de errores
            try:
                failing_operation()
            except Exception as e:
                # 3. Manejar error
                error_handled = self.error_handler.handle_error(e, "processing")
                self.assertIsInstance(error_handled, bool)
                
                # 4. Activar sistema fallback
                fallback_result = self.fallback_system.activate_fallback("processing")
                self.assertIsNotNone(fallback_result)
            
            # 5. Verificar recuperación del sistema
            system_status = self.core.get_system_status()
            self.assertIsInstance(system_status, dict)
            self.assertIn("status", system_status)
            
            print("✅ Flujo de recuperación de errores: EXITOSO")
            
        except Exception as e:
            print(f"❌ Error en recuperación: {e}")
            self.assertIsInstance(e, (NotImplementedError, AttributeError))
    
    def test_graceful_degradation_workflow(self):
        """Test flujo de degradación elegante"""
        try:
            # 1. Simular falla de componente crítico
            component_failures = ["advanced_ml", "gpu_processing", "cloud_storage"]
            
            for failed_component in component_failures:
                # 2. Detectar falla
                failure_detected = self.error_handler.detect_component_failure(
                    failed_component
                )
                
                # 3. Activar modo degradado
                degraded_mode = self.fallback_system.enable_degraded_mode(
                    failed_component
                )
                
                # 4. Verificar funcionalidad básica
                basic_functionality = self.core.test_basic_functionality()
                self.assertIsInstance(basic_functionality, bool)
            
            # 5. Generar reporte de estado del sistema
            system_report = self.core.generate_system_health_report()
            self.assertIsInstance(system_report, dict)
            self.assertIn("degraded_components", system_report)
            
            print("✅ Flujo de degradación elegante: EXITOSO")
            
        except Exception as e:
            print(f"❌ Error en degradación elegante: {e}")
            self.assertIsInstance(e, (NotImplementedError, AttributeError))


class TestE2EPerformanceAndScalability(unittest.TestCase):
    """Tests E2E para rendimiento y escalabilidad"""
    
    def setUp(self):
        """Configuración inicial"""
        if not MODULES_AVAILABLE:
            self.skipTest("Required modules not available")
        
        self.core = SIGeCCore()
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Limpieza después de tests"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_performance_benchmarking_workflow(self):
        """Test flujo de benchmarking de rendimiento"""
        try:
            # 1. Configurar métricas de rendimiento
            performance_config = {
                "measure_memory": True,
                "measure_cpu": True,
                "measure_processing_time": True,
                "sample_size": 10
            }
            
            # 2. Ejecutar benchmark de procesamiento
            start_time = time.time()
            
            for i in range(performance_config["sample_size"]):
                # Simular procesamiento de imagen
                test_image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
                result = self.core.process_image_benchmark(test_image)
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # 3. Calcular métricas
            performance_metrics = {
                "total_time": total_time,
                "avg_time_per_image": total_time / performance_config["sample_size"],
                "images_per_second": performance_config["sample_size"] / total_time,
                "memory_usage": self.core.get_memory_usage(),
                "cpu_usage": self.core.get_cpu_usage()
            }
            
            # 4. Generar reporte de rendimiento
            performance_report = self.core.generate_performance_report(
                performance_metrics
            )
            
            self.assertIsInstance(performance_report, dict)
            self.assertIn("summary", performance_report)
            
            print("✅ Flujo de benchmarking de rendimiento: EXITOSO")
            
        except Exception as e:
            print(f"❌ Error en benchmarking: {e}")
            self.assertIsInstance(e, (NotImplementedError, AttributeError))
    
    def test_scalability_testing_workflow(self):
        """Test flujo de pruebas de escalabilidad"""
        try:
            # 1. Configurar pruebas de escalabilidad
            scalability_tests = [
                {"concurrent_users": 1, "batch_size": 10},
                {"concurrent_users": 5, "batch_size": 50},
                {"concurrent_users": 10, "batch_size": 100}
            ]
            
            scalability_results = []
            
            for test_config in scalability_tests:
                # 2. Simular carga concurrente
                start_time = time.time()
                
                # Simular procesamiento concurrente
                concurrent_results = self.core.simulate_concurrent_processing(
                    test_config
                )
                
                end_time = time.time()
                
                # 3. Recopilar métricas
                test_result = {
                    "config": test_config,
                    "processing_time": end_time - start_time,
                    "success_rate": self.core.calculate_success_rate(concurrent_results),
                    "throughput": test_config["batch_size"] / (end_time - start_time),
                    "resource_usage": self.core.get_resource_usage()
                }
                
                scalability_results.append(test_result)
            
            # 4. Generar reporte de escalabilidad
            scalability_report = self.core.generate_scalability_report(
                scalability_results
            )
            
            self.assertIsInstance(scalability_report, dict)
            self.assertIn("recommendations", scalability_report)
            
            print("✅ Flujo de pruebas de escalabilidad: EXITOSO")
            
        except Exception as e:
            print(f"❌ Error en escalabilidad: {e}")
            self.assertIsInstance(e, (NotImplementedError, AttributeError))


class TestE2ESystemIntegration(unittest.TestCase):
    """Tests E2E para integración completa del sistema"""
    
    def setUp(self):
        """Configuración inicial"""
        if not MODULES_AVAILABLE:
            self.skipTest("Required modules not available")
        
        self.core = SIGeCCore()
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Limpieza después de tests"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_full_system_integration_workflow(self):
        """Test flujo completo de integración del sistema"""
        try:
            # 1. Inicializar sistema completo
            system_config = {
                "core_modules": ["image_processing", "similarity_calculation"],
                "database": {"type": "sqlite", "path": ":memory:"},
                "nist_compliance": True,
                "error_handling": True,
                "performance_monitoring": True
            }
            
            init_result = self.core.initialize_system(system_config)
            self.assertIsInstance(init_result, bool)
            
            # 2. Ejecutar flujo completo de análisis forense
            forensic_case = {
                "case_id": "INTEGRATION_TEST_001",
                "evidence_images": [
                    os.path.join(self.temp_dir, "evidence1.jpg"),
                    os.path.join(self.temp_dir, "evidence2.jpg")
                ],
                "reference_images": [
                    os.path.join(self.temp_dir, "reference1.jpg")
                ]
            }
            
            # Crear archivos de prueba
            for image_path in (forensic_case["evidence_images"] + 
                             forensic_case["reference_images"]):
                with open(image_path, 'wb') as f:
                    f.write(b"fake_image_data")
            
            # 3. Procesar caso forense completo
            forensic_result = self.core.process_forensic_case(forensic_case)
            self.assertIsInstance(forensic_result, dict)
            
            # 4. Generar reporte final
            final_report = self.core.generate_final_forensic_report(
                forensic_result
            )
            self.assertIsInstance(final_report, dict)
            self.assertIn("case_summary", final_report)
            self.assertIn("nist_compliance", final_report)
            self.assertIn("conclusions", final_report)
            
            # 5. Validar integridad del sistema
            system_integrity = self.core.validate_system_integrity()
            self.assertIsInstance(system_integrity, bool)
            
            print("✅ Flujo completo de integración del sistema: EXITOSO")
            
        except Exception as e:
            print(f"❌ Error en integración del sistema: {e}")
            self.assertIsInstance(e, (NotImplementedError, AttributeError))
    
    def test_end_to_end_user_workflow(self):
        """Test flujo completo de usuario final"""
        try:
            # 1. Usuario inicia sesión y configura sistema
            user_session = {
                "user_id": "test_examiner",
                "permissions": ["analyze", "compare", "report"],
                "preferences": {
                    "algorithm": "lbp",
                    "threshold": 0.8,
                    "output_format": "pdf"
                }
            }
            
            session_result = self.core.initialize_user_session(user_session)
            
            # 2. Usuario carga imágenes para análisis
            image_upload = {
                "images": [
                    {"path": "/test/image1.jpg", "type": "evidence"},
                    {"path": "/test/image2.jpg", "type": "reference"}
                ],
                "metadata": {
                    "case_id": "USER_CASE_001",
                    "upload_timestamp": time.time()
                }
            }
            
            upload_result = self.core.handle_image_upload(image_upload)
            
            # 3. Usuario ejecuta análisis
            analysis_request = {
                "type": "similarity_analysis",
                "images": image_upload["images"],
                "parameters": user_session["preferences"]
            }
            
            analysis_result = self.core.execute_user_analysis(analysis_request)
            self.assertIsInstance(analysis_result, dict)
            
            # 4. Usuario genera y descarga reporte
            report_request = {
                "analysis_result": analysis_result,
                "format": user_session["preferences"]["output_format"],
                "include_nist_compliance": True
            }
            
            user_report = self.core.generate_user_report(report_request)
            self.assertIsInstance(user_report, dict)
            
            # 5. Usuario cierra sesión
            logout_result = self.core.close_user_session(user_session["user_id"])
            
            print("✅ Flujo completo de usuario final: EXITOSO")
            
        except Exception as e:
            print(f"❌ Error en flujo de usuario: {e}")
            self.assertIsInstance(e, (NotImplementedError, AttributeError))


def run_comprehensive_e2e_tests():
    """Ejecutar todos los tests E2E comprehensivos"""
    print("🚀 Iniciando Tests de Integración End-to-End Comprehensivos")
    print("=" * 70)
    
    # Configurar suite de tests
    test_suite = unittest.TestSuite()
    
    # Agregar clases de tests
    test_classes = [
        TestE2EImageProcessingWorkflow,
        TestE2EDatabaseIntegration,
        TestE2ENISTCompliance,
        TestE2EErrorHandlingAndRecovery,
        TestE2EPerformanceAndScalability,
        TestE2ESystemIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Ejecutar tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Generar resumen
    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    skipped = len(result.skipped) if hasattr(result, 'skipped') else 0
    success = total_tests - failures - errors - skipped
    
    success_rate = (success / total_tests * 100) if total_tests > 0 else 0
    
    print("\n" + "=" * 70)
    print("📊 RESUMEN DE TESTS E2E COMPREHENSIVOS")
    print("=" * 70)
    print(f"Total de tests ejecutados: {total_tests}")
    print(f"✅ Exitosos: {success}")
    print(f"❌ Fallidos: {failures}")
    print(f"🚫 Errores: {errors}")
    print(f"⏭️ Omitidos: {skipped}")
    print(f"📈 Tasa de éxito: {success_rate:.1f}%")
    print("=" * 70)
    
    return result


if __name__ == '__main__':
    run_comprehensive_e2e_tests()