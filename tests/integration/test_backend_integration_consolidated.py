#!/usr/bin/env python3
"""
Tests de Integración Backend Consolidados
Sistema Balístico Forense SIGeC-Balisticar

Consolida todos los tests de integración backend en un solo archivo
Migrado desde: test_backend_integration.py, test_integration.py, test_complete_integration.py
"""

import sys
import os
import time
import traceback
import tempfile
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
import unittest
from unittest.mock import Mock, patch

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Imports del sistema
from config.unified_config import get_unified_config
from utils.logger import setup_logging, get_logger
from utils.validators import SystemValidator, SecurityUtils, FileUtils
from utils.dependency_manager import DependencyManager

# Procesamiento de imágenes
from image_processing.feature_extractor import BallisticFeatureExtractor
from image_processing.unified_roi_detector import UnifiedROIDetector
from image_processing.statistical_analyzer import StatisticalAnalyzer
from image_processing.unified_preprocessor import UnifiedPreprocessor

# Base de datos
from database.vector_db import VectorDatabase, BallisticCase, BallisticImage, FeatureVector

# Matching
from matching.unified_matcher import UnifiedMatcher

# Core
from core.unified_pipeline import UnifiedPipeline
from core.intelligent_cache import IntelligentCache


class BackendIntegrationTestSuite(unittest.TestCase):
    """Suite consolidada de tests de integración backend"""
    
    @classmethod
    def setUpClass(cls):
        """Configuración inicial para toda la suite"""
        cls.config = get_unified_config()
        cls.logger = get_logger(__name__)
        cls.test_assets_path = Path(__file__).parent.parent / "assets"
        
        # Configurar logging para tests
        setup_logging(level="DEBUG")
        
        # Inicializar componentes principales
        cls.dependency_manager = DependencyManager()
        cls.cache = IntelligentCache(cls.config)
        
        cls.logger.info("Backend Integration Test Suite initialized")
    
    def setUp(self):
        """Configuración para cada test individual"""
        self.start_time = time.time()
        
    def tearDown(self):
        """Limpieza después de cada test"""
        execution_time = time.time() - self.start_time
        self.logger.debug(f"Test executed in {execution_time:.2f}s")

    def test_module_imports(self):
        """Test consolidado de importaciones de módulos"""
        self.logger.info("🔍 Testing module imports...")
        
        # Test imports básicos
        try:
            from config.unified_config import get_unified_config
            from utils.logger import setup_logging, get_logger
            self.assertTrue(True, "Basic utilities imported successfully")
        except ImportError as e:
            self.fail(f"Failed to import basic utilities: {e}")
        
        # Test imports de procesamiento
        try:
            from image_processing.feature_extractor import BallisticFeatureExtractor
            from image_processing.unified_roi_detector import UnifiedROIDetector
            self.assertTrue(True, "Image processing modules imported successfully")
        except ImportError as e:
            self.fail(f"Failed to import image processing modules: {e}")
        
        # Test imports de base de datos
        try:
            from database.vector_db import VectorDatabase
            self.assertTrue(True, "Database modules imported successfully")
        except ImportError as e:
            self.fail(f"Failed to import database modules: {e}")
        
        # Test imports de matching
        try:
            from matching.unified_matcher import UnifiedMatcher
            self.assertTrue(True, "Matching modules imported successfully")
        except ImportError as e:
            self.fail(f"Failed to import matching modules: {e}")

    def test_configuration_system(self):
        """Test consolidado del sistema de configuración"""
        self.logger.info("🔧 Testing configuration system...")
        
        config = get_unified_config()
        self.assertIsNotNone(config, "Configuration should not be None")
        
        # Verificar configuraciones críticas
        self.assertTrue(hasattr(config, 'database'), "Config should have database section")
        self.assertTrue(hasattr(config, 'image_processing'), "Config should have image_processing section")
        self.assertTrue(hasattr(config, 'matching'), "Config should have matching section")
        
        # Test validación de configuración
        validator = SystemValidator(config)
        validation_result = validator.validate_system()
        self.assertTrue(validation_result.get('valid', False), 
                       f"System validation failed: {validation_result}")

    def test_dependency_management(self):
        """Test del sistema de gestión de dependencias"""
        self.logger.info("📦 Testing dependency management...")
        
        # Test inicialización del dependency manager
        dm = DependencyManager()
        self.assertIsNotNone(dm, "DependencyManager should initialize")
        
        # Test verificación de dependencias críticas
        critical_deps = ['numpy', 'opencv-python', 'scikit-learn']
        for dep in critical_deps:
            available = dm.is_available(dep)
            self.assertTrue(available, f"Critical dependency {dep} should be available")
        
        # Test fallbacks
        fallback_result = dm.get_fallback_implementation('non_existent_module')
        self.assertIsNotNone(fallback_result, "Should provide fallback for missing modules")

    def test_image_processing_pipeline(self):
        """Test consolidado del pipeline de procesamiento de imágenes"""
        self.logger.info("🖼️ Testing image processing pipeline...")
        
        # Buscar imagen de test
        test_image_path = self._find_test_image()
        if not test_image_path:
            self.skipTest("No test image available")
        
        try:
            # Test preprocessor
            preprocessor = UnifiedPreprocessor(self.config)
            processed_image = preprocessor.preprocess_image(test_image_path)
            self.assertIsNotNone(processed_image, "Preprocessed image should not be None")
            
            # Test feature extractor
            feature_extractor = BallisticFeatureExtractor(self.config)
            features = feature_extractor.extract_features(processed_image)
            self.assertIsNotNone(features, "Features should not be None")
            self.assertGreater(len(features.get('keypoints', [])), 0, "Should extract keypoints")
            
            # Test ROI detector
            roi_detector = UnifiedROIDetector(self.config)
            roi_result = roi_detector.detect_roi(processed_image)
            self.assertIsNotNone(roi_result, "ROI detection should not be None")
            
        except Exception as e:
            self.fail(f"Image processing pipeline failed: {e}")

    def test_database_operations(self):
        """Test consolidado de operaciones de base de datos"""
        self.logger.info("🗄️ Testing database operations...")
        
        try:
            # Test inicialización de base de datos
            db = VectorDatabase(self.config)
            self.assertIsNotNone(db, "Database should initialize")
            
            # Test creación de caso balístico
            test_case = BallisticCase(
                case_id="TEST_CASE_001",
                case_name="Test Case Integration",
                description="Test case for integration testing"
            )
            
            # Test operaciones CRUD básicas
            case_id = db.create_case(test_case)
            self.assertIsNotNone(case_id, "Should create case successfully")
            
            retrieved_case = db.get_case(case_id)
            self.assertIsNotNone(retrieved_case, "Should retrieve case successfully")
            self.assertEqual(retrieved_case.case_name, test_case.case_name)
            
            # Cleanup
            db.delete_case(case_id)
            
        except Exception as e:
            self.fail(f"Database operations failed: {e}")

    def test_matching_system(self):
        """Test consolidado del sistema de matching"""
        self.logger.info("🔍 Testing matching system...")
        
        test_image_path = self._find_test_image()
        if not test_image_path:
            self.skipTest("No test image available")
        
        try:
            # Test inicialización del matcher
            matcher = UnifiedMatcher(self.config)
            self.assertIsNotNone(matcher, "Matcher should initialize")
            
            # Test matching con la misma imagen (debería dar alta similitud)
            result = matcher.compare_images(test_image_path, test_image_path)
            self.assertIsNotNone(result, "Matching result should not be None")
            self.assertGreater(result.similarity_score, 80.0, 
                             "Same image should have high similarity")
            
            # Test con diferentes algoritmos
            algorithms = ['ORB', 'SIFT', 'AKAZE']
            for algorithm in algorithms:
                try:
                    result = matcher.compare_images(
                        test_image_path, test_image_path, 
                        algorithm=algorithm
                    )
                    self.assertIsNotNone(result, f"{algorithm} matching should work")
                except Exception as e:
                    self.logger.warning(f"{algorithm} algorithm failed: {e}")
            
        except Exception as e:
            self.fail(f"Matching system failed: {e}")

    def test_unified_pipeline(self):
        """Test del pipeline unificado completo"""
        self.logger.info("🔄 Testing unified pipeline...")
        
        test_image_path = self._find_test_image()
        if not test_image_path:
            self.skipTest("No test image available")
        
        try:
            # Test inicialización del pipeline
            pipeline = UnifiedPipeline(self.config)
            self.assertIsNotNone(pipeline, "Pipeline should initialize")
            
            # Test procesamiento completo
            result = pipeline.process_image(test_image_path)
            self.assertIsNotNone(result, "Pipeline result should not be None")
            
            # Verificar componentes del resultado
            self.assertIn('features', result, "Result should contain features")
            self.assertIn('roi', result, "Result should contain ROI")
            self.assertIn('quality_metrics', result, "Result should contain quality metrics")
            
        except Exception as e:
            self.fail(f"Unified pipeline failed: {e}")

    def test_cache_system(self):
        """Test del sistema de cache inteligente"""
        self.logger.info("💾 Testing cache system...")
        
        try:
            cache = IntelligentCache(self.config)
            self.assertIsNotNone(cache, "Cache should initialize")
            
            # Test operaciones básicas de cache
            test_key = "test_cache_key"
            test_value = {"test": "data", "timestamp": time.time()}
            
            # Test set/get
            cache.set(test_key, test_value)
            retrieved_value = cache.get(test_key)
            self.assertEqual(retrieved_value, test_value, "Cache should store and retrieve correctly")
            
            # Test invalidación
            cache.invalidate(test_key)
            retrieved_after_invalidation = cache.get(test_key)
            self.assertIsNone(retrieved_after_invalidation, "Cache should be invalidated")
            
        except Exception as e:
            self.fail(f"Cache system failed: {e}")

    def test_error_handling(self):
        """Test consolidado del manejo de errores"""
        self.logger.info("⚠️ Testing error handling...")
        
        # Test con archivo inexistente
        try:
            preprocessor = UnifiedPreprocessor(self.config)
            result = preprocessor.preprocess_image("non_existent_file.jpg")
            # Debería manejar el error gracefully
            self.assertIsNone(result, "Should handle missing file gracefully")
        except Exception as e:
            # Si lanza excepción, debería ser manejada apropiadamente
            self.assertIsInstance(e, (FileNotFoundError, ValueError))

    def test_performance_benchmarks(self):
        """Test de benchmarks de rendimiento"""
        self.logger.info("⚡ Testing performance benchmarks...")
        
        test_image_path = self._find_test_image()
        if not test_image_path:
            self.skipTest("No test image available")
        
        # Test tiempo de procesamiento
        start_time = time.time()
        
        try:
            preprocessor = UnifiedPreprocessor(self.config)
            processed_image = preprocessor.preprocess_image(test_image_path)
            
            feature_extractor = BallisticFeatureExtractor(self.config)
            features = feature_extractor.extract_features(processed_image)
            
            processing_time = time.time() - start_time
            
            # Verificar que el procesamiento sea razonablemente rápido
            self.assertLess(processing_time, 30.0, 
                           f"Processing should complete within 30s, took {processing_time:.2f}s")
            
            self.logger.info(f"Processing completed in {processing_time:.2f}s")
            
        except Exception as e:
            self.fail(f"Performance benchmark failed: {e}")

    def _find_test_image(self) -> Optional[str]:
        """Buscar imagen de test disponible"""
        test_images = [
            self.test_assets_path / "test_image.png",
            self.test_assets_path / "FBI 58A008995 RP1_BFR.png",
            self.test_assets_path / "FBI B240793 RP1_BFR.png",
            self.test_assets_path / "SS007_CCI BF R.png"
        ]
        
        for img_path in test_images:
            if img_path.exists():
                return str(img_path)
        
        return None


def run_backend_integration_tests():
    """Ejecutar todos los tests de integración backend"""
    print("🚀 Ejecutando Tests de Integración Backend Consolidados")
    print("=" * 60)
    
    # Configurar test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(BackendIntegrationTestSuite)
    
    # Ejecutar tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Resumen de resultados
    print("\n" + "=" * 60)
    print(f"Tests ejecutados: {result.testsRun}")
    print(f"Errores: {len(result.errors)}")
    print(f"Fallos: {len(result.failures)}")
    print(f"Saltados: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
    
    success = len(result.errors) == 0 and len(result.failures) == 0
    print(f"Estado: {'✅ ÉXITO' if success else '❌ FALLÓ'}")
    
    return success


if __name__ == "__main__":
    success = run_backend_integration_tests()
    sys.exit(0 if success else 1)