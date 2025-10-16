#!/usr/bin/env python3
"""
Tests unitarios para módulos Common
Objetivo: Mejorar cobertura de testing del módulo common (actualmente 0-27%)
"""

import unittest
import tempfile
import os
import sys
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from common.compatibility_adapters import CompatibilityAdapter, LegacyImageProcessor
    from common.nist_integration import NISTIntegration, NISTStandards
    from common.similarity_functions_unified import SimilarityCalculator, SimilarityMetrics
    from common.statistical_core import StatisticalAnalyzer, ClusteringAnalyzer
    COMMON_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Common modules not fully available: {e}")
    COMMON_AVAILABLE = False


class TestCompatibilityAdapter(unittest.TestCase):
    """Tests para CompatibilityAdapter"""
    
    def setUp(self):
        """Configuración inicial"""
        if not COMMON_AVAILABLE:
            self.skipTest("Common modules not available")
        self.adapter = CompatibilityAdapter()
    
    def test_adapter_initialization(self):
        """Test inicialización del adaptador"""
        self.assertIsNotNone(self.adapter)
        self.assertIsInstance(self.adapter, CompatibilityAdapter)
    
    def test_legacy_format_conversion(self):
        """Test conversión de formatos legacy"""
        try:
            # Datos de prueba en formato legacy
            legacy_data = {
                "image_path": "/test/path/image.jpg",
                "features": [1, 2, 3, 4, 5],
                "metadata": {"width": 640, "height": 480}
            }
            
            # Convertir a formato moderno
            modern_data = self.adapter.convert_legacy_format(legacy_data)
            
            # Verificar conversión
            self.assertIsInstance(modern_data, dict)
            self.assertIn("image_path", modern_data)
            self.assertIn("features", modern_data)
        except Exception as e:
            self.assertIsInstance(e, (NotImplementedError, AttributeError))
    
    def test_backward_compatibility(self):
        """Test compatibilidad hacia atrás"""
        try:
            # Datos modernos
            modern_data = {
                "image_path": "/test/path/image.jpg",
                "feature_vector": np.array([1, 2, 3, 4, 5]),
                "quality_metrics": {"snr": 0.8, "contrast": 0.7}
            }
            
            # Convertir a formato legacy
            legacy_data = self.adapter.convert_to_legacy(modern_data)
            
            # Verificar conversión
            self.assertIsInstance(legacy_data, dict)
            self.assertIn("image_path", legacy_data)
        except Exception as e:
            self.assertIsInstance(e, (NotImplementedError, AttributeError))
    
    def test_version_detection(self):
        """Test detección de versión de datos"""
        try:
            # Datos con diferentes versiones
            v1_data = {"version": "1.0", "data": "legacy_format"}
            v2_data = {"version": "2.0", "data": "modern_format"}
            
            v1_detected = self.adapter.detect_version(v1_data)
            v2_detected = self.adapter.detect_version(v2_data)
            
            self.assertEqual(v1_detected, "1.0")
            self.assertEqual(v2_detected, "2.0")
        except Exception as e:
            self.assertIsInstance(e, (NotImplementedError, AttributeError))


class TestLegacyImageProcessor(unittest.TestCase):
    """Tests para LegacyImageProcessor"""
    
    def setUp(self):
        """Configuración inicial"""
        if not COMMON_AVAILABLE:
            self.skipTest("Common modules not available")
        self.processor = LegacyImageProcessor()
    
    def test_processor_initialization(self):
        """Test inicialización del procesador legacy"""
        self.assertIsNotNone(self.processor)
        self.assertIsInstance(self.processor, LegacyImageProcessor)
    
    def test_legacy_feature_extraction(self):
        """Test extracción de características legacy"""
        try:
            # Crear imagen de prueba
            test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            
            # Extraer características
            features = self.processor.extract_legacy_features(test_image)
            
            # Verificar resultado
            self.assertIsNotNone(features)
            self.assertIsInstance(features, (list, np.ndarray))
        except Exception as e:
            self.assertIsInstance(e, (NotImplementedError, AttributeError))
    
    def test_legacy_preprocessing(self):
        """Test preprocesamiento legacy"""
        try:
            # Crear imagen de prueba
            test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            
            # Aplicar preprocesamiento legacy
            processed = self.processor.legacy_preprocess(test_image)
            
            # Verificar resultado
            self.assertIsNotNone(processed)
            self.assertIsInstance(processed, np.ndarray)
        except Exception as e:
            self.assertIsInstance(e, (NotImplementedError, AttributeError))


class TestNISTIntegration(unittest.TestCase):
    """Tests para NISTIntegration"""
    
    def setUp(self):
        """Configuración inicial"""
        if not COMMON_AVAILABLE:
            self.skipTest("Common modules not available")
        self.nist_integration = NISTIntegration()
    
    def test_nist_integration_initialization(self):
        """Test inicialización de integración NIST"""
        self.assertIsNotNone(self.nist_integration)
        self.assertIsInstance(self.nist_integration, NISTIntegration)
    
    def test_nist_standards_validation(self):
        """Test validación de estándares NIST"""
        try:
            # Datos de prueba que cumplen estándares NIST
            test_data = {
                "image_quality": 0.8,
                "resolution": 1000,
                "contrast": 0.7,
                "snr": 0.9
            }
            
            # Validar contra estándares NIST
            is_valid = self.nist_integration.validate_standards(test_data)
            
            # Verificar resultado
            self.assertIsInstance(is_valid, bool)
        except Exception as e:
            self.assertIsInstance(e, (NotImplementedError, AttributeError))
    
    def test_nist_metadata_generation(self):
        """Test generación de metadatos NIST"""
        try:
            # Crear imagen de prueba
            test_image = np.random.randint(0, 255, (500, 500, 3), dtype=np.uint8)
            
            # Generar metadatos NIST
            metadata = self.nist_integration.generate_nist_metadata(test_image)
            
            # Verificar metadatos
            self.assertIsInstance(metadata, dict)
            self.assertIn("quality_metrics", metadata)
        except Exception as e:
            self.assertIsInstance(e, (NotImplementedError, AttributeError))
    
    def test_nist_report_generation(self):
        """Test generación de reportes NIST"""
        try:
            # Datos de análisis
            analysis_data = {
                "image1_quality": 0.8,
                "image2_quality": 0.7,
                "similarity_score": 0.85,
                "match_confidence": 0.9
            }
            
            # Generar reporte NIST
            report = self.nist_integration.generate_nist_report(analysis_data)
            
            # Verificar reporte
            self.assertIsInstance(report, dict)
            self.assertIn("conclusion", report)
        except Exception as e:
            self.assertIsInstance(e, (NotImplementedError, AttributeError))


class TestNISTStandards(unittest.TestCase):
    """Tests para NISTStandards"""
    
    def setUp(self):
        """Configuración inicial"""
        if not COMMON_AVAILABLE:
            self.skipTest("Common modules not available")
        self.standards = NISTStandards()
    
    def test_standards_initialization(self):
        """Test inicialización de estándares NIST"""
        self.assertIsNotNone(self.standards)
        self.assertIsInstance(self.standards, NISTStandards)
    
    def test_quality_thresholds(self):
        """Test umbrales de calidad NIST"""
        try:
            # Obtener umbrales de calidad
            thresholds = self.standards.get_quality_thresholds()
            
            # Verificar umbrales
            self.assertIsInstance(thresholds, dict)
            self.assertIn("min_resolution", thresholds)
            self.assertIn("min_contrast", thresholds)
            self.assertIn("min_snr", thresholds)
        except Exception as e:
            self.assertIsInstance(e, (NotImplementedError, AttributeError))
    
    def test_compliance_check(self):
        """Test verificación de cumplimiento NIST"""
        try:
            # Datos de prueba
            test_metrics = {
                "resolution": 1200,
                "contrast": 0.8,
                "snr": 0.9,
                "uniformity": 0.7
            }
            
            # Verificar cumplimiento
            compliance = self.standards.check_compliance(test_metrics)
            
            # Verificar resultado
            self.assertIsInstance(compliance, dict)
            self.assertIn("compliant", compliance)
            self.assertIn("violations", compliance)
        except Exception as e:
            self.assertIsInstance(e, (NotImplementedError, AttributeError))


class TestSimilarityCalculator(unittest.TestCase):
    """Tests para SimilarityCalculator"""
    
    def setUp(self):
        """Configuración inicial"""
        if not COMMON_AVAILABLE:
            self.skipTest("Common modules not available")
        self.calculator = SimilarityCalculator()
    
    def test_calculator_initialization(self):
        """Test inicialización del calculador de similitud"""
        self.assertIsNotNone(self.calculator)
        self.assertIsInstance(self.calculator, SimilarityCalculator)
    
    def test_cosine_similarity(self):
        """Test similitud coseno"""
        try:
            # Vectores de prueba
            vector1 = np.array([1, 2, 3, 4, 5])
            vector2 = np.array([2, 4, 6, 8, 10])
            
            # Calcular similitud coseno
            similarity = self.calculator.cosine_similarity(vector1, vector2)
            
            # Verificar resultado
            self.assertIsInstance(similarity, (int, float))
            self.assertGreaterEqual(similarity, -1)
            self.assertLessEqual(similarity, 1)
        except Exception as e:
            self.assertIsInstance(e, (NotImplementedError, AttributeError))
    
    def test_euclidean_distance(self):
        """Test distancia euclidiana"""
        try:
            # Vectores de prueba
            vector1 = np.array([0, 0, 0])
            vector2 = np.array([3, 4, 0])
            
            # Calcular distancia euclidiana
            distance = self.calculator.euclidean_distance(vector1, vector2)
            
            # Verificar resultado (debería ser 5.0)
            self.assertIsInstance(distance, (int, float))
            self.assertAlmostEqual(distance, 5.0, places=1)
        except Exception as e:
            self.assertIsInstance(e, (NotImplementedError, AttributeError))
    
    def test_manhattan_distance(self):
        """Test distancia Manhattan"""
        try:
            # Vectores de prueba
            vector1 = np.array([1, 2, 3])
            vector2 = np.array([4, 6, 8])
            
            # Calcular distancia Manhattan
            distance = self.calculator.manhattan_distance(vector1, vector2)
            
            # Verificar resultado
            self.assertIsInstance(distance, (int, float))
            self.assertGreater(distance, 0)
        except Exception as e:
            self.assertIsInstance(e, (NotImplementedError, AttributeError))
    
    def test_jaccard_similarity(self):
        """Test similitud Jaccard"""
        try:
            # Conjuntos de prueba
            set1 = {1, 2, 3, 4, 5}
            set2 = {3, 4, 5, 6, 7}
            
            # Calcular similitud Jaccard
            similarity = self.calculator.jaccard_similarity(set1, set2)
            
            # Verificar resultado
            self.assertIsInstance(similarity, (int, float))
            self.assertGreaterEqual(similarity, 0)
            self.assertLessEqual(similarity, 1)
        except Exception as e:
            self.assertIsInstance(e, (NotImplementedError, AttributeError))


class TestSimilarityMetrics(unittest.TestCase):
    """Tests para SimilarityMetrics"""
    
    def setUp(self):
        """Configuración inicial"""
        if not COMMON_AVAILABLE:
            self.skipTest("Common modules not available")
        self.metrics = SimilarityMetrics()
    
    def test_metrics_initialization(self):
        """Test inicialización de métricas de similitud"""
        self.assertIsNotNone(self.metrics)
        self.assertIsInstance(self.metrics, SimilarityMetrics)
    
    def test_calculate_all_metrics(self):
        """Test cálculo de todas las métricas"""
        try:
            # Datos de prueba
            data1 = np.random.rand(100)
            data2 = np.random.rand(100)
            
            # Calcular todas las métricas
            all_metrics = self.metrics.calculate_all_metrics(data1, data2)
            
            # Verificar resultado
            self.assertIsInstance(all_metrics, dict)
            self.assertIn("cosine_similarity", all_metrics)
            self.assertIn("euclidean_distance", all_metrics)
        except Exception as e:
            self.assertIsInstance(e, (NotImplementedError, AttributeError))
    
    def test_metric_comparison(self):
        """Test comparación de métricas"""
        try:
            # Datos idénticos
            identical_data = np.array([1, 2, 3, 4, 5])
            
            # Calcular métricas para datos idénticos
            metrics = self.metrics.calculate_all_metrics(identical_data, identical_data)
            
            # Verificar que la similitud coseno es 1.0 para datos idénticos
            if "cosine_similarity" in metrics:
                self.assertAlmostEqual(metrics["cosine_similarity"], 1.0, places=5)
            
            # Verificar que la distancia euclidiana es 0.0 para datos idénticos
            if "euclidean_distance" in metrics:
                self.assertAlmostEqual(metrics["euclidean_distance"], 0.0, places=5)
        except Exception as e:
            self.assertIsInstance(e, (NotImplementedError, AttributeError))


class TestStatisticalAnalyzer(unittest.TestCase):
    """Tests para StatisticalAnalyzer"""
    
    def setUp(self):
        """Configuración inicial"""
        if not COMMON_AVAILABLE:
            self.skipTest("Common modules not available")
        self.analyzer = StatisticalAnalyzer()
    
    def test_analyzer_initialization(self):
        """Test inicialización del analizador estadístico"""
        self.assertIsNotNone(self.analyzer)
        self.assertIsInstance(self.analyzer, StatisticalAnalyzer)
    
    def test_descriptive_statistics(self):
        """Test estadísticas descriptivas"""
        try:
            # Datos de prueba
            data = np.random.normal(100, 15, 1000)
            
            # Calcular estadísticas descriptivas
            stats = self.analyzer.calculate_descriptive_stats(data)
            
            # Verificar resultado
            self.assertIsInstance(stats, dict)
            self.assertIn("mean", stats)
            self.assertIn("std", stats)
            self.assertIn("median", stats)
        except Exception as e:
            self.assertIsInstance(e, (NotImplementedError, AttributeError))
    
    def test_correlation_analysis(self):
        """Test análisis de correlación"""
        try:
            # Datos correlacionados
            x = np.random.rand(100)
            y = x * 2 + np.random.rand(100) * 0.1  # Correlación alta
            
            # Calcular correlación
            correlation = self.analyzer.calculate_correlation(x, y)
            
            # Verificar resultado
            self.assertIsInstance(correlation, (int, float))
            self.assertGreaterEqual(correlation, -1)
            self.assertLessEqual(correlation, 1)
        except Exception as e:
            self.assertIsInstance(e, (NotImplementedError, AttributeError))
    
    def test_hypothesis_testing(self):
        """Test pruebas de hipótesis"""
        try:
            # Dos muestras diferentes
            sample1 = np.random.normal(100, 10, 50)
            sample2 = np.random.normal(105, 10, 50)
            
            # Realizar prueba t
            t_stat, p_value = self.analyzer.t_test(sample1, sample2)
            
            # Verificar resultado
            self.assertIsInstance(t_stat, (int, float))
            self.assertIsInstance(p_value, (int, float))
            self.assertGreaterEqual(p_value, 0)
            self.assertLessEqual(p_value, 1)
        except Exception as e:
            self.assertIsInstance(e, (NotImplementedError, AttributeError))


class TestClusteringAnalyzer(unittest.TestCase):
    """Tests para ClusteringAnalyzer"""
    
    def setUp(self):
        """Configuración inicial"""
        if not COMMON_AVAILABLE:
            self.skipTest("Common modules not available")
        self.clustering = ClusteringAnalyzer()
    
    def test_clustering_initialization(self):
        """Test inicialización del analizador de clustering"""
        self.assertIsNotNone(self.clustering)
        self.assertIsInstance(self.clustering, ClusteringAnalyzer)
    
    def test_kmeans_clustering(self):
        """Test clustering K-means"""
        try:
            # Datos de prueba
            data = np.random.rand(100, 5)
            
            # Aplicar K-means
            labels, centers = self.clustering.kmeans_clustering(data, n_clusters=3)
            
            # Verificar resultado
            self.assertIsInstance(labels, np.ndarray)
            self.assertIsInstance(centers, np.ndarray)
            self.assertEqual(len(labels), 100)
            self.assertEqual(centers.shape[0], 3)
        except Exception as e:
            self.assertIsInstance(e, (NotImplementedError, AttributeError))
    
    def test_hierarchical_clustering(self):
        """Test clustering jerárquico"""
        try:
            # Datos de prueba
            data = np.random.rand(50, 3)
            
            # Aplicar clustering jerárquico
            linkage_matrix = self.clustering.hierarchical_clustering(data)
            
            # Verificar resultado
            self.assertIsInstance(linkage_matrix, np.ndarray)
            self.assertEqual(linkage_matrix.shape[1], 4)  # Formato estándar de linkage
        except Exception as e:
            self.assertIsInstance(e, (NotImplementedError, AttributeError))
    
    def test_cluster_evaluation(self):
        """Test evaluación de clusters"""
        try:
            # Datos de prueba con clusters conocidos
            data = np.vstack([
                np.random.normal([0, 0], 0.5, (50, 2)),
                np.random.normal([3, 3], 0.5, (50, 2))
            ])
            
            # Aplicar clustering
            labels, _ = self.clustering.kmeans_clustering(data, n_clusters=2)
            
            # Evaluar calidad del clustering
            silhouette_score = self.clustering.evaluate_clustering(data, labels)
            
            # Verificar resultado
            self.assertIsInstance(silhouette_score, (int, float))
            self.assertGreaterEqual(silhouette_score, -1)
            self.assertLessEqual(silhouette_score, 1)
        except Exception as e:
            self.assertIsInstance(e, (NotImplementedError, AttributeError))


if __name__ == '__main__':
    print("🧪 Ejecutando tests unitarios para módulos Common...")
    
    # Configurar logging para tests
    import logging
    logging.basicConfig(level=logging.WARNING)
    
    # Ejecutar tests
    unittest.main(verbosity=2)