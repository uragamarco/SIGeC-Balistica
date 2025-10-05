#!/usr/bin/env python3
"""
Tests de Validación de Compatibilidad para Refactorización Estadística
Sistema SEACABA - Análisis Balístico Forense

Este módulo valida que los adaptadores de compatibilidad mantengan
equivalencia funcional exacta con las implementaciones originales,
preservando la trazabilidad NIST y la precisión requerida.
"""

import unittest
import numpy as np
import sys
import os
from pathlib import Path
from unittest.mock import Mock, patch
import warnings

# Agregar el directorio raíz al path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importar implementaciones originales
from nist_standards.statistical_analysis import (
    AdvancedStatisticalAnalysis as OriginalAdvancedStatisticalAnalysis,
    StatisticalTest,
    CorrectionMethod,
    BootstrapResult,
    StatisticalTestResult,
    MultipleComparisonResult
)

from matching.bootstrap_similarity import (
    BootstrapSimilarityAnalyzer as OriginalBootstrapSimilarityAnalyzer,
    SimilarityBootstrapResult,
    MatchingBootstrapConfig
)

from image_processing.statistical_analyzer import (
    StatisticalAnalyzer as OriginalStatisticalAnalyzer
)

# Importar adaptadores de compatibilidad
from common.compatibility_adapters import (
    AdvancedStatisticalAnalysisAdapter,
    BootstrapSimilarityAnalyzerAdapter,
    StatisticalAnalyzerAdapter,
    get_adapter
)

from common.statistical_core import UnifiedStatisticalAnalysis


class TestAdvancedStatisticalAnalysisCompatibility(unittest.TestCase):
    """
    Validar equivalencia funcional entre AdvancedStatisticalAnalysis original
    y AdvancedStatisticalAnalysisAdapter
    """
    
    def setUp(self):
        """Configurar instancias para comparación"""
        self.original = OriginalAdvancedStatisticalAnalysis()
        self.adapter = AdvancedStatisticalAnalysisAdapter()
        
        # Datos de prueba estándar
        self.sample_data = [1.2, 2.3, 1.8, 2.1, 1.9, 2.4, 1.7, 2.0, 1.6, 2.2]
        self.sample_data2 = [2.1, 2.8, 2.3, 2.6, 2.4, 2.9, 2.2, 2.5, 2.1, 2.7]
        
        # Configurar semilla para reproducibilidad
        np.random.seed(42)
    
    def test_bootstrap_sampling_equivalence(self):
        """Validar equivalencia en bootstrap sampling"""
        # Configurar parámetros idénticos
        params = {
            'data': self.sample_data,
            'statistic_func': np.mean,
            'n_bootstrap': 100,
            'confidence_level': 0.95,
            'method': 'percentile'
        }
        
        # Ejecutar con ambas implementaciones
        np.random.seed(42)  # Reset seed
        original_result = self.original.bootstrap_sampling(**params)
        
        np.random.seed(42)  # Reset seed
        adapter_result = self.adapter.bootstrap_sampling(**params)
        
        # Validar estructura de resultados
        self.assertIsInstance(original_result, BootstrapResult)
        self.assertIsInstance(adapter_result, BootstrapResult)
        
        # Validar equivalencia de valores críticos
        self.assertEqual(original_result.n_bootstrap, adapter_result.n_bootstrap)
        self.assertEqual(original_result.confidence_level, adapter_result.confidence_level)
        
        # Validar precisión estadística (tolerancia NIST)
        np.testing.assert_allclose(
            original_result.original_statistic,
            adapter_result.original_statistic,
            rtol=1e-10,  # Precisión NIST requerida
            err_msg="Bootstrap statistic values must be identical"
        )
        
        np.testing.assert_allclose(
            original_result.confidence_interval,
            adapter_result.confidence_interval,
            rtol=1e-8,  # Tolerancia para intervalos de confianza
            err_msg="Bootstrap confidence intervals must be equivalent"
        )
    
    def test_p_value_calculation_equivalence(self):
        """Validar equivalencia en cálculo de p-values"""
        test_cases = [
            (StatisticalTest.T_TEST, self.sample_data, self.sample_data2),
            (StatisticalTest.MANN_WHITNEY, self.sample_data, self.sample_data2),
            (StatisticalTest.KOLMOGOROV_SMIRNOV, self.sample_data, self.sample_data2),
        ]
        
        for test_type, data1, data2 in test_cases:
            with self.subTest(test=test_type):
                # Ejecutar con ambas implementaciones
                original_result = self.original.calculate_p_value(data1, data2, test_type)
                adapter_result = self.adapter.calculate_p_value(data1, data2, test_type)
                
                # Validar estructura
                self.assertIsInstance(original_result, StatisticalTestResult)
                self.assertIsInstance(adapter_result, StatisticalTestResult)
                
                # Validar equivalencia exacta de p-values (crítico para NIST)
                np.testing.assert_allclose(
                    original_result.p_value,
                    adapter_result.p_value,
                    rtol=1e-12,  # Máxima precisión para p-values
                    err_msg=f"P-values must be identical for {test_type}"
                )
                
                # Validar estadísticos de prueba
                np.testing.assert_allclose(
                    original_result.statistic,
                    adapter_result.statistic,
                    rtol=1e-10,
                    err_msg=f"Test statistics must be identical for {test_type}"
                )
    
    def test_multiple_comparison_correction_equivalence(self):
        """Validar equivalencia en corrección de múltiples comparaciones"""
        from nist_standards.statistical_analysis import CorrectionMethod
        
        # Datos de prueba
        p_values = [0.01, 0.05, 0.1, 0.2, 0.3]
        
        # Test original
        original_result = self.original.multiple_comparison_correction(
            p_values, 
            method=CorrectionMethod.BONFERRONI
        )
        
        # Test adapter
        adapter_result = self.adapter.multiple_comparison_correction(
            p_values, 
            method=CorrectionMethod.BONFERRONI
        )
        
        # Validar equivalencia
        np.testing.assert_array_equal(
            original_result.corrected_p_values,
            adapter_result.corrected_p_values,
            err_msg="Corrected p-values must be identical"
        )
        
        np.testing.assert_array_equal(
            original_result.rejected_hypotheses,
            adapter_result.rejected_hypotheses,
            err_msg="Rejected hypotheses must be identical"
        )


class TestBootstrapSimilarityAnalyzerCompatibility(unittest.TestCase):
    """
    Validar equivalencia funcional entre BootstrapSimilarityAnalyzer original
    y BootstrapSimilarityAnalyzerAdapter
    """
    
    def setUp(self):
        """Configurar instancias para comparación"""
        self.config = MatchingBootstrapConfig(
            n_bootstrap=50,
            confidence_level=0.95,
            method='percentile'
        )
        
        self.original = OriginalBootstrapSimilarityAnalyzer(self.config)
        self.adapter = BootstrapSimilarityAnalyzerAdapter(self.config)
        
        # Datos de similitud simulados
        self.similarity_scores = np.array([0.85, 0.92, 0.78, 0.88, 0.91, 0.83, 0.87, 0.89])
        
        np.random.seed(42)
    
    def test_bootstrap_similarity_equivalence(self):
        """Validar equivalencia en análisis bootstrap de similitud"""
        # Crear datos de matches de prueba
        matches_data = [
            {'distance': 0.1, 'kp1_idx': 0, 'kp2_idx': 0},
            {'distance': 0.2, 'kp1_idx': 1, 'kp2_idx': 1},
            {'distance': 0.15, 'kp1_idx': 2, 'kp2_idx': 2},
            {'distance': 0.3, 'kp1_idx': 3, 'kp2_idx': 3},
            {'distance': 0.25, 'kp1_idx': 4, 'kp2_idx': 4}
        ]
        
        def similarity_func(data):
            return 75.0  # Score fijo para prueba
        
        # Ejecutar con ambas implementaciones
        np.random.seed(42)
        original_result = self.original.bootstrap_similarity_confidence(matches_data, similarity_func)
        
        np.random.seed(42)
        adapter_result = self.adapter.bootstrap_similarity_confidence(matches_data, similarity_func)
        
        # Validar estructura
        self.assertIsInstance(original_result, SimilarityBootstrapResult)
        self.assertIsInstance(adapter_result, SimilarityBootstrapResult)
        
        # Validar equivalencia de métricas críticas
        np.testing.assert_allclose(
            original_result.similarity_score,
            adapter_result.similarity_score,
            rtol=1e-10,
            err_msg="Mean similarity must be identical"
        )
        
        np.testing.assert_allclose(
            original_result.confidence_interval,
            adapter_result.confidence_interval,
            rtol=1e-8,
            err_msg="Confidence intervals must be equivalent"
        )
        
        np.testing.assert_allclose(
            original_result.bootstrap_scores,
            adapter_result.bootstrap_scores,
            rtol=1e-8,
            err_msg="Bootstrap distributions must be equivalent"
        )


class TestStatisticalAnalyzerCompatibility(unittest.TestCase):
    """
    Validar equivalencia funcional entre StatisticalAnalyzer original
    y StatisticalAnalyzerAdapter
    """
    
    def setUp(self):
        """Configurar instancias para comparación"""
        self.original = OriginalStatisticalAnalyzer()
        self.adapter = StatisticalAnalyzerAdapter()
        
        # Datos de prueba para análisis estadístico
        np.random.seed(42)
        self.features = np.random.randn(100, 10)  # 100 muestras, 10 características
        self.sample1 = np.random.normal(0, 1, 50)
        self.sample2 = np.random.normal(0.5, 1, 50)
    
    def test_pca_analysis_equivalence(self):
        """Validar equivalencia en análisis PCA"""
        # Convertir features numpy a formato de lista de diccionarios
        features_data = []
        for i in range(self.features.shape[0]):
            feature_dict = {f'feature_{j}': self.features[i, j] for j in range(self.features.shape[1])}
            features_data.append(feature_dict)
        
        # Ejecutar con ambas implementaciones
        original_result = self.original.perform_pca_analysis(features_data, n_components=5)
        adapter_result = self.adapter.perform_pca_analysis(features_data, n_components=5)
        
        # Validar equivalencia de componentes principales
        # Nota: Los signos pueden diferir, pero los valores absolutos deben ser iguales
        if 'components' in original_result and 'components' in adapter_result:
            np.testing.assert_allclose(
                np.abs(original_result['components']),
                np.abs(adapter_result['components']),
                rtol=1e-8,
                err_msg="PCA components must be equivalent"
            )
        
        # Validar varianza explicada (debe ser idéntica)
        if 'explained_variance_ratio' in original_result and 'explained_variance_ratio' in adapter_result:
            np.testing.assert_allclose(
                original_result['explained_variance_ratio'],
                adapter_result['explained_variance_ratio'],
                rtol=1e-10,
                err_msg="Explained variance ratios must be identical"
            )
    
    def test_significance_tests_equivalence(self):
        """Validar equivalencia en tests de significancia"""
        # Usar el método unificado perform_significance_tests
        original_result = self.original.perform_significance_tests(self.sample1, self.sample2)
        adapter_result = self.adapter.perform_significance_tests(self.sample1, self.sample2)
        
        # Validar que ambos resultados tengan la misma estructura
        self.assertEqual(set(original_result.keys()), set(adapter_result.keys()))
        
        # Validar equivalencia de p-values para cada test
        for test_name in original_result.keys():
            if isinstance(original_result[test_name], dict) and 'p_value' in original_result[test_name]:
                if isinstance(adapter_result[test_name], dict) and 'p_value' in adapter_result[test_name]:
                    np.testing.assert_allclose(
                        original_result[test_name]['p_value'],
                        adapter_result[test_name]['p_value'],
                        rtol=1e-10,
                        err_msg=f"P-values must be identical for {test_name}"
                    )


class TestNISTTraceabilityPreservation(unittest.TestCase):
    """
    Validar que la trazabilidad NIST se preserve en todos los adaptadores
    """
    
    def test_nist_metadata_preservation(self):
        """Validar que los metadatos NIST se preserven"""
        adapter = AdvancedStatisticalAnalysisAdapter()
        
        # Verificar que el adaptador mantenga referencias a implementación original
        self.assertTrue(hasattr(adapter, '_original'))
        self.assertIsInstance(adapter._original, OriginalAdvancedStatisticalAnalysis)
        
        # Verificar trazabilidad en resultados
        data = np.random.randn(100)
        result = adapter.bootstrap_sampling(data, np.mean, n_bootstrap=100)
        
        # Validar que el resultado mantiene estructura NIST
        self.assertTrue(hasattr(result, 'n_bootstrap'))
        self.assertTrue(hasattr(result, 'confidence_level'))
        self.assertTrue(hasattr(result, 'original_statistic'))
        self.assertTrue(hasattr(result, 'confidence_interval'))
        self.assertTrue(hasattr(result, 'bootstrap_statistics'))
        self.assertTrue(hasattr(result, 'bias'))
        self.assertTrue(hasattr(result, 'standard_error'))
        
        # Validar preservación de metadatos NIST en p-values
        data1 = np.random.randn(50)
        data2 = np.random.randn(50)
        
        p_result = adapter.calculate_p_value(data1, data2)
        
        # Validar metadatos NIST requeridos
        self.assertTrue(hasattr(p_result, 'test_name'))
        self.assertTrue(hasattr(p_result, 'statistic'))
        self.assertTrue(hasattr(p_result, 'p_value'))
        self.assertTrue(hasattr(p_result, 'is_significant'))
        self.assertTrue(hasattr(p_result, 'alpha'))
        self.assertTrue(hasattr(p_result, 'sample_size'))
        
        # Validar tipos (convertir numpy bool a Python bool)
        self.assertIsInstance(p_result.test_name, str)
        self.assertIsInstance(p_result.statistic, (int, float))
        self.assertIsInstance(p_result.p_value, (int, float))
        self.assertIsInstance(bool(p_result.is_significant), bool)
        self.assertIsInstance(p_result.alpha, (int, float))
        self.assertIsInstance(p_result.sample_size, int)
    
    def test_chain_of_custody_preservation(self):
        """Validar que la cadena de custodia se preserve"""
        # Verificar que los adaptadores mantengan información de origen
        adapters = [
            AdvancedStatisticalAnalysisAdapter(),
            BootstrapSimilarityAnalyzerAdapter(MatchingBootstrapConfig()),
            StatisticalAnalyzerAdapter()
        ]
        
        for adapter in adapters:
            # Cada adaptador debe tener referencia a la implementación original
            self.assertTrue(hasattr(adapter, '_original'))
            
            # La implementación original debe existir y ser del tipo correcto
            original = adapter._original
            self.assertIsNotNone(original)
            
            # Verificar que el adaptador preserva la trazabilidad según su tipo
            if isinstance(adapter, AdvancedStatisticalAnalysisAdapter):
                # Solo este adaptador tiene random_state
                self.assertTrue(hasattr(adapter, 'random_state'))
            elif isinstance(adapter, BootstrapSimilarityAnalyzerAdapter):
                # Este adaptador tiene config
                self.assertTrue(hasattr(adapter, 'config'))
            elif isinstance(adapter, StatisticalAnalyzerAdapter):
                # Este adaptador tiene _original
                self.assertTrue(hasattr(adapter, '_original'))


class TestAdapterFactoryFunctionality(unittest.TestCase):
    """
    Validar funcionalidad de la función factory get_adapter
    """
    
    def test_get_adapter_functionality(self):
        """Validar que get_adapter retorne adaptadores correctos"""
        # Test para AdvancedStatisticalAnalysis
        adapter1 = get_adapter('AdvancedStatisticalAnalysis')
        self.assertIsInstance(adapter1, AdvancedStatisticalAnalysisAdapter)
        
        # Test para BootstrapSimilarityAnalyzer
        config = MatchingBootstrapConfig()
        adapter2 = get_adapter('BootstrapSimilarityAnalyzer', config=config)
        self.assertIsInstance(adapter2, BootstrapSimilarityAnalyzerAdapter)
        
        # Test para StatisticalAnalyzer
        adapter3 = get_adapter('StatisticalAnalyzer')
        self.assertIsInstance(adapter3, StatisticalAnalyzerAdapter)
        
        # Test para adaptador no existente
        with self.assertRaises(ValueError):
            get_adapter('NonExistentAnalyzer')


if __name__ == '__main__':
    # Configurar warnings para detectar problemas de precisión
    warnings.filterwarnings('error', category=RuntimeWarning)
    
    # Ejecutar tests con máximo detalle
    unittest.main(verbosity=2, buffer=True)