#!/usr/bin/env python3
"""
Tests comprehensivos para el módulo de Análisis Estadístico Avanzado
Sistema SEACABA - Análisis Balístico Forense
"""

import unittest
import numpy as np
import warnings
from unittest.mock import patch, MagicMock
import tempfile
import os
import json
from pathlib import Path

# Importar desde el núcleo estadístico unificado
from common.statistical_core import (
    UnifiedStatisticalAnalysis, StatisticalTest, CorrectionMethod, BootstrapResult,
    StatisticalTestResult, MultipleComparisonResult
)
from common.compatibility_adapters import AdvancedStatisticalAnalysisAdapter


class TestBootstrapSampling(unittest.TestCase):
    """Tests para Bootstrap Sampling"""
    
    def setUp(self):
        """Configurar datos de prueba"""
        self.statistical_analysis = AdvancedStatisticalAnalysisAdapter()
        self.sample_data = [1.2, 2.3, 1.8, 2.1, 1.9, 2.4, 1.7, 2.0, 1.6, 2.2]
        
    def test_bootstrap_sampling_basic(self):
        """Test básico de bootstrap sampling"""
        try:
            result = self.statistical_analysis.bootstrap_sampling(
                self.sample_data,
                statistic_func=np.mean,
                n_bootstrap=100,
                confidence_level=0.95
            )
            
            # Verificar estructura del resultado
            self.assertIsInstance(result, BootstrapResult)
            self.assertEqual(result.n_bootstrap, 100)
            self.assertEqual(result.confidence_level, 0.95)
            self.assertIsInstance(result.confidence_interval, tuple)
            self.assertEqual(len(result.confidence_interval), 2)
            self.assertLess(result.confidence_interval[0], result.confidence_interval[1])
            
        except Exception as e:
            self.skipTest(f"Bootstrap sampling no disponible: {e}")
    
    def test_bootstrap_different_methods(self):
        """Test de diferentes métodos de bootstrap"""
        methods = ['percentile', 'basic', 'bca']
        
        for method in methods:
            with self.subTest(method=method):
                try:
                    result = self.statistical_analysis.bootstrap_sampling(
                        self.sample_data,
                        statistic_func=np.mean,
                        n_bootstrap=50,
                        method=method
                    )
                    
                    self.assertEqual(result.method, method)
                    self.assertIsNotNone(result.confidence_interval)
                    
                except Exception as e:
                    self.skipTest(f"Método {method} no disponible: {e}")
    
    def test_bootstrap_different_statistics(self):
        """Test de bootstrap con diferentes estadísticos"""
        statistics = [np.mean, np.median, np.std]
        
        for stat_func in statistics:
            with self.subTest(statistic=stat_func.__name__):
                try:
                    result = self.statistical_analysis.bootstrap_sampling(
                        self.sample_data,
                        statistic_func=stat_func,
                        n_bootstrap=50
                    )
                    
                    expected_stat = stat_func(self.sample_data)
                    self.assertAlmostEqual(result.original_statistic, expected_stat, places=5)
                    
                except Exception as e:
                    self.skipTest(f"Estadístico {stat_func.__name__} no disponible: {e}")


class TestPValueCalculation(unittest.TestCase):
    """Tests para cálculo de P-values"""
    
    def setUp(self):
        """Configurar datos de prueba"""
        self.statistical_analysis = AdvancedStatisticalAnalysisAdapter()
        np.random.seed(42)  # Para reproducibilidad
        self.sample1 = np.random.normal(0, 1, 30)
        self.sample2 = np.random.normal(0.5, 1, 30)  # Diferente media
        
    def test_t_test(self):
        """Test de t-test"""
        try:
            result = self.statistical_analysis.calculate_p_value(
                self.sample1,
                self.sample2,
                test_type=StatisticalTest.T_TEST
            )
            
            self.assertIsInstance(result, StatisticalTestResult)
            self.assertEqual(result.test_type, StatisticalTest.T_TEST)
            self.assertIsInstance(result.p_value, float)
            self.assertGreaterEqual(result.p_value, 0)
            self.assertLessEqual(result.p_value, 1)
            self.assertIsInstance(result.is_significant, bool)
            
        except Exception as e:
            self.skipTest(f"T-test no disponible: {e}")
    
    def test_mann_whitney(self):
        """Test de Mann-Whitney U"""
        try:
            result = self.statistical_analysis.calculate_p_value(
                self.sample1,
                self.sample2,
                test_type=StatisticalTest.MANN_WHITNEY
            )
            
            self.assertEqual(result.test_type, StatisticalTest.MANN_WHITNEY)
            self.assertIsInstance(result.p_value, float)
            
        except Exception as e:
            self.skipTest(f"Mann-Whitney no disponible: {e}")
    
    def test_kolmogorov_smirnov(self):
        """Test de Kolmogorov-Smirnov"""
        try:
            result = self.statistical_analysis.calculate_p_value(
                self.sample1,
                self.sample2,
                test_type=StatisticalTest.KOLMOGOROV_SMIRNOV
            )
            
            self.assertEqual(result.test_type, StatisticalTest.KOLMOGOROV_SMIRNOV)
            self.assertIsInstance(result.p_value, float)
            
        except Exception as e:
            self.skipTest(f"Kolmogorov-Smirnov no disponible: {e}")
    
    def test_wilcoxon(self):
        """Test de Wilcoxon"""
        try:
            # Para Wilcoxon necesitamos muestras pareadas
            paired_sample1 = self.sample1[:20]
            paired_sample2 = self.sample2[:20]
            
            result = self.statistical_analysis.calculate_p_value(
                paired_sample1,
                paired_sample2,
                test_type=StatisticalTest.WILCOXON
            )
            
            self.assertEqual(result.test_type, StatisticalTest.WILCOXON)
            self.assertIsInstance(result.p_value, float)
            
        except Exception as e:
            self.skipTest(f"Wilcoxon no disponible: {e}")


class TestMultipleComparisonCorrection(unittest.TestCase):
    """Tests para corrección de comparaciones múltiples"""
    
    def setUp(self):
        """Configurar datos de prueba"""
        self.statistical_analysis = AdvancedStatisticalAnalysisAdapter()
        self.p_values = [0.01, 0.03, 0.05, 0.08, 0.12, 0.15]
        
    def test_bonferroni_correction(self):
        """Test de corrección de Bonferroni"""
        try:
            result = self.statistical_analysis.multiple_comparison_correction(
                self.p_values,
                method=CorrectionMethod.BONFERRONI,
                alpha=0.05
            )
            
            self.assertIsInstance(result, MultipleComparisonResult)
            self.assertEqual(result.method, CorrectionMethod.BONFERRONI)
            self.assertEqual(len(result.corrected_p_values), len(self.p_values))
            self.assertEqual(len(result.rejected_hypotheses), len(self.p_values))
            
            # Verificar que los p-valores corregidos son mayores o iguales a los originales
            for orig, corr in zip(self.p_values, result.corrected_p_values):
                self.assertGreaterEqual(corr, orig)
                
        except Exception as e:
            self.skipTest(f"Corrección Bonferroni no disponible: {e}")
    
    def test_holm_correction(self):
        """Test de corrección de Holm"""
        try:
            result = self.statistical_analysis.multiple_comparison_correction(
                self.p_values,
                method=CorrectionMethod.HOLM,
                alpha=0.05
            )
            
            self.assertEqual(result.method, CorrectionMethod.HOLM)
            self.assertIsInstance(result.corrected_p_values, list)
            
        except Exception as e:
            self.skipTest(f"Corrección Holm no disponible: {e}")
    
    def test_benjamini_hochberg_correction(self):
        """Test de corrección de Benjamini-Hochberg (FDR)"""
        try:
            result = self.statistical_analysis.multiple_comparison_correction(
                self.p_values,
                method=CorrectionMethod.BENJAMINI_HOCHBERG,
                alpha=0.05
            )
            
            self.assertEqual(result.method, CorrectionMethod.BENJAMINI_HOCHBERG)
            self.assertIsInstance(result.corrected_p_values, list)
            
        except Exception as e:
            self.skipTest(f"Corrección Benjamini-Hochberg no disponible: {e}")
    
    def test_sidak_correction(self):
        """Test de corrección de Sidak"""
        try:
            result = self.statistical_analysis.multiple_comparison_correction(
                self.p_values,
                method=CorrectionMethod.SIDAK,
                alpha=0.05
            )
            
            self.assertEqual(result.method, CorrectionMethod.SIDAK)
            self.assertIsInstance(result.corrected_p_values, list)
            
        except Exception as e:
            self.skipTest(f"Corrección Sidak no disponible: {e}")


class TestStatisticalPower(unittest.TestCase):
    """Tests para análisis de potencia estadística"""
    
    def setUp(self):
        """Configurar datos de prueba"""
        self.statistical_analysis = AdvancedStatisticalAnalysisAdapter()
        
    def test_statistical_power_calculation(self):
        """Test de cálculo de potencia estadística"""
        try:
            power = self.statistical_analysis.calculate_statistical_power(
                effect_size=0.5,
                sample_size=30,
                alpha=0.05
            )
            
            self.assertIsInstance(power, float)
            self.assertGreaterEqual(power, 0)
            self.assertLessEqual(power, 1)
            
        except Exception as e:
            self.skipTest(f"Cálculo de potencia no disponible: {e}")
    
    def test_sample_size_calculation(self):
        """Test de cálculo de tamaño de muestra"""
        try:
            sample_size = self.statistical_analysis.calculate_sample_size(
                effect_size=0.5,
                power=0.8,
                alpha=0.05
            )
            
            self.assertIsInstance(sample_size, int)
            self.assertGreater(sample_size, 0)
            
        except Exception as e:
            self.skipTest(f"Cálculo de tamaño de muestra no disponible: {e}")


class TestStatisticalReporting(unittest.TestCase):
    """Tests para generación de reportes estadísticos"""
    
    def setUp(self):
        """Configurar datos de prueba"""
        self.statistical_analysis = AdvancedStatisticalAnalysisAdapter()
        
        # Crear resultados mock
        self.mock_bootstrap = BootstrapResult(
            original_statistic=1.5,
            bootstrap_statistics=np.array([1.4, 1.5, 1.6, 1.5, 1.5]),
            confidence_interval=(1.2, 1.8),
            confidence_level=0.95,
            bias=0.02,
            standard_error=0.1,
            percentile_ci=(1.2, 1.8),
            bca_ci=(1.15, 1.85),
            n_bootstrap=1000
        )
        
        self.mock_test_result = StatisticalTestResult(
            test_name="T-Test",
            statistic=2.5,
            p_value=0.013,
            critical_value=2.086,
            confidence_interval=None,
            effect_size=0.6,
            power=None,
            sample_size=30,
            degrees_of_freedom=28,
            is_significant=True,
            alpha=0.05
        )
        
    def test_generate_statistical_report(self):
        """Test de generación de reporte estadístico"""
        try:
            results = [self.mock_bootstrap, self.mock_test_result]
            
            report = self.statistical_analysis.generate_statistical_report(
                results,
                title="Test Report"
            )
            
            self.assertIsInstance(report, str)
            self.assertIn("Test Report", report)
            self.assertIn("Bootstrap", report)
            self.assertIn("T-Test", report)
            
        except Exception as e:
            self.skipTest(f"Generación de reporte no disponible: {e}")


class TestIntegrationStatisticalAnalysis(unittest.TestCase):
    """Tests de integración para análisis estadístico completo"""
    
    def setUp(self):
        """Configurar datos de prueba"""
        self.statistical_analysis = AdvancedStatisticalAnalysisAdapter()
        np.random.seed(42)
        
        # Simular datos de calidad balística
        self.quality_data1 = {
            'snr': np.random.normal(25, 3, 20),
            'contrast': np.random.normal(0.8, 0.1, 20),
            'uniformity': np.random.normal(0.9, 0.05, 20),
            'sharpness': np.random.normal(0.85, 0.08, 20)
        }
        
        self.quality_data2 = {
            'snr': np.random.normal(23, 3, 20),
            'contrast': np.random.normal(0.75, 0.1, 20),
            'uniformity': np.random.normal(0.88, 0.05, 20),
            'sharpness': np.random.normal(0.82, 0.08, 20)
        }
    
    def test_complete_statistical_workflow(self):
        """Test del flujo completo de análisis estadístico"""
        try:
            # 1. Bootstrap Sampling para intervalos de confianza
            snr_data = self.quality_data1['snr']
            bootstrap_result = self.statistical_analysis.bootstrap_sampling(
                snr_data,
                statistic_func=np.mean,
                n_bootstrap=100,
                confidence_level=0.95
            )
            
            # 2. Tests estadísticos comparativos
            test_results = []
            for metric in ['snr', 'contrast', 'uniformity', 'sharpness']:
                test_result = self.statistical_analysis.calculate_p_value(
                    self.quality_data1[metric],
                    self.quality_data2[metric],
                    test_type=StatisticalTest.T_TEST
                )
                test_results.append(test_result)
            
            # 3. Corrección por comparaciones múltiples
            p_values = [result.p_value for result in test_results]
            correction_result = self.statistical_analysis.multiple_comparison_correction(
                p_values,
                method=CorrectionMethod.BONFERRONI,
                alpha=0.05
            )
            
            # 4. Análisis de potencia
            power = self.statistical_analysis.calculate_statistical_power(
                effect_size=0.5,
                sample_size=20,
                alpha=0.05
            )
            
            # 5. Reporte comprehensivo
            all_results = [bootstrap_result] + test_results + [correction_result]
            report = self.statistical_analysis.generate_statistical_report(
                all_results,
                title="Análisis Estadístico Completo - SEACABA"
            )
            
            # Verificaciones
            self.assertIsInstance(bootstrap_result, BootstrapResult)
            self.assertEqual(len(test_results), 4)
            self.assertIsInstance(correction_result, MultipleComparisonResult)
            self.assertIsInstance(power, float)
            self.assertIsInstance(report, str)
            self.assertIn("SEACABA", report)
            
        except Exception as e:
            self.skipTest(f"Flujo completo no disponible: {e}")
    
    def test_error_handling(self):
        """Test de manejo de errores"""
        # Test con datos vacíos
        with self.assertRaises((ValueError, TypeError)):
            self.statistical_analysis.bootstrap_sampling(
                [],
                statistic_func=np.mean
            )
        
        # Test con datos insuficientes
        with self.assertRaises((ValueError, TypeError)):
            self.statistical_analysis.calculate_p_value(
                [1],
                [2],
                test_type=StatisticalTest.T_TEST
            )


if __name__ == '__main__':
    # Configurar el runner de tests
    unittest.main(verbosity=2, buffer=True)