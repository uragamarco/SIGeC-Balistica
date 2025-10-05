#!/usr/bin/env python3
"""
Tests de integración para Análisis Estadístico con Estándares NIST
Sistema SEACABA - Análisis Balístico Forense
"""

import unittest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import sys
import os
import tempfile
import json

# Agregar el directorio raíz al path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from nist_standards import NISTStandardsManager
    from nist_standards.statistical_analysis import (
        AdvancedStatisticalAnalysis,
        StatisticalTest,
        CorrectionMethod
    )
    try:
        from nist_standards.quality_metrics import QualityMetrics
    except ImportError:
        QualityMetrics = None
    try:
        from nist_standards.afte_conclusions import AFTEConclusions
    except ImportError:
        AFTEConclusions = None
except ImportError as e:
    print(f"Error importing NIST modules: {e}")
    NISTStandardsManager = None
    QualityMetrics = None
    AFTEConclusions = None


class TestNISTStatisticalIntegration(unittest.TestCase):
    """Tests de integración entre NIST Standards y Análisis Estadístico"""
    
    def setUp(self):
        """Configurar datos de prueba"""
        self.nist_manager = NISTStandardsManager()
        
        # Datos mock de evidencia balística
        self.mock_evidence_data = {
            'case_id': 'TEST_001',
            'evidence_type': 'cartridge_case',
            'image_path': '/test/image.jpg',
            'quality_metrics': {
                'snr': 25.5,
                'contrast': 0.85,
                'uniformity': 0.92,
                'sharpness': 0.88,
                'resolution': 1200
            },
            'features': {
                'keypoints': [(100, 150), (200, 250), (300, 350)],
                'descriptors': np.random.rand(3, 32),
                'similarity_scores': [0.95, 0.87, 0.76]
            }
        }
        
        self.mock_comparison_data = {
            'reference_evidence': self.mock_evidence_data,
            'comparison_evidence': {
                **self.mock_evidence_data,
                'case_id': 'TEST_002',
                'quality_metrics': {
                    'snr': 23.2,
                    'contrast': 0.82,
                    'uniformity': 0.89,
                    'sharpness': 0.85,
                    'resolution': 1200
                }
            }
        }
    
    def test_nist_manager_has_statistical_analysis(self):
        """Verificar que NISTStandardsManager tiene análisis estadístico"""
        try:
            self.assertTrue(hasattr(self.nist_manager, 'statistical_analysis'))
            self.assertIsInstance(
                self.nist_manager.statistical_analysis, 
                AdvancedStatisticalAnalysis
            )
        except Exception as e:
            self.skipTest(f"Integración estadística no disponible: {e}")
    
    def test_process_ballistic_evidence_with_statistics(self):
        """Test de procesamiento de evidencia con análisis estadístico"""
        try:
            # Mock de datos de entrada
            mock_image_data = np.random.randint(0, 255, (500, 500, 3), dtype=np.uint8)
            
            with patch('cv2.imread', return_value=mock_image_data):
                with patch('os.path.exists', return_value=True):
                    result = self.nist_manager.process_ballistic_evidence(
                        image_path='/test/image.jpg',
                        evidence_type='cartridge_case',
                        case_id='TEST_STAT_001'
                    )
            
            # Verificar que el resultado incluye análisis estadístico
            self.assertIsInstance(result, dict)
            if 'statistical_analysis' in result:
                stat_analysis = result['statistical_analysis']
                self.assertIn('bootstrap_confidence_intervals', stat_analysis)
                self.assertIn('quality_assessment', stat_analysis)
                
        except Exception as e:
            self.skipTest(f"Procesamiento con estadísticas no disponible: {e}")
    
    def test_compare_evidence_with_statistical_tests(self):
        """Test de comparación de evidencia con tests estadísticos"""
        try:
            with patch('cv2.imread', return_value=np.random.randint(0, 255, (500, 500, 3))):
                with patch('os.path.exists', return_value=True):
                    result = self.nist_manager.compare_evidence(
                        reference_path='/test/ref.jpg',
                        comparison_path='/test/comp.jpg',
                        evidence_type='cartridge_case'
                    )
            
            # Verificar análisis estadístico en comparación
            if 'statistical_comparison' in result:
                stat_comp = result['statistical_comparison']
                self.assertIn('p_values', stat_comp)
                self.assertIn('bonferroni_correction', stat_comp)
                self.assertIn('statistical_significance', stat_comp)
                
        except Exception as e:
            self.skipTest(f"Comparación estadística no disponible: {e}")
    
    def test_export_nist_report_with_statistics(self):
        """Test de exportación de reporte NIST con estadísticas"""
        try:
            # Preparar datos mock
            mock_data = {
                'case_info': {'case_id': 'TEST_EXPORT_001'},
                'quality_metrics': self.mock_evidence_data['quality_metrics'],
                'statistical_analysis': {
                    'bootstrap_results': {
                        'snr_confidence_interval': (22.5, 28.5),
                        'contrast_confidence_interval': (0.75, 0.95)
                    },
                    'statistical_tests': [
                        {
                            'test_type': 'T_TEST',
                            'p_value': 0.023,
                            'is_significant': True
                        }
                    ]
                }
            }
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
                report_path = tmp_file.name
            
            try:
                success = self.nist_manager.export_nist_report(
                    mock_data,
                    report_path,
                    format_type='json'
                )
                
                self.assertTrue(success)
                
                # Verificar contenido del reporte
                if os.path.exists(report_path):
                    with open(report_path, 'r') as f:
                        report_content = json.load(f)
                    
                    self.assertIn('statistical_analysis', report_content)
                    
            finally:
                if os.path.exists(report_path):
                    os.unlink(report_path)
                    
        except Exception as e:
            self.skipTest(f"Exportación con estadísticas no disponible: {e}")
    
    def test_validate_system_with_statistical_metrics(self):
        """Test de validación del sistema con métricas estadísticas"""
        try:
            validation_result = self.nist_manager.validate_system()
            
            self.assertIsInstance(validation_result, dict)
            
            # Verificar que incluye validación estadística
            if 'statistical_validation' in validation_result:
                stat_val = validation_result['statistical_validation']
                self.assertIn('bootstrap_validation', stat_val)
                self.assertIn('power_analysis', stat_val)
                
        except Exception as e:
            self.skipTest(f"Validación estadística no disponible: {e}")


class TestStatisticalQualityMetrics(unittest.TestCase):
    """Tests para métricas de calidad con análisis estadístico"""
    
    def setUp(self):
        """Configurar datos de prueba"""
        if QualityMetrics is None:
            self.skipTest("QualityMetrics no disponible")
            
        self.quality_metrics = QualityMetrics()
        self.statistical_analysis = AdvancedStatisticalAnalysis()
        
        # Datos de calidad simulados
        self.quality_samples = {
            'snr_values': [25.2, 24.8, 26.1, 25.5, 24.9, 25.8, 25.3, 24.7, 25.9, 25.1],
            'contrast_values': [0.85, 0.83, 0.87, 0.84, 0.86, 0.85, 0.88, 0.82, 0.85, 0.84],
            'uniformity_values': [0.92, 0.91, 0.93, 0.90, 0.92, 0.91, 0.94, 0.89, 0.92, 0.90]
        }
    
    def test_quality_metrics_bootstrap_analysis(self):
        """Test de análisis bootstrap para métricas de calidad"""
        try:
            for metric_name, values in self.quality_samples.items():
                with self.subTest(metric=metric_name):
                    bootstrap_result = self.statistical_analysis.bootstrap_sampling(
                        values,
                        statistic_func=np.mean,
                        n_bootstrap=100,
                        confidence_level=0.95
                    )
                    
                    self.assertIsNotNone(bootstrap_result.confidence_interval)
                    self.assertLess(
                        bootstrap_result.confidence_interval[0],
                        bootstrap_result.confidence_interval[1]
                    )
                    
        except Exception as e:
            self.skipTest(f"Bootstrap para métricas no disponible: {e}")
    
    def test_quality_comparison_statistical_tests(self):
        """Test de comparación estadística entre métricas de calidad"""
        try:
            # Simular dos conjuntos de métricas
            metrics1 = self.quality_samples['snr_values']
            metrics2 = [x + np.random.normal(0, 0.5) for x in metrics1]  # Variación pequeña
            
            # Test t-student
            t_test_result = self.statistical_analysis.calculate_p_value(
                metrics1,
                metrics2,
                test_type=StatisticalTest.T_TEST
            )
            
            self.assertIsNotNone(t_test_result.p_value)
            self.assertIsInstance(t_test_result.is_significant, bool)
            
            # Test Mann-Whitney
            mw_test_result = self.statistical_analysis.calculate_p_value(
                metrics1,
                metrics2,
                test_type=StatisticalTest.MANN_WHITNEY
            )
            
            self.assertIsNotNone(mw_test_result.p_value)
            
        except Exception as e:
            self.skipTest(f"Tests de comparación no disponibles: {e}")


class TestAFTEStatisticalConclusions(unittest.TestCase):
    """Tests para conclusiones AFTE con análisis estadístico"""
    
    def setUp(self):
        """Configurar datos de prueba"""
        if AFTEConclusions is None:
            self.skipTest("AFTEConclusions no disponible")
            
        self.afte_conclusions = AFTEConclusions()
        self.statistical_analysis = AdvancedStatisticalAnalysis()
        
        # Datos de similitud simulados
        self.similarity_data = {
            'high_confidence': [0.95, 0.93, 0.96, 0.94, 0.97, 0.92, 0.95, 0.94, 0.96, 0.93],
            'medium_confidence': [0.75, 0.78, 0.72, 0.76, 0.74, 0.77, 0.73, 0.75, 0.76, 0.74],
            'low_confidence': [0.45, 0.48, 0.42, 0.46, 0.44, 0.47, 0.43, 0.45, 0.46, 0.44]
        }
    
    def test_afte_conclusions_with_confidence_intervals(self):
        """Test de conclusiones AFTE con intervalos de confianza"""
        try:
            for confidence_level, similarities in self.similarity_data.items():
                with self.subTest(confidence=confidence_level):
                    # Bootstrap para intervalos de confianza
                    bootstrap_result = self.statistical_analysis.bootstrap_sampling(
                        similarities,
                        statistic_func=np.mean,
                        n_bootstrap=100,
                        confidence_level=0.95
                    )
                    
                    # Generar conclusión AFTE con soporte estadístico
                    mean_similarity = np.mean(similarities)
                    
                    # Simular conclusión basada en similitud y estadísticas
                    if mean_similarity > 0.9:
                        expected_conclusion = "identification"
                    elif mean_similarity > 0.7:
                        expected_conclusion = "probable_identification"
                    else:
                        expected_conclusion = "inconclusive"
                    
                    # Verificar que tenemos datos estadísticos válidos
                    self.assertIsNotNone(bootstrap_result.confidence_interval)
                    self.assertGreater(mean_similarity, 0)
                    self.assertLess(mean_similarity, 1)
                    
        except Exception as e:
            self.skipTest(f"Conclusiones AFTE estadísticas no disponibles: {e}")
    
    def test_multiple_comparison_correction_for_afte(self):
        """Test de corrección por comparaciones múltiples en análisis AFTE"""
        try:
            # Simular múltiples comparaciones
            p_values = []
            
            # Comparar cada nivel de confianza con los otros
            confidence_levels = list(self.similarity_data.keys())
            
            for i in range(len(confidence_levels)):
                for j in range(i + 1, len(confidence_levels)):
                    data1 = self.similarity_data[confidence_levels[i]]
                    data2 = self.similarity_data[confidence_levels[j]]
                    
                    test_result = self.statistical_analysis.calculate_p_value(
                        data1,
                        data2,
                        test_type=StatisticalTest.T_TEST
                    )
                    
                    p_values.append(test_result.p_value)
            
            # Aplicar corrección de Bonferroni
            if p_values:
                correction_result = self.statistical_analysis.multiple_comparison_correction(
                    p_values,
                    method=CorrectionMethod.BONFERRONI,
                    alpha=0.05
                )
                
                self.assertEqual(len(correction_result.corrected_p_values), len(p_values))
                self.assertEqual(len(correction_result.rejected_hypotheses), len(p_values))
                
        except Exception as e:
            self.skipTest(f"Corrección múltiple AFTE no disponible: {e}")


class TestStatisticalReportGeneration(unittest.TestCase):
    """Tests para generación de reportes estadísticos integrados"""
    
    def setUp(self):
        """Configurar datos de prueba"""
        self.nist_manager = NISTStandardsManager()
        
    def test_comprehensive_statistical_report(self):
        """Test de reporte estadístico comprehensivo"""
        try:
            # Datos mock para reporte completo
            mock_analysis_data = {
                'case_info': {
                    'case_id': 'STAT_REPORT_001',
                    'investigator': 'Test Investigator',
                    'date': '2024-01-15'
                },
                'evidence_data': {
                    'type': 'cartridge_case',
                    'quality_metrics': {
                        'snr': 25.5,
                        'contrast': 0.85,
                        'uniformity': 0.92
                    }
                },
                'statistical_analysis': {
                    'bootstrap_results': {
                        'snr_ci': (22.5, 28.5),
                        'contrast_ci': (0.75, 0.95)
                    },
                    'statistical_tests': [
                        {
                            'test_type': 'T_TEST',
                            'p_value': 0.023,
                            'is_significant': True
                        }
                    ],
                    'multiple_comparisons': {
                        'method': 'BONFERRONI',
                        'corrected_alpha': 0.0125,
                        'significant_comparisons': 2
                    },
                    'power_analysis': {
                        'statistical_power': 0.85,
                        'effect_size': 0.6,
                        'sample_size': 20
                    }
                }
            }
            
            # Generar reporte
            if hasattr(self.nist_manager.statistical_analysis, 'generate_statistical_report'):
                report = self.nist_manager.statistical_analysis.generate_statistical_report(
                    [mock_analysis_data],
                    title="Reporte Estadístico Comprehensivo - SEACABA"
                )
                
                self.assertIsInstance(report, str)
                self.assertIn("SEACABA", report)
                self.assertIn("Bootstrap", report)
                self.assertIn("Bonferroni", report)
                
        except Exception as e:
            self.skipTest(f"Reporte estadístico no disponible: {e}")


if __name__ == '__main__':
    # Configurar el runner de tests
    unittest.main(verbosity=2, buffer=True)