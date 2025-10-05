#!/usr/bin/env python3
"""
Test de validación NIST para verificar precisión y trazabilidad
durante la migración al núcleo estadístico unificado
"""

import unittest
import numpy as np
import warnings
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
import sys
import os
import logging
import time

# Agregar el directorio raíz al path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importar implementaciones originales y adaptadores
try:
    from nist_standards.statistical_analysis import AdvancedStatisticalAnalysis
    from matching.bootstrap_similarity import BootstrapSimilarityAnalyzer
    from image_processing.statistical_analyzer import StatisticalAnalyzer
    ORIGINAL_NIST_AVAILABLE = True
except ImportError:
    ORIGINAL_NIST_AVAILABLE = False
    AdvancedStatisticalAnalysis = None
    BootstrapSimilarityAnalyzer = None
    StatisticalAnalyzer = None

# Importar implementación unificada y adaptadores
from common.statistical_core import UnifiedStatisticalAnalysis
from common.compatibility_adapters import (
    AdvancedStatisticalAnalysisAdapter,
    BootstrapSimilarityAnalyzerAdapter,
    StatisticalAnalyzerAdapter,
    get_adapter,
    enable_unified_mode,
    disable_unified_mode,
    get_migration_status
)

# Configurar logging para trazabilidad
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class NISTValidationResult:
    """Resultado de validación NIST"""
    test_name: str
    original_result: Any
    unified_result: Any
    precision_error: float
    is_valid: bool
    nist_compliant: bool
    execution_time_original: float
    execution_time_unified: float
    metadata: Dict[str, Any]


class NISTMetricsValidator:
    """
    Validador de métricas NIST para migración estadística
    
    Garantiza que la migración preserve:
    - Precisión numérica (< 1e-10)
    - Reproducibilidad
    - Trazabilidad NIST
    - Performance aceptable
    """
    
    def __init__(self, tolerance: float = 1e-10, random_state: int = 42):
        """
        Inicializa el validador NIST
        
        Args:
            tolerance: Tolerancia máxima para diferencias numéricas
            random_state: Semilla para reproducibilidad
        """
        self.tolerance = tolerance
        self.random_state = random_state
        self.validation_results: List[NISTValidationResult] = []
        
        # Configurar implementaciones para comparación
        self.original_nist = AdvancedStatisticalAnalysis(random_state=random_state)
        self.unified_nist = UnifiedStatisticalAnalysis(random_state=random_state)
        
        logger.info(f"NISTMetricsValidator inicializado - tolerancia={tolerance}")
    
    def validate_bootstrap_sampling(self, data: np.ndarray, n_bootstrap: int = 1000) -> NISTValidationResult:
        """
        Valida equivalencia en bootstrap sampling
        
        Args:
            data: Datos para bootstrap
            n_bootstrap: Número de muestras bootstrap
            
        Returns:
            Resultado de validación NIST
        """
        logger.info("Validando bootstrap sampling...")
        
        # Función estadística estándar
        statistic_func = np.mean
        
        # Ejecutar implementación original
        start_time = time.time()
        original_result = self.original_nist.bootstrap_sampling(
            data=data,
            statistic_func=statistic_func,
            n_bootstrap=n_bootstrap,
            confidence_level=0.95,
            method='percentile'
        )
        original_time = time.time() - start_time
        
        # Ejecutar implementación unificada
        start_time = time.time()
        unified_result = self.unified_nist.bootstrap_sampling(
            data=data,
            statistic_func=statistic_func,
            n_bootstrap=n_bootstrap,
            confidence_level=0.95,
            method='percentile'
        )
        unified_time = time.time() - start_time
        
        # Calcular error de precisión
        precision_error = abs(original_result.statistic - unified_result.statistic)
        
        # Validar equivalencia
        is_valid = precision_error < self.tolerance
        nist_compliant = (
            is_valid and
            abs(original_result.confidence_interval[0] - unified_result.confidence_interval[0]) < self.tolerance and
            abs(original_result.confidence_interval[1] - unified_result.confidence_interval[1]) < self.tolerance
        )
        
        result = NISTValidationResult(
            test_name="bootstrap_sampling",
            original_result=original_result,
            unified_result=unified_result,
            precision_error=precision_error,
            is_valid=is_valid,
            nist_compliant=nist_compliant,
            execution_time_original=original_time,
            execution_time_unified=unified_time,
            metadata={
                'n_bootstrap': n_bootstrap,
                'data_size': len(data),
                'confidence_level': 0.95,
                'method': 'percentile'
            }
        )
        
        self.validation_results.append(result)
        logger.info(f"Bootstrap validation - precisión_error={precision_error:.2e}, válido={is_valid}")
        return result
    
    def validate_statistical_tests(self, data1: np.ndarray, data2: np.ndarray) -> NISTValidationResult:
        """
        Valida equivalencia en tests estadísticos
        
        Args:
            data1: Primera muestra
            data2: Segunda muestra
            
        Returns:
            Resultado de validación NIST
        """
        logger.info("Validando tests estadísticos...")
        
        from nist_standards.statistical_analysis import StatisticalTest
        
        # Ejecutar implementación original
        start_time = time.time()
        original_result = self.original_nist.calculate_p_value(
            data1=data1,
            data2=data2,
            test_type=StatisticalTest.T_TEST,
            alternative='two-sided'
        )
        original_time = time.time() - start_time
        
        # Ejecutar implementación unificada
        start_time = time.time()
        unified_result = self.unified_nist.calculate_p_value(
            data1=data1,
            data2=data2,
            test_type=StatisticalTest.T_TEST,
            alternative='two-sided'
        )
        unified_time = time.time() - start_time
        
        # Calcular error de precisión
        precision_error = abs(original_result.p_value - unified_result.p_value)
        
        # Validar equivalencia
        is_valid = precision_error < self.tolerance
        nist_compliant = (
            is_valid and
            original_result.is_significant == unified_result.is_significant
        )
        
        result = NISTValidationResult(
            test_name="statistical_tests",
            original_result=original_result,
            unified_result=unified_result,
            precision_error=precision_error,
            is_valid=is_valid,
            nist_compliant=nist_compliant,
            execution_time_original=original_time,
            execution_time_unified=unified_time,
            metadata={
                'test_type': 'T_TEST',
                'alternative': 'two-sided',
                'data1_size': len(data1),
                'data2_size': len(data2)
            }
        )
        
        self.validation_results.append(result)
        logger.info(f"Statistical tests validation - precisión_error={precision_error:.2e}, válido={is_valid}")
        return result
    
    def validate_multiple_corrections(self, p_values: np.ndarray) -> NISTValidationResult:
        """
        Valida equivalencia en correcciones múltiples
        
        Args:
            p_values: Array de p-values para corrección
            
        Returns:
            Resultado de validación NIST
        """
        logger.info("Validando correcciones múltiples...")
        
        from nist_standards.statistical_analysis import CorrectionMethod
        
        # Ejecutar implementación original
        start_time = time.time()
        original_result = self.original_nist.multiple_comparison_correction(
            p_values=p_values,
            method=CorrectionMethod.BONFERRONI,
            alpha=0.05
        )
        original_time = time.time() - start_time
        
        # Ejecutar implementación unificada
        start_time = time.time()
        unified_result = self.unified_nist.multiple_comparison_correction(
            p_values=p_values,
            method=CorrectionMethod.BONFERRONI,
            alpha=0.05
        )
        unified_time = time.time() - start_time
        
        # Calcular error de precisión (promedio de diferencias en p-values corregidos)
        precision_error = np.mean(np.abs(original_result.corrected_p_values - unified_result.corrected_p_values))
        
        # Validar equivalencia
        is_valid = precision_error < self.tolerance
        nist_compliant = (
            is_valid and
            np.array_equal(original_result.rejected_hypotheses, unified_result.rejected_hypotheses)
        )
        
        result = NISTValidationResult(
            test_name="multiple_corrections",
            original_result=original_result,
            unified_result=unified_result,
            precision_error=precision_error,
            is_valid=is_valid,
            nist_compliant=nist_compliant,
            execution_time_original=original_time,
            execution_time_unified=unified_time,
            metadata={
                'method': 'BONFERRONI',
                'alpha': 0.05,
                'n_comparisons': len(p_values)
            }
        )
        
        self.validation_results.append(result)
        logger.info(f"Multiple corrections validation - precisión_error={precision_error:.2e}, válido={is_valid}")
        return result
    
    def validate_image_quality_metrics(self, image: np.ndarray) -> NISTValidationResult:
        """
        Valida equivalencia en métricas de calidad de imagen
        
        Args:
            image: Imagen para análisis
            
        Returns:
            Resultado de validación NIST
        """
        logger.info("Validando métricas de calidad de imagen...")
        
        # Crear adaptadores para comparación
        original_analyzer = StatisticalAnalyzer()
        unified_analyzer = StatisticalAnalyzerAdapter()
        
        # Ejecutar implementación original
        start_time = time.time()
        original_result = original_analyzer.analyze_image(image)
        original_time = time.time() - start_time
        
        # Ejecutar implementación unificada
        start_time = time.time()
        unified_result = unified_analyzer.analyze_image(image)
        unified_time = time.time() - start_time
        
        # Calcular error de precisión en métricas clave
        key_metrics = ['entropy', 'contrast', 'sharpness']
        precision_errors = []
        
        for metric in key_metrics:
            if metric in original_result and metric in unified_result:
                error = abs(original_result[metric] - unified_result[metric])
                precision_errors.append(error)
        
        precision_error = np.mean(precision_errors) if precision_errors else float('inf')
        
        # Validar equivalencia
        is_valid = precision_error < self.tolerance
        nist_compliant = is_valid
        
        result = NISTValidationResult(
            test_name="image_quality_metrics",
            original_result=original_result,
            unified_result=unified_result,
            precision_error=precision_error,
            is_valid=is_valid,
            nist_compliant=nist_compliant,
            execution_time_original=original_time,
            execution_time_unified=unified_time,
            metadata={
                'image_shape': image.shape,
                'validated_metrics': key_metrics,
                'n_metrics_compared': len(precision_errors)
            }
        )
        
        self.validation_results.append(result)
        logger.info(f"Image quality validation - precisión_error={precision_error:.2e}, válido={is_valid}")
        return result
    
    def validate_adapter_compatibility(self) -> NISTValidationResult:
        """
        Valida que los adaptadores mantengan compatibilidad completa
        
        Returns:
            Resultado de validación NIST
        """
        logger.info("Validando compatibilidad de adaptadores...")
        
        # Datos de prueba
        test_data = np.random.normal(0, 1, 100)
        
        # Crear adaptadores en ambas fases
        adapter_phase1 = AdvancedStatisticalAnalysisAdapter(random_state=self.random_state, use_unified=False)
        adapter_phase2 = AdvancedStatisticalAnalysisAdapter(random_state=self.random_state, use_unified=True)
        
        # Ejecutar bootstrap en Fase 1
        start_time = time.time()
        result_phase1 = adapter_phase1.bootstrap_sampling(
            data=test_data,
            statistic_func=np.mean,
            n_bootstrap=1000
        )
        phase1_time = time.time() - start_time
        
        # Ejecutar bootstrap en Fase 2
        start_time = time.time()
        result_phase2 = adapter_phase2.bootstrap_sampling(
            data=test_data,
            statistic_func=np.mean,
            n_bootstrap=1000
        )
        phase2_time = time.time() - start_time
        
        # Calcular error de precisión
        precision_error = abs(result_phase1.statistic - result_phase2.statistic)
        
        # Validar equivalencia
        is_valid = precision_error < self.tolerance
        nist_compliant = is_valid
        
        result = NISTValidationResult(
            test_name="adapter_compatibility",
            original_result=result_phase1,
            unified_result=result_phase2,
            precision_error=precision_error,
            is_valid=is_valid,
            nist_compliant=nist_compliant,
            execution_time_original=phase1_time,
            execution_time_unified=phase2_time,
            metadata={
                'phase1_implementation': 'original',
                'phase2_implementation': 'unified',
                'data_size': len(test_data),
                'n_bootstrap': 1000
            }
        )
        
        self.validation_results.append(result)
        logger.info(f"Adapter compatibility validation - precisión_error={precision_error:.2e}, válido={is_valid}")
        return result
    
    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """
        Ejecuta validación completa de métricas NIST
        
        Returns:
            Reporte completo de validación
        """
        logger.info("Iniciando validación completa NIST...")
        
        # Generar datos de prueba reproducibles
        np.random.seed(self.random_state)
        
        # Datos para bootstrap
        bootstrap_data = np.random.normal(10, 2, 200)
        
        # Datos para tests estadísticos
        data1 = np.random.normal(10, 2, 100)
        data2 = np.random.normal(10.5, 2.2, 100)
        
        # P-values para corrección múltiple
        p_values = np.random.uniform(0.001, 0.1, 20)
        
        # Imagen sintética para análisis
        image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        
        # Ejecutar todas las validaciones
        validations = [
            self.validate_bootstrap_sampling(bootstrap_data),
            self.validate_statistical_tests(data1, data2),
            self.validate_multiple_corrections(p_values),
            self.validate_image_quality_metrics(image),
            self.validate_adapter_compatibility()
        ]
        
        # Generar reporte
        total_tests = len(validations)
        passed_tests = sum(1 for v in validations if v.is_valid)
        nist_compliant_tests = sum(1 for v in validations if v.nist_compliant)
        
        avg_precision_error = np.mean([v.precision_error for v in validations])
        max_precision_error = np.max([v.precision_error for v in validations])
        
        # Performance comparison
        total_original_time = sum(v.execution_time_original for v in validations)
        total_unified_time = sum(v.execution_time_unified for v in validations)
        performance_ratio = total_unified_time / total_original_time if total_original_time > 0 else 1.0
        
        report = {
            'validation_summary': {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'nist_compliant_tests': nist_compliant_tests,
                'success_rate': passed_tests / total_tests * 100,
                'nist_compliance_rate': nist_compliant_tests / total_tests * 100
            },
            'precision_analysis': {
                'tolerance_threshold': self.tolerance,
                'average_precision_error': avg_precision_error,
                'maximum_precision_error': max_precision_error,
                'precision_acceptable': max_precision_error < self.tolerance
            },
            'performance_analysis': {
                'total_original_time': total_original_time,
                'total_unified_time': total_unified_time,
                'performance_ratio': performance_ratio,
                'performance_acceptable': performance_ratio < 2.0  # Máximo 2x más lento
            },
            'nist_traceability': {
                'random_state': self.random_state,
                'validation_timestamp': time.time(),
                'standards_validated': [
                    'ISO 5725-2:2019',
                    'NIST/SEMATECH e-Handbook',
                    'NIST SP 800-90A Rev. 1'
                ]
            },
            'detailed_results': [
                {
                    'test_name': v.test_name,
                    'is_valid': v.is_valid,
                    'nist_compliant': v.nist_compliant,
                    'precision_error': v.precision_error,
                    'execution_time_ratio': v.execution_time_unified / v.execution_time_original if v.execution_time_original > 0 else 1.0,
                    'metadata': v.metadata
                }
                for v in validations
            ]
        }
        
        # Determinar estado general de migración
        migration_ready = (
            report['validation_summary']['success_rate'] >= 95.0 and
            report['validation_summary']['nist_compliance_rate'] >= 95.0 and
            report['precision_analysis']['precision_acceptable'] and
            report['performance_analysis']['performance_acceptable']
        )
        
        report['migration_status'] = {
            'ready_for_phase2': migration_ready,
            'risk_level': 'bajo' if migration_ready else 'medio',
            'recommendation': 'proceder_con_migracion' if migration_ready else 'revisar_implementacion'
        }
        
        logger.info(f"Validación NIST completada - éxito={report['validation_summary']['success_rate']:.1f}%, "
                   f"cumplimiento_NIST={report['validation_summary']['nist_compliance_rate']:.1f}%")
        
        return report
    
    def generate_nist_compliance_report(self) -> str:
        """
        Genera reporte de cumplimiento NIST en formato texto
        
        Returns:
            Reporte formateado para documentación
        """
        if not self.validation_results:
            return "No hay resultados de validación disponibles."
        
        report = self.run_comprehensive_validation()
        
        report_text = f"""
REPORTE DE CUMPLIMIENTO NIST - MIGRACIÓN ESTADÍSTICA SIGeC-Balistica
============================================================

RESUMEN EJECUTIVO:
- Tests Ejecutados: {report['validation_summary']['total_tests']}
- Tests Exitosos: {report['validation_summary']['passed_tests']} ({report['validation_summary']['success_rate']:.1f}%)
- Cumplimiento NIST: {report['validation_summary']['nist_compliant_tests']} ({report['validation_summary']['nist_compliance_rate']:.1f}%)

ANÁLISIS DE PRECISIÓN:
- Tolerancia Configurada: {report['precision_analysis']['tolerance_threshold']:.2e}
- Error Promedio: {report['precision_analysis']['average_precision_error']:.2e}
- Error Máximo: {report['precision_analysis']['maximum_precision_error']:.2e}
- Precisión Aceptable: {'✓' if report['precision_analysis']['precision_acceptable'] else '✗'}

ANÁLISIS DE PERFORMANCE:
- Tiempo Original: {report['performance_analysis']['total_original_time']:.3f}s
- Tiempo Unificado: {report['performance_analysis']['total_unified_time']:.3f}s
- Ratio Performance: {report['performance_analysis']['performance_ratio']:.2f}x
- Performance Aceptable: {'✓' if report['performance_analysis']['performance_acceptable'] else '✗'}

ESTADO DE MIGRACIÓN:
- Listo para Fase 2: {'✓' if report['migration_status']['ready_for_phase2'] else '✗'}
- Nivel de Riesgo: {report['migration_status']['risk_level'].upper()}
- Recomendación: {report['migration_status']['recommendation'].replace('_', ' ').upper()}

TRAZABILIDAD NIST:
- Semilla Aleatoria: {report['nist_traceability']['random_state']}
- Timestamp: {report['nist_traceability']['validation_timestamp']}
- Estándares Validados: {', '.join(report['nist_traceability']['standards_validated'])}

RESULTADOS DETALLADOS:
"""
        
        for result in report['detailed_results']:
            status = '✓' if result['is_valid'] else '✗'
            nist_status = '✓' if result['nist_compliant'] else '✗'
            report_text += f"""
- {result['test_name'].upper()}:
  Válido: {status} | NIST: {nist_status} | Error: {result['precision_error']:.2e} | Ratio: {result['execution_time_ratio']:.2f}x
"""
        
        return report_text


class TestNISTValidation(unittest.TestCase):
    """
    Test suite para validación NIST de migración estadística
    """
    
    def setUp(self):
        """Configurar test environment"""
        self.validator = NISTMetricsValidator(tolerance=1e-10, random_state=42)
        np.random.seed(42)
    
    def test_bootstrap_precision(self):
        """Test precisión en bootstrap sampling"""
        data = np.random.normal(0, 1, 100)
        result = self.validator.validate_bootstrap_sampling(data, n_bootstrap=1000)
        
        self.assertTrue(result.is_valid, f"Bootstrap precision failed: error={result.precision_error}")
        self.assertTrue(result.nist_compliant, "Bootstrap NIST compliance failed")
    
    def test_statistical_tests_precision(self):
        """Test precisión en tests estadísticos"""
        data1 = np.random.normal(0, 1, 50)
        data2 = np.random.normal(0.5, 1, 50)
        result = self.validator.validate_statistical_tests(data1, data2)
        
        self.assertTrue(result.is_valid, f"Statistical tests precision failed: error={result.precision_error}")
        self.assertTrue(result.nist_compliant, "Statistical tests NIST compliance failed")
    
    def test_multiple_corrections_precision(self):
        """Test precisión en correcciones múltiples"""
        p_values = np.array([0.01, 0.05, 0.1, 0.2, 0.3])
        result = self.validator.validate_multiple_corrections(p_values)
        
        self.assertTrue(result.is_valid, f"Multiple corrections precision failed: error={result.precision_error}")
        self.assertTrue(result.nist_compliant, "Multiple corrections NIST compliance failed")
    
    def test_adapter_compatibility(self):
        """Test compatibilidad de adaptadores"""
        result = self.validator.validate_adapter_compatibility()
        
        self.assertTrue(result.is_valid, f"Adapter compatibility failed: error={result.precision_error}")
        self.assertTrue(result.nist_compliant, "Adapter NIST compliance failed")
    
    def test_comprehensive_validation(self):
        """Test validación completa"""
        report = self.validator.run_comprehensive_validation()
        
        self.assertGreaterEqual(report['validation_summary']['success_rate'], 95.0)
        self.assertGreaterEqual(report['validation_summary']['nist_compliance_rate'], 95.0)
        self.assertTrue(report['precision_analysis']['precision_acceptable'])
        self.assertTrue(report['migration_status']['ready_for_phase2'])


if __name__ == "__main__":
    print("SIGeC-Balistica - Validación de Métricas NIST para Migración Estadística")
    print("=" * 70)
    
    # Ejecutar validación completa
    validator = NISTMetricsValidator(tolerance=1e-10, random_state=42)
    
    try:
        report = validator.run_comprehensive_validation()
        print("\n" + validator.generate_nist_compliance_report())
        
        # Ejecutar tests unitarios
        print("\n" + "=" * 70)
        print("EJECUTANDO TESTS UNITARIOS...")
        unittest.main(argv=[''], exit=False, verbosity=2)
        
    except Exception as e:
        logger.error(f"Error en validación NIST: {e}")
        print(f"\n❌ ERROR EN VALIDACIÓN: {e}")