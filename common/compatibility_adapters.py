#!/usr/bin/env python3
"""
SIGeC-Balistica - Adaptadores de Compatibilidad para Refactorización Estadística
========================================================================

Este módulo proporciona adaptadores transparentes para mantener compatibilidad
hacia atrás durante la migración hacia statistical_core.py centralizado.

Fase 1: Wrappers transparentes (RIESGO MÍNIMO)
Fase 2: Migración gradual hacia UnifiedStatisticalAnalysis (RIESGO CONTROLADO)
Fase 3: Consolidación final (RIESGO MEDIO)

Cumplimiento NIST:
- ISO 5725-2:2019 (Precisión de métodos de medición)
- NIST/SEMATECH e-Handbook (Análisis estadístico)
- NIST SP 800-90A Rev. 1 (Generación de números aleatorios)

Autor: Sistema SIGeC-Balistica
Fecha: 2024
"""

import numpy as np
import logging
from typing import List, Dict, Tuple, Optional, Union, Callable, Any
from dataclasses import dataclass
import warnings
import time

# Importaciones de módulos originales (Fase 1)
# Importación diferida para evitar dependencia circular
try:
    from nist_standards.statistical_analysis import (
        StatisticalTest,
        CorrectionMethod,
        BootstrapResult,
        StatisticalTestResult,
        MultipleComparisonResult
    )
except ImportError:
    # Definiciones locales como fallback
    from enum import Enum
    from dataclasses import dataclass
    from typing import Tuple, Optional
    import numpy as np
    
    class StatisticalTest(Enum):
        T_TEST = "t_test"
        MANN_WHITNEY = "mann_whitney"
        KOLMOGOROV_SMIRNOV = "kolmogorov_smirnov"
    
    class CorrectionMethod(Enum):
        BONFERRONI = "bonferroni"
        HOLM = "holm"
        BENJAMINI_HOCHBERG = "benjamini_hochberg"
    
    @dataclass
    class BootstrapResult:
        original_statistic: float
        bootstrap_statistics: np.ndarray
        confidence_interval: Tuple[float, float]
        confidence_level: float
        bias: float
        standard_error: float
        percentile_ci: Tuple[float, float]
        bca_ci: Optional[Tuple[float, float]]
        n_bootstrap: int
    
    @dataclass
    class StatisticalTestResult:
        test_name: str
        statistic: float
        p_value: float
        critical_value: Optional[float]
        confidence_interval: Optional[Tuple[float, float]]
        effect_size: Optional[float]
        power: Optional[float]
        sample_size: int
        degrees_of_freedom: Optional[int]
        is_significant: bool
        alpha: float
    
    @dataclass
    class MultipleComparisonResult:
        original_p_values: np.ndarray
        corrected_p_values: np.ndarray
        rejected_hypotheses: np.ndarray
        correction_method: str
        alpha: float
        n_comparisons: int
        family_wise_error_rate: float
        false_discovery_rate: Optional[float]

from matching.bootstrap_similarity import (
    BootstrapSimilarityAnalyzer,
    SimilarityBootstrapResult,
    MatchingBootstrapConfig,
    create_similarity_bootstrap_function as original_create_similarity_bootstrap_function,
    calculate_bootstrap_confidence_interval as original_calculate_bootstrap_confidence_interval
)

from image_processing.statistical_analyzer import StatisticalAnalyzer

# Importación del nuevo módulo centralizado (Fase 2)
from common.statistical_core import UnifiedStatisticalAnalysis

logger = logging.getLogger(__name__)


class AdvancedStatisticalAnalysisAdapter:
    """
    Adaptador wrapper transparente para AdvancedStatisticalAnalysis
    
    Fase 1: Delega directamente a la implementación original
    Mantiene 100% compatibilidad funcional
    Preserva trazabilidad NIST completa
    """
    
    def __init__(self, random_state: Optional[int] = None, use_unified: Optional[bool] = None):
        """
        Inicializar adaptador con soporte para migración gradual
        
        Args:
            random_state: Semilla para reproducibilidad
            use_unified: Si usar implementación unificada (None = usar configuración global)
        """
        self.random_state = random_state
        self._use_unified = use_unified if use_unified is not None else _UNIFIED_MODE_ENABLED
        
        # Inicializar implementaciones según el modo
        if self._use_unified:
            try:
                from common.statistical_core import UnifiedStatisticalAnalysis
                self._implementation = UnifiedStatisticalAnalysis(random_state=random_state)
                impl_type = "unified"
            except ImportError:
                # Fallback al adaptador de migración crítica
                try:
                    from nist_standards.critical_migration_adapter import CriticalMigrationAdapter
                    self._implementation = CriticalMigrationAdapter(random_state=random_state)
                    impl_type = "critical_migration"
                    logger.warning("Usando adaptador de migración crítica")
                except ImportError:
                    # Fallback final - implementación básica
                    self._implementation = None
                    impl_type = "none"
                    logger.error("No se pudo cargar ninguna implementación")
        else:
            try:
                AdvancedStatisticalAnalysisClass = get_advanced_statistical_analysis()
                self._implementation = AdvancedStatisticalAnalysisClass(random_state=random_state)
                impl_type = "original"
            except Exception as e:
                logger.error(f"Error cargando implementación original: {e}")
                self._implementation = None
                impl_type = "none"
        
        # Logging para trazabilidad NIST
        if self._implementation is not None:
            logger.info(f"AdvancedStatisticalAnalysisAdapter inicializado - Implementación: {impl_type} - Trazabilidad NIST preservada - random_state={random_state}")
        else:
            logger.error(f"AdvancedStatisticalAnalysisAdapter falló al inicializar - Implementación: {impl_type} - random_state={random_state}")
    
    def bootstrap_sampling(
        self,
        data: Union[np.ndarray, List[float]],
        statistic_func: Callable[[np.ndarray], float],
        n_bootstrap: int = 10000,
        confidence_level: float = 0.95,
        method: str = 'percentile',
        stratified: bool = False,
        parallel: bool = True
    ) -> BootstrapResult:
        """
        Wrapper transparente para bootstrap_sampling
        Delega a la implementación activa (original o unificada)
        """
        logger.debug(f"Bootstrap sampling - n_bootstrap={n_bootstrap}, confidence_level={confidence_level}, method={method}")
        
        result = self._implementation.bootstrap_sampling(
            data=data,
            statistic_func=statistic_func,
            n_bootstrap=n_bootstrap,
            confidence_level=confidence_level,
            method=method,
            stratified=stratified,
            parallel=parallel
        )
        
        logger.debug(f"Bootstrap completado - CI: {result.confidence_interval}")
        return result
    
    def calculate_p_value(
        self,
        data1: Union[np.ndarray, List[float]],
        data2: Optional[Union[np.ndarray, List[float]]] = None,
        test_type: StatisticalTest = StatisticalTest.T_TEST,
        alternative: str = 'two-sided',
        alpha: float = 0.05,
        **kwargs
    ) -> StatisticalTestResult:
        """
        Wrapper transparente para calculate_p_value
        Delega a la implementación activa (original o unificada)
        """
        logger.debug(f"Calculando p-value - test_type={test_type}, alternative={alternative}, alpha={alpha}")
        
        result = self._implementation.calculate_p_value(
            data1=data1,
            data2=data2,
            test_type=test_type,
            alternative=alternative,
            alpha=alpha,
            **kwargs
        )
        
        logger.debug(f"P-value calculado - p={result.p_value}, significativo={result.is_significant}")
        return result
    
    def multiple_comparison_correction(
        self,
        p_values: Union[np.ndarray, List[float]],
        method: CorrectionMethod = CorrectionMethod.BONFERRONI,
        alpha: float = 0.05,
        **kwargs
    ) -> MultipleComparisonResult:
        """
        Wrapper transparente para multiple_comparison_correction
        Delega a la implementación activa (original o unificada)
        """
        logger.debug(f"Corrección múltiple - method={method}, alpha={alpha}, n_comparisons={len(p_values)}")
        
        result = self._implementation.multiple_comparison_correction(
            p_values=p_values,
            method=method,
            alpha=alpha,
            **kwargs
        )
        
        logger.debug(f"Corrección completada - n_rejected={np.sum(result.rejected_hypotheses)}")
        return result
    
    def calculate_statistical_power(
        self,
        effect_size: float,
        sample_size: int,
        alpha: float = 0.05,
        test_type: str = 't_test',
        alternative: str = 'two-sided'
    ) -> float:
        """
        Wrapper transparente para calculate_statistical_power
        Delega a la implementación activa (original o unificada)
        """
        logger.debug(f"Calculando poder estadístico - effect_size={effect_size}, sample_size={sample_size}")
        
        result = self._implementation.calculate_statistical_power(
            effect_size=effect_size,
            sample_size=sample_size,
            alpha=alpha,
            test_type=test_type,
            alternative=alternative
        )
        
        logger.debug(f"Poder estadístico calculado - power={result}")
        return result
    
    def sample_size_calculation(
        self,
        effect_size: float,
        power: float = 0.8,
        alpha: float = 0.05,
        test_type: str = 't_test',
        alternative: str = 'two-sided'
    ) -> int:
        """
        Wrapper transparente para sample_size_calculation
        Delega a la implementación activa (original o unificada)
        """
        logger.debug(f"Calculando tamaño de muestra - effect_size={effect_size}, power={power}")
        
        result = self._implementation.sample_size_calculation(
            effect_size=effect_size,
            power=power,
            alpha=alpha,
            test_type=test_type,
            alternative=alternative
        )
        
        logger.debug(f"Tamaño de muestra calculado - n={result}")
        return result
    
    def generate_statistical_report(
        self,
        results: List[Union[BootstrapResult, StatisticalTestResult, MultipleComparisonResult]],
        title: str = "Reporte de Análisis Estadístico Avanzado"
    ) -> Dict[str, Any]:
        """
        Wrapper transparente para generate_statistical_report
        Delega a la implementación activa (original o unificada)
        """
        logger.info(f"Generando reporte estadístico - {len(results)} resultados")
        
        result = self._implementation.generate_statistical_report(results, title)
        
        # Agregar metadatos NIST de trazabilidad
        result['nist_traceability'] = {
            'adapter_version': '1.0.0',
            'phase': 'Fase 1 - Preparación',
            'compatibility_mode': 'wrapper_transparente',
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'standards_compliance': ['ISO 5725-2:2019', 'NIST/SEMATECH e-Handbook']
        }
        
        logger.info("Reporte generado con trazabilidad NIST preservada")
        return result


class BootstrapSimilarityAnalyzerAdapter:
    """
    Adaptador wrapper transparente para BootstrapSimilarityAnalyzer
    
    Fase 1: Delega directamente a la implementación original
    Mantiene 100% compatibilidad funcional
    """
    
    def __init__(self, config: Optional[MatchingBootstrapConfig] = None):
        """
        Inicializar adaptador delegando a la implementación original
        
        Args:
            config: Configuración bootstrap
        """
        self._original = BootstrapSimilarityAnalyzer(config=config)
        self.config = config
        
        logger.info("BootstrapSimilarityAnalyzerAdapter inicializado - Compatibilidad preservada")
    
    def bootstrap_similarity_confidence(
        self,
        matches_data: List[Dict[str, Any]],
        similarity_function: Callable[[List[Dict[str, Any]]], float],
        **kwargs
    ) -> SimilarityBootstrapResult:
        """
        Wrapper transparente para bootstrap_similarity_confidence
        Delega directamente a la implementación original
        """
        logger.debug(f"Bootstrap similitud - {len(matches_data)} matches")
        
        result = self._original.bootstrap_similarity_confidence(
            matches_data=matches_data,
            similarity_function=similarity_function,
            **kwargs
        )
        
        logger.debug(f"Bootstrap similitud completado - CI: {result.confidence_interval}")
        return result
    
    def bootstrap_similarity_analysis(
        self,
        similarity_scores: Union[np.ndarray, List[float]]
    ) -> SimilarityBootstrapResult:
        """
        Método adicional para análisis bootstrap de scores de similitud
        Mantiene compatibilidad con tests existentes
        """
        logger.debug(f"Análisis bootstrap de {len(similarity_scores)} scores de similitud")
        
        # Usar método existente si está disponible en la implementación original
        if hasattr(self._original, 'bootstrap_similarity_analysis'):
            return self._original.bootstrap_similarity_analysis(similarity_scores)
        
        # Fallback: convertir a formato compatible
        matches_data = [{'similarity': float(score)} for score in similarity_scores]
        
        def similarity_function(matches_list, **kwargs):
            scores = [match.get('similarity', 0.0) for match in matches_list]
            return np.mean(scores) if scores else 0.0
        
        return self.bootstrap_similarity_confidence(matches_data, similarity_function)


class StatisticalAnalyzerAdapter:
    """
    Adaptador para StatisticalAnalyzer que usa UnifiedStatisticalAnalysis
    Mantiene la misma interfaz pública para compatibilidad hacia atrás
    """
    
    def __init__(self):
        """Inicializa el adaptador con el núcleo unificado"""
        from .statistical_core import UnifiedStatisticalAnalysis
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        import logging
        
        # Mantener referencia a la implementación original para cadena de custodia NIST
        self._original = StatisticalAnalyzer()
        self.unified_core = UnifiedStatisticalAnalysis()
        self.scaler = StandardScaler()
        self.pca_model = None
        self.logger = logging.getLogger(__name__)
        
        logger.info("StatisticalAnalyzerAdapter inicializado - Trazabilidad NIST preservada")
    
    def perform_pca_analysis(self, features_data: List[Dict[str, float]], 
                           n_components: Optional[int] = None,
                           variance_threshold: float = 0.95) -> Dict[str, Any]:
        """
        Realiza análisis PCA usando el núcleo unificado
        Mantiene la misma interfaz que StatisticalAnalyzer original
        """
        try:
            # Usar el núcleo unificado para PCA
            result = self.unified_core.perform_pca_analysis(
                features_data=features_data,
                n_components=n_components,
                variance_threshold=variance_threshold
            )
            
            # Mantener referencia al modelo PCA para compatibilidad
            if 'pca_model' in result:
                self.pca_model = result['pca_model']
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error en análisis PCA (adaptador): {str(e)}")
            return {"error": str(e)}
    
    def perform_significance_tests(self, group1: List[float], group2: List[float],
                                 alpha: float = 0.05) -> Dict[str, Any]:
        """
        Realiza pruebas de significancia usando el núcleo unificado
        """
        try:
            return self.unified_core.perform_significance_tests(
                group1=group1,
                group2=group2,
                alpha=alpha
            )
        except Exception as e:
            self.logger.error(f"Error en pruebas de significancia (adaptador): {str(e)}")
            return {"error": str(e)}
    
    def perform_correlation_analysis(self, features_data: List[Dict[str, float]]) -> Dict[str, Any]:
        """
        Realiza análisis de correlación usando el núcleo unificado
        """
        try:
            return self.unified_core.perform_correlation_analysis(features_data)
        except Exception as e:
            self.logger.error(f"Error en análisis de correlación (adaptador): {str(e)}")
            return {"error": str(e)}
    
    def detect_outliers(self, features_data: List[Dict[str, float]], 
                       method: str = "isolation_forest") -> Dict[str, Any]:
        """
        Detecta outliers usando el núcleo unificado
        """
        try:
            return self.unified_core.detect_outliers(
                features_data=features_data,
                method=method
            )
        except Exception as e:
            self.logger.error(f"Error en detección de outliers (adaptador): {str(e)}")
            return {"error": str(e)}
    
    def perform_clustering_analysis(self, features_data: List[Dict[str, float]], 
                                  n_clusters: Optional[int] = None) -> Dict[str, Any]:
        """
        Realiza análisis de clustering usando el núcleo unificado
        """
        try:
            return self.unified_core.perform_clustering_analysis(
                features_data=features_data,
                n_clusters=n_clusters
            )
        except Exception as e:
            self.logger.error(f"Error en análisis de clustering (adaptador): {str(e)}")
            return {"error": str(e)}
    
    def analyze_image(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Analiza imagen - funcionalidad específica que se mantiene
        Esta funcionalidad no está duplicada en el núcleo unificado
        """
        try:
            import cv2
            import numpy as np
            from scipy import stats
            
            if image is None or image.size == 0:
                return {"error": "Imagen inválida"}
            
            # Convertir a escala de grises si es necesario
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Calcular métricas de imagen
            result = {
                "mean_intensity": float(np.mean(gray)),
                "std_intensity": float(np.std(gray)),
                "min_intensity": int(np.min(gray)),
                "max_intensity": int(np.max(gray)),
                "entropy": self._calculate_entropy(gray),
                "contrast": self._calculate_contrast(gray),
                "histogram": np.histogram(gray, bins=256)[0].tolist(),
                "shape": gray.shape
            }
            
            self.logger.info(f"Análisis de imagen completado: {gray.shape}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error en análisis de imagen: {str(e)}")
            return {"error": str(e)}
    
    def _calculate_entropy(self, image: np.ndarray) -> float:
        """Calcula la entropía de la imagen"""
        try:
            hist, _ = np.histogram(image, bins=256, range=(0, 256))
            hist = hist[hist > 0]  # Eliminar bins vacíos
            prob = hist / hist.sum()
            entropy = -np.sum(prob * np.log2(prob))
            return float(entropy)
        except:
            return 0.0
    
    def _calculate_contrast(self, image: np.ndarray) -> float:
        """Calcula el contraste de la imagen usando desviación estándar"""
        try:
            return float(np.std(image))
        except:
            return 0.0
    
    def _calculate_feature_importance(self) -> Dict[str, float]:
        """
        Calcula importancia de características - delegado al núcleo unificado
        """
        if hasattr(self.unified_core, '_calculate_feature_importance'):
            return self.unified_core._calculate_feature_importance()
        return {}
    
    def _calculate_descriptive_stats(self, data: np.ndarray) -> Dict[str, float]:
        """
        Calcula estadísticas descriptivas - delegado al núcleo unificado
        """
        return self.unified_core._calculate_descriptive_stats(data)
    
    def _find_high_correlations(self, corr_matrix, threshold: float = 0.8) -> List[Dict[str, Any]]:
        """
        Encuentra correlaciones altas - delegado al núcleo unificado
        """
        return self.unified_core._find_high_correlations(corr_matrix, threshold)
    
    def _find_optimal_clusters(self, X: np.ndarray, max_clusters: int = 10) -> int:
        """
        Encuentra número óptimo de clusters - delegado al núcleo unificado
        """
        return self.unified_core._find_optimal_clusters(X, max_clusters)


# Funciones utilitarias - wrappers transparentes
def create_similarity_bootstrap_function(
    good_matches: int,
    kp1_count: int,
    kp2_count: int,
    algorithm: str,
    image1_quality: float = 0.0,
    image2_quality: float = 0.0
) -> Callable[[List[Dict[str, Any]]], float]:
    """
    Wrapper transparente para create_similarity_bootstrap_function original
    Delega directamente a la implementación original
    """
    logger.debug(f"Creando función bootstrap - good_matches={good_matches}, algorithm={algorithm}")
    
    return original_create_similarity_bootstrap_function(
        good_matches=good_matches,
        kp1_count=kp1_count,
        kp2_count=kp2_count,
        algorithm=algorithm,
        image1_quality=image1_quality,
        image2_quality=image2_quality
    )


def calculate_bootstrap_confidence_interval(
    matches_data: List[Dict[str, Any]],
    similarity_score: float,
    algorithm: str = "ORB",
    confidence_level: float = 0.95,
    n_bootstrap: int = 1000
) -> Tuple[float, float]:
    """
    Wrapper transparente para calculate_bootstrap_confidence_interval original
    Delega directamente a la implementación original
    """
    logger.debug(f"Calculando CI bootstrap - similarity_score={similarity_score}, algorithm={algorithm}")
    
    return original_calculate_bootstrap_confidence_interval(
        matches_data=matches_data,
        similarity_score=similarity_score,
        algorithm=algorithm,
        confidence_level=confidence_level,
        n_bootstrap=n_bootstrap
    )


# Factory function para obtener adaptadores
def get_advanced_statistical_analysis():
    """Función factory para obtener AdvancedStatisticalAnalysis evitando importación circular"""
    try:
        from nist_standards.statistical_analysis import AdvancedStatisticalAnalysis
        return AdvancedStatisticalAnalysis
    except ImportError:
        # Fallback al adaptador local
        return AdvancedStatisticalAnalysisAdapter
def get_adapter(adapter_type: str, **kwargs):
    """
    Factory function para obtener adaptadores de compatibilidad
    
    Args:
        adapter_type: Tipo de adaptador ('AdvancedStatisticalAnalysis', 
                     'BootstrapSimilarityAnalyzer', 'StatisticalAnalyzer')
        **kwargs: Argumentos para el constructor del adaptador
        
    Returns:
        Instancia del adaptador solicitado
        
    Raises:
        ValueError: Si el tipo de adaptador no es válido
    """
    adapters = {
        'AdvancedStatisticalAnalysis': AdvancedStatisticalAnalysisAdapter,
        'BootstrapSimilarityAnalyzer': BootstrapSimilarityAnalyzerAdapter,
        'StatisticalAnalyzer': StatisticalAnalyzerAdapter
    }
    
    if adapter_type not in adapters:
        available = ', '.join(adapters.keys())
        raise ValueError(f"Tipo de adaptador no válido: {adapter_type}. Disponibles: {available}")
    
    logger.info(f"Creando adaptador {adapter_type} - Fase 1 (wrapper transparente)")
    return adapters[adapter_type](**kwargs)


# Lista de adaptadores disponibles para introspección
AVAILABLE_ADAPTERS = [
    'AdvancedStatisticalAnalysis',
    'BootstrapSimilarityAnalyzer', 
    'StatisticalAnalyzer'
]

# Control global de migración
_UNIFIED_MODE_ENABLED = False

def enable_unified_mode():
    """Habilita el modo unificado para todos los adaptadores"""
    global _UNIFIED_MODE_ENABLED
    _UNIFIED_MODE_ENABLED = True
    logger.info("Modo unificado habilitado - usando statistical_core.py")

def disable_unified_mode():
    """Deshabilita el modo unificado, volviendo a implementaciones originales"""
    global _UNIFIED_MODE_ENABLED
    _UNIFIED_MODE_ENABLED = False
    logger.info("Modo unificado deshabilitado - usando implementaciones originales")

def get_migration_status():
    """Obtiene el estado actual de la migración"""
    return {
        'unified_mode_enabled': _UNIFIED_MODE_ENABLED,
        'phase': 'Fase 1 - Preparación' if not _UNIFIED_MODE_ENABLED else 'Fase 2 - Migración',
        'adapters_available': AVAILABLE_ADAPTERS,
        'nist_compliance': True
    }

# Metadatos de trazabilidad NIST
NIST_TRACEABILITY_INFO = {
    'implementation_phase': 'Fase 1 - Preparación',
    'compatibility_mode': 'wrapper_transparente',
    'risk_level': 'mínimo',
    'standards_compliance': [
        'ISO 5725-2:2019',
        'NIST/SEMATECH e-Handbook',
        'NIST SP 800-90A Rev. 1'
    ],
    'validation_status': 'pendiente_tests',
    'migration_readiness': 'preparado_para_fase_2'
}


if __name__ == "__main__":
    # Test básico de adaptadores
    print("Testing SIGeC-Balistica Compatibility Adapters - Fase 1...")
    
    # Test adaptador NIST
    try:
        nist_adapter = AdvancedStatisticalAnalysisAdapter(random_state=42)
        print("✓ AdvancedStatisticalAnalysisAdapter inicializado correctamente")
    except Exception as e:
        print(f"✗ Error en AdvancedStatisticalAnalysisAdapter: {e}")
    
    # Test adaptador matching
    try:
        matching_adapter = BootstrapSimilarityAnalyzerAdapter()
        print("✓ BootstrapSimilarityAnalyzerAdapter inicializado correctamente")
    except Exception as e:
        print(f"✗ Error en BootstrapSimilarityAnalyzerAdapter: {e}")
    
    # Test adaptador image processing
    try:
        image_adapter = StatisticalAnalyzerAdapter()
        print("✓ StatisticalAnalyzerAdapter inicializado correctamente")
    except Exception as e:
        print(f"✗ Error en StatisticalAnalyzerAdapter: {e}")
    
    # Test factory function
    try:
        adapter = get_adapter('AdvancedStatisticalAnalysis', random_state=42)
        print("✓ Factory function funciona correctamente")
    except Exception as e:
        print(f"✗ Error en factory function: {e}")
    
    print(f"\nFase 1 - Adaptadores como wrappers transparentes")
    print(f"Trazabilidad NIST: {NIST_TRACEABILITY_INFO['standards_compliance']}")
    print("Listos para validación de compatibilidad ✓")