#!/usr/bin/env python3
"""
Módulo Unificado de Funciones de Similitud
Sistema Balístico Forense SIGeC-Balisticar

Consolida todas las funciones de similitud duplicadas encontradas en:
- matching/bootstrap_similarity.py
- gui/comparison_worker.py  
- matching/unified_matcher.py
- common/statistical_core.py

Elimina duplicación y proporciona interfaz unificada
"""

import numpy as np
import logging
import time
from typing import List, Dict, Tuple, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

# Imports del sistema unificado
from common.statistical_core import UnifiedStatisticalAnalysis, BootstrapResult
from config.unified_config import get_unified_config

logger = logging.getLogger(__name__)

@dataclass
class SimilarityBootstrapResult:
    """Resultado unificado del análisis bootstrap para métricas de similitud"""
    similarity_score: float
    confidence_interval: Tuple[float, float]
    confidence_level: float
    bootstrap_scores: np.ndarray
    bias: float
    standard_error: float
    percentile_ci: Tuple[float, float]
    bca_ci: Optional[Tuple[float, float]]
    n_bootstrap: int
    method: str
    processing_time: float
    
    # Métricas específicas de matching
    match_statistics: Dict[str, float] = field(default_factory=dict)
    quality_weighted_ci: Optional[Tuple[float, float]] = None
    geometric_consistency_ci: Optional[Tuple[float, float]] = None

@dataclass
class SimilarityConfig:
    """Configuración unificada para análisis de similitud"""
    # Bootstrap configuration
    n_bootstrap: int = 1000
    confidence_level: float = 0.95
    method: str = 'bca'  # 'percentile', 'basic', 'bca'
    parallel: bool = True
    max_workers: int = 4
    
    # Similarity calculation parameters
    min_matches_for_bootstrap: int = 5
    include_quality_weighting: bool = True
    include_geometric_consistency: bool = True
    
    # Algorithm-specific adjustments
    algorithm_factors: Dict[str, float] = field(default_factory=lambda: {
        'ORB': 1.2,
        'SIFT': 1.1,
        'AKAZE': 1.15,
        'SURF': 1.05
    })
    
    # Quality thresholds
    high_similarity_threshold: float = 0.7
    medium_similarity_threshold: float = 0.4
    
    # Performance limits
    max_processing_time: float = 30.0  # seconds


class UnifiedSimilarityAnalyzer:
    """
    Analizador unificado de similitud que consolida toda la funcionalidad duplicada
    """
    
    def __init__(self, config: Optional[SimilarityConfig] = None):
        """
        Inicializar analizador unificado de similitud
        
        Args:
            config: Configuración del analizador (opcional)
        """
        self.config = config or SimilarityConfig()
        self.logger = logging.getLogger(__name__)
        
        # Inicializar análisis estadístico unificado
        self.statistical_analysis = UnifiedStatisticalAnalysis()
        
        # Cargar configuración del sistema
        self.system_config = get_unified_config()
        
        self.logger.info("Unified similarity analyzer initialized")

    def calculate_basic_similarity(
        self,
        good_matches: int,
        total_features_a: int,
        total_features_b: int,
        algorithm: str = "SIFT",
        image_quality_a: float = 1.0,
        image_quality_b: float = 1.0
    ) -> float:
        """
        Calcular similitud básica unificada
        
        Args:
            good_matches: Número de matches buenos
            total_features_a: Total de características imagen A
            total_features_b: Total de características imagen B
            algorithm: Algoritmo utilizado
            image_quality_a: Calidad de imagen A (0-1)
            image_quality_b: Calidad de imagen B (0-1)
            
        Returns:
            Score de similitud (0-100)
        """
        if total_features_a == 0 or total_features_b == 0:
            return 0.0
        
        # Score base: proporción de matches buenos
        max_possible_matches = min(total_features_a, total_features_b)
        base_score = good_matches / max_possible_matches
        
        # Aplicar factor específico del algoritmo
        algorithm_factor = self.config.algorithm_factors.get(algorithm, 1.0)
        if good_matches >= self._get_min_matches_for_algorithm(algorithm):
            base_score *= algorithm_factor
        
        # Aplicar factor de calidad de imagen
        if self.config.include_quality_weighting:
            quality_factor = self._calculate_quality_factor(image_quality_a, image_quality_b)
            base_score *= quality_factor
        
        # Normalizar a 0-100 y limitar
        similarity_score = min(base_score * 100, 100.0)
        
        return similarity_score

    def calculate_similarity_with_distances(
        self,
        matches_data: List[Dict[str, Any]],
        total_features_a: int,
        total_features_b: int,
        algorithm: str = "SIFT",
        image_quality_a: float = 1.0,
        image_quality_b: float = 1.0
    ) -> Dict[str, Any]:
        """
        Calcular similitud considerando distancias de matches
        
        Args:
            matches_data: Lista de matches con distancias
            total_features_a: Total de características imagen A
            total_features_b: Total de características imagen B
            algorithm: Algoritmo utilizado
            image_quality_a: Calidad de imagen A
            image_quality_b: Calidad de imagen B
            
        Returns:
            Diccionario con análisis completo de similitud
        """
        num_good_matches = len(matches_data)
        
        # Calcular similitud básica
        basic_similarity = self.calculate_basic_similarity(
            num_good_matches, total_features_a, total_features_b,
            algorithm, image_quality_a, image_quality_b
        )
        
        # Analizar distancias de matches
        distances = [match.get('distance', 1.0) for match in matches_data if 'distance' in match]
        
        distance_analysis = {}
        if distances:
            distance_analysis = {
                'mean_distance': np.mean(distances),
                'std_distance': np.std(distances),
                'min_distance': np.min(distances),
                'max_distance': np.max(distances),
                'median_distance': np.median(distances)
            }
            
            # Ajustar similitud por calidad de distancias
            # Distancias menores indican mejores matches
            distance_factor = max(0.1, 1.0 - distance_analysis['mean_distance'])
            basic_similarity *= distance_factor
        
        # Clasificar nivel de similitud
        similarity_level, similarity_color = self._classify_similarity_level(basic_similarity)
        
        return {
            'similarity_score': basic_similarity,
            'similarity_level': similarity_level,
            'similarity_color': similarity_color,
            'num_good_matches': num_good_matches,
            'total_features_a': total_features_a,
            'total_features_b': total_features_b,
            'distance_analysis': distance_analysis,
            'algorithm': algorithm,
            'quality_factor': self._calculate_quality_factor(image_quality_a, image_quality_b)
        }

    def bootstrap_similarity_confidence(
        self,
        matches_data: List[Dict[str, Any]],
        similarity_function: Optional[Callable] = None,
        **kwargs
    ) -> SimilarityBootstrapResult:
        """
        Calcular intervalo de confianza bootstrap para similitud
        
        Args:
            matches_data: Datos de matches
            similarity_function: Función de similitud personalizada (opcional)
            **kwargs: Argumentos adicionales
            
        Returns:
            Resultado del análisis bootstrap
        """
        start_time = time.time()
        
        if len(matches_data) < self.config.min_matches_for_bootstrap:
            return self._create_fallback_result(matches_data, similarity_function, **kwargs)
        
        # Usar función de similitud por defecto si no se proporciona
        if similarity_function is None:
            similarity_function = self._create_default_similarity_function(**kwargs)
        
        # Crear función adaptada para bootstrap
        def adapted_statistic_func(indices_array):
            resampled_matches = [matches_data[i % len(matches_data)] for i in indices_array]
            return similarity_function(resampled_matches)
        
        # Ejecutar bootstrap
        indices_data = np.arange(len(matches_data))
        bootstrap_result = self.statistical_analysis.bootstrap_sampling(
            data=indices_data,
            statistic_func=adapted_statistic_func,
            n_bootstrap=self.config.n_bootstrap,
            confidence_level=self.config.confidence_level,
            method=self.config.method,
            parallel=self.config.parallel
        )
        
        processing_time = time.time() - start_time
        
        # Calcular estadísticas específicas de matching
        match_statistics = self._calculate_match_statistics(matches_data)
        
        # Crear resultado unificado
        result = SimilarityBootstrapResult(
            similarity_score=bootstrap_result.original_statistic,
            confidence_interval=bootstrap_result.confidence_interval,
            confidence_level=bootstrap_result.confidence_level,
            bootstrap_scores=bootstrap_result.bootstrap_statistics,
            bias=bootstrap_result.bias,
            standard_error=bootstrap_result.standard_error,
            percentile_ci=bootstrap_result.percentile_ci,
            bca_ci=bootstrap_result.bca_ci,
            n_bootstrap=bootstrap_result.n_bootstrap,
            method=self.config.method,
            processing_time=processing_time,
            match_statistics=match_statistics
        )
        
        # Calcular intervalos de confianza adicionales si están habilitados
        if self.config.include_quality_weighting:
            result.quality_weighted_ci = self._calculate_quality_weighted_ci(
                matches_data, bootstrap_result
            )
        
        if self.config.include_geometric_consistency:
            result.geometric_consistency_ci = self._calculate_geometric_consistency_ci(
                matches_data, bootstrap_result
            )
        
        return result

    def create_similarity_function(
        self,
        good_matches: int,
        kp1_count: int,
        kp2_count: int,
        algorithm: str,
        image1_quality: float = 1.0,
        image2_quality: float = 1.0
    ) -> Callable[[List[Dict[str, Any]]], float]:
        """
        Crear función de similitud unificada para bootstrap
        
        Args:
            good_matches: Número de buenos matches
            kp1_count: Número de keypoints imagen 1
            kp2_count: Número de keypoints imagen 2
            algorithm: Algoritmo utilizado
            image1_quality: Calidad imagen 1
            image2_quality: Calidad imagen 2
            
        Returns:
            Función de similitud para bootstrap
        """
        def similarity_function(matches_data: List[Dict[str, Any]]) -> float:
            if not matches_data or kp1_count == 0 or kp2_count == 0:
                return 0.0
            
            # Usar el método unificado de cálculo de similitud
            result = self.calculate_similarity_with_distances(
                matches_data, kp1_count, kp2_count, algorithm,
                image1_quality, image2_quality
            )
            
            return result['similarity_score']
        
        return similarity_function

    def analyze_bootstrap_confidence_interval(
        self,
        matches_data: List[Dict[str, Any]],
        base_confidence: float,
        algorithm: str = "SIFT",
        confidence_level: float = 0.95,
        n_bootstrap: int = 500
    ) -> Tuple[float, float, Dict[str, Any]]:
        """
        Analizar intervalo de confianza bootstrap para ajustar confianza base
        
        Args:
            matches_data: Datos de matches
            base_confidence: Confianza base
            algorithm: Algoritmo utilizado
            confidence_level: Nivel de confianza
            n_bootstrap: Número de muestras bootstrap
            
        Returns:
            Tupla con (ci_lower, ci_upper, bootstrap_info)
        """
        if len(matches_data) < self.config.min_matches_for_bootstrap:
            # Fallback para pocos matches
            margin = base_confidence * 0.1  # 10% de margen
            return (
                max(0, base_confidence - margin),
                min(100, base_confidence + margin),
                {'bootstrap_used': False, 'reason': 'insufficient_matches'}
            )
        
        # Crear función de similitud para este análisis
        similarity_function = lambda matches: base_confidence * (len(matches) / len(matches_data))
        
        # Configurar bootstrap temporal
        original_n_bootstrap = self.config.n_bootstrap
        self.config.n_bootstrap = n_bootstrap
        
        try:
            # Ejecutar bootstrap
            bootstrap_result = self.bootstrap_similarity_confidence(
                matches_data, similarity_function
            )
            
            ci_lower, ci_upper = bootstrap_result.confidence_interval
            
            bootstrap_info = {
                'confidence_interval_lower': ci_lower,
                'confidence_interval_upper': ci_upper,
                'confidence_interval_method': bootstrap_result.method,
                'bootstrap_samples': bootstrap_result.n_bootstrap,
                'bootstrap_confidence_level': bootstrap_result.confidence_level,
                'bootstrap_bias': bootstrap_result.bias,
                'bootstrap_std_error': bootstrap_result.standard_error,
                'bootstrap_used': True,
                'processing_time': bootstrap_result.processing_time
            }
            
            return ci_lower, ci_upper, bootstrap_info
            
        finally:
            # Restaurar configuración original
            self.config.n_bootstrap = original_n_bootstrap

    def _get_min_matches_for_algorithm(self, algorithm: str) -> int:
        """Obtener número mínimo de matches para aplicar factor de algoritmo"""
        min_matches = {
            'ORB': 10,
            'SIFT': 20,
            'AKAZE': 8,
            'SURF': 15
        }
        return min_matches.get(algorithm, 10)

    def _calculate_quality_factor(self, quality_a: float, quality_b: float) -> float:
        """Calcular factor de calidad combinado"""
        if quality_a <= 0 or quality_b <= 0:
            return 1.0
        
        # Media armónica para penalizar calidades muy diferentes
        combined_quality = 2 * (quality_a * quality_b) / (quality_a + quality_b)
        
        # Factor entre 0.5 y 1.5
        return 0.5 + combined_quality

    def _classify_similarity_level(self, similarity_score: float) -> Tuple[str, str]:
        """Clasificar nivel de similitud y color asociado"""
        if similarity_score >= self.config.high_similarity_threshold * 100:
            return "Alta", "green"
        elif similarity_score >= self.config.medium_similarity_threshold * 100:
            return "Media", "yellow"
        else:
            return "Baja", "red"

    def _calculate_match_statistics(self, matches_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calcular estadísticas específicas de matches"""
        if not matches_data:
            return {}
        
        distances = [match.get('distance', 1.0) for match in matches_data if 'distance' in match]
        
        stats = {
            'num_matches': len(matches_data),
            'mean_distance': np.mean(distances) if distances else 0.0,
            'std_distance': np.std(distances) if distances else 0.0,
            'min_distance': np.min(distances) if distances else 0.0,
            'max_distance': np.max(distances) if distances else 0.0
        }
        
        # Calcular métricas de calidad
        if distances:
            # Porcentaje de matches de alta calidad (distancia < 0.3)
            high_quality_matches = sum(1 for d in distances if d < 0.3)
            stats['high_quality_ratio'] = high_quality_matches / len(distances)
            
            # Consistencia de distancias (menor std = más consistente)
            stats['distance_consistency'] = 1.0 / (1.0 + stats['std_distance'])
        
        return stats

    def _create_default_similarity_function(self, **kwargs) -> Callable:
        """Crear función de similitud por defecto"""
        algorithm = kwargs.get('algorithm', 'SIFT')
        kp1_count = kwargs.get('kp1_count', 100)
        kp2_count = kwargs.get('kp2_count', 100)
        image1_quality = kwargs.get('image1_quality', 1.0)
        image2_quality = kwargs.get('image2_quality', 1.0)
        
        return self.create_similarity_function(
            len(kwargs.get('matches_data', [])),
            kp1_count, kp2_count, algorithm,
            image1_quality, image2_quality
        )

    def _create_fallback_result(
        self,
        matches_data: List[Dict[str, Any]],
        similarity_function: Optional[Callable],
        **kwargs
    ) -> SimilarityBootstrapResult:
        """Crear resultado fallback para casos con pocos matches"""
        if similarity_function is None:
            similarity_function = self._create_default_similarity_function(**kwargs)
        
        similarity_score = similarity_function(matches_data)
        
        # Crear intervalo de confianza amplio para reflejar incertidumbre
        margin = similarity_score * 0.2  # 20% de margen
        ci_lower = max(0, similarity_score - margin)
        ci_upper = min(100, similarity_score + margin)
        
        return SimilarityBootstrapResult(
            similarity_score=similarity_score,
            confidence_interval=(ci_lower, ci_upper),
            confidence_level=self.config.confidence_level,
            bootstrap_scores=np.array([similarity_score]),
            bias=0.0,
            standard_error=margin / 2,
            percentile_ci=(ci_lower, ci_upper),
            bca_ci=None,
            n_bootstrap=1,
            method='fallback',
            processing_time=0.001,
            match_statistics=self._calculate_match_statistics(matches_data)
        )

    def _calculate_quality_weighted_ci(
        self,
        matches_data: List[Dict[str, Any]],
        bootstrap_result: BootstrapResult
    ) -> Optional[Tuple[float, float]]:
        """Calcular intervalo de confianza ponderado por calidad"""
        try:
            # Implementación simplificada
            # En una implementación completa, esto consideraría la calidad de cada match
            distances = [match.get('distance', 1.0) for match in matches_data if 'distance' in match]
            if not distances:
                return None
            
            quality_factor = 1.0 - np.mean(distances)  # Mejor calidad = menor distancia
            
            ci_lower, ci_upper = bootstrap_result.confidence_interval
            ci_width = ci_upper - ci_lower
            
            # Ajustar ancho del intervalo según calidad
            adjusted_width = ci_width * (2.0 - quality_factor)  # Mejor calidad = intervalo más estrecho
            
            center = (ci_lower + ci_upper) / 2
            return (
                max(0, center - adjusted_width / 2),
                min(100, center + adjusted_width / 2)
            )
            
        except Exception as e:
            self.logger.warning(f"Error calculating quality weighted CI: {e}")
            return None

    def _calculate_geometric_consistency_ci(
        self,
        matches_data: List[Dict[str, Any]],
        bootstrap_result: BootstrapResult
    ) -> Optional[Tuple[float, float]]:
        """Calcular intervalo de confianza basado en consistencia geométrica"""
        try:
            # Implementación simplificada
            # En una implementación completa, esto analizaría la consistencia geométrica
            if len(matches_data) < 4:  # Mínimo para análisis geométrico
                return None
            
            # Simular factor de consistencia geométrica
            consistency_factor = min(1.0, len(matches_data) / 20.0)  # Más matches = mayor consistencia
            
            ci_lower, ci_upper = bootstrap_result.confidence_interval
            ci_width = ci_upper - ci_lower
            
            # Ajustar según consistencia
            adjusted_width = ci_width * (2.0 - consistency_factor)
            
            center = (ci_lower + ci_upper) / 2
            return (
                max(0, center - adjusted_width / 2),
                min(100, center + adjusted_width / 2)
            )
            
        except Exception as e:
            self.logger.warning(f"Error calculating geometric consistency CI: {e}")
            return None


# ============================================================================
# FUNCIONES DE COMPATIBILIDAD HACIA ATRÁS
# ============================================================================

# Instancia global del analizador unificado
_unified_analyzer = None

def get_unified_similarity_analyzer() -> UnifiedSimilarityAnalyzer:
    """Obtener instancia global del analizador unificado"""
    global _unified_analyzer
    if _unified_analyzer is None:
        _unified_analyzer = UnifiedSimilarityAnalyzer()
    return _unified_analyzer


def create_similarity_bootstrap_function(
    good_matches: int,
    kp1_count: int,
    kp2_count: int,
    algorithm: str,
    image1_quality: float = 1.0,
    image2_quality: float = 1.0
) -> Callable[[List[Dict[str, Any]]], float]:
    """
    Función de compatibilidad para crear función de similitud bootstrap
    Mantiene compatibilidad con código existente
    """
    analyzer = get_unified_similarity_analyzer()
    return analyzer.create_similarity_function(
        good_matches, kp1_count, kp2_count, algorithm,
        image1_quality, image2_quality
    )


def calculate_basic_similarity(
    good_matches: int,
    total_features_a: int,
    total_features_b: int,
    algorithm: str = "SIFT"
) -> float:
    """
    Función de compatibilidad para cálculo básico de similitud
    """
    analyzer = get_unified_similarity_analyzer()
    return analyzer.calculate_basic_similarity(
        good_matches, total_features_a, total_features_b, algorithm
    )


def bootstrap_similarity_confidence(
    matches_data: List[Dict[str, Any]],
    similarity_function: Callable[[List[Dict[str, Any]]], float],
    **kwargs
) -> SimilarityBootstrapResult:
    """
    Función de compatibilidad para análisis bootstrap de confianza
    """
    analyzer = get_unified_similarity_analyzer()
    return analyzer.bootstrap_similarity_confidence(matches_data, similarity_function, **kwargs)


# ============================================================================
# UTILIDADES ADICIONALES
# ============================================================================

def validate_similarity_config(config: SimilarityConfig) -> List[str]:
    """
    Validar configuración de similitud
    
    Returns:
        Lista de errores de validación (vacía si es válida)
    """
    errors = []
    
    if config.n_bootstrap < 100:
        errors.append("n_bootstrap should be at least 100")
    
    if not 0.8 <= config.confidence_level <= 0.99:
        errors.append("confidence_level should be between 0.8 and 0.99")
    
    if config.method not in ['percentile', 'basic', 'bca']:
        errors.append("method should be one of: percentile, basic, bca")
    
    if config.max_workers < 1:
        errors.append("max_workers should be at least 1")
    
    if config.min_matches_for_bootstrap < 3:
        errors.append("min_matches_for_bootstrap should be at least 3")
    
    return errors


def benchmark_similarity_methods(
    matches_data: List[Dict[str, Any]],
    n_iterations: int = 10
) -> Dict[str, float]:
    """
    Benchmark de diferentes métodos de similitud
    
    Returns:
        Diccionario con tiempos de ejecución por método
    """
    analyzer = get_unified_similarity_analyzer()
    results = {}
    
    # Test basic similarity
    start_time = time.time()
    for _ in range(n_iterations):
        analyzer.calculate_basic_similarity(
            len(matches_data), 100, 100, "SIFT"
        )
    results['basic_similarity'] = (time.time() - start_time) / n_iterations
    
    # Test similarity with distances
    start_time = time.time()
    for _ in range(n_iterations):
        analyzer.calculate_similarity_with_distances(
            matches_data, 100, 100, "SIFT"
        )
    results['similarity_with_distances'] = (time.time() - start_time) / n_iterations
    
    # Test bootstrap (fewer iterations due to computational cost)
    if len(matches_data) >= analyzer.config.min_matches_for_bootstrap:
        start_time = time.time()
        for _ in range(min(3, n_iterations)):
            analyzer.bootstrap_similarity_confidence(matches_data)
        results['bootstrap_similarity'] = (time.time() - start_time) / min(3, n_iterations)
    
    return results


if __name__ == "__main__":
    # Test básico del módulo
    print("🔧 Testing Unified Similarity Functions Module")
    
    # Crear datos de test
    test_matches = [
        {'distance': 0.1, 'kp1_idx': 0, 'kp2_idx': 0},
        {'distance': 0.2, 'kp1_idx': 1, 'kp2_idx': 1},
        {'distance': 0.15, 'kp1_idx': 2, 'kp2_idx': 2},
        {'distance': 0.3, 'kp1_idx': 3, 'kp2_idx': 3},
        {'distance': 0.25, 'kp1_idx': 4, 'kp2_idx': 4}
    ]
    
    # Test analizador
    analyzer = UnifiedSimilarityAnalyzer()
    
    # Test similitud básica
    basic_sim = analyzer.calculate_basic_similarity(5, 100, 100, "SIFT")
    print(f"Basic similarity: {basic_sim:.2f}")
    
    # Test similitud con distancias
    sim_result = analyzer.calculate_similarity_with_distances(
        test_matches, 100, 100, "SIFT"
    )
    print(f"Similarity with distances: {sim_result['similarity_score']:.2f}")
    
    # Test bootstrap
    bootstrap_result = analyzer.bootstrap_similarity_confidence(test_matches)
    print(f"Bootstrap similarity: {bootstrap_result.similarity_score:.2f} "
          f"CI: [{bootstrap_result.confidence_interval[0]:.2f}, {bootstrap_result.confidence_interval[1]:.2f}]")
    
    print("✅ Unified Similarity Functions Module test completed")