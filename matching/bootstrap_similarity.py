#!/usr/bin/env python3
"""
Bootstrap Similarity Analysis Module for SEACABAr
Integra bootstrap sampling con métricas de similitud para intervalos de confianza robustos
"""

import numpy as np
import logging
from typing import List, Dict, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Importar análisis estadístico unificado
try:
    # Usar el núcleo estadístico unificado directamente
    from common.statistical_core import UnifiedStatisticalAnalysis, BootstrapResult
    
    # Crear instancia del núcleo unificado
    statistical_analysis = UnifiedStatisticalAnalysis()
    UNIFIED_CORE_AVAILABLE = True
    
except ImportError:
    # Fallback a adaptadores de compatibilidad
    try:
        from common.compatibility_adapters import AdvancedStatisticalAnalysisAdapter
        from common.statistical_core import BootstrapResult
        
        # Crear instancia del adaptador (que internamente usa el núcleo unificado)
        statistical_analysis = AdvancedStatisticalAnalysisAdapter()
        UNIFIED_CORE_AVAILABLE = True
        
    except ImportError:
        # Último fallback a imports legacy
        try:
            from nist_standards.statistical_analysis import AdvancedStatisticalAnalysis, BootstrapResult
            statistical_analysis = AdvancedStatisticalAnalysis()
            UNIFIED_CORE_AVAILABLE = False
        except ImportError:
            warnings.warn("NIST statistical analysis not available. Bootstrap functionality will be limited.")
            statistical_analysis = None
            BootstrapResult = None
            UNIFIED_CORE_AVAILABLE = False

@dataclass
class SimilarityBootstrapResult:
    """Resultado del análisis bootstrap para métricas de similitud"""
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
class MatchingBootstrapConfig:
    """Configuración para bootstrap de métricas de matching"""
    n_bootstrap: int = 1000
    confidence_level: float = 0.95
    method: str = 'bca'  # 'percentile', 'basic', 'bca'
    parallel: bool = True
    max_workers: int = 4
    stratified: bool = False
    include_quality_weighting: bool = True
    include_geometric_consistency: bool = True
    min_matches_for_bootstrap: int = 5

class BootstrapSimilarityAnalyzer:
    """
    Analizador de similitud con bootstrap sampling para intervalos de confianza robustos
    """
    
    def __init__(self, config: Optional[MatchingBootstrapConfig] = None):
        """
        Inicializar analizador bootstrap para similitud
        
        Args:
            config: Configuración del bootstrap (opcional)
        """
        self.config = config or MatchingBootstrapConfig()
        self.logger = logging.getLogger(__name__)
        
        # Usar instancia global del análisis estadístico unificado
        self.statistical_analysis = statistical_analysis
        
        if not UNIFIED_CORE_AVAILABLE:
            self.logger.warning("Unified statistical core not available. Some features may be limited.")
        else:
            self.logger.info("Bootstrap similarity analyzer initialized with unified statistical core")
    
    def bootstrap_similarity_confidence(
        self,
        matches_data: List[Dict[str, Any]],
        similarity_function: Callable[[List[Dict[str, Any]]], float],
        **kwargs
    ) -> SimilarityBootstrapResult:
        """
        Calcular intervalos de confianza bootstrap para similarity scores
        
        Args:
            matches_data: Lista de datos de matches
            similarity_function: Función para calcular similitud
            **kwargs: Argumentos adicionales para la función de similitud
            
        Returns:
            SimilarityBootstrapResult con intervalos de confianza
        """
        start_time = time.time()
        
        if len(matches_data) < self.config.min_matches_for_bootstrap:
            self.logger.warning(f"Insufficient matches for bootstrap: {len(matches_data)}")
            return self._create_fallback_result(matches_data, similarity_function, **kwargs)
        
        try:
            # Calcular similitud original
            original_similarity = similarity_function(matches_data, **kwargs)
            
            # Generar muestras bootstrap
            if self.config.parallel and self.config.n_bootstrap > 100:
                bootstrap_scores = self._parallel_bootstrap_similarity(
                    matches_data, similarity_function, **kwargs
                )
            else:
                bootstrap_scores = self._sequential_bootstrap_similarity(
                    matches_data, similarity_function, **kwargs
                )
            
            # Calcular estadísticas bootstrap
            bias = np.mean(bootstrap_scores) - original_similarity
            standard_error = np.std(bootstrap_scores, ddof=1)
            
            # Calcular intervalos de confianza
            alpha = 1 - self.config.confidence_level
            
            # Intervalo percentil
            percentile_ci = self._percentile_confidence_interval(bootstrap_scores, alpha)
            
            # Intervalo BCA si está disponible y se solicita
            bca_ci = None
            if self.config.method == 'bca' and self.statistical_analysis:
                try:
                    bca_ci = self._bca_confidence_interval_similarity(
                        matches_data, similarity_function, bootstrap_scores, 
                        original_similarity, alpha, **kwargs
                    )
                except Exception as e:
                    self.logger.warning(f"BCA CI calculation failed: {e}")
                    bca_ci = percentile_ci
            
            # Seleccionar CI según método
            if self.config.method == 'bca' and bca_ci is not None:
                confidence_interval = bca_ci
            elif self.config.method == 'basic':
                confidence_interval = self._basic_confidence_interval(
                    bootstrap_scores, original_similarity, alpha
                )
            else:  # percentile
                confidence_interval = percentile_ci
            
            # Calcular estadísticas de matches
            match_statistics = self._calculate_match_statistics(matches_data)
            
            # Intervalos de confianza adicionales si se solicitan
            quality_weighted_ci = None
            geometric_consistency_ci = None
            
            if self.config.include_quality_weighting:
                quality_weighted_ci = self._bootstrap_quality_weighted_similarity(
                    matches_data, **kwargs
                )
            
            if self.config.include_geometric_consistency:
                geometric_consistency_ci = self._bootstrap_geometric_consistency(
                    matches_data, **kwargs
                )
            
            processing_time = time.time() - start_time
            
            return SimilarityBootstrapResult(
                similarity_score=original_similarity,
                confidence_interval=confidence_interval,
                confidence_level=self.config.confidence_level,
                bootstrap_scores=bootstrap_scores,
                bias=bias,
                standard_error=standard_error,
                percentile_ci=percentile_ci,
                bca_ci=bca_ci,
                n_bootstrap=self.config.n_bootstrap,
                method=self.config.method,
                processing_time=processing_time,
                match_statistics=match_statistics,
                quality_weighted_ci=quality_weighted_ci,
                geometric_consistency_ci=geometric_consistency_ci
            )
            
        except Exception as e:
            self.logger.error(f"Error in bootstrap similarity analysis: {e}")
            return self._create_fallback_result(matches_data, similarity_function, **kwargs)
    
    def _parallel_bootstrap_similarity(
        self,
        matches_data: List[Dict[str, Any]],
        similarity_function: Callable,
        **kwargs
    ) -> np.ndarray:
        """Ejecutar bootstrap en paralelo"""
        bootstrap_scores = []
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Dividir trabajo entre workers
            chunk_size = max(1, self.config.n_bootstrap // self.config.max_workers)
            futures = []
            
            for i in range(0, self.config.n_bootstrap, chunk_size):
                n_samples = min(chunk_size, self.config.n_bootstrap - i)
                future = executor.submit(
                    self._bootstrap_chunk, matches_data, similarity_function, 
                    n_samples, **kwargs
                )
                futures.append(future)
            
            # Recopilar resultados
            for future in as_completed(futures):
                try:
                    chunk_scores = future.result()
                    bootstrap_scores.extend(chunk_scores)
                except Exception as e:
                    self.logger.warning(f"Bootstrap chunk failed: {e}")
        
        return np.array(bootstrap_scores[:self.config.n_bootstrap])
    
    def _sequential_bootstrap_similarity(
        self,
        matches_data: List[Dict[str, Any]],
        similarity_function: Callable,
        **kwargs
    ) -> np.ndarray:
        """Ejecutar bootstrap secuencial"""
        return self._bootstrap_chunk(
            matches_data, similarity_function, self.config.n_bootstrap, **kwargs
        )
    
    def _bootstrap_chunk(
        self,
        matches_data: List[Dict[str, Any]],
        similarity_function: Callable,
        n_samples: int,
        **kwargs
    ) -> List[float]:
        """Ejecutar un chunk de muestras bootstrap"""
        bootstrap_scores = []
        
        for _ in range(n_samples):
            # Resample con reemplazo
            if self.config.stratified:
                resampled_data = self._stratified_resample(matches_data)
            else:
                indices = np.random.choice(len(matches_data), size=len(matches_data), replace=True)
                resampled_data = [matches_data[i] for i in indices]
            
            # Calcular similitud para la muestra bootstrap
            try:
                bootstrap_score = similarity_function(resampled_data, **kwargs)
                bootstrap_scores.append(bootstrap_score)
            except Exception as e:
                self.logger.debug(f"Bootstrap sample failed: {e}")
                # Usar score original como fallback
                bootstrap_scores.append(similarity_function(matches_data, **kwargs))
        
        return bootstrap_scores
    
    def _stratified_resample(self, matches_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Resample estratificado basado en calidad de matches"""
        if not matches_data:
            return matches_data
        
        # Estratificar por distancia de match (calidad)
        distances = [match.get('distance', 0) for match in matches_data]
        median_distance = np.median(distances)
        
        high_quality = [match for match in matches_data if match.get('distance', 0) <= median_distance]
        low_quality = [match for match in matches_data if match.get('distance', 0) > median_distance]
        
        # Mantener proporción original
        n_high = len(high_quality)
        n_low = len(low_quality)
        total = len(matches_data)
        
        if n_high > 0 and n_low > 0:
            # Resample manteniendo proporción
            n_high_sample = int((n_high / total) * total)
            n_low_sample = total - n_high_sample
            
            high_indices = np.random.choice(n_high, size=n_high_sample, replace=True)
            low_indices = np.random.choice(n_low, size=n_low_sample, replace=True)
            
            resampled = ([high_quality[i] for i in high_indices] + 
                        [low_quality[i] for i in low_indices])
            np.random.shuffle(resampled)
            return resampled
        else:
            # Fallback a resample simple
            indices = np.random.choice(len(matches_data), size=len(matches_data), replace=True)
            return [matches_data[i] for i in indices]
    
    def _percentile_confidence_interval(
        self, 
        bootstrap_scores: np.ndarray, 
        alpha: float
    ) -> Tuple[float, float]:
        """Calcular intervalo de confianza percentil"""
        lower_percentile = 100 * (alpha / 2)
        upper_percentile = 100 * (1 - alpha / 2)
        
        lower_bound = np.percentile(bootstrap_scores, lower_percentile)
        upper_bound = np.percentile(bootstrap_scores, upper_percentile)
        
        return (lower_bound, upper_bound)
    
    def _basic_confidence_interval(
        self,
        bootstrap_scores: np.ndarray,
        original_score: float,
        alpha: float
    ) -> Tuple[float, float]:
        """Calcular intervalo de confianza básico"""
        lower_percentile = 100 * (alpha / 2)
        upper_percentile = 100 * (1 - alpha / 2)
        
        lower_bootstrap = np.percentile(bootstrap_scores, lower_percentile)
        upper_bootstrap = np.percentile(bootstrap_scores, upper_percentile)
        
        # Intervalo básico: 2*θ̂ - θ*_upper, 2*θ̂ - θ*_lower
        lower_bound = 2 * original_score - upper_bootstrap
        upper_bound = 2 * original_score - lower_bootstrap
        
        return (lower_bound, upper_bound)
    
    def _bca_confidence_interval_similarity(
        self,
        matches_data: List[Dict[str, Any]],
        similarity_function: Callable,
        bootstrap_scores: np.ndarray,
        original_score: float,
        alpha: float,
        **kwargs
    ) -> Tuple[float, float]:
        """Calcular intervalo de confianza BCA para similitud"""
        if not self.statistical_analysis:
            return self._percentile_confidence_interval(bootstrap_scores, alpha)
        
        try:
            # Usar el método BCA del análisis estadístico existente
            # Adaptar para datos de matching
            similarity_values = [similarity_function([match], **kwargs) for match in matches_data]
            
            # Usar bootstrap del análisis estadístico
            bootstrap_result = self.statistical_analysis.bootstrap_sampling(
                similarity_values,
                statistic_func=np.mean,
                n_bootstrap=len(bootstrap_scores),
                confidence_level=self.config.confidence_level,
                method='bca'
            )
            
            return bootstrap_result.bca_ci or bootstrap_result.confidence_interval
            
        except Exception as e:
            self.logger.warning(f"BCA calculation failed, using percentile: {e}")
            return self._percentile_confidence_interval(bootstrap_scores, alpha)
    
    def _calculate_match_statistics(self, matches_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calcular estadísticas de los matches"""
        if not matches_data:
            return {}
        
        distances = [match.get('distance', 0) for match in matches_data]
        
        return {
            'mean_distance': np.mean(distances),
            'std_distance': np.std(distances),
            'min_distance': np.min(distances),
            'max_distance': np.max(distances),
            'median_distance': np.median(distances),
            'n_matches': len(matches_data)
        }
    
    def _bootstrap_quality_weighted_similarity(
        self,
        matches_data: List[Dict[str, Any]],
        **kwargs
    ) -> Optional[Tuple[float, float]]:
        """Bootstrap para similitud ponderada por calidad"""
        try:
            def quality_weighted_similarity(matches):
                if not matches:
                    return 0.0
                
                # Calcular similitud ponderada por calidad inversa de distancia
                weights = [1.0 / (1.0 + match.get('distance', 1.0)) for match in matches]
                total_weight = sum(weights)
                
                if total_weight == 0:
                    return 0.0
                
                return sum(weights) / len(matches)  # Similitud normalizada
            
            # Bootstrap rápido para quality weighting
            n_bootstrap_quick = min(500, self.config.n_bootstrap)
            bootstrap_scores = []
            
            for _ in range(n_bootstrap_quick):
                indices = np.random.choice(len(matches_data), size=len(matches_data), replace=True)
                resampled_data = [matches_data[i] for i in indices]
                score = quality_weighted_similarity(resampled_data)
                bootstrap_scores.append(score)
            
            alpha = 1 - self.config.confidence_level
            return self._percentile_confidence_interval(np.array(bootstrap_scores), alpha)
            
        except Exception as e:
            self.logger.debug(f"Quality weighted bootstrap failed: {e}")
            return None
    
    def _bootstrap_geometric_consistency(
        self,
        matches_data: List[Dict[str, Any]],
        **kwargs
    ) -> Optional[Tuple[float, float]]:
        """Bootstrap para consistencia geométrica"""
        try:
            def geometric_consistency(matches):
                if len(matches) < 4:  # Mínimo para homografía
                    return 0.0
                
                # Simular consistencia geométrica basada en distribución de matches
                distances = [match.get('distance', 0) for match in matches]
                consistency = 1.0 - (np.std(distances) / (np.mean(distances) + 1e-6))
                return max(0.0, min(1.0, consistency))
            
            # Bootstrap rápido para consistencia geométrica
            n_bootstrap_quick = min(300, self.config.n_bootstrap)
            bootstrap_scores = []
            
            for _ in range(n_bootstrap_quick):
                indices = np.random.choice(len(matches_data), size=len(matches_data), replace=True)
                resampled_data = [matches_data[i] for i in indices]
                score = geometric_consistency(resampled_data)
                bootstrap_scores.append(score)
            
            alpha = 1 - self.config.confidence_level
            return self._percentile_confidence_interval(np.array(bootstrap_scores), alpha)
            
        except Exception as e:
            self.logger.debug(f"Geometric consistency bootstrap failed: {e}")
            return None
    
    def _create_fallback_result(
        self,
        matches_data: List[Dict[str, Any]],
        similarity_function: Callable,
        **kwargs
    ) -> SimilarityBootstrapResult:
        """Crear resultado fallback cuando bootstrap falla"""
        try:
            original_similarity = similarity_function(matches_data, **kwargs)
        except:
            original_similarity = 0.0
        
        # Intervalo de confianza conservador
        margin = 0.1 * original_similarity if original_similarity > 0 else 0.05
        confidence_interval = (
            max(0.0, original_similarity - margin),
            min(1.0, original_similarity + margin)
        )
        
        return SimilarityBootstrapResult(
            similarity_score=original_similarity,
            confidence_interval=confidence_interval,
            confidence_level=self.config.confidence_level,
            bootstrap_scores=np.array([original_similarity]),
            bias=0.0,
            standard_error=margin,
            percentile_ci=confidence_interval,
            bca_ci=None,
            n_bootstrap=1,
            method='fallback',
            processing_time=0.0,
            match_statistics=self._calculate_match_statistics(matches_data)
        )

def create_similarity_bootstrap_function(
    good_matches: int,
    kp1_count: int,
    kp2_count: int,
    algorithm: str,
    image1_quality: float = 0.0,
    image2_quality: float = 0.0
) -> Callable[[List[Dict[str, Any]]], float]:
    """
    Crear función de similitud compatible con bootstrap para UnifiedMatcher
    
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
    def similarity_function(matches_data: List[Dict[str, Any]], **kwargs) -> float:
        """Función de similitud para bootstrap sampling"""
        if not matches_data or kp1_count == 0 or kp2_count == 0:
            return 0.0
        
        # Calcular similitud basada en matches resampled
        n_good_matches = len(matches_data)
        max_possible_matches = min(kp1_count, kp2_count)
        
        if max_possible_matches == 0:
            return 0.0
        
        # Score base: proporción de matches buenos
        base_score = n_good_matches / max_possible_matches
        
        # Ajustar según algoritmo (similar a UnifiedMatcher)
        if algorithm == "ORB" and n_good_matches >= 10:
            base_score *= 1.2
        elif algorithm == "SIFT" and n_good_matches >= 20:
            base_score *= 1.1
        elif algorithm == "AKAZE" and n_good_matches >= 8:
            base_score *= 1.15
        
        # Normalizar a 0-100
        base_score = min(base_score * 100, 100.0)
        
        # Aplicar quality weighting si está disponible
        if image1_quality > 0 and image2_quality > 0:
            combined_quality = 2 * (image1_quality * image2_quality) / (image1_quality + image2_quality)
            quality_factor = 0.5 + 0.5 * combined_quality
            base_score *= quality_factor
        
        return base_score
    
    return similarity_function

# Función de utilidad para integración fácil
def calculate_bootstrap_confidence_interval(
    matches_data: List[Dict[str, Any]],
    similarity_score: float,
    algorithm: str = "ORB",
    confidence_level: float = 0.95,
    n_bootstrap: int = 1000
) -> Tuple[float, float]:
    """
    Función de utilidad para calcular intervalos de confianza bootstrap
    
    Args:
        matches_data: Datos de matches
        similarity_score: Score de similitud original
        algorithm: Algoritmo utilizado
        confidence_level: Nivel de confianza
        n_bootstrap: Número de muestras bootstrap
        
    Returns:
        Tupla con (lower_bound, upper_bound) del intervalo de confianza
    """
    try:
        config = MatchingBootstrapConfig(
            n_bootstrap=n_bootstrap,
            confidence_level=confidence_level,
            method='percentile'  # Método más rápido para utilidad
        )
        
        analyzer = BootstrapSimilarityAnalyzer(config)
        
        # Función de similitud simple
        def simple_similarity(matches):
            return len(matches) / max(1, len(matches_data)) * similarity_score
        
        result = analyzer.bootstrap_similarity_confidence(matches_data, simple_similarity)
        return result.confidence_interval
        
    except Exception as e:
        logging.getLogger(__name__).warning(f"Bootstrap CI calculation failed: {e}")
        # Fallback conservador
        margin = 0.1 * similarity_score if similarity_score > 0 else 0.05
        return (
            max(0.0, similarity_score - margin),
            min(100.0, similarity_score + margin)
        )

if __name__ == "__main__":
    # Test básico del módulo
    logging.basicConfig(level=logging.INFO)
    
    # Datos de prueba
    test_matches = [
        {'distance': 0.1, 'kp1_idx': 0, 'kp2_idx': 0},
        {'distance': 0.2, 'kp1_idx': 1, 'kp2_idx': 1},
        {'distance': 0.15, 'kp1_idx': 2, 'kp2_idx': 2},
        {'distance': 0.3, 'kp1_idx': 3, 'kp2_idx': 3},
        {'distance': 0.25, 'kp1_idx': 4, 'kp2_idx': 4},
    ]
    
    # Test función de utilidad
    ci = calculate_bootstrap_confidence_interval(test_matches, 75.0, "ORB")
    print(f"Intervalo de confianza: {ci}")
    
    # Test analizador completo
    config = MatchingBootstrapConfig(n_bootstrap=100)
    analyzer = BootstrapSimilarityAnalyzer(config)
    
    similarity_func = create_similarity_bootstrap_function(
        good_matches=len(test_matches),
        kp1_count=100,
        kp2_count=120,
        algorithm="ORB"
    )
    
    result = analyzer.bootstrap_similarity_confidence(test_matches, similarity_func)
    print(f"Resultado bootstrap completo:")
    print(f"  Score: {result.similarity_score:.2f}")
    print(f"  CI: {result.confidence_interval}")
    print(f"  Método: {result.method}")
    print(f"  Tiempo: {result.processing_time:.3f}s")