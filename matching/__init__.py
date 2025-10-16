"""Módulo de Matching - Sistema Balístico Forense SIGeC-Balisticar
==============================================================

Módulo especializado en comparación y matching de características balísticas.
Implementa múltiples algoritmos de matching y análisis de similitud.

Componentes principales:
- UnifiedMatcher: Matcher principal con múltiples algoritmos
- CMCAlgorithm: Implementación del algoritmo CMC
- BallisticFeatureExtractor: Extractor de características balísticas
"""

# Importaciones principales
try:
    from .unified_matcher import UnifiedMatcher, MatchResult, MatchingConfig, AlgorithmType, MatchingLevel
    UNIFIED_MATCHER_AVAILABLE = True
except ImportError:
    UNIFIED_MATCHER_AVAILABLE = False

try:
    from .cmc_algorithm import CMCAlgorithm, CMCParameters, CMCMatchResult
    CMC_AVAILABLE = True
except ImportError:
    CMC_AVAILABLE = False

# Interfaces públicas principales
__all__ = []

if UNIFIED_MATCHER_AVAILABLE:
    __all__.extend([
        'UnifiedMatcher',
        'MatchResult', 
        'MatchingConfig',
        'AlgorithmType',
        'MatchingLevel'
    ])

if CMC_AVAILABLE:
    __all__.extend([
        'CMCAlgorithm',
        'CMCParameters',
        'CMCMatchResult'
    ])

__version__ = "1.0.0"
__author__ = "SIGeC-Balisticar Development Team"