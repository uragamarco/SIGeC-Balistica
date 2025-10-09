"""Common Statistical Core Module for SIGeC-Balistica
==========================================

Este módulo proporciona funcionalidades estadísticas unificadas y centralizadas
para todo el sistema SIGeC-Balistica, manteniendo compatibilidad con interfaces existentes
y preservando trazabilidad NIST.

Componentes principales:
- UnifiedStatisticalAnalysis: Núcleo estadístico unificado
- Adaptadores de compatibilidad para módulos existentes
- Clases de datos para resultados estadísticos
- Utilidades para bootstrap y análisis estadístico

Autor: SIGeC-Balistica Development Team
Fecha: 2024
Licencia: MIT
"""

# Importar núcleo estadístico unificado
from .statistical_core import (
    # Clase principal
    UnifiedStatisticalAnalysis,
    StatisticalCore,
    
    # Clases de datos
    BootstrapResult,
    StatisticalTestResult,
    MultipleComparisonResult,
    SimilarityBootstrapResult,
    MatchingBootstrapConfig,
    
    # Enums
    StatisticalTest,
    CorrectionMethod,
    
    # Funciones de utilidad
    create_bootstrap_adapter,
    create_statistical_adapter,
    create_similarity_bootstrap_function,
    calculate_bootstrap_confidence_interval
)

# Alias para compatibilidad hacia atrás
StatisticalAnalyzer = UnifiedStatisticalAnalysis

# Importar adaptadores de compatibilidad
from .compatibility_adapters import (
    # Adaptadores principales
    AdvancedStatisticalAnalysisAdapter,
    BootstrapSimilarityAnalyzerAdapter,
    StatisticalAnalyzerAdapter,
    
    # Funciones de utilidad de compatibilidad
    get_adapter,
    AVAILABLE_ADAPTERS
)

__version__ = "1.0.0"
__author__ = "SIGeC-Balistica Development Team"
__license__ = "MIT"

# Interfaces públicas principales
__all__ = [
    # Núcleo estadístico
    'UnifiedStatisticalAnalysis',
    'StatisticalCore',
    'StatisticalAnalyzer',  # Alias para compatibilidad
    
    # Clases de datos
    'BootstrapResult',
    'StatisticalTestResult', 
    'MultipleComparisonResult',
    'SimilarityBootstrapResult',
    'MatchingBootstrapConfig',
    
    # Enums
    'StatisticalTest',
    'CorrectionMethod',
    
    # Utilidades del núcleo
    'create_bootstrap_adapter',
    'create_statistical_adapter',
    'create_similarity_bootstrap_function',
    'calculate_bootstrap_confidence_interval',
    
    # Adaptadores de compatibilidad
    'AdvancedStatisticalAnalysisAdapter',
    'BootstrapSimilarityAnalyzerAdapter', 
    'StatisticalAnalyzerAdapter',
    'get_adapter',
    'AVAILABLE_ADAPTERS'
]