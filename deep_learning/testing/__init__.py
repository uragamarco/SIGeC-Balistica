"""
Testing Framework para Deep Learning SIGeC-Balistica
============================================

Este módulo proporciona un framework completo de testing y evaluación
para modelos de deep learning aplicados al análisis balístico.

Componentes principales:
- Métricas especializadas para análisis balístico
- Framework de evaluación con validación cruzada
- Análisis estadístico y visualización de resultados
- Testing automatizado de modelos y pipelines

Autor: SIGeC-BalisticaTeam
Versión: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "SIGeC-BalisticaTeam"

# Importar componentes principales
from .ballistic_metrics import (
    BallisticMetrics,
    CMCCurve,
    ROCAnalysis,
    MatchingMetrics,
    ClassificationMetrics
)

from .evaluation_framework import (
    ModelEvaluator,
    CrossValidator,
    PerformanceAnalyzer,
    ResultsVisualizer
)

from .test_runner import (
    TestRunner,
    ModelTester,
    PipelineTester,
    DatasetTester
)

__all__ = [
    # Métricas
    'BallisticMetrics',
    'CMCCurve', 
    'ROCAnalysis',
    'MatchingMetrics',
    'ClassificationMetrics',
    
    # Framework de evaluación
    'ModelEvaluator',
    'CrossValidator',
    'PerformanceAnalyzer',
    'ResultsVisualizer',
    
    # Testing
    'TestRunner',
    'ModelTester',
    'PipelineTester',
    'DatasetTester'
]