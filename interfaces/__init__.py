"""Interfaces públicas para SIGeC-Balisticar
===========================================

Este paquete agrupa las interfaces abstractas usadas por los módulos
principales (pipeline, matching, preprocesamiento, deep learning, base de datos).
Provee importaciones estables de alto nivel:

from interfaces import IFeatureMatcher, IPipelineProcessor, ProcessingResult
from interfaces import IDatabaseManager
from interfaces import IImagePreprocessor, ISegmentationProcessor
from interfaces import IDeepLearningModel, ISegmentationModel
"""

# Re-exportar interfaces de submódulos
from .matcher_interfaces import IFeatureMatcher, IDescriptorExtractor
from .pipeline_interfaces import IPipelineProcessor, ProcessingResult, IQualityAssessor, IQualityMetricsProvider
from .preprocessor_interfaces import IImagePreprocessor, ISegmentationProcessor
from .deep_learning_interfaces import IDeepLearningModel, ISegmentationModel
from .database_interfaces import IDatabaseManager

__all__ = [
    # Matcher
    'IFeatureMatcher',
    'IDescriptorExtractor',
    # Pipeline
    'IPipelineProcessor',
    'ProcessingResult',
    'IQualityAssessor',
    'IQualityMetricsProvider',
    # Preprocessing
    'IImagePreprocessor',
    'ISegmentationProcessor',
    # Deep Learning
    'IDeepLearningModel',
    'ISegmentationModel',
    # Database
    'IDatabaseManager',
]