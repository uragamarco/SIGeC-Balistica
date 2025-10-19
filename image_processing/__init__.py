"""Módulo de Procesamiento de Imágenes - Sistema Balístico Forense SIGeC-Balisticar
===============================================================================

Módulo especializado en procesamiento y análisis de imágenes balísticas.
Implementa algoritmos avanzados de preprocesamiento, extracción de características
y análisis de calidad específicos para el dominio balístico.

Componentes principales:
- ChunkedImageProcessor: Procesamiento por chunks para imágenes grandes
- UnifiedPreprocessor: Preprocesador unificado con múltiples algoritmos
- BallisticFeatureExtractor: Extractor de características balísticas
- GPUAccelerator: Aceleración por GPU para operaciones intensivas
- UnifiedROIDetector: Detector de regiones de interés (ROI)
- OptimizedImageLoader: Cargador optimizado con caché y carga progresiva
"""

# Importaciones principales
try:
    from .chunked_processor import ChunkedImageProcessor, ChunkConfig, ChunkStrategy
    CHUNKED_PROCESSOR_AVAILABLE = True
except ImportError:
    CHUNKED_PROCESSOR_AVAILABLE = False

try:
    from .unified_preprocessor import UnifiedPreprocessor, PreprocessingConfig
    UNIFIED_PREPROCESSOR_AVAILABLE = True
except ImportError:
    UNIFIED_PREPROCESSOR_AVAILABLE = False

try:
    from .ballistic_features import BallisticFeatureExtractor, FeatureConfig
    BALLISTIC_FEATURES_AVAILABLE = True
except ImportError:
    BALLISTIC_FEATURES_AVAILABLE = False

try:
    from .gpu_accelerator import GPUAccelerator
    GPU_ACCELERATOR_AVAILABLE = True
except ImportError:
    GPU_ACCELERATOR_AVAILABLE = False

try:
    from .unified_roi_detector import UnifiedROIDetector, ROIDetectionConfig, DetectionLevel
    ROI_DETECTOR_AVAILABLE = True
except ImportError:
    ROI_DETECTOR_AVAILABLE = False

try:
    from .optimized_loader import (
        OptimizedImageLoader,
        LoadingConfig,
        LoadingStrategy,
        ImageFormat,
        QualityLevel,
        ImageMetadata,
        LoadingResult,
        CacheEntry,
        LRUCache,
        ImageFormatDetector,
        ProgressiveLoader,
        PersistentImageCache,
        create_optimized_config,
        estimate_memory_usage,
    )
    OPTIMIZED_LOADER_AVAILABLE = True
except ImportError:
    OPTIMIZED_LOADER_AVAILABLE = False

try:
    from .feature_extractor import FeatureExtractor, extract_features
    FEATURE_EXTRACTOR_AVAILABLE = True
except ImportError:
    FEATURE_EXTRACTOR_AVAILABLE = False

# Interfaces públicas principales
__all__ = []

if CHUNKED_PROCESSOR_AVAILABLE:
    __all__.extend([
        'ChunkedImageProcessor',
        'ChunkConfig',
        'ChunkStrategy'
    ])

if UNIFIED_PREPROCESSOR_AVAILABLE:
    __all__.extend([
        'UnifiedPreprocessor',
        'PreprocessingConfig'
    ])

if BALLISTIC_FEATURES_AVAILABLE:
    __all__.extend([
        'BallisticFeatureExtractor',
        'FeatureConfig'
    ])

if GPU_ACCELERATOR_AVAILABLE:
    __all__.extend(['GPUAccelerator'])

if ROI_DETECTOR_AVAILABLE:
    __all__.extend([
        'UnifiedROIDetector',
        'ROIDetectionConfig',
        'DetectionLevel'
    ])

if OPTIMIZED_LOADER_AVAILABLE:
    __all__.extend([
        'OptimizedImageLoader',
        'LoadingConfig',
        'LoadingStrategy',
        'ImageFormat',
        'QualityLevel',
        'ImageMetadata',
        'LoadingResult',
        'CacheEntry',
        'LRUCache',
        'ImageFormatDetector',
        'ProgressiveLoader',
        'PersistentImageCache',
        'create_optimized_config',
        'estimate_memory_usage',
    ])

if FEATURE_EXTRACTOR_AVAILABLE:
    __all__.extend([
        'FeatureExtractor',
        'extract_features'
    ])

__version__ = "1.0.0"
__author__ = "SIGeC-Balisticar Development Team"