"""
Core module for SIGeC-Balisticar ballistic analysis system.

Exports stable core interfaces for pipeline orchestration, configuration,
performance monitoring, notifications, and intelligent cache.
"""

# Unified pipeline and configuration helpers
try:
    from .unified_pipeline import ScientificPipeline, create_pipeline_config, PipelineResult
    PIPELINE_AVAILABLE = True
except Exception:
    PIPELINE_AVAILABLE = False

try:
    from .pipeline_config import (
        PipelineConfiguration,
        PipelineLevel,
        DeepLearningConfig,
        QualityAssessmentConfig,
        PreprocessingConfig,
        ROIDetectionConfig,
        MatchingConfig,
        CMCAnalysisConfig,
        AFTEConclusionConfig,
        get_available_levels,
        get_level_description,
        get_recommended_level,
        get_predefined_config,
        list_predefined_configs,
    )
    PIPELINE_CONFIG_AVAILABLE = True
except Exception:
    PIPELINE_CONFIG_AVAILABLE = False

# Performance monitoring utilities
try:
    from .performance_monitor import (
        OperationType,
        PerformanceThreshold,
        PerformanceMetrics,
        PerformanceMonitor,
        get_performance_monitor,
        monitor_performance,
        monitor_image_analysis,
        monitor_image_processing,
        monitor_database_operation,
        monitor_image_comparison,
        monitor_nist_validation,
        monitor_pipeline_execution,
        PerformanceContext,
    )
    PERFORMANCE_AVAILABLE = True
except Exception:
    PERFORMANCE_AVAILABLE = False

# Notification system
try:
    from .notification_system import (
        NotificationType,
        NotificationChannel,
        Notification,
        NotificationManager,
        get_notification_manager,
        notify_info,
        notify_warning,
        notify_error,
        notify_critical,
        notify_success,
        NotificationSystem,
    )
    NOTIFICATIONS_AVAILABLE = True
except Exception:
    NOTIFICATIONS_AVAILABLE = False

# Intelligent cache interfaces
try:
    from .intelligent_cache import (
        CacheStrategy,
        CompressionType,
        CacheLevel,
        CacheEntry,
        CacheStats,
        IntelligentCache,
        get_cache,
        initialize_cache,
        cached,
        cache_invalidate,
        MemoryCache,
        get_global_cache,
        cache_features,
        cache_matches,
        clear_cache,
        get_cache_stats,
        configure_cache,
    )
    CACHE_AVAILABLE = True
except Exception:
    CACHE_AVAILABLE = False

try:
    from .error_handler import (
        ErrorSeverity,
        ErrorRecoveryManager,
        ErrorContext,
        RecoveryStrategy,
        get_error_manager,
        handle_error,
        with_error_handling,
    )
    ERROR_HANDLER_AVAILABLE = True
except Exception:
    ERROR_HANDLER_AVAILABLE = False

try:
    from .fallback_registry import (
        FallbackRegistry,
        get_fallback,
        create_robust_import,
        fallback_registry,
        DeepLearningFallback,
        WebServiceFallback,
        ImageProcessingFallback,
        DatabaseFallback,
        TorchFallback,
        TensorFlowFallback,
        FlaskFallback,
        RawPyFallback,
        CoreComponentsFallback,
    )
    FALLBACK_REGISTRY_AVAILABLE = True
except Exception:
    FALLBACK_REGISTRY_AVAILABLE = False

try:
    from .fallback_system import FallbackSystem, FallbackStrategy
    FALLBACK_SYSTEM_AVAILABLE = True
except Exception:
    FALLBACK_SYSTEM_AVAILABLE = False

__all__ = []

if PIPELINE_AVAILABLE:
    __all__.extend([
        'ScientificPipeline',
        'create_pipeline_config',
        'PipelineResult',
    ])

if PIPELINE_CONFIG_AVAILABLE:
    __all__.extend([
        'PipelineConfiguration',
        'PipelineLevel',
        'DeepLearningConfig',
        'QualityAssessmentConfig',
        'PreprocessingConfig',
        'ROIDetectionConfig',
        'MatchingConfig',
        'CMCAnalysisConfig',
        'AFTEConclusionConfig',
        'get_available_levels',
        'get_level_description',
        'get_recommended_level',
        'get_predefined_config',
        'list_predefined_configs',
    ])

if PERFORMANCE_AVAILABLE:
    __all__.extend([
        'OperationType',
        'PerformanceThreshold',
        'PerformanceMetrics',
        'PerformanceMonitor',
        'get_performance_monitor',
        'monitor_performance',
        'monitor_image_analysis',
        'monitor_image_processing',
        'monitor_database_operation',
        'monitor_image_comparison',
        'monitor_nist_validation',
        'monitor_pipeline_execution',
        'PerformanceContext',
    ])

if NOTIFICATIONS_AVAILABLE:
    __all__.extend([
        'NotificationType',
        'NotificationChannel',
        'Notification',
        'NotificationManager',
        'get_notification_manager',
        'notify_info',
        'notify_warning',
        'notify_error',
        'notify_critical',
        'notify_success',
        'NotificationSystem',
    ])

if CACHE_AVAILABLE:
    __all__.extend([
        'CacheStrategy',
        'CompressionType',
        'CacheLevel',
        'CacheEntry',
        'CacheStats',
        'IntelligentCache',
        'get_cache',
        'initialize_cache',
        'cached',
        'cache_invalidate',
        'MemoryCache',
        'get_global_cache',
        'cache_features',
        'cache_matches',
        'clear_cache',
        'get_cache_stats',
        'configure_cache',
    ])

if ERROR_HANDLER_AVAILABLE:
    __all__.extend([
        'ErrorSeverity',
        'ErrorRecoveryManager',
        'ErrorContext',
        'RecoveryStrategy',
        'get_error_manager',
        'handle_error',
        'with_error_handling',
    ])

if FALLBACK_REGISTRY_AVAILABLE:
    __all__.extend([
        'FallbackRegistry',
        'get_fallback',
        'create_robust_import',
        'fallback_registry',
        'DeepLearningFallback',
        'WebServiceFallback',
        'ImageProcessingFallback',
        'DatabaseFallback',
        'TorchFallback',
        'TensorFlowFallback',
        'FlaskFallback',
        'RawPyFallback',
        'CoreComponentsFallback',
    ])

if FALLBACK_SYSTEM_AVAILABLE:
    __all__.extend(['FallbackSystem', 'FallbackStrategy'])

__version__ = "1.0.0"
__author__ = "SIGeC-Balisticar Development Team"