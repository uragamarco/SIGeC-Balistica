#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wrapper de Compatibilidad - utils/config.py
==========================================

Este archivo mantiene compatibilidad con código legacy que importa
configuraciones desde utils.config. Redirige todas las llamadas
al nuevo sistema de configuración unificado.

DEPRECADO: Use config.unified_config directamente en código nuevo.

Autor: SEACABAr Team
"""

import warnings
from config.unified_config import (
    get_unified_config,
    get_database_config,
    get_image_processing_config,
    get_matching_config,
    get_gui_config,
    get_logging_config,
    get_deep_learning_config,
    get_nist_config,
    UnifiedConfig,
    DatabaseConfig,
    ImageProcessingConfig,
    MatchingConfig,
    GUIConfig,
    LoggingConfig,
    DeepLearningConfig,
    NISTConfig
)

# Emitir advertencia de deprecación
warnings.warn(
    "utils.config está deprecado. Use config.unified_config directamente.",
    DeprecationWarning,
    stacklevel=2
)

# Alias para compatibilidad
Config = get_unified_config

# Funciones de compatibilidad
def get_config():
    """Función de compatibilidad"""
    return get_unified_config()

# Exportar clases para compatibilidad
__all__ = [
    'Config',
    'get_config',
    'UnifiedConfig',
    'DatabaseConfig',
    'ImageProcessingConfig',
    'MatchingConfig',
    'GUIConfig',
    'LoggingConfig',
    'DeepLearningConfig',
    'NISTConfig',
    'get_unified_config',
    'get_database_config',
    'get_image_processing_config',
    'get_matching_config',
    'get_gui_config',
    'get_logging_config',
    'get_deep_learning_config',
    'get_nist_config'
]
