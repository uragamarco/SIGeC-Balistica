"""Utilidades del Sistema - SIGeC-Balistica
========================================

Módulo de utilidades que proporciona funciones auxiliares y herramientas
comunes para todo el sistema.

Componentes principales:
- Configuración (deprecado, usar config.unified_config)
- Logging y monitoreo
- Gestión de dependencias
- Validación de datos
- Manejo de errores
"""

# Importaciones principales
try:
    from .logger import setup_logger, get_logger, LoggerMixin
    LOGGER_AVAILABLE = True
except ImportError:
    LOGGER_AVAILABLE = False

try:
    from .dependency_manager import DependencyManager, DependencyInfo
    DEPENDENCY_MANAGER_AVAILABLE = True
except ImportError:
    DEPENDENCY_MANAGER_AVAILABLE = False

# Configuración (deprecado)
try:
    from .config import get_config, Config
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False

__all__ = []

if LOGGER_AVAILABLE:
    __all__.extend(['setup_logger', 'get_logger', 'LoggerMixin'])

if DEPENDENCY_MANAGER_AVAILABLE:
    __all__.extend(['DependencyManager', 'DependencyInfo'])

if CONFIG_AVAILABLE:
    __all__.extend(['get_config', 'Config'])

__version__ = "1.0.0"
__author__ = "SIGeC-Balisticar Development Team"