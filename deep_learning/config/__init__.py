"""
Sistema de Configuración para Deep Learning SIGeC-Balistica
===================================================

Este módulo proporciona un sistema completo de configuración
para experimentos y modelos de deep learning balístico.

Componentes principales:
- Configuración de modelos y experimentos
- Gestión de hiperparámetros
- Configuración de entrenamiento y evaluación
- Perfiles de configuración predefinidos

Autor: SIGeC-BalisticaTeam
Versión: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "SIGeC-BalisticaTeam"

# Importar componentes principales
from .experiment_config import (
    ExperimentConfig,
    ModelConfig,
    TrainingConfig,
    DataConfig,
    EvaluationConfig,
    create_ballistic_classification_config,
    create_ballistic_matching_config
)

from .config_manager import (
    ConfigManager,
    load_config,
    save_config,
    create_default_config
)

from .hyperparameter_config import (
    HyperparameterConfig,
    OptimizationConfig,
    SchedulerConfig
)

# Alias para compatibilidad
BallisticClassificationConfig = ExperimentConfig
BallisticClassificationConfig.create_default = staticmethod(create_ballistic_classification_config)

__all__ = [
    # Configuraciones principales
    'ExperimentConfig',
    'ModelConfig', 
    'TrainingConfig',
    'DataConfig',
    'EvaluationConfig',
    
    # Gestión de configuración
    'ConfigManager',
    'load_config',
    'save_config',
    'create_default_config',
    
    # Hiperparámetros
    'HyperparameterConfig',
    'OptimizationConfig',
    'SchedulerConfig',
    
    # Alias de compatibilidad
    'BallisticClassificationConfig'
]