"""
Factory Pattern para Modelos de Deep Learning Balístico
"""

from typing import Dict, Any, Optional
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class ModelType(Enum):
    """Tipos de modelo disponibles"""
    BALLISTIC_CNN = "ballistic_cnn"
    SEGMENTATION = "segmentation"
    FEATURE_EXTRACTOR = "feature_extractor"
    SIMILARITY_NET = "similarity_net"

class DeepLearningFactory:
    """Factory para crear instancias de modelos DL"""
    
    _models = {}
    
    @classmethod
    def register_model(cls, model_type: ModelType, model_class):
        """Registra un tipo de modelo"""
        cls._models[model_type] = model_class
        logger.info(f"Modelo DL registrado: {model_type.value}")
    
    @classmethod
    def create_model(cls, model_type: ModelType, config: Dict[str, Any] = None):
        """Crea una instancia de modelo"""
        if model_type not in cls._models:
            raise ValueError(f"Model type {model_type.value} not registered")
        
        model_class = cls._models[model_type]
        
        try:
            if config:
                return model_class(config)
            else:
                return model_class()
        except Exception as e:
            logger.error(f"Error creating model {model_type.value}: {e}")
            raise
    
    @classmethod
    def get_available_models(cls) -> list:
        """Retorna lista de modelos disponibles"""
        return list(cls._models.keys())
    
    @classmethod
    def is_deep_learning_available(cls) -> bool:
        """Verifica si deep learning está disponible"""
        try:
            import torch
            return len(cls._models) > 0
        except ImportError:
            return False
    
    @classmethod
    def get_model_info(cls, model_type: ModelType) -> Dict[str, Any]:
        """Retorna información sobre un modelo"""
        if model_type not in cls._models:
            return {}
        
        model_class = cls._models[model_type]
        return {
            'name': model_type.value,
            'class': model_class.__name__,
            'module': model_class.__module__,
            'description': getattr(model_class, '__doc__', 'No description available')
        }

# Auto-registro de modelos disponibles
def _auto_register_models():
    """Auto-registra modelos disponibles"""
    try:
        from deep_learning.ballistic_dl_models import BallisticDLModels
        DeepLearningFactory.register_model(ModelType.BALLISTIC_CNN, BallisticDLModels)
    except ImportError:
        logger.warning("BallisticDLModels not available")

# Ejecutar auto-registro al importar
_auto_register_models()
