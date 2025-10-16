"""
Factory Pattern para Preprocesadores de Imágenes Balísticas
"""

from typing import Dict, Any, Optional
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class PreprocessorType(Enum):
    """Tipos de preprocessor disponibles"""
    UNIFIED = "unified"
    BASIC = "basic"
    ADVANCED = "advanced"
    NIST_COMPLIANT = "nist_compliant"

class PreprocessorFactory:
    """Factory para crear instancias de preprocessors"""
    
    _preprocessors = {}
    
    @classmethod
    def register_preprocessor(cls, preprocessor_type: PreprocessorType, preprocessor_class):
        """Registra un tipo de preprocessor"""
        cls._preprocessors[preprocessor_type] = preprocessor_class
        logger.info(f"Preprocessor registrado: {preprocessor_type.value}")
    
    @classmethod
    def create_preprocessor(cls, preprocessor_type: PreprocessorType, config: Dict[str, Any] = None):
        """Crea una instancia de preprocessor"""
        if preprocessor_type not in cls._preprocessors:
            raise ValueError(f"Preprocessor type {preprocessor_type.value} not registered")
        
        preprocessor_class = cls._preprocessors[preprocessor_type]
        
        try:
            if config:
                return preprocessor_class(config)
            else:
                return preprocessor_class()
        except Exception as e:
            logger.error(f"Error creating preprocessor {preprocessor_type.value}: {e}")
            raise
    
    @classmethod
    def get_available_preprocessors(cls) -> list:
        """Retorna lista de preprocessors disponibles"""
        return list(cls._preprocessors.keys())
    
    @classmethod
    def get_preprocessor_info(cls, preprocessor_type: PreprocessorType) -> Dict[str, Any]:
        """Retorna información sobre un preprocessor"""
        if preprocessor_type not in cls._preprocessors:
            return {}
        
        preprocessor_class = cls._preprocessors[preprocessor_type]
        return {
            'name': preprocessor_type.value,
            'class': preprocessor_class.__name__,
            'module': preprocessor_class.__module__,
            'description': getattr(preprocessor_class, '__doc__', 'No description available')
        }

# Auto-registro de preprocessors disponibles
def _auto_register_preprocessors():
    """Auto-registra preprocessors disponibles"""
    try:
        from image_processing.unified_preprocessor import UnifiedPreprocessor
        PreprocessorFactory.register_preprocessor(PreprocessorType.UNIFIED, UnifiedPreprocessor)
    except ImportError:
        logger.warning("UnifiedPreprocessor not available")

# Ejecutar auto-registro al importar
_auto_register_preprocessors()
