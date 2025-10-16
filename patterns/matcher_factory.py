"""
Factory Pattern para Algoritmos de Matching Balístico
"""

from typing import Dict, Any, Optional
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class MatcherType(Enum):
    """Tipos de matcher disponibles"""
    SIFT = "sift"
    ORB = "orb"
    AKAZE = "akaze"
    BRISK = "brisk"
    UNIFIED = "unified"

class MatcherFactory:
    """Factory para crear instancias de matchers"""
    
    _matchers = {}
    
    @classmethod
    def register_matcher(cls, matcher_type: MatcherType, matcher_class):
        """Registra un tipo de matcher"""
        cls._matchers[matcher_type] = matcher_class
        logger.info(f"Matcher registrado: {matcher_type.value}")
    
    @classmethod
    def create_matcher(cls, matcher_type: MatcherType, config: Dict[str, Any] = None):
        """Crea una instancia de matcher"""
        if matcher_type not in cls._matchers:
            raise ValueError(f"Matcher type {matcher_type.value} not registered")
        
        matcher_class = cls._matchers[matcher_type]
        
        try:
            if config:
                return matcher_class(config)
            else:
                return matcher_class()
        except Exception as e:
            logger.error(f"Error creating matcher {matcher_type.value}: {e}")
            raise
    
    @classmethod
    def get_available_matchers(cls) -> list:
        """Retorna lista de matchers disponibles"""
        return list(cls._matchers.keys())
    
    @classmethod
    def get_matcher_info(cls, matcher_type: MatcherType) -> Dict[str, Any]:
        """Retorna información sobre un matcher"""
        if matcher_type not in cls._matchers:
            return {}
        
        matcher_class = cls._matchers[matcher_type]
        return {
            'name': matcher_type.value,
            'class': matcher_class.__name__,
            'module': matcher_class.__module__,
            'description': getattr(matcher_class, '__doc__', 'No description available')
        }

# Auto-registro de matchers disponibles
def _auto_register_matchers():
    """Auto-registra matchers disponibles"""
    try:
        from matching.unified_matcher import UnifiedMatcher
        MatcherFactory.register_matcher(MatcherType.UNIFIED, UnifiedMatcher)
    except ImportError:
        logger.warning("UnifiedMatcher not available")
    
    # Registrar otros matchers según disponibilidad
    # TODO: Implementar registro automático de otros matchers

# Ejecutar auto-registro al importar
_auto_register_matchers()
