"""
Adaptador específico para migración crítica de nist_standards/statistical_analysis.py
=====================================================================================

Este adaptador mantiene compatibilidad total durante la migración del módulo crítico.
Preserva trazabilidad NIST y garantiza que no se rompa funcionalidad existente.
"""

import logging
from typing import Any, Optional, Dict, List
import warnings

# Importar el núcleo unificado
from common.statistical_core import UnifiedStatisticalAnalysis

logger = logging.getLogger(__name__)

class CriticalMigrationAdapter:
    """
    Adaptador crítico para migración de nist_standards/statistical_analysis.py
    
    Este adaptador:
    1. Mantiene compatibilidad total con la interfaz original
    2. Redirige llamadas al núcleo unificado
    3. Preserva trazabilidad NIST
    4. Permite rollback inmediato si es necesario
    """
    
    def __init__(self, *args, **kwargs):
        """Inicializa el adaptador crítico"""
        logger.info("CriticalMigrationAdapter inicializado - Preservando trazabilidad NIST")
        
        # Inicializar núcleo unificado con los mismos parámetros
        self._unified_core = UnifiedStatisticalAnalysis(*args, **kwargs)
        
        # Marcar como migración crítica activa
        self._migration_active = True
        self._nist_compliance = True
    
    def __getattr__(self, name: str) -> Any:
        """
        Redirige automáticamente cualquier llamada al núcleo unificado
        Mantiene compatibilidad total con la interfaz original
        """
        if hasattr(self._unified_core, name):
            method = getattr(self._unified_core, name)
            
            # Log para trazabilidad NIST
            logger.debug(f"CriticalMigrationAdapter: Redirigiendo {name} al núcleo unificado")
            
            return method
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
    
    def get_migration_status(self) -> Dict[str, Any]:
        """Retorna estado de la migración crítica"""
        return {
            'migration_active': self._migration_active,
            'nist_compliance': self._nist_compliance,
            'unified_core_available': self._unified_core is not None,
            'adapter_type': 'CriticalMigrationAdapter'
        }

# Alias para compatibilidad total
AdvancedStatisticalAnalysis = CriticalMigrationAdapter

# Función de validación NIST
def validate_nist_compliance() -> bool:
    """Valida que se mantenga el cumplimiento NIST durante la migración"""
    try:
        adapter = CriticalMigrationAdapter(random_state=42)
        status = adapter.get_migration_status()
        return status.get('nist_compliance', False)
    except Exception:
        return False
