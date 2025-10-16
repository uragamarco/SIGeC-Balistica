"""Gestor de Configuración Unificado
================================

Este módulo proporciona un gestor de configuración centralizado que unifica:
- Configuración general del proyecto
- Configuración de deep learning
- Configuración de producción
- Configuración de GUI
- Configuración de testing

Integra el sistema de configuración en capas para herencia y sobrescritura.
Mantiene compatibilidad con las APIs existentes mientras centraliza la gestión.
"""

import os
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import logging
from datetime import datetime
from dataclasses import dataclass, asdict
import copy

# Integración con sistema de configuración unificado
from .unified_config import (
    get_unified_config,
    reset_unified_config,
    Environment,
)

# Importar gestores específicos para compatibilidad
try:
    from deep_learning.config.config_manager import ConfigManager as DLConfigManager
except ImportError:
    DLConfigManager = None

try:
    from production.production_config import ProductionConfigManager
except ImportError:
    ProductionConfigManager = None

logger = logging.getLogger(__name__)

@dataclass
class ConfigInfo:
    """Información sobre una configuración"""
    name: str
    type: str
    path: Path
    last_modified: datetime
    size: int
    valid: bool = True
    errors: List[str] = None

class UnifiedConfigManager:
    """Gestor de configuración unificado con soporte de capas"""
    
    def __init__(self, config_dir: Optional[Path] = None):
        """
        Inicializar gestor unificado
        
        Args:
            config_dir: Directorio de configuraciones
        """
        self.config_dir = config_dir or Path(__file__).parent
        self.project_root = self._find_project_root()
        self.unified_config_path = self.config_dir / "unified_config.yaml"
        # Mantener referencia legacy si existiera, pero el sistema usa unified_config_* por entorno
        self.production_config_path = self.config_dir / "production.json"
        self.deep_learning_config_dir = self.project_root / "deep_learning" / "config"
        
        # Gestores específicos
        self.dl_manager = None
        self.production_manager = None
        
        # Cache de configuraciones
        self._config_cache: Dict[str, Any] = {}
        self._last_reload = None
        
        # Inicializar gestores específicos
        self._init_specific_managers()
        
        # Migrar configuraciones legacy si es necesario (lo gestiona UnifiedConfig en su init)
        self._migrate_legacy_configs()
        
        logger.info("UnifiedConfigManager inicializado usando UnifiedConfig por entorno")
    
    def _find_project_root(self) -> Path:
        """Encuentra la raíz del proyecto"""
        current = Path(__file__).parent
        while current.parent != current:
            if (current / "config" / "unified_config.yaml").exists():
                return current
            current = current.parent
        # Fallback al directorio padre de config
        return Path(__file__).parent.parent
    
    def _init_specific_managers(self):
        """Inicializa gestores específicos para compatibilidad"""
        try:
            if DLConfigManager:
                self.dl_manager = DLConfigManager()
        except Exception as e:
            logger.warning(f"No se pudo inicializar DL ConfigManager: {e}")
        
        try:
            if ProductionConfigManager:
                self.production_manager = ProductionConfigManager()
        except Exception as e:
            logger.warning(f"No se pudo inicializar ProductionConfigManager: {e}")
    
    def _migrate_legacy_configs(self):
        """Migra configuraciones legacy al nuevo sistema"""
        try:
            # Instanciar el sistema unificado para ejecutar su migración automática
            get_unified_config(force_reload=True)
            logger.info("Verificación de migración legacy completada por UnifiedConfig")
        except Exception as e:
            logger.warning(f"Error migrando configuraciones legacy: {e}")

    def _map_environment(self, environment: Optional[str]) -> Environment:
        """Mapea nombre de entorno a Enum Environment"""
        if not environment:
            return Environment.DEVELOPMENT
        env = (environment or "").strip().lower()
        if env in ("base", "dev", "development"):
            return Environment.DEVELOPMENT
        if env in ("test", "testing"):
            return Environment.TESTING
        if env in ("prod", "production"):
            return Environment.PRODUCTION
        # Fallback seguro
        return Environment.DEVELOPMENT
    
    def load_config(self, config_type: str = "unified", environment: Optional[str] = None) -> Dict[str, Any]:
        """
        Carga configuración del tipo especificado
        
        Args:
            config_type: Tipo de configuración (unified, deep_learning, production, gui, testing)
            environment: Entorno específico (base, testing, production)
            
        Returns:
            Diccionario con la configuración
        """
        cache_key = f"{config_type}_{environment or 'default'}"
        
        if cache_key in self._config_cache:
            return self._config_cache[cache_key]
        
        config = {}
        
        try:
            if config_type == "unified":
                env_enum = self._map_environment(environment)
                ucfg = get_unified_config(environment=env_enum, force_reload=True)
                config = ucfg.get_config_dict()
            elif config_type == "deep_learning":
                config = self._load_deep_learning_config()
            elif config_type == "production":
                ucfg = get_unified_config(environment=Environment.PRODUCTION, force_reload=True)
                config = ucfg.get_config_dict()
            elif config_type == "gui":
                env_enum = self._map_environment(environment)
                ucfg = get_unified_config(environment=env_enum, force_reload=True)
                config = asdict(ucfg.gui)
            elif config_type == "testing":
                ucfg = get_unified_config(environment=Environment.TESTING, force_reload=True)
                config = ucfg.get_config_dict()
            
            self._config_cache[cache_key] = config
            
        except Exception as e:
            logger.error(f"Error cargando configuración {config_type}: {e}")
            config = {}
        
        return config
    
    def _load_deep_learning_config(self) -> Dict[str, Any]:
        """Carga configuraciones de deep learning"""
        config = {}
        if self.deep_learning_config_dir.exists():
            # Cargar presets y experimentos
            for config_file in self.deep_learning_config_dir.rglob("*.json"):
                try:
                    with open(config_file, 'r', encoding='utf-8') as f:
                        file_config = json.load(f)
                        config[config_file.stem] = file_config
                except Exception as e:
                    logger.warning(f"Error cargando {config_file}: {e}")
        return config
    
    def get_config_value(self, key: str, default: Any = None, 
                        config_type: str = "unified", environment: Optional[str] = None) -> Any:
        """
        Obtiene un valor de configuración usando notación de punto
        
        Args:
            key: Clave de configuración (ej: "database.host")
            default: Valor por defecto
            config_type: Tipo de configuración
            environment: Entorno específico
            
        Returns:
            Valor de configuración o default
        """
        config = self.load_config(config_type, environment)
        
        # Navegar por claves anidadas
        keys = key.split('.')
        value = config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set_config_value(self, key: str, value: Any, 
                        config_type: str = "unified", environment: Optional[str] = None):
        """
        Establece un valor de configuración
        
        Args:
            key: Clave de configuración
            value: Valor a establecer
            config_type: Tipo de configuración
            environment: Entorno específico
        """
        cache_key = f"{config_type}_{environment or 'default'}"
        config = self.load_config(config_type, environment)
        
        # Navegar y crear estructura anidada
        keys = key.split('.')
        current = config
        
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        current[keys[-1]] = value
        self._config_cache[cache_key] = config
    
    def save_config(self, config_type: str = "unified", environment: Optional[str] = None):
        """
        Guarda configuración del tipo especificado
        
        Args:
            config_type: Tipo de configuración a guardar
            environment: Entorno específico
        """
        cache_key = f"{config_type}_{environment or 'default'}"
        
        if cache_key not in self._config_cache:
            logger.warning(f"No hay configuración en cache para {config_type}")
            return
        
        config = self._config_cache[cache_key]
        
        try:
            env_enum = self._map_environment(environment)
            # Forzar recarga para asegurar escritura en el archivo del entorno correcto
            ucfg = get_unified_config(environment=env_enum, force_reload=True)

            if config_type in ("unified", "production", "testing"):
                if isinstance(config, dict):
                    for section in ['database', 'image_processing', 'matching', 'gui', 'logging', 'deep_learning', 'nist']:
                        section_data = config.get(section)
                        if isinstance(section_data, dict):
                            ucfg.update_config(section, **section_data)
                ucfg.save_config()
            elif config_type == "gui":
                if isinstance(config, dict):
                    ucfg.update_config('gui', **config)
                ucfg.save_config()

            logger.info(f"Configuración {config_type} guardada")
            
        except Exception as e:
            logger.error(f"Error guardando configuración {config_type}: {e}")
    
    def reload_config(self, config_type: Optional[str] = None):
        """
        Recarga configuración desde disco
        
        Args:
            config_type: Tipo específico a recargar, None para todos
        """
        if config_type:
            # Limpiar cache para el tipo específico
            keys_to_remove = [k for k in self._config_cache.keys() if k.startswith(config_type)]
            for key in keys_to_remove:
                del self._config_cache[key]
        else:
            self._config_cache.clear()
        
        # Reiniciar instancia global de UnifiedConfig para recargar desde disco
        reset_unified_config()
        
        logger.info(f"Configuración recargada: {config_type or 'todas'}")
    
    def list_configs(self, config_type: str) -> List[str]:
        """
        Lista configuraciones disponibles del tipo especificado
        
        Args:
            config_type: Tipo de configuración
            
        Returns:
            Lista de nombres de configuración
        """
        configs = []
        
        if config_type == "deep_learning":
            if self.deep_learning_config_dir.exists():
                configs = [f.stem for f in self.deep_learning_config_dir.rglob("*.json")]
        elif config_type in ["unified", "gui", "testing", "production"]:
            configs = ["base", "testing", "production"]
        
        return sorted(configs)
    
    def validate_config(self, config_type: str, environment: Optional[str] = None) -> List[str]:
        """
        Valida configuración del tipo especificado
        
        Args:
            config_type: Tipo de configuración a validar
            environment: Entorno específico
            
        Returns:
            Lista de errores encontrados
        """
        errors = []
        config = self.load_config(config_type, environment)
        
        if not config:
            errors.append(f"Configuración {config_type} vacía o no encontrada")
            return errors
        
        # Validaciones específicas por tipo
        if config_type == "unified":
            errors.extend(self._validate_unified_config(config))
        elif config_type == "production":
            errors.extend(self._validate_production_config(config))
        
        return errors
    
    def _validate_unified_config(self, config: Dict[str, Any]) -> List[str]:
        """Valida configuración unificada"""
        errors = []
        required_sections = ['database', 'image_processing', 'matching']
        
        for section in required_sections:
            if section not in config:
                errors.append(f"Sección requerida '{section}' no encontrada")
        
        return errors
    
    def _validate_production_config(self, config: Dict[str, Any]) -> List[str]:
        """Valida configuración de producción"""
        errors = []
        required_fields = ['environment', 'database', 'security']
        
        for field in required_fields:
            if field not in config:
                errors.append(f"Campo requerido '{field}' no encontrado")
        
        return errors

# Instancia global del gestor unificado
_unified_manager = None

def get_unified_manager() -> UnifiedConfigManager:
    """Obtiene la instancia global del gestor unificado"""
    global _unified_manager
    if _unified_manager is None:
        _unified_manager = UnifiedConfigManager()
    return _unified_manager

# Funciones de compatibilidad con la API anterior
def get_project_root() -> str:
    """Obtiene la ruta raíz del proyecto de forma portable"""
    return str(get_unified_manager().project_root)

def load_config() -> dict:
    """Carga la configuración unificada"""
    return get_unified_manager().load_config("unified")

def get_config_value(key: str, default: Any = None) -> Any:
    """Obtiene un valor de configuración usando notación de punto"""
    return get_unified_manager().get_config_value(key, default, "unified")

def reload_config():
    """Recarga la configuración desde disco"""
    get_unified_manager().reload_config()
