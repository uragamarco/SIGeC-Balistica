"""
Gestor de Configuración Centralizada
===================================

Proporciona funciones para acceder a la configuración unificada
y obtener rutas del proyecto de forma portable.
"""

import os
import yaml
from pathlib import Path
from typing import Any, Optional

_config_cache = None
_project_root = None

def get_project_root() -> str:
    """Obtiene la ruta raíz del proyecto de forma portable"""
    global _project_root
    
    if _project_root is None:
        # Buscar desde el directorio actual hacia arriba
        current = Path(__file__).parent
        while current.parent != current:
            if (current / "config" / "unified_config.yaml").exists():
                _project_root = str(current)
                break
            current = current.parent
        else:
            # Fallback al directorio padre de config
            _project_root = str(Path(__file__).parent.parent)
    
    return _project_root

def load_config() -> dict:
    """Carga la configuración unificada"""
    global _config_cache
    
    if _config_cache is None:
        config_path = Path(get_project_root()) / "config" / "unified_config.yaml"
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                _config_cache = yaml.safe_load(f) or {}
        except FileNotFoundError:
            print(f"⚠️  Configuración no encontrada: {config_path}")
            _config_cache = {}
    
    return _config_cache

def get_config_value(key: str, default: Any = None) -> Any:
    """Obtiene un valor de configuración usando notación de punto"""
    config = load_config()
    
    # Navegar por claves anidadas (ej: "api.host")
    keys = key.split('.')
    value = config
    
    for k in keys:
        if isinstance(value, dict) and k in value:
            value = value[k]
        else:
            return default
    
    return value

def reload_config():
    """Recarga la configuración desde disco"""
    global _config_cache
    _config_cache = None
