#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utils Integration Module - SIGeC-Balistica GUI
===========================================

Módulo de integración específico para utilidades del sistema,
proporcionando acceso directo a funcionalidades de utils desde la GUI.

Funcionalidades:
- Validación de archivos e imágenes
- Logging unificado
- Cache de memoria
- Configuración del sistema
- Manejo de dependencias

Autor: SIGeC-Balistica Team
Fecha: Octubre 2025
"""

import os
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

# Importaciones de utilidades
try:
    from utils.logger import get_logger
    from utils.validators import SystemValidator
    from core.intelligent_cache import MemoryCache
    from utils.config import load_config, save_config
    from utils.dependency_manager import DependencyManager
    UTILS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Módulo utils no disponible: {e}")
    UTILS_AVAILABLE = False

# Instanciar el validador una vez
_system_validator = SystemValidator()

class UtilsIntegration:
    """
    Clase de integración para utilidades del sistema
    """
    
    def __init__(self):
        self.logger = None
        self.memory_cache = None
        self.dependency_manager = None
        self.config_cache = {}
        
        if UTILS_AVAILABLE:
            self._initialize_utils()
    
    def _initialize_utils(self):
        """Inicializa los componentes de utilidades"""
        try:
            # Logger unificado
            self.logger = get_logger(__name__)
            
            # Cache de memoria
            self.memory_cache = MemoryCache()
            
            # Gestor de dependencias
            self.dependency_manager = DependencyManager()
            
            self.logger.info("Utilidades del sistema inicializadas correctamente")
            
        except Exception as e:
            logging.error(f"Error inicializando utilidades: {e}")
    
    def validate_image_file(self, image_path: str) -> tuple[bool, str]:
        """
        Valida un archivo de imagen
        
        Args:
            image_path: Ruta al archivo de imagen
            
        Returns:
            Tupla (es_válido, mensaje)
        """
        if not UTILS_AVAILABLE:
            return self._fallback_image_validation(image_path)
        
        try:
            return _system_validator.validate_image_path(image_path)
        except Exception as e:
            return False, f"Error validando imagen: {e}"
    
    def _fallback_image_validation(self, image_path: str) -> tuple[bool, str]:
        """Validación básica de imagen sin utils"""
        if not os.path.exists(image_path):
            return False, "El archivo no existe"
        
        valid_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp']
        file_ext = Path(image_path).suffix.lower()
        
        if file_ext not in valid_extensions:
            return False, f"Formato no soportado: {file_ext}"
        
        return True, "Archivo válido"
    
    def get_cached_data(self, key: str) -> Any:
        """
        Obtiene datos del cache de memoria
        
        Args:
            key: Clave del dato en cache
            
        Returns:
            Datos cacheados o None si no existen
        """
        if not UTILS_AVAILABLE or not self.memory_cache:
            return self.config_cache.get(key)
        
        return self.memory_cache.get(key)
    
    def set_cached_data(self, key: str, data: Any, ttl: int = 3600):
        """
        Almacena datos en cache de memoria
        
        Args:
            key: Clave para el dato
            data: Datos a cachear
            ttl: Tiempo de vida en segundos
        """
        if not UTILS_AVAILABLE or not self.memory_cache:
            self.config_cache[key] = data
            return
        
        self.memory_cache.set(key, data, ttl)
    
    def clear_cache(self):
        """Limpia el cache de memoria"""
        if not UTILS_AVAILABLE or not self.memory_cache:
            self.config_cache.clear()
            return
        
        self.memory_cache.clear()
    
    def load_system_config(self, config_path: str) -> Optional[Dict[str, Any]]:
        """
        Carga configuración del sistema
        
        Args:
            config_path: Ruta al archivo de configuración
            
        Returns:
            Diccionario con la configuración o None si hay error
        """
        if not UTILS_AVAILABLE:
            return self._fallback_load_config(config_path)
        
        try:
            return load_config(config_path)
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error cargando configuración: {e}")
            return None
    
    def _fallback_load_config(self, config_path: str) -> Optional[Dict[str, Any]]:
        """Carga básica de configuración sin utils"""
        try:
            import json
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return None
    
    def save_system_config(self, config_path: str, config_data: Dict[str, Any]) -> bool:
        """
        Guarda configuración del sistema
        
        Args:
            config_path: Ruta donde guardar la configuración
            config_data: Datos de configuración
            
        Returns:
            True si se guardó correctamente, False en caso contrario
        """
        if not UTILS_AVAILABLE:
            return self._fallback_save_config(config_path, config_data)
        
        try:
            return save_config(config_path, config_data)
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error guardando configuración: {e}")
            return False
    
    def _fallback_save_config(self, config_path: str, config_data: Dict[str, Any]) -> bool:
        """Guardado básico de configuración sin utils"""
        try:
            import json
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            return True
        except Exception:
            return False
    
    def check_dependencies(self) -> Dict[str, bool]:
        """
        Verifica el estado de las dependencias del sistema
        
        Returns:
            Diccionario con el estado de cada dependencia
        """
        if not UTILS_AVAILABLE or not self.dependency_manager:
            return self._fallback_dependency_check()
        
        return self.dependency_manager.check_all_dependencies()
    
    def _fallback_dependency_check(self) -> Dict[str, bool]:
        """Verificación básica de dependencias sin utils"""
        dependencies = {}
        
        # Verificar dependencias básicas
        try:
            import numpy
            dependencies['numpy'] = True
        except ImportError:
            dependencies['numpy'] = False
        
        try:
            import cv2
            dependencies['opencv'] = True
        except ImportError:
            dependencies['opencv'] = False
        
        try:
            import matplotlib
            dependencies['matplotlib'] = True
        except ImportError:
            dependencies['matplotlib'] = False
        
        return dependencies
    
    def get_system_info(self) -> Dict[str, Any]:
        """
        Obtiene información del sistema
        
        Returns:
            Diccionario con información del sistema
        """
        import platform
        import sys
        
        return {
            'platform': platform.system(),
            'platform_version': platform.version(),
            'python_version': sys.version,
            'utils_available': UTILS_AVAILABLE,
            'cache_size': len(self.config_cache) if not UTILS_AVAILABLE else self.memory_cache.size() if self.memory_cache else 0,
            'dependencies': self.check_dependencies()
        }

# Instancia global
_utils_integration = None

def get_utils_integration() -> UtilsIntegration:
    """Obtiene la instancia global de integración de utilidades"""
    global _utils_integration
    if _utils_integration is None:
        _utils_integration = UtilsIntegration()
    return _utils_integration

if __name__ == "__main__":
    # Prueba básica del módulo
    utils_int = UtilsIntegration()
    print("=== Utils Integration Test ===")
    print("Sistema info:", utils_int.get_system_info())
    print("Dependencias:", utils_int.check_dependencies())