"""
Gestor de Configuraciones para Deep Learning SIGeC-Balistica
===================================================

Este módulo proporciona funciones para gestionar, cargar y guardar
configuraciones de experimentos de deep learning.

Autor: SIGeC-BalisticaTeam
"""

import os
import json
import yaml
from typing import Dict, Any, Optional, List
from pathlib import Path
import logging
from datetime import datetime

from .experiment_config import ExperimentConfig, create_ballistic_classification_config, create_ballistic_matching_config

logger = logging.getLogger(__name__)

class ConfigManager:
    """Gestor centralizado de configuraciones."""
    
    def __init__(self, config_dir: str = "deep_learning/configs"):
        """
        Inicializar el gestor de configuraciones.
        
        Args:
            config_dir: Directorio base para almacenar configuraciones
        """
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Subdirectorios para diferentes tipos
        self.templates_dir = self.config_dir / "templates"
        self.experiments_dir = self.config_dir / "experiments"
        self.presets_dir = self.config_dir / "presets"
        
        # Crear subdirectorios
        for subdir in [self.templates_dir, self.experiments_dir, self.presets_dir]:
            subdir.mkdir(exist_ok=True)
        
        logger.info(f"ConfigManager inicializado en: {self.config_dir}")
    
    def save_config(self, config: ExperimentConfig, 
                   filename: Optional[str] = None,
                   config_type: str = "experiment") -> str:
        """
        Guardar configuración en archivo.
        
        Args:
            config: Configuración a guardar
            filename: Nombre del archivo (opcional)
            config_type: Tipo de configuración (experiment, template, preset)
            
        Returns:
            Ruta del archivo guardado
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{config.experiment_name}_{timestamp}.json"
        
        # Seleccionar directorio según tipo
        if config_type == "template":
            target_dir = self.templates_dir
        elif config_type == "preset":
            target_dir = self.presets_dir
        else:
            target_dir = self.experiments_dir
        
        filepath = target_dir / filename
        
        # Guardar configuración
        config.save(str(filepath))
        
        logger.info(f"Configuración guardada: {filepath}")
        return str(filepath)
    
    def load_config(self, filename: str, 
                   config_type: str = "experiment") -> ExperimentConfig:
        """
        Cargar configuración desde archivo.
        
        Args:
            filename: Nombre del archivo
            config_type: Tipo de configuración
            
        Returns:
            Configuración cargada
        """
        # Seleccionar directorio según tipo
        if config_type == "template":
            target_dir = self.templates_dir
        elif config_type == "preset":
            target_dir = self.presets_dir
        else:
            target_dir = self.experiments_dir
        
        filepath = target_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Archivo de configuración no encontrado: {filepath}")
        
        config = ExperimentConfig.load(str(filepath))
        logger.info(f"Configuración cargada: {filepath}")
        
        return config
    
    def list_configs(self, config_type: str = "experiment") -> List[str]:
        """
        Listar configuraciones disponibles.
        
        Args:
            config_type: Tipo de configuración
            
        Returns:
            Lista de nombres de archivos
        """
        if config_type == "template":
            target_dir = self.templates_dir
        elif config_type == "preset":
            target_dir = self.presets_dir
        else:
            target_dir = self.experiments_dir
        
        configs = []
        for filepath in target_dir.glob("*.json"):
            configs.append(filepath.name)
        
        return sorted(configs)
    
    def create_preset_configs(self):
        """Crear configuraciones predefinidas."""
        presets = {
            "ballistic_classification.json": create_ballistic_classification_config(),
            "ballistic_matching.json": create_ballistic_matching_config(),
        }
        
        for filename, config in presets.items():
            filepath = self.presets_dir / filename
            config.save(str(filepath))
            logger.info(f"Preset creado: {filepath}")
    
    def get_config_info(self, filename: str, 
                       config_type: str = "experiment") -> Dict[str, Any]:
        """
        Obtener información básica de una configuración.
        
        Args:
            filename: Nombre del archivo
            config_type: Tipo de configuración
            
        Returns:
            Diccionario con información de la configuración
        """
        try:
            config = self.load_config(filename, config_type)
            
            return {
                "name": config.experiment_name,
                "description": config.description,
                "version": config.version,
                "model_type": config.model.model_type,
                "architecture": config.model.architecture,
                "epochs": config.training.epochs,
                "batch_size": config.data.batch_size,
                "created": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error obteniendo info de configuración: {e}")
            return {"error": str(e)}

def load_config(filepath: str) -> ExperimentConfig:
    """
    Función de conveniencia para cargar configuración.
    
    Args:
        filepath: Ruta del archivo de configuración
        
    Returns:
        Configuración cargada
    """
    return ExperimentConfig.load(filepath)

def save_config(config: ExperimentConfig, filepath: str):
    """
    Función de conveniencia para guardar configuración.
    
    Args:
        config: Configuración a guardar
        filepath: Ruta donde guardar
    """
    config.save(filepath)

def create_default_config(experiment_name: str = "default_experiment") -> ExperimentConfig:
    """
    Crear configuración por defecto.
    
    Args:
        experiment_name: Nombre del experimento
        
    Returns:
        Configuración por defecto
    """
    config = ExperimentConfig(experiment_name=experiment_name)
    
    # Configuración optimizada para CPU
    config.data.batch_size = 16  # Reducir para CPU
    config.data.num_workers = 2  # Menos workers para CPU
    config.training.epochs = 20  # Menos épocas para pruebas
    config.device = "cpu"
    
    logger.info(f"Configuración por defecto creada: {experiment_name}")
    return config

def create_config_from_template(template_name: str, 
                              experiment_name: str,
                              modifications: Optional[Dict[str, Any]] = None) -> ExperimentConfig:
    """
    Crear configuración desde template con modificaciones.
    
    Args:
        template_name: Nombre del template
        experiment_name: Nombre del nuevo experimento
        modifications: Modificaciones a aplicar
        
    Returns:
        Nueva configuración
    """
    manager = ConfigManager()
    
    # Cargar template
    try:
        config = manager.load_config(template_name, "preset")
    except FileNotFoundError:
        logger.warning(f"Template no encontrado: {template_name}, usando configuración por defecto")
        config = create_default_config(experiment_name)
    
    # Cambiar nombre del experimento
    config.experiment_name = experiment_name
    
    # Aplicar modificaciones si se proporcionan
    if modifications:
        config_dict = config.to_dict()
        
        # Aplicar modificaciones anidadas
        def apply_nested_modifications(target_dict, mod_dict):
            for key, value in mod_dict.items():
                if isinstance(value, dict) and key in target_dict:
                    apply_nested_modifications(target_dict[key], value)
                else:
                    target_dict[key] = value
        
        apply_nested_modifications(config_dict, modifications)
        config = ExperimentConfig.from_dict(config_dict)
    
    logger.info(f"Configuración creada desde template {template_name}: {experiment_name}")
    return config

def validate_config(config: ExperimentConfig) -> List[str]:
    """
    Validar configuración y retornar lista de problemas.
    
    Args:
        config: Configuración a validar
        
    Returns:
        Lista de problemas encontrados
    """
    problems = []
    
    # Validar rutas de datos
    if not os.path.exists(config.data.data_root):
        problems.append(f"Directorio de datos no existe: {config.data.data_root}")
    
    # Validar splits
    total_split = config.data.train_split + config.data.val_split + config.data.test_split
    if abs(total_split - 1.0) > 1e-6:
        problems.append(f"Los splits no suman 1.0: {total_split}")
    
    # Validar configuración de modelo
    if config.model.num_classes <= 0:
        problems.append("Número de clases debe ser positivo")
    
    if config.model.feature_dim <= 0:
        problems.append("Dimensión de features debe ser positiva")
    
    # Validar configuración de entrenamiento
    if config.training.epochs <= 0:
        problems.append("Número de épocas debe ser positivo")
    
    if config.training.learning_rate <= 0:
        problems.append("Learning rate debe ser positivo")
    
    # Validar batch size
    if config.data.batch_size <= 0:
        problems.append("Batch size debe ser positivo")
    
    return problems

# Test rápido
def quick_test():
    """Test rápido del gestor de configuraciones."""
    try:
        # Crear gestor
        manager = ConfigManager("/tmp/test_configs")
        print("✓ ConfigManager creado")
        
        # Crear configuración de prueba
        config = create_default_config("test_experiment")
        print("✓ Configuración por defecto creada")
        
        # Guardar configuración
        filepath = manager.save_config(config, "test_config.json")
        print("✓ Configuración guardada")
        
        # Cargar configuración
        loaded_config = manager.load_config("test_config.json")
        print("✓ Configuración cargada")
        
        # Validar configuración
        problems = validate_config(loaded_config)
        if not problems:
            print("✓ Configuración válida")
        else:
            print(f"⚠ Problemas encontrados: {problems}")
        
        # Crear presets
        manager.create_preset_configs()
        print("✓ Presets creados")
        
        # Listar configuraciones
        configs = manager.list_configs("preset")
        print(f"✓ Presets disponibles: {configs}")
        
        print("✅ Todas las pruebas del ConfigManager pasaron")
        return True
        
    except Exception as e:
        print(f"❌ Error en pruebas del ConfigManager: {e}")
        return False

if __name__ == "__main__":
    quick_test()