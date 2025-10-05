#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sistema de Configuración Unificado - SIGeC-Balistica
============================================

Sistema centralizado de configuración que unifica todas las configuraciones
dispersas del proyecto SIGeC-Balistica, proporcionando validación, migración automática
y gestión de configuraciones por entorno.

Características:
- Configuración centralizada y tipada
- Validación automática de valores
- Migración de configuraciones legacy
- Soporte para múltiples entornos (dev, test, prod)
- Configuración por variables de entorno
- Sistema de respaldo y recuperación

Autor: SIGeC-BalisticaTeam
Fecha: Octubre 2025
"""

import os
import sys
import yaml
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Type
from dataclasses import dataclass, field, asdict, fields
from enum import Enum
from datetime import datetime
import shutil

# Importar MatchingLevel y AlgorithmType desde unified_matcher
try:
    from matching.unified_matcher import MatchingLevel, AlgorithmType
except ImportError:
    # Fallback si no se puede importar
    class MatchingLevel(Enum):
        BASIC = "basic"
        STANDARD = "standard"
        ADVANCED = "advanced"
    
    class AlgorithmType(Enum):
        ORB = "ORB"
        SIFT = "SIFT"
        AKAZE = "AKAZE"
        BRISK = "BRISK"
        KAZE = "KAZE"
        CMC = "CMC"

# Configurar logging para este módulo
logger = logging.getLogger(__name__)

class Environment(Enum):
    """Entornos de ejecución soportados"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"

class ConfigValidationError(Exception):
    """Error de validación de configuración"""
    pass

class ConfigMigrationError(Exception):
    """Error durante migración de configuración"""
    pass

@dataclass
class DatabaseConfig:
    """Configuración unificada de base de datos"""
    # Rutas de base de datos
    sqlite_path: str = "database/ballistics.db"
    faiss_index_path: str = "database/faiss_index"
    db_path: str = "database"
    
    # Configuración de respaldo
    backup_path: str = "database/backups"
    backup_enabled: bool = True
    backup_interval_hours: int = 24
    backup_retention_days: int = 30
    
    # Configuración de rendimiento
    connection_pool_size: int = 10
    query_timeout: int = 30
    enable_wal_mode: bool = True
    
    def validate(self) -> List[str]:
        """Valida la configuración de base de datos"""
        errors = []
        
        if self.backup_interval_hours < 1:
            errors.append("backup_interval_hours debe ser mayor a 0")
        
        if self.backup_retention_days < 1:
            errors.append("backup_retention_days debe ser mayor a 0")
            
        if self.connection_pool_size < 1:
            errors.append("connection_pool_size debe ser mayor a 0")
            
        if self.query_timeout < 1:
            errors.append("query_timeout debe ser mayor a 0")
        
        return errors

@dataclass
class ImageProcessingConfig:
    """Configuración unificada de procesamiento de imágenes"""
    # Configuración básica
    max_image_size: int = 2048
    min_image_size: int = 64
    supported_formats: List[str] = field(default_factory=lambda: [
        '.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'
    ])
    
    # Rutas de procesamiento
    temp_path: str = "temp"
    output_path: str = "output"
    cache_path: str = "cache/images"
    
    # Configuración de algoritmos
    orb_features: int = 5000
    sift_features: int = 1000
    roi_detection_method: str = "simple"  # "simple", "advanced", "ml"
    
    # Configuración de preprocesamiento
    resize_images: bool = True
    maintain_aspect_ratio: bool = True
    gaussian_kernel_size: int = 5
    clahe_clip_limit: float = 2.0
    clahe_grid_size: int = 8
    enable_illumination_correction: bool = True
    
    # Configuración de mejoras específicas
    enhance_striations: bool = False
    enhance_breech_marks: bool = False
    enhance_firing_pin: bool = False
    
    # Configuración de rendimiento
    enable_parallel_processing: bool = True
    max_workers: int = 4
    memory_limit_mb: int = 1024
    
    def validate(self) -> List[str]:
        """Valida la configuración de procesamiento de imágenes"""
        errors = []
        
        if self.max_image_size < self.min_image_size:
            errors.append("max_image_size debe ser mayor que min_image_size")
            
        if self.orb_features < 100:
            errors.append("orb_features debe ser al menos 100")
            
        if self.sift_features < 100:
            errors.append("sift_features debe ser al menos 100")
            
        if self.roi_detection_method not in ["simple", "advanced", "ml"]:
            errors.append("roi_detection_method debe ser 'simple', 'advanced' o 'ml'")
            
        if self.gaussian_kernel_size % 2 == 0:
            errors.append("gaussian_kernel_size debe ser impar")
            
        if not (0.1 <= self.clahe_clip_limit <= 10.0):
            errors.append("clahe_clip_limit debe estar entre 0.1 y 10.0")
            
        if self.max_workers < 1:
            errors.append("max_workers debe ser mayor a 0")
            
        if self.memory_limit_mb < 128:
            errors.append("memory_limit_mb debe ser al menos 128")
        
        return errors

@dataclass
class MatchingConfig:
    """Configuración unificada de algoritmos de matching"""
    # Configuración básica de matching
    algorithm: AlgorithmType = AlgorithmType.ORB
    matcher_type: str = "BF"  # "BF" (BruteForce), "FLANN", "HYBRID"
    distance_threshold: float = 0.75  # Lowe's ratio test
    min_matches: int = 10
    similarity_threshold: float = 0.3
    max_results: int = 5
    max_features: int = 1000  # Número máximo de características a extraer
    
    # Configuración avanzada
    enable_geometric_verification: bool = True
    ransac_threshold: float = 5.0
    ransac_confidence: float = 0.99
    max_iterations: int = 2000
    
    # Configuración de filtros
    enable_cross_check: bool = True
    enable_ratio_test: bool = True
    enable_symmetry_test: bool = False
    
    # Configuración de rendimiento
    enable_parallel_matching: bool = True
    batch_size: int = 100
    cache_descriptors: bool = True
    
    # Configuración de aceleración GPU
    enable_gpu_acceleration: bool = False
    gpu_device_id: int = 0
    gpu_fallback_to_cpu: bool = True
    
    # Configuración específica de balística
    enable_firing_pin_analysis: bool = True
    enable_breech_face_analysis: bool = True
    enable_striation_analysis: bool = True
    enable_impression_analysis: bool = True
    
    # Atributo level requerido por UnifiedMatcher
    level: MatchingLevel = MatchingLevel.STANDARD
    
    def validate(self) -> List[str]:
        """Valida la configuración de matching"""
        errors = []
        
        if self.matcher_type not in ["BF", "FLANN", "HYBRID"]:
            errors.append("matcher_type debe ser 'BF', 'FLANN' o 'HYBRID'")
            
        if not (0.1 <= self.distance_threshold <= 1.0):
            errors.append("distance_threshold debe estar entre 0.1 y 1.0")
            
        if self.min_matches < 4:
            errors.append("min_matches debe ser al menos 4")
            
        if not (0.0 <= self.similarity_threshold <= 1.0):
            errors.append("similarity_threshold debe estar entre 0.0 y 1.0")
            
        if self.max_results < 1:
            errors.append("max_results debe ser mayor a 0")
            
        if not (0.1 <= self.ransac_threshold <= 50.0):
            errors.append("ransac_threshold debe estar entre 0.1 y 50.0")
            
        if not (0.5 <= self.ransac_confidence <= 0.999):
            errors.append("ransac_confidence debe estar entre 0.5 y 0.999")
            
        if self.batch_size < 1:
            errors.append("batch_size debe ser mayor a 0")
        
        return errors

@dataclass
class GUIConfig:
    """Configuración unificada de interfaz gráfica"""
    # Configuración de ventana
    window_width: int = 1200
    window_height: int = 800
    min_window_width: int = 800
    min_window_height: int = 600
    
    # Configuración de apariencia
    theme: str = "default"  # "default", "dark", "light"
    language: str = "es"  # "es", "en"
    font_family: str = "Arial"
    font_size: int = 10
    
    # Configuración de comportamiento
    auto_save: bool = True
    auto_save_interval: int = 300  # segundos
    remember_window_state: bool = True
    show_tooltips: bool = True
    enable_animations: bool = True
    
    # Configuración de visualización
    image_zoom_step: float = 0.1
    max_zoom_level: float = 10.0
    min_zoom_level: float = 0.1
    default_image_quality: int = 95
    
    # Configuración de paneles
    show_metadata_panel: bool = True
    show_statistics_panel: bool = True
    show_progress_panel: bool = True
    panel_transparency: float = 0.9
    
    def validate(self) -> List[str]:
        """Valida la configuración de GUI"""
        errors = []
        
        if self.window_width < self.min_window_width:
            errors.append(f"window_width debe ser al menos {self.min_window_width}")
            
        if self.window_height < self.min_window_height:
            errors.append(f"window_height debe ser al menos {self.min_window_height}")
            
        if self.theme not in ["default", "dark", "light"]:
            errors.append("theme debe ser 'default', 'dark' o 'light'")
            
        if self.language not in ["es", "en"]:
            errors.append("language debe ser 'es' o 'en'")
            
        if self.font_size < 6 or self.font_size > 24:
            errors.append("font_size debe estar entre 6 y 24")
            
        if self.auto_save_interval < 60:
            errors.append("auto_save_interval debe ser al menos 60 segundos")
            
        if not (0.01 <= self.image_zoom_step <= 1.0):
            errors.append("image_zoom_step debe estar entre 0.01 y 1.0")
            
        if self.max_zoom_level <= self.min_zoom_level:
            errors.append("max_zoom_level debe ser mayor que min_zoom_level")
            
        if not (50 <= self.default_image_quality <= 100):
            errors.append("default_image_quality debe estar entre 50 y 100")
            
        if not (0.1 <= self.panel_transparency <= 1.0):
            errors.append("panel_transparency debe estar entre 0.1 y 1.0")
        
        return errors

@dataclass
class LoggingConfig:
    """Configuración unificada de logging"""
    # Configuración básica
    level: str = "INFO"  # "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
    console_output: bool = True
    
    # Configuración de archivos
    file_path: str = "logs/ballistics.log"
    log_path: str = "logs"
    max_file_size: str = "10MB"
    backup_count: int = 5
    
    # Configuración de formato
    format_string: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"
    
    # Configuración avanzada
    enable_rotation: bool = True
    enable_compression: bool = True
    log_performance_metrics: bool = True
    log_memory_usage: bool = False
    
    # Configuración por módulos
    module_levels: Dict[str, str] = field(default_factory=dict)
    
    def validate(self) -> List[str]:
        """Valida la configuración de logging"""
        errors = []
        
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.level not in valid_levels:
            errors.append(f"level debe ser uno de: {', '.join(valid_levels)}")
            
        if self.backup_count < 0:
            errors.append("backup_count debe ser mayor o igual a 0")
            
        # Validar formato de tamaño de archivo
        if not self.max_file_size.endswith(('B', 'KB', 'MB', 'GB')):
            errors.append("max_file_size debe terminar en B, KB, MB o GB")
        
        return errors

@dataclass
class DeepLearningConfig:
    """Configuración unificada de deep learning"""
    # Configuración básica
    enabled: bool = False
    device: str = "cpu"  # "cpu", "cuda", "auto"
    
    # Configuración de entrenamiento
    batch_size: int = 16
    learning_rate: float = 0.001
    epochs: int = 100
    dropout_rate: float = 0.5
    
    # Configuración de modelos
    models_path: str = "deep_learning/models"
    checkpoints_path: str = "deep_learning/checkpoints"
    tensorboard_path: str = "deep_learning/logs"
    
    # Configuración específica por modelo
    cnn_config: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": False,
        "input_size": 224,
        "num_classes": 10,
        "hidden_layers": [512, 256, 128]
    })
    
    autoencoder_config: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": False,
        "input_size": 224,
        "latent_dim": 128,
        "hidden_layers": [512, 256]
    })
    
    siamese_config: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": False,
        "input_size": 224,
        "embedding_dim": 128,
        "margin": 1.0
    })
    
    def validate(self) -> List[str]:
        """Valida la configuración de deep learning"""
        errors = []
        
        if self.device not in ["cpu", "cuda", "auto"]:
            errors.append("device debe ser 'cpu', 'cuda' o 'auto'")
            
        if self.batch_size < 1:
            errors.append("batch_size debe ser mayor a 0")
            
        if not (0.0001 <= self.learning_rate <= 1.0):
            errors.append("learning_rate debe estar entre 0.0001 y 1.0")
            
        if self.epochs < 1:
            errors.append("epochs debe ser mayor a 0")
            
        if not (0.0 <= self.dropout_rate <= 1.0):
            errors.append("dropout_rate debe estar entre 0.0 y 1.0")
        
        return errors

@dataclass
class NISTConfig:
    """Configuración para estándares NIST/AFTE"""
    # Configuración de estándares
    enable_nist_compliance: bool = True
    enable_afte_standards: bool = True
    
    # Configuración de métricas de calidad
    min_quality_score: float = 0.7
    enable_quality_weighting: bool = True
    quality_threshold: float = 0.8
    
    # Configuración de reportes
    generate_compliance_reports: bool = True
    report_format: str = "json"  # "json", "xml", "pdf"
    include_metadata: bool = True
    
    # Configuración de validación
    enable_chain_of_custody: bool = True
    require_examiner_certification: bool = False
    enable_peer_review: bool = True
    
    def validate(self) -> List[str]:
        """Valida la configuración NIST"""
        errors = []
        
        if not (0.0 <= self.min_quality_score <= 1.0):
            errors.append("min_quality_score debe estar entre 0.0 y 1.0")
            
        if not (0.0 <= self.quality_threshold <= 1.0):
            errors.append("quality_threshold debe estar entre 0.0 y 1.0")
            
        if self.report_format not in ["json", "xml", "pdf"]:
            errors.append("report_format debe ser 'json', 'xml' o 'pdf'")
        
        return errors

class UnifiedConfig:
    """
    Sistema de configuración unificado para SIGeC-Balistica
    
    Centraliza todas las configuraciones del sistema y proporciona:
    - Validación automática
    - Migración de configuraciones legacy
    - Soporte para múltiples entornos
    - Configuración por variables de entorno
    """
    
    def __init__(self, 
                 config_file: Optional[str] = None,
                 environment: Optional[Environment] = None,
                 auto_migrate: bool = True):
        """
        Inicializa el sistema de configuración unificado
        
        Args:
            config_file: Ruta al archivo de configuración
            environment: Entorno de ejecución
            auto_migrate: Si debe migrar automáticamente configuraciones legacy
        """
        # Configuración básica
        self.environment = environment or self._detect_environment()
        self.config_file = config_file or self._get_default_config_file()
        self.config_path = Path(self.config_file)
        self.auto_migrate = auto_migrate
        
        # Rutas importantes
        self.project_root = self._get_project_root()
        self.config_dir = self.project_root / "config"
        self.backup_dir = self.config_dir / "backups"
        
        # Crear directorios necesarios
        self._create_directories()
        
        # Inicializar configuraciones con valores por defecto
        self.database = DatabaseConfig()
        self.image_processing = ImageProcessingConfig()
        self.matching = MatchingConfig()
        self.gui = GUIConfig()
        self.logging = LoggingConfig()
        self.deep_learning = DeepLearningConfig()
        self.nist = NISTConfig()
        
        # Cargar configuración
        self._load_configuration()
        
        # Migrar configuraciones legacy si es necesario
        if self.auto_migrate:
            self._migrate_legacy_configs()
        
        # Validar configuración
        self.validate()
        
        logger.info(f"Sistema de configuración inicializado para entorno: {self.environment.value}")
    
    def _detect_environment(self) -> Environment:
        """Detecta el entorno de ejecución"""
        env_var = os.getenv('SIGeC-Balistica_ENV', 'development').lower()
        
        if env_var in ['prod', 'production']:
            return Environment.PRODUCTION
        elif env_var in ['test', 'testing']:
            return Environment.TESTING
        else:
            return Environment.DEVELOPMENT
    
    def _get_default_config_file(self) -> str:
        """Obtiene el archivo de configuración por defecto según el entorno"""
        env_suffix = "" if self.environment == Environment.DEVELOPMENT else f"_{self.environment.value}"
        return f"config/unified_config{env_suffix}.yaml"
    
    def _get_project_root(self) -> Path:
        """Obtiene la ruta raíz del proyecto"""
        # Buscar desde el directorio actual hacia arriba
        current = Path.cwd()
        
        # Buscar indicadores del proyecto SIGeC-Balistica
        indicators = ['main.py', 'gui', 'matching', 'requirements.txt']
        
        for parent in [current] + list(current.parents):
            if all((parent / indicator).exists() for indicator in indicators[:2]):
                return parent
        
        # Si no se encuentra, usar directorio actual
        return current
    
    def _create_directories(self) -> None:
        """Crea los directorios necesarios"""
        directories = [
            self.config_dir,
            self.backup_dir,
            self.project_root / "logs",
            self.project_root / "temp",
            self.project_root / "cache",
            self.project_root / "database" / "backups"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def _load_configuration(self) -> None:
        """Carga la configuración desde archivo"""
        if not self.config_path.exists():
            logger.info(f"Archivo de configuración no encontrado: {self.config_path}")
            logger.info("Usando configuración por defecto")
            self.save_config()
            return
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            
            if config_data:
                self._update_from_dict(config_data)
                logger.info(f"Configuración cargada desde: {self.config_path}")
            
        except Exception as e:
            logger.error(f"Error al cargar configuración: {e}")
            logger.info("Usando configuración por defecto")
    
    def _update_from_dict(self, config_data: Dict[str, Any]) -> None:
        """Actualiza la configuración desde un diccionario"""
        # Mapeo de secciones a objetos de configuración
        section_mapping = {
            'database': self.database,
            'image_processing': self.image_processing,
            'matching': self.matching,
            'gui': self.gui,
            'logging': self.logging,
            'deep_learning': self.deep_learning,
            'nist': self.nist
        }
        
        for section_name, section_data in config_data.items():
            if section_name in section_mapping and isinstance(section_data, dict):
                config_obj = section_mapping[section_name]
                
                # Actualizar campos del dataclass
                for field_name, field_value in section_data.items():
                    if hasattr(config_obj, field_name):
                        # Manejar conversiones especiales para enums
                        if section_name == 'matching' and field_name == 'algorithm' and isinstance(field_value, str):
                            # Convertir string a AlgorithmType enum
                            try:
                                algorithm_enum = next((a for a in AlgorithmType if a.value == field_value), AlgorithmType.ORB)
                                setattr(config_obj, field_name, algorithm_enum)
                            except Exception as e:
                                logger.warning(f"Error convirtiendo algorithm '{field_value}' a enum, usando ORB por defecto: {e}")
                                setattr(config_obj, field_name, AlgorithmType.ORB)
                        elif section_name == 'matching' and field_name == 'level' and isinstance(field_value, str):
                            # Convertir string a MatchingLevel enum
                            try:
                                level_enum = next((l for l in MatchingLevel if l.value == field_value), MatchingLevel.STANDARD)
                                setattr(config_obj, field_name, level_enum)
                            except Exception as e:
                                logger.warning(f"Error convirtiendo level '{field_value}' a enum, usando STANDARD por defecto: {e}")
                                setattr(config_obj, field_name, MatchingLevel.STANDARD)
                        else:
                            setattr(config_obj, field_name, field_value)
    
    def _migrate_legacy_configs(self) -> None:
        """Migra configuraciones legacy al sistema unificado"""
        legacy_files = [
            self.project_root / "config.yaml",
            self.project_root / "tests" / "config.yaml"
        ]
        
        migrated_any = False
        
        for legacy_file in legacy_files:
            if legacy_file.exists():
                try:
                    self._migrate_single_config(legacy_file)
                    migrated_any = True
                except Exception as e:
                    logger.error(f"Error migrando {legacy_file}: {e}")
        
        if migrated_any:
            logger.info("Configuraciones legacy migradas exitosamente")
            self.save_config()
    
    def _migrate_single_config(self, config_file: Path) -> None:
        """Migra un archivo de configuración individual"""
        with open(config_file, 'r', encoding='utf-8') as f:
            legacy_data = yaml.safe_load(f)
        
        if not legacy_data:
            return
        
        # Crear respaldo
        backup_file = self.backup_dir / f"{config_file.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
        shutil.copy2(config_file, backup_file)
        
        # Migrar datos
        self._update_from_dict(legacy_data)
        
        logger.info(f"Configuración migrada desde: {config_file}")
        logger.info(f"Respaldo creado en: {backup_file}")
    
    def validate(self) -> None:
        """Valida toda la configuración"""
        all_errors = []
        
        # Validar cada sección
        sections = {
            'database': self.database,
            'image_processing': self.image_processing,
            'matching': self.matching,
            'gui': self.gui,
            'logging': self.logging,
            'deep_learning': self.deep_learning,
            'nist': self.nist
        }
        
        for section_name, section_config in sections.items():
            if hasattr(section_config, 'validate'):
                errors = section_config.validate()
                if errors:
                    all_errors.extend([f"{section_name}.{error}" for error in errors])
        
        if all_errors:
            error_msg = "Errores de validación de configuración:\n" + "\n".join(f"- {error}" for error in all_errors)
            raise ConfigValidationError(error_msg)
        
        logger.info("Configuración validada exitosamente")
    
    def save_config(self) -> None:
        """Guarda la configuración actual"""
        config_dict = {
            'database': asdict(self.database),
            'image_processing': asdict(self.image_processing),
            'matching': asdict(self.matching),
            'gui': asdict(self.gui),
            'logging': asdict(self.logging),
            'deep_learning': asdict(self.deep_learning),
            'nist': asdict(self.nist),
            '_metadata': {
                'version': '1.0.0',
                'environment': self.environment.value,
                'created': datetime.now().isoformat(),
                'project_root': str(self.project_root)
            }
        }
        
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_dict, f, default_flow_style=False, 
                         allow_unicode=True, indent=2, sort_keys=True)
            
            logger.info(f"Configuración guardada en: {self.config_path}")
            
        except Exception as e:
            logger.error(f"Error al guardar configuración: {e}")
            raise
    
    def get_absolute_path(self, relative_path: str) -> Path:
        """Convierte una ruta relativa en absoluta basada en project_root"""
        if Path(relative_path).is_absolute():
            return Path(relative_path)
        return self.project_root / relative_path
    
    def get_config_dict(self) -> Dict[str, Any]:
        """Retorna la configuración completa como diccionario"""
        return {
            'database': asdict(self.database),
            'image_processing': asdict(self.image_processing),
            'matching': asdict(self.matching),
            'gui': asdict(self.gui),
            'logging': asdict(self.logging),
            'deep_learning': asdict(self.deep_learning),
            'nist': asdict(self.nist)
        }
    
    def update_config(self, section: str, **kwargs) -> None:
        """Actualiza una sección específica de la configuración"""
        section_mapping = {
            'database': self.database,
            'image_processing': self.image_processing,
            'matching': self.matching,
            'gui': self.gui,
            'logging': self.logging,
            'deep_learning': self.deep_learning,
            'nist': self.nist
        }
        
        if section not in section_mapping:
            raise ValueError(f"Sección desconocida: {section}")
        
        config_obj = section_mapping[section]
        
        for key, value in kwargs.items():
            if hasattr(config_obj, key):
                setattr(config_obj, key, value)
            else:
                logger.warning(f"Campo desconocido {key} en sección {section}")
        
        # Validar después de actualizar
        if hasattr(config_obj, 'validate'):
            errors = config_obj.validate()
            if errors:
                error_msg = f"Errores de validación en {section}:\n" + "\n".join(f"- {error}" for error in errors)
                raise ConfigValidationError(error_msg)
    
    def reset_to_defaults(self, section: Optional[str] = None) -> None:
        """Resetea la configuración a valores por defecto"""
        if section:
            section_mapping = {
                'database': DatabaseConfig,
                'image_processing': ImageProcessingConfig,
                'matching': MatchingConfig,
                'gui': GUIConfig,
                'logging': LoggingConfig,
                'deep_learning': DeepLearningConfig,
                'nist': NISTConfig
            }
            
            if section in section_mapping:
                setattr(self, section, section_mapping[section]())
                logger.info(f"Sección {section} reseteada a valores por defecto")
            else:
                raise ValueError(f"Sección desconocida: {section}")
        else:
            # Resetear toda la configuración
            self.database = DatabaseConfig()
            self.image_processing = ImageProcessingConfig()
            self.matching = MatchingConfig()
            self.gui = GUIConfig()
            self.logging = LoggingConfig()
            self.deep_learning = DeepLearningConfig()
            self.nist = NISTConfig()
            logger.info("Toda la configuración reseteada a valores por defecto")
    
    def export_config(self, format: str = "yaml", file_path: Optional[str] = None) -> str:
        """Exporta la configuración a un archivo"""
        if not file_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = f"config_export_{timestamp}.{format}"
        
        config_dict = self.get_config_dict()
        
        if format.lower() == "json":
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
        elif format.lower() == "yaml":
            with open(file_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_dict, f, default_flow_style=False, 
                         allow_unicode=True, indent=2)
        else:
            raise ValueError(f"Formato no soportado: {format}")
        
        logger.info(f"Configuración exportada a: {file_path}")
        return file_path

# Instancia global del sistema de configuración
_unified_config_instance: Optional[UnifiedConfig] = None

def get_unified_config(config_file: Optional[str] = None,
                      environment: Optional[Environment] = None,
                      force_reload: bool = False) -> UnifiedConfig:
    """
    Obtiene la instancia global del sistema de configuración unificado
    
    Args:
        config_file: Archivo de configuración personalizado
        environment: Entorno específico
        force_reload: Forzar recarga de la configuración
    
    Returns:
        Instancia del sistema de configuración unificado
    """
    global _unified_config_instance
    
    if _unified_config_instance is None or force_reload:
        _unified_config_instance = UnifiedConfig(
            config_file=config_file,
            environment=environment
        )
    
    return _unified_config_instance

def reset_unified_config() -> None:
    """Resetea la instancia global de configuración"""
    global _unified_config_instance
    _unified_config_instance = None

# Funciones de conveniencia para compatibilidad con código existente
def get_config() -> UnifiedConfig:
    """Función de compatibilidad con utils/config.py"""
    return get_unified_config()

def get_database_config() -> DatabaseConfig:
    """Obtiene configuración de base de datos"""
    return get_unified_config().database

def get_image_processing_config() -> ImageProcessingConfig:
    """Obtiene configuración de procesamiento de imágenes"""
    return get_unified_config().image_processing

def get_matching_config() -> MatchingConfig:
    """Obtiene configuración de matching"""
    return get_unified_config().matching

def get_gui_config() -> GUIConfig:
    """Obtiene configuración de GUI"""
    return get_unified_config().gui

def get_logging_config() -> LoggingConfig:
    """Obtiene configuración de logging"""
    return get_unified_config().logging

def get_deep_learning_config() -> DeepLearningConfig:
    """Obtiene configuración de deep learning"""
    return get_unified_config().deep_learning

def get_nist_config() -> NISTConfig:
    """Obtiene configuración NIST"""
    return get_unified_config().nist

# Configuración de logging para este módulo
if __name__ != "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

# Ejemplo de uso
if __name__ == "__main__":
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Crear instancia de configuración
        config = UnifiedConfig()
        
        # Mostrar configuración actual
        print("=== Configuración Unificada SIGeC-Balistica===")
        print(f"Entorno: {config.environment.value}")
        print(f"Archivo de configuración: {config.config_path}")
        print(f"Directorio del proyecto: {config.project_root}")
        
        # Validar configuración
        config.validate()
        print("✅ Configuración validada exitosamente")
        
        # Guardar configuración
        config.save_config()
        print("✅ Configuración guardada")
        
        # Exportar configuración
        export_file = config.export_config("json")
        print(f"✅ Configuración exportada a: {export_file}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)