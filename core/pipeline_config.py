"""
Configuración del Pipeline Científico Unificado
==============================================

Este módulo define las configuraciones específicas para el pipeline científico,
incluyendo diferentes niveles de análisis y parámetros optimizados para cada caso de uso.

Niveles de Análisis:
- Basic: Análisis rápido para casos simples
- Standard: Análisis balanceado para uso general
- Advanced: Análisis detallado para casos complejos
- Forensic: Análisis exhaustivo para uso forense
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from enum import Enum

from image_processing.unified_preprocessor import PreprocessingLevel
from image_processing.unified_roi_detector import DetectionLevel
from matching.unified_matcher import AlgorithmType


class PipelineLevel(Enum):
    """Niveles de análisis del pipeline"""
    BASIC = "basic"
    STANDARD = "standard"
    ADVANCED = "advanced"
    FORENSIC = "forensic"


@dataclass
class QualityAssessmentConfig:
    """Configuración para evaluación de calidad NIST"""
    enabled: bool = True
    min_snr: float = 20.0
    min_contrast: float = 0.3
    min_uniformity: float = 0.7
    min_sharpness: float = 0.5
    min_resolution: float = 300.0
    strict_mode: bool = False
    
    # Umbrales específicos por nivel
    level_thresholds: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "basic": {
            "min_snr": 15.0,
            "min_contrast": 0.2,
            "min_uniformity": 0.6,
            "min_sharpness": 0.4,
            "min_resolution": 200.0
        },
        "standard": {
            "min_snr": 20.0,
            "min_contrast": 0.3,
            "min_uniformity": 0.7,
            "min_sharpness": 0.5,
            "min_resolution": 300.0
        },
        "advanced": {
            "min_snr": 25.0,
            "min_contrast": 0.4,
            "min_uniformity": 0.75,
            "min_sharpness": 0.6,
            "min_resolution": 400.0
        },
        "forensic": {
            "min_snr": 30.0,
            "min_contrast": 0.5,
            "min_uniformity": 0.8,
            "min_sharpness": 0.7,
            "min_resolution": 600.0
        }
    })


@dataclass
class PreprocessingConfig:
    """Configuración para preprocesamiento"""
    level: PreprocessingLevel = PreprocessingLevel.STANDARD
    enable_noise_reduction: bool = True
    enable_contrast_enhancement: bool = True
    enable_sharpening: bool = True
    enable_normalization: bool = True
    
    # Parámetros específicos
    gaussian_kernel_size: int = 5
    gaussian_sigma: float = 1.0
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: tuple = (8, 8)
    unsharp_strength: float = 1.5
    unsharp_radius: float = 1.0
    
    # Configuración por nivel
    level_configs: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "basic": {
            "enable_noise_reduction": True,
            "enable_contrast_enhancement": False,
            "enable_sharpening": False,
            "enable_normalization": True,
            "gaussian_sigma": 0.8,
            "clahe_clip_limit": 1.5
        },
        "standard": {
            "enable_noise_reduction": True,
            "enable_contrast_enhancement": True,
            "enable_sharpening": True,
            "enable_normalization": True,
            "gaussian_sigma": 1.0,
            "clahe_clip_limit": 2.0,
            "unsharp_strength": 1.5
        },
        "advanced": {
            "enable_noise_reduction": True,
            "enable_contrast_enhancement": True,
            "enable_sharpening": True,
            "enable_normalization": True,
            "gaussian_sigma": 1.2,
            "clahe_clip_limit": 2.5,
            "unsharp_strength": 2.0
        },
        "forensic": {
            "enable_noise_reduction": True,
            "enable_contrast_enhancement": True,
            "enable_sharpening": True,
            "enable_normalization": True,
            "gaussian_sigma": 1.5,
            "clahe_clip_limit": 3.0,
            "unsharp_strength": 2.5
        }
    })


@dataclass
class ROIDetectionConfig:
    """Configuración para detección de ROI"""
    enabled: bool = True
    detection_level: DetectionLevel = DetectionLevel.STANDARD
    min_roi_area: int = 1000
    max_roi_count: int = 5
    
    # Parámetros de watershed
    watershed_markers: int = 10
    watershed_compactness: float = 0.1
    watershed_sigma: float = 1.0
    
    # Configuración por nivel
    level_configs: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "basic": {
            "detection_level": DetectionLevel.SIMPLE,
            "min_roi_area": 2000,
            "max_roi_count": 3,
            "watershed_markers": 5
        },
        "standard": {
            "detection_level": DetectionLevel.STANDARD,
            "min_roi_area": 1000,
            "max_roi_count": 5,
            "watershed_markers": 10
        },
        "advanced": {
            "detection_level": DetectionLevel.ADVANCED,
            "min_roi_area": 800,
            "max_roi_count": 7,
            "watershed_markers": 15
        },
        "forensic": {
            "detection_level": DetectionLevel.ADVANCED,  # FORENSIC no existe, usar ADVANCED
            "min_roi_area": 500,
            "max_roi_count": 10,
            "watershed_markers": 20
        }
    })


@dataclass
class MatchingConfig:
    """Configuración para matching de características"""
    algorithm: AlgorithmType = AlgorithmType.ORB
    similarity_threshold: float = 0.7
    max_features: int = 5000
    enable_ransac: bool = True
    ransac_threshold: float = 5.0
    ransac_max_trials: int = 1000
    
    # Parámetros específicos por algoritmo
    orb_n_features: int = 5000
    orb_scale_factor: float = 1.2
    orb_n_levels: int = 8
    
    sift_n_features: int = 0  # 0 = sin límite
    sift_n_octave_layers: int = 3
    sift_contrast_threshold: float = 0.04
    sift_edge_threshold: float = 10.0
    sift_sigma: float = 1.6
    
    # Configuración por nivel
    level_configs: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "basic": {
            "algorithm": AlgorithmType.ORB,
            "similarity_threshold": 0.5,
            "max_features": 2000,
            "orb_n_features": 2000,
            "ransac_threshold": 8.0
        },
        "standard": {
            "algorithm": AlgorithmType.ORB,
            "similarity_threshold": 0.7,
            "max_features": 5000,
            "orb_n_features": 5000,
            "ransac_threshold": 5.0
        },
        "advanced": {
            "algorithm": AlgorithmType.SIFT,
            "similarity_threshold": 0.75,
            "max_features": 8000,
            "sift_contrast_threshold": 0.03,
            "ransac_threshold": 3.0
        },
        "forensic": {
            "algorithm": AlgorithmType.SIFT,
            "similarity_threshold": 0.8,
            "max_features": 10000,
            "sift_contrast_threshold": 0.02,
            "ransac_threshold": 2.0
        }
    })


@dataclass
class CMCAnalysisConfig:
    """Configuración para análisis CMC"""
    enabled: bool = True
    cmc_threshold: int = 6
    min_cell_size: int = 10
    max_cell_size: int = 100
    overlap_threshold: float = 0.3
    
    # Configuración por nivel
    level_configs: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "basic": {
            "cmc_threshold": 4,
            "min_cell_size": 15,
            "max_cell_size": 80
        },
        "standard": {
            "cmc_threshold": 6,
            "min_cell_size": 10,
            "max_cell_size": 100
        },
        "advanced": {
            "cmc_threshold": 6,
            "min_cell_size": 8,
            "max_cell_size": 120
        },
        "forensic": {
            "cmc_threshold": 8,
            "min_cell_size": 5,
            "max_cell_size": 150
        }
    })


@dataclass
class AFTEConclusionConfig:
    """Configuración para determinación de conclusiones AFTE"""
    # Umbrales para conclusiones
    identification_threshold: float = 0.8
    elimination_threshold: float = 0.3
    
    # Pesos para diferentes métricas
    similarity_weight: float = 0.4
    quality_weight: float = 0.2
    cmc_weight: float = 0.3
    consistency_weight: float = 0.1
    
    # Configuración por nivel
    level_configs: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "basic": {
            "identification_threshold": 0.7,
            "elimination_threshold": 0.4,
            "similarity_weight": 0.6,
            "quality_weight": 0.1,
            "cmc_weight": 0.2,
            "consistency_weight": 0.1
        },
        "standard": {
            "identification_threshold": 0.8,
            "elimination_threshold": 0.3,
            "similarity_weight": 0.4,
            "quality_weight": 0.2,
            "cmc_weight": 0.3,
            "consistency_weight": 0.1
        },
        "advanced": {
            "identification_threshold": 0.85,
            "elimination_threshold": 0.25,
            "similarity_weight": 0.35,
            "quality_weight": 0.25,
            "cmc_weight": 0.3,
            "consistency_weight": 0.1
        },
        "forensic": {
            "identification_threshold": 0.9,
            "elimination_threshold": 0.2,
            "similarity_weight": 0.3,
            "quality_weight": 0.3,
            "cmc_weight": 0.3,
            "consistency_weight": 0.1
        }
    })


@dataclass
class PipelineConfiguration:
    """Configuración completa del pipeline científico"""
    level: PipelineLevel = PipelineLevel.STANDARD
    
    # Configuraciones de componentes
    quality_assessment: QualityAssessmentConfig = field(default_factory=QualityAssessmentConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    roi_detection: ROIDetectionConfig = field(default_factory=ROIDetectionConfig)
    matching: MatchingConfig = field(default_factory=MatchingConfig)
    cmc_analysis: CMCAnalysisConfig = field(default_factory=CMCAnalysisConfig)
    afte_conclusion: AFTEConclusionConfig = field(default_factory=AFTEConclusionConfig)
    
    # Configuraciones generales
    enable_parallel_processing: bool = True
    max_processing_threads: int = 4
    enable_caching: bool = True
    cache_directory: str = "cache/pipeline"
    
    # Configuraciones de exportación
    export_intermediate_results: bool = False
    export_visualizations: bool = True
    export_detailed_report: bool = True
    
    def apply_level_configuration(self, level: str):
        """Aplica configuración específica del nivel"""
        level = level.lower()
        
        # Actualizar el nivel del pipeline
        self.level = PipelineLevel(level)
        
        # Aplicar configuraciones de calidad
        if level in self.quality_assessment.level_thresholds:
            thresholds = self.quality_assessment.level_thresholds[level]
            for key, value in thresholds.items():
                setattr(self.quality_assessment, key, value)
        
        # Aplicar configuraciones de preprocesamiento
        if level in self.preprocessing.level_configs:
            config = self.preprocessing.level_configs[level]
            for key, value in config.items():
                setattr(self.preprocessing, key, value)
        
        # Aplicar configuraciones de ROI
        if level in self.roi_detection.level_configs:
            config = self.roi_detection.level_configs[level]
            for key, value in config.items():
                if key == "detection_level":
                    self.roi_detection.detection_level = DetectionLevel(value)
                else:
                    setattr(self.roi_detection, key, value)
        
        # Aplicar configuraciones de matching
        if level in self.matching.level_configs:
            config = self.matching.level_configs[level]
            for key, value in config.items():
                if key == "algorithm":
                    self.matching.algorithm = AlgorithmType(value)
                else:
                    setattr(self.matching, key, value)
        
        # Aplicar configuraciones de CMC
        if level in self.cmc_analysis.level_configs:
            config = self.cmc_analysis.level_configs[level]
            for key, value in config.items():
                setattr(self.cmc_analysis, key, value)
        
        # Aplicar configuraciones de conclusión AFTE
        if level in self.afte_conclusion.level_configs:
            config = self.afte_conclusion.level_configs[level]
            for key, value in config.items():
                setattr(self.afte_conclusion, key, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte la configuración a diccionario"""
        return {
            "level": {"value": self.level.value},
            "quality_assessment": {
                "enabled": self.quality_assessment.enabled,
                "min_snr": self.quality_assessment.min_snr,
                "min_contrast": self.quality_assessment.min_contrast,
                "min_uniformity": self.quality_assessment.min_uniformity,
                "min_sharpness": self.quality_assessment.min_sharpness,
                "min_resolution": self.quality_assessment.min_resolution,
                "strict_mode": self.quality_assessment.strict_mode
            },
            "preprocessing": {
                "level": self.preprocessing.level.value,
                "enable_noise_reduction": self.preprocessing.enable_noise_reduction,
                "enable_contrast_enhancement": self.preprocessing.enable_contrast_enhancement,
                "enable_sharpening": self.preprocessing.enable_sharpening,
                "enable_normalization": self.preprocessing.enable_normalization
            },
            "roi_detection": {
                "enabled": self.roi_detection.enabled,
                "detection_level": self.roi_detection.detection_level.value,
                "min_roi_area": self.roi_detection.min_roi_area,
                "max_roi_count": self.roi_detection.max_roi_count
            },
            "matching": {
                "algorithm": self.matching.algorithm.value,
                "similarity_threshold": self.matching.similarity_threshold,
                "max_features": self.matching.max_features,
                "enable_ransac": self.matching.enable_ransac
            },
            "cmc_analysis": {
                "enabled": self.cmc_analysis.enabled,
                "cmc_threshold": self.cmc_analysis.cmc_threshold,
                "min_cell_size": self.cmc_analysis.min_cell_size,
                "max_cell_size": self.cmc_analysis.max_cell_size
            },
            "afte_conclusion": {
                "identification_threshold": self.afte_conclusion.identification_threshold,
                "elimination_threshold": self.afte_conclusion.elimination_threshold,
                "similarity_weight": self.afte_conclusion.similarity_weight,
                "quality_weight": self.afte_conclusion.quality_weight,
                "cmc_weight": self.afte_conclusion.cmc_weight
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PipelineConfiguration':
        """Crea configuración desde diccionario"""
        config = cls()
        
        # Aplicar nivel si está especificado
        if "level" in data:
            level_value = data["level"]
            # Manejar tanto formato string como diccionario
            if isinstance(level_value, dict) and "value" in level_value:
                level_value = level_value["value"]
            config.level = PipelineLevel(level_value)
            config.apply_level_configuration(level_value)
        
        # Aplicar configuraciones específicas
        if "quality_assessment" in data:
            qa_data = data["quality_assessment"]
            for key, value in qa_data.items():
                if hasattr(config.quality_assessment, key):
                    setattr(config.quality_assessment, key, value)
        
        if "roi_detection" in data:
            roi_data = data["roi_detection"]
            for key, value in roi_data.items():
                if hasattr(config.roi_detection, key):
                    setattr(config.roi_detection, key, value)
        
        if "cmc_analysis" in data:
            cmc_data = data["cmc_analysis"]
            for key, value in cmc_data.items():
                if hasattr(config.cmc_analysis, key):
                    setattr(config.cmc_analysis, key, value)
        
        if "matching" in data:
            match_data = data["matching"]
            for key, value in match_data.items():
                if key == "algorithm":
                    config.matching.algorithm = AlgorithmType(value)
                elif hasattr(config.matching, key):
                    setattr(config.matching, key, value)
        
        # Aplicar configuraciones generales
        for key in ["export_intermediate_results", "export_visualizations", 
                   "export_detailed_report", "enable_parallel_processing",
                   "max_processing_threads", "enable_caching", "cache_directory"]:
            if key in data:
                setattr(config, key, data[key])
        
        return config


def create_pipeline_config(level: str = "standard") -> PipelineConfiguration:
    """
    Crea una configuración de pipeline para el nivel especificado
    
    Args:
        level: Nivel de análisis ("basic", "standard", "advanced", "forensic")
    
    Returns:
        PipelineConfiguration configurada para el nivel especificado
    """
    config = PipelineConfiguration()
    config.level = PipelineLevel(level.lower())
    config.apply_level_configuration(level.lower())
    
    return config


def get_available_levels() -> List[str]:
    """Retorna los niveles de análisis disponibles"""
    return [level.value for level in PipelineLevel]


def get_level_description(level: str) -> str:
    """Retorna la descripción de un nivel de análisis"""
    descriptions = {
        "basic": "Análisis rápido con configuración básica para casos simples",
        "standard": "Análisis balanceado con configuración estándar para uso general",
        "advanced": "Análisis detallado con configuración avanzada para casos complejos",
        "forensic": "Análisis exhaustivo con configuración forense para máxima precisión"
    }
    return descriptions.get(level.lower(), "Nivel no reconocido")


def get_recommended_level(image_quality: float, case_complexity: str = "medium") -> str:
    """
    Recomienda un nivel de análisis basado en la calidad de imagen y complejidad del caso
    
    Args:
        image_quality: Puntuación de calidad de imagen (0.0 - 1.0)
        case_complexity: Complejidad del caso ("low", "medium", "high", "forensic")
    
    Returns:
        Nivel recomendado
    """
    if case_complexity == "forensic":
        return "forensic"
    elif case_complexity == "high":
        return "forensic"
    elif image_quality < 0.5:
        return "basic"
    elif image_quality < 0.7:
        return "standard"
    else:
        return "advanced"


# Configuraciones predefinidas para casos específicos
PREDEFINED_CONFIGS = {
    "quick_screening": {
        "level": "basic",
        "quality_assessment": {"enabled": False},
        "roi_detection": {"enabled": False},
        "cmc_analysis": {"enabled": False}
    },
    "standard_comparison": {
        "level": "standard",
        "quality_assessment": {"enabled": True},
        "roi_detection": {"enabled": True},
        "cmc_analysis": {"enabled": True}
    },
    "forensic_analysis": {
        "level": "forensic",
        "quality_assessment": {"enabled": True, "strict_mode": True},
        "roi_detection": {"enabled": True},
        "cmc_analysis": {"enabled": True},
        "export_intermediate_results": True,
        "export_detailed_report": True
    },
    "research_mode": {
        "level": "advanced",
        "quality_assessment": {"enabled": True},
        "roi_detection": {"enabled": True},
        "cmc_analysis": {"enabled": True},
        "export_intermediate_results": True,
        "export_visualizations": True,
        "enable_parallel_processing": True
    }
}


def get_predefined_config(config_name: str) -> PipelineConfiguration:
    """
    Obtiene una configuración predefinida
    
    Args:
        config_name: Nombre de la configuración predefinida
    
    Returns:
        PipelineConfiguration configurada
    """
    if config_name not in PREDEFINED_CONFIGS:
        raise ValueError(f"Configuración '{config_name}' no encontrada")
    
    config_data = PREDEFINED_CONFIGS[config_name]
    return PipelineConfiguration.from_dict(config_data)


def list_predefined_configs() -> List[str]:
    """Lista las configuraciones predefinidas disponibles"""
    return list(PREDEFINED_CONFIGS.keys())