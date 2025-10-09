"""
Preprocesador Unificado de Imágenes Balísticas
Sistema Balístico Forense MVP

Módulo consolidado para preprocesamiento automático de imágenes balísticas
que combina las funcionalidades de preprocessor.py y ballistic_preprocessor.py

Implementa técnicas avanzadas de preprocesamiento específicas para:
- Corrección de iluminación no uniforme
- Mejora de contraste adaptativo
- Filtrado de ruido especializado
- Normalización de orientación
- Mejora de características balísticas

Basado en literatura científica:
- NIST Special Publication 1500-9
- AFTE Theory and Practice of Firearm Identification
- Técnicas de procesamiento de imágenes forenses
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
import logging
from pathlib import Path
from enum import Enum
import json
import time
import hashlib
import os

# Importar nuevos módulos NIST
try:
    from spatial_calibration import SpatialCalibrator, CalibrationData
    from nist_compliance_validator import NISTComplianceValidator, NISTProcessingReport
    NIST_MODULES_AVAILABLE = True
except ImportError:
    NIST_MODULES_AVAILABLE = False
    # Crear clases dummy para mantener compatibilidad
    class SpatialCalibrator:
        def __init__(self, *args, **kwargs): pass
        def calibrate(self, *args, **kwargs): return None
    
    class CalibrationData:
        def __init__(self, *args, **kwargs): pass
    
    class NISTComplianceValidator:
        def __init__(self, *args, **kwargs): pass
        def validate(self, *args, **kwargs): return None
    
    class NISTProcessingReport:
        def __init__(self, *args, **kwargs): pass

# Importaciones opcionales con manejo de errores
try:
    from scipy import ndimage
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from skimage import filters, morphology, segmentation, measure
    from skimage.restoration import denoise_bilateral, denoise_tv_chambolle
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False

try:
    from utils.logger import LoggerMixin
    LOGGER_MIXIN_AVAILABLE = True
except ImportError:
    LOGGER_MIXIN_AVAILABLE = False
    # Crear clase dummy para mantener compatibilidad
    class LoggerMixin:
        def __init__(self, *args, **kwargs): 
            self.logger = logging.getLogger(self.__class__.__name__)
        
        def log_info(self, message): 
            self.logger.info(message)
        
        def log_error(self, message): 
            self.logger.error(message)
        
        def log_warning(self, message): 
            self.logger.warning(message)

# Importar detector de ROI
try:
    from image_processing.unified_roi_detector import UnifiedROIDetector, ROIDetectionConfig, DetectionLevel
    ROI_DETECTOR_AVAILABLE = True
except ImportError:
    ROI_DETECTOR_AVAILABLE = False

# Importar acelerador GPU
try:
    from image_processing.gpu_accelerator import get_gpu_accelerator, is_gpu_available, GPUAccelerator
    GPU_ACCELERATION_AVAILABLE = True
except ImportError:
    GPU_ACCELERATION_AVAILABLE = False
    # Crear clase dummy para mantener compatibilidad
    class GPUAccelerator:
        def __init__(self, *args, **kwargs): pass
        def is_gpu_enabled(self): return False
        def resize(self, *args, **kwargs): return cv2.resize(*args, **kwargs)
        def gaussian_blur(self, *args, **kwargs): return cv2.GaussianBlur(*args, **kwargs)
        def bilateral_filter(self, *args, **kwargs): return cv2.bilateralFilter(*args, **kwargs)
        def morphology_ex(self, *args, **kwargs): return cv2.morphologyEx(*args, **kwargs)
        def threshold(self, *args, **kwargs): return cv2.threshold(*args, **kwargs)
    
    def get_gpu_accelerator(*args, **kwargs):
        return GPUAccelerator()
    
    def is_gpu_available():
        return False

class PreprocessingLevel(Enum):
    """Niveles de preprocesamiento"""
    BASIC = "basic"
    STANDARD = "standard"
    ADVANCED = "advanced"
    FORENSIC = "forensic"

@dataclass
class PreprocessingConfig:
    """Configuración para el preprocesamiento de imágenes balísticas"""
    # Configuración general
    level: PreprocessingLevel = PreprocessingLevel.STANDARD
    target_size: Tuple[int, int] = (800, 600)
    resize_images: bool = True
    
    # Configuración de aceleración GPU
    enable_gpu_acceleration: bool = True
    gpu_device_id: int = 0
    gpu_fallback_to_cpu: bool = True
    
    # Corrección de iluminación
    illumination_correction: bool = True
    clahe_clip_limit: float = 2.0
    clahe_tile_size: Tuple[int, int] = (8, 8)
    
    # Reducción de ruido
    noise_reduction: bool = True
    bilateral_d: int = 9
    bilateral_sigma_color: float = 75
    bilateral_sigma_space: float = 75
    gaussian_kernel_size: Tuple[int, int] = (3, 3)
    gaussian_sigma: float = 0.8
    
    # Mejora de contraste
    contrast_enhancement: bool = True
    gamma_correction: float = 1.1
    histogram_equalization: bool = True
    
    # Mejora de bordes y operaciones morfológicas
    edge_enhancement: bool = True
    morphological_operations: bool = True
    
    # Normalización
    normalize_orientation: bool = True
    normalize_scale: bool = True
    
    # Mejoras específicas para evidencia balística
    enhance_striations: bool = False
    enhance_breech_marks: bool = False
    enhance_firing_pin: bool = False
    
    # Configuración de detección automática de ROI
    enable_roi_detection: bool = False
    roi_detection_level: str = "standard"  # basic, standard, advanced
    roi_use_enhanced_watershed: bool = True
    roi_watershed_method: str = "ballistic_optimized"  # classic, distance, marker_controlled, hybrid, ballistic_optimized
    
    # Configuración de visualización
    enable_visualization: bool = False
    save_intermediate_steps: bool = False
    visualization_output_dir: Optional[str] = None

@dataclass
class PreprocessingResult:
    """Resultado del preprocesamiento con métricas detalladas"""
    original_image: np.ndarray
    processed_image: np.ndarray
    metadata: Dict[str, Any]
    success: bool
    error_message: str = ""
    
    # Métricas de calidad
    processing_time: float = 0.0
    contrast_improvement: float = 0.0
    noise_reduction_score: float = 0.0
    sharpness_score: float = 0.0
    steps_applied: List[str] = field(default_factory=list)
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Imágenes intermedias y visualización
    intermediate_images: Dict[str, np.ndarray] = field(default_factory=dict)
    visualization_paths: Dict[str, str] = field(default_factory=dict)
    
    # Información de ROI
    roi_regions: List[Dict[str, Any]] = field(default_factory=list)
    roi_detection_success: bool = False
    roi_detection_time: float = 0.0
    
    # Información de calibración NIST
    calibration_data: Optional[CalibrationData] = None
    nist_compliance_report: Optional[NISTProcessingReport] = None
    illumination_uniformity: float = 0.0
    nist_compliant: bool = False
    
    @property
    def shape(self) -> Tuple[int, int]:
        """Retorna las dimensiones de la imagen procesada"""
        if self.processed_image is not None:
            return self.processed_image.shape[:2]
        elif self.original_image is not None:
            return self.original_image.shape[:2]
        else:
            return (0, 0)


class UnifiedPreprocessor(LoggerMixin):
    """
    Preprocesador unificado para imágenes balísticas
    
    Combina técnicas de preprocesamiento específicas para análisis forense
    con capacidades de visualización de pasos intermedios.
    """
    
    def __init__(self, config: Optional[Union[Dict, PreprocessingConfig]] = None):
        """
        Inicializar el preprocesador
        
        Args:
            config: Configuración de preprocesamiento
        """
        super().__init__()
        
        # Verificar dependencias
        self._check_dependencies()
        
        # Configuración por defecto
        if config is None:
            self.config = PreprocessingConfig()
        elif isinstance(config, dict):
            # Convertir diccionario a PreprocessingConfig
            self.config = PreprocessingConfig(**config)
        else:
            # Verificar si es ImageProcessingConfig y convertir a PreprocessingConfig
            if hasattr(config, '__class__') and config.__class__.__name__ == 'ImageProcessingConfig':
                # Convertir ImageProcessingConfig a PreprocessingConfig usando campos compatibles
                self.config = PreprocessingConfig(
                    resize_images=getattr(config, 'resize_images', True),
                    illumination_correction=getattr(config, 'enable_illumination_correction', True),
                    clahe_clip_limit=getattr(config, 'clahe_clip_limit', 2.0),
                    enhance_striations=getattr(config, 'enhance_striations', False),
                    enhance_breech_marks=getattr(config, 'enhance_breech_marks', False),
                    enhance_firing_pin=getattr(config, 'enhance_firing_pin', False),
                    enable_gpu_acceleration=False  # ImageProcessingConfig no tiene este campo, usar False por defecto
                )
            else:
                self.config = config
        
        # Inicializar acelerador GPU
        self.gpu_accelerator = None
        if self.config.enable_gpu_acceleration and GPU_ACCELERATION_AVAILABLE:
            try:
                self.gpu_accelerator = get_gpu_accelerator(
                    enable_gpu=True, 
                    device_id=self.config.gpu_device_id
                )
                if self.gpu_accelerator.is_gpu_enabled():
                    self.logger.info(f"GPU acceleration habilitada: {self.gpu_accelerator.get_gpu_info().gpu_name}")
                else:
                    self.logger.warning("GPU no disponible, usando CPU")
                    if not self.config.gpu_fallback_to_cpu:
                        raise RuntimeError("GPU requerida pero no disponible")
            except Exception as e:
                self.logger.error(f"Error inicializando GPU: {e}")
                if not self.config.gpu_fallback_to_cpu:
                    raise
                self.gpu_accelerator = None
        
        # Inicializar componentes NIST
        self.spatial_calibrator = SpatialCalibrator()
        self.nist_validator = NISTComplianceValidator()
        
        # Configuraciones predefinidas por nivel
        self.default_configs = {
            PreprocessingLevel.BASIC: PreprocessingConfig(
                level=PreprocessingLevel.BASIC,
                illumination_correction=True,
                noise_reduction=True,
                contrast_enhancement=False,
                edge_enhancement=False,
                morphological_operations=False,
                normalize_orientation=False,
                enhance_striations=False,
                enhance_breech_marks=False,
                enhance_firing_pin=False,
                enable_roi_detection=False
            ),
            PreprocessingLevel.STANDARD: PreprocessingConfig(
                level=PreprocessingLevel.STANDARD,
                illumination_correction=True,
                noise_reduction=True,
                contrast_enhancement=True,
                edge_enhancement=True,
                morphological_operations=True,
                normalize_orientation=True,
                enhance_striations=False,
                enhance_breech_marks=False,
                enhance_firing_pin=False,
                enable_roi_detection=True,
                roi_detection_level="standard"
            ),
            PreprocessingLevel.ADVANCED: PreprocessingConfig(
                level=PreprocessingLevel.ADVANCED,
                illumination_correction=True,
                noise_reduction=True,
                contrast_enhancement=True,
                edge_enhancement=True,
                morphological_operations=True,
                normalize_orientation=True,
                normalize_scale=True,
                enhance_striations=True,
                enhance_breech_marks=False,
                enhance_firing_pin=False,
                enable_roi_detection=True,
                roi_detection_level="advanced",
                roi_use_enhanced_watershed=True
            ),
            PreprocessingLevel.FORENSIC: PreprocessingConfig(
                level=PreprocessingLevel.FORENSIC,
                illumination_correction=True,
                noise_reduction=True,
                contrast_enhancement=True,
                edge_enhancement=True,
                morphological_operations=True,
                normalize_orientation=True,
                normalize_scale=True,
                enhance_striations=True,
                enhance_breech_marks=True,
                enhance_firing_pin=True,
                enable_roi_detection=True,
                roi_detection_level="advanced",
                roi_use_enhanced_watershed=True,
                roi_watershed_method="ballistic_optimized"
            )
        }
        
        # Inicializar detector de ROI si está disponible y habilitado
        self.roi_detector = None
        if ROI_DETECTOR_AVAILABLE and self.config.enable_roi_detection:
            try:
                # Configurar detector de ROI
                roi_config = ROIDetectionConfig(
                    detection_level=DetectionLevel(self.config.roi_detection_level),
                    use_enhanced_watershed=self.config.roi_use_enhanced_watershed,
                    watershed_method=self.config.roi_watershed_method
                )
                self.roi_detector = UnifiedROIDetector(roi_config)
                self.logger.info("Detector de ROI inicializado correctamente")
            except Exception as e:
                self.logger.warning(f"No se pudo inicializar el detector de ROI: {e}")
                self.roi_detector = None
        
        # Inicializar visualizador si está habilitado
        self.visualizer = None
        if self.config.enable_visualization:
            try:
                from .preprocessing_visualizer import PreprocessingVisualizer
                self.visualizer = PreprocessingVisualizer(self)
            except ImportError:
                self.logger.warning("PreprocessingVisualizer no disponible")
        
        # Estadísticas de procesamiento
        self.processing_stats = {
            'total_processed': 0,
            'nist_compliant_count': 0,
            'average_processing_time': 0.0,
            'calibration_success_rate': 0.0
        }
    
    def _check_dependencies(self):
        """Verificar dependencias opcionales"""
        if not SCIPY_AVAILABLE:
            self.logger.warning("SciPy no disponible - algunas funciones pueden estar limitadas")
        
        if not SKIMAGE_AVAILABLE:
            self.logger.warning("scikit-image no disponible - algunas funciones pueden estar limitadas")
    
    def load_image(self, image_path: str) -> Optional[np.ndarray]:
        """
        Cargar imagen desde archivo
        
        Args:
            image_path: Ruta al archivo de imagen
            
        Returns:
            np.ndarray: Imagen cargada o None si hay error
        """
        try:
            if not os.path.exists(image_path):
                self.logger.error(f"Archivo de imagen no encontrado: {image_path}")
                return None
            
            # Cargar imagen usando OpenCV
            image = cv2.imread(image_path)
            
            if image is None:
                self.logger.error(f"No se pudo cargar la imagen: {image_path}")
                return None
            
            self.logger.info(f"Imagen cargada exitosamente: {image_path} - Shape: {image.shape}")
            return image
            
        except Exception as e:
            self.logger.error(f"Error cargando imagen {image_path}: {str(e)}")
            return None
    
    def convert_to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """
        Convertir imagen a escala de grises
        
        Args:
            image: Imagen de entrada (BGR o ya en escala de grises)
            
        Returns:
            Imagen en escala de grises
        """
        try:
            if len(image.shape) == 3:
                # Imagen en color, convertir a escala de grises
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                self.logger.info("Imagen convertida a escala de grises")
                return gray
            elif len(image.shape) == 2:
                # Ya está en escala de grises
                self.logger.info("Imagen ya está en escala de grises")
                return image
            else:
                raise ValueError(f"Formato de imagen no soportado: {image.shape}")
                
        except Exception as e:
            self.logger.error(f"Error al convertir imagen a escala de grises: {str(e)}")
            # Retornar imagen original en caso de error
            return image
    
    def _save_intermediate_step(self, image: np.ndarray, step_name: str, 
                               intermediate_images: Dict[str, np.ndarray],
                               visualization_paths: Dict[str, str]) -> None:
        """
        Guardar paso intermedio para visualización
        
        Args:
            image: Imagen del paso intermedio
            step_name: Nombre del paso
            intermediate_images: Diccionario para almacenar imágenes intermedias
            visualization_paths: Diccionario para almacenar rutas de visualización
        """
        if self.config.save_intermediate_steps:
            intermediate_images[step_name] = image.copy()
            
            if self.config.visualization_output_dir:
                output_path = Path(self.config.visualization_output_dir)
                output_path.mkdir(parents=True, exist_ok=True)
                
                filename = f"{step_name}.png"
                filepath = output_path / filename
                cv2.imwrite(str(filepath), image)
                visualization_paths[step_name] = str(filepath)
    
    def preprocess_with_visualization(self, image_path: str, 
                                    evidence_type: str = "unknown",
                                    level: Optional[str] = None) -> PreprocessingResult:
        """
        Preprocesar imagen con visualización habilitada temporalmente
        
        Args:
            image_path: Ruta de la imagen
            evidence_type: Tipo de evidencia
            level: Nivel de preprocesamiento
            
        Returns:
            PreprocessingResult con visualizaciones
        """
        # Guardar configuración original
        original_visualization = self.config.enable_visualization
        original_save_steps = self.config.save_intermediate_steps
        
        try:
            # Habilitar visualización temporalmente
            self.config.enable_visualization = True
            self.config.save_intermediate_steps = True
            
            # Procesar imagen
            return self.preprocess_image(image_path, evidence_type, level)
        
        finally:
            # Restaurar configuración original
            self.config.enable_visualization = original_visualization
            self.config.save_intermediate_steps = original_save_steps
    
    def preprocess_image(self, image_path: str, 
                        evidence_type: str = "unknown",
                        level: Optional[str] = None) -> PreprocessingResult:
        """
        Preprocesar imagen balística
        
        Args:
            image_path: Ruta de la imagen
            evidence_type: Tipo de evidencia (bullet, cartridge, etc.)
            level: Nivel de preprocesamiento
            
        Returns:
            PreprocessingResult con imagen procesada y metadatos
        """
        start_time = time.time()
        
        try:
            # Cargar imagen
            if isinstance(image_path, str):
                if not os.path.exists(image_path):
                    return PreprocessingResult(
                        original_image=np.array([]),
                        processed_image=np.array([]),
                        metadata={},
                        success=False,
                        error_message=f"Archivo no encontrado: {image_path}"
                    )
                
                image = cv2.imread(image_path)
                if image is None:
                    return PreprocessingResult(
                        original_image=np.array([]),
                        processed_image=np.array([]),
                        metadata={},
                        success=False,
                        error_message=f"No se pudo cargar la imagen: {image_path}"
                    )
            else:
                image = image_path.copy()
            
            original_image = image.copy()
            
            # Configurar nivel si se especifica
            if level:
                try:
                    preprocessing_level = PreprocessingLevel(level)
                    if preprocessing_level in self.default_configs:
                        self.config = self.default_configs[preprocessing_level]
                except ValueError:
                    self.logger.warning(f"Nivel de preprocesamiento no válido: {level}")
            
            # Inicializar contenedores para visualización
            intermediate_images = {}
            visualization_paths = {}
            steps_applied = []
            
            # Guardar imagen original
            self._save_intermediate_step(original_image, "original", intermediate_images, visualization_paths)
            
            # Aplicar preprocesamiento con calibración espacial
            calibration_data = None
            nist_compliance_report = None
            
            processed_image, calibration_data, nist_compliance_report = self.preprocess_ballistic_image(
                image, evidence_type, intermediate_images, visualization_paths, steps_applied,
                image_path=str(image_path) if isinstance(image_path, str) else None,
                enable_nist_validation=True,
                calibration_method='auto'
            )
            
            # Detección automática de ROI si está habilitada
            roi_regions = []
            roi_detection_success = False
            roi_detection_time = 0.0
            
            if self.config.enable_roi_detection and self.roi_detector is not None:
                roi_start_time = time.time()
                try:
                    self.logger.info("Iniciando detección automática de ROI...")
                    
                    # Determinar tipo de evidencia para el detector
                    evidence_type_mapping = {
                        "vaina": "cartridge",
                        "cartridge": "cartridge", 
                        "proyectil": "bullet",
                        "bullet": "bullet",
                        "unknown": "cartridge"  # Por defecto
                    }
                    
                    detector_evidence_type = evidence_type_mapping.get(evidence_type.lower(), "cartridge")
                    
                    # Detectar ROI usando la imagen procesada
                    roi_result = self.roi_detector.detect_roi(processed_image, detector_evidence_type)
                    
                    if roi_result.success and roi_result.regions:
                        roi_detection_success = True
                        
                        # Convertir regiones a formato serializable
                        for region in roi_result.regions:
                            roi_data = {
                                "x": int(region.x),
                                "y": int(region.y), 
                                "width": int(region.width),
                                "height": int(region.height),
                                "confidence": float(region.confidence),
                                "region_type": region.region_type,
                                "area": int(region.area),
                                "center": (int(region.center[0]), int(region.center[1]))
                            }
                            roi_regions.append(roi_data)
                        
                        steps_applied.append("roi_detection")
                        self.logger.info(f"Detectadas {len(roi_regions)} regiones de interés")
                        
                        # Guardar imagen con ROI marcadas para visualización
                        if self.config.save_intermediate_steps:
                            roi_image = processed_image.copy()
                            for region in roi_result.regions:
                                cv2.rectangle(roi_image, 
                                            (region.x, region.y), 
                                            (region.x + region.width, region.y + region.height),
                                            (0, 255, 0), 2)
                                cv2.putText(roi_image, f"{region.region_type} ({region.confidence:.2f})",
                                          (region.x, region.y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                            
                            self._save_intermediate_step(roi_image, "roi_detected", intermediate_images, visualization_paths)
                    else:
                        self.logger.warning("No se detectaron regiones de interés válidas")
                        
                except Exception as e:
                    self.logger.error(f"Error en detección de ROI: {e}")
                    roi_detection_success = False
                
                roi_detection_time = time.time() - roi_start_time
            
            # Calcular métricas de calidad
            processing_time = time.time() - start_time
            quality_metrics = self._calculate_quality_metrics(original_image, processed_image)
            
            # Crear resultado
            result = PreprocessingResult(
                original_image=original_image,
                processed_image=processed_image,
                metadata={
                    "evidence_type": evidence_type,
                    "preprocessing_level": self.config.level.value,
                    "image_path": str(image_path) if isinstance(image_path, str) else "array",
                    "original_shape": original_image.shape,
                    "processed_shape": processed_image.shape,
                    "timestamp": time.time(),
                    "roi_detection_enabled": self.config.enable_roi_detection,
                    "roi_regions_count": len(roi_regions)
                },
                success=True,
                processing_time=processing_time,
                steps_applied=steps_applied,
                quality_metrics=quality_metrics,
                intermediate_images=intermediate_images,
                visualization_paths=visualization_paths,
                roi_regions=roi_regions,
                roi_detection_success=roi_detection_success,
                roi_detection_time=roi_detection_time,
                calibration_data=calibration_data,
                nist_compliance_report=nist_compliance_report,
                nist_compliant=nist_compliance_report.nist_compliant if nist_compliance_report else False
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error en preprocesamiento: {str(e)}")
            return PreprocessingResult(
                original_image=np.array([]),
                processed_image=np.array([]),
                metadata={},
                success=False,
                error_message=str(e),
                processing_time=time.time() - start_time
            )
    
    def preprocess_ballistic_image(self, 
                                 image: np.ndarray,
                                 evidence_type: str = "unknown",
                                 intermediate_images: Optional[Dict[str, np.ndarray]] = None,
                                 visualization_paths: Optional[Dict[str, str]] = None,
                                 steps_applied: Optional[List[str]] = None,
                                 image_path: Optional[str] = None,
                                 enable_nist_validation: bool = True,
                                 calibration_method: str = 'auto',
                                 reference_object: Optional[str] = None) -> Tuple[np.ndarray, Optional[CalibrationData], Optional[NISTProcessingReport]]:
        """
        Pipeline principal de preprocesamiento balístico con validación NIST
        
        Args:
            image: Imagen a procesar
            evidence_type: Tipo de evidencia
            intermediate_images: Diccionario para imágenes intermedias
            visualization_paths: Diccionario para rutas de visualización
            steps_applied: Lista de pasos aplicados
            image_path: Ruta de la imagen (para calibración DPI)
            enable_nist_validation: Habilitar validación NIST completa
            calibration_method: Método de calibración ('auto', 'metadata', 'reference', 'manual')
            reference_object: Tipo de objeto de referencia para calibración
            
        Returns:
            Imagen procesada
        """
        processed = image.copy()
        
        # Inicializar contenedores si no se proporcionan
        if intermediate_images is None:
            intermediate_images = {}
        if visualization_paths is None:
            visualization_paths = {}
        if steps_applied is None:
            steps_applied = []
        
        # 1. Corrección de iluminación avanzada (NIST)
        if self.config.illumination_correction:
            processed = self._correct_illumination(processed)
            steps_applied.append("advanced_illumination_correction")
            self._save_intermediate_step(processed, "illumination_corrected", intermediate_images, visualization_paths)
        
        # 2. Reducción de ruido
        if self.config.noise_reduction:
            processed = self._reduce_noise(processed)
            steps_applied.append("noise_reduction")
            self._save_intermediate_step(processed, "noise_reduced", intermediate_images, visualization_paths)
        
        # 3. Mejora de contraste
        if self.config.contrast_enhancement:
            processed = self._enhance_contrast(processed)
            steps_applied.append("contrast_enhancement")
            self._save_intermediate_step(processed, "contrast_enhanced", intermediate_images, visualization_paths)
        
        # 4. Corrección de rotación
        if self.config.normalize_orientation:
            processed = self._correct_rotation(processed)
            steps_applied.append("rotation_correction")
            self._save_intermediate_step(processed, "rotation_corrected", intermediate_images, visualization_paths)
        
        # 5. Mejora de bordes
        if self.config.edge_enhancement:
            processed = self._enhance_edges(processed)
            steps_applied.append("edge_enhancement")
            self._save_intermediate_step(processed, "edges_enhanced", intermediate_images, visualization_paths)
        
        # 6. Operaciones morfológicas
        if self.config.morphological_operations:
            processed = self._apply_morphological_operations(processed)
            steps_applied.append("morphological_operations")
            self._save_intermediate_step(processed, "morphological_processed", intermediate_images, visualization_paths)
        
        # 7. Redimensionamiento
        if self.config.resize_images:
            processed = self._resize_image(processed)
            steps_applied.append("resize")
            self._save_intermediate_step(processed, "resized", intermediate_images, visualization_paths)
        
        # 8. Mejoras específicas por tipo de evidencia
        if evidence_type in ["bullet", "projectile"] and self.config.enhance_striations:
            processed = self._enhance_striations(processed)
            steps_applied.append("striation_enhancement")
            self._save_intermediate_step(processed, "striations_enhanced", intermediate_images, visualization_paths)
        
        if evidence_type in ["cartridge", "case"] and self.config.enhance_breech_marks:
            processed = self._enhance_breech_marks(processed)
            steps_applied.append("breech_mark_enhancement")
            self._save_intermediate_step(processed, "breech_marks_enhanced", intermediate_images, visualization_paths)
        
        if evidence_type in ["cartridge", "case"] and self.config.enhance_firing_pin:
            processed = self._enhance_firing_pin_marks(processed)
            steps_applied.append("firing_pin_enhancement")
            self._save_intermediate_step(processed, "firing_pin_enhanced", intermediate_images, visualization_paths)
        
        # 9. Calibración espacial DPI (si está habilitada)
        calibration_data = None
        nist_compliance_report = None
        
        if enable_nist_validation and NIST_MODULES_AVAILABLE and image_path:
            try:
                # Inicializar calibrador espacial
                spatial_calibrator = SpatialCalibrator()
                
                # Intentar calibración según método especificado
                if calibration_method == 'auto' or calibration_method == 'metadata':
                    calibration_data = spatial_calibrator.calibrate_from_metadata(image_path)
                    
                    # Si no hay metadatos y es auto, intentar con objeto de referencia
                    if calibration_data is None and calibration_method == 'auto' and reference_object:
                        calibration_data = spatial_calibrator.calibrate_with_reference_object(
                            processed, reference_object
                        )
                
                elif calibration_method == 'reference' and reference_object:
                    calibration_data = spatial_calibrator.calibrate_with_reference_object(
                        processed, reference_object
                    )
                
                # Si se obtuvo calibración, validar cumplimiento NIST
                if calibration_data:
                    nist_result = spatial_calibrator.validate_nist_compliance(calibration_data)
                    steps_applied.append(f"spatial_calibration_{calibration_method}")
                    
                    # Crear reporte de cumplimiento NIST
                    validator = NISTComplianceValidator()
                    nist_compliance_report = validator.validate_image_processing(
                        image_path, processed, calibration_method, reference_object
                    )
                    
                    self.log_info(f"Calibración DPI: {calibration_data.dpi:.1f} DPI, "
                                f"Cumplimiento NIST: {nist_result.nist_compliant}")
                else:
                    self.log_warning("No se pudo realizar calibración espacial DPI")
                    
            except Exception as e:
                self.log_error(f"Error en calibración espacial: {e}")
        
        # Guardar resultado final
        self._save_intermediate_step(processed, "final_result", intermediate_images, visualization_paths)
        
        return processed, calibration_data, nist_compliance_report
    
    def correct_illumination(self, image: np.ndarray) -> np.ndarray:
        """
        Método público para corrección de iluminación
        
        Args:
            image: Imagen a corregir
            
        Returns:
            Imagen con iluminación corregida
        """
        return self._correct_illumination(image)
    
    def _correct_illumination(self, image: np.ndarray) -> np.ndarray:
        """
        Corrección de iluminación completa según estándares NIST
        
        Implementa múltiples técnicas para lograr uniformidad de iluminación < 10%:
        1. Estimación y corrección del fondo de iluminación
        2. CLAHE adaptativo
        3. Corrección de gradientes de iluminación
        4. Normalización global
        
        Args:
            image: Imagen a corregir
            
        Returns:
            Imagen con iluminación corregida
        """
        try:
            # Convertir a escala de grises para análisis
            if len(image.shape) == 3:
                working_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                is_color = True
            else:
                working_image = image.copy()
                is_color = False
            
            # 1. Estimación del fondo de iluminación usando morfología
            # Crear kernel grande para capturar variaciones de iluminación
            kernel_size = max(working_image.shape) // 20
            if kernel_size % 2 == 0:
                kernel_size += 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            
            # Estimación del fondo usando apertura morfológica
            background = cv2.morphologyEx(working_image, cv2.MORPH_OPEN, kernel)
            
            # Suavizar el fondo estimado
            background = cv2.GaussianBlur(background, (kernel_size//4*2+1, kernel_size//4*2+1), 0)
            
            # 2. Corrección del fondo
            # Evitar división por cero
            background_safe = np.where(background < 10, 10, background)
            
            # Normalizar usando el fondo estimado
            corrected = (working_image.astype(np.float32) / background_safe.astype(np.float32)) * 128
            corrected = np.clip(corrected, 0, 255).astype(np.uint8)
            
            # 3. Corrección de gradientes de iluminación
            # Calcular gradientes de iluminación
            grad_x = cv2.Sobel(corrected, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(corrected, cv2.CV_64F, 0, 1, ksize=3)
            
            # Suavizar gradientes para obtener tendencia global
            grad_x_smooth = cv2.GaussianBlur(grad_x, (kernel_size//2*2+1, kernel_size//2*2+1), 0)
            grad_y_smooth = cv2.GaussianBlur(grad_y, (kernel_size//2*2+1, kernel_size//2*2+1), 0)
            
            # Crear mapa de corrección basado en gradientes
            h, w = corrected.shape
            y_coords, x_coords = np.mgrid[0:h, 0:w]
            
            # Corrección suave basada en gradientes
            correction_map = np.ones_like(corrected, dtype=np.float32)
            if np.std(grad_x_smooth) > 0.1:
                correction_map *= (1.0 - grad_x_smooth / (np.max(np.abs(grad_x_smooth)) + 1e-6) * 0.1)
            if np.std(grad_y_smooth) > 0.1:
                correction_map *= (1.0 - grad_y_smooth / (np.max(np.abs(grad_y_smooth)) + 1e-6) * 0.1)
            
            corrected = (corrected.astype(np.float32) * correction_map).astype(np.uint8)
            
            # 4. CLAHE adaptativo mejorado
            # Usar CLAHE con parámetros optimizados para uniformidad
            clahe = cv2.createCLAHE(
                clipLimit=self.config.clahe_clip_limit * 0.8,  # Reducir para evitar sobre-corrección
                tileGridSize=(16, 16)  # Tiles más pequeños para mejor uniformidad local
            )
            corrected = clahe.apply(corrected)
            
            # 5. Normalización global final
            # Ajustar el rango dinámico manteniendo la uniformidad
            mean_intensity = np.mean(corrected)
            std_intensity = np.std(corrected)
            
            # Normalizar para centrar en 128 con desviación controlada
            if std_intensity > 0:
                target_std = min(std_intensity, 40)  # Limitar variación para uniformidad
                corrected = ((corrected - mean_intensity) / std_intensity * target_std + 128)
                corrected = np.clip(corrected, 0, 255).astype(np.uint8)
            
            # 6. Aplicar corrección a imagen original si es color
            if is_color:
                # Convertir imagen original a LAB
                lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
                l_channel, a_channel, b_channel = cv2.split(lab)
                
                # Aplicar la corrección calculada al canal L
                # Calcular factor de corrección
                l_original = l_channel.astype(np.float32)
                l_corrected = corrected.astype(np.float32)
                
                # Evitar división por cero
                l_safe = np.where(l_original < 1, 1, l_original)
                correction_factor = l_corrected / l_safe
                
                # Aplicar corrección suave
                l_final = (l_original * correction_factor * 0.7 + l_corrected * 0.3)
                l_final = np.clip(l_final, 0, 255).astype(np.uint8)
                
                # Recombinar canales
                lab_corrected = cv2.merge([l_final, a_channel, b_channel])
                result = cv2.cvtColor(lab_corrected, cv2.COLOR_LAB2BGR)
            else:
                result = corrected
            
            # 7. Validación de uniformidad NIST
            uniformity = self._calculate_illumination_uniformity(result)
            if uniformity < 0.9:  # NIST requiere uniformidad > 90%
                self.logger.warning(f"Uniformidad de iluminación por debajo del estándar NIST: {uniformity:.3f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error en corrección de iluminación avanzada: {e}")
            # Fallback a CLAHE básico
            return self._correct_illumination_basic(image)
    
    def _correct_illumination_basic(self, image: np.ndarray) -> np.ndarray:
        """Método básico de corrección de iluminación (fallback)"""
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(
                clipLimit=self.config.clahe_clip_limit,
                tileGridSize=self.config.clahe_tile_size
            )
            l = clahe.apply(l)
            lab = cv2.merge([l, a, b])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            clahe = cv2.createCLAHE(
                clipLimit=self.config.clahe_clip_limit,
                tileGridSize=self.config.clahe_tile_size
            )
            return clahe.apply(image)
    
    def _calculate_illumination_uniformity(self, image: np.ndarray) -> float:
        """
        Calcula la uniformidad de iluminación según estándares NIST
        
        Args:
            image: Imagen a evaluar
            
        Returns:
            float: Valor de uniformidad (0-1), donde 1 es perfectamente uniforme
        """
        try:
            # Convertir a escala de grises si es necesario
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Dividir imagen en regiones para análisis de uniformidad
            h, w = gray.shape
            region_size = min(64, h // 8, w // 8)
            
            if region_size < 16:
                return 0.5  # Imagen muy pequeña
            
            region_means = []
            
            # Calcular media de cada región
            for i in range(0, h - region_size, region_size // 2):
                for j in range(0, w - region_size, region_size // 2):
                    region = gray[i:i+region_size, j:j+region_size]
                    if region.size > 0:
                        region_means.append(np.mean(region))
            
            if len(region_means) < 4:
                return 0.5
            
            # Calcular uniformidad como inverso del coeficiente de variación
            mean_of_means = np.mean(region_means)
            std_of_means = np.std(region_means)
            
            if mean_of_means > 0:
                cv_coefficient = std_of_means / mean_of_means
                # NIST requiere variación < 10%, convertir a score de uniformidad
                uniformity = max(0.0, 1.0 - cv_coefficient / 0.1)
            else:
                uniformity = 0.0
            
            return min(1.0, uniformity)
            
        except Exception as e:
            self.logger.error(f"Error calculando uniformidad: {e}")
            return 0.0
    
    def reduce_noise(self, image: np.ndarray) -> np.ndarray:
        """Método público para reducir ruido usando filtro bilateral con aceleración GPU"""
        return self._reduce_noise(image)
    
    def _reduce_noise(self, image: np.ndarray) -> np.ndarray:
        """Reducir ruido usando filtro bilateral con aceleración GPU"""
        if self.gpu_accelerator and self.gpu_accelerator.is_gpu_enabled():
            return self.gpu_accelerator.bilateral_filter(
                image,
                self.config.bilateral_d,
                self.config.bilateral_sigma_color,
                self.config.bilateral_sigma_space
            )
        else:
            return cv2.bilateralFilter(
                image,
                self.config.bilateral_d,
                self.config.bilateral_sigma_color,
                self.config.bilateral_sigma_space
            )
    
    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Mejorar contraste usando corrección gamma y ecualización"""
        # Corrección gamma
        gamma_corrected = np.power(image / 255.0, self.config.gamma_correction) * 255.0
        gamma_corrected = gamma_corrected.astype(np.uint8)
        
        if self.config.histogram_equalization:
            if len(image.shape) == 3:
                # Ecualización en espacio YUV
                yuv = cv2.cvtColor(gamma_corrected, cv2.COLOR_BGR2YUV)
                yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
                return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
            else:
                return cv2.equalizeHist(gamma_corrected)
        
        return gamma_corrected
    
    def _correct_rotation(self, image: np.ndarray) -> np.ndarray:
        """Corregir rotación automáticamente"""
        # Implementación básica - puede mejorarse con detección de características
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Detectar líneas usando transformada de Hough
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        
        if lines is not None and len(lines) > 0:
            # Calcular ángulo promedio
            angles = []
            for line in lines[:10]:  # Usar solo las primeras 10 líneas
                rho, theta = line[0]  # HoughLines devuelve [[rho, theta]] por línea
                angle = theta * 180 / np.pi
                if angle > 90:
                    angle -= 180
                angles.append(angle)
            
            if angles:
                avg_angle = np.mean(angles)
                if abs(avg_angle) > 1:  # Solo rotar si el ángulo es significativo
                    center = (image.shape[1] // 2, image.shape[0] // 2)
                    rotation_matrix = cv2.getRotationMatrix2D(center, avg_angle, 1.0)
                    return cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))
        
        return image
    
    def _enhance_edges(self, image: np.ndarray) -> np.ndarray:
        """Mejorar bordes usando filtros de realce"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Filtro de realce de bordes
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        
        enhanced = cv2.filter2D(gray, -1, kernel)
        
        if len(image.shape) == 3:
            return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        return enhanced
    
    def _apply_morphological_operations(self, image: np.ndarray) -> np.ndarray:
        """Aplicar operaciones morfológicas para limpiar la imagen con aceleración GPU"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Elemento estructurante
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        
        # Operación de apertura (erosión seguida de dilatación)
        if self.gpu_accelerator and self.gpu_accelerator.is_gpu_enabled():
            opened = self.gpu_accelerator.morphology_ex(gray, cv2.MORPH_OPEN, kernel)
            # Operación de cierre (dilatación seguida de erosión)
            closed = self.gpu_accelerator.morphology_ex(opened, cv2.MORPH_CLOSE, kernel)
        else:
            opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
            # Operación de cierre (dilatación seguida de erosión)
            closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
        
        if len(image.shape) == 3:
            return cv2.cvtColor(closed, cv2.COLOR_GRAY2BGR)
        return closed
    
    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        """Redimensionar imagen al tamaño objetivo con aceleración GPU"""
        if self.gpu_accelerator and self.gpu_accelerator.is_gpu_enabled():
            return self.gpu_accelerator.resize(image, self.config.target_size, cv2.INTER_LANCZOS4)
        else:
            return cv2.resize(image, self.config.target_size, interpolation=cv2.INTER_LANCZOS4)
    
    def _enhance_striations(self, image: np.ndarray) -> np.ndarray:
        """Mejorar estrías en proyectiles"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Filtro direccional para realzar líneas verticales (estrías)
        kernel = np.array([[-1, 2, -1],
                          [-1, 2, -1],
                          [-1, 2, -1]]) / 3.0
        
        enhanced = cv2.filter2D(gray, -1, kernel)
        
        if len(image.shape) == 3:
            return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        return enhanced
    
    def _enhance_breech_marks(self, image: np.ndarray) -> np.ndarray:
        """Mejorar marcas de recámara en casquillos"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Filtro para realzar patrones circulares
        enhanced = cv2.GaussianBlur(gray, (3, 3), 0)
        enhanced = cv2.addWeighted(gray, 1.5, enhanced, -0.5, 0)
        
        if len(image.shape) == 3:
            return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        return enhanced
    
    def _enhance_firing_pin_marks(self, image: np.ndarray) -> np.ndarray:
        """Mejorar marcas de percutor"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Realzar características centrales pequeñas
        kernel = np.array([[0, -1, 0],
                          [-1, 5, -1],
                          [0, -1, 0]])
        
        enhanced = cv2.filter2D(gray, -1, kernel)
        
        if len(image.shape) == 3:
            return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        return enhanced
    
    def _calculate_quality_metrics(self, original: np.ndarray, processed: np.ndarray) -> Dict[str, float]:
        """Calcular métricas de calidad del preprocesamiento"""
        metrics = {}
        
        try:
            # Convertir a escala de grises si es necesario
            if len(original.shape) == 3:
                orig_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
                proc_gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
            else:
                orig_gray = original
                proc_gray = processed
            
            # Mejora de contraste (desviación estándar)
            orig_std = np.std(orig_gray)
            proc_std = np.std(proc_gray)
            metrics['contrast_improvement'] = (proc_std - orig_std) / orig_std if orig_std > 0 else 0
            
            # Nitidez (varianza del Laplaciano)
            laplacian = cv2.Laplacian(proc_gray, cv2.CV_64F)
            metrics['sharpness_score'] = laplacian.var()
            
            # Relación señal-ruido estimada
            mean_intensity = np.mean(proc_gray)
            noise_estimate = np.std(proc_gray - cv2.GaussianBlur(proc_gray, (5, 5), 0))
            metrics['snr_estimate'] = mean_intensity / noise_estimate if noise_estimate > 0 else 0
            
        except Exception as e:
            self.logger.warning(f"Error calculando métricas de calidad: {e}")
        
        return metrics
    
    def get_config_for_level(self, level: PreprocessingLevel) -> PreprocessingConfig:
        """Obtener configuración predefinida para un nivel específico"""
        return self.default_configs.get(level, self.config)
    
    def update_config(self, **kwargs):
        """Actualizar configuración actual"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                self.logger.warning(f"Parámetro de configuración desconocido: {key}")
    
    def _update_processing_stats(self, result: PreprocessingResult):
        """Actualizar estadísticas de procesamiento con métricas NIST"""
        try:
            self.processing_stats['total_processed'] += 1
            
            # Actualizar tiempo promedio
            current_avg = self.processing_stats['average_processing_time']
            total = self.processing_stats['total_processed']
            new_avg = ((current_avg * (total - 1)) + result.processing_time) / total
            self.processing_stats['average_processing_time'] = new_avg
            
            # Actualizar conteo de cumplimiento NIST
            if result.nist_compliant:
                self.processing_stats['nist_compliant_count'] += 1
            
            # Actualizar tasa de éxito de calibración
            if result.calibration_data and result.calibration_data.confidence > 0.7:
                calibration_successes = getattr(self, '_calibration_successes', 0) + 1
                self._calibration_successes = calibration_successes
                self.processing_stats['calibration_success_rate'] = calibration_successes / total
            
        except Exception as e:
            self.logger.error(f"Error actualizando estadísticas: {e}")
    
    def get_nist_compliance_summary(self) -> Dict[str, Any]:
        """Obtener resumen de cumplimiento NIST"""
        total = self.processing_stats['total_processed']
        if total == 0:
            return {'message': 'No se han procesado imágenes'}
        
        compliant_rate = self.processing_stats['nist_compliant_count'] / total
        
        return {
            'total_processed': total,
            'nist_compliant_count': self.processing_stats['nist_compliant_count'],
            'compliance_rate': compliant_rate,
            'calibration_success_rate': self.processing_stats['calibration_success_rate'],
            'average_processing_time': self.processing_stats['average_processing_time'],
            'compliance_status': 'GOOD' if compliant_rate >= 0.8 else 'NEEDS_IMPROVEMENT'
        }
    
    def get_image_info(self, image_path: str) -> Dict[str, Any]:
        """
        Obtiene información detallada de una imagen sin procesarla completamente
        
        Args:
            image_path: Ruta de la imagen
            
        Returns:
            Diccionario con información de la imagen
        """
        try:
            # Verificar que el archivo existe
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Archivo no encontrado: {image_path}")
            
            # Obtener información básica del archivo
            file_stat = os.stat(image_path)
            
            # Cargar imagen para obtener dimensiones y formato
            from PIL import Image
            with Image.open(image_path) as img:
                width, height = img.size
                format_name = img.format or 'Unknown'
                mode = img.mode
                
                # Determinar número de canales
                channels = len(img.getbands()) if hasattr(img, 'getbands') else 3
                
                # Obtener información de color
                color_space = mode
                if mode == 'RGB':
                    color_space = 'RGB'
                elif mode == 'RGBA':
                    color_space = 'RGBA'
                elif mode == 'L':
                    color_space = 'Grayscale'
                elif mode == 'CMYK':
                    color_space = 'CMYK'
                
                # Obtener DPI si está disponible
                dpi = img.info.get('dpi', (72, 72))
                
                # Calcular puntuación de calidad básica
                quality_score = self._calculate_basic_quality_score(width, height, file_stat.st_size)
            
            return {
                'width': width,
                'height': height,
                'format': format_name,
                'channels': channels,
                'color_space': color_space,
                'file_size': file_stat.st_size,
                'dpi': dpi,
                'quality_score': quality_score,
                'last_modified': file_stat.st_mtime,
                'path': image_path
            }
            
        except Exception as e:
            self.logger.error(f"Error obteniendo información de imagen {image_path}: {e}")
            return {
                'width': 0,
                'height': 0,
                'format': 'Unknown',
                'channels': 3,
                'color_space': 'Unknown',
                'file_size': 0,
                'dpi': (72, 72),
                'quality_score': 0.0,
                'error': str(e)
            }
    
    def _calculate_basic_quality_score(self, width: int, height: int, file_size: int) -> float:
        """
        Calcula una puntuación básica de calidad basada en dimensiones y tamaño de archivo
        
        Args:
            width: Ancho de la imagen
            height: Alto de la imagen
            file_size: Tamaño del archivo en bytes
            
        Returns:
            Puntuación de calidad entre 0.0 y 1.0
        """
        try:
            # Calcular resolución total
            total_pixels = width * height
            
            # Puntuación basada en resolución (normalizada)
            resolution_score = min(total_pixels / (2048 * 2048), 1.0)  # Normalizar a 2048x2048
            
            # Puntuación basada en tamaño de archivo (bytes por pixel)
            if total_pixels > 0:
                bytes_per_pixel = file_size / total_pixels
                # Asumir que 3-4 bytes por pixel es óptimo para imágenes de calidad
                size_score = min(bytes_per_pixel / 4.0, 1.0)
            else:
                size_score = 0.0
            
            # Combinar puntuaciones
            quality_score = (resolution_score * 0.6) + (size_score * 0.4)
            
            return min(max(quality_score, 0.0), 1.0)  # Asegurar rango [0, 1]
            
        except Exception as e:
            self.logger.warning(f"Error calculando puntuación de calidad: {e}")
            return 0.5  # Valor por defecto