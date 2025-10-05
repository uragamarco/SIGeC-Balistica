"""
Pipeline Científico Unificado para Análisis Balístico
====================================================

Este módulo implementa el pipeline completo de análisis balístico forense,
integrando todos los componentes del sistema SIGeC-Balisticaen un flujo unificado:

1. Preprocesamiento NIST
2. Evaluación de calidad de imagen
3. Detección de ROI con Watershed
4. Extracción de características (ORB/SIFT)
5. Matching con ponderación de calidad
6. Análisis CMC
7. Conclusión AFTE

Basado en:
- NIST Special Publication 800-101 Rev. 1
- AFTE Theory and Practice of Firearm Identification
- Song et al. (2013-2014) CMC Algorithm
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
from pathlib import Path
import json

# Importar configuración del pipeline
try:
    from core.pipeline_config import PipelineConfiguration, PipelineLevel, AFTEConclusion
    PIPELINE_CONFIG_AVAILABLE = True
except ImportError:
    PIPELINE_CONFIG_AVAILABLE = False
    
    # Definir clases básicas si no están disponibles
    class PipelineLevel(Enum):
        BASIC = "basic"
        STANDARD = "standard"
        ADVANCED = "advanced"
        FORENSIC = "forensic"
    
    class AFTEConclusion(Enum):
        IDENTIFICATION = "identification"
        INCONCLUSIVE = "inconclusive"
        ELIMINATION = "elimination"
        UNSUITABLE = "unsuitable"

# Importar componentes del sistema
try:
    from image_processing.unified_preprocessor import UnifiedPreprocessor, PreprocessingConfig, PreprocessingLevel
    PREPROCESSOR_AVAILABLE = True
except ImportError:
    PREPROCESSOR_AVAILABLE = False

try:
    from image_processing.unified_roi_detector import UnifiedROIDetector, ROIDetectionConfig, DetectionLevel
    ROI_DETECTOR_AVAILABLE = True
except ImportError:
    ROI_DETECTOR_AVAILABLE = False

try:
    from nist_standards.quality_metrics import NISTQualityMetrics, NISTQualityReport
    QUALITY_METRICS_AVAILABLE = True
except ImportError:
    QUALITY_METRICS_AVAILABLE = False

try:
    from matching.unified_matcher import UnifiedMatcher, MatchingConfig, MatchResult, AlgorithmType
    MATCHER_AVAILABLE = True
except ImportError:
    MATCHER_AVAILABLE = False

try:
    from matching.cmc_algorithm import CMCAlgorithm, CMCParameters, CMCMatchResult
    CMC_AVAILABLE = True
except ImportError:
    CMC_AVAILABLE = False

try:
    from utils.logger import LoggerMixin
    LOGGER_MIXIN_AVAILABLE = True
except ImportError:
    LOGGER_MIXIN_AVAILABLE = False
    class LoggerMixin:
        def __init__(self, *args, **kwargs):
            self.logger = logging.getLogger(self.__class__.__name__)


@dataclass
class PipelineResult:
    """Resultado completo del análisis del pipeline"""
    # Información básica
    image1_path: str
    image2_path: str
    analysis_timestamp: str
    processing_time: float
    
    # Resultados de calidad
    image1_quality: Optional[NISTQualityReport] = None
    image2_quality: Optional[NISTQualityReport] = None
    quality_assessment_passed: bool = False
    
    # Resultados de preprocesamiento
    preprocessing_successful: bool = False
    preprocessing_steps: List[str] = field(default_factory=list)
    
    # Resultados de ROI
    roi1_detected: bool = False
    roi2_detected: bool = False
    roi1_regions: List[Dict] = field(default_factory=list)
    roi2_regions: List[Dict] = field(default_factory=list)
    
    # Resultados de matching
    match_result: Optional[MatchResult] = None
    similarity_score: float = 0.0
    quality_weighted_score: float = 0.0
    
    # Resultados CMC
    cmc_result: Optional[CMCMatchResult] = None
    cmc_count: int = 0
    cmc_passed: bool = False
    
    # Conclusión final
    afte_conclusion: AFTEConclusion = AFTEConclusion.UNSUITABLE
    confidence: float = 0.0
    
    # Datos adicionales
    intermediate_results: Dict[str, Any] = field(default_factory=dict)
    error_messages: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte el resultado a diccionario"""
        return {
            'image1_path': self.image1_path,
            'image2_path': self.image2_path,
            'analysis_timestamp': self.analysis_timestamp,
            'processing_time': self.processing_time,
            'quality_assessment_passed': self.quality_assessment_passed,
            'preprocessing_successful': self.preprocessing_successful,
            'preprocessing_steps': self.preprocessing_steps,
            'roi1_detected': self.roi1_detected,
            'roi2_detected': self.roi2_detected,
            'similarity_score': self.similarity_score,
            'quality_weighted_score': self.quality_weighted_score,
            'cmc_count': self.cmc_count,
            'cmc_passed': self.cmc_passed,
            'afte_conclusion': self.afte_conclusion.value,
            'confidence': self.confidence,
            'error_messages': self.error_messages,
            'warnings': self.warnings
        }


class ScientificPipeline(LoggerMixin):
    """
    Pipeline científico unificado para análisis balístico forense
    
    Integra todos los componentes del sistema SIGeC-Balisticaen un flujo
    de trabajo científicamente validado y conforme a estándares NIST/AFTE.
    """
    
    def __init__(self, config: Optional[Union[Dict, PipelineConfiguration]] = None):
        """
        Inicializa el pipeline científico
        
        Args:
            config: Configuración del pipeline
        """
        super().__init__()
        
        # Procesar configuración
        if config is None:
            if PIPELINE_CONFIG_AVAILABLE:
                from core.pipeline_config import create_pipeline_config
                self.config = create_pipeline_config("standard")
            else:
                # Configuración básica de fallback
                self.config = self._create_fallback_config()
        elif isinstance(config, dict):
            if PIPELINE_CONFIG_AVAILABLE:
                self.config = PipelineConfiguration.from_dict(config)
            else:
                self.config = self._create_fallback_config()
        else:
            self.config = config
        
        # Verificar dependencias
        self._check_dependencies()
        
        # Inicializar componentes
        self._initialize_components()
        
        self.logger.info(f"Pipeline científico inicializado (nivel: {self.config.level.value})")
    
    def _create_fallback_config(self) -> 'PipelineConfiguration':
        """Crea una configuración básica de fallback"""
        from core.pipeline_config import (
            PipelineConfiguration, PipelineLevel,
            QualityAssessmentConfig, PreprocessingConfig, 
            ROIDetectionConfig, MatchingConfig, 
            CMCAnalysisConfig, AFTEConclusionConfig
        )
        
        return PipelineConfiguration(
            level=PipelineLevel.BASIC,
            quality_assessment=QualityAssessmentConfig(),
            preprocessing=PreprocessingConfig(),
            roi_detection=ROIDetectionConfig(),
            matching=MatchingConfig(),
            cmc_analysis=CMCAnalysisConfig(),
            afte_conclusion=AFTEConclusionConfig()
        )
    
    def _check_dependencies(self):
        """Verifica la disponibilidad de componentes"""
        missing_components = []
        
        if not PREPROCESSOR_AVAILABLE:
            missing_components.append("UnifiedPreprocessor")
        if not ROI_DETECTOR_AVAILABLE:
            missing_components.append("UnifiedROIDetector")
        if not QUALITY_METRICS_AVAILABLE:
            missing_components.append("NISTQualityMetrics")
        if not MATCHER_AVAILABLE:
            missing_components.append("UnifiedMatcher")
        if not CMC_AVAILABLE:
            missing_components.append("CMCAlgorithm")
        
        if missing_components:
            self.logger.warning(f"Componentes no disponibles: {', '.join(missing_components)}")
    
    def _initialize_components(self):
        """Inicializa los componentes del pipeline según la configuración"""
        
        # Inicializar preprocesador
        if PREPROCESSOR_AVAILABLE:
            try:
                if hasattr(self.config, 'preprocessing'):
                    # Usar configuración específica de preprocessing
                    preproc_config = PreprocessingConfig()
                    preproc_config.level = self.config.preprocessing.level
                    self.preprocessor = UnifiedPreprocessor(preproc_config)
                else:
                    # Configuración básica
                    self.preprocessor = UnifiedPreprocessor()
                self.logger.info("Preprocesador inicializado")
            except Exception as e:
                self.logger.warning(f"Error inicializando preprocesador: {e}")
                self.preprocessor = None
        else:
            self.preprocessor = None
            self.logger.warning("Preprocesador no disponible")
        
        # Inicializar detector de ROI
        if ROI_DETECTOR_AVAILABLE:
            try:
                if hasattr(self.config, 'roi_detection'):
                    # Usar configuración específica de ROI
                    roi_config = ROIDetectionConfig()
                    roi_config.detection_level = self.config.roi_detection.detection_level
                    roi_config.enabled = self.config.roi_detection.enabled
                    self.roi_detector = UnifiedROIDetector(roi_config)
                else:
                    # Configuración básica
                    self.roi_detector = UnifiedROIDetector()
                self.logger.info("Detector de ROI inicializado")
            except Exception as e:
                self.logger.warning(f"Error inicializando detector ROI: {e}")
                self.roi_detector = None
        else:
            self.roi_detector = None
            self.logger.warning("Detector de ROI no disponible")
        
        # Inicializar evaluador de calidad
        if QUALITY_METRICS_AVAILABLE:
            try:
                self.quality_metrics = NISTQualityMetrics()
                self.logger.info("Evaluador de calidad inicializado")
            except Exception as e:
                self.logger.warning(f"Error inicializando evaluador de calidad: {e}")
                self.quality_metrics = None
        else:
            self.quality_metrics = None
            self.logger.warning("Evaluador de calidad no disponible")
        
        # Inicializar matcher
        if MATCHER_AVAILABLE:
            try:
                if hasattr(self.config, 'matching'):
                    # Usar configuración específica de matching
                    match_config = MatchingConfig()
                    match_config.algorithm = self.config.matching.algorithm
                    match_config.similarity_threshold = self.config.matching.similarity_threshold
                    self.matcher = UnifiedMatcher(match_config)
                else:
                    # Configuración básica
                    self.matcher = UnifiedMatcher()
                self.logger.info("Matcher inicializado")
            except Exception as e:
                self.logger.warning(f"Error inicializando matcher: {e}")
                self.matcher = None
        else:
            self.matcher = None
            self.logger.warning("Matcher no disponible")
        
        # Inicializar analizador CMC
        if CMC_AVAILABLE:
            try:
                if hasattr(self.config, 'cmc_analysis'):
                    # Usar configuración específica de CMC
                    cmc_params = CMCParameters()
                    cmc_params.cmc_threshold = self.config.cmc_analysis.cmc_threshold
                    cmc_params.min_cell_size = self.config.cmc_analysis.min_cell_size
                    cmc_params.max_cell_size = self.config.cmc_analysis.max_cell_size
                    self.cmc_analyzer = CMCAlgorithm(cmc_params)
                else:
                    # Configuración básica
                    self.cmc_analyzer = CMCAlgorithm()
                self.logger.info("Analizador CMC inicializado")
            except Exception as e:
                self.logger.warning(f"Error inicializando analizador CMC: {e}")
                self.cmc_analyzer = None
        else:
            self.cmc_analyzer = None
            self.logger.warning("Analizador CMC no disponible")
    
    def process_comparison(self, image1: Union[str, np.ndarray], 
                          image2: Union[str, np.ndarray]) -> PipelineResult:
        """
        Ejecuta el pipeline completo de comparación balística
        
        Args:
            image1: Primera imagen (ruta o array numpy)
            image2: Segunda imagen (ruta o array numpy)
            
        Returns:
            Resultado completo del análisis
        """
        start_time = time.time()
        
        # Crear resultado
        result = PipelineResult(
            image1_path=str(image1) if isinstance(image1, str) else "array",
            image2_path=str(image2) if isinstance(image2, str) else "array",
            analysis_timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            processing_time=0.0
        )
        
        try:
            # 1. Cargar imágenes
            img1_array, img2_array = self._load_images(image1, image2)
            if img1_array is None or img2_array is None:
                result.error_messages.append("Error cargando imágenes")
                result.afte_conclusion = AFTEConclusion.UNSUITABLE
                return result
            
            # 2. Evaluación de calidad NIST
            if self.quality_metrics:
                result = self._assess_quality(img1_array, img2_array, result)
                if not result.quality_assessment_passed:
                    result.afte_conclusion = AFTEConclusion.UNSUITABLE
                    return result
            
            # 3. Preprocesamiento NIST
            if self.preprocessor:
                processed_img1, processed_img2, result = self._preprocess_images(
                    img1_array, img2_array, result
                )
                if not result.preprocessing_successful:
                    result.error_messages.append("Error en preprocesamiento")
                    result.afte_conclusion = AFTEConclusion.UNSUITABLE
                    return result
            else:
                processed_img1, processed_img2 = img1_array, img2_array
            
            # 4. Detección de ROI
            if self.roi_detector:
                result = self._detect_roi(processed_img1, processed_img2, result)
            
            # 5. Extracción de características y matching
            if self.matcher:
                result = self._perform_matching(processed_img1, processed_img2, result)
            
            # 6. Realizar análisis CMC
            if self.cmc_analyzer and result.match_result:
                result = self._perform_cmc_analysis(processed_img1, processed_img2, result)
            
            # 7. Determinar conclusión AFTE
            result = self._determine_afte_conclusion(result)
            
        except Exception as e:
            self.logger.error(f"Error en pipeline: {e}")
            result.error_messages.append(f"Error crítico: {str(e)}")
            result.afte_conclusion = AFTEConclusion.UNSUITABLE
        
        finally:
            result.processing_time = time.time() - start_time
        
        return result
    
    def _load_images(self, image1: Union[str, np.ndarray], 
                    image2: Union[str, np.ndarray]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Carga las imágenes desde archivo o usa arrays directamente"""
        try:
            if isinstance(image1, str):
                img1 = cv2.imread(image1)
            else:
                img1 = image1.copy()
            
            if isinstance(image2, str):
                img2 = cv2.imread(image2)
            else:
                img2 = image2.copy()
            
            return img1, img2
        except Exception as e:
            self.logger.error(f"Error cargando imágenes: {e}")
            return None, None
    
    def _assess_quality(self, img1: np.ndarray, img2: np.ndarray, 
                       result: PipelineResult) -> PipelineResult:
        """Evalúa la calidad de las imágenes según estándares NIST"""
        try:
            if self.quality_metrics is None:
                result.warnings.append("Evaluador de calidad no disponible")
                return result
                
            # Evaluar calidad de imagen 1
            result.image1_quality = self.quality_metrics.analyze_image_quality(img1, "image1")
            
            # Evaluar calidad de imagen 2
            result.image2_quality = self.quality_metrics.analyze_image_quality(img2, "image2")
            
            # Verificar umbrales mínimos usando configuración
            min_quality = 0.5  # Valor por defecto
            if hasattr(self.config, 'quality_assessment') and hasattr(self.config.quality_assessment, 'min_quality_score'):
                min_quality = self.config.quality_assessment.min_quality_score
            
            quality1_passed = result.image1_quality.quality_score >= min_quality
            quality2_passed = result.image2_quality.quality_score >= min_quality
            
            result.quality_assessment_passed = quality1_passed and quality2_passed
            
            if not result.quality_assessment_passed:
                result.warnings.append("Una o ambas imágenes no cumplen los estándares mínimos de calidad")
            
            self.logger.info(f"Calidad imagen 1: {result.image1_quality.quality_score:.3f}")
            self.logger.info(f"Calidad imagen 2: {result.image2_quality.quality_score:.3f}")
            
        except Exception as e:
            self.logger.error(f"Error en evaluación de calidad: {e}")
            result.error_messages.append(f"Error evaluando calidad: {str(e)}")
        
        return result
    
    def _preprocess_images(self, img1: np.ndarray, img2: np.ndarray, 
                          result: PipelineResult) -> Tuple[np.ndarray, np.ndarray, PipelineResult]:
        """Preprocesa las imágenes según estándares NIST"""
        try:
            if self.preprocessor is None:
                result.warnings.append("Preprocesador no disponible")
                return img1, img2, result
                
            # Preprocesar imagen 1
            preprocessing_result1 = self.preprocessor.preprocess_image(img1)
            processed_img1 = preprocessing_result1.processed_image
            
            # Preprocesar imagen 2
            preprocessing_result2 = self.preprocessor.preprocess_image(img2)
            processed_img2 = preprocessing_result2.processed_image
            
            # Registrar pasos aplicados
            result.preprocessing_steps = preprocessing_result1.steps_applied
            result.preprocessing_successful = True
            
            self.logger.info(f"Preprocesamiento completado: {len(result.preprocessing_steps)} pasos")
            
            return processed_img1, processed_img2, result
            
        except Exception as e:
            self.logger.error(f"Error en preprocesamiento: {e}")
            result.error_messages.append(f"Error en preprocesamiento: {str(e)}")
            return img1, img2, result
    
    def _detect_roi(self, img1: np.ndarray, img2: np.ndarray, 
                   result: PipelineResult) -> PipelineResult:
        """Detecta regiones de interés en ambas imágenes"""
        try:
            if self.roi_detector is None:
                result.warnings.append("Detector de ROI no disponible")
                return result
                
            # Detectar ROI en imagen 1
            roi_regions1 = self.roi_detector.detect_roi(img1, 'cartridge_case')
            result.roi1_detected = len(roi_regions1) > 0
            result.roi1_regions = [region.to_dict() for region in roi_regions1]
            
            # Detectar ROI en imagen 2
            roi_regions2 = self.roi_detector.detect_roi(img2, 'cartridge_case')
            result.roi2_detected = len(roi_regions2) > 0
            result.roi2_regions = [region.to_dict() for region in roi_regions2]
            
            self.logger.info(f"ROI detectadas: {len(roi_regions1)} en img1, {len(roi_regions2)} en img2")
            
        except Exception as e:
            self.logger.error(f"Error en detección de ROI: {e}")
            result.error_messages.append(f"Error detectando ROI: {str(e)}")
        
        return result
    
    def _perform_matching(self, img1: np.ndarray, img2: np.ndarray, 
                         result: PipelineResult) -> PipelineResult:
        """Realiza el matching de características con ponderación de calidad"""
        try:
            if self.matcher is None:
                result.warnings.append("Matcher no disponible")
                return result
                
            # Obtener scores de calidad para ponderación
            quality1 = result.image1_quality.quality_score if result.image1_quality else 1.0
            quality2 = result.image2_quality.quality_score if result.image2_quality else 1.0
            
            # Realizar matching
            match_result = self.matcher.compare_images(img1, img2)
            
            # Aplicar ponderación de calidad si está habilitada
            enable_quality_weighting = False
            if hasattr(self.config, 'matching') and hasattr(self.config.matching, 'enable_quality_weighting'):
                enable_quality_weighting = self.config.matching.enable_quality_weighting
            
            if enable_quality_weighting and result.image1_quality and result.image2_quality:
                # El UnifiedMatcher ya maneja la ponderación internamente
                match_result.image1_quality_score = quality1
                match_result.image2_quality_score = quality2
                match_result.combined_quality_score = (quality1 * quality2) ** 0.5  # Media geométrica
                match_result.quality_weighted_similarity = match_result.similarity_score * match_result.combined_quality_score
            
            result.match_result = match_result
            result.similarity_score = match_result.similarity_score
            result.quality_weighted_score = getattr(match_result, 'quality_weighted_similarity', match_result.similarity_score)
            
            self.logger.info(f"Matching completado: score={result.similarity_score:.3f}, weighted={result.quality_weighted_score:.3f}")
            
        except Exception as e:
            self.logger.error(f"Error en matching: {e}")
            result.error_messages.append(f"Error en matching: {str(e)}")
        
        return result
    
    def _perform_cmc_analysis(self, img1: np.ndarray, img2: np.ndarray, 
                             result: PipelineResult) -> PipelineResult:
        """Realiza análisis CMC para validación científica"""
        try:
            if self.cmc_analyzer is None:
                result.warnings.append("Analizador CMC no disponible")
                return result
                
            # Convertir a escala de grises si es necesario
            if len(img1.shape) == 3:
                img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            else:
                img1_gray = img1
            
            if len(img2.shape) == 3:
                img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            else:
                img2_gray = img2
            
            # Realizar análisis CMC
            cmc_result = self.cmc_analyzer.analyze_cmc(img1_gray, img2_gray)
            
            result.cmc_result = cmc_result
            result.cmc_count = cmc_result.cmc_count
            
            # Verificar umbral CMC usando configuración
            cmc_threshold = 6  # Valor por defecto
            if hasattr(self.config, 'cmc_analysis') and hasattr(self.config.cmc_analysis, 'cmc_threshold'):
                cmc_threshold = self.config.cmc_analysis.cmc_threshold
            
            result.cmc_passed = result.cmc_count >= cmc_threshold
            
            self.logger.info(f"Análisis CMC completado: {result.cmc_count} CMCs detectados")
            
        except Exception as e:
            self.logger.error(f"Error en análisis CMC: {e}")
            result.error_messages.append(f"Error en análisis CMC: {str(e)}")
        
        return result
    
    def _determine_afte_conclusion(self, result: PipelineResult) -> PipelineResult:
        """Determina la conclusión AFTE basada en todos los análisis"""
        try:
            # Verificar si hay errores críticos
            if result.error_messages:
                result.afte_conclusion = AFTEConclusion.UNSUITABLE
                result.confidence = 0.0
                return result
            
            # Calcular confianza basada en múltiples factores
            confidence_factors = []
            
            # Factor de calidad
            if result.image1_quality and result.image2_quality:
                quality_factor = min(result.image1_quality.quality_score, result.image2_quality.quality_score)
                confidence_factors.append(quality_factor)
            
            # Factor de matching
            if result.match_result:
                matching_factor = result.match_result.confidence
                confidence_factors.append(matching_factor)
            
            # Factor CMC usando configuración
            cmc_threshold = 6  # Valor por defecto
            if hasattr(self.config, 'cmc_analysis') and hasattr(self.config.cmc_analysis, 'cmc_threshold'):
                cmc_threshold = self.config.cmc_analysis.cmc_threshold
                
            if result.cmc_result:
                cmc_factor = min(1.0, result.cmc_count / cmc_threshold)
                confidence_factors.append(cmc_factor)
            
            # Calcular confianza combinada
            if confidence_factors:
                result.confidence = np.mean(confidence_factors)
            else:
                result.confidence = 0.0
            
            # Obtener umbrales de configuración
            confidence_threshold = 0.8  # Valor por defecto
            similarity_threshold = 0.7  # Valor por defecto
            
            if hasattr(self.config, 'afte_conclusion'):
                if hasattr(self.config.afte_conclusion, 'confidence_threshold'):
                    confidence_threshold = self.config.afte_conclusion.confidence_threshold
                if hasattr(self.config.afte_conclusion, 'identification_threshold'):
                    similarity_threshold = self.config.afte_conclusion.identification_threshold
            
            # Determinar conclusión AFTE
            if result.confidence >= confidence_threshold:
                if result.quality_weighted_score >= similarity_threshold:
                    if result.cmc_passed or not hasattr(self.config, 'cmc_analysis') or not self.config.cmc_analysis.enabled:
                        result.afte_conclusion = AFTEConclusion.IDENTIFICATION
                    else:
                        result.afte_conclusion = AFTEConclusion.INCONCLUSIVE
                else:
                    result.afte_conclusion = AFTEConclusion.ELIMINATION
            else:
                if result.quality_weighted_score >= 0.3:  # Umbral mínimo para inconclusive
                    result.afte_conclusion = AFTEConclusion.INCONCLUSIVE
                else:
                    result.afte_conclusion = AFTEConclusion.ELIMINATION
            
            self.logger.info(f"Conclusión AFTE: {result.afte_conclusion.value} (confianza: {result.confidence:.3f})")
            
        except Exception as e:
            self.logger.error(f"Error determinando conclusión AFTE: {e}")
            result.afte_conclusion = AFTEConclusion.UNSUITABLE
            result.confidence = 0.0
        
        return result
    
    def export_report(self, result: PipelineResult, output_path: str) -> bool:
        """Exporta un reporte detallado del análisis"""
        try:
            report_data = result.to_dict()
            
            # Agregar información adicional del pipeline usando configuración centralizada
            pipeline_config_data = {
                'level': self.config.level.value if hasattr(self.config, 'level') else 'standard'
            }
            
            # Agregar configuraciones específicas si están disponibles
            if hasattr(self.config, 'preprocessing') and hasattr(self.config.preprocessing, 'level'):
                pipeline_config_data['preprocessing_level'] = self.config.preprocessing.level.value
            
            if hasattr(self.config, 'matching') and hasattr(self.config.matching, 'algorithm'):
                pipeline_config_data['matching_algorithm'] = self.config.matching.algorithm.value
            
            if hasattr(self.config, 'afte_conclusion'):
                if hasattr(self.config.afte_conclusion, 'identification_threshold'):
                    pipeline_config_data['similarity_threshold'] = self.config.afte_conclusion.identification_threshold
                if hasattr(self.config.afte_conclusion, 'confidence_threshold'):
                    pipeline_config_data['confidence_threshold'] = self.config.afte_conclusion.confidence_threshold
            
            if hasattr(self.config, 'cmc_analysis') and hasattr(self.config.cmc_analysis, 'cmc_threshold'):
                pipeline_config_data['cmc_threshold'] = self.config.cmc_analysis.cmc_threshold
            
            report_data['pipeline_config'] = pipeline_config_data
            
            # Guardar reporte
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Reporte exportado: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exportando reporte: {e}")
            return False


def create_pipeline_config(level: str = "standard") -> 'PipelineConfiguration':
    """
    Crea una configuración predefinida del pipeline
    
    Args:
        level: Nivel de análisis ("basic", "standard", "advanced", "forensic")
        
    Returns:
        Configuración del pipeline
    """
    if PIPELINE_CONFIG_AVAILABLE:
        from core.pipeline_config import (
            PipelineConfiguration, PipelineLevel, 
            QualityAssessmentConfig, PreprocessingConfig, ROIDetectionConfig,
            MatchingConfig, CMCAnalysisConfig, AFTEConclusionConfig,
            PreprocessingLevel, DetectionLevel, AlgorithmType
        )
        
        level_enum = PipelineLevel(level.lower())
        
        if level_enum == PipelineLevel.BASIC:
            return PipelineConfiguration(
                level=PipelineLevel.BASIC,
                quality_assessment=QualityAssessmentConfig(
                    enabled=True,
                    min_quality_score=0.3
                ),
                preprocessing=PreprocessingConfig(
                    level=PreprocessingLevel.BASIC,
                    enabled=True
                ),
                roi_detection=ROIDetectionConfig(
                    level=DetectionLevel.SIMPLE,
                    enabled=True
                ),
                matching=MatchingConfig(
                    algorithm=AlgorithmType.ORB,
                    enabled=True
                ),
                cmc_analysis=CMCAnalysisConfig(
                    enabled=False,
                    cmc_threshold=4
                ),
                afte_conclusion=AFTEConclusionConfig(
                    identification_threshold=0.5,
                    confidence_threshold=0.6
                )
            )
        elif level_enum == PipelineLevel.STANDARD:
            return PipelineConfiguration(
                level=PipelineLevel.STANDARD,
                quality_assessment=QualityAssessmentConfig(
                    enabled=True,
                    min_quality_score=0.5
                ),
                preprocessing=PreprocessingConfig(
                    level=PreprocessingLevel.STANDARD,
                    enabled=True
                ),
                roi_detection=ROIDetectionConfig(
                    level=DetectionLevel.STANDARD,
                    enabled=True
                ),
                matching=MatchingConfig(
                    algorithm=AlgorithmType.ORB,
                    enabled=True
                ),
                cmc_analysis=CMCAnalysisConfig(
                    enabled=True,
                    cmc_threshold=6
                ),
                afte_conclusion=AFTEConclusionConfig(
                    identification_threshold=0.7,
                    confidence_threshold=0.8
                )
            )
        elif level_enum == PipelineLevel.ADVANCED:
            return PipelineConfiguration(
                level=PipelineLevel.ADVANCED,
                quality_assessment=QualityAssessmentConfig(
                    enabled=True,
                    min_quality_score=0.6
                ),
                preprocessing=PreprocessingConfig(
                    level=PreprocessingLevel.ADVANCED,
                    enabled=True
                ),
                roi_detection=ROIDetectionConfig(
                    level=DetectionLevel.ADVANCED,
                    enabled=True
                ),
                matching=MatchingConfig(
                    algorithm=AlgorithmType.SIFT,
                    enabled=True
                ),
                cmc_analysis=CMCAnalysisConfig(
                    enabled=True,
                    cmc_threshold=8
                ),
                afte_conclusion=AFTEConclusionConfig(
                    identification_threshold=0.75,
                    confidence_threshold=0.85
                )
            )
        elif level_enum == PipelineLevel.FORENSIC:
            return PipelineConfiguration(
                level=PipelineLevel.FORENSIC,
                quality_assessment=QualityAssessmentConfig(
                    enabled=True,
                    min_quality_score=0.7
                ),
                preprocessing=PreprocessingConfig(
                    level=PreprocessingLevel.FORENSIC,
                    enabled=True
                ),
                roi_detection=ROIDetectionConfig(
                    level=DetectionLevel.ADVANCED,
                    enabled=True
                ),
                matching=MatchingConfig(
                    algorithm=AlgorithmType.SIFT,
                    enabled=True
                ),
                cmc_analysis=CMCAnalysisConfig(
                    enabled=True,
                    cmc_threshold=10
                ),
                afte_conclusion=AFTEConclusionConfig(
                    identification_threshold=0.8,
                    confidence_threshold=0.9
                )
            )
    else:
        # Fallback para cuando no está disponible la configuración centralizada
        return None


if __name__ == "__main__":
    import argparse
    
    # Configurar logging
    logging.basicConfig(level=logging.INFO)
    
    # Parser de argumentos
    parser = argparse.ArgumentParser(description="Pipeline científico unificado para análisis balístico")
    parser.add_argument("image1", help="Primera imagen para comparación")
    parser.add_argument("image2", help="Segunda imagen para comparación")
    parser.add_argument("--level", "-l", choices=["basic", "standard", "advanced", "forensic"], 
                       default="standard", help="Nivel de análisis")
    parser.add_argument("--output", "-o", help="Archivo de salida para reporte")
    
    args = parser.parse_args()
    
    # Crear pipeline
    config = create_pipeline_config(args.level)
    pipeline = ScientificPipeline(config)
    
    # Ejecutar análisis
    result = pipeline.process_comparison(args.image1, args.image2)
    
    # Mostrar resultados
    print("\n=== RESULTADOS DEL ANÁLISIS BALÍSTICO ===")
    print(f"Conclusión AFTE: {result.afte_conclusion.value.upper()}")
    print(f"Confianza: {result.confidence:.3f}")
    print(f"Score de similitud: {result.similarity_score:.3f}")
    print(f"Score ponderado por calidad: {result.quality_weighted_score:.3f}")
    print(f"CMCs detectados: {result.cmc_count}")
    print(f"Tiempo de procesamiento: {result.processing_time:.2f}s")
    
    if result.error_messages:
        print(f"\nErrores: {', '.join(result.error_messages)}")
    
    if result.warnings:
        print(f"\nAdvertencias: {', '.join(result.warnings)}")
    
    # Exportar reporte si se especifica
    if args.output:
        pipeline.export_report(result, args.output)
        print(f"\nReporte guardado en: {args.output}")