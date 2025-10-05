"""
Módulo de Validación de Cumplimiento NIST para Procesamiento de Imágenes

Este módulo integra todos los aspectos de cumplimiento NIST para imágenes balísticas:
- Validación de calibración espacial (DPI)
- Verificación de métricas de calidad de imagen
- Validación de uniformidad de iluminación
- Generación de reportes de cumplimiento
- Recomendaciones para mejora de calidad

Autor: Sistema SIGeC-Balistica
Fecha: 2024
"""

import cv2
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path
import json
from datetime import datetime

# Importar módulos NIST existentes
from nist_standards.quality_metrics import NISTQualityMetrics, NISTQualityReport
from nist_standards.nist_schema import NISTImageData
from image_processing.spatial_calibration import SpatialCalibrator, CalibrationData, NISTCalibrationResult

@dataclass
class NISTProcessingReport:
    """Reporte completo de cumplimiento NIST para procesamiento"""
    image_path: str
    timestamp: str
    
    # Datos de calibración
    calibration_result: Optional[NISTCalibrationResult] = None
    
    # Métricas de calidad
    quality_report: Optional[NISTQualityReport] = None
    
    # Validación de procesamiento
    illumination_uniformity: float = 0.0
    preprocessing_applied: List[str] = field(default_factory=list)
    
    # Cumplimiento general
    overall_compliance: bool = False
    compliance_score: float = 0.0
    critical_issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    # Metadatos adicionales
    processing_metadata: Dict[str, Any] = field(default_factory=dict)

class NISTComplianceValidator:
    """
    Validador completo de cumplimiento NIST para procesamiento de imágenes balísticas
    
    Integra:
    - Calibración espacial DPI
    - Métricas de calidad de imagen
    - Validación de procesamiento
    - Generación de reportes
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Inicializar componentes
        self.spatial_calibrator = SpatialCalibrator()
        self.quality_metrics = NISTQualityMetrics()
        
        # Umbrales NIST
        self.nist_requirements = {
            'min_dpi': 1000,
            'min_snr_db': 20,
            'min_contrast': 0.3,
            'max_illumination_variation': 0.1,  # 10%
            'max_geometric_distortion': 0.02,  # 2%
            'min_sharpness': 100,
            'calibration_accuracy': 0.02  # ±2%
        }
        
        # Pesos para score de cumplimiento
        self.compliance_weights = {
            'spatial_calibration': 0.25,
            'image_quality': 0.30,
            'illumination_uniformity': 0.20,
            'preprocessing_quality': 0.15,
            'metadata_completeness': 0.10
        }
    
    def validate_full_compliance(self, 
                               image_path: str,
                               image: Optional[np.ndarray] = None,
                               calibration_method: str = 'auto',
                               reference_object: Optional[str] = None) -> NISTProcessingReport:
        """
        Validación completa de cumplimiento NIST
        
        Args:
            image_path: Ruta de la imagen
            image: Imagen cargada (opcional)
            calibration_method: Método de calibración ('auto', 'metadata', 'reference', 'manual')
            reference_object: Tipo de objeto de referencia si se usa calibración por referencia
            
        Returns:
            NISTProcessingReport con validación completa
        """
        try:
            # Cargar imagen si no se proporciona
            if image is None:
                image = cv2.imread(image_path)
                if image is None:
                    raise ValueError(f"No se pudo cargar la imagen: {image_path}")
            
            # Inicializar reporte
            report = NISTProcessingReport(
                image_path=image_path,
                timestamp=datetime.now().isoformat()
            )
            
            # 1. Validar calibración espacial
            self.logger.info("Validando calibración espacial...")
            report.calibration_result = self._validate_spatial_calibration(
                image_path, image, calibration_method, reference_object
            )
            
            # 2. Evaluar métricas de calidad
            self.logger.info("Evaluando métricas de calidad...")
            report.quality_report = self._evaluate_image_quality(image)
            
            # 3. Validar uniformidad de iluminación
            self.logger.info("Validando uniformidad de iluminación...")
            report.illumination_uniformity = self._calculate_illumination_uniformity(image)
            
            # 4. Evaluar calidad de procesamiento
            self.logger.info("Evaluando procesamiento aplicado...")
            report.preprocessing_applied = self._detect_preprocessing(image)
            
            # 5. Calcular cumplimiento general
            self._calculate_overall_compliance(report)
            
            # 6. Generar recomendaciones
            self._generate_recommendations(report)
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error en validación de cumplimiento: {e}")
            # Retornar reporte con error
            return NISTProcessingReport(
                image_path=image_path,
                timestamp=datetime.now().isoformat(),
                overall_compliance=False,
                compliance_score=0.0,
                critical_issues=[f"Error en validación: {str(e)}"]
            )
    
    def _validate_spatial_calibration(self, 
                                    image_path: str,
                                    image: np.ndarray,
                                    method: str,
                                    reference_object: Optional[str]) -> NISTCalibrationResult:
        """Validar calibración espacial según método especificado"""
        try:
            calibration_data = None
            
            if method == 'auto' or method == 'metadata':
                # Intentar extraer de metadatos primero
                calibration_data = self.spatial_calibrator.calibrate_from_metadata(image_path)
                
                if calibration_data is None and method == 'auto':
                    # Si no hay metadatos, intentar detección automática
                    if reference_object:
                        calibration_data = self.spatial_calibrator.calibrate_with_reference_object(
                            image, reference_object
                        )
            
            elif method == 'reference' and reference_object:
                calibration_data = self.spatial_calibrator.calibrate_with_reference_object(
                    image, reference_object
                )
            
            # Si no se pudo calibrar, crear datos por defecto
            if calibration_data is None:
                calibration_data = CalibrationData(
                    pixels_per_mm=39.37,  # Asumiendo 1000 DPI
                    dpi=1000,
                    calibration_method="assumed_default",
                    confidence=0.1
                )
            
            # Validar cumplimiento NIST
            return self.spatial_calibrator.validate_nist_compliance(calibration_data)
            
        except Exception as e:
            self.logger.error(f"Error validando calibración espacial: {e}")
            # Retornar resultado de error
            default_calibration = CalibrationData(
                pixels_per_mm=0,
                dpi=0,
                calibration_method="error",
                confidence=0.0
            )
            return self.spatial_calibrator.validate_nist_compliance(default_calibration)
    
    def _evaluate_image_quality(self, image: np.ndarray) -> NISTQualityReport:
        """Evaluar métricas de calidad de imagen"""
        try:
            # Crear datos de imagen NIST
            nist_data = NISTImageData(
                image_data=image,
                resolution_dpi=1000,  # Valor por defecto
                bit_depth=8 if image.dtype == np.uint8 else 16,
                color_space="BGR" if len(image.shape) == 3 else "GRAY",
                compression="none",
                metadata={}
            )
            
            # Evaluar calidad
            return self.quality_metrics.assess_overall_quality(nist_data)
            
        except Exception as e:
            self.logger.error(f"Error evaluando calidad de imagen: {e}")
            # Retornar reporte vacío en caso de error
            return NISTQualityReport(
                overall_score=0.0,
                snr_db=0.0,
                contrast_score=0.0,
                sharpness_score=0.0,
                noise_level=1.0,
                illumination_uniformity=0.0,
                meets_nist_standards=False,
                quality_issues=[f"Error en evaluación: {str(e)}"],
                recommendations=["Revisar imagen y repetir evaluación"]
            )
    
    def _calculate_illumination_uniformity(self, image: np.ndarray) -> float:
        """Calcular uniformidad de iluminación"""
        try:
            # Convertir a escala de grises si es necesario
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Dividir imagen en regiones
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
            
            # Calcular uniformidad
            mean_of_means = np.mean(region_means)
            std_of_means = np.std(region_means)
            
            if mean_of_means > 0:
                cv_coefficient = std_of_means / mean_of_means
                # Convertir a uniformidad (1 = perfecta uniformidad)
                uniformity = max(0.0, 1.0 - cv_coefficient / 0.1)
            else:
                uniformity = 0.0
            
            return min(1.0, uniformity)
            
        except Exception as e:
            self.logger.error(f"Error calculando uniformidad: {e}")
            return 0.0
    
    def _detect_preprocessing(self, image: np.ndarray) -> List[str]:
        """Detectar qué procesamiento se ha aplicado a la imagen"""
        applied_processing = []
        
        try:
            # Convertir a escala de grises para análisis
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Detectar suavizado (blur)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            if laplacian_var < 100:
                applied_processing.append("noise_reduction")
            
            # Detectar mejora de contraste
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist_spread = np.sum(hist > np.max(hist) * 0.01)
            if hist_spread > 200:
                applied_processing.append("contrast_enhancement")
            
            # Detectar corrección de iluminación
            # Verificar uniformidad de histograma
            hist_uniform = np.std(hist.flatten())
            if hist_uniform < np.mean(hist) * 0.5:
                applied_processing.append("illumination_correction")
            
            # Detectar filtrado de ruido
            # Calcular gradientes para detectar suavizado
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            if np.mean(gradient_magnitude) < 20:
                if "noise_reduction" not in applied_processing:
                    applied_processing.append("noise_reduction")
            
        except Exception as e:
            self.logger.error(f"Error detectando procesamiento: {e}")
        
        return applied_processing
    
    def _calculate_overall_compliance(self, report: NISTProcessingReport):
        """Calcular cumplimiento general y score"""
        try:
            scores = {}
            issues = []
            warnings = []
            
            # 1. Score de calibración espacial
            if report.calibration_result:
                if report.calibration_result.nist_compliant:
                    scores['spatial_calibration'] = 1.0
                else:
                    scores['spatial_calibration'] = report.calibration_result.quality_score
                    issues.extend(report.calibration_result.validation_errors)
            else:
                scores['spatial_calibration'] = 0.0
                issues.append("No se pudo realizar calibración espacial")
            
            # 2. Score de calidad de imagen
            if report.quality_report:
                scores['image_quality'] = report.quality_report.overall_score
                if not report.quality_report.meets_nist_standards:
                    issues.extend(report.quality_report.quality_issues)
            else:
                scores['image_quality'] = 0.0
                issues.append("No se pudo evaluar calidad de imagen")
            
            # 3. Score de uniformidad de iluminación
            if report.illumination_uniformity >= 0.9:  # NIST requiere >90%
                scores['illumination_uniformity'] = 1.0
            else:
                scores['illumination_uniformity'] = report.illumination_uniformity
                if report.illumination_uniformity < 0.8:
                    issues.append(f"Uniformidad de iluminación insuficiente: {report.illumination_uniformity:.1%}")
                else:
                    warnings.append(f"Uniformidad de iluminación marginal: {report.illumination_uniformity:.1%}")
            
            # 4. Score de calidad de procesamiento
            expected_processing = ['illumination_correction', 'noise_reduction', 'contrast_enhancement']
            processing_score = len(set(report.preprocessing_applied) & set(expected_processing)) / len(expected_processing)
            scores['preprocessing_quality'] = processing_score
            
            if processing_score < 0.5:
                warnings.append("Procesamiento de imagen incompleto")
            
            # 5. Score de completitud de metadatos
            metadata_score = 0.5  # Base score
            if report.calibration_result and report.calibration_result.calibration_data.calibration_method != "assumed_default":
                metadata_score += 0.3
            if report.quality_report and report.quality_report.overall_score > 0:
                metadata_score += 0.2
            
            scores['metadata_completeness'] = min(1.0, metadata_score)
            
            # Calcular score ponderado
            weighted_score = sum(
                scores.get(component, 0) * weight 
                for component, weight in self.compliance_weights.items()
            )
            
            # Determinar cumplimiento general
            overall_compliant = (
                weighted_score >= 0.8 and
                len(issues) == 0 and
                scores.get('spatial_calibration', 0) >= 0.8 and
                scores.get('image_quality', 0) >= 0.7
            )
            
            # Actualizar reporte
            report.overall_compliance = overall_compliant
            report.compliance_score = weighted_score
            report.critical_issues = issues
            report.warnings = warnings
            report.processing_metadata = {
                'component_scores': scores,
                'weights_used': self.compliance_weights,
                'nist_requirements': self.nist_requirements
            }
            
        except Exception as e:
            self.logger.error(f"Error calculando cumplimiento general: {e}")
            report.overall_compliance = False
            report.compliance_score = 0.0
            report.critical_issues = [f"Error en cálculo de cumplimiento: {str(e)}"]
    
    def _generate_recommendations(self, report: NISTProcessingReport):
        """Generar recomendaciones para mejorar cumplimiento"""
        recommendations = []
        
        try:
            # Recomendaciones de calibración
            if report.calibration_result:
                recommendations.extend(report.calibration_result.recommended_actions)
            
            # Recomendaciones de calidad
            if report.quality_report:
                recommendations.extend(report.quality_report.recommendations)
            
            # Recomendaciones de uniformidad
            if report.illumination_uniformity < 0.9:
                recommendations.append("Aplicar corrección de iluminación avanzada")
                recommendations.append("Verificar condiciones de captura de imagen")
            
            # Recomendaciones de procesamiento
            expected_processing = ['illumination_correction', 'noise_reduction', 'contrast_enhancement']
            missing_processing = set(expected_processing) - set(report.preprocessing_applied)
            
            for process in missing_processing:
                if process == 'illumination_correction':
                    recommendations.append("Aplicar corrección de iluminación")
                elif process == 'noise_reduction':
                    recommendations.append("Aplicar reducción de ruido")
                elif process == 'contrast_enhancement':
                    recommendations.append("Aplicar mejora de contraste")
            
            # Recomendaciones generales
            if report.compliance_score < 0.6:
                recommendations.append("Considerar recaptura de imagen con mejores condiciones")
            elif report.compliance_score < 0.8:
                recommendations.append("Aplicar procesamiento adicional para mejorar calidad")
            
            # Eliminar duplicados y ordenar
            recommendations = list(set(recommendations))
            recommendations.sort()
            
            report.recommendations = recommendations
            
        except Exception as e:
            self.logger.error(f"Error generando recomendaciones: {e}")
            report.recommendations = ["Error generando recomendaciones"]
    
    def save_compliance_report(self, report: NISTProcessingReport, filepath: str) -> bool:
        """Guardar reporte de cumplimiento en archivo JSON"""
        try:
            # Convertir reporte a diccionario serializable
            report_dict = {
                'image_path': report.image_path,
                'timestamp': report.timestamp,
                'overall_compliance': report.overall_compliance,
                'compliance_score': report.compliance_score,
                'illumination_uniformity': report.illumination_uniformity,
                'preprocessing_applied': report.preprocessing_applied,
                'critical_issues': report.critical_issues,
                'warnings': report.warnings,
                'recommendations': report.recommendations,
                'processing_metadata': report.processing_metadata
            }
            
            # Agregar datos de calibración si existen
            if report.calibration_result:
                report_dict['calibration'] = {
                    'nist_compliant': report.calibration_result.nist_compliant,
                    'quality_score': report.calibration_result.quality_score,
                    'dpi': report.calibration_result.calibration_data.dpi,
                    'pixels_per_mm': report.calibration_result.calibration_data.pixels_per_mm,
                    'method': report.calibration_result.calibration_data.calibration_method,
                    'confidence': report.calibration_result.calibration_data.confidence
                }
            
            # Agregar datos de calidad si existen
            if report.quality_report:
                report_dict['quality'] = {
                    'overall_score': report.quality_report.overall_score,
                    'meets_nist_standards': report.quality_report.meets_nist_standards,
                    'snr_db': report.quality_report.snr_db,
                    'contrast_score': report.quality_report.contrast_score,
                    'sharpness_score': report.quality_report.sharpness_score,
                    'noise_level': report.quality_report.noise_level,
                    'illumination_uniformity': report.quality_report.illumination_uniformity
                }
            
            # Guardar archivo
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report_dict, f, indent=2, ensure_ascii=False)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error guardando reporte: {e}")
            return False
    
    def load_compliance_report(self, filepath: str) -> Optional[NISTProcessingReport]:
        """Cargar reporte de cumplimiento desde archivo JSON"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Crear reporte básico
            report = NISTProcessingReport(
                image_path=data.get('image_path', ''),
                timestamp=data.get('timestamp', ''),
                overall_compliance=data.get('overall_compliance', False),
                compliance_score=data.get('compliance_score', 0.0),
                illumination_uniformity=data.get('illumination_uniformity', 0.0),
                preprocessing_applied=data.get('preprocessing_applied', []),
                critical_issues=data.get('critical_issues', []),
                warnings=data.get('warnings', []),
                recommendations=data.get('recommendations', []),
                processing_metadata=data.get('processing_metadata', {})
            )
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error cargando reporte: {e}")
            return None