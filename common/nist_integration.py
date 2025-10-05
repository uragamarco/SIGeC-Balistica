"""
Módulo de integración NIST para análisis estadístico avanzado
Proporciona funcionalidades específicas para estándares NIST en análisis balístico
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

try:
    from common.statistical_core import UnifiedStatisticalAnalysis, StatisticalTest
    STATISTICAL_CORE_AVAILABLE = True
except ImportError:
    STATISTICAL_CORE_AVAILABLE = False

try:
    from nist_standards.quality_metrics import NISTQualityReport, QualityLevel
    NIST_STANDARDS_AVAILABLE = True
except ImportError:
    NIST_STANDARDS_AVAILABLE = False


class NISTComplianceLevel(Enum):
    """Niveles de cumplimiento con estándares NIST"""
    FULL_COMPLIANCE = "full_compliance"
    PARTIAL_COMPLIANCE = "partial_compliance"
    NON_COMPLIANCE = "non_compliance"
    UNKNOWN = "unknown"


@dataclass
class NISTStatisticalReport:
    """Reporte estadístico integrado con estándares NIST"""
    compliance_level: NISTComplianceLevel
    quality_score: float
    statistical_confidence: float
    recommendations: List[str]
    detailed_metrics: Dict[str, Any]
    bootstrap_results: Optional[Any] = None
    test_results: Optional[Dict[str, Any]] = None


class NISTStatisticalIntegration:
    """Clase principal para integración estadística con estándares NIST"""
    
    def __init__(self):
        self.statistical_core = None
        if STATISTICAL_CORE_AVAILABLE:
            self.statistical_core = UnifiedStatisticalAnalysis()
        
        # Umbrales NIST para diferentes métricas
        self.nist_thresholds = {
            'snr_min': 20.0,
            'contrast_min': 0.5,
            'uniformity_min': 0.7,
            'sharpness_min': 0.6,
            'resolution_min': 0.5,
            'noise_max': 0.4,
            'brightness_range': (0.3, 0.8),
            'saturation_range': (0.2, 0.9)
        }
        
        # Pesos para cálculo de score global
        self.metric_weights = {
            'snr': 0.25,
            'contrast': 0.20,
            'uniformity': 0.20,
            'sharpness': 0.15,
            'resolution': 0.10,
            'noise': 0.10
        }
    
    def analyze_nist_compliance(self, quality_report) -> NISTStatisticalReport:
        """
        Analiza el cumplimiento con estándares NIST
        
        Args:
            quality_report: Reporte de calidad NIST
            
        Returns:
            NISTStatisticalReport con análisis completo
        """
        if not NIST_STANDARDS_AVAILABLE or quality_report is None:
            return self._create_unknown_report()
        
        # Extraer métricas del reporte
        metrics = self._extract_metrics_from_report(quality_report)
        
        # Evaluar cumplimiento individual
        compliance_scores = self._evaluate_individual_compliance(metrics)
        
        # Calcular score global
        global_score = self._calculate_global_score(compliance_scores)
        
        # Determinar nivel de cumplimiento
        compliance_level = self._determine_compliance_level(global_score)
        
        # Realizar análisis estadístico si está disponible
        statistical_results = None
        if self.statistical_core:
            statistical_results = self._perform_statistical_analysis(metrics)
        
        # Generar recomendaciones
        recommendations = self._generate_recommendations(compliance_scores, metrics)
        
        return NISTStatisticalReport(
            compliance_level=compliance_level,
            quality_score=global_score,
            statistical_confidence=self._calculate_statistical_confidence(statistical_results),
            recommendations=recommendations,
            detailed_metrics=compliance_scores,
            bootstrap_results=statistical_results.get('bootstrap') if statistical_results else None,
            test_results=statistical_results.get('tests') if statistical_results else None
        )
    
    def _extract_metrics_from_report(self, report) -> Dict[str, float]:
        """Extrae métricas numéricas del reporte NIST"""
        metrics = {}
        
        # Métricas principales
        if hasattr(report, 'snr_value'):
            metrics['snr'] = float(report.snr_value)
        if hasattr(report, 'contrast_value'):
            metrics['contrast'] = float(report.contrast_value)
        if hasattr(report, 'uniformity_value'):
            metrics['uniformity'] = float(report.uniformity_value)
        if hasattr(report, 'sharpness_value'):
            metrics['sharpness'] = float(report.sharpness_value)
        
        # Métricas adicionales con valores por defecto
        metrics['resolution'] = getattr(report, 'resolution_value', 0.5)
        metrics['noise'] = getattr(report, 'noise_level', 0.3)
        metrics['brightness'] = getattr(report, 'brightness_value', 0.5)
        metrics['saturation'] = getattr(report, 'saturation_value', 0.5)
        
        return metrics
    
    def _evaluate_individual_compliance(self, metrics: Dict[str, float]) -> Dict[str, Dict[str, Any]]:
        """Evalúa el cumplimiento individual de cada métrica"""
        compliance_scores = {}
        
        for metric, value in metrics.items():
            if metric == 'snr':
                compliance_scores[metric] = {
                    'value': value,
                    'threshold': self.nist_thresholds['snr_min'],
                    'compliant': value >= self.nist_thresholds['snr_min'],
                    'score': min(1.0, value / self.nist_thresholds['snr_min'])
                }
            elif metric == 'contrast':
                compliance_scores[metric] = {
                    'value': value,
                    'threshold': self.nist_thresholds['contrast_min'],
                    'compliant': value >= self.nist_thresholds['contrast_min'],
                    'score': min(1.0, value / self.nist_thresholds['contrast_min'])
                }
            elif metric == 'uniformity':
                compliance_scores[metric] = {
                    'value': value,
                    'threshold': self.nist_thresholds['uniformity_min'],
                    'compliant': value >= self.nist_thresholds['uniformity_min'],
                    'score': min(1.0, value / self.nist_thresholds['uniformity_min'])
                }
            elif metric == 'sharpness':
                compliance_scores[metric] = {
                    'value': value,
                    'threshold': self.nist_thresholds['sharpness_min'],
                    'compliant': value >= self.nist_thresholds['sharpness_min'],
                    'score': min(1.0, value / self.nist_thresholds['sharpness_min'])
                }
            elif metric == 'noise':
                compliance_scores[metric] = {
                    'value': value,
                    'threshold': self.nist_thresholds['noise_max'],
                    'compliant': value <= self.nist_thresholds['noise_max'],
                    'score': max(0.0, 1.0 - (value / self.nist_thresholds['noise_max']))
                }
            elif metric in ['brightness', 'saturation']:
                range_key = f"{metric}_range"
                min_val, max_val = self.nist_thresholds[range_key]
                compliance_scores[metric] = {
                    'value': value,
                    'threshold': f"[{min_val}, {max_val}]",
                    'compliant': min_val <= value <= max_val,
                    'score': 1.0 if min_val <= value <= max_val else 0.5
                }
        
        return compliance_scores
    
    def _calculate_global_score(self, compliance_scores: Dict[str, Dict[str, Any]]) -> float:
        """Calcula el score global ponderado"""
        total_score = 0.0
        total_weight = 0.0
        
        for metric, data in compliance_scores.items():
            if metric in self.metric_weights:
                weight = self.metric_weights[metric]
                score = data['score']
                total_score += weight * score
                total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def _determine_compliance_level(self, global_score: float) -> NISTComplianceLevel:
        """Determina el nivel de cumplimiento basado en el score global"""
        if global_score >= 0.9:
            return NISTComplianceLevel.FULL_COMPLIANCE
        elif global_score >= 0.7:
            return NISTComplianceLevel.PARTIAL_COMPLIANCE
        else:
            return NISTComplianceLevel.NON_COMPLIANCE
    
    def _perform_statistical_analysis(self, metrics: Dict[str, float]) -> Optional[Dict[str, Any]]:
        """Realiza análisis estadístico de las métricas"""
        if not self.statistical_core:
            return None
        
        try:
            # Convertir métricas a array
            metric_values = np.array(list(metrics.values()))
            
            # Bootstrap sampling
            bootstrap_result = self.statistical_core.bootstrap_sampling(
                metric_values,
                statistic_func=np.mean,
                n_bootstrap=1000,
                confidence_level=0.95
            )
            
            # Test de normalidad
            normality_test = self.statistical_core.perform_statistical_test(
                metric_values,
                test_type=StatisticalTest.KOLMOGOROV_SMIRNOV,
                alpha=0.05
            )
            
            return {
                'bootstrap': bootstrap_result,
                'tests': {
                    'normality': normality_test
                },
                'descriptive': {
                    'mean': np.mean(metric_values),
                    'std': np.std(metric_values),
                    'min': np.min(metric_values),
                    'max': np.max(metric_values)
                }
            }
        
        except Exception as e:
            print(f"Error en análisis estadístico: {e}")
            return None
    
    def _calculate_statistical_confidence(self, statistical_results: Optional[Dict[str, Any]]) -> float:
        """Calcula la confianza estadística del análisis"""
        if not statistical_results:
            return 0.5  # Confianza neutral sin análisis estadístico
        
        try:
            # Basado en el ancho del intervalo de confianza del bootstrap
            bootstrap = statistical_results.get('bootstrap')
            if bootstrap and hasattr(bootstrap, 'confidence_interval'):
                ci_lower, ci_upper = bootstrap.confidence_interval
                ci_width = ci_upper - ci_lower
                
                # Confianza inversamente proporcional al ancho del IC
                # Normalizar entre 0.5 y 1.0
                confidence = max(0.5, min(1.0, 1.0 - ci_width))
                return confidence
            
            return 0.7  # Confianza por defecto si hay análisis estadístico
        
        except Exception:
            return 0.5
    
    def _generate_recommendations(self, compliance_scores: Dict[str, Dict[str, Any]], 
                                metrics: Dict[str, float]) -> List[str]:
        """Genera recomendaciones basadas en el análisis"""
        recommendations = []
        
        # Revisar métricas no conformes
        for metric, data in compliance_scores.items():
            if not data['compliant']:
                if metric == 'snr':
                    recommendations.append(
                        f"Mejorar la relación señal-ruido (SNR actual: {data['value']:.2f}, "
                        f"mínimo NIST: {data['threshold']})"
                    )
                elif metric == 'contrast':
                    recommendations.append(
                        f"Aumentar el contraste de la imagen (actual: {data['value']:.2f}, "
                        f"mínimo NIST: {data['threshold']})"
                    )
                elif metric == 'uniformity':
                    recommendations.append(
                        f"Mejorar la uniformidad de iluminación (actual: {data['value']:.2f}, "
                        f"mínimo NIST: {data['threshold']})"
                    )
                elif metric == 'sharpness':
                    recommendations.append(
                        f"Mejorar la nitidez de la imagen (actual: {data['value']:.2f}, "
                        f"mínimo NIST: {data['threshold']})"
                    )
                elif metric == 'noise':
                    recommendations.append(
                        f"Reducir el nivel de ruido (actual: {data['value']:.2f}, "
                        f"máximo NIST: {data['threshold']})"
                    )
        
        # Recomendaciones generales si no hay específicas
        if not recommendations:
            recommendations.append("La imagen cumple con los estándares NIST básicos")
            recommendations.append("Considerar optimizaciones adicionales para mejorar la calidad")
        
        return recommendations
    
    def _create_unknown_report(self) -> NISTStatisticalReport:
        """Crea un reporte para casos donde no se puede determinar el cumplimiento"""
        return NISTStatisticalReport(
            compliance_level=NISTComplianceLevel.UNKNOWN,
            quality_score=0.0,
            statistical_confidence=0.0,
            recommendations=["No se pudo evaluar el cumplimiento NIST"],
            detailed_metrics={},
            bootstrap_results=None,
            test_results=None
        )
    
    def get_nist_thresholds(self) -> Dict[str, Any]:
        """Retorna los umbrales NIST configurados"""
        return self.nist_thresholds.copy()
    
    def update_nist_thresholds(self, new_thresholds: Dict[str, Any]):
        """Actualiza los umbrales NIST"""
        self.nist_thresholds.update(new_thresholds)
    
    def export_compliance_report(self, report: NISTStatisticalReport) -> Dict[str, Any]:
        """Exporta el reporte de cumplimiento en formato estructurado"""
        return {
            'compliance_level': report.compliance_level.value,
            'quality_score': report.quality_score,
            'statistical_confidence': report.statistical_confidence,
            'recommendations': report.recommendations,
            'detailed_metrics': report.detailed_metrics,
            'has_statistical_analysis': report.bootstrap_results is not None,
            'timestamp': np.datetime64('now').astype(str)
        }