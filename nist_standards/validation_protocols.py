"""
Protocolos de Validación NIST para Sistemas Balísticos Forenses
==============================================================

Este módulo implementa protocolos de validación según estándares NIST para
sistemas de análisis balístico forense, incluyendo:

- Cross-validation k-fold
- Métricas de confiabilidad y reproducibilidad
- Validación de precisión y exactitud
- Análisis de incertidumbre
- Protocolos de verificación y calibración

Basado en:
- NIST Special Publication 800-101 Rev. 1
- ISO/IEC 17025:2017 (General requirements for testing and calibration laboratories)
- ASTM E2927 (Standard Test Method for Forensic Comparison of Glass)
- SWGGUN Guidelines for Validation Studies
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
import math
from datetime import datetime
import json
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
import warnings
from scipy import stats
from scipy.stats import chi2_contingency, fisher_exact
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')


class ValidationLevel(Enum):
    """Niveles de validación según NIST"""
    DEVELOPMENTAL = "developmental"    # Validación durante desarrollo
    INTERNAL = "internal"            # Validación interna del laboratorio
    COLLABORATIVE = "collaborative"   # Validación colaborativa entre laboratorios
    OPERATIONAL = "operational"      # Validación operacional continua


class MetricType(Enum):
    """Tipos de métricas de validación"""
    ACCURACY = "accuracy"           # Exactitud
    PRECISION = "precision"         # Precisión
    RECALL = "recall"              # Sensibilidad/Recall
    SPECIFICITY = "specificity"     # Especificidad
    F1_SCORE = "f1_score"          # F1-Score
    AUC_ROC = "auc_roc"            # Área bajo curva ROC
    REPRODUCIBILITY = "reproducibility"  # Reproducibilidad
    REPEATABILITY = "repeatability"      # Repetibilidad


@dataclass
class ValidationResult:
    """Resultado de validación"""
    validation_id: str
    validation_level: ValidationLevel
    validation_date: datetime
    dataset_size: int
    k_folds: int
    metrics: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]
    statistical_tests: Dict[str, Any]
    cross_validation_scores: List[float]
    confusion_matrices: List[np.ndarray]
    roc_curves: List[Tuple[np.ndarray, np.ndarray, float]]
    reliability_metrics: Dict[str, float]
    uncertainty_analysis: Dict[str, Any]
    validation_summary: str
    recommendations: List[str]
    is_valid: bool
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte resultado a diccionario"""
        return {
            'validation_id': self.validation_id,
            'validation_level': self.validation_level.value,
            'validation_date': self.validation_date.isoformat(),
            'dataset_size': self.dataset_size,
            'k_folds': self.k_folds,
            'metrics': self.metrics,
            'confidence_intervals': {k: list(v) for k, v in self.confidence_intervals.items()},
            'statistical_tests': self.statistical_tests,
            'cross_validation_scores': self.cross_validation_scores,
            'confusion_matrices': [cm.tolist() for cm in self.confusion_matrices],
            'roc_curves': [(fpr.tolist(), tpr.tolist(), auc) for fpr, tpr, auc in self.roc_curves],
            'reliability_metrics': self.reliability_metrics,
            'uncertainty_analysis': self.uncertainty_analysis,
            'validation_summary': self.validation_summary,
            'recommendations': self.recommendations,
            'is_valid': self.is_valid
        }


@dataclass
class ValidationDataset:
    """Dataset para validación"""
    features: np.ndarray
    labels: np.ndarray
    metadata: Dict[str, Any]
    ground_truth: Optional[np.ndarray] = None
    quality_scores: Optional[np.ndarray] = None


class NISTValidationProtocols:
    """
    Implementación de protocolos de validación NIST
    """
    
    def __init__(self):
        # Umbrales de aceptación según NIST
        self.acceptance_thresholds = {
            MetricType.ACCURACY: 0.95,      # 95% exactitud mínima
            MetricType.PRECISION: 0.90,     # 90% precisión mínima
            MetricType.RECALL: 0.85,        # 85% sensibilidad mínima
            MetricType.SPECIFICITY: 0.95,   # 95% especificidad mínima
            MetricType.F1_SCORE: 0.90,      # 90% F1-score mínimo
            MetricType.AUC_ROC: 0.90,       # 90% AUC mínimo
            MetricType.REPRODUCIBILITY: 0.85, # 85% reproducibilidad mínima
            MetricType.REPEATABILITY: 0.90    # 90% repetibilidad mínima
        }
        
        # Configuración de validación cruzada
        self.default_k_folds = 10
        self.min_samples_per_fold = 20
        self.confidence_level = 0.95
        
        # Configuración de análisis estadístico
        self.alpha_level = 0.05  # Nivel de significancia
        
    def perform_k_fold_validation(self, 
                                 dataset: ValidationDataset,
                                 classifier_func: Callable,
                                 k_folds: Optional[int] = None,
                                 stratified: bool = True,
                                 validation_level: ValidationLevel = ValidationLevel.INTERNAL) -> ValidationResult:
        """
        Realiza validación cruzada k-fold según protocolos NIST
        
        Args:
            dataset: Dataset de validación
            classifier_func: Función clasificadora a validar
            k_folds: Número de folds (por defecto usa configuración)
            stratified: Si usar estratificación
            validation_level: Nivel de validación
            
        Returns:
            ValidationResult: Resultado completo de validación
        """
        try:
            # Configurar k-folds
            if k_folds is None:
                k_folds = min(self.default_k_folds, len(dataset.labels) // self.min_samples_per_fold)
            
            k_folds = max(3, min(k_folds, len(dataset.labels) // 2))
            
            # Configurar cross-validation
            if stratified and len(np.unique(dataset.labels)) > 1:
                cv = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
            else:
                cv = KFold(n_splits=k_folds, shuffle=True, random_state=42)
            
            # Almacenar resultados por fold
            fold_results = []
            confusion_matrices = []
            roc_curves = []
            
            # Realizar validación cruzada
            for fold_idx, (train_idx, test_idx) in enumerate(cv.split(dataset.features, dataset.labels)):
                X_train, X_test = dataset.features[train_idx], dataset.features[test_idx]
                y_train, y_test = dataset.labels[train_idx], dataset.labels[test_idx]
                
                # Entrenar y predecir
                try:
                    predictions, probabilities = classifier_func(X_train, y_train, X_test)
                    
                    # Calcular métricas para este fold
                    fold_metrics = self._calculate_fold_metrics(y_test, predictions, probabilities)
                    fold_results.append(fold_metrics)
                    
                    # Matriz de confusión
                    cm = confusion_matrix(y_test, predictions)
                    confusion_matrices.append(cm)
                    
                    # Curva ROC (si hay probabilidades)
                    if probabilities is not None and len(np.unique(y_test)) == 2:
                        fpr, tpr, _ = roc_curve(y_test, probabilities[:, 1] if probabilities.ndim > 1 else probabilities)
                        auc = roc_auc_score(y_test, probabilities[:, 1] if probabilities.ndim > 1 else probabilities)
                        roc_curves.append((fpr, tpr, auc))
                    
                except Exception as e:
                    print(f"Error en fold {fold_idx}: {e}")
                    continue
            
            if not fold_results:
                raise ValueError("No se pudieron completar los folds de validación")
            
            # Agregar resultados
            aggregated_metrics = self._aggregate_fold_results(fold_results)
            
            # Calcular intervalos de confianza
            confidence_intervals = self._calculate_confidence_intervals(fold_results)
            
            # Realizar pruebas estadísticas
            statistical_tests = self._perform_statistical_tests(fold_results, dataset)
            
            # Calcular métricas de confiabilidad
            reliability_metrics = self._calculate_reliability_metrics(fold_results, dataset)
            
            # Análisis de incertidumbre
            uncertainty_analysis = self._perform_uncertainty_analysis(fold_results, dataset)
            
            # Generar resumen y recomendaciones
            validation_summary, recommendations, is_valid = self._generate_validation_summary(
                aggregated_metrics, confidence_intervals, statistical_tests, reliability_metrics
            )
            
            return ValidationResult(
                validation_id=f"validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                validation_level=validation_level,
                validation_date=datetime.now(),
                dataset_size=len(dataset.labels),
                k_folds=k_folds,
                metrics=aggregated_metrics,
                confidence_intervals=confidence_intervals,
                statistical_tests=statistical_tests,
                cross_validation_scores=[result['accuracy'] for result in fold_results],
                confusion_matrices=confusion_matrices,
                roc_curves=roc_curves,
                reliability_metrics=reliability_metrics,
                uncertainty_analysis=uncertainty_analysis,
                validation_summary=validation_summary,
                recommendations=recommendations,
                is_valid=is_valid
            )
            
        except Exception as e:
            return self._create_error_validation_result(str(e), validation_level)
    
    def _calculate_fold_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                               y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Calcula métricas para un fold individual
        
        Args:
            y_true: Etiquetas verdaderas
            y_pred: Predicciones
            y_prob: Probabilidades (opcional)
            
        Returns:
            Dict: Métricas del fold
        """
        metrics = {}
        
        try:
            # Métricas básicas
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
            
            # Métricas para clasificación binaria y multiclase
            if len(np.unique(y_true)) == 2:
                # Clasificación binaria
                metrics['precision'] = precision_score(y_true, y_pred, average='binary')
                metrics['recall'] = recall_score(y_true, y_pred, average='binary')
                metrics['f1_score'] = f1_score(y_true, y_pred, average='binary')
                
                # Especificidad
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
                metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                
                # AUC-ROC si hay probabilidades
                if y_prob is not None:
                    try:
                        if y_prob.ndim > 1:
                            metrics['auc_roc'] = roc_auc_score(y_true, y_prob[:, 1])
                        else:
                            metrics['auc_roc'] = roc_auc_score(y_true, y_prob)
                    except ValueError:
                        metrics['auc_roc'] = 0.5  # AUC neutral si hay error
            else:
                # Clasificación multiclase
                metrics['precision'] = precision_score(y_true, y_pred, average='weighted')
                metrics['recall'] = recall_score(y_true, y_pred, average='weighted')
                metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted')
                metrics['specificity'] = 0.0  # No aplicable directamente a multiclase
                
                # AUC-ROC multiclase si hay probabilidades
                if y_prob is not None:
                    try:
                        metrics['auc_roc'] = roc_auc_score(y_true, y_prob, multi_class='ovr', average='weighted')
                    except ValueError:
                        metrics['auc_roc'] = 0.5
            
            return metrics
            
        except Exception as e:
            print(f"Error calculando métricas del fold: {e}")
            return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0, 'specificity': 0.0, 'auc_roc': 0.5}
    
    def _aggregate_fold_results(self, fold_results: List[Dict[str, float]]) -> Dict[str, float]:
        """
        Agrega resultados de todos los folds
        
        Args:
            fold_results: Lista de resultados por fold
            
        Returns:
            Dict: Métricas agregadas
        """
        aggregated = {}
        
        try:
            if not fold_results:
                return {}
            
            # Obtener todas las métricas disponibles
            all_metrics = set()
            for result in fold_results:
                all_metrics.update(result.keys())
            
            # Calcular estadísticas para cada métrica
            for metric in all_metrics:
                values = [result.get(metric, 0.0) for result in fold_results]
                
                aggregated[f'{metric}_mean'] = np.mean(values)
                aggregated[f'{metric}_std'] = np.std(values)
                aggregated[f'{metric}_min'] = np.min(values)
                aggregated[f'{metric}_max'] = np.max(values)
                aggregated[f'{metric}_median'] = np.median(values)
                
                # Métrica principal (promedio)
                aggregated[metric] = np.mean(values)
            
            return aggregated
            
        except Exception as e:
            print(f"Error agregando resultados: {e}")
            return {}
    
    def _calculate_confidence_intervals(self, fold_results: List[Dict[str, float]]) -> Dict[str, Tuple[float, float]]:
        """
        Calcula intervalos de confianza para las métricas
        
        Args:
            fold_results: Lista de resultados por fold
            
        Returns:
            Dict: Intervalos de confianza
        """
        confidence_intervals = {}
        
        try:
            if not fold_results:
                return {}
            
            # Obtener métricas principales
            main_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'specificity', 'auc_roc']
            
            for metric in main_metrics:
                values = [result.get(metric, 0.0) for result in fold_results if metric in result]
                
                if len(values) > 1:
                    # Calcular intervalo de confianza usando t-student
                    mean = np.mean(values)
                    std_err = stats.sem(values)  # Error estándar de la media
                    dof = len(values) - 1  # Grados de libertad
                    
                    # Intervalo de confianza
                    t_critical = stats.t.ppf((1 + self.confidence_level) / 2, dof)
                    margin_error = t_critical * std_err
                    
                    ci_lower = mean - margin_error
                    ci_upper = mean + margin_error
                    
                    confidence_intervals[metric] = (ci_lower, ci_upper)
                else:
                    # Si solo hay un valor, usar el valor mismo
                    value = values[0] if values else 0.0
                    confidence_intervals[metric] = (value, value)
            
            return confidence_intervals
            
        except Exception as e:
            print(f"Error calculando intervalos de confianza: {e}")
            return {}
    
    def _perform_statistical_tests(self, fold_results: List[Dict[str, float]], 
                                 dataset: ValidationDataset) -> Dict[str, Any]:
        """
        Realiza pruebas estadísticas de validación
        
        Args:
            fold_results: Resultados por fold
            dataset: Dataset de validación
            
        Returns:
            Dict: Resultados de pruebas estadísticas
        """
        tests = {}
        
        try:
            # Test de normalidad para las métricas
            main_metrics = ['accuracy', 'precision', 'recall', 'f1_score']
            
            for metric in main_metrics:
                values = [result.get(metric, 0.0) for result in fold_results if metric in result]
                
                if len(values) >= 3:
                    # Test de Shapiro-Wilk para normalidad
                    statistic, p_value = stats.shapiro(values)
                    tests[f'{metric}_normality'] = {
                        'test': 'shapiro_wilk',
                        'statistic': float(statistic),
                        'p_value': float(p_value),
                        'is_normal': p_value > self.alpha_level
                    }
                    
                    # Test t de una muestra contra umbral de aceptación
                    threshold = self.acceptance_thresholds.get(MetricType(metric), 0.5)
                    t_stat, t_p_value = stats.ttest_1samp(values, threshold)
                    tests[f'{metric}_threshold_test'] = {
                        'test': 'one_sample_t_test',
                        'threshold': threshold,
                        'statistic': float(t_stat),
                        'p_value': float(t_p_value),
                        'exceeds_threshold': np.mean(values) > threshold and t_p_value < self.alpha_level
                    }
            
            # Test de consistencia entre folds
            accuracy_values = [result.get('accuracy', 0.0) for result in fold_results]
            if len(accuracy_values) > 2:
                # Coeficiente de variación
                cv = np.std(accuracy_values) / np.mean(accuracy_values) if np.mean(accuracy_values) > 0 else float('inf')
                tests['consistency'] = {
                    'coefficient_of_variation': float(cv),
                    'is_consistent': cv < 0.1  # CV < 10% se considera consistente
                }
            
            # Test de balance del dataset
            unique_labels, counts = np.unique(dataset.labels, return_counts=True)
            if len(unique_labels) > 1:
                # Test chi-cuadrado de bondad de ajuste
                expected = np.full_like(counts, len(dataset.labels) / len(unique_labels))
                chi2_stat, chi2_p = stats.chisquare(counts, expected)
                tests['dataset_balance'] = {
                    'test': 'chi_square_goodness_of_fit',
                    'statistic': float(chi2_stat),
                    'p_value': float(chi2_p),
                    'is_balanced': chi2_p > self.alpha_level,
                    'class_distribution': dict(zip(unique_labels.tolist(), counts.tolist()))
                }
            
            return tests
            
        except Exception as e:
            print(f"Error en pruebas estadísticas: {e}")
            return {}
    
    def _calculate_reliability_metrics(self, fold_results: List[Dict[str, float]], 
                                     dataset: ValidationDataset) -> Dict[str, float]:
        """
        Calcula métricas de confiabilidad
        
        Args:
            fold_results: Resultados por fold
            dataset: Dataset de validación
            
        Returns:
            Dict: Métricas de confiabilidad
        """
        reliability = {}
        
        try:
            # Reproducibilidad (consistencia entre folds)
            accuracy_values = [result.get('accuracy', 0.0) for result in fold_results]
            if len(accuracy_values) > 1:
                mean_accuracy = np.mean(accuracy_values)
                std_accuracy = np.std(accuracy_values)
                
                # Reproducibilidad como 1 - CV (coeficiente de variación)
                cv = std_accuracy / mean_accuracy if mean_accuracy > 0 else 1.0
                reliability['reproducibility'] = max(0.0, 1.0 - cv)
                
                # Repetibilidad (basada en la varianza intra-fold)
                reliability['repeatability'] = max(0.0, 1.0 - (std_accuracy / 0.1))  # Normalizado
            
            # Estabilidad del modelo
            precision_values = [result.get('precision', 0.0) for result in fold_results]
            recall_values = [result.get('recall', 0.0) for result in fold_results]
            
            if len(precision_values) > 1 and len(recall_values) > 1:
                # Estabilidad como inverso de la varianza promedio
                precision_var = np.var(precision_values)
                recall_var = np.var(recall_values)
                avg_variance = (precision_var + recall_var) / 2
                reliability['stability'] = max(0.0, 1.0 - (avg_variance * 10))  # Escalado
            
            # Robustez (basada en el rango de métricas)
            f1_values = [result.get('f1_score', 0.0) for result in fold_results]
            if len(f1_values) > 1:
                f1_range = np.max(f1_values) - np.min(f1_values)
                reliability['robustness'] = max(0.0, 1.0 - (f1_range * 2))  # Escalado
            
            # Confiabilidad general (promedio ponderado)
            if reliability:
                weights = {'reproducibility': 0.4, 'repeatability': 0.3, 'stability': 0.2, 'robustness': 0.1}
                total_reliability = 0.0
                total_weight = 0.0
                
                for metric, weight in weights.items():
                    if metric in reliability:
                        total_reliability += reliability[metric] * weight
                        total_weight += weight
                
                if total_weight > 0:
                    reliability['overall_reliability'] = total_reliability / total_weight
            
            return reliability
            
        except Exception as e:
            print(f"Error calculando confiabilidad: {e}")
            return {}
    
    def _perform_uncertainty_analysis(self, fold_results: List[Dict[str, float]], 
                                    dataset: ValidationDataset) -> Dict[str, Any]:
        """
        Realiza análisis de incertidumbre
        
        Args:
            fold_results: Resultados por fold
            dataset: Dataset de validación
            
        Returns:
            Dict: Análisis de incertidumbre
        """
        uncertainty = {}
        
        try:
            # Incertidumbre tipo A (estadística)
            accuracy_values = [result.get('accuracy', 0.0) for result in fold_results]
            if len(accuracy_values) > 1:
                # Incertidumbre estándar
                standard_uncertainty = np.std(accuracy_values) / np.sqrt(len(accuracy_values))
                uncertainty['type_a_uncertainty'] = float(standard_uncertainty)
                
                # Incertidumbre expandida (k=2 para ~95% confianza)
                expanded_uncertainty = 2 * standard_uncertainty
                uncertainty['expanded_uncertainty'] = float(expanded_uncertainty)
            
            # Incertidumbre tipo B (sistemática)
            # Basada en factores como tamaño del dataset, balance de clases, etc.
            dataset_size_factor = min(1.0, len(dataset.labels) / 1000)  # Factor de tamaño
            
            unique_labels, counts = np.unique(dataset.labels, return_counts=True)
            balance_factor = min(counts) / max(counts) if len(counts) > 1 else 1.0  # Factor de balance
            
            # Incertidumbre sistemática estimada
            systematic_uncertainty = (1.0 - dataset_size_factor) * 0.05 + (1.0 - balance_factor) * 0.03
            uncertainty['type_b_uncertainty'] = float(systematic_uncertainty)
            
            # Incertidumbre combinada
            if 'type_a_uncertainty' in uncertainty:
                combined_uncertainty = np.sqrt(uncertainty['type_a_uncertainty']**2 + systematic_uncertainty**2)
                uncertainty['combined_uncertainty'] = float(combined_uncertainty)
            
            # Análisis de sensibilidad
            metric_variations = {}
            for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
                values = [result.get(metric, 0.0) for result in fold_results if metric in result]
                if len(values) > 1:
                    metric_variations[metric] = {
                        'coefficient_of_variation': float(np.std(values) / np.mean(values)) if np.mean(values) > 0 else 0.0,
                        'range': float(np.max(values) - np.min(values)),
                        'interquartile_range': float(np.percentile(values, 75) - np.percentile(values, 25))
                    }
            
            uncertainty['sensitivity_analysis'] = metric_variations
            
            return uncertainty
            
        except Exception as e:
            print(f"Error en análisis de incertidumbre: {e}")
            return {}
    
    def _generate_validation_summary(self, metrics: Dict[str, float],
                                   confidence_intervals: Dict[str, Tuple[float, float]],
                                   statistical_tests: Dict[str, Any],
                                   reliability_metrics: Dict[str, float]) -> Tuple[str, List[str], bool]:
        """
        Genera resumen de validación y recomendaciones
        
        Args:
            metrics: Métricas agregadas
            confidence_intervals: Intervalos de confianza
            statistical_tests: Pruebas estadísticas
            reliability_metrics: Métricas de confiabilidad
            
        Returns:
            Tuple: (resumen, recomendaciones, es_válido)
        """
        summary_parts = []
        recommendations = []
        is_valid = True
        
        try:
            # Resumen de métricas principales
            main_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'specificity']
            
            summary_parts.append("=== RESUMEN DE VALIDACIÓN NIST ===")
            
            for metric in main_metrics:
                if metric in metrics:
                    value = metrics[metric]
                    threshold = self.acceptance_thresholds.get(MetricType(metric), 0.5)
                    
                    # Intervalo de confianza
                    ci_text = ""
                    if metric in confidence_intervals:
                        ci_lower, ci_upper = confidence_intervals[metric]
                        ci_text = f" (IC 95%: {ci_lower:.3f}-{ci_upper:.3f})"
                    
                    # Estado de cumplimiento
                    status = "✓ CUMPLE" if value >= threshold else "✗ NO CUMPLE"
                    if value < threshold:
                        is_valid = False
                    
                    summary_parts.append(f"{metric.upper()}: {value:.3f}{ci_text} - Umbral: {threshold:.3f} - {status}")
            
            # Resumen de confiabilidad
            if reliability_metrics:
                summary_parts.append("\n=== MÉTRICAS DE CONFIABILIDAD ===")
                for metric, value in reliability_metrics.items():
                    summary_parts.append(f"{metric.upper()}: {value:.3f}")
            
            # Resumen de pruebas estadísticas
            summary_parts.append("\n=== PRUEBAS ESTADÍSTICAS ===")
            
            # Consistencia
            if 'consistency' in statistical_tests:
                cv = statistical_tests['consistency']['coefficient_of_variation']
                is_consistent = statistical_tests['consistency']['is_consistent']
                status = "✓ CONSISTENTE" if is_consistent else "✗ INCONSISTENTE"
                summary_parts.append(f"Consistencia entre folds: CV={cv:.3f} - {status}")
                
                if not is_consistent:
                    recommendations.append("Mejorar consistencia entre folds - considerar más datos o mejor preprocesamiento")
            
            # Balance del dataset
            if 'dataset_balance' in statistical_tests:
                is_balanced = statistical_tests['dataset_balance']['is_balanced']
                status = "✓ BALANCEADO" if is_balanced else "✗ DESBALANCEADO"
                summary_parts.append(f"Balance del dataset: {status}")
                
                if not is_balanced:
                    recommendations.append("Considerar técnicas de balanceamiento de clases")
            
            # Generar recomendaciones específicas
            for metric in main_metrics:
                if metric in metrics:
                    value = metrics[metric]
                    threshold = self.acceptance_thresholds.get(MetricType(metric), 0.5)
                    
                    if value < threshold:
                        if metric == 'accuracy':
                            recommendations.append("Mejorar exactitud: revisar algoritmo de clasificación y características")
                        elif metric == 'precision':
                            recommendations.append("Reducir falsos positivos: ajustar umbrales de decisión")
                        elif metric == 'recall':
                            recommendations.append("Reducir falsos negativos: mejorar sensibilidad del modelo")
                        elif metric == 'specificity':
                            recommendations.append("Mejorar especificidad: reducir falsos positivos")
            
            # Recomendaciones de confiabilidad
            if reliability_metrics.get('overall_reliability', 1.0) < 0.8:
                recommendations.append("Mejorar confiabilidad general del sistema")
            
            if reliability_metrics.get('reproducibility', 1.0) < 0.85:
                recommendations.append("Mejorar reproducibilidad: estandarizar procedimientos")
            
            # Recomendaciones generales
            if is_valid:
                recommendations.append("Sistema validado según estándares NIST")
                summary_parts.append("\n✓ VALIDACIÓN EXITOSA - Sistema cumple estándares NIST")
            else:
                recommendations.append("Sistema requiere mejoras antes de uso operacional")
                summary_parts.append("\n✗ VALIDACIÓN FALLIDA - Sistema no cumple estándares NIST")
            
            summary = "\n".join(summary_parts)
            
            return summary, recommendations, is_valid
            
        except Exception as e:
            error_summary = f"Error generando resumen: {e}"
            error_recommendations = ["Revisar proceso de validación"]
            return error_summary, error_recommendations, False
    
    def _create_error_validation_result(self, error_msg: str, 
                                      validation_level: ValidationLevel) -> ValidationResult:
        """
        Crea resultado de validación de error
        
        Args:
            error_msg: Mensaje de error
            validation_level: Nivel de validación
            
        Returns:
            ValidationResult: Resultado de error
        """
        return ValidationResult(
            validation_id=f"error_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            validation_level=validation_level,
            validation_date=datetime.now(),
            dataset_size=0,
            k_folds=0,
            metrics={},
            confidence_intervals={},
            statistical_tests={},
            cross_validation_scores=[],
            confusion_matrices=[],
            roc_curves=[],
            reliability_metrics={},
            uncertainty_analysis={},
            validation_summary=f"Error en validación: {error_msg}",
            recommendations=["Revisar configuración de validación", "Verificar datos de entrada"],
            is_valid=False
        )
    
    def generate_validation_report(self, result: ValidationResult, 
                                 output_path: str, include_plots: bool = True) -> bool:
        """
        Genera reporte completo de validación
        
        Args:
            result: Resultado de validación
            output_path: Ruta de salida
            include_plots: Si incluir gráficos
            
        Returns:
            bool: True si se generó correctamente
        """
        try:
            # Crear reporte en formato JSON
            report_data = result.to_dict()
            
            # Agregar metadatos del reporte
            report_data['report_metadata'] = {
                'generated_at': datetime.now().isoformat(),
                'nist_compliance': result.is_valid,
                'validation_standard': 'NIST SP 800-101 Rev. 1',
                'confidence_level': self.confidence_level
            }
            
            # Guardar reporte JSON
            json_path = output_path.replace('.json', '') + '.json'
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)
            
            # Generar gráficos si se solicita
            if include_plots and result.cross_validation_scores:
                self._generate_validation_plots(result, output_path.replace('.json', ''))
            
            return True
            
        except Exception as e:
            print(f"Error generando reporte: {e}")
            return False
    
    def _generate_validation_plots(self, result: ValidationResult, base_path: str):
        """
        Genera gráficos de validación
        
        Args:
            result: Resultado de validación
            base_path: Ruta base para archivos
        """
        try:
            # Configurar matplotlib para backend no interactivo
            plt.switch_backend('Agg')
            
            # Gráfico de scores de cross-validation
            if result.cross_validation_scores:
                plt.figure(figsize=(10, 6))
                plt.plot(range(1, len(result.cross_validation_scores) + 1), 
                        result.cross_validation_scores, 'bo-', linewidth=2, markersize=8)
                plt.axhline(y=np.mean(result.cross_validation_scores), color='r', 
                           linestyle='--', label=f'Media: {np.mean(result.cross_validation_scores):.3f}')
                plt.xlabel('Fold')
                plt.ylabel('Accuracy')
                plt.title('Cross-Validation Scores')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(f'{base_path}_cv_scores.png', dpi=300, bbox_inches='tight')
                plt.close()
            
            # Matriz de confusión promedio
            if result.confusion_matrices:
                avg_cm = np.mean(result.confusion_matrices, axis=0)
                plt.figure(figsize=(8, 6))
                sns.heatmap(avg_cm, annot=True, fmt='.1f', cmap='Blues')
                plt.title('Matriz de Confusión Promedio')
                plt.ylabel('Etiqueta Verdadera')
                plt.xlabel('Etiqueta Predicha')
                plt.tight_layout()
                plt.savefig(f'{base_path}_confusion_matrix.png', dpi=300, bbox_inches='tight')
                plt.close()
            
            # Curvas ROC
            if result.roc_curves:
                plt.figure(figsize=(8, 8))
                for i, (fpr, tpr, auc) in enumerate(result.roc_curves):
                    plt.plot(fpr, tpr, alpha=0.7, label=f'Fold {i+1} (AUC = {auc:.3f})')
                
                # ROC promedio
                mean_fpr = np.linspace(0, 1, 100)
                tprs = []
                for fpr, tpr, _ in result.roc_curves:
                    tprs.append(np.interp(mean_fpr, fpr, tpr))
                
                mean_tpr = np.mean(tprs, axis=0)
                mean_auc = np.mean([auc for _, _, auc in result.roc_curves])
                
                plt.plot(mean_fpr, mean_tpr, 'b-', linewidth=3, 
                        label=f'ROC Promedio (AUC = {mean_auc:.3f})')
                plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
                plt.xlabel('Tasa de Falsos Positivos')
                plt.ylabel('Tasa de Verdaderos Positivos')
                plt.title('Curvas ROC - Cross-Validation')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(f'{base_path}_roc_curves.png', dpi=300, bbox_inches='tight')
                plt.close()
            
        except Exception as e:
            print(f"Error generando gráficos: {e}")
    
    def compare_validation_results(self, results: List[ValidationResult]) -> Dict[str, Any]:
        """
        Compara múltiples resultados de validación
        
        Args:
            results: Lista de resultados de validación
            
        Returns:
            Dict: Comparación detallada
        """
        try:
            if len(results) < 2:
                return {'error': 'Se requieren al menos 2 resultados para comparar'}
            
            comparison = {
                'num_results': len(results),
                'comparison_date': datetime.now().isoformat(),
                'metric_comparison': {},
                'best_result': None,
                'recommendations': []
            }
            
            # Comparar métricas principales
            main_metrics = ['accuracy', 'precision', 'recall', 'f1_score']
            
            for metric in main_metrics:
                values = []
                result_ids = []
                
                for result in results:
                    if metric in result.metrics:
                        values.append(result.metrics[metric])
                        result_ids.append(result.validation_id)
                
                if values:
                    comparison['metric_comparison'][metric] = {
                        'values': dict(zip(result_ids, values)),
                        'best_result_id': result_ids[np.argmax(values)],
                        'worst_result_id': result_ids[np.argmin(values)],
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values)),
                        'range': float(np.max(values) - np.min(values))
                    }
            
            # Determinar mejor resultado general
            overall_scores = []
            for result in results:
                score = 0.0
                count = 0
                for metric in main_metrics:
                    if metric in result.metrics:
                        score += result.metrics[metric]
                        count += 1
                
                overall_scores.append(score / count if count > 0 else 0.0)
            
            if overall_scores:
                best_idx = np.argmax(overall_scores)
                comparison['best_result'] = {
                    'validation_id': results[best_idx].validation_id,
                    'overall_score': float(overall_scores[best_idx]),
                    'is_valid': results[best_idx].is_valid
                }
            
            # Generar recomendaciones comparativas
            valid_results = [r for r in results if r.is_valid]
            
            if len(valid_results) == 0:
                comparison['recommendations'].append("Ningún resultado cumple estándares NIST")
            elif len(valid_results) < len(results):
                comparison['recommendations'].append(f"Solo {len(valid_results)}/{len(results)} resultados cumplen estándares")
            else:
                comparison['recommendations'].append("Todos los resultados cumplen estándares NIST")
            
            return comparison
            
        except Exception as e:
            return {'error': f'Error comparando resultados: {e}'}
    
    def continuous_validation_monitoring(self, validation_history: List[ValidationResult]) -> Dict[str, Any]:
        """
        Monitoreo continuo de validación para uso operacional
        
        Args:
            validation_history: Historial de validaciones
            
        Returns:
            Dict: Estado del monitoreo
        """
        try:
            if not validation_history:
                return {'status': 'no_data', 'message': 'No hay datos de validación'}
            
            # Analizar tendencias
            recent_results = validation_history[-10:]  # Últimos 10 resultados
            
            monitoring = {
                'status': 'monitoring',
                'last_validation': recent_results[-1].validation_date.isoformat(),
                'trend_analysis': {},
                'alerts': [],
                'recommendations': []
            }
            
            # Análisis de tendencias para métricas principales
            main_metrics = ['accuracy', 'precision', 'recall', 'f1_score']
            
            for metric in main_metrics:
                values = []
                dates = []
                
                for result in recent_results:
                    if metric in result.metrics:
                        values.append(result.metrics[metric])
                        dates.append(result.validation_date)
                
                if len(values) >= 3:
                    # Calcular tendencia (regresión lineal simple)
                    x = np.arange(len(values))
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
                    
                    monitoring['trend_analysis'][metric] = {
                        'slope': float(slope),
                        'r_squared': float(r_value**2),
                        'p_value': float(p_value),
                        'trend': 'improving' if slope > 0.001 else 'declining' if slope < -0.001 else 'stable',
                        'current_value': float(values[-1]),
                        'mean_value': float(np.mean(values))
                    }
                    
                    # Generar alertas
                    threshold = self.acceptance_thresholds.get(MetricType(metric), 0.5)
                    
                    if values[-1] < threshold:
                        monitoring['alerts'].append(f"ALERTA: {metric} por debajo del umbral ({values[-1]:.3f} < {threshold:.3f})")
                    
                    if slope < -0.01 and p_value < 0.05:
                        monitoring['alerts'].append(f"TENDENCIA NEGATIVA: {metric} está declinando significativamente")
            
            # Estado general del sistema
            latest_result = recent_results[-1]
            if not latest_result.is_valid:
                monitoring['status'] = 'critical'
                monitoring['alerts'].append("CRÍTICO: Última validación no cumple estándares NIST")
            elif len(monitoring['alerts']) > 0:
                monitoring['status'] = 'warning'
            else:
                monitoring['status'] = 'healthy'
            
            # Recomendaciones
            if monitoring['status'] == 'critical':
                monitoring['recommendations'].append("Suspender uso operacional hasta resolver problemas")
            elif monitoring['status'] == 'warning':
                monitoring['recommendations'].append("Investigar causas de las alertas generadas")
                monitoring['recommendations'].append("Considerar recalibración del sistema")
            else:
                monitoring['recommendations'].append("Sistema operando dentro de parámetros normales")
            
            return monitoring
            
        except Exception as e:
            return {'status': 'error', 'message': f'Error en monitoreo: {e}'}