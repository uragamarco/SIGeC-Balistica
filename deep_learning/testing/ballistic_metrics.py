"""
Métricas Especializadas para Análisis Balístico
==============================================

Este módulo implementa métricas especializadas para la evaluación
de modelos de deep learning en análisis balístico, incluyendo:
- Curvas CMC (Cumulative Match Characteristic)
- Análisis ROC para matching
- Métricas de clasificación jerárquica
- Análisis de similitud y distancia

Autor: SIGeC-BalisticaTeam
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, 
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class MetricResult:
    """Resultado de una métrica específica"""
    name: str
    value: float
    confidence_interval: Optional[Tuple[float, float]] = None
    metadata: Optional[Dict] = None

class BallisticMetrics:
    """
    Clase principal para métricas balísticas especializadas
    """
    
    def __init__(self, save_plots: bool = True, output_dir: str = "metrics_output"):
        self.save_plots = save_plots
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Configurar estilo de plots
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def compute_all_metrics(self, 
                          y_true: np.ndarray,
                          y_pred: np.ndarray,
                          y_scores: Optional[np.ndarray] = None,
                          class_names: Optional[List[str]] = None) -> Dict[str, MetricResult]:
        """
        Computa todas las métricas disponibles
        
        Args:
            y_true: Etiquetas verdaderas
            y_pred: Predicciones del modelo
            y_scores: Scores de confianza (opcional)
            class_names: Nombres de las clases (opcional)
            
        Returns:
            Diccionario con todos los resultados de métricas
        """
        results = {}
        
        # Métricas básicas de clasificación
        results.update(self._compute_classification_metrics(y_true, y_pred))
        
        # Métricas con scores si están disponibles
        if y_scores is not None:
            results.update(self._compute_score_based_metrics(y_true, y_scores))
        
        # Análisis de confusión
        results['confusion_analysis'] = self._analyze_confusion_matrix(
            y_true, y_pred, class_names
        )
        
        return results
    
    def _compute_classification_metrics(self, 
                                     y_true: np.ndarray, 
                                     y_pred: np.ndarray) -> Dict[str, MetricResult]:
        """Computa métricas básicas de clasificación"""
        results = {}
        
        # Accuracy
        acc = accuracy_score(y_true, y_pred)
        results['accuracy'] = MetricResult('Accuracy', acc)
        
        # Precision, Recall, F1 (macro y weighted)
        for avg in ['macro', 'weighted']:
            prec = precision_score(y_true, y_pred, average=avg, zero_division=0)
            rec = recall_score(y_true, y_pred, average=avg, zero_division=0)
            f1 = f1_score(y_true, y_pred, average=avg, zero_division=0)
            
            results[f'precision_{avg}'] = MetricResult(f'Precision ({avg})', prec)
            results[f'recall_{avg}'] = MetricResult(f'Recall ({avg})', rec)
            results[f'f1_{avg}'] = MetricResult(f'F1-Score ({avg})', f1)
        
        return results
    
    def _compute_score_based_metrics(self, 
                                   y_true: np.ndarray, 
                                   y_scores: np.ndarray) -> Dict[str, MetricResult]:
        """Computa métricas basadas en scores de confianza"""
        results = {}
        
        # Determinar si es binario o multiclase
        n_classes = len(np.unique(y_true))
        
        if n_classes == 2:
            # Caso binario
            fpr, tpr, _ = roc_curve(y_true, y_scores[:, 1] if y_scores.ndim > 1 else y_scores)
            roc_auc = auc(fpr, tpr)
            results['roc_auc'] = MetricResult('ROC AUC', roc_auc)
            
            # Precision-Recall curve
            precision, recall, _ = precision_recall_curve(
                y_true, y_scores[:, 1] if y_scores.ndim > 1 else y_scores
            )
            pr_auc = auc(recall, precision)
            results['pr_auc'] = MetricResult('PR AUC', pr_auc)
            
        else:
            # Caso multiclase - ROC AUC macro y micro
            y_true_bin = label_binarize(y_true, classes=range(n_classes))
            
            # Macro AUC
            macro_auc = 0
            for i in range(n_classes):
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
                macro_auc += auc(fpr, tpr)
            macro_auc /= n_classes
            
            results['roc_auc_macro'] = MetricResult('ROC AUC (Macro)', macro_auc)
        
        return results
    
    def _analyze_confusion_matrix(self, 
                                y_true: np.ndarray, 
                                y_pred: np.ndarray,
                                class_names: Optional[List[str]] = None) -> MetricResult:
        """Analiza la matriz de confusión"""
        cm = confusion_matrix(y_true, y_pred)
        
        # Normalizar matriz de confusión
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Calcular métricas por clase
        per_class_accuracy = np.diag(cm_normalized)
        
        # Crear visualización si se requiere
        if self.save_plots:
            self._plot_confusion_matrix(cm, cm_normalized, class_names)
        
        metadata = {
            'confusion_matrix': cm.tolist(),
            'normalized_cm': cm_normalized.tolist(),
            'per_class_accuracy': per_class_accuracy.tolist()
        }
        
        return MetricResult(
            'Confusion Matrix Analysis', 
            np.mean(per_class_accuracy),
            metadata=metadata
        )
    
    def _plot_confusion_matrix(self, 
                             cm: np.ndarray, 
                             cm_normalized: np.ndarray,
                             class_names: Optional[List[str]] = None):
        """Crea visualización de matriz de confusión"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Matriz de confusión absoluta
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1)
        ax1.set_title('Matriz de Confusión (Absoluta)')
        ax1.set_ylabel('Etiqueta Verdadera')
        ax1.set_xlabel('Predicción')
        
        # Matriz de confusión normalizada
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', ax=ax2)
        ax2.set_title('Matriz de Confusión (Normalizada)')
        ax2.set_ylabel('Etiqueta Verdadera')
        ax2.set_xlabel('Predicción')
        
        if class_names:
            for ax in [ax1, ax2]:
                ax.set_xticklabels(class_names, rotation=45)
                ax.set_yticklabels(class_names, rotation=0)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()

class CMCCurve:
    """
    Implementación de Curvas CMC (Cumulative Match Characteristic)
    Especialmente útil para análisis de matching balístico
    """
    
    def __init__(self, max_rank: int = 50):
        self.max_rank = max_rank
    
    def compute_cmc(self, 
                   similarities: np.ndarray, 
                   true_matches: np.ndarray,
                   gallery_labels: np.ndarray,
                   probe_labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computa la curva CMC
        
        Args:
            similarities: Matriz de similitudes (probe x gallery)
            true_matches: Matriz booleana de matches verdaderos
            gallery_labels: Etiquetas del gallery
            probe_labels: Etiquetas de las probes
            
        Returns:
            ranks: Array de ranks (1 a max_rank)
            cmc_scores: Scores CMC para cada rank
        """
        n_probes = similarities.shape[0]
        ranks = np.arange(1, self.max_rank + 1)
        cmc_scores = np.zeros(self.max_rank)
        
        for i in range(n_probes):
            # Ordenar por similitud descendente
            sorted_indices = np.argsort(similarities[i])[::-1]
            
            # Encontrar el rank del primer match correcto
            probe_label = probe_labels[i]
            for rank, gallery_idx in enumerate(sorted_indices[:self.max_rank]):
                if gallery_labels[gallery_idx] == probe_label:
                    # Match encontrado en este rank
                    cmc_scores[rank:] += 1
                    break
        
        # Normalizar por número de probes
        cmc_scores = cmc_scores / n_probes
        
        return ranks, cmc_scores
    
    def plot_cmc_curve(self, 
                      ranks: np.ndarray, 
                      cmc_scores: np.ndarray,
                      title: str = "CMC Curve",
                      save_path: Optional[Path] = None):
        """Visualiza la curva CMC"""
        plt.figure(figsize=(10, 6))
        plt.plot(ranks, cmc_scores, 'b-', linewidth=2, marker='o', markersize=4)
        plt.xlabel('Rank')
        plt.ylabel('Cumulative Match Accuracy')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.xlim(1, len(ranks))
        plt.ylim(0, 1)
        
        # Añadir anotaciones para ranks importantes
        for rank in [1, 5, 10, 20]:
            if rank <= len(ranks):
                score = cmc_scores[rank-1]
                plt.annotate(f'Rank-{rank}: {score:.3f}', 
                           xy=(rank, score), 
                           xytext=(rank+2, score+0.05),
                           arrowprops=dict(arrowstyle='->', alpha=0.7))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()

class ROCAnalysis:
    """
    Análisis ROC especializado para matching balístico
    """
    
    def __init__(self):
        self.curves = {}
    
    def compute_roc_matching(self, 
                           similarities: np.ndarray,
                           true_matches: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Computa curva ROC para matching
        
        Args:
            similarities: Scores de similitud
            true_matches: Etiquetas binarias de match verdadero
            
        Returns:
            fpr: False Positive Rate
            tpr: True Positive Rate  
            auc_score: Area Under Curve
        """
        fpr, tpr, thresholds = roc_curve(true_matches.flatten(), similarities.flatten())
        auc_score = auc(fpr, tpr)
        
        self.curves['matching'] = {
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds,
            'auc': auc_score
        }
        
        return fpr, tpr, auc_score
    
    def find_optimal_threshold(self, 
                             fpr: np.ndarray, 
                             tpr: np.ndarray, 
                             thresholds: np.ndarray,
                             method: str = 'youden') -> float:
        """
        Encuentra el threshold óptimo usando diferentes criterios
        
        Args:
            fpr, tpr, thresholds: Resultados de ROC curve
            method: 'youden', 'closest_to_topleft', 'f1_optimal'
            
        Returns:
            Threshold óptimo
        """
        if method == 'youden':
            # Youden's J statistic
            j_scores = tpr - fpr
            optimal_idx = np.argmax(j_scores)
        elif method == 'closest_to_topleft':
            # Punto más cercano a (0,1)
            distances = np.sqrt((fpr - 0)**2 + (tpr - 1)**2)
            optimal_idx = np.argmin(distances)
        else:
            # Default: Youden
            j_scores = tpr - fpr
            optimal_idx = np.argmax(j_scores)
        
        return thresholds[optimal_idx]
    
    def plot_roc_curves(self, 
                       curves_data: Dict[str, Dict],
                       title: str = "ROC Curves Comparison",
                       save_path: Optional[Path] = None):
        """Visualiza múltiples curvas ROC"""
        plt.figure(figsize=(10, 8))
        
        for name, data in curves_data.items():
            plt.plot(data['fpr'], data['tpr'], 
                    linewidth=2, label=f'{name} (AUC = {data["auc"]:.3f})')
        
        # Línea diagonal de referencia
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
        
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()

class MatchingMetrics:
    """
    Métricas especializadas para evaluación de matching balístico
    """
    
    def __init__(self):
        self.results = {}
    
    def compute_matching_accuracy(self, 
                                similarities: np.ndarray,
                                true_matches: np.ndarray,
                                threshold: float) -> Dict[str, float]:
        """
        Computa métricas de accuracy para matching
        
        Args:
            similarities: Matriz de similitudes
            true_matches: Matriz de matches verdaderos
            threshold: Threshold para decisión de match
            
        Returns:
            Diccionario con métricas de matching
        """
        # Convertir a predicciones binarias
        predictions = (similarities >= threshold).astype(int)
        
        # Calcular métricas
        tp = np.sum((predictions == 1) & (true_matches == 1))
        tn = np.sum((predictions == 0) & (true_matches == 0))
        fp = np.sum((predictions == 1) & (true_matches == 0))
        fn = np.sum((predictions == 0) & (true_matches == 1))
        
        # Métricas derivadas
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # False Match Rate y False Non-Match Rate
        fmr = fp / (tn + fp) if (tn + fp) > 0 else 0  # False Match Rate
        fnmr = fn / (tp + fn) if (tp + fn) > 0 else 0  # False Non-Match Rate
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'f1_score': f1,
            'false_match_rate': fmr,
            'false_non_match_rate': fnmr,
            'true_positives': tp,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn
        }
        
        return results
    
    def compute_eer(self, 
                   similarities: np.ndarray,
                   true_matches: np.ndarray) -> Tuple[float, float]:
        """
        Computa Equal Error Rate (EER)
        
        Args:
            similarities: Scores de similitud
            true_matches: Etiquetas de match verdadero
            
        Returns:
            eer: Equal Error Rate
            eer_threshold: Threshold en el EER
        """
        fpr, tpr, thresholds = roc_curve(true_matches.flatten(), similarities.flatten())
        fnr = 1 - tpr
        
        # Encontrar punto donde FPR ≈ FNR
        eer_idx = np.argmin(np.abs(fpr - fnr))
        eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
        eer_threshold = thresholds[eer_idx]
        
        return eer, eer_threshold

class ClassificationMetrics:
    """
    Métricas especializadas para clasificación jerárquica balística
    """
    
    def __init__(self):
        self.hierarchy_levels = ['manufacturer', 'model', 'caliber', 'individual']
    
    def compute_hierarchical_accuracy(self, 
                                    predictions: Dict[str, np.ndarray],
                                    true_labels: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Computa accuracy jerárquica para diferentes niveles
        
        Args:
            predictions: Predicciones por nivel jerárquico
            true_labels: Etiquetas verdaderas por nivel
            
        Returns:
            Accuracy por cada nivel jerárquico
        """
        results = {}
        
        for level in self.hierarchy_levels:
            if level in predictions and level in true_labels:
                acc = accuracy_score(true_labels[level], predictions[level])
                results[f'{level}_accuracy'] = acc
        
        # Accuracy jerárquica promedio
        if results:
            results['hierarchical_mean_accuracy'] = np.mean(list(results.values()))
        
        return results
    
    def compute_top_k_accuracy(self, 
                             scores: np.ndarray,
                             true_labels: np.ndarray,
                             k_values: List[int] = [1, 3, 5, 10]) -> Dict[str, float]:
        """
        Computa Top-K accuracy
        
        Args:
            scores: Scores de predicción (n_samples x n_classes)
            true_labels: Etiquetas verdaderas
            k_values: Valores de K para evaluar
            
        Returns:
            Top-K accuracy para cada valor de K
        """
        results = {}
        
        # Obtener top-k predicciones
        top_k_preds = np.argsort(scores, axis=1)[:, ::-1]
        
        for k in k_values:
            if k <= scores.shape[1]:
                # Verificar si la etiqueta verdadera está en top-k
                top_k_correct = np.any(top_k_preds[:, :k] == true_labels.reshape(-1, 1), axis=1)
                top_k_acc = np.mean(top_k_correct)
                results[f'top_{k}_accuracy'] = top_k_acc
        
        return results

# Función de utilidad para testing rápido
def quick_test():
    """Test rápido de las métricas implementadas"""
    print("🧪 Testing Ballistic Metrics...")
    
    # Datos sintéticos para testing
    np.random.seed(42)
    n_samples = 1000
    n_classes = 5
    
    # Generar datos sintéticos
    y_true = np.random.randint(0, n_classes, n_samples)
    y_scores = np.random.rand(n_samples, n_classes)
    y_pred = np.argmax(y_scores, axis=1)
    
    # Test métricas básicas
    metrics = BallisticMetrics(save_plots=False)
    results = metrics.compute_all_metrics(y_true, y_pred, y_scores)
    
    print("✅ Métricas básicas computadas:")
    for name, result in results.items():
        if hasattr(result, 'value'):
            print(f"  - {result.name}: {result.value:.3f}")
    
    # Test CMC curve
    cmc = CMCCurve(max_rank=20)
    similarities = np.random.rand(100, 200)
    gallery_labels = np.random.randint(0, 50, 200)
    probe_labels = np.random.randint(0, 50, 100)
    
    ranks, cmc_scores = cmc.compute_cmc(similarities, None, gallery_labels, probe_labels)
    print(f"✅ CMC Curve - Rank-1 Accuracy: {cmc_scores[0]:.3f}")
    
    # Test ROC Analysis
    roc = ROCAnalysis()
    similarities_flat = np.random.rand(1000)
    matches_flat = np.random.randint(0, 2, 1000)
    
    fpr, tpr, auc_score = roc.compute_roc_matching(similarities_flat, matches_flat)
    print(f"✅ ROC Analysis - AUC: {auc_score:.3f}")
    
    # Test Matching Metrics
    matching = MatchingMetrics()
    threshold = 0.5
    match_results = matching.compute_matching_accuracy(
        similarities_flat.reshape(-1, 1), 
        matches_flat.reshape(-1, 1), 
        threshold
    )
    print(f"✅ Matching Metrics - F1 Score: {match_results['f1_score']:.3f}")
    
    print("🎉 Todos los tests completados exitosamente!")

if __name__ == "__main__":
    quick_test()