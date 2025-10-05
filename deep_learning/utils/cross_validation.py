"""
Módulo de validación cruzada y métricas de evaluación para modelos balísticos
Incluye K-Fold, Stratified K-Fold y métricas específicas para análisis balístico
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    precision_recall_curve, roc_curve, auc
)
import seaborn as sns
from datetime import datetime
import logging

class CrossValidator:
    """Clase para realizar validación cruzada en modelos balísticos"""
    
    def __init__(self, n_splits: int = 5, stratified: bool = True, random_state: int = 42):
        """
        Inicializar validador cruzado
        
        Args:
            n_splits: Número de folds para la validación cruzada
            stratified: Si usar estratificación para mantener distribución de clases
            random_state: Semilla para reproducibilidad
        """
        self.n_splits = n_splits
        self.stratified = stratified
        self.random_state = random_state
        self.logger = logging.getLogger(__name__)
        
        if stratified:
            self.cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        else:
            self.cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        
        self.results = {
            'fold_results': [],
            'mean_metrics': {},
            'std_metrics': {},
            'best_fold': None,
            'worst_fold': None
        }
    
    def cross_validate_model(self, model_class, model_params: Dict, 
                           X: np.ndarray, y: np.ndarray, 
                           train_params: Dict, device: str = "cpu") -> Dict[str, Any]:
        """
        Realizar validación cruzada en un modelo
        
        Args:
            model_class: Clase del modelo a entrenar
            model_params: Parámetros para inicializar el modelo
            X: Datos de entrada
            y: Etiquetas
            train_params: Parámetros de entrenamiento
            device: Dispositivo de cómputo
            
        Returns:
            Diccionario con resultados de validación cruzada
        """
        self.logger.info(f"Iniciando validación cruzada con {self.n_splits} folds")
        
        fold_results = []
        
        for fold, (train_idx, val_idx) in enumerate(self.cv.split(X, y)):
            self.logger.info(f"Procesando fold {fold + 1}/{self.n_splits}")
            
            # Dividir datos
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            try:
                # Crear y entrenar modelo
                model = model_class(**model_params)
                if hasattr(model, 'to'):
                    model = model.to(device)
                
                # Entrenar modelo (esto requiere implementación específica)
                fold_result = self._train_and_evaluate_fold(
                    model, X_train, y_train, X_val, y_val, 
                    train_params, fold, device
                )
                
                fold_results.append(fold_result)
                
            except Exception as e:
                self.logger.error(f"Error en fold {fold + 1}: {str(e)}")
                continue
        
        # Calcular estadísticas agregadas
        self.results['fold_results'] = fold_results
        self._calculate_aggregate_metrics()
        
        return self.results
    
    def _train_and_evaluate_fold(self, model, X_train, y_train, X_val, y_val,
                                train_params: Dict, fold: int, device: str) -> Dict[str, Any]:
        """
        Entrenar y evaluar modelo en un fold específico
        
        Args:
            model: Modelo a entrenar
            X_train, y_train: Datos de entrenamiento
            X_val, y_val: Datos de validación
            train_params: Parámetros de entrenamiento
            fold: Número del fold
            device: Dispositivo de cómputo
            
        Returns:
            Diccionario con métricas del fold
        """
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            from torch.utils.data import DataLoader, TensorDataset
            
            # Convertir a tensores de PyTorch
            X_train_tensor = torch.FloatTensor(X_train).to(device)
            y_train_tensor = torch.LongTensor(y_train).to(device)
            X_val_tensor = torch.FloatTensor(X_val).to(device)
            y_val_tensor = torch.LongTensor(y_val).to(device)
            
            # Crear datasets
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            
            train_loader = DataLoader(train_dataset, batch_size=train_params.get('batch_size', 32), shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=train_params.get('batch_size', 32), shuffle=False)
            
            # Configurar entrenamiento
            optimizer = optim.Adam(model.parameters(), lr=train_params.get('learning_rate', 0.001))
            criterion = nn.CrossEntropyLoss()
            
            # Entrenar modelo
            model.train()
            for epoch in range(train_params.get('epochs', 10)):
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    
                    # Manejar salida del modelo (dict o tensor)
                    if isinstance(outputs, dict):
                        logits = outputs['logits']
                    else:
                        logits = outputs
                    
                    loss = criterion(logits, batch_y)
                    loss.backward()
                    optimizer.step()
            
            # Evaluar modelo
            model.eval()
            all_predictions = []
            all_probabilities = []
            all_targets = []
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    outputs = model(batch_X)
                    
                    if isinstance(outputs, dict):
                        logits = outputs['logits']
                    else:
                        logits = outputs
                    
                    probabilities = torch.softmax(logits, dim=1)
                    predictions = torch.argmax(logits, dim=1)
                    
                    all_predictions.extend(predictions.cpu().numpy())
                    all_probabilities.extend(probabilities.cpu().numpy())
                    all_targets.extend(batch_y.cpu().numpy())
            
            # Calcular métricas
            metrics = self._calculate_fold_metrics(
                np.array(all_targets), 
                np.array(all_predictions), 
                np.array(all_probabilities)
            )
            
            metrics['fold'] = fold
            return metrics
            
        except ImportError:
            self.logger.error("PyTorch no disponible para validación cruzada")
            return {'fold': fold, 'error': 'PyTorch no disponible'}
        except Exception as e:
            self.logger.error(f"Error en entrenamiento del fold {fold}: {str(e)}")
            return {'fold': fold, 'error': str(e)}
    
    def _calculate_fold_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                               y_prob: np.ndarray) -> Dict[str, float]:
        """
        Calcular métricas para un fold específico
        
        Args:
            y_true: Etiquetas verdaderas
            y_pred: Predicciones
            y_prob: Probabilidades de predicción
            
        Returns:
            Diccionario con métricas calculadas
        """
        metrics = {}
        
        # Métricas básicas
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        # Métricas ponderadas
        metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # AUC para clasificación multiclase (one-vs-rest)
        try:
            if len(np.unique(y_true)) > 2:
                metrics['auc_ovr'] = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
            else:
                metrics['auc'] = roc_auc_score(y_true, y_prob[:, 1])
        except Exception:
            metrics['auc_ovr'] = 0.0
        
        # Métricas específicas para análisis balístico
        metrics.update(self._calculate_ballistic_metrics(y_true, y_pred))
        
        return metrics
    
    def _calculate_ballistic_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calcular métricas específicas para análisis balístico
        
        Args:
            y_true: Etiquetas verdaderas
            y_pred: Predicciones
            
        Returns:
            Diccionario con métricas balísticas
        """
        metrics = {}
        
        # Matriz de confusión
        cm = confusion_matrix(y_true, y_pred)
        
        # Tasa de identificación correcta (True Positive Rate por clase)
        class_accuracies = cm.diagonal() / cm.sum(axis=1)
        metrics['mean_class_accuracy'] = np.mean(class_accuracies)
        metrics['min_class_accuracy'] = np.min(class_accuracies)
        metrics['max_class_accuracy'] = np.max(class_accuracies)
        
        # Tasa de falsos positivos por clase
        fp_rates = []
        for i in range(len(cm)):
            fp = cm[:, i].sum() - cm[i, i]
            tn = cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]
            fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
            fp_rates.append(fp_rate)
        
        metrics['mean_fp_rate'] = np.mean(fp_rates)
        metrics['max_fp_rate'] = np.max(fp_rates)
        
        # Métricas de confiabilidad
        correct_predictions = (y_true == y_pred)
        metrics['reliability_score'] = np.mean(correct_predictions)
        
        # Distribución de errores por clase
        unique_classes = np.unique(y_true)
        error_distribution = {}
        for cls in unique_classes:
            class_mask = (y_true == cls)
            class_errors = np.sum(y_pred[class_mask] != cls)
            class_total = np.sum(class_mask)
            error_distribution[f'class_{cls}_error_rate'] = class_errors / class_total if class_total > 0 else 0
        
        metrics.update(error_distribution)
        
        return metrics
    
    def _calculate_aggregate_metrics(self):
        """Calcular métricas agregadas de todos los folds"""
        if not self.results['fold_results']:
            return
        
        # Extraer métricas de todos los folds válidos
        valid_folds = [fold for fold in self.results['fold_results'] if 'error' not in fold]
        
        if not valid_folds:
            self.logger.warning("No hay folds válidos para calcular métricas agregadas")
            return
        
        # Obtener nombres de métricas
        metric_names = [key for key in valid_folds[0].keys() if key != 'fold']
        
        # Calcular media y desviación estándar
        for metric in metric_names:
            values = [fold[metric] for fold in valid_folds if metric in fold]
            if values:
                self.results['mean_metrics'][metric] = np.mean(values)
                self.results['std_metrics'][metric] = np.std(values)
        
        # Encontrar mejor y peor fold basado en accuracy
        if 'accuracy' in self.results['mean_metrics']:
            accuracies = [(fold['fold'], fold.get('accuracy', 0)) for fold in valid_folds]
            accuracies.sort(key=lambda x: x[1])
            
            self.results['worst_fold'] = accuracies[0][0]
            self.results['best_fold'] = accuracies[-1][0]
    
    def plot_cv_results(self, output_dir: str = "cv_results"):
        """
        Generar gráficos de los resultados de validación cruzada
        
        Args:
            output_dir: Directorio donde guardar los gráficos
        """
        os.makedirs(output_dir, exist_ok=True)
        
        if not self.results['fold_results']:
            self.logger.warning("No hay resultados para graficar")
            return
        
        valid_folds = [fold for fold in self.results['fold_results'] if 'error' not in fold]
        
        if not valid_folds:
            return
        
        # Gráfico de métricas por fold
        metrics_to_plot = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.ravel()
        
        for i, metric in enumerate(metrics_to_plot):
            if metric in self.results['mean_metrics']:
                values = [fold.get(metric, 0) for fold in valid_folds]
                folds = [fold['fold'] + 1 for fold in valid_folds]
                
                axes[i].bar(folds, values, alpha=0.7)
                axes[i].axhline(y=self.results['mean_metrics'][metric], 
                              color='red', linestyle='--', 
                              label=f'Media: {self.results["mean_metrics"][metric]:.3f}')
                axes[i].set_title(f'{metric.replace("_", " ").title()} por Fold')
                axes[i].set_xlabel('Fold')
                axes[i].set_ylabel(metric.replace("_", " ").title())
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'cv_metrics_by_fold.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Gráfico de distribución de métricas
        self._plot_metric_distributions(output_dir, valid_folds)
        
        # Gráfico de comparación de folds
        self._plot_fold_comparison(output_dir, valid_folds)
    
    def _plot_metric_distributions(self, output_dir: str, valid_folds: List[Dict]):
        """Generar gráfico de distribución de métricas"""
        metrics_to_plot = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        data_for_boxplot = []
        labels = []
        
        for metric in metrics_to_plot:
            if metric in self.results['mean_metrics']:
                values = [fold.get(metric, 0) for fold in valid_folds]
                data_for_boxplot.append(values)
                labels.append(metric.replace("_", " ").title())
        
        if data_for_boxplot:
            bp = ax.boxplot(data_for_boxplot, labels=labels, patch_artist=True)
            
            # Colorear las cajas
            colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']
            for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
                patch.set_facecolor(color)
            
            ax.set_title('Distribución de Métricas en Validación Cruzada')
            ax.set_ylabel('Valor de la Métrica')
            ax.grid(True, alpha=0.3)
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'cv_metric_distributions.png'), dpi=300, bbox_inches='tight')
            plt.close()
    
    def _plot_fold_comparison(self, output_dir: str, valid_folds: List[Dict]):
        """Generar gráfico de comparación entre folds"""
        if len(valid_folds) < 2:
            return
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
        x = np.arange(len(valid_folds))
        width = 0.2
        
        colors = ['blue', 'green', 'red', 'orange']
        
        for i, metric in enumerate(metrics):
            if metric in self.results['mean_metrics']:
                values = [fold.get(metric, 0) for fold in valid_folds]
                ax.bar(x + i * width, values, width, label=metric.replace("_", " ").title(), 
                      color=colors[i], alpha=0.7)
        
        ax.set_xlabel('Fold')
        ax.set_ylabel('Valor de la Métrica')
        ax.set_title('Comparación de Métricas por Fold')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels([f'Fold {fold["fold"] + 1}' for fold in valid_folds])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'cv_fold_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_results(self, output_path: str):
        """
        Guardar resultados de validación cruzada en archivo JSON
        
        Args:
            output_path: Ruta donde guardar los resultados
        """
        # Preparar datos para serialización JSON
        results_to_save = {
            'metadata': {
                'n_splits': self.n_splits,
                'stratified': self.stratified,
                'random_state': self.random_state,
                'timestamp': datetime.now().isoformat()
            },
            'results': self.results
        }
        
        # Convertir arrays numpy a listas para JSON
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj
        
        results_to_save = convert_numpy(results_to_save)
        
        with open(output_path, 'w') as f:
            json.dump(results_to_save, f, indent=2)
        
        self.logger.info(f"Resultados de validación cruzada guardados en: {output_path}")
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Obtener resumen de los resultados de validación cruzada
        
        Returns:
            Diccionario con resumen de resultados
        """
        if not self.results['mean_metrics']:
            return {'error': 'No hay resultados disponibles'}
        
        summary = {
            'configuration': {
                'n_splits': self.n_splits,
                'stratified': self.stratified,
                'successful_folds': len([f for f in self.results['fold_results'] if 'error' not in f])
            },
            'performance': {
                'mean_accuracy': self.results['mean_metrics'].get('accuracy', 0),
                'std_accuracy': self.results['std_metrics'].get('accuracy', 0),
                'mean_f1': self.results['mean_metrics'].get('f1_macro', 0),
                'std_f1': self.results['std_metrics'].get('f1_macro', 0)
            },
            'reliability': {
                'best_fold': self.results.get('best_fold'),
                'worst_fold': self.results.get('worst_fold'),
                'consistency': 1 - (self.results['std_metrics'].get('accuracy', 1) / 
                                  max(self.results['mean_metrics'].get('accuracy', 1), 0.001))
            }
        }
        
        return summary