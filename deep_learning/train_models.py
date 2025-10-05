#!/usr/bin/env python3
"""
Script de entrenamiento para modelos de deep learning con datos NIST FADB
Entrena modelos CNN y Siamese para clasificación y matching balístico
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from datetime import datetime
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Agregar el directorio raíz al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importar módulos del sistema
from deep_learning.data.nist_loader import NISTFADBLoader, NISTFADBDataset
from deep_learning.models.cnn_models import BallisticCNN
from deep_learning.models.siamese_models import SiameseNetwork
from deep_learning.utils.performance_optimizer import PerformanceOptimizer
from deep_learning.config.experiment_config import ExperimentConfig, ModelConfig, DataConfig, TrainingConfig
from deep_learning.config.training_configs import (
    get_recommended_config, validate_config_for_dataset, 
    NIST_FADB_CONFIGS, list_available_configs
)
from deep_learning.utils.cross_validation import CrossValidator

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ModelTrainer:
    """Clase principal para entrenar modelos de deep learning con datos NIST FADB"""
    
    def __init__(self, config: ExperimentConfig, output_dir: str = "training_results"):
        """
        Inicializar el entrenador de modelos
        
        Args:
            config: Configuración del experimento
            output_dir: Directorio para guardar resultados
        """
        self.config = config
        self.output_dir = output_dir
        self.logger = logging.getLogger(__name__)  # Inicializar logger primero
        self.device = self._setup_device()
        
        # Inicializar num_classes desde la configuración
        self.num_classes = config.model.num_classes
        
        # Crear directorio de salida
        os.makedirs(output_dir, exist_ok=True)
        
        # Optimizar rendimiento del sistema
        self.optimizer = PerformanceOptimizer()
        self.optimizer.setup_torch_optimizations()
        
        # Inicializar variables de entrenamiento
        self.train_history = {"loss": [], "accuracy": []}
        self.val_history = {"loss": [], "accuracy": []}
        self.best_val_accuracy = 0.0
        self.best_model_path = None
        
        # Inicializar variables que se usarán más tarde
        self.data_path = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        
        self.logger.info(f"Entrenador inicializado - Dispositivo: {self.device}")
        self.logger.info(f"Configuración: {config.experiment_name}")
        self.logger.info(f"Número de clases: {self.num_classes}")
    
    def _setup_device(self):
        """Configurar dispositivo de cómputo (CPU/GPU)"""
        try:
            import torch
            if torch.cuda.is_available():
                device = torch.device("cuda")
                self.logger.info(f"GPU disponible: {torch.cuda.get_device_name()}")
            else:
                device = torch.device("cpu")
                self.logger.info("Usando CPU para entrenamiento")
            return device
        except ImportError:
            self.logger.warning("PyTorch no disponible, usando CPU")
            return "cpu"
    
    def load_and_prepare_data(self) -> Tuple[Any, Any, Any]:
        """
        Cargar y preparar los datos para entrenamiento
        
        Returns:
            Tuple con (train_loader, val_loader, test_loader)
        """
        self.logger.info("Cargando datos NIST FADB...")
        
        # Inicializar el cargador de datos
        data_loader = NISTFADBLoader(self.config.data.data_root)
        
        # Obtener estadísticas del dataset
        dataset_stats = data_loader.get_dataset_statistics()
        self.logger.info(f"Dataset cargado: {dataset_stats['total_images']} imágenes, "
                        f"{dataset_stats['total_studies']} estudios")
        
        # Validar configuración con el dataset
        validation_results = validate_config_for_dataset(self.config, dataset_stats)
        
        # Aplicar ajustes recomendados
        if validation_results["adjustments"]:
            self.logger.info("Aplicando ajustes recomendados a la configuración:")
            for key, value in validation_results["adjustments"].items():
                if key == "num_classes":
                    self.config.model.num_classes = value
                    self.num_classes = value  # Actualizar también el atributo de instancia
                elif key == "batch_size":
                    self.config.data.batch_size = value
                self.logger.info(f"  - {key}: {value}")
        
        # Mostrar advertencias
        for warning in validation_results["warnings"]:
            self.logger.warning(warning)
        
        # Mostrar sugerencias
        for suggestion in validation_results["suggestions"]:
            self.logger.info(f"Sugerencia: {suggestion}")
        
        # Crear datasets
        try:
            # Verificar si hay datos disponibles
            if dataset_stats['total_images'] == 0:
                self.logger.warning("No hay imágenes disponibles en el dataset")
                raise AttributeError("No data available")
            
            # Crear splits estratificados
            splits = data_loader.create_stratified_splits(
                train_ratio=self.config.data.train_split,
                val_ratio=self.config.data.val_split,
                test_ratio=1.0 - self.config.data.train_split - self.config.data.val_split
            )
            
            # Crear datasets usando NISTFADBDataset
            train_dataset = NISTFADBDataset(
                metadata=data_loader.metadata,
                indices=splits['train'],
                task='classification',
                image_size=getattr(self.config.data, 'target_size', (224, 224))
            )
            
            val_dataset = NISTFADBDataset(
                metadata=data_loader.metadata,
                indices=splits['val'],
                task='classification',
                image_size=getattr(self.config.data, 'target_size', (224, 224))
            )
            
            test_dataset = NISTFADBDataset(
                metadata=data_loader.metadata,
                indices=splits['test'],
                task='classification',
                image_size=getattr(self.config.data, 'target_size', (224, 224))
            )
            
            self.logger.info(f"División de datos - Train: {len(train_dataset)}, "
                           f"Val: {len(val_dataset)}, Test: {len(test_dataset)}")
            
            # Crear data loaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config.data.batch_size,
                shuffle=True,
                num_workers=getattr(self.config.data, 'num_workers', 0),
                pin_memory=getattr(self.config.data, 'pin_memory', False)
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.data.batch_size,
                shuffle=False,
                num_workers=getattr(self.config.data, 'num_workers', 0),
                pin_memory=getattr(self.config.data, 'pin_memory', False)
            )
            
            test_loader = DataLoader(
                test_dataset,
                batch_size=self.config.data.batch_size,
                shuffle=False,
                num_workers=getattr(self.config.data, 'num_workers', 0),
                pin_memory=getattr(self.config.data, 'pin_memory', False)
            )
            
            return train_loader, val_loader, test_loader
            
        except Exception as e:
            self.logger.error(f"Error al cargar datos: {str(e)}")
            # Fallback: crear datos sintéticos para prueba
            self.logger.info("Creando datos sintéticos para prueba...")
            return self._create_synthetic_data()
    
    def _create_synthetic_data(self):
        """Crear datos sintéticos para pruebas cuando no se pueden cargar datos reales"""
        try:
            import torch
            from torch.utils.data import DataLoader, TensorDataset
            
            # Crear datos sintéticos
            num_samples = 1000
            num_classes = max(self.config.model.num_classes, 2)  # Asegurar al menos 2 clases
            image_size = getattr(self.config.data, 'target_size', (224, 224))  # Usar target_size o valor por defecto
            
            # Actualizar num_classes en caso de que haya cambiado
            if num_classes != self.num_classes:
                self.config.model.num_classes = num_classes
                self.num_classes = num_classes
            
            self.logger.info(f"Creando datos sintéticos con {num_classes} clases y {num_samples} muestras")
            
            # Generar imágenes sintéticas
            images = torch.randn(num_samples, 3, image_size[0], image_size[1])
            labels = torch.randint(0, num_classes, (num_samples,))
            
            # Crear dataset
            dataset = TensorDataset(images, labels)
            
            # Dividir en train/val/test
            train_size = int(0.7 * num_samples)
            val_size = int(0.15 * num_samples)
            test_size = num_samples - train_size - val_size
            
            train_dataset = TensorDataset(images[:train_size], labels[:train_size])
            val_dataset = TensorDataset(images[train_size:train_size+val_size], 
                                      labels[train_size:train_size+val_size])
            test_dataset = TensorDataset(images[train_size+val_size:], 
                                       labels[train_size+val_size:])
            
            # Crear data loaders
            train_loader = DataLoader(train_dataset, batch_size=self.config.data.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=self.config.data.batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=self.config.data.batch_size, shuffle=False)
            
            self.logger.info(f"Datos sintéticos creados - Train: {len(train_dataset)}, "
                           f"Val: {len(val_dataset)}, Test: {len(test_dataset)}")
            
            return train_loader, val_loader, test_loader
            
        except ImportError:
            self.logger.error("PyTorch no disponible para crear datos sintéticos")
            return None, None, None
    
    def create_models(self):
        """Crear y configurar los modelos para entrenamiento"""
        self.logger.info("Creando modelos...")
        
        try:
            import torch
            import torch.nn as nn
            
            # Crear modelo CNN para clasificación
            self.cnn_model = BallisticCNN(
                input_channels=3,
                num_classes=max(self.config.model.num_classes, 2),  # Asegurar al menos 2 clases
                feature_dim=self.config.model.feature_dim,
                use_attention=self.config.model.use_attention,
                dropout_rate=self.config.model.dropout_rate
            ).to(self.device)
            
            # Crear modelo Siamese para matching (opcional)
            backbone_config = {
                "type": "ballistic_cnn",  # Usar tipo soportado
                "input_channels": 3,
                "num_classes": max(self.config.model.num_classes, 2),
                "backbone_feature_dim": self.config.model.feature_dim,
                "use_attention": self.config.model.use_attention
            }
            
            self.siamese_model = SiameseNetwork(
                backbone_config=backbone_config,
                feature_dim=self.config.model.feature_dim,
                similarity_method="cosine",
                temperature=0.1
            ).to(self.device)
            
            # Contar parámetros
            cnn_params = sum(p.numel() for p in self.cnn_model.parameters() if p.requires_grad)
            siamese_params = sum(p.numel() for p in self.siamese_model.parameters() if p.requires_grad)
            
            self.logger.info(f"Modelo CNN creado - Parámetros entrenables: {cnn_params:,}")
            self.logger.info(f"Modelo Siamese creado - Parámetros entrenables: {siamese_params:,}")
            
            return self.cnn_model, self.siamese_model
            
        except ImportError:
            self.logger.error("PyTorch no disponible para crear modelos")
            return None, None
        except Exception as e:
            self.logger.error(f"Error al crear modelos: {str(e)}")
            return None, None
    
    def train_cnn_model(self, train_loader, val_loader):
        """
        Entrenar el modelo CNN para clasificación
        
        Args:
            train_loader: DataLoader de entrenamiento
            val_loader: DataLoader de validación
        """
        if self.cnn_model is None:
            self.logger.error("Modelo CNN no inicializado")
            return
        
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            from torch.optim.lr_scheduler import ReduceLROnPlateau
            
            self.logger.info("Iniciando entrenamiento del modelo CNN...")
            
            # Inicializar listas para el historial de entrenamiento
            train_losses = []
            val_losses = []
            train_accuracies = []
            val_accuracies = []
            
            # Configurar optimizador
            if self.config.training.optimizer.lower() == "adam":
                optimizer = optim.Adam(
                    self.cnn_model.parameters(),
                    lr=self.config.training.learning_rate,
                    weight_decay=self.config.training.weight_decay
                )
            elif self.config.training.optimizer.lower() == "sgd":
                optimizer = optim.SGD(
                    self.cnn_model.parameters(),
                    lr=self.config.training.learning_rate,
                    momentum=0.9,
                    weight_decay=self.config.training.weight_decay
                )
            else:
                optimizer = optim.Adam(self.cnn_model.parameters(), lr=self.config.training.learning_rate)
            
            # Configurar scheduler si está habilitado
            if self.config.training.use_scheduler:
                if self.config.training.scheduler_type == "plateau":
                    scheduler = ReduceLROnPlateau(
                        optimizer,
                        mode="min",
                        factor=self.config.training.gamma,
                        patience=self.config.training.patience,
                        min_lr=1e-6
                    )
                else:
                    scheduler = None
            else:
                scheduler = None
            
            # Función de pérdida
            criterion = nn.CrossEntropyLoss()
            
            # Variables para early stopping
            best_val_loss = float('inf')
            patience_counter = 0
            
            # Entrenamiento
            for epoch in range(self.config.training.epochs):
                # Fase de entrenamiento
                self.cnn_model.train()
                train_loss = 0.0
                train_correct = 0
                train_total = 0
                
                for batch_idx, (data, target) in enumerate(train_loader):
                    data, target = data.to(self.device), target.to(self.device)
                    
                    optimizer.zero_grad()
                    output = self.cnn_model(data)
                    
                    # El modelo puede devolver un diccionario o tensor
                    if isinstance(output, dict):
                        logits = output['logits']
                    else:
                        logits = output
                    
                    loss = criterion(logits, target)
                    loss.backward()
                    
                    # Gradient clipping (opcional, usando valor por defecto)
                    gradient_clip_value = getattr(self.config.training, 'gradient_clipping', 0.0)
                    if gradient_clip_value > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.cnn_model.parameters(), 
                            gradient_clip_value
                        )
                    
                    optimizer.step()
                    
                    train_loss += loss.item()
                    _, predicted = logits.max(1)
                    train_total += target.size(0)
                    train_correct += predicted.eq(target).sum().item()
                    
                    if batch_idx % 10 == 0:
                        self.logger.info(f'Epoch {epoch+1}/{self.config.training.epochs}, '
                                       f'Batch {batch_idx}/{len(train_loader)}, '
                                       f'Loss: {loss.item():.4f}')
                
                # Fase de validación
                self.cnn_model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for data, target in val_loader:
                        data, target = data.to(self.device), target.to(self.device)
                        output = self.cnn_model(data)
                        
                        if isinstance(output, dict):
                            logits = output['logits']
                        else:
                            logits = output
                        
                        loss = criterion(logits, target)
                        val_loss += loss.item()
                        
                        _, predicted = logits.max(1)
                        val_total += target.size(0)
                        val_correct += predicted.eq(target).sum().item()
                
                # Calcular métricas
                train_loss /= len(train_loader)
                val_loss /= len(val_loader)
                train_acc = 100. * train_correct / train_total
                val_acc = 100. * val_correct / val_total
                
                # Actualizar listas del historial
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                train_accuracies.append(train_acc)
                val_accuracies.append(val_acc)
                
                # Guardar historial (mantener compatibilidad con atributos de clase)
                self.train_history["loss"].append(train_loss)
                self.train_history["accuracy"].append(train_acc)
                self.val_history["loss"].append(val_loss)
                self.val_history["accuracy"].append(val_acc)
                
                self.logger.info(f'Epoch {epoch+1}/{self.config.training.epochs}:')
                self.logger.info(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
                self.logger.info(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
                
                # Guardar mejor modelo
                if val_acc > self.best_val_accuracy:
                    self.best_val_accuracy = val_acc
                    self.best_model_path = os.path.join(self.output_dir, "best_cnn_model.pth")
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.cnn_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_accuracy': val_acc,
                        'config': self.config
                    }, self.best_model_path)
                    self.logger.info(f'Nuevo mejor modelo guardado con accuracy: {val_acc:.2f}%')
                
                # Scheduler step
                if scheduler:
                    scheduler.step(val_loss)
                
                # Early stopping
                if self.config.training.use_early_stopping:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= self.config.training.early_stopping_patience:
                            self.logger.info(f'Early stopping en epoch {epoch+1}')
                            break
            
            self.logger.info("Entrenamiento completado!")
            self.logger.info(f"Mejor accuracy de validación: {self.best_val_accuracy:.2f}%")
            
            # Retornar historial de entrenamiento
            return {
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_accuracies': train_accuracies,
                'val_accuracies': val_accuracies
            }
            
        except Exception as e:
            self.logger.error(f"Error durante el entrenamiento: {str(e)}")
            raise
    
    def evaluate_cnn_model(self, test_loader=None):
        """Evaluar el modelo CNN en el conjunto de prueba"""
        logger.info("Evaluando modelo CNN...")
        
        # Usar test_loader pasado como parámetro o self.test_loader si existe
        if test_loader is None:
            test_loader = getattr(self, 'test_loader', None)
        
        if test_loader is None:
            logger.error("No hay test_loader disponible para evaluación")
            return 0.0, {}
        
        # Cargar mejor modelo
        best_model_path = os.path.join(self.output_dir, 'best_cnn_model.pth')
        checkpoint = torch.load(best_model_path, map_location=self.device, weights_only=False)
        self.cnn_model.load_state_dict(checkpoint['model_state_dict'])
        
        self.cnn_model.eval()
        test_correct = 0
        test_total = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, targets in test_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.cnn_model(data)
                
                if isinstance(outputs, dict):
                    logits = outputs['logits']
                else:
                    logits = outputs
                
                _, predicted = torch.max(logits.data, 1)
                test_total += targets.size(0)
                test_correct += (predicted == targets).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        test_acc = 100. * test_correct / test_total
        logger.info(f"Precisión en conjunto de prueba: {test_acc:.2f}%")
        
        # Generar reporte de clasificación
        class_names = [f'Class_{i}' for i in range(self.num_classes)]
        report = classification_report(all_targets, all_predictions, 
                                     target_names=class_names, output_dict=True)
        
        # Guardar reporte
        with open(os.path.join(self.output_dir, 'cnn_evaluation_report.json'), 'w') as f:
            json.dump(report, f, indent=2)
        
        # Crear matriz de confusión
        cm = confusion_matrix(all_targets, all_predictions)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Matriz de Confusión - Modelo CNN')
        plt.ylabel('Etiqueta Real')
        plt.xlabel('Predicción')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'cnn_confusion_matrix.png'), dpi=300)
        plt.close()
        
        return test_acc, report
    
    def plot_training_history(self, history, model_name):
        """Graficar historial de entrenamiento"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss
        ax1.plot(history['train_losses'], label='Entrenamiento', color='blue')
        ax1.plot(history['val_losses'], label='Validación', color='red')
        ax1.set_title(f'Pérdida durante Entrenamiento - {model_name}')
        ax1.set_xlabel('Época')
        ax1.set_ylabel('Pérdida')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy
        ax2.plot(history['train_accuracies'], label='Entrenamiento', color='blue')
        ax2.plot(history['val_accuracies'], label='Validación', color='red')
        ax2.set_title(f'Precisión durante Entrenamiento - {model_name}')
        ax2.set_xlabel('Época')
        ax2.set_ylabel('Precisión (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'{model_name.lower()}_training_history.png'), dpi=300)
        plt.close()
    
    def run_cross_validation(self) -> Dict[str, Any]:
        """
        Ejecutar validación cruzada en el modelo CNN
        
        Returns:
            Diccionario con resultados de validación cruzada
        """
        self.logger.info("Iniciando validación cruzada...")
        
        try:
            # Preparar datos para validación cruzada
            if not hasattr(self, 'train_dataset') or self.train_dataset is None:
                self.logger.error("Datos de entrenamiento no disponibles para validación cruzada")
                return {'error': 'Datos no disponibles'}
            
            # Extraer datos y etiquetas
            X_data = []
            y_data = []
            
            for i in range(len(self.train_dataset)):
                sample, label = self.train_dataset[i]
                if isinstance(sample, torch.Tensor):
                    X_data.append(sample.numpy())
                else:
                    X_data.append(sample)
                y_data.append(label)
            
            X_data = np.array(X_data)
            y_data = np.array(y_data)
            
            self.logger.info(f"Datos preparados para CV: {X_data.shape}, {len(np.unique(y_data))} clases")
            
            # Configurar validación cruzada
            cv_config = self.config.training_config.cross_validation
            n_splits = cv_config.get('n_splits', 5)
            stratified = cv_config.get('stratified', True)
            
            validator = CrossValidator(
                n_splits=n_splits,
                stratified=stratified,
                random_state=self.config.training_config.random_seed
            )
            
            # Parámetros del modelo
            model_params = {
                'backbone_config': self.config.model_config.cnn_config,
                'num_classes': self.config.data_config.num_classes
            }
            
            # Parámetros de entrenamiento para CV
            train_params = {
                'epochs': min(self.config.training_config.epochs, 10),  # Reducir épocas para CV
                'batch_size': self.config.training_config.batch_size,
                'learning_rate': self.config.training_config.learning_rate
            }
            
            # Ejecutar validación cruzada
            cv_results = validator.cross_validate_model(
                model_class=BallisticCNN,
                model_params=model_params,
                X=X_data,
                y=y_data,
                train_params=train_params,
                device=self.device
            )
            
            # Guardar resultados
            cv_output_dir = os.path.join(self.output_dir, 'cross_validation')
            os.makedirs(cv_output_dir, exist_ok=True)
            
            # Generar gráficos
            validator.plot_cv_results(cv_output_dir)
            
            # Guardar resultados detallados
            cv_results_path = os.path.join(cv_output_dir, 'cv_results.json')
            validator.save_results(cv_results_path)
            
            # Obtener resumen
            cv_summary = validator.get_summary()
            
            self.logger.info("Validación cruzada completada exitosamente")
            self.logger.info(f"Accuracy promedio: {cv_summary.get('performance', {}).get('mean_accuracy', 0):.4f}")
            self.logger.info(f"F1-Score promedio: {cv_summary.get('performance', {}).get('mean_f1', 0):.4f}")
            
            return {
                'summary': cv_summary,
                'detailed_results': cv_results,
                'output_directory': cv_output_dir
            }
            
        except Exception as e:
            self.logger.error(f"Error en validación cruzada: {str(e)}")
            return {'error': str(e)}
    
    def generate_training_report(self, cnn_history, cnn_test_acc):
        """Generar reporte completo de entrenamiento"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'device': str(self.device),
            'data_path': self.data_path if hasattr(self, 'data_path') and self.data_path else 'synthetic_data',
            'num_classes': self.num_classes,
            'dataset_splits': {
                'train': len(self.train_loader.dataset) if self.train_loader else 0,
                'validation': len(self.val_loader.dataset) if self.val_loader else 0,
                'test': len(self.test_loader.dataset) if self.test_loader else 0
            },
            'cnn_results': {
                'best_val_accuracy': cnn_history.get('best_val_acc', 0.0),
                'test_accuracy': cnn_test_acc,
                'final_train_loss': cnn_history.get('train_losses', [0.0])[-1],
                'final_val_loss': cnn_history.get('val_losses', [0.0])[-1]
            },
            'training_config': {
                'epochs': self.config.training.epochs,
                'batch_size': self.config.data.batch_size,
                'learning_rate': self.config.training.learning_rate,
                'weight_decay': self.config.training.weight_decay
            }
        }
        
        with open(os.path.join(self.output_dir, 'training_report.json'), 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info("Reporte de entrenamiento generado")
        return report

def main():
    """Función principal para ejecutar el entrenamiento"""
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("=== Iniciando Sistema de Entrenamiento SIGeC-Balistica===")
        
        # Mostrar configuraciones disponibles
        logger.info("Configuraciones disponibles:")
        for name, desc in list_available_configs().items():
            logger.info(f"  - {name}: {desc}")
        
        # Obtener información del sistema para recomendación automática
        try:
            import psutil
            memory_gb = psutil.virtual_memory().total / (1024**3)
            logger.info(f"Memoria RAM disponible: {memory_gb:.1f} GB")
        except ImportError:
            memory_gb = 8.0  # Valor por defecto
            logger.warning("psutil no disponible, usando valor por defecto de memoria")
        
        # Analizar dataset para obtener recomendación
        data_path = "uploads/Muestras NIST FADB"
        
        if os.path.exists(data_path):
            logger.info("Analizando dataset NIST FADB...")
            try:
                data_loader = NISTFADBLoader(data_path)
                dataset_stats = data_loader.get_dataset_statistics()
                num_images = dataset_stats.get('total_images', 0)
                logger.info(f"Dataset encontrado: {num_images} imágenes")
                
                # Obtener configuración recomendada
                config = get_recommended_config(num_images, memory_gb)
                logger.info(f"Usando configuración recomendada: {config.experiment_name}")
                
            except Exception as e:
                logger.warning(f"Error al analizar dataset: {str(e)}")
                logger.info("Usando configuración de prueba rápida")
                config = NIST_FADB_CONFIGS["quick_test"]
        else:
            logger.warning(f"Dataset no encontrado en {data_path}")
            logger.info("Usando configuración de prueba rápida")
            config = NIST_FADB_CONFIGS["quick_test"]
        
        # Crear directorio de resultados con timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"training_results_{timestamp}"
        
        # Inicializar entrenador
        trainer = ModelTrainer(config, output_dir)
        
        # Cargar y preparar datos
        logger.info("Cargando y preparando datos...")
        train_loader, val_loader, test_loader = trainer.load_and_prepare_data()
        
        # Asignar los data loaders a las variables de instancia del trainer
        trainer.train_loader = train_loader
        trainer.val_loader = val_loader
        trainer.test_loader = test_loader
        trainer.data_path = data_path
        
        if train_loader is None:
            logger.error("No se pudieron cargar los datos. Abortando entrenamiento.")
            return
        
        # Crear modelos
        logger.info("Creando modelos...")
        cnn_model, siamese_model = trainer.create_models()
        
        if cnn_model is None:
            logger.error("No se pudieron crear los modelos. Abortando entrenamiento.")
            return
        
        # Entrenar modelo CNN
        logger.info("Iniciando entrenamiento del modelo CNN...")
        cnn_history = trainer.train_cnn_model(train_loader, val_loader)
        
        # Evaluar modelo
        cnn_test_acc = 0.0
        if test_loader is not None:
            logger.info("Evaluando modelo en conjunto de prueba...")
            cnn_test_acc, _ = trainer.evaluate_cnn_model(test_loader)
        
        # Generar gráficos de entrenamiento
        if cnn_history:
            logger.info("Generando gráficos de entrenamiento...")
            trainer.plot_training_history(cnn_history, "CNN")
        
        # 6. Ejecutar validación cruzada (opcional)
        cv_results = None
        if config.evaluation.use_cross_validation:
            logger.info("Ejecutando validación cruzada...")
            cv_results = trainer.run_cross_validation()
            
            if 'error' not in cv_results:
                logger.info("Validación cruzada completada exitosamente")
                cv_summary = cv_results.get('summary', {})
                logger.info(f"CV Accuracy: {cv_summary.get('performance', {}).get('mean_accuracy', 0):.4f} ± {cv_summary.get('performance', {}).get('std_accuracy', 0):.4f}")
            else:
                logger.warning(f"Error en validación cruzada: {cv_results['error']}")
        
        # 7. Generar reporte final
        logger.info("Generando reporte final...")
        trainer.generate_training_report(cnn_history, cnn_test_acc)
        
        logger.info("=== Entrenamiento Completado Exitosamente ===")
        logger.info(f"Resultados guardados en: {output_dir}")
        logger.info(f"Mejor accuracy de validación: {trainer.best_val_accuracy:.2f}%")
        
        if trainer.best_model_path:
            logger.info(f"Mejor modelo guardado en: {trainer.best_model_path}")
        
    except KeyboardInterrupt:
        logger.info("Entrenamiento interrumpido por el usuario")
    except Exception as e:
        logger.error(f"Error durante el entrenamiento: {str(e)}")
        raise

if __name__ == "__main__":
    main()