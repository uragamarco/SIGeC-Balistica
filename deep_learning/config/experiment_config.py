"""
Configuración de Experimentos para Deep Learning SIGeC-Balistica
========================================================

Este módulo define las clases de configuración para experimentos,
modelos, entrenamiento, datos y evaluación del sistema de deep learning.

Autor: SIGeC-BalisticaTeam
"""

import os
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuración para modelos de deep learning."""
    
    # Tipo de modelo
    model_type: str = "BallisticCNN"  # BallisticCNN, SiameseNetwork, TripletNetwork
    architecture: str = "custom"  # custom, resnet18, resnet50, efficientnet_b0
    
    # Parámetros del modelo
    input_size: tuple = (224, 224)
    num_classes: int = 10
    feature_dim: int = 512
    dropout_rate: float = 0.3
    
    # Configuración específica por tipo
    use_attention: bool = True
    attention_type: str = "spatial"  # spatial, channel, both
    
    # Para redes Siamese/Triplet
    similarity_metric: str = "cosine"  # cosine, euclidean, learned
    margin: float = 1.0
    
    # Configuración de capas
    hidden_layers: List[int] = field(default_factory=lambda: [1024, 512, 256])
    activation: str = "relu"
    batch_norm: bool = True
    
    def __post_init__(self):
        """Validación post-inicialización."""
        if self.model_type not in ["BallisticCNN", "SiameseNetwork", "TripletNetwork", "HierarchicalClassifier"]:
            raise ValueError(f"Tipo de modelo no soportado: {self.model_type}")
        
        if self.architecture not in ["custom", "resnet18", "resnet50", "efficientnet_b0", "efficientnet_b3"]:
            raise ValueError(f"Arquitectura no soportada: {self.architecture}")

@dataclass
class DataConfig:
    """Configuración para el pipeline de datos."""
    
    # Rutas de datos
    data_root: str = "uploads/Muestras NIST FADB"
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    
    # Configuración de carga
    batch_size: int = 32
    num_workers: int = 4
    shuffle_train: bool = True
    pin_memory: bool = True
    
    # Preprocesamiento
    normalize: bool = True
    resize_images: bool = True
    target_size: tuple = (224, 224)
    
    # Augmentación
    use_augmentation: bool = True
    rotation_range: float = 15.0
    brightness_range: tuple = (0.8, 1.2)
    contrast_range: tuple = (0.8, 1.2)
    noise_std: float = 0.01
    
    # Filtros de calidad
    min_image_size: tuple = (100, 100)
    max_blur_threshold: float = 100.0
    min_contrast_threshold: float = 0.1
    
    # Estratificación
    stratify_by: List[str] = field(default_factory=lambda: ["study", "firearm_type"])
    balance_classes: bool = True
    
    def __post_init__(self):
        """Validación de configuración de datos."""
        if abs(self.train_split + self.val_split + self.test_split - 1.0) > 1e-6:
            raise ValueError("Los splits deben sumar 1.0")
        
        if not os.path.exists(self.data_root):
            logger.warning(f"Directorio de datos no encontrado: {self.data_root}")

@dataclass
class TrainingConfig:
    """Configuración para el entrenamiento."""
    
    # Parámetros básicos
    epochs: int = 100
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    
    # Optimizador
    optimizer: str = "adam"  # adam, sgd, adamw
    momentum: float = 0.9  # Para SGD
    beta1: float = 0.9     # Para Adam
    beta2: float = 0.999   # Para Adam
    
    # Scheduler
    use_scheduler: bool = True
    scheduler_type: str = "cosine"  # cosine, step, plateau
    step_size: int = 30
    gamma: float = 0.1
    patience: int = 10
    
    # Early stopping
    use_early_stopping: bool = True
    early_stopping_patience: int = 15
    early_stopping_delta: float = 1e-4
    
    # Regularización
    use_mixup: bool = False
    mixup_alpha: float = 0.2
    use_cutmix: bool = False
    cutmix_alpha: float = 1.0
    
    # Checkpoints
    save_checkpoints: bool = True
    checkpoint_frequency: int = 10
    save_best_only: bool = True
    
    # Logging
    log_frequency: int = 10
    validate_frequency: int = 1
    
    def __post_init__(self):
        """Validación de configuración de entrenamiento."""
        if self.optimizer not in ["adam", "sgd", "adamw"]:
            raise ValueError(f"Optimizador no soportado: {self.optimizer}")
        
        if self.scheduler_type not in ["cosine", "step", "plateau", "none"]:
            raise ValueError(f"Scheduler no soportado: {self.scheduler_type}")

@dataclass
class EvaluationConfig:
    """Configuración para evaluación y métricas."""
    
    # Métricas principales
    primary_metric: str = "accuracy"  # accuracy, f1, precision, recall
    compute_confusion_matrix: bool = True
    compute_roc_curves: bool = True
    compute_cmc_curves: bool = True
    
    # Métricas específicas balísticas
    compute_matching_metrics: bool = True
    matching_threshold: float = 0.5
    compute_eer: bool = True
    
    # Top-K accuracy
    top_k_values: List[int] = field(default_factory=lambda: [1, 3, 5])
    
    # Análisis por clase
    per_class_metrics: bool = True
    class_names: Optional[List[str]] = None
    
    # Visualizaciones
    save_plots: bool = True
    plot_format: str = "png"  # png, pdf, svg
    
    # Reportes
    generate_html_report: bool = True
    include_sample_predictions: bool = True
    num_sample_predictions: int = 20
    
    # Cross-validation
    use_cross_validation: bool = False
    cv_folds: int = 5
    cv_stratified: bool = True
    
    def __post_init__(self):
        """Validación de configuración de evaluación."""
        if self.primary_metric not in ["accuracy", "f1", "precision", "recall", "auc"]:
            raise ValueError(f"Métrica primaria no soportada: {self.primary_metric}")

@dataclass
class ExperimentConfig:
    """Configuración completa del experimento."""
    
    # Información del experimento
    experiment_name: str = "ballistic_experiment"
    description: str = "Experimento de análisis balístico con deep learning"
    version: str = "1.0.0"
    
    # Configuraciones componentes
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    
    # Rutas de salida
    output_dir: str = "deep_learning/experiments"
    checkpoint_dir: str = "checkpoints"
    logs_dir: str = "logs"
    results_dir: str = "results"
    
    # Configuración de hardware
    device: str = "auto"  # auto, cpu, cuda
    mixed_precision: bool = False
    
    # Reproducibilidad
    seed: int = 42
    deterministic: bool = True
    
    # Logging
    log_level: str = "INFO"
    tensorboard_logging: bool = True
    wandb_logging: bool = False
    wandb_project: Optional[str] = None
    
    def __post_init__(self):
        """Configuración post-inicialización."""
        # Crear directorios de salida
        self.experiment_dir = os.path.join(self.output_dir, self.experiment_name)
        self.full_checkpoint_dir = os.path.join(self.experiment_dir, self.checkpoint_dir)
        self.full_logs_dir = os.path.join(self.experiment_dir, self.logs_dir)
        self.full_results_dir = os.path.join(self.experiment_dir, self.results_dir)
        
        # Validar configuración de device
        if self.device not in ["auto", "cpu", "cuda"]:
            raise ValueError(f"Device no soportado: {self.device}")
    
    def create_directories(self):
        """Crear directorios necesarios para el experimento."""
        directories = [
            self.experiment_dir,
            self.full_checkpoint_dir,
            self.full_logs_dir,
            self.full_results_dir
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            logger.info(f"Directorio creado/verificado: {directory}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir configuración a diccionario."""
        return asdict(self)
    
    def save(self, filepath: str):
        """Guardar configuración en archivo JSON."""
        config_dict = self.to_dict()
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Configuración guardada en: {filepath}")
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ExperimentConfig':
        """Crear configuración desde diccionario."""
        # Crear sub-configuraciones
        model_config = ModelConfig(**config_dict.get('model', {}))
        data_config = DataConfig(**config_dict.get('data', {}))
        training_config = TrainingConfig(**config_dict.get('training', {}))
        evaluation_config = EvaluationConfig(**config_dict.get('evaluation', {}))
        
        # Remover sub-configuraciones del diccionario principal
        main_config = {k: v for k, v in config_dict.items() 
                      if k not in ['model', 'data', 'training', 'evaluation']}
        
        return cls(
            model=model_config,
            data=data_config,
            training=training_config,
            evaluation=evaluation_config,
            **main_config
        )
    
    @classmethod
    def load(cls, filepath: str) -> 'ExperimentConfig':
        """Cargar configuración desde archivo JSON."""
        with open(filepath, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        logger.info(f"Configuración cargada desde: {filepath}")
        return cls.from_dict(config_dict)

def create_ballistic_classification_config() -> ExperimentConfig:
    """Crear configuración predefinida para clasificación balística."""
    config = ExperimentConfig(
        experiment_name="ballistic_classification",
        description="Clasificación de imágenes balísticas por tipo de arma"
    )
    
    # Configurar modelo para clasificación
    config.model.model_type = "BallisticCNN"
    config.model.num_classes = 5  # Ajustar según estudios NIST
    config.model.use_attention = True
    
    # Configurar datos
    config.data.use_augmentation = True
    config.data.stratify_by = ["study", "firearm_type"]
    
    # Configurar entrenamiento
    config.training.epochs = 50
    config.training.learning_rate = 0.001
    config.training.use_scheduler = True
    
    return config

def create_ballistic_matching_config() -> ExperimentConfig:
    """Crear configuración predefinida para matching balístico."""
    config = ExperimentConfig(
        experiment_name="ballistic_matching",
        description="Matching de imágenes balísticas usando redes Siamese"
    )
    
    # Configurar modelo para matching
    config.model.model_type = "SiameseNetwork"
    config.model.similarity_metric = "cosine"
    config.model.feature_dim = 512
    
    # Configurar evaluación para matching
    config.evaluation.compute_matching_metrics = True
    config.evaluation.compute_cmc_curves = True
    config.evaluation.compute_eer = True
    
    return config

# Test rápido
def quick_test():
    """Test rápido de las configuraciones."""
    try:
        # Test configuración básica
        config = Experimentget_unified_config()
        print("✓ Configuración básica creada")
        
        # Test configuración de clasificación
        class_config = create_ballistic_classification_config()
        print("✓ Configuración de clasificación creada")
        
        # Test configuración de matching
        match_config = create_ballistic_matching_config()
        print("✓ Configuración de matching creada")
        
        # Test serialización
        config_dict = config.to_dict()
        config_restored = ExperimentConfig.from_dict(config_dict)
        print("✓ Serialización/deserialización exitosa")
        
        print("✅ Todas las pruebas de configuración pasaron")
        return True
        
    except Exception as e:
        print(f"❌ Error en pruebas de configuración: {e}")
        return False

if __name__ == "__main__":
    quick_test()