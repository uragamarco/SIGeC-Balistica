"""
Módulo de Deep Learning para Análisis Balístico
Sistema Balístico Forense MVP

Este módulo implementa modelos de deep learning especializados para:
- Extracción de características profundas de imágenes balísticas
- Matching automático usando redes neuronales
- Segmentación de ROI con U-Net
- Clasificación de armas con ResNet
- Reducción dimensional con Autoencoders

Modelos disponibles:
- CNNFeatureExtractor: Extracción de características con CNN personalizada
- SiameseNetwork: Comparación de pares de imágenes
- UNetSegmentation: Segmentación de regiones de interés
- ResNetClassifier: Clasificación de tipos de arma
- BallisticAutoencoder: Reducción dimensional y detección de anomalías

Uso:
    from deep_learning.ballistic_dl_models import evaluate_dl_models, ModelType
    
    # Evaluar todos los modelos
    results = evaluate_dl_models("data/images", "output/models")
    
    # Usar modelo específico
    config = ModelConfig(
        model_type=ModelType.CNN_FEATURE_EXTRACTOR,
        input_size=(3, 224, 224),
        num_classes=10,
        learning_rate=0.001,
        batch_size=16,
        epochs=50,
        device='cuda'
    )
    
    trainer = BallisticDLTrainer(config)
    result = trainer.train(train_loader, val_loader)
"""

from .ballistic_dl_models import (
    # Enums y configuraciones
    ModelType,
    ModelConfig,
    TrainingResult,
    
    # Modelos
    CNNFeatureExtractor,
    SiameseNetwork,
    UNetSegmentation,
    ResNetClassifier,
    BallisticAutoencoder,
    
    # Dataset y entrenamiento
    BallisticDataset,
    BallisticDLTrainer,
    
    # Funciones principales
    evaluate_dl_models,
    create_data_transforms,
    load_ballistic_dataset,
    generate_dl_evaluation_report
)

__all__ = [
    'ModelType',
    'ModelConfig',
    'TrainingResult',
    'CNNFeatureExtractor',
    'SiameseNetwork',
    'UNetSegmentation',
    'ResNetClassifier',
    'BallisticAutoencoder',
    'BallisticDataset',
    'BallisticDLTrainer',
    'evaluate_dl_models',
    'create_data_transforms',
    'load_ballistic_dataset',
    'generate_dl_evaluation_report'
]

__version__ = "1.0.0"
__author__ = "Sistema Balístico Forense MVP"