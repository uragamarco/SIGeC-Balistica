"""
Módulo de modelos de Deep Learning - SIGeC-Balistica
===========================================

Este módulo contiene todas las arquitecturas de modelos especializadas para balística:
- CNN para extracción de características
- Redes Siamese para matching
- U-Net para segmentación de ROI
- Transformers para análisis secuencial
- Clasificadores jerárquicos

Componentes principales:
- cnn_models.py: Modelos CNN base y especializados
- siamese_models.py: Redes Siamese y Triplet
- unet_models.py: Modelos de segmentación U-Net
- transformer_models.py: Modelos basados en Transformer
- hierarchical_models.py: Clasificadores jerárquicos
"""

__version__ = "1.0.0"
__author__ = "SIGeC-BalisticaTeam"

from .cnn_models import (
    BallisticCNN,
    ResNetFeatureExtractor,
    EfficientNetFeatureExtractor,
    MultiScaleFeatureExtractor
)

from .siamese_models import (
    SiameseNetwork,
    TripletNetwork,
    ContrastiveLoss,
    BallisticMatcher
)

# Comentar importaciones de segmentation_models hasta que se implemente
# from .segmentation_models import (
#     BallisticUNet,
#     AttentionUNet,
#     ROISegmentationModel
# )

# Comentar importaciones de hierarchical_models hasta que se implemente
# from .hierarchical_models import (
#     HierarchicalClassifier,
#     ManufacturerClassifier,
#     WeaponTypeClassifier
# )

__all__ = [
    # CNN Models
    'BallisticCNN',
    'ResNetFeatureExtractor', 
    'EfficientNetFeatureExtractor',
    'MultiScaleFeatureExtractor',
    
    # Siamese/Triplet Models
    'SiameseNetwork',
    'TripletNetwork',
    'ContrastiveLoss',
    'BallisticMatcher',
    
    # U-Net Models (comentado hasta implementación)
    # 'BallisticUNet',
    # 'AttentionUNet',
    # 'ROISegmentationModel',
    
    # Hierarchical Models (comentado hasta implementación)
    # 'HierarchicalClassifier',
    # 'ManufacturerClassifier',
    # 'WeaponTypeClassifier'
]