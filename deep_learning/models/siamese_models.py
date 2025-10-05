"""
Redes Siamese y Triplet para matching balístico - SEACABAr
=========================================================

Implementación de arquitecturas especializadas para comparación de imágenes balísticas,
incluyendo redes Siamese, Triplet y métricas de similitud especializadas.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
import logging
from .cnn_models import BallisticCNN, ResNetFeatureExtractor, EfficientNetFeatureExtractor

logger = logging.getLogger(__name__)


class SiameseNetwork(nn.Module):
    """
    Red Siamese para comparación de imágenes balísticas
    
    Arquitectura que aprende a determinar si dos imágenes provienen
    del mismo arma de fuego basándose en características microscópicas.
    """
    
    def __init__(self,
                 backbone_config: Dict,
                 feature_dim: int = 512,
                 similarity_method: str = 'cosine',
                 temperature: float = 0.1):
        """
        Inicializar red Siamese
        
        Args:
            backbone_config: Configuración del backbone CNN
            feature_dim: Dimensión del vector de características
            similarity_method: Método de similitud ('cosine', 'euclidean', 'learned')
            temperature: Temperatura para softmax en similitud
        """
        super(SiameseNetwork, self).__init__()
        
        self.feature_dim = feature_dim
        self.similarity_method = similarity_method
        self.temperature = temperature
        
        # Backbone compartido para extracción de características
        self.backbone = self._create_backbone(backbone_config)
        
        # Capa de proyección para características finales
        self.projection = nn.Sequential(
            nn.Linear(self._get_backbone_dim(), feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(feature_dim, feature_dim)
        )
        
        # Normalizador L2 para características
        self.l2_norm = nn.functional.normalize
        
        # Red de similitud aprendida (si se usa)
        if similarity_method == 'learned':
            self.similarity_net = nn.Sequential(
                nn.Linear(feature_dim * 2, feature_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(feature_dim, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 1),
                nn.Sigmoid()
            )
        
        logger.info(f"SiameseNetwork inicializada: feature_dim={feature_dim}, "
                   f"similarity={similarity_method}, temperature={temperature}")
    
    def _create_backbone(self, config: Dict) -> nn.Module:
        """Crear backbone según configuración"""
        backbone_type = config.get('type', 'ballistic_cnn')
        
        if backbone_type == 'ballistic_cnn':
            return BallisticCNN(
                input_channels=config.get('input_channels', 3),
                num_classes=config.get('num_classes', 10),
                feature_dim=config.get('backbone_feature_dim', 512),
                use_attention=config.get('use_attention', True)
            )
        elif backbone_type == 'resnet':
            return ResNetFeatureExtractor(
                model_name=config.get('model_name', 'resnet50'),
                pretrained=config.get('pretrained', True),
                feature_dim=config.get('backbone_feature_dim', 512)
            )
        elif backbone_type == 'efficientnet':
            return EfficientNetFeatureExtractor(
                model_name=config.get('model_name', 'efficientnet_b0'),
                pretrained=config.get('pretrained', True),
                feature_dim=config.get('backbone_feature_dim', 512)
            )
        else:
            raise ValueError(f"Backbone no soportado: {backbone_type}")
    
    def _get_backbone_dim(self) -> int:
        """Obtener dimensión de salida del backbone"""
        if hasattr(self.backbone, 'feature_dim'):
            return self.backbone.feature_dim
        elif hasattr(self.backbone, 'classifier') and len(self.backbone.classifier) > 1:
            # Para BallisticCNN
            return self.backbone.classifier[1].out_features
        else:
            return 512  # Default
    
    def forward_once(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass para una imagen
        
        Args:
            x: Tensor de imagen [batch_size, channels, height, width]
        
        Returns:
            Vector de características normalizado
        """
        # Extracción de características con backbone
        if isinstance(self.backbone, BallisticCNN):
            backbone_output = self.backbone(x)
            features = backbone_output['features']
        else:
            features = self.backbone(x)
        
        # Proyección
        features = self.projection(features)
        
        # Normalización L2
        features = self.l2_norm(features, p=2, dim=1)
        
        return features
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass para par de imágenes
        
        Args:
            x1: Primera imagen del par
            x2: Segunda imagen del par
        
        Returns:
            Dict con características y similitud
        """
        # Extraer características de ambas imágenes
        features1 = self.forward_once(x1)
        features2 = self.forward_once(x2)
        
        # Calcular similitud
        similarity = self.compute_similarity(features1, features2)
        
        return {
            'features1': features1,
            'features2': features2,
            'similarity': similarity,
            'distance': 1.0 - similarity if self.similarity_method != 'euclidean' else similarity
        }
    
    def compute_similarity(self, features1: torch.Tensor, features2: torch.Tensor) -> torch.Tensor:
        """
        Calcular similitud entre vectores de características
        
        Args:
            features1: Características de la primera imagen
            features2: Características de la segunda imagen
        
        Returns:
            Tensor de similitud
        """
        if self.similarity_method == 'cosine':
            # Similitud coseno (ya normalizadas las características)
            similarity = torch.sum(features1 * features2, dim=1)
            
        elif self.similarity_method == 'euclidean':
            # Distancia euclidiana
            distance = torch.norm(features1 - features2, p=2, dim=1)
            similarity = torch.exp(-distance / self.temperature)
            
        elif self.similarity_method == 'learned':
            # Similitud aprendida
            combined_features = torch.cat([features1, features2], dim=1)
            similarity = self.similarity_net(combined_features).squeeze()
            
        else:
            raise ValueError(f"Método de similitud no soportado: {self.similarity_method}")
        
        return similarity


class TripletNetwork(nn.Module):
    """
    Red Triplet para aprendizaje de embeddings balísticos
    
    Aprende representaciones donde imágenes del mismo arma están más cerca
    que imágenes de armas diferentes en el espacio de características.
    """
    
    def __init__(self,
                 backbone_config: Dict,
                 feature_dim: int = 512,
                 margin: float = 1.0,
                 mining_strategy: str = 'hard'):
        """
        Inicializar red Triplet
        
        Args:
            backbone_config: Configuración del backbone CNN
            feature_dim: Dimensión del vector de características
            margin: Margen para triplet loss
            mining_strategy: Estrategia de minería ('hard', 'semi_hard', 'easy')
        """
        super(TripletNetwork, self).__init__()
        
        self.feature_dim = feature_dim
        self.margin = margin
        self.mining_strategy = mining_strategy
        
        # Backbone compartido
        self.backbone = self._create_backbone(backbone_config)
        
        # Capa de proyección
        self.projection = nn.Sequential(
            nn.Linear(self._get_backbone_dim(), feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(feature_dim, feature_dim)
        )
        
        # Triplet loss
        self.triplet_loss = nn.TripletMarginLoss(margin=margin, p=2)
        
        logger.info(f"TripletNetwork inicializada: feature_dim={feature_dim}, "
                   f"margin={margin}, mining={mining_strategy}")
    
    def _create_backbone(self, config: Dict) -> nn.Module:
        """Crear backbone según configuración"""
        # Misma lógica que SiameseNetwork
        backbone_type = config.get('type', 'ballistic_cnn')
        
        if backbone_type == 'ballistic_cnn':
            return BallisticCNN(
                input_channels=config.get('input_channels', 3),
                num_classes=config.get('num_classes', 10),
                feature_dim=config.get('backbone_feature_dim', 512),
                use_attention=config.get('use_attention', True)
            )
        elif backbone_type == 'resnet':
            return ResNetFeatureExtractor(
                model_name=config.get('model_name', 'resnet50'),
                pretrained=config.get('pretrained', True),
                feature_dim=config.get('backbone_feature_dim', 512)
            )
        else:
            raise ValueError(f"Backbone no soportado: {backbone_type}")
    
    def _get_backbone_dim(self) -> int:
        """Obtener dimensión de salida del backbone"""
        if hasattr(self.backbone, 'feature_dim'):
            return self.backbone.feature_dim
        elif hasattr(self.backbone, 'classifier') and len(self.backbone.classifier) > 1:
            return self.backbone.classifier[1].out_features
        else:
            return 512
    
    def forward_once(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass para una imagen
        
        Args:
            x: Tensor de imagen
        
        Returns:
            Vector de características normalizado
        """
        # Extracción de características
        if isinstance(self.backbone, BallisticCNN):
            backbone_output = self.backbone(x)
            features = backbone_output['features']
        else:
            features = self.backbone(x)
        
        # Proyección
        features = self.projection(features)
        
        # Normalización L2
        features = F.normalize(features, p=2, dim=1)
        
        return features
    
    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, 
                negative: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass para triplet
        
        Args:
            anchor: Imagen ancla
            positive: Imagen positiva (misma clase que ancla)
            negative: Imagen negativa (clase diferente)
        
        Returns:
            Dict con características y pérdida
        """
        # Extraer características
        anchor_features = self.forward_once(anchor)
        positive_features = self.forward_once(positive)
        negative_features = self.forward_once(negative)
        
        # Calcular pérdida triplet
        loss = self.triplet_loss(anchor_features, positive_features, negative_features)
        
        # Calcular distancias
        pos_distance = F.pairwise_distance(anchor_features, positive_features, p=2)
        neg_distance = F.pairwise_distance(anchor_features, negative_features, p=2)
        
        return {
            'anchor_features': anchor_features,
            'positive_features': positive_features,
            'negative_features': negative_features,
            'triplet_loss': loss,
            'positive_distance': pos_distance,
            'negative_distance': neg_distance,
            'margin_violation': (pos_distance - neg_distance + self.margin > 0).float()
        }
    
    def mine_triplets(self, features: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Minería de triplets según estrategia
        
        Args:
            features: Características extraídas [batch_size, feature_dim]
            labels: Etiquetas [batch_size]
        
        Returns:
            Índices de anchor, positive, negative
        """
        batch_size = features.size(0)
        
        # Calcular matriz de distancias
        distances = torch.cdist(features, features, p=2)
        
        anchors, positives, negatives = [], [], []
        
        for i in range(batch_size):
            anchor_label = labels[i]
            
            # Encontrar positivos (misma etiqueta, diferente índice)
            positive_mask = (labels == anchor_label) & (torch.arange(batch_size) != i)
            positive_indices = torch.where(positive_mask)[0]
            
            # Encontrar negativos (etiqueta diferente)
            negative_mask = labels != anchor_label
            negative_indices = torch.where(negative_mask)[0]
            
            if len(positive_indices) == 0 or len(negative_indices) == 0:
                continue
            
            if self.mining_strategy == 'hard':
                # Hard positive: el más lejano de la misma clase
                pos_distances = distances[i, positive_indices]
                hardest_positive = positive_indices[torch.argmax(pos_distances)]
                
                # Hard negative: el más cercano de clase diferente
                neg_distances = distances[i, negative_indices]
                hardest_negative = negative_indices[torch.argmin(neg_distances)]
                
                anchors.append(i)
                positives.append(hardest_positive.item())
                negatives.append(hardest_negative.item())
                
            elif self.mining_strategy == 'semi_hard':
                # Semi-hard negative: más cercano que el positive pero de clase diferente
                pos_distances = distances[i, positive_indices]
                closest_positive_dist = torch.min(pos_distances)
                
                neg_distances = distances[i, negative_indices]
                semi_hard_mask = (neg_distances > closest_positive_dist) & (neg_distances < closest_positive_dist + self.margin)
                semi_hard_negatives = negative_indices[semi_hard_mask]
                
                if len(semi_hard_negatives) > 0:
                    random_positive = positive_indices[torch.randint(len(positive_indices), (1,))]
                    random_negative = semi_hard_negatives[torch.randint(len(semi_hard_negatives), (1,))]
                    
                    anchors.append(i)
                    positives.append(random_positive.item())
                    negatives.append(random_negative.item())
        
        if len(anchors) == 0:
            # Fallback a selección aleatoria
            return self._random_triplet_selection(batch_size, labels)
        
        return torch.tensor(anchors), torch.tensor(positives), torch.tensor(negatives)
    
    def _random_triplet_selection(self, batch_size: int, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Selección aleatoria de triplets como fallback"""
        anchors, positives, negatives = [], [], []
        
        for i in range(batch_size):
            anchor_label = labels[i]
            
            positive_indices = torch.where((labels == anchor_label) & (torch.arange(batch_size) != i))[0]
            negative_indices = torch.where(labels != anchor_label)[0]
            
            if len(positive_indices) > 0 and len(negative_indices) > 0:
                anchors.append(i)
                positives.append(positive_indices[torch.randint(len(positive_indices), (1,))].item())
                negatives.append(negative_indices[torch.randint(len(negative_indices), (1,))].item())
        
        return torch.tensor(anchors), torch.tensor(positives), torch.tensor(negatives)


class ContrastiveLoss(nn.Module):
    """
    Pérdida contrastiva para redes Siamese
    """
    
    def __init__(self, margin: float = 1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    
    def forward(self, output1: torch.Tensor, output2: torch.Tensor, 
                label: torch.Tensor) -> torch.Tensor:
        """
        Calcular pérdida contrastiva
        
        Args:
            output1: Características de la primera imagen
            output2: Características de la segunda imagen
            label: 1 si son de la misma clase, 0 si no
        
        Returns:
            Pérdida contrastiva
        """
        euclidean_distance = F.pairwise_distance(output1, output2, p=2)
        
        loss_contrastive = torch.mean(
            label * torch.pow(euclidean_distance, 2) +
            (1 - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )
        
        return loss_contrastive


class BallisticMatcher(nn.Module):
    """
    Sistema completo de matching balístico
    
    Combina múltiples arquitecturas y métricas para comparación robusta
    """
    
    def __init__(self,
                 siamese_config: Dict,
                 triplet_config: Dict,
                 ensemble_weights: Optional[List[float]] = None):
        """
        Inicializar matcher balístico
        
        Args:
            siamese_config: Configuración de red Siamese
            triplet_config: Configuración de red Triplet
            ensemble_weights: Pesos para ensemble de modelos
        """
        super(BallisticMatcher, self).__init__()
        
        # Redes especializadas
        self.siamese_net = SiameseNetwork(**siamese_config)
        self.triplet_net = TripletNetwork(**triplet_config)
        
        # Pesos de ensemble
        self.ensemble_weights = ensemble_weights or [0.6, 0.4]
        
        logger.info("BallisticMatcher inicializado con redes Siamese y Triplet")
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass para matching
        
        Args:
            x1: Primera imagen
            x2: Segunda imagen
        
        Returns:
            Dict con similitudes y características
        """
        # Similitud Siamese
        siamese_output = self.siamese_net(x1, x2)
        siamese_similarity = siamese_output['similarity']
        
        # Características Triplet
        triplet_features1 = self.triplet_net.forward_once(x1)
        triplet_features2 = self.triplet_net.forward_once(x2)
        triplet_similarity = F.cosine_similarity(triplet_features1, triplet_features2)
        
        # Ensemble de similitudes
        ensemble_similarity = (
            self.ensemble_weights[0] * siamese_similarity +
            self.ensemble_weights[1] * triplet_similarity
        )
        
        return {
            'siamese_similarity': siamese_similarity,
            'triplet_similarity': triplet_similarity,
            'ensemble_similarity': ensemble_similarity,
            'siamese_features1': siamese_output['features1'],
            'siamese_features2': siamese_output['features2'],
            'triplet_features1': triplet_features1,
            'triplet_features2': triplet_features2
        }


def create_siamese_model(config: Dict) -> SiameseNetwork:
    """Factory function para crear modelo Siamese"""
    return SiameseNetwork(
        backbone_config=config['backbone'],
        feature_dim=config.get('feature_dim', 512),
        similarity_method=config.get('similarity_method', 'cosine'),
        temperature=config.get('temperature', 0.1)
    )


def create_triplet_model(config: Dict) -> TripletNetwork:
    """Factory function para crear modelo Triplet"""
    return TripletNetwork(
        backbone_config=config['backbone'],
        feature_dim=config.get('feature_dim', 512),
        margin=config.get('margin', 1.0),
        mining_strategy=config.get('mining_strategy', 'hard')
    )


if __name__ == "__main__":
    # Ejemplo de uso y pruebas
    print("=== PRUEBAS DE MODELOS SIAMESE Y TRIPLET ===")
    
    # Configuración de backbone
    backbone_config = {
        'type': 'ballistic_cnn',
        'input_channels': 3,
        'num_classes': 10,
        'backbone_feature_dim': 256,
        'use_attention': True
    }
    
    # Crear tensores de prueba
    batch_size = 4
    x1 = torch.randn(batch_size, 3, 224, 224)
    x2 = torch.randn(batch_size, 3, 224, 224)
    x3 = torch.randn(batch_size, 3, 224, 224)
    
    print(f"Tensores de entrada: {x1.shape}")
    
    # Probar SiameseNetwork
    print("\n1. SiameseNetwork:")
    siamese_model = SiameseNetwork(backbone_config, feature_dim=128)
    siamese_output = siamese_model(x1, x2)
    print(f"   Similitud: {siamese_output['similarity'].shape}")
    print(f"   Features1: {siamese_output['features1'].shape}")
    print(f"   Features2: {siamese_output['features2'].shape}")
    
    # Probar TripletNetwork
    print("\n2. TripletNetwork:")
    triplet_model = TripletNetwork(backbone_config, feature_dim=128, margin=0.5)
    triplet_output = triplet_model(x1, x2, x3)
    print(f"   Triplet loss: {triplet_output['triplet_loss'].item():.4f}")
    print(f"   Positive distance: {triplet_output['positive_distance'].mean().item():.4f}")
    print(f"   Negative distance: {triplet_output['negative_distance'].mean().item():.4f}")
    
    # Probar ContrastiveLoss
    print("\n3. ContrastiveLoss:")
    contrastive_loss = ContrastiveLoss(margin=1.0)
    labels = torch.randint(0, 2, (batch_size,)).float()
    features1 = torch.randn(batch_size, 128)
    features2 = torch.randn(batch_size, 128)
    loss = contrastive_loss(features1, features2, labels)
    print(f"   Contrastive loss: {loss.item():.4f}")
    
    # Probar BallisticMatcher
    print("\n4. BallisticMatcher:")
    siamese_config = {
        'backbone_config': backbone_config,
        'feature_dim': 128,
        'similarity_method': 'cosine'
    }
    triplet_config = {
        'backbone_config': backbone_config,
        'feature_dim': 128,
        'margin': 0.5
    }
    
    matcher = BallisticMatcher(siamese_config, triplet_config)
    matcher_output = matcher(x1, x2)
    print(f"   Ensemble similarity: {matcher_output['ensemble_similarity'].shape}")
    print(f"   Siamese similarity: {matcher_output['siamese_similarity'].shape}")
    print(f"   Triplet similarity: {matcher_output['triplet_similarity'].shape}")
    
    print("\n✅ Todas las pruebas completadas exitosamente!")