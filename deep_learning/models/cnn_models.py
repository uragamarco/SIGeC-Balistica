"""
Modelos CNN para extracción de características balísticas - SIGeC-Balistica
==================================================================

Implementación de arquitecturas CNN especializadas para imágenes balísticas,
incluyendo modelos base y extractores de características multi-escala.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import timm
from typing import Dict, List, Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)


class BallisticCNN(nn.Module):
    """
    CNN base especializado para imágenes balísticas
    
    Arquitectura optimizada para:
    - Detección de características microscópicas
    - Patrones de estriado y marcas de percusión
    - Invarianza a rotación y escala
    """
    
    def __init__(self,
                 input_channels: int = 3,
                 num_classes: int = 10,
                 feature_dim: int = 512,
                 dropout_rate: float = 0.3,
                 use_attention: bool = True):
        """
        Inicializar CNN balístico
        
        Args:
            input_channels: Canales de entrada (3 para RGB)
            num_classes: Número de clases para clasificación
            feature_dim: Dimensión del vector de características
            dropout_rate: Tasa de dropout
            use_attention: Usar mecanismo de atención
        """
        super(BallisticCNN, self).__init__()
        
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.use_attention = use_attention
        
        # Capas convolucionales con filtros especializados
        self.conv_layers = self._build_conv_layers()
        
        # Mecanismo de atención espacial
        if self.use_attention:
            self.attention = SpatialAttention()
        
        # Capas de clasificación
        self.classifier = self._build_classifier()
        
        # Inicialización de pesos
        self._initialize_weights()
        
        logger.info(f"BallisticCNN inicializado: {num_classes} clases, "
                   f"feature_dim={feature_dim}, attention={use_attention}")
    
    def _build_conv_layers(self) -> nn.Sequential:
        """Construir capas convolucionales especializadas"""
        layers = []
        
        # Bloque 1: Detección de características básicas
        layers.extend([
            nn.Conv2d(self.input_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        ])
        
        # Bloque 2: Características de nivel medio
        layers.extend([
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ])
        
        # Bloque 3: Características específicas de balística
        layers.extend([
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ])
        
        # Bloque 4: Características de alto nivel
        layers.extend([
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((7, 7))
        ])
        
        return nn.Sequential(*layers)
    
    def _build_classifier(self) -> nn.Sequential:
        """Construir capas de clasificación"""
        return nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512 * 7 * 7, self.feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(self.feature_dim, self.num_classes)
        )
    
    def _initialize_weights(self):
        """Inicializar pesos de la red"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Tensor de entrada [batch_size, channels, height, width]
        
        Returns:
            Dict con logits y características
        """
        # Extracción de características
        features = self.conv_layers(x)
        
        # Aplicar atención si está habilitada
        if self.use_attention:
            features = self.attention(features)
        
        # Aplanar características
        features_flat = features.view(features.size(0), -1)
        
        # Clasificación
        logits = self.classifier(features_flat)
        
        return {
            'logits': logits,
            'features': features_flat,
            'feature_maps': features
        }
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extraer solo características sin clasificación
        
        Args:
            x: Tensor de entrada
        
        Returns:
            Vector de características
        """
        with torch.no_grad():
            features = self.conv_layers(x)
            if self.use_attention:
                features = self.attention(features)
            features_flat = features.view(features.size(0), -1)
            
            # Pasar por la primera parte del clasificador para obtener features
            features_processed = self.classifier[1](self.classifier[0](features_flat))
            return features_processed


class SpatialAttention(nn.Module):
    """Mecanismo de atención espacial para CNN balístico"""
    
    def __init__(self, kernel_size: int = 7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Calcular estadísticas espaciales
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # Concatenar y procesar
        attention_input = torch.cat([avg_out, max_out], dim=1)
        attention_weights = self.sigmoid(self.conv(attention_input))
        
        return x * attention_weights


class ResNetFeatureExtractor(nn.Module):
    """
    Extractor de características basado en ResNet preentrenado
    """
    
    def __init__(self,
                 model_name: str = 'resnet50',
                 pretrained: bool = True,
                 feature_dim: int = 512,
                 freeze_backbone: bool = False):
        """
        Inicializar extractor ResNet
        
        Args:
            model_name: Nombre del modelo ResNet
            pretrained: Usar pesos preentrenados
            feature_dim: Dimensión final de características
            freeze_backbone: Congelar backbone preentrenado
        """
        super(ResNetFeatureExtractor, self).__init__()
        
        # Cargar modelo base
        if model_name == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            backbone_dim = 512
        elif model_name == 'resnet34':
            self.backbone = models.resnet34(pretrained=pretrained)
            backbone_dim = 512
        elif model_name == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            backbone_dim = 2048
        elif model_name == 'resnet101':
            self.backbone = models.resnet101(pretrained=pretrained)
            backbone_dim = 2048
        else:
            raise ValueError(f"Modelo no soportado: {model_name}")
        
        # Remover la capa de clasificación final
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # Congelar backbone si se solicita
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Capa de proyección
        self.projection = nn.Sequential(
            nn.Linear(backbone_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        
        self.feature_dim = feature_dim
        logger.info(f"ResNetFeatureExtractor inicializado: {model_name}, "
                   f"pretrained={pretrained}, feature_dim={feature_dim}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Tensor de entrada
        
        Returns:
            Vector de características
        """
        # Extracción con backbone
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        
        # Proyección
        features = self.projection(features)
        
        return features


class EfficientNetFeatureExtractor(nn.Module):
    """
    Extractor de características basado en EfficientNet
    """
    
    def __init__(self,
                 model_name: str = 'efficientnet_b0',
                 pretrained: bool = True,
                 feature_dim: int = 512,
                 freeze_backbone: bool = False):
        """
        Inicializar extractor EfficientNet
        
        Args:
            model_name: Nombre del modelo EfficientNet
            pretrained: Usar pesos preentrenados
            feature_dim: Dimensión final de características
            freeze_backbone: Congelar backbone preentrenado
        """
        super(EfficientNetFeatureExtractor, self).__init__()
        
        # Cargar modelo usando timm
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Sin clasificador
            global_pool='avg'
        )
        
        # Obtener dimensión del backbone
        backbone_dim = self.backbone.num_features
        
        # Congelar backbone si se solicita
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Capa de proyección
        self.projection = nn.Sequential(
            nn.Linear(backbone_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        
        self.feature_dim = feature_dim
        logger.info(f"EfficientNetFeatureExtractor inicializado: {model_name}, "
                   f"pretrained={pretrained}, feature_dim={feature_dim}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Tensor de entrada
        
        Returns:
            Vector de características
        """
        # Extracción con backbone
        features = self.backbone(x)
        
        # Proyección
        features = self.projection(features)
        
        return features


class MultiScaleFeatureExtractor(nn.Module):
    """
    Extractor de características multi-escala para capturar detalles a diferentes niveles
    """
    
    def __init__(self,
                 input_channels: int = 3,
                 feature_dim: int = 512,
                 scales: List[int] = [1, 2, 4]):
        """
        Inicializar extractor multi-escala
        
        Args:
            input_channels: Canales de entrada
            feature_dim: Dimensión final de características
            scales: Escalas para procesamiento multi-escala
        """
        super(MultiScaleFeatureExtractor, self).__init__()
        
        self.scales = scales
        self.feature_dim = feature_dim
        
        # Extractores para cada escala
        self.scale_extractors = nn.ModuleList()
        for scale in scales:
            extractor = self._build_scale_extractor(input_channels, scale)
            self.scale_extractors.append(extractor)
        
        # Fusión de características multi-escala
        fusion_dim = len(scales) * 256  # Cada extractor produce 256 features
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(feature_dim, feature_dim)
        )
        
        logger.info(f"MultiScaleFeatureExtractor inicializado: scales={scales}, "
                   f"feature_dim={feature_dim}")
    
    def _build_scale_extractor(self, input_channels: int, scale: int) -> nn.Module:
        """Construir extractor para una escala específica"""
        return nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3*scale, stride=scale, padding=scale),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass multi-escala
        
        Args:
            x: Tensor de entrada
        
        Returns:
            Vector de características fusionadas
        """
        scale_features = []
        
        # Extraer características en cada escala
        for extractor in self.scale_extractors:
            features = extractor(x)
            scale_features.append(features)
        
        # Concatenar características de todas las escalas
        combined_features = torch.cat(scale_features, dim=1)
        
        # Fusionar características
        fused_features = self.fusion(combined_features)
        
        return fused_features


def create_cnn_model(config: Dict) -> nn.Module:
    """
    Factory function para crear modelos CNN
    
    Args:
        config: Configuración del modelo
    
    Returns:
        Modelo CNN configurado
    """
    model_type = config.get('type', 'ballistic_cnn')
    
    if model_type == 'ballistic_cnn':
        return BallisticCNN(
            input_channels=config.get('input_channels', 3),
            num_classes=config.get('num_classes', 10),
            feature_dim=config.get('feature_dim', 512),
            dropout_rate=config.get('dropout_rate', 0.3),
            use_attention=config.get('use_attention', True)
        )
    
    elif model_type == 'resnet':
        return ResNetFeatureExtractor(
            model_name=config.get('model_name', 'resnet50'),
            pretrained=config.get('pretrained', True),
            feature_dim=config.get('feature_dim', 512),
            freeze_backbone=config.get('freeze_backbone', False)
        )
    
    elif model_type == 'efficientnet':
        return EfficientNetFeatureExtractor(
            model_name=config.get('model_name', 'efficientnet_b0'),
            pretrained=config.get('pretrained', True),
            feature_dim=config.get('feature_dim', 512),
            freeze_backbone=config.get('freeze_backbone', False)
        )
    
    elif model_type == 'multiscale':
        return MultiScaleFeatureExtractor(
            input_channels=config.get('input_channels', 3),
            feature_dim=config.get('feature_dim', 512),
            scales=config.get('scales', [1, 2, 4])
        )
    
    else:
        raise ValueError(f"Tipo de modelo no soportado: {model_type}")


if __name__ == "__main__":
    # Ejemplo de uso y pruebas
    print("=== PRUEBAS DE MODELOS CNN ===")
    
    # Crear tensor de prueba
    batch_size = 4
    channels = 3
    height, width = 224, 224
    x = torch.randn(batch_size, channels, height, width)
    
    print(f"Tensor de entrada: {x.shape}")
    
    # Probar BallisticCNN
    print("\n1. BallisticCNN:")
    model1 = BallisticCNN(num_classes=5, feature_dim=256)
    output1 = model1(x)
    print(f"   Logits: {output1['logits'].shape}")
    print(f"   Features: {output1['features'].shape}")
    print(f"   Feature maps: {output1['feature_maps'].shape}")
    
    # Probar ResNetFeatureExtractor
    print("\n2. ResNetFeatureExtractor:")
    model2 = ResNetFeatureExtractor(model_name='resnet18', feature_dim=256)
    output2 = model2(x)
    print(f"   Features: {output2.shape}")
    
    # Probar EfficientNetFeatureExtractor
    print("\n3. EfficientNetFeatureExtractor:")
    try:
        model3 = EfficientNetFeatureExtractor(model_name='efficientnet_b0', feature_dim=256)
        output3 = model3(x)
        print(f"   Features: {output3.shape}")
    except Exception as e:
        print(f"   Error (timm no disponible): {e}")
    
    # Probar MultiScaleFeatureExtractor
    print("\n4. MultiScaleFeatureExtractor:")
    model4 = MultiScaleFeatureExtractor(feature_dim=256, scales=[1, 2])
    output4 = model4(x)
    print(f"   Features: {output4.shape}")
    
    # Probar factory function
    print("\n5. Factory function:")
    config = {
        'type': 'ballistic_cnn',
        'num_classes': 8,
        'feature_dim': 128,
        'use_attention': True
    }
    model5 = create_cnn_model(config)
    output5 = model5(x)
    print(f"   Logits: {output5['logits'].shape}")
    
    print("\n✅ Todas las pruebas completadas exitosamente!")