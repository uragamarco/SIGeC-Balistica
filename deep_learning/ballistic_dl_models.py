"""
Modelos de Deep Learning para Análisis Balístico
Sistema Balístico Forense MVP

Implementa y evalúa modelos de deep learning para:
- Extracción de características profundas
- Matching automático de imágenes balísticas
- Segmentación de ROI con redes neuronales
- Clasificación de armas de fuego

Modelos implementados:
- CNN para extracción de características
- Siamese Networks para comparación
- U-Net para segmentación de ROI
- ResNet para clasificación
- Autoencoder para reducción dimensional

Basado en literatura científica:
- Song et al. (2018) - CNN para análisis balístico
- Chu et al. (2020) - Siamese networks para matching
- Ronneberger et al. (2015) - U-Net para segmentación
- He et al. (2016) - ResNet para clasificación profunda
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import logging
from pathlib import Path
from enum import Enum
import json
import time
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

class ModelType(Enum):
    """Tipos de modelos de deep learning"""
    CNN_FEATURE_EXTRACTOR = "cnn_feature_extractor"
    SIAMESE_NETWORK = "siamese_network"
    UNET_SEGMENTATION = "unet_segmentation"
    RESNET_CLASSIFIER = "resnet_classifier"
    AUTOENCODER = "autoencoder"

@dataclass
class ModelConfig:
    """Configuración de modelo"""
    model_type: ModelType
    input_size: Tuple[int, int, int]  # (channels, height, width)
    num_classes: int
    learning_rate: float
    batch_size: int
    epochs: int
    device: str
    
    # Parámetros específicos
    feature_dim: int = 512
    dropout_rate: float = 0.5
    use_pretrained: bool = True
    freeze_backbone: bool = False

@dataclass
class TrainingResult:
    """Resultado del entrenamiento"""
    model_type: ModelType
    train_loss: List[float]
    val_loss: List[float]
    train_accuracy: List[float]
    val_accuracy: List[float]
    best_accuracy: float
    training_time: float
    model_path: str
    
    # Métricas adicionales
    precision: float
    recall: float
    f1_score: float
    confusion_matrix: np.ndarray

class BallisticDataset(Dataset):
    """Dataset para imágenes balísticas"""
    
    def __init__(self, image_paths: List[str], labels: List[int], 
                 transform: Optional[transforms.Compose] = None,
                 pairs: bool = False):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.pairs = pairs
        
        if pairs:
            self._create_pairs()
    
    def _create_pairs(self):
        """Crea pares de imágenes para Siamese Network"""
        self.pairs_data = []
        
        # Crear pares positivos (misma clase)
        class_indices = {}
        for i, label in enumerate(self.labels):
            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append(i)
        
        # Pares positivos
        for class_label, indices in class_indices.items():
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    self.pairs_data.append((indices[i], indices[j], 1))  # Similar
        
        # Pares negativos (diferentes clases)
        classes = list(class_indices.keys())
        for i in range(len(classes)):
            for j in range(i + 1, len(classes)):
                class1_indices = class_indices[classes[i]]
                class2_indices = class_indices[classes[j]]
                
                # Tomar muestra de pares negativos
                for idx1 in class1_indices[:min(5, len(class1_indices))]:
                    for idx2 in class2_indices[:min(5, len(class2_indices))]:
                        self.pairs_data.append((idx1, idx2, 0))  # Diferente
    
    def __len__(self):
        if self.pairs:
            return len(self.pairs_data)
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        if self.pairs:
            idx1, idx2, label = self.pairs_data[idx]
            
            image1 = self._load_image(self.image_paths[idx1])
            image2 = self._load_image(self.image_paths[idx2])
            
            if self.transform:
                image1 = self.transform(image1)
                image2 = self.transform(image2)
            
            return (image1, image2), torch.tensor(label, dtype=torch.float32)
        else:
            image = self._load_image(self.image_paths[idx])
            label = self.labels[idx]
            
            if self.transform:
                image = self.transform(image)
            
            return image, torch.tensor(label, dtype=torch.long)
    
    def _load_image(self, path: str) -> np.ndarray:
        """Carga y preprocesa imagen"""
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"No se pudo cargar imagen: {path}")
        
        # Normalizar a [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Convertir a 3 canales si es necesario
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=0)
        
        return image

class CNNFeatureExtractor(nn.Module):
    """CNN para extracción de características balísticas"""
    
    def __init__(self, input_channels: int = 3, feature_dim: int = 512):
        super(CNNFeatureExtractor, self).__init__()
        
        self.features = nn.Sequential(
            # Bloque 1
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Bloque 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Bloque 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Bloque 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((7, 7))
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, feature_dim)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class SiameseNetwork(nn.Module):
    """Red Siamese para comparación de imágenes balísticas"""
    
    def __init__(self, feature_extractor: nn.Module, feature_dim: int = 512):
        super(SiameseNetwork, self).__init__()
        
        self.feature_extractor = feature_extractor
        
        # Capas de comparación
        self.comparison = nn.Sequential(
            nn.Linear(feature_dim * 2, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x1, x2):
        # Extraer características
        feat1 = self.feature_extractor(x1)
        feat2 = self.feature_extractor(x2)
        
        # Concatenar características
        combined = torch.cat([feat1, feat2], dim=1)
        
        # Predicción de similitud
        similarity = self.comparison(combined)
        
        return similarity, feat1, feat2

class UNetSegmentation(nn.Module):
    """U-Net para segmentación de ROI balísticas"""
    
    def __init__(self, input_channels: int = 3, num_classes: int = 4):
        super(UNetSegmentation, self).__init__()
        
        # Encoder
        self.enc1 = self._conv_block(input_channels, 64)
        self.enc2 = self._conv_block(64, 128)
        self.enc3 = self._conv_block(128, 256)
        self.enc4 = self._conv_block(256, 512)
        
        # Bottleneck
        self.bottleneck = self._conv_block(512, 1024)
        
        # Decoder
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = self._conv_block(1024, 512)
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self._conv_block(512, 256)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self._conv_block(256, 128)
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self._conv_block(128, 64)
        
        # Output
        self.output = nn.Conv2d(64, num_classes, kernel_size=1)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def _conv_block(self, in_channels: int, out_channels: int):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc4))
        
        # Decoder
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.dec4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)
        
        output = self.output(dec1)
        
        return output

class ResNetClassifier(nn.Module):
    """ResNet para clasificación de armas"""
    
    def __init__(self, num_classes: int, pretrained: bool = True):
        super(ResNetClassifier, self).__init__()
        
        # Usar ResNet50 preentrenado
        self.backbone = models.resnet50(pretrained=pretrained)
        
        # Modificar primera capa para imágenes en escala de grises
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Modificar capa final
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_features, num_classes)
    
    def forward(self, x):
        return self.backbone(x)

class BallisticAutoencoder(nn.Module):
    """Autoencoder para reducción dimensional y detección de anomalías"""
    
    def __init__(self, input_channels: int = 3, latent_dim: int = 256):
        super(BallisticAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        self.fc_encoder = nn.Linear(512 * 4 * 4, latent_dim)
        self.fc_decoder = nn.Linear(latent_dim, 512 * 4 * 4)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, input_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, 1)
        x = self.fc_encoder(x)
        return x
    
    def decode(self, z):
        z = self.fc_decoder(z)
        z = z.view(-1, 512, 4, 4)
        z = self.decoder(z)
        return z
    
    def forward(self, x):
        latent = self.encode(x)
        reconstructed = self.decode(latent)
        return reconstructed, latent

class BallisticDLTrainer:
    """Entrenador para modelos de deep learning balísticos"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Configurar dispositivo
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Usando dispositivo: {self.device}")
        
        # Inicializar modelo
        self.model = self._create_model()
        self.model.to(self.device)
        
        # Configurar optimizador y loss
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.criterion = self._get_criterion()
        
        # Métricas de entrenamiento
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
    
    def _create_model(self) -> nn.Module:
        """Crea el modelo según la configuración"""
        if self.config.model_type == ModelType.CNN_FEATURE_EXTRACTOR:
            return CNNFeatureExtractor(
                input_channels=self.config.input_size[0],
                feature_dim=self.config.feature_dim
            )
        
        elif self.config.model_type == ModelType.SIAMESE_NETWORK:
            feature_extractor = CNNFeatureExtractor(
                input_channels=self.config.input_size[0],
                feature_dim=self.config.feature_dim
            )
            return SiameseNetwork(feature_extractor, self.config.feature_dim)
        
        elif self.config.model_type == ModelType.UNET_SEGMENTATION:
            return UNetSegmentation(
                input_channels=self.config.input_size[0],
                num_classes=self.config.num_classes
            )
        
        elif self.config.model_type == ModelType.RESNET_CLASSIFIER:
            return ResNetClassifier(
                num_classes=self.config.num_classes,
                pretrained=self.config.use_pretrained
            )
        
        elif self.config.model_type == ModelType.AUTOENCODER:
            return BallisticAutoencoder(
                input_channels=self.config.input_size[0],
                latent_dim=self.config.feature_dim
            )
        
        else:
            raise ValueError(f"Tipo de modelo no soportado: {self.config.model_type}")
    
    def _get_criterion(self):
        """Obtiene la función de pérdida apropiada"""
        if self.config.model_type == ModelType.SIAMESE_NETWORK:
            return nn.BCELoss()
        elif self.config.model_type == ModelType.UNET_SEGMENTATION:
            return nn.CrossEntropyLoss()
        elif self.config.model_type == ModelType.AUTOENCODER:
            return nn.MSELoss()
        else:
            return nn.CrossEntropyLoss()
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> TrainingResult:
        """Entrena el modelo"""
        self.logger.info(f"Iniciando entrenamiento de {self.config.model_type.value}")
        
        start_time = time.time()
        best_val_accuracy = 0.0
        best_model_path = None
        
        for epoch in range(self.config.epochs):
            # Entrenamiento
            train_loss, train_acc = self._train_epoch(train_loader)
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            
            # Validación
            val_loss, val_acc = self._validate_epoch(val_loader)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            # Guardar mejor modelo
            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                best_model_path = f"models/best_{self.config.model_type.value}_epoch_{epoch}.pth"
                self._save_model(best_model_path)
            
            self.logger.info(
                f"Epoch {epoch+1}/{self.config.epochs}: "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
            )
        
        training_time = time.time() - start_time
        
        # Calcular métricas finales
        precision, recall, f1, cm = self._calculate_final_metrics(val_loader)
        
        return TrainingResult(
            model_type=self.config.model_type,
            train_loss=self.train_losses,
            val_loss=self.val_losses,
            train_accuracy=self.train_accuracies,
            val_accuracy=self.val_accuracies,
            best_accuracy=best_val_accuracy,
            training_time=training_time,
            model_path=best_model_path or "",
            precision=precision,
            recall=recall,
            f1_score=f1,
            confusion_matrix=cm
        )
    
    def _train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Entrena una época"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            if isinstance(data, tuple):  # Siamese network
                data1, data2 = data
                data1, data2, target = data1.to(self.device), data2.to(self.device), target.to(self.device)
                
                self.optimizer.zero_grad()
                output, _, _ = self.model(data1, data2)
                loss = self.criterion(output.squeeze(), target)
                
                # Accuracy para clasificación binaria
                predicted = (output.squeeze() > 0.5).float()
                correct += (predicted == target).sum().item()
                
            else:  # Otros modelos
                data, target = data.to(self.device), target.to(self.device)
                
                self.optimizer.zero_grad()
                
                if self.config.model_type == ModelType.AUTOENCODER:
                    output, _ = self.model(data)
                    loss = self.criterion(output, data)
                    # Para autoencoder, usar MSE como métrica
                    correct += data.size(0)  # Placeholder
                else:
                    output = self.model(data)
                    loss = self.criterion(output, target)
                    
                    # Accuracy
                    _, predicted = torch.max(output.data, 1)
                    correct += (predicted == target).sum().item()
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            total += target.size(0)
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def _validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Valida una época"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                if isinstance(data, tuple):  # Siamese network
                    data1, data2 = data
                    data1, data2, target = data1.to(self.device), data2.to(self.device), target.to(self.device)
                    
                    output, _, _ = self.model(data1, data2)
                    loss = self.criterion(output.squeeze(), target)
                    
                    predicted = (output.squeeze() > 0.5).float()
                    correct += (predicted == target).sum().item()
                    
                else:
                    data, target = data.to(self.device), target.to(self.device)
                    
                    if self.config.model_type == ModelType.AUTOENCODER:
                        output, _ = self.model(data)
                        loss = self.criterion(output, data)
                        correct += data.size(0)  # Placeholder
                    else:
                        output = self.model(data)
                        loss = self.criterion(output, target)
                        
                        _, predicted = torch.max(output.data, 1)
                        correct += (predicted == target).sum().item()
                
                total_loss += loss.item()
                total += target.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def _calculate_final_metrics(self, val_loader: DataLoader) -> Tuple[float, float, float, np.ndarray]:
        """Calcula métricas finales"""
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in val_loader:
                if isinstance(data, tuple):  # Siamese network
                    data1, data2 = data
                    data1, data2 = data1.to(self.device), data2.to(self.device)
                    
                    output, _, _ = self.model(data1, data2)
                    predicted = (output.squeeze() > 0.5).float()
                    
                    all_predictions.extend(predicted.cpu().numpy())
                    all_targets.extend(target.numpy())
                    
                else:
                    data = data.to(self.device)
                    
                    if self.config.model_type != ModelType.AUTOENCODER:
                        output = self.model(data)
                        _, predicted = torch.max(output.data, 1)
                        
                        all_predictions.extend(predicted.cpu().numpy())
                        all_targets.extend(target.numpy())
        
        if all_predictions and all_targets:
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_targets, all_predictions, average='weighted'
            )
            
            # Matriz de confusión
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(all_targets, all_predictions)
            
            return precision, recall, f1, cm
        
        return 0.0, 0.0, 0.0, np.array([])
    
    def _save_model(self, path: str):
        """Guarda el modelo"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }, path)
    
    def load_model(self, path: str):
        """Carga el modelo"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

def create_data_transforms(input_size: Tuple[int, int]) -> Dict[str, transforms.Compose]:
    """Crea transformaciones de datos"""
    
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(input_size),
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return {'train': train_transform, 'val': val_transform}

def evaluate_dl_models(image_dir: str, output_dir: str) -> Dict[ModelType, TrainingResult]:
    """
    Evalúa todos los modelos de deep learning
    
    Args:
        image_dir: Directorio con imágenes organizadas por clase
        output_dir: Directorio de salida para modelos y reportes
        
    Returns:
        Diccionario con resultados de cada modelo
    """
    
    # Configurar logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Crear directorio de salida
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Cargar datos
    image_paths, labels = load_ballistic_dataset(image_dir)
    
    if not image_paths:
        logger.error("No se encontraron imágenes")
        return {}
    
    # Dividir datos
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Configuraciones de modelos
    base_config = {
        'input_size': (3, 224, 224),
        'num_classes': len(set(labels)),
        'learning_rate': 0.001,
        'batch_size': 16,
        'epochs': 50,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    model_configs = {
        ModelType.CNN_FEATURE_EXTRACTOR: ModelConfig(
            model_type=ModelType.CNN_FEATURE_EXTRACTOR,
            **base_config
        ),
        ModelType.SIAMESE_NETWORK: ModelConfig(
            model_type=ModelType.SIAMESE_NETWORK,
            **base_config
        ),
        ModelType.RESNET_CLASSIFIER: ModelConfig(
            model_type=ModelType.RESNET_CLASSIFIER,
            **base_config,
            use_pretrained=True
        ),
        ModelType.AUTOENCODER: ModelConfig(
            model_type=ModelType.AUTOENCODER,
            **base_config,
            num_classes=1  # No aplica para autoencoder
        )
    }
    
    results = {}
    
    # Evaluar cada modelo
    for model_type, config in model_configs.items():
        try:
            logger.info(f"Evaluando modelo: {model_type.value}")
            
            # Crear transformaciones
            transforms_dict = create_data_transforms((224, 224))
            
            # Crear datasets
            if model_type == ModelType.SIAMESE_NETWORK:
                train_dataset = BallisticDataset(train_paths, train_labels, transforms_dict['train'], pairs=True)
                val_dataset = BallisticDataset(val_paths, val_labels, transforms_dict['val'], pairs=True)
            else:
                train_dataset = BallisticDataset(train_paths, train_labels, transforms_dict['train'])
                val_dataset = BallisticDataset(val_paths, val_labels, transforms_dict['val'])
            
            # Crear data loaders
            train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
            
            # Entrenar modelo
            trainer = BallisticDLTrainer(config)
            result = trainer.train(train_loader, val_loader)
            
            results[model_type] = result
            
            logger.info(f"Modelo {model_type.value} completado. Mejor accuracy: {result.best_accuracy:.4f}")
            
        except Exception as e:
            logger.error(f"Error evaluando modelo {model_type.value}: {e}")
    
    # Generar reporte
    generate_dl_evaluation_report(results, str(output_path / "dl_evaluation_report.json"))
    
    return results

def load_ballistic_dataset(image_dir: str) -> Tuple[List[str], List[int]]:
    """Carga dataset de imágenes balísticas"""
    image_paths = []
    labels = []
    
    image_dir = Path(image_dir)
    
    # Buscar imágenes organizadas por subdirectorio (clase)
    class_dirs = [d for d in image_dir.iterdir() if d.is_dir()]
    
    for class_idx, class_dir in enumerate(class_dirs):
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.tiff']:
            for image_path in class_dir.glob(ext):
                image_paths.append(str(image_path))
                labels.append(class_idx)
    
    return image_paths, labels

def generate_dl_evaluation_report(results: Dict[ModelType, TrainingResult], output_path: str):
    """Genera reporte de evaluación de modelos DL"""
    
    report = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'models_evaluated': len(results),
        'summary': {},
        'detailed_results': {},
        'recommendations': []
    }
    
    if not results:
        report['error'] = "No se evaluaron modelos"
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        return
    
    # Resumen
    best_accuracy = max(results.values(), key=lambda r: r.best_accuracy)
    fastest_training = min(results.values(), key=lambda r: r.training_time)
    
    report['summary'] = {
        'best_accuracy_model': best_accuracy.model_type.value,
        'best_accuracy': best_accuracy.best_accuracy,
        'fastest_training_model': fastest_training.model_type.value,
        'fastest_training_time': fastest_training.training_time
    }
    
    # Resultados detallados
    for model_type, result in results.items():
        report['detailed_results'][model_type.value] = {
            'best_accuracy': result.best_accuracy,
            'training_time': result.training_time,
            'precision': result.precision,
            'recall': result.recall,
            'f1_score': result.f1_score,
            'final_train_loss': result.train_loss[-1] if result.train_loss else 0,
            'final_val_loss': result.val_loss[-1] if result.val_loss else 0,
            'model_path': result.model_path
        }
    
    # Recomendaciones
    if best_accuracy.best_accuracy > 0.8:
        report['recommendations'].append(
            f"Modelo {best_accuracy.model_type.value} muestra excelente rendimiento "
            f"(accuracy: {best_accuracy.best_accuracy:.3f})"
        )
    
    if fastest_training.training_time < 3600:  # Menos de 1 hora
        report['recommendations'].append(
            f"Modelo {fastest_training.model_type.value} es eficiente para entrenamiento "
            f"(tiempo: {fastest_training.training_time:.1f}s)"
        )
    
    # Guardar reporte
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        image_dir = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else "dl_models"
        
        # Evaluar modelos
        results = evaluate_dl_models(image_dir, output_dir)
        
        print("\n=== EVALUACIÓN DE MODELOS DE DEEP LEARNING ===")
        print(f"Modelos evaluados: {len(results)}")
        
        for model_type, result in results.items():
            print(f"\n{model_type.value}:")
            print(f"  Mejor accuracy: {result.best_accuracy:.4f}")
            print(f"  Tiempo entrenamiento: {result.training_time:.1f}s")
            print(f"  F1-score: {result.f1_score:.4f}")
    else:
        print("Uso: python ballistic_dl_models.py <image_directory> [output_directory]")
        print("Ejemplo: python ballistic_dl_models.py data/ballistic_images dl_models")