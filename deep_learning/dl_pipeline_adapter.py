"""
Adaptador para integrar modelos de Deep Learning con IPipelineProcessor
"""

import os
import time
import torch
import numpy as np
import cv2
from typing import Dict, Any, Optional, List, Tuple

# Importar la interfaz y clases necesarias
from interfaces.pipeline_interfaces import IPipelineProcessor, ProcessingResult
from deep_learning.ballistic_dl_models import (
    ModelType, SiameseNetwork, CNNFeatureExtractor, 
    UNetSegmentation, ResNetClassifier
)

class DeepLearningPipelineAdapter(IPipelineProcessor):
    """
    Adaptador que implementa IPipelineProcessor para modelos de Deep Learning
    
    Este adaptador permite integrar los modelos de deep learning con el pipeline
    principal del sistema, implementando la interfaz estándar IPipelineProcessor.
    """
    
    def __init__(self):
        """Inicializa el adaptador"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.model_type = None
        self.transform = None
        self.initialized = False
        self.feature_extractor = None
        self.config = {}
    
    def initialize(self, config: Dict[str, Any]) -> bool:
        """
        Inicializa el procesador con la configuración proporcionada
        
        Args:
            config: Diccionario con la configuración del modelo
                - model_type: Tipo de modelo a utilizar
                - model_path: Ruta al modelo pre-entrenado
                - input_size: Tamaño de entrada para las imágenes
                - threshold: Umbral de similitud para matching
        
        Returns:
            bool: True si la inicialización fue exitosa, False en caso contrario
        """
        try:
            self.config = config
            self.model_type = config.get('model_type', 'siamese_network')
            model_path = config.get('model_path', '')
            input_size = config.get('input_size', (224, 224))
            self.threshold = config.get('threshold', 0.7)
            
            # Configurar transformación para preprocesamiento
            self.transform = torch.nn.Sequential(
                torch.nn.Resize(input_size),
                torch.nn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            )
            
            # Cargar el modelo según el tipo
            if self.model_type == 'siamese_network':
                # Primero cargar el extractor de características
                self.feature_extractor = CNNFeatureExtractor(input_channels=3, feature_dim=512)
                self.model = SiameseNetwork(self.feature_extractor, feature_dim=512)
                
                # Cargar pesos pre-entrenados si existen
                if os.path.exists(model_path):
                    checkpoint = torch.load(model_path, map_location=self.device)
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    
            elif self.model_type == 'unet_segmentation':
                self.model = UNetSegmentation(input_channels=3, num_classes=4)
                if os.path.exists(model_path):
                    checkpoint = torch.load(model_path, map_location=self.device)
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    
            elif self.model_type == 'resnet_classifier':
                self.model = ResNetClassifier(num_classes=config.get('num_classes', 10))
                if os.path.exists(model_path):
                    checkpoint = torch.load(model_path, map_location=self.device)
                    self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # Mover modelo a dispositivo y modo evaluación
            self.model.to(self.device)
            self.model.eval()
            
            self.initialized = True
            return True
            
        except Exception as e:
            print(f"Error al inicializar el adaptador de Deep Learning: {str(e)}")
            self.initialized = False
            return False
    
    def process_images(self, image1_path: str, image2_path: str) -> ProcessingResult:
        """
        Procesa un par de imágenes utilizando el modelo de deep learning
        
        Args:
            image1_path: Ruta a la primera imagen
            image2_path: Ruta a la segunda imagen
            
        Returns:
            ProcessingResult: Resultado del procesamiento con métricas de similitud
        """
        if not self.initialized or self.model is None:
            return ProcessingResult(
                success=False,
                similarity_score=0.0,
                quality_score=0.0,
                processing_time=0.0,
                metadata={},
                error_message="El adaptador no ha sido inicializado correctamente"
            )
        
        try:
            start_time = time.time()
            
            # Cargar y preprocesar imágenes
            img1 = self._load_and_preprocess_image(image1_path)
            img2 = self._load_and_preprocess_image(image2_path)
            
            # Procesar según el tipo de modelo
            if self.model_type == 'siamese_network':
                # Desactivar cálculo de gradientes para inferencia
                with torch.no_grad():
                    similarity, feat1, feat2 = self.model(img1, img2)
                    similarity_score = similarity.item()
                
                # Calcular métricas adicionales
                quality_score = self._calculate_quality_score(feat1, feat2)
                
                # Preparar metadatos
                metadata = {
                    'model_type': self.model_type,
                    'threshold': self.threshold,
                    'match': similarity_score >= self.threshold,
                    'feature_distance': torch.nn.functional.pairwise_distance(feat1, feat2).item(),
                    'confidence': abs(similarity_score - 0.5) * 2  # Normalizar confianza
                }
                
            else:
                # Para otros tipos de modelos, implementar lógica específica
                # Por ahora, devolver valores por defecto
                similarity_score = 0.0
                quality_score = 0.0
                metadata = {'model_type': self.model_type, 'message': 'Tipo de modelo no soportado para comparación'}
            
            processing_time = time.time() - start_time
            
            return ProcessingResult(
                success=True,
                similarity_score=similarity_score,
                quality_score=quality_score,
                processing_time=processing_time,
                metadata=metadata
            )
            
        except Exception as e:
            return ProcessingResult(
                success=False,
                similarity_score=0.0,
                quality_score=0.0,
                processing_time=0.0,
                metadata={'model_type': self.model_type},
                error_message=f"Error en el procesamiento: {str(e)}"
            )
    
    def get_capabilities(self) -> Dict[str, bool]:
        """
        Retorna las capacidades del procesador
        
        Returns:
            Dict[str, bool]: Diccionario con las capacidades
        """
        return {
            'similarity_matching': self.model_type == 'siamese_network',
            'segmentation': self.model_type == 'unet_segmentation',
            'classification': self.model_type == 'resnet_classifier',
            'gpu_acceleration': torch.cuda.is_available(),
            'batch_processing': True,
            'roi_detection': self.model_type == 'unet_segmentation'
        }
    
    def cleanup(self):
        """Limpia recursos utilizados por el modelo"""
        self.model = None
        self.feature_extractor = None
        torch.cuda.empty_cache()
        self.initialized = False
    
    def _load_and_preprocess_image(self, image_path: str) -> torch.Tensor:
        """
        Carga y preprocesa una imagen para el modelo
        
        Args:
            image_path: Ruta a la imagen
            
        Returns:
            torch.Tensor: Tensor con la imagen preprocesada
        """
        # Cargar imagen
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"No se pudo cargar la imagen: {image_path}")
        
        # Convertir de BGR a RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Normalizar a [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Convertir a tensor y añadir dimensión de batch
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        
        # Aplicar transformaciones
        if self.transform:
            image_tensor = self.transform(image_tensor)
        
        # Mover a dispositivo
        return image_tensor.to(self.device)
    
    def _calculate_quality_score(self, feat1: torch.Tensor, feat2: torch.Tensor) -> float:
        """
        Calcula un score de calidad basado en las características extraídas
        
        Args:
            feat1: Características de la primera imagen
            feat2: Características de la segunda imagen
            
        Returns:
            float: Score de calidad entre 0 y 1
        """
        # Calcular distancia coseno (mayor valor = mayor similitud)
        cosine_sim = torch.nn.functional.cosine_similarity(feat1, feat2).item()
        
        # Calcular norma de los vectores (mayor valor = características más fuertes)
        norm1 = torch.norm(feat1).item()
        norm2 = torch.norm(feat2).item()
        
        # Combinar métricas para obtener score de calidad
        # Normalizar normas a [0,1] usando una función sigmoide
        norm_quality = (torch.sigmoid(torch.tensor((norm1 + norm2) / 2 - 10))).item()
        
        # Combinar similitud coseno con calidad de norma
        quality_score = (cosine_sim * 0.7) + (norm_quality * 0.3)
        
        return max(0.0, min(1.0, quality_score))  # Limitar entre 0 y 1