"""
Interface abstracta para Preprocesamiento de Imágenes Balísticas
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional
import numpy as np

class IImagePreprocessor(ABC):
    """Interface para preprocesadores de imágenes"""
    
    @abstractmethod
    def preprocess(self, image: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
        """Preprocesa una imagen"""
        pass
    
    @abstractmethod
    def enhance_quality(self, image: np.ndarray) -> np.ndarray:
        """Mejora la calidad de la imagen"""
        pass
    
    @abstractmethod
    def detect_roi(self, image: np.ndarray) -> Tuple[int, int, int, int]:
        """Detecta región de interés"""
        pass
    
    @abstractmethod
    def get_preprocessing_steps(self) -> List[str]:
        """Retorna los pasos de preprocesamiento aplicados"""
        pass

class ISegmentationProcessor(ABC):
    """Interface para procesadores de segmentación"""
    
    @abstractmethod
    def segment_image(self, image: np.ndarray) -> np.ndarray:
        """Segmenta una imagen"""
        pass
    
    @abstractmethod
    def get_segmentation_confidence(self) -> float:
        """Retorna confianza de la segmentación"""
        pass
