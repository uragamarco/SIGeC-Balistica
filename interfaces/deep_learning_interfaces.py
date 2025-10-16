"""
Interface abstracta para Modelos de Deep Learning Balístico
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

class IDeepLearningModel(ABC):
    """Interface para modelos de deep learning"""
    
    @abstractmethod
    def load_model(self, model_path: str, device: str = 'cpu') -> bool:
        """Carga un modelo"""
        pass
    
    @abstractmethod
    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """Extrae características usando DL"""
        pass
    
    @abstractmethod
    def calculate_similarity(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """Calcula similitud entre características"""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Retorna información del modelo"""
        pass
    
    @abstractmethod
    def is_model_loaded(self) -> bool:
        """Verifica si el modelo está cargado"""
        pass

class ISegmentationModel(ABC):
    """Interface para modelos de segmentación"""
    
    @abstractmethod
    def segment(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """Segmenta imagen y retorna confianza"""
        pass
    
    @abstractmethod
    def get_supported_classes(self) -> List[str]:
        """Retorna clases soportadas"""
        pass
