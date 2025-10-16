"""
Interface abstracta para Algoritmos de Matching Balístico
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional
import numpy as np

class IFeatureMatcher(ABC):
    """Interface para algoritmos de matching de características"""
    
    @abstractmethod
    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """Extrae características de una imagen"""
        pass
    
    @abstractmethod
    def match_features(self, features1: np.ndarray, features2: np.ndarray) -> Dict[str, float]:
        """Realiza matching entre características"""
        pass
    
    @abstractmethod
    def calculate_similarity(self, matches: Dict[str, float]) -> float:
        """Calcula score de similitud"""
        pass
    
    @abstractmethod
    def get_algorithm_info(self) -> Dict[str, str]:
        """Retorna información del algoritmo"""
        pass

class IDescriptorExtractor(ABC):
    """Interface para extractores de descriptores"""
    
    @abstractmethod
    def extract_descriptors(self, image: np.ndarray, keypoints: List) -> np.ndarray:
        """Extrae descriptores de puntos clave"""
        pass
    
    @abstractmethod
    def get_descriptor_size(self) -> int:
        """Retorna el tamaño del descriptor"""
        pass
