"""
Interface abstracta para el Pipeline de Procesamiento Balístico
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np

@dataclass
class ProcessingResult:
    """Resultado del procesamiento"""
    success: bool
    similarity_score: float
    quality_score: float
    processing_time: float
    metadata: Dict[str, Any]
    error_message: Optional[str] = None

class IPipelineProcessor(ABC):
    """Interface para procesadores de pipeline"""
    
    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> bool:
        """Inicializa el procesador"""
        pass
    
    @abstractmethod
    def process_images(self, image1_path: str, image2_path: str) -> ProcessingResult:
        """Procesa un par de imágenes"""
        pass
    
    @abstractmethod
    def get_capabilities(self) -> Dict[str, bool]:
        """Retorna las capacidades del procesador"""
        pass
    
    @abstractmethod
    def cleanup(self):
        """Limpia recursos"""
        pass

class IQualityAssessor(ABC):
    """Interface para evaluadores de calidad"""
    
    @abstractmethod
    def assess_quality(self, image: np.ndarray) -> Dict[str, float]:
        """Evalúa la calidad de una imagen"""
        pass
    
    @abstractmethod
    def meets_minimum_quality(self, quality_metrics: Dict[str, float]) -> bool:
        """Verifica si cumple calidad mínima"""
        pass

class IQualityMetricsProvider(ABC):
    """Interface para proveedores de métricas de calidad (NIST)."""

    @abstractmethod
    def analyze_image_quality(self, image: np.ndarray, image_id: str) -> Any:
        """Calcula métricas NIST y retorna un reporte estructurado."""
        pass

    @abstractmethod
    def get_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Retorna umbrales por métrica organizados por nivel de calidad."""
        pass

    @abstractmethod
    def get_quality_score(self, report: Any) -> float:
        """Extrae el puntaje de calidad global del reporte."""
        pass

    @abstractmethod
    def meets_minimum_quality(self, report: Any, min_score: float) -> bool:
        """Verifica si el reporte cumple un puntaje mínimo establecido."""
        pass
