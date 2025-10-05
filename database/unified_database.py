"""
Base de Datos Unificada
Sistema Balístico Forense MVP

Interfaz unificada para acceso a la base de datos
"""

import logging
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

from .vector_db import VectorDatabase, BallisticCase, BallisticImage, FeatureVector
from config.unified_config import get_unified_config, UnifiedConfig

logger = logging.getLogger(__name__)

class UnifiedDatabase:
    """
    Interfaz unificada para acceso a la base de datos balística
    Proporciona una API simplificada para operaciones comunes
    """
    
    def __init__(self, config: Union[UnifiedConfig, str] = None):
        """
        Inicializa la base de datos unificada
        
        Args:
            config: Configuración unificada o ruta al archivo de configuración
        """
        self.logger = logger
        
        try:
            # Inicializar la base de datos vectorial
            self.vector_db = VectorDatabase(config)
            self.logger.info("Base de datos unificada inicializada correctamente")
        except Exception as e:
            self.logger.error(f"Error al inicializar la base de datos unificada: {e}")
            raise
    
    def search_similar_images(self, query_vector: Any, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Busca imágenes similares en la base de datos
        
        Args:
            query_vector: Vector de características de la imagen de consulta
            top_k: Número máximo de resultados a devolver
            
        Returns:
            Lista de resultados similares
        """
        try:
            return self.vector_db.search_similar(query_vector, top_k)
        except Exception as e:
            self.logger.error(f"Error en búsqueda de imágenes similares: {e}")
            return []
    
    def add_case(self, case_data: Dict[str, Any]) -> Optional[int]:
        """
        Añade un nuevo caso a la base de datos
        
        Args:
            case_data: Datos del caso
            
        Returns:
            ID del caso creado o None si hay error
        """
        try:
            case = BallisticCase(**case_data)
            return self.vector_db.add_case(case)
        except Exception as e:
            self.logger.error(f"Error al añadir caso: {e}")
            return None
    
    def add_image(self, image_data: Dict[str, Any]) -> Optional[int]:
        """
        Añade una nueva imagen a la base de datos
        
        Args:
            image_data: Datos de la imagen
            
        Returns:
            ID de la imagen creada o None si hay error
        """
        try:
            image = BallisticImage(**image_data)
            return self.vector_db.add_image(image)
        except Exception as e:
            self.logger.error(f"Error al añadir imagen: {e}")
            return None
    
    def get_cases(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Obtiene lista de casos
        
        Args:
            limit: Número máximo de casos a devolver
            
        Returns:
            Lista de casos
        """
        try:
            return self.vector_db.get_cases(limit)
        except Exception as e:
            self.logger.error(f"Error al obtener casos: {e}")
            return []
    
    def get_images_by_case(self, case_id: int) -> List[Dict[str, Any]]:
        """
        Obtiene imágenes de un caso específico
        
        Args:
            case_id: ID del caso
            
        Returns:
            Lista de imágenes del caso
        """
        try:
            return self.vector_db.get_images_by_case(case_id)
        except Exception as e:
            self.logger.error(f"Error al obtener imágenes del caso {case_id}: {e}")
            return []
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas de la base de datos
        
        Returns:
            Diccionario con estadísticas
        """
        try:
            return self.vector_db.get_statistics()
        except Exception as e:
            self.logger.error(f"Error al obtener estadísticas: {e}")
            return {
                "total_cases": 0,
                "total_images": 0,
                "total_vectors": 0,
                "database_size": 0
            }
    
    def get_database_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas detalladas de la base de datos
        
        Returns:
            Diccionario con estadísticas completas
        """
        try:
            return self.vector_db.get_database_stats()
        except Exception as e:
            self.logger.error(f"Error al obtener estadísticas de la base de datos: {e}")
            return {
                "active_cases": 0,
                "total_images": 0,
                "total_vectors": 0,
                "faiss_vectors": 0,
                "evidence_counts": {},
                "vector_dimension": 0,
                "cache_size": 0,
                "connection_pool_size": 0,
                "error": str(e)
            }
    
    def __enter__(self):
        """Soporte para context manager"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Soporte para context manager"""
        self.close()