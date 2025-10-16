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

# Import database interface
try:
    from interfaces.database_interfaces import IDatabaseManager
    INTERFACES_AVAILABLE = True
except ImportError:
    INTERFACES_AVAILABLE = False
    # Fallback empty interface
    class IDatabaseManager:
        pass

# Importar sistemas de validación y manejo de errores
try:
    from core.data_validator import get_data_validator, ValidationResult
    from core.error_handler import get_error_manager, with_error_handling, ErrorSeverity
except ImportError:
    # Fallback si los módulos no están disponibles
    def get_data_validator():
        return None
    
    def get_error_manager():
        return None
    
    def with_error_handling(component, operation=None):
        def decorator(func):
            return func
        return decorator
    
    class ErrorSeverity:
        HIGH = "high"
        MEDIUM = "medium"

logger = logging.getLogger(__name__)

class UnifiedDatabase(IDatabaseManager):
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
    
    @with_error_handling("unified_database", "add_case")
    def add_case(self, case_data: Dict[str, Any]) -> Optional[int]:
        """
        Añade un nuevo caso a la base de datos con validación robusta
        
        Args:
            case_data: Datos del caso
            
        Returns:
            ID del caso creado o None si hay error
        """
        try:
            # Validar datos de entrada
            validator = get_data_validator()
            if validator:
                validation_result = validator.validate_data(case_data, "ballistic_case")
                
                if not validation_result.is_valid:
                    error_messages = [error.error_message for error in validation_result.errors]
                    raise ValueError(f"Case validation failed: {'; '.join(error_messages)}")
                
                # Usar datos sanitizados
                case_data = validation_result.sanitized_data
            
            case = BallisticCase(**case_data)
            return self.vector_db.add_case(case)
        except Exception as e:
            self.logger.error(f"Error al añadir caso: {e}")
            return None
    
    @with_error_handling("unified_database", "add_image")
    def add_image(self, image_data: Dict[str, Any]) -> Optional[int]:
        """
        Añade una nueva imagen a la base de datos con validación robusta
        
        Args:
            image_data: Datos de la imagen
            
        Returns:
            ID de la imagen creada o None si hay error
        """
        try:
            # Validar datos de entrada
            validator = get_data_validator()
            if validator:
                validation_result = validator.validate_data(image_data, "ballistic_image")
                
                if not validation_result.is_valid:
                    error_messages = [error.error_message for error in validation_result.errors]
                    raise ValueError(f"Image validation failed: {'; '.join(error_messages)}")
                
                # Usar datos sanitizados
                image_data = validation_result.sanitized_data
            
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
    
    def close(self):
        """Cierra la conexión a la base de datos"""
        try:
            if hasattr(self.vector_db, 'close'):
                self.vector_db.close()
            self.logger.info("Conexión a la base de datos cerrada")
        except Exception as e:
            self.logger.error(f"Error al cerrar la base de datos: {e}")
    
    # Interface-compliant methods for IDatabaseManager
    def connect(self, connection_string: str) -> bool:
        """
        Interface-compliant method for database connection
        """
        try:
            # UnifiedDatabase initializes connection in __init__
            # This method can be used to reconnect or validate connection
            return True
        except Exception as e:
            self.logger.error(f"Error connecting to database: {e}")
            return False
    
    def store_analysis(self, analysis_data: Dict[str, Any]) -> str:
        """
        Interface-compliant method for storing analysis data
        """
        try:
            # Store as a case with analysis data
            case_id = self.add_case(analysis_data)
            return str(case_id) if case_id else ""
        except Exception as e:
            self.logger.error(f"Error storing analysis: {e}")
            return ""
    
    def retrieve_analysis(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        """
        Interface-compliant method for retrieving analysis data
        """
        try:
            # Get cases and find by ID
            cases = self.get_cases(limit=1000)  # Adjust limit as needed
            for case in cases:
                if str(case.get('id', '')) == analysis_id:
                    return case
            return None
        except Exception as e:
            self.logger.error(f"Error retrieving analysis {analysis_id}: {e}")
            return None
    
    def search_similar_cases(self, features: Dict[str, Any], threshold: float) -> List[Dict]:
        """
        Interface-compliant method for searching similar cases
        """
        try:
            # Use existing search_similar_images method
            query_vector = features.get('vector', features.get('descriptors'))
            if query_vector is not None:
                results = self.search_similar_images(query_vector, top_k=10)
                # Filter by threshold if similarity score is available
                filtered_results = []
                for result in results:
                    similarity = result.get('similarity', result.get('score', 1.0))
                    if similarity >= threshold:
                        filtered_results.append(result)
                return filtered_results
            return []
        except Exception as e:
            self.logger.error(f"Error searching similar cases: {e}")
            return []
    
    def backup_database(self, backup_path: str) -> bool:
        """
        Interface-compliant method for database backup
        """
        try:
            # Implement basic backup functionality
            if hasattr(self.vector_db, 'backup'):
                return self.vector_db.backup(backup_path)
            else:
                self.logger.warning("Backup functionality not implemented in vector database")
                return False
        except Exception as e:
            self.logger.error(f"Error creating database backup: {e}")
            return False
    
    def close_connection(self):
        """
        Interface-compliant method for closing connection
        """
        self.close()