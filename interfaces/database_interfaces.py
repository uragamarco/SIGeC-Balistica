"""
Interface abstracta para Gestión de Base de Datos Balística
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from datetime import datetime

class IDatabaseManager(ABC):
    """Interface para gestores de base de datos"""
    
    @abstractmethod
    def connect(self, connection_string: str) -> bool:
        """Conecta a la base de datos"""
        pass
    
    @abstractmethod
    def store_analysis(self, analysis_data: Dict[str, Any]) -> str:
        """Almacena un análisis y retorna ID"""
        pass
    
    @abstractmethod
    def retrieve_analysis(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        """Recupera un análisis por ID"""
        pass
    
    @abstractmethod
    def search_similar_cases(self, features: Dict[str, Any], threshold: float) -> List[Dict]:
        """Busca casos similares"""
        pass
    
    @abstractmethod
    def get_statistics(self) -> Dict[str, Any]:
        """Retorna estadísticas de la base de datos"""
        pass
    
    @abstractmethod
    def backup_database(self, backup_path: str) -> bool:
        """Crea backup de la base de datos"""
        pass
    
    @abstractmethod
    def close_connection(self):
        """Cierra la conexión"""
        pass
