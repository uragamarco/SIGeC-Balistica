"""
Módulo de Base de Datos - Sistema Balístico Forense SIGeC-Balisticar
===================================================================

Este módulo proporciona gestión unificada de bases de datos para el sistema:

Componentes principales:
- UnifiedDatabase: Gestión unificada de base de datos SQLite
- VectorDatabase: Índice vectorial con FAISS para búsqueda por similitud
- DatabaseManager: Coordinación entre bases de datos relacionales y vectoriales

Interfaces públicas:
- Gestión de base de datos SQLite
- Índice vectorial con FAISS
- Almacenamiento de descriptores balísticos
- Búsqueda por similitud
- Operaciones CRUD optimizadas
"""

from .unified_database import UnifiedDatabase
from .vector_db import VectorDatabase

# Interfaces públicas principales
__all__ = [
    # Base de datos principal
    'UnifiedDatabase',
    
    # Base de datos vectorial
    'VectorDatabase'
]

__version__ = "1.0.0"
__author__ = "SIGeC-Balisticar Development Team"