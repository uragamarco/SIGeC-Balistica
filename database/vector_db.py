"""
Base de Datos Vectorial
Sistema Balístico Forense MVP

Gestión de base de datos SQLite + FAISS para almacenamiento y búsqueda vectorial
"""

import sqlite3
import numpy as np
import faiss
import json
import pickle
import threading
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from functools import lru_cache
from contextlib import contextmanager

from utils.logger import LoggerMixin
from config.unified_config import get_unified_config, UnifiedConfig

# Importar sistema de monitoreo de rendimiento
try:
    from core.performance_monitor import monitor_performance, monitor_database_operation, OperationType
except ImportError:
    # Fallback si el módulo no está disponible
    def monitor_performance(operation_type):
        def decorator(func):
            return func
        return decorator
    
    def monitor_database_operation(func):
        return func
    
    class OperationType:
        DATABASE_OPERATION = "database_operation"

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

@dataclass
class BallisticCase:
    """Estructura de datos para un caso balístico"""
    id: Optional[int] = None
    case_number: str = ""
    investigator: str = ""
    date_created: str = ""
    weapon_type: str = ""
    weapon_model: str = ""
    caliber: str = ""
    description: str = ""
    status: str = "active"  # active, archived, deleted
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

@dataclass
class BallisticImage:
    """Estructura de datos para una imagen balística"""
    id: Optional[int] = None
    case_id: int = 0
    filename: str = ""
    file_path: str = ""
    evidence_type: str = ""  # "vaina" o "proyectil"
    image_hash: str = ""
    width: int = 0
    height: int = 0
    file_size: int = 0
    date_added: str = ""
    processed: bool = False
    feature_vector_id: Optional[int] = None
    created_at: Optional[str] = None

@dataclass
class FeatureVector:
    """Estructura de datos para vectores de características"""
    id: Optional[int] = None
    image_id: int = 0
    algorithm: str = ""  # "ORB", "SIFT", etc.
    vector_data: bytes = b""  # Datos serializados del vector
    vector_size: int = 0
    extraction_params: str = ""  # JSON con parámetros usados
    date_extracted: str = ""

class VectorDatabase(LoggerMixin):
    """Clase principal para manejo de base de datos vectorial optimizada"""
    
    def __init__(self, config: Union[UnifiedConfig, str] = None):
        """
        Inicializar la base de datos vectorial
        
        Args:
            config: Configuración unificada o path al archivo de configuración
        """
        super().__init__()
        
        if isinstance(config, str):
            # Si es string, cargar configuración desde archivo
            from config.unified_config import get_unified_config as ConfigClass
            self.config = ConfigClass(config)
        elif config is None:
            # Si no se proporciona configuración, usar la por defecto
            self.config = get_unified_config()
        else:
            # Si ya es una instancia de UnifiedConfig
            self.config = config
            
        # Configurar rutas de base de datos
        if hasattr(self.config, 'get_absolute_path'):
            # Es UnifiedConfig
            self.db_path = self.config.get_absolute_path(self.config.database.sqlite_path)
            self.faiss_path = self.config.get_absolute_path(self.config.database.faiss_index_path)
        else:
            # Es DatabaseConfig directamente - usar rutas como están
            from pathlib import Path
            self.db_path = Path(self.config.sqlite_path) if hasattr(self.config, 'sqlite_path') else Path(self.config.database.sqlite_path)
            self.faiss_path = Path(self.config.faiss_index_path) if hasattr(self.config, 'faiss_index_path') else Path(self.config.database.faiss_index_path)
        
        # Pool de conexiones y cache
        self._connection_pool = []
        self._pool_lock = threading.Lock()
        self._query_cache = {}
        self._cache_lock = threading.Lock()
        self._cache_ttl = 300  # 5 minutos
        
        # Inicializar base de datos SQLite
        self._init_sqlite()
        
        # Inicializar índice FAISS
        self.faiss_index = None
        self.vector_dimension = None
        self._init_faiss()
        
        self.logger.info(f"Base de datos vectorial optimizada inicializada: {self.db_path}")
    
    @contextmanager
    def _get_connection(self):
        """Context manager para obtener conexión del pool"""
        with self._pool_lock:
            if self._connection_pool:
                conn = self._connection_pool.pop()
            else:
                conn = sqlite3.connect(self.db_path, timeout=30.0)
                conn.row_factory = sqlite3.Row
                # Optimizaciones SQLite
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")
                conn.execute("PRAGMA cache_size=10000")
                conn.execute("PRAGMA temp_store=MEMORY")
        
        try:
            yield conn
        finally:
            with self._pool_lock:
                if len(self._connection_pool) < 5:  # Máximo 5 conexiones en pool
                    self._connection_pool.append(conn)
                else:
                    conn.close()

    def _cache_get(self, key: str) -> Optional[Any]:
        """Obtener valor del cache"""
        with self._cache_lock:
            if key in self._query_cache:
                value, timestamp = self._query_cache[key]
                if time.time() - timestamp < self._cache_ttl:
                    return value
                else:
                    del self._query_cache[key]
        return None

    def _cache_set(self, key: str, value: Any):
        """Establecer valor en cache"""
        with self._cache_lock:
            self._query_cache[key] = (value, time.time())
            # Limpiar cache si es muy grande
            if len(self._query_cache) > 1000:
                # Eliminar entradas más antiguas
                current_time = time.time()
                expired_keys = [k for k, (_, t) in self._query_cache.items() 
                              if current_time - t > self._cache_ttl]
                for k in expired_keys:
                    del self._query_cache[k]

    def _init_sqlite(self):
        """Inicializa la base de datos SQLite con optimizaciones"""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Tabla de casos
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ballistic_cases (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    case_number TEXT UNIQUE NOT NULL,
                    investigator TEXT NOT NULL,
                    date_created TEXT NOT NULL,
                    weapon_type TEXT,
                    weapon_model TEXT,
                    caliber TEXT,
                    description TEXT,
                    status TEXT DEFAULT 'active',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Tabla de imágenes
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ballistic_images (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    case_id INTEGER NOT NULL,
                    filename TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    evidence_type TEXT NOT NULL,
                    image_hash TEXT UNIQUE,
                    width INTEGER,
                    height INTEGER,
                    file_size INTEGER,
                    date_added TEXT NOT NULL,
                    processed BOOLEAN DEFAULT FALSE,
                    feature_vector_id INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (case_id) REFERENCES ballistic_cases (id)
                )
            """)
            
            # Tabla de vectores de características
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS feature_vectors (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    image_id INTEGER NOT NULL,
                    algorithm TEXT NOT NULL,
                    vector_data BLOB NOT NULL,
                    vector_size INTEGER NOT NULL,
                    extraction_params TEXT,
                    date_extracted TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (image_id) REFERENCES ballistic_images (id)
                )
            """)
            
            # Índices optimizados
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_case_number ON ballistic_cases (case_number)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_case_status ON ballistic_cases (status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_image_hash ON ballistic_images (image_hash)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_evidence_type ON ballistic_images (evidence_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_case_id ON ballistic_images (case_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_algorithm ON feature_vectors (algorithm)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_image_id ON feature_vectors (image_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_date_created ON ballistic_cases (date_created)")
            
            conn.commit()
            self.logger.info("Esquema de base de datos SQLite optimizado inicializado")
    
    def _init_faiss(self):
        """Inicializa el índice FAISS"""
        faiss_file = f"{self.faiss_path}.index"
        
        if Path(faiss_file).exists():
            try:
                self.faiss_index = faiss.read_index(faiss_file)
                self.vector_dimension = self.faiss_index.d
                self.logger.info(f"Índice FAISS cargado: {faiss_file} (dim={self.vector_dimension})")
            except Exception as e:
                self.logger.error(f"Error cargando índice FAISS: {e}")
                self.faiss_index = None
        
        if self.faiss_index is None:
            # Crear nuevo índice (se inicializará cuando se agregue el primer vector)
            self.vector_dimension = None
            self.logger.info("Índice FAISS será creado al agregar el primer vector")
    


    
    def add_feature_vector(self, vector: FeatureVector, numpy_vector: np.ndarray) -> int:
        """Agrega un vector de características a la base de datos y al índice FAISS"""
        if not vector.date_extracted:
            vector.date_extracted = datetime.now().isoformat()
        
        # Serializar el vector numpy
        vector_bytes = pickle.dumps(numpy_vector)
        vector.vector_data = vector_bytes
        vector.vector_size = len(numpy_vector)
        
        # Agregar a SQLite
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO feature_vectors 
                (image_id, algorithm, vector_data, vector_size, extraction_params, date_extracted)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                vector.image_id, vector.algorithm, vector.vector_data,
                vector.vector_size, vector.extraction_params, vector.date_extracted
            ))
            vector_id = cursor.lastrowid
            conn.commit()
        
        # Agregar al índice FAISS
        self._add_to_faiss_index(numpy_vector, vector_id)
        
        self.logger.info(f"Vector de características agregado (ID: {vector_id})")
        return vector_id
    
    def _add_to_faiss_index(self, vector: np.ndarray, vector_id: int):
        """Agrega un vector al índice FAISS"""
        # Asegurar que el vector sea float32 y 2D
        if vector.dtype != np.float32:
            vector = vector.astype(np.float32)
        
        if len(vector.shape) == 1:
            vector = vector.reshape(1, -1)
        
        # Inicializar índice si es necesario
        if self.faiss_index is None:
            self.vector_dimension = vector.shape[1]
            self.faiss_index = faiss.IndexFlatL2(self.vector_dimension)
            self.logger.info(f"Índice FAISS creado con dimensión {self.vector_dimension}")
        
        # Verificar dimensión
        if vector.shape[1] != self.vector_dimension:
            raise ValueError(f"Dimensión del vector ({vector.shape[1]}) no coincide con el índice ({self.vector_dimension})")
        
        # Agregar al índice
        self.faiss_index.add(vector)
        
        # Guardar índice actualizado
        self._save_faiss_index()
    
    def _save_faiss_index(self):
        """Guarda el índice FAISS a disco"""
        if self.faiss_index is not None:
            Path(self.faiss_path).parent.mkdir(parents=True, exist_ok=True)
            faiss_file = f"{self.faiss_path}.index"
            faiss.write_index(self.faiss_index, faiss_file)
    
    def search_similar_vectors(self, query_vector: np.ndarray, k: int = 5, 
                             distance_threshold: float = None) -> List[Tuple[int, float]]:
        """Busca vectores similares usando FAISS con optimizaciones"""
        if self.faiss_index is None or self.faiss_index.ntotal == 0:
            return []
        
        # Cache key para búsquedas repetidas
        cache_key = f"search_{hash(query_vector.tobytes())}_{k}_{distance_threshold}"
        cached_result = self._cache_get(cache_key)
        if cached_result is not None:
            return cached_result
        
        # Preparar vector de consulta
        if query_vector.dtype != np.float32:
            query_vector = query_vector.astype(np.float32)
        
        if len(query_vector.shape) == 1:
            query_vector = query_vector.reshape(1, -1)
        
        # Verificar dimensión
        if query_vector.shape[1] != self.vector_dimension:
            raise ValueError(f"Dimensión del vector de consulta ({query_vector.shape[1]}) no coincide con el índice ({self.vector_dimension})")
        
        # Buscar con más candidatos para mejor precisión
        search_k = min(k * 2, self.faiss_index.ntotal)
        distances, indices = self.faiss_index.search(query_vector, search_k)
        
        # Filtrar resultados
        results = []
        for i, (idx, dist) in enumerate(zip(indices[0], distances[0])):
            if idx != -1:  # -1 indica que no se encontró resultado válido
                if distance_threshold is None or dist <= distance_threshold:
                    results.append((int(idx), float(dist)))
                    if len(results) >= k:
                        break
        
        # Cache del resultado
        self._cache_set(cache_key, results)
        return results

    def batch_search_similar_vectors(self, query_vectors: np.ndarray, k: int = 5) -> List[List[Tuple[int, float]]]:
        """Búsqueda batch optimizada de vectores similares"""
        if self.faiss_index is None or self.faiss_index.ntotal == 0:
            return [[] for _ in range(len(query_vectors))]
        
        # Preparar vectores de consulta
        if query_vectors.dtype != np.float32:
            query_vectors = query_vectors.astype(np.float32)
        
        if len(query_vectors.shape) == 1:
            query_vectors = query_vectors.reshape(1, -1)
        
        # Verificar dimensión
        if query_vectors.shape[1] != self.vector_dimension:
            raise ValueError(f"Dimensión de vectores de consulta no coincide con el índice")
        
        # Búsqueda batch
        distances, indices = self.faiss_index.search(query_vectors, min(k, self.faiss_index.ntotal))
        
        # Procesar resultados
        batch_results = []
        for i in range(len(query_vectors)):
            results = []
            for idx, dist in zip(indices[i], distances[i]):
                if idx != -1:
                    results.append((int(idx), float(dist)))
            batch_results.append(results)
        
        return batch_results
    
    @monitor_performance(OperationType.DATABASE_OPERATION)
    @lru_cache(maxsize=128)
    def get_case_by_id(self, case_id: int) -> Optional[BallisticCase]:
        """Obtiene un caso por ID con cache"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM ballistic_cases WHERE id = ?", (case_id,))
            row = cursor.fetchone()
            
            if row:
                return BallisticCase(**dict(row))
        return None

    @lru_cache(maxsize=128)
    @with_error_handling("database", "get_case_by_number")
    def get_case_by_number(self, case_number: str) -> Optional[BallisticCase]:
        """Obtiene un caso por número de caso con cache y validación"""
        # Validar entrada
        if not case_number or not case_number.strip():
            raise ValueError("case_number no puede estar vacío")
        
        # Sanitizar entrada
        validator = get_data_validator()
        if validator:
            sanitized_number = validator.sanitize_string(case_number.strip())
        else:
            sanitized_number = case_number.strip()
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM ballistic_cases WHERE case_number = ? AND status = 'active'", (sanitized_number,))
            row = cursor.fetchone()
            
            if row:
                return BallisticCase(**dict(row))
        return None

    def get_cases_batch(self, case_ids: List[int]) -> List[BallisticCase]:
        """Obtiene múltiples casos en una sola consulta"""
        if not case_ids:
            return []
        
        placeholders = ','.join('?' * len(case_ids))
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"SELECT * FROM ballistic_cases WHERE id IN ({placeholders})", case_ids)
            rows = cursor.fetchall()
            
            return [BallisticCase(**dict(row)) for row in rows]

    def get_images_batch(self, image_ids: List[int]) -> List[BallisticImage]:
        """Obtiene múltiples imágenes en una sola consulta"""
        if not image_ids:
            return []
        
        placeholders = ','.join('?' * len(image_ids))
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"SELECT * FROM ballistic_images WHERE id IN ({placeholders})", image_ids)
            rows = cursor.fetchall()
            
            return [BallisticImage(**dict(row)) for row in rows]
    
    def get_image_by_id(self, image_id: int) -> Optional[BallisticImage]:
        """Obtiene una imagen por ID"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM ballistic_images WHERE id = ?", (image_id,))
            row = cursor.fetchone()
            
            if row:
                return BallisticImage(**dict(row))
        return None
    
    def get_image_by_hash(self, image_hash: str) -> Optional[BallisticImage]:
        """Obtiene una imagen por hash"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM ballistic_images WHERE image_hash = ?", (image_hash,))
            row = cursor.fetchone()
            
            if row:
                return BallisticImage(**dict(row))
        return None
    
    def get_vector_by_id(self, vector_id: int) -> Optional[Tuple[FeatureVector, np.ndarray]]:
        """Obtiene un vector por ID"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM feature_vectors WHERE id = ?", (vector_id,))
            row = cursor.fetchone()
            
            if row:
                vector_info = FeatureVector(**dict(row))
                numpy_vector = pickle.loads(vector_info.vector_data)
                return vector_info, numpy_vector
        return None
    
    def get_cases(self, status: str = "active") -> List[BallisticCase]:
        """Obtiene lista de casos"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM ballistic_cases WHERE status = ? ORDER BY date_created DESC", (status,))
            rows = cursor.fetchall()
            
            return [BallisticCase(**dict(row)) for row in rows]
    
    def get_images_by_case(self, case_id: int) -> List[BallisticImage]:
        """Obtiene imágenes de un caso"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM ballistic_images WHERE case_id = ? ORDER BY date_added", (case_id,))
            rows = cursor.fetchall()
            
            return [BallisticImage(**dict(row)) for row in rows]
    
    @monitor_database_operation
    @with_error_handling("database", "add_case")
    def add_case(self, case: BallisticCase) -> int:
        """Agrega un nuevo caso balístico con validación robusta"""
        try:
            # Validar datos de entrada usando el sistema de validación
            validator = get_data_validator()
            if validator:
                case_data = {
                    "case_number": case.case_number,
                    "investigator": case.investigator,
                    "weapon_type": case.weapon_type,
                    "weapon_model": case.weapon_model,
                    "caliber": case.caliber,
                    "description": case.description,
                    "date_created": case.date_created or datetime.now().isoformat()
                }
                
                validation_result = validator.validate_data(case_data, "ballistic_case")
                
                if not validation_result.is_valid:
                    error_messages = [error.error_message for error in validation_result.errors]
                    raise ValueError(f"Validation failed: {'; '.join(error_messages)}")
                
                # Usar datos sanitizados
                case.case_number = validation_result.sanitized_data.get("case_number", case.case_number)
                case.investigator = validation_result.sanitized_data.get("investigator", case.investigator)
                case.weapon_type = validation_result.sanitized_data.get("weapon_type", case.weapon_type)
                case.weapon_model = validation_result.sanitized_data.get("weapon_model", case.weapon_model)
                case.caliber = validation_result.sanitized_data.get("caliber", case.caliber)
                case.description = validation_result.sanitized_data.get("description", case.description)
                case.date_created = validation_result.sanitized_data.get("date_created", case.date_created)
            
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Establecer timestamps
                now = datetime.now().isoformat()
                if not case.created_at:
                    case.created_at = now
                case.updated_at = now
                
                # Verificar que el número de caso no exista
                cursor.execute("SELECT id FROM ballistic_cases WHERE case_number = ?", (case.case_number,))
                if cursor.fetchone():
                    raise ValueError(f"Case number '{case.case_number}' already exists")
                
                # Validar campos requeridos (validación adicional)
                if not case.case_number or not case.case_number.strip():
                    raise ValueError("case_number es requerido")
                if not case.investigator or not case.investigator.strip():
                    raise ValueError("investigator es requerido")
                if not case.date_created:
                    case.date_created = now
                    case.date_created = now
                
                # Verificar si el case_number ya existe
                cursor.execute("SELECT id FROM ballistic_cases WHERE case_number = ?", (case.case_number,))
                if cursor.fetchone():
                    # Generar un case_number único
                    base_number = case.case_number
                    counter = 1
                    while True:
                        new_case_number = f"{base_number}_{counter}"
                        cursor.execute("SELECT id FROM ballistic_cases WHERE case_number = ?", (new_case_number,))
                        if not cursor.fetchone():
                            case.case_number = new_case_number
                            break
                        counter += 1
                
                cursor.execute("""
                    INSERT INTO ballistic_cases 
                    (case_number, investigator, date_created, weapon_type, weapon_model, 
                     caliber, description, status, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    case.case_number, case.investigator, case.date_created,
                    case.weapon_type, case.weapon_model, case.caliber,
                    case.description, case.status, case.created_at, case.updated_at
                ))
                
                case_id = cursor.lastrowid
                conn.commit()
                
            self.logger.info(f"Caso agregado (ID: {case_id}, Número: {case.case_number})")
            return case_id
            
        except sqlite3.IntegrityError as e:
            self.logger.error(f"Error de integridad al agregar caso: {e}")
            raise ValueError(f"Error de integridad en base de datos: {e}")
        except Exception as e:
            self.logger.error(f"Error al agregar caso: {e}")
            raise
    
    @with_error_handling("database", "add_image")
    def add_image(self, image: BallisticImage) -> int:
        """Agrega una nueva imagen balística con validación robusta"""
        try:
            # Validar datos de entrada usando el sistema de validación
            validator = get_data_validator()
            if validator:
                image_data = {
                    "filename": image.filename,
                    "file_path": image.file_path,
                    "evidence_type": image.evidence_type,
                    "width": image.width,
                    "height": image.height,
                    "file_size": image.file_size
                }
                
                validation_result = validator.validate_data(image_data, "ballistic_image")
                
                if not validation_result.is_valid:
                    error_messages = [error.error_message for error in validation_result.errors]
                    raise ValueError(f"Image validation failed: {'; '.join(error_messages)}")
                
                # Usar datos sanitizados
                image.filename = validation_result.sanitized_data.get("filename", image.filename)
                image.file_path = validation_result.sanitized_data.get("file_path", image.file_path)
                image.evidence_type = validation_result.sanitized_data.get("evidence_type", image.evidence_type)
                image.width = validation_result.sanitized_data.get("width", image.width)
                image.height = validation_result.sanitized_data.get("height", image.height)
                image.file_size = validation_result.sanitized_data.get("file_size", image.file_size)
            
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Establecer timestamp
                now = datetime.now().isoformat()
                if not image.created_at:
                    image.created_at = now
                if not image.date_added:
                    image.date_added = now
                
                # Validar campos requeridos (validación adicional)
                if not image.case_id:
                    raise ValueError("case_id es requerido")
                if not image.filename or not image.filename.strip():
                    raise ValueError("filename es requerido")
                if not image.file_path or not image.file_path.strip():
                    raise ValueError("file_path es requerido")
                if not image.evidence_type or not image.evidence_type.strip():
                    raise ValueError("evidence_type es requerido")
                
                # Verificar que el caso existe
                cursor.execute("SELECT id FROM ballistic_cases WHERE id = ?", (image.case_id,))
                if not cursor.fetchone():
                    raise ValueError(f"El caso con ID {image.case_id} no existe")
                
                # Manejar image_hash duplicado
                if image.image_hash:
                    cursor.execute("SELECT id FROM ballistic_images WHERE image_hash = ?", (image.image_hash,))
                    if cursor.fetchone():
                        # Generar un hash único agregando timestamp
                        import hashlib
                        unique_suffix = hashlib.md5(now.encode()).hexdigest()[:8]
                        image.image_hash = f"{image.image_hash}_{unique_suffix}"
                
                cursor.execute("""
                    INSERT INTO ballistic_images 
                    (case_id, filename, file_path, evidence_type, image_hash, 
                     width, height, file_size, date_added, processed, feature_vector_id, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    image.case_id, image.filename, image.file_path, image.evidence_type,
                    image.image_hash, image.width, image.height, image.file_size,
                    image.date_added, image.processed, image.feature_vector_id, image.created_at
                ))
                
                image_id = cursor.lastrowid
                conn.commit()
                
            self.logger.info(f"Imagen agregada (ID: {image_id}, Archivo: {image.filename})")
            return image_id
            
        except sqlite3.IntegrityError as e:
            self.logger.error(f"Error de integridad al agregar imagen: {e}")
            raise ValueError(f"Error de integridad en base de datos: {e}")
        except Exception as e:
            self.logger.error(f"Error al agregar imagen: {e}")
            raise

    @monitor_performance(OperationType.DATABASE_OPERATION)
    def add_cases_batch(self, cases: List[BallisticCase]) -> List[int]:
        """Agrega múltiples casos en una transacción"""
        if not cases:
            return []
        
        case_ids = []
        with self._get_connection() as conn:
            cursor = conn.cursor()
            now = datetime.now().isoformat()
            
            for case in cases:
                if not case.created_at:
                    case.created_at = now
                case.updated_at = now
                
                cursor.execute("""
                    INSERT INTO ballistic_cases 
                    (case_number, investigator, date_created, weapon_type, weapon_model, 
                     caliber, description, status, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    case.case_number, case.investigator, case.date_created,
                    case.weapon_type, case.weapon_model, case.caliber,
                    case.description, case.status, case.created_at, case.updated_at
                ))
                
                case_ids.append(cursor.lastrowid)
            
            conn.commit()
        
        self.logger.info(f"Agregados {len(case_ids)} casos en batch")
        return case_ids

    def add_images_batch(self, images: List[BallisticImage]) -> List[int]:
        """Agrega múltiples imágenes en una transacción"""
        if not images:
            return []
        
        image_ids = []
        with self._get_connection() as conn:
            cursor = conn.cursor()
            now = datetime.now().isoformat()
            
            for image in images:
                if not image.created_at:
                    image.created_at = now
                if not image.date_added:
                    image.date_added = now
                
                cursor.execute("""
                    INSERT INTO ballistic_images 
                    (case_id, filename, file_path, evidence_type, image_hash, 
                     width, height, file_size, date_added, processed, feature_vector_id, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    image.case_id, image.filename, image.file_path, image.evidence_type,
                    image.image_hash, image.width, image.height, image.file_size,
                    image.date_added, image.processed, image.feature_vector_id, image.created_at
                ))
                
                image_ids.append(cursor.lastrowid)
            
            conn.commit()
        
        self.logger.info(f"Agregadas {len(image_ids)} imágenes en batch")
        return image_ids

    def get_database_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas de la base de datos con cache"""
        cache_key = "database_stats"
        cached_stats = self._cache_get(cache_key)
        if cached_stats is not None:
            return cached_stats
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Contar casos
            cursor.execute("SELECT COUNT(*) FROM ballistic_cases WHERE status = 'active'")
            active_cases = cursor.fetchone()[0]
            
            # Contar imágenes
            cursor.execute("SELECT COUNT(*) FROM ballistic_images")
            total_images = cursor.fetchone()[0]
            
            # Contar vectores
            cursor.execute("SELECT COUNT(*) FROM feature_vectors")
            total_vectors = cursor.fetchone()[0]
            
            # Contar por tipo de evidencia
            cursor.execute("SELECT evidence_type, COUNT(*) FROM ballistic_images GROUP BY evidence_type")
            evidence_counts = dict(cursor.fetchall())
        
        faiss_vectors = self.faiss_index.ntotal if self.faiss_index else 0
        
        stats = {
            "active_cases": active_cases,
            "total_images": total_images,
            "total_vectors": total_vectors,
            "faiss_vectors": faiss_vectors,
            "evidence_counts": evidence_counts,
            "vector_dimension": self.vector_dimension,
            "cache_size": len(self._query_cache),
            "connection_pool_size": len(self._connection_pool)
        }
        
        # Cache por 1 minuto
        self._cache_set(cache_key, stats)
        return stats

    def optimize_database(self):
        """Optimiza la base de datos"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # VACUUM para compactar
            cursor.execute("VACUUM")
            
            # ANALYZE para actualizar estadísticas
            cursor.execute("ANALYZE")
            
            conn.commit()
        
        # Limpiar cache
        with self._cache_lock:
            self._query_cache.clear()
        
        self.logger.info("Base de datos optimizada")

    def close(self):
        """Cierra todas las conexiones y limpia recursos"""
        with self._pool_lock:
            for conn in self._connection_pool:
                conn.close()
            self._connection_pool.clear()
        
        with self._cache_lock:
            self._query_cache.clear()
        
        if self.faiss_index is not None:
            self._save_faiss_index()
        
        self.logger.info("Base de datos cerrada correctamente")