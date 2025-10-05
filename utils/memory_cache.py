"""
Sistema de caché inteligente para optimizar el rendimiento del sistema SIGeC-Balisticar
"""

import time
import threading
import weakref
import hashlib
import pickle
import gc
from typing import Any, Dict, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field
from collections import OrderedDict
import psutil
import numpy as np
import logging

logger = logging.getLogger(__name__)

@dataclass
class CacheStats:
    """Estadísticas del caché"""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    memory_usage_mb: float = 0.0
    total_items: int = 0
    
    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

@dataclass
class CacheEntry:
    """Entrada del caché"""
    value: Any
    timestamp: float
    access_count: int = 0
    size_bytes: int = 0
    ttl: Optional[float] = None
    
    def is_expired(self) -> bool:
        """Verifica si la entrada ha expirado"""
        if self.ttl is None:
            return False
        return time.time() - self.timestamp > self.ttl
    
    def touch(self):
        """Actualiza el timestamp y contador de acceso"""
        self.timestamp = time.time()
        self.access_count += 1

class MemoryCache:
    """
    Sistema de caché inteligente con gestión automática de memoria
    """
    
    def __init__(self, 
                 max_size: int = 1000,
                 max_memory_mb: float = 500.0,
                 default_ttl: Optional[float] = 3600.0,  # 1 hora
                 cleanup_interval: float = 300.0,  # 5 minutos
                 memory_threshold: float = 0.8):
        """
        Inicializa el caché
        
        Args:
            max_size: Número máximo de elementos
            max_memory_mb: Memoria máxima en MB
            default_ttl: TTL por defecto en segundos
            cleanup_interval: Intervalo de limpieza en segundos
            memory_threshold: Umbral de memoria para limpieza automática
        """
        self.max_size = max_size
        self.max_memory_mb = max_memory_mb
        self.default_ttl = default_ttl
        self.cleanup_interval = cleanup_interval
        self.memory_threshold = memory_threshold
        
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        self._stats = CacheStats()
        
        # Hilo de limpieza automática
        self._cleanup_thread = None
        self._stop_cleanup = threading.Event()
        self._start_cleanup_thread()
    
    def _start_cleanup_thread(self):
        """Inicia el hilo de limpieza automática"""
        if self._cleanup_thread is None or not self._cleanup_thread.is_alive():
            self._cleanup_thread = threading.Thread(
                target=self._cleanup_worker,
                daemon=True,
                name="CacheCleanup"
            )
            self._cleanup_thread.start()
    
    def _cleanup_worker(self):
        """Worker para limpieza automática del caché"""
        while not self._stop_cleanup.wait(self.cleanup_interval):
            try:
                self._cleanup_expired()
                self._check_memory_pressure()
            except Exception as e:
                logger.error(f"Error en limpieza automática del caché: {e}")
    
    def _generate_key(self, key: Union[str, tuple, Any]) -> str:
        """Genera una clave hash para el caché"""
        if isinstance(key, str):
            return key
        
        # Convertir a string serializable
        try:
            if isinstance(key, (tuple, list)):
                key_str = str(key)
            elif isinstance(key, np.ndarray):
                # Para arrays numpy, usar hash del contenido
                key_str = hashlib.md5(key.tobytes()).hexdigest()
            else:
                key_str = str(key)
            
            return hashlib.md5(key_str.encode()).hexdigest()
        except Exception:
            return str(hash(str(key)))
    
    def _estimate_size(self, obj: Any) -> int:
        """Estima el tamaño de un objeto en bytes"""
        try:
            if isinstance(obj, np.ndarray):
                return obj.nbytes
            elif isinstance(obj, (str, bytes)):
                return len(obj)
            elif isinstance(obj, (list, tuple, dict)):
                return len(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))
            else:
                return len(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))
        except Exception:
            return 1024  # Estimación por defecto
    
    def _get_memory_usage_mb(self) -> float:
        """Obtiene el uso actual de memoria del caché"""
        total_bytes = sum(entry.size_bytes for entry in self._cache.values())
        return total_bytes / (1024 * 1024)
    
    def _check_memory_pressure(self):
        """Verifica y maneja la presión de memoria"""
        current_memory = self._get_memory_usage_mb()
        
        if current_memory > self.max_memory_mb * self.memory_threshold:
            logger.info(f"Presión de memoria detectada: {current_memory:.1f}MB")
            self._evict_lru_items(target_memory=self.max_memory_mb * 0.6)
    
    def _evict_lru_items(self, target_memory: Optional[float] = None):
        """Elimina elementos usando estrategia LRU"""
        if target_memory is None:
            target_memory = self.max_memory_mb * 0.8
        
        with self._lock:
            # Ordenar por timestamp (LRU)
            items_by_age = sorted(
                self._cache.items(),
                key=lambda x: x[1].timestamp
            )
            
            current_memory = self._get_memory_usage_mb()
            
            for key, entry in items_by_age:
                if current_memory <= target_memory:
                    break
                
                current_memory -= entry.size_bytes / (1024 * 1024)
                del self._cache[key]
                self._stats.evictions += 1
    
    def _cleanup_expired(self):
        """Limpia elementos expirados"""
        with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items()
                if entry.is_expired()
            ]
            
            for key in expired_keys:
                del self._cache[key]
                self._stats.evictions += 1
    
    def put(self, key: Union[str, tuple, Any], value: Any, ttl: Optional[float] = None) -> bool:
        """
        Almacena un valor en el caché
        
        Args:
            key: Clave del elemento
            value: Valor a almacenar
            ttl: Tiempo de vida en segundos (None para usar default)
            
        Returns:
            True si se almacenó correctamente
        """
        cache_key = self._generate_key(key)
        size_bytes = self._estimate_size(value)
        
        # Verificar límites de memoria
        if size_bytes > self.max_memory_mb * 1024 * 1024:
            logger.warning(f"Objeto demasiado grande para caché: {size_bytes / 1024 / 1024:.1f}MB")
            return False
        
        with self._lock:
            # Verificar si necesitamos espacio
            current_memory = self._get_memory_usage_mb()
            required_memory = size_bytes / (1024 * 1024)
            
            if current_memory + required_memory > self.max_memory_mb:
                self._evict_lru_items(self.max_memory_mb - required_memory)
            
            # Verificar límite de elementos
            if len(self._cache) >= self.max_size:
                # Eliminar el más antiguo
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                self._stats.evictions += 1
            
            # Crear entrada
            entry = CacheEntry(
                value=value,
                timestamp=time.time(),
                size_bytes=size_bytes,
                ttl=ttl if ttl is not None else self.default_ttl
            )
            
            self._cache[cache_key] = entry
            self._cache.move_to_end(cache_key)  # Mover al final (más reciente)
            
            self._stats.total_items = len(self._cache)
            self._stats.memory_usage_mb = self._get_memory_usage_mb()
            
            return True
    
    def get(self, key: Union[str, tuple, Any]) -> Optional[Any]:
        """
        Obtiene un valor del caché
        
        Args:
            key: Clave del elemento
            
        Returns:
            Valor almacenado o None si no existe
        """
        cache_key = self._generate_key(key)
        
        with self._lock:
            entry = self._cache.get(cache_key)
            
            if entry is None:
                self._stats.misses += 1
                return None
            
            if entry.is_expired():
                del self._cache[cache_key]
                self._stats.misses += 1
                self._stats.evictions += 1
                return None
            
            # Actualizar estadísticas de acceso
            entry.touch()
            self._cache.move_to_end(cache_key)  # Mover al final (más reciente)
            self._stats.hits += 1
            
            return entry.value
    
    def get_or_compute(self, key: Union[str, tuple, Any], 
                      compute_func: Callable[[], Any],
                      ttl: Optional[float] = None) -> Any:
        """
        Obtiene un valor del caché o lo computa si no existe
        
        Args:
            key: Clave del elemento
            compute_func: Función para computar el valor
            ttl: Tiempo de vida en segundos
            
        Returns:
            Valor del caché o computado
        """
        value = self.get(key)
        if value is not None:
            return value
        
        # Computar valor
        computed_value = compute_func()
        self.put(key, computed_value, ttl)
        return computed_value
    
    def invalidate(self, key: Union[str, tuple, Any]) -> bool:
        """
        Invalida una entrada del caché
        
        Args:
            key: Clave del elemento
            
        Returns:
            True si se eliminó
        """
        cache_key = self._generate_key(key)
        
        with self._lock:
            if cache_key in self._cache:
                del self._cache[cache_key]
                self._stats.total_items = len(self._cache)
                self._stats.memory_usage_mb = self._get_memory_usage_mb()
                return True
            return False
    
    def clear(self):
        """Limpia todo el caché"""
        with self._lock:
            self._cache.clear()
            self._stats = CacheStats()
    
    def get_stats(self) -> CacheStats:
        """Obtiene estadísticas del caché"""
        with self._lock:
            self._stats.total_items = len(self._cache)
            self._stats.memory_usage_mb = self._get_memory_usage_mb()
            return self._stats
    
    def shutdown(self):
        """Cierra el caché y detiene hilos"""
        self._stop_cleanup.set()
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=5.0)
        self.clear()

# Instancia global del caché
_global_cache: Optional[MemoryCache] = None
_cache_lock = threading.Lock()

def get_global_cache() -> MemoryCache:
    """Obtiene la instancia global del caché"""
    global _global_cache
    
    if _global_cache is None:
        with _cache_lock:
            if _global_cache is None:
                _global_cache = MemoryCache()
    
    return _global_cache

def cache_features(func: Callable) -> Callable:
    """
    Decorador para cachear extracción de características
    """
    def wrapper(*args, **kwargs):
        # Generar clave basada en argumentos
        cache_key = (func.__name__, args, tuple(sorted(kwargs.items())))
        
        cache = get_global_cache()
        return cache.get_or_compute(
            cache_key,
            lambda: func(*args, **kwargs),
            ttl=1800  # 30 minutos para características
        )
    
    return wrapper

def cache_matches(func: Callable) -> Callable:
    """
    Decorador para cachear resultados de matching
    """
    def wrapper(*args, **kwargs):
        # Generar clave basada en argumentos
        cache_key = (func.__name__, args, tuple(sorted(kwargs.items())))
        
        cache = get_global_cache()
        return cache.get_or_compute(
            cache_key,
            lambda: func(*args, **kwargs),
            ttl=900  # 15 minutos para matches
        )
    
    return wrapper

# Funciones de utilidad
def clear_cache():
    """Limpia el caché global"""
    cache = get_global_cache()
    cache.clear()

def get_cache_stats() -> CacheStats:
    """Obtiene estadísticas del caché global"""
    cache = get_global_cache()
    return cache.get_stats()

def configure_cache(max_size: int = 1000, max_memory_mb: float = 500.0):
    """Configura el caché global"""
    global _global_cache
    
    with _cache_lock:
        if _global_cache is not None:
            _global_cache.shutdown()
        
        _global_cache = MemoryCache(
            max_size=max_size,
            max_memory_mb=max_memory_mb
        )