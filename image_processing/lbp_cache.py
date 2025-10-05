"""
Sistema de Caché para Patrones LBP Frecuentes
Sistema Balístico Forense MVP

Implementa un sistema de caché inteligente para optimizar el cálculo de patrones LBP
frecuentemente utilizados en el análisis de características balísticas.

Características:
- Caché basado en hash de imagen y parámetros LBP
- Seguimiento de frecuencia de uso
- Políticas de evicción LRU y basada en frecuencia
- Gestión automática de memoria
- Estadísticas de rendimiento
"""

import hashlib
import time
import threading
from collections import OrderedDict, defaultdict
from typing import Dict, Tuple, Optional, Any, List
from dataclasses import dataclass, field
import numpy as np
import logging
from skimage.feature import local_binary_pattern

logger = logging.getLogger(__name__)

@dataclass
class LBPCacheEntry:
    """Entrada del caché LBP"""
    lbp_result: np.ndarray
    histogram: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    access_count: int = 0
    last_access: float = field(default_factory=time.time)
    creation_time: float = field(default_factory=time.time)
    memory_size: int = 0

@dataclass
class LBPCacheStats:
    """Estadísticas del caché LBP"""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_memory_used: int = 0
    max_memory_used: int = 0
    average_computation_time: float = 0.0
    cache_save_time: float = 0.0

class LBPPatternCache:
    """
    Caché inteligente para patrones LBP frecuentes
    
    Utiliza una combinación de LRU y frecuencia de uso para optimizar
    el rendimiento en el cálculo de patrones LBP repetitivos.
    """
    
    def __init__(self, max_memory_mb: int = 100, max_entries: int = 1000):
        """
        Inicializa el caché LBP
        
        Args:
            max_memory_mb: Memoria máxima en MB para el caché
            max_entries: Número máximo de entradas en el caché
        """
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.max_entries = max_entries
        
        # Almacenamiento principal del caché
        self._cache: OrderedDict[str, LBPCacheEntry] = OrderedDict()
        
        # Seguimiento de frecuencia de patrones
        self._pattern_frequency: defaultdict[Tuple, int] = defaultdict(int)
        
        # Estadísticas
        self.stats = LBPCacheStats()
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Configuración de evicción
        self._eviction_threshold = 0.8  # Evictar cuando se use 80% de memoria
        self.eviction_policy = 'lru'  # Política de evicción por defecto
        
        logger.info(f"Inicializado caché LBP: {max_memory_mb}MB, {max_entries} entradas máx")
    
    def _generate_cache_key(self, image: np.ndarray, n_points: int, 
                          radius: float, method: str = 'uniform') -> str:
        """
        Genera una clave única para la combinación imagen-parámetros
        
        Args:
            image: Imagen de entrada
            n_points: Número de puntos LBP
            radius: Radio LBP
            method: Método LBP
            
        Returns:
            Clave hash única
        """
        # Hash de la imagen (usando una muestra para eficiencia)
        h, w = image.shape
        sample_points = min(1000, h * w // 10)  # Muestrear 10% o máximo 1000 puntos
        
        if sample_points < h * w:
            # Muestreo estratificado
            step_h = max(1, h // int(np.sqrt(sample_points)))
            step_w = max(1, w // int(np.sqrt(sample_points)))
            sample = image[::step_h, ::step_w]
        else:
            sample = image
        
        # Crear hash combinado
        image_hash = hashlib.md5(sample.tobytes()).hexdigest()[:16]
        param_hash = hashlib.md5(f"{n_points}_{radius}_{method}".encode()).hexdigest()[:8]
        
        return f"{image_hash}_{param_hash}"
    
    def _calculate_memory_size(self, lbp_result: np.ndarray, 
                             histogram: Optional[np.ndarray] = None) -> int:
        """Calcula el tamaño en memoria de una entrada del caché"""
        size = lbp_result.nbytes
        if histogram is not None:
            size += histogram.nbytes
        return size
    
    def _should_evict(self) -> bool:
        """Determina si se debe realizar evicción"""
        return (self.stats.total_memory_used > self.max_memory_bytes * self._eviction_threshold or
                len(self._cache) > self.max_entries * self._eviction_threshold)
    
    def _evict_entries(self):
        """
        Evicta entradas usando una estrategia híbrida LRU + frecuencia
        """
        if not self._cache:
            return
        
        target_size = int(self.max_memory_bytes * 0.6)  # Reducir a 60% de capacidad
        target_entries = int(self.max_entries * 0.6)
        
        # Crear lista de candidatos para evicción con puntuación
        candidates = []
        current_time = time.time()
        
        for key, entry in self._cache.items():
            # Puntuación basada en frecuencia, recencia y edad
            frequency_score = entry.access_count
            recency_score = 1.0 / (current_time - entry.last_access + 1)
            age_penalty = current_time - entry.creation_time
            
            # Puntuación combinada (menor = más probable de evictar)
            score = frequency_score * recency_score - age_penalty * 0.001
            candidates.append((score, key, entry.memory_size))
        
        # Ordenar por puntuación (menor primero)
        candidates.sort(key=lambda x: x[0])
        
        # Evictar hasta alcanzar objetivos
        evicted_count = 0
        for score, key, size in candidates:
            if (self.stats.total_memory_used <= target_size and 
                len(self._cache) <= target_entries):
                break
            
            if key in self._cache:
                del self._cache[key]
                self.stats.total_memory_used -= size
                evicted_count += 1
        
        self.stats.evictions += evicted_count
        logger.debug(f"Evictadas {evicted_count} entradas del caché LBP")
    
    def get_lbp_pattern(self, image: np.ndarray, n_points: int, radius: float,
                       method: str = 'uniform', compute_histogram: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Obtiene patrón LBP del caché o lo calcula si no existe
        
        Args:
            image: Imagen de entrada
            n_points: Número de puntos LBP
            radius: Radio LBP
            method: Método LBP
            compute_histogram: Si calcular histograma
            
        Returns:
            Tupla (patrón_lbp, histograma)
        """
        with self._lock:
            # Generar clave
            cache_key = self._generate_cache_key(image, n_points, radius, method)
            
            # Actualizar frecuencia de patrón
            pattern_key = (n_points, radius, method)
            self._pattern_frequency[pattern_key] += 1
            
            # Verificar caché
            if cache_key in self._cache:
                entry = self._cache[cache_key]
                entry.access_count += 1
                entry.last_access = time.time()
                
                # Mover al final (LRU)
                self._cache.move_to_end(cache_key)
                
                self.stats.hits += 1
                logger.debug(f"Cache hit para patrón LBP {pattern_key}")
                
                return entry.lbp_result, entry.histogram
            
            # Cache miss - calcular patrón
            self.stats.misses += 1
            start_time = time.time()
            
            try:
                # Calcular LBP
                lbp_result = local_binary_pattern(image, n_points, radius, method=method)
                
                # Calcular histograma si se solicita
                histogram = None
                if compute_histogram:
                    if method == 'uniform':
                        n_bins = n_points + 2
                    else:
                        n_bins = 2 ** n_points
                    
                    histogram, _ = np.histogram(lbp_result.flatten(), 
                                             bins=n_bins, 
                                             range=(0, n_bins))
                    histogram = histogram.astype(np.float32)
                    histogram /= (histogram.sum() + 1e-7)  # Normalizar
                
                computation_time = time.time() - start_time
                
                # Actualizar estadísticas de tiempo
                if self.stats.average_computation_time == 0:
                    self.stats.average_computation_time = computation_time
                else:
                    self.stats.average_computation_time = (
                        self.stats.average_computation_time * 0.9 + 
                        computation_time * 0.1
                    )
                
                # Crear entrada del caché
                memory_size = self._calculate_memory_size(lbp_result, histogram)
                entry = LBPCacheEntry(
                    lbp_result=lbp_result.copy(),
                    histogram=histogram.copy() if histogram is not None else None,
                    metadata={
                        'n_points': n_points,
                        'radius': radius,
                        'method': method,
                        'image_shape': image.shape,
                        'computation_time': computation_time
                    },
                    memory_size=memory_size
                )
                
                # Verificar si necesitamos evictar antes de agregar
                if self._should_evict():
                    self._evict_entries()
                
                # Agregar al caché
                self._cache[cache_key] = entry
                self.stats.total_memory_used += memory_size
                self.stats.max_memory_used = max(self.stats.max_memory_used, 
                                               self.stats.total_memory_used)
                
                logger.debug(f"Calculado y cacheado patrón LBP {pattern_key} en {computation_time:.3f}s")
                
                return lbp_result, histogram
                
            except Exception as e:
                logger.error(f"Error calculando patrón LBP: {e}")
                raise
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Obtiene estadísticas detalladas del caché"""
        with self._lock:
            total_requests = self.stats.hits + self.stats.misses
            hit_rate = self.stats.hits / total_requests if total_requests > 0 else 0
            
            # Patrones más frecuentes
            top_patterns = sorted(self._pattern_frequency.items(), 
                                key=lambda x: x[1], reverse=True)[:5]
            
            return {
                'cache_size': len(self._cache),
                'memory_used_mb': self.stats.total_memory_used / (1024 * 1024),
                'memory_limit_mb': self.max_memory_bytes / (1024 * 1024),
                'hit_rate': hit_rate,
                'total_hits': self.stats.hits,
                'total_misses': self.stats.misses,
                'total_evictions': self.stats.evictions,
                'average_computation_time': self.stats.average_computation_time,
                'top_patterns': [
                    {
                        'n_points': pattern[0],
                        'radius': pattern[1], 
                        'method': pattern[2],
                        'frequency': freq
                    }
                    for pattern, freq in top_patterns
                ]
            }
    
    def clear_cache(self):
        """Limpia completamente el caché"""
        with self._lock:
            self._cache.clear()
            self._pattern_frequency.clear()
            self.stats = LBPCacheStats()
            logger.info("Caché LBP limpiado completamente")
    
    def optimize_cache(self):
        """Optimiza el caché eliminando entradas poco utilizadas"""
        with self._lock:
            if not self._cache:
                return
            
            # Encontrar umbral de frecuencia
            frequencies = [entry.access_count for entry in self._cache.values()]
            threshold = np.percentile(frequencies, 25)  # Eliminar 25% menos usados
            
            keys_to_remove = []
            for key, entry in self._cache.items():
                if entry.access_count <= threshold:
                    keys_to_remove.append(key)
            
            removed_count = 0
            for key in keys_to_remove:
                if key in self._cache:
                    entry = self._cache[key]
                    self.stats.total_memory_used -= entry.memory_size
                    del self._cache[key]
                    removed_count += 1
            
            logger.info(f"Optimización del caché: eliminadas {removed_count} entradas poco utilizadas")

# Instancia global del caché
_global_lbp_cache: Optional[LBPPatternCache] = None

def get_lbp_cache() -> LBPPatternCache:
    """Obtiene la instancia global del caché LBP"""
    global _global_lbp_cache
    if _global_lbp_cache is None:
        _global_lbp_cache = LBPPatternCache()
    return _global_lbp_cache

def cached_local_binary_pattern(image: np.ndarray, n_points: int, radius: float,
                               method: str = 'uniform', compute_histogram: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Versión cacheada de local_binary_pattern
    
    Args:
        image: Imagen de entrada
        n_points: Número de puntos LBP
        radius: Radio LBP
        method: Método LBP
        compute_histogram: Si calcular histograma normalizado
        
    Returns:
        Tupla (patrón_lbp, histograma_opcional)
    """
    cache = get_lbp_cache()
    return cache.get_lbp_pattern(image, n_points, radius, method, compute_histogram)

def get_lbp_cache_stats() -> Dict[str, Any]:
    """Obtiene estadísticas del caché LBP global"""
    cache = get_lbp_cache()
    return cache.get_cache_statistics()

def clear_lbp_cache():
    """Limpia el caché LBP global"""
    cache = get_lbp_cache()
    cache.clear_cache()

def optimize_lbp_cache():
    """Optimiza el cache LBP global"""
    cache = get_lbp_cache()
    cache.optimize_cache()


def set_lbp_cache_config(max_size: int = 1000, max_memory_mb: int = 100, 
                        eviction_policy: str = 'lru'):
    """
    Configura el cache LBP global
    
    Args:
        max_size: Número máximo de entradas en el cache
        max_memory_mb: Memoria máxima en MB
        eviction_policy: Política de evicción ('lru' o 'frequency')
    """
    global _global_lbp_cache
    
    # Crear nuevo cache con la configuración especificada
    _global_lbp_cache = LBPPatternCache(
        max_memory_mb=max_memory_mb,
        max_entries=max_size
    )
    
    # Configurar política de evicción
    _global_lbp_cache.eviction_policy = eviction_policy