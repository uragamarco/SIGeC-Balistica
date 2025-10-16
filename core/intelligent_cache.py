#!/usr/bin/env python3
"""
Sistema de Caché Inteligente para SIGeC-Balistica.
Proporciona caché adaptativo con invalidación automática, compresión y análisis de patrones.
"""

import time
import json
import pickle
import hashlib
import threading
import weakref
import gzip
import lz4.frame
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import defaultdict, OrderedDict
import numpy as np

# Importar psutil de forma opcional
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    # Mock básico para psutil
    class psutil:
        @staticmethod
        def virtual_memory():
            return type('obj', (object,), {'available': 1024*1024*1024, 'percent': 50})()
        
        @staticmethod
        def Process():
            return type('obj', (object,), {'memory_info': lambda: type('obj', (object,), {'rss': 1024*1024})()})()

# Configurar logging
logger = logging.getLogger(__name__)

class CacheStrategy(Enum):
    """Estrategias de caché."""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    TTL = "ttl"  # Time To Live
    ADAPTIVE = "adaptive"  # Adaptativo basado en patrones
    PREDICTIVE = "predictive"  # Predictivo basado en ML

class CompressionType(Enum):
    """Tipos de compresión."""
    NONE = "none"
    GZIP = "gzip"
    LZ4 = "lz4"
    AUTO = "auto"  # Selección automática

class CacheLevel(Enum):
    """Niveles de caché."""
    MEMORY = "memory"
    DISK = "disk"
    DISTRIBUTED = "distributed"

@dataclass
class CacheEntry:
    """Entrada de caché."""
    key: str
    value: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    ttl: Optional[float] = None
    size_bytes: int = 0
    compression: CompressionType = CompressionType.NONE
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_expired(self) -> bool:
        """Verificar si la entrada ha expirado."""
        if self.ttl is None:
            return False
        return time.time() - self.created_at.timestamp() > self.ttl
    
    @property
    def age_seconds(self) -> float:
        """Edad de la entrada en segundos."""
        return time.time() - self.created_at.timestamp()
    
    def touch(self):
        """Marcar como accedida."""
        self.last_accessed = datetime.now()
        self.access_count += 1

@dataclass
class CacheStats:
    """Estadísticas de caché."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size_bytes: int = 0
    entry_count: int = 0
    compression_ratio: float = 0.0
    
    @property
    def hit_rate(self) -> float:
        """Tasa de aciertos."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    @property
    def miss_rate(self) -> float:
        """Tasa de fallos."""
        return 1.0 - self.hit_rate

class IntelligentCache:
    """Sistema de caché inteligente con múltiples estrategias."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Inicializar caché inteligente."""
        self.config = config or {}
        
        # Configuración
        self.max_memory_mb = self.config.get('max_memory_mb', 512)
        self.max_entries = self.config.get('max_entries', 10000)
        self.default_ttl = self.config.get('default_ttl', 3600)  # 1 hora
        self.strategy = CacheStrategy(self.config.get('strategy', 'adaptive'))
        self.compression = CompressionType(self.config.get('compression', 'auto'))
        self.enable_disk_cache = self.config.get('enable_disk_cache', True)
        self.disk_cache_dir = Path(self.config.get('disk_cache_dir', 'cache'))
        self.cleanup_interval = self.config.get('cleanup_interval', 300)  # 5 minutos
        self.enable_analytics = self.config.get('enable_analytics', True)
        
        # Estado interno
        self._memory_cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._access_patterns: Dict[str, List[float]] = defaultdict(list)
        self._key_dependencies: Dict[str, Set[str]] = defaultdict(set)
        self._invalidation_callbacks: Dict[str, List[Callable]] = defaultdict(list)
        self._stats = CacheStats()
        
        # Threading
        self._lock = threading.RLock()
        self._cleanup_thread: Optional[threading.Thread] = None
        self._running = False
        
        # Análisis predictivo
        self._prediction_model = None
        self._feature_history: List[Dict[str, float]] = []
        
        # Inicializar
        self._initialize()
    
    def _initialize(self):
        """Inicializar sistema de caché."""
        
        # Crear directorio de caché en disco
        if self.enable_disk_cache:
            self.disk_cache_dir.mkdir(parents=True, exist_ok=True)
            self._load_disk_cache()
        
        # Iniciar hilo de limpieza
        self._start_cleanup_thread()
        
        logger.info(f"Caché inteligente inicializado - Estrategia: {self.strategy.value}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Obtener valor del caché."""
        
        with self._lock:
            # Buscar en memoria
            if key in self._memory_cache:
                entry = self._memory_cache[key]
                
                # Verificar expiración
                if entry.is_expired:
                    self._remove_entry(key)
                    self._stats.misses += 1
                    return default
                
                # Actualizar acceso
                entry.touch()
                self._record_access_pattern(key)
                
                # Mover al final (LRU)
                self._memory_cache.move_to_end(key)
                
                self._stats.hits += 1
                
                # Descomprimir si es necesario
                value = self._decompress_value(entry.value, entry.compression)
                return value
            
            # Buscar en disco
            if self.enable_disk_cache:
                disk_value = self._get_from_disk(key)
                if disk_value is not None:
                    # Promover a memoria
                    self._promote_to_memory(key, disk_value)
                    self._stats.hits += 1
                    return disk_value
            
            self._stats.misses += 1
            return default
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None,
            dependencies: Optional[Set[str]] = None,
            metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Almacenar valor en caché."""
        
        with self._lock:
            try:
                # Preparar entrada
                now = datetime.now()
                entry_ttl = ttl if ttl is not None else self.default_ttl
                
                # Comprimir valor
                compressed_value, compression_type = self._compress_value(value)
                
                # Calcular tamaño
                size_bytes = self._calculate_size(compressed_value)
                
                # Crear entrada
                entry = CacheEntry(
                    key=key,
                    value=compressed_value,
                    created_at=now,
                    last_accessed=now,
                    ttl=entry_ttl,
                    size_bytes=size_bytes,
                    compression=compression_type,
                    metadata=metadata or {}
                )
                
                # Verificar espacio disponible
                if not self._ensure_space(size_bytes):
                    logger.warning(f"No se pudo hacer espacio para clave: {key}")
                    return False
                
                # Almacenar en memoria
                if key in self._memory_cache:
                    old_entry = self._memory_cache[key]
                    self._stats.size_bytes -= old_entry.size_bytes
                
                self._memory_cache[key] = entry
                self._stats.size_bytes += size_bytes
                self._stats.entry_count = len(self._memory_cache)
                
                # Gestionar dependencias
                if dependencies:
                    self._key_dependencies[key] = dependencies
                    for dep_key in dependencies:
                        self._invalidation_callbacks[dep_key].append(
                            lambda: self.invalidate(key)
                        )
                
                # Almacenar en disco si está habilitado
                if self.enable_disk_cache:
                    self._save_to_disk(key, value, entry)
                
                # Registrar patrón de acceso
                self._record_access_pattern(key)
                
                logger.debug(f"Cacheado: {key} ({size_bytes} bytes, {compression_type.value})")
                return True
                
            except Exception as e:
                logger.error(f"Error almacenando en caché {key}: {e}")
                return False
    
    def invalidate(self, key: str) -> bool:
        """Invalidar entrada específica."""
        
        with self._lock:
            removed = False
            
            # Remover de memoria
            if key in self._memory_cache:
                self._remove_entry(key)
                removed = True
            
            # Remover de disco
            if self.enable_disk_cache:
                disk_removed = self._remove_from_disk(key)
                removed = removed or disk_removed
            
            # Ejecutar callbacks de invalidación
            if key in self._invalidation_callbacks:
                for callback in self._invalidation_callbacks[key]:
                    try:
                        callback()
                    except Exception as e:
                        logger.error(f"Error en callback de invalidación: {e}")
                del self._invalidation_callbacks[key]
            
            # Limpiar dependencias
            if key in self._key_dependencies:
                del self._key_dependencies[key]
            
            if removed:
                logger.debug(f"Invalidado: {key}")
            
            return removed
    
    def invalidate_pattern(self, pattern: str) -> int:
        """Invalidar entradas que coincidan con patrón."""
        
        import fnmatch
        
        with self._lock:
            keys_to_remove = []
            
            # Buscar claves que coincidan
            for key in self._memory_cache.keys():
                if fnmatch.fnmatch(key, pattern):
                    keys_to_remove.append(key)
            
            # Remover claves
            for key in keys_to_remove:
                self.invalidate(key)
            
            logger.info(f"Invalidadas {len(keys_to_remove)} entradas con patrón: {pattern}")
            return len(keys_to_remove)
    
    def invalidate_dependencies(self, dependency_key: str) -> int:
        """Invalidar todas las entradas que dependan de una clave."""
        
        with self._lock:
            invalidated = 0
            
            # Buscar entradas dependientes
            keys_to_invalidate = []
            for key, deps in self._key_dependencies.items():
                if dependency_key in deps:
                    keys_to_invalidate.append(key)
            
            # Invalidar entradas dependientes
            for key in keys_to_invalidate:
                if self.invalidate(key):
                    invalidated += 1
            
            logger.info(f"Invalidadas {invalidated} entradas dependientes de: {dependency_key}")
            return invalidated
    
    def clear(self) -> bool:
        """Limpiar todo el caché."""
        
        with self._lock:
            try:
                # Limpiar memoria
                self._memory_cache.clear()
                
                # Limpiar disco
                if self.enable_disk_cache:
                    self._clear_disk_cache()
                
                # Resetear estadísticas
                self._stats = CacheStats()
                
                # Limpiar estructuras auxiliares
                self._access_patterns.clear()
                self._key_dependencies.clear()
                self._invalidation_callbacks.clear()
                
                logger.info("Caché completamente limpiado")
                return True
                
            except Exception as e:
                logger.error(f"Error limpiando caché: {e}")
                return False
    
    def get_stats(self) -> CacheStats:
        """Obtener estadísticas del caché."""
        
        with self._lock:
            # Actualizar estadísticas actuales
            self._stats.entry_count = len(self._memory_cache)
            
            # Calcular ratio de compresión
            if self._stats.entry_count > 0:
                total_compressed = sum(entry.size_bytes for entry in self._memory_cache.values())
                total_uncompressed = sum(
                    self._calculate_size(self._decompress_value(entry.value, entry.compression))
                    for entry in self._memory_cache.values()
                )
                if total_uncompressed > 0:
                    self._stats.compression_ratio = 1.0 - (total_compressed / total_uncompressed)
            
            return self._stats
    
    def get_analytics(self) -> Dict[str, Any]:
        """Obtener análisis detallado del caché."""
        
        with self._lock:
            stats = self.get_stats()
            
            # Análisis de patrones de acceso
            access_analysis = self._analyze_access_patterns()
            
            # Top entradas por acceso
            top_entries = sorted(
                self._memory_cache.items(),
                key=lambda x: x[1].access_count,
                reverse=True
            )[:10]
            
            # Distribución de tamaños
            sizes = [entry.size_bytes for entry in self._memory_cache.values()]
            size_stats = {
                'min': min(sizes) if sizes else 0,
                'max': max(sizes) if sizes else 0,
                'avg': sum(sizes) / len(sizes) if sizes else 0,
                'total': sum(sizes)
            }
            
            # Distribución de TTL
            ttls = [entry.ttl for entry in self._memory_cache.values() if entry.ttl]
            ttl_stats = {
                'min': min(ttls) if ttls else 0,
                'max': max(ttls) if ttls else 0,
                'avg': sum(ttls) / len(ttls) if ttls else 0
            }
            
            # Uso de memoria
            memory_usage = {
                'used_mb': self._stats.size_bytes / (1024 * 1024),
                'max_mb': self.max_memory_mb,
                'usage_percent': (self._stats.size_bytes / (1024 * 1024)) / self.max_memory_mb * 100
            }
            
            return {
                'timestamp': datetime.now().isoformat(),
                'basic_stats': {
                    'hits': stats.hits,
                    'misses': stats.misses,
                    'hit_rate': stats.hit_rate,
                    'evictions': stats.evictions,
                    'entry_count': stats.entry_count,
                    'compression_ratio': stats.compression_ratio
                },
                'memory_usage': memory_usage,
                'size_distribution': size_stats,
                'ttl_distribution': ttl_stats,
                'access_patterns': access_analysis,
                'top_entries': [
                    {
                        'key': key,
                        'access_count': entry.access_count,
                        'size_bytes': entry.size_bytes,
                        'age_seconds': entry.age_seconds
                    }
                    for key, entry in top_entries
                ],
                'strategy': self.strategy.value,
                'compression': self.compression.value
            }
    
    def optimize(self) -> Dict[str, Any]:
        """Optimizar caché basado en patrones de uso."""
        
        with self._lock:
            optimization_results = {
                'actions_taken': [],
                'space_freed_mb': 0,
                'entries_optimized': 0
            }
            
            # 1. Limpiar entradas expiradas
            expired_keys = [
                key for key, entry in self._memory_cache.items()
                if entry.is_expired
            ]
            
            for key in expired_keys:
                entry = self._memory_cache[key]
                optimization_results['space_freed_mb'] += entry.size_bytes / (1024 * 1024)
                self._remove_entry(key)
                optimization_results['entries_optimized'] += 1
            
            if expired_keys:
                optimization_results['actions_taken'].append(
                    f"Removidas {len(expired_keys)} entradas expiradas"
                )
            
            # 2. Optimizar compresión
            compression_optimized = self._optimize_compression()
            optimization_results['actions_taken'].extend(compression_optimized)
            
            # 3. Ajustar estrategia si es adaptativa
            if self.strategy == CacheStrategy.ADAPTIVE:
                strategy_changes = self._adapt_strategy()
                optimization_results['actions_taken'].extend(strategy_changes)
            
            # 4. Predicción y precarga
            if self.strategy == CacheStrategy.PREDICTIVE:
                predictions = self._predict_access_patterns()
                optimization_results['predictions'] = predictions
            
            logger.info(f"Optimización completada: {optimization_results}")
            return optimization_results
    
    # Métodos privados
    
    def _compress_value(self, value: Any) -> Tuple[Any, CompressionType]:
        """Comprimir valor según configuración."""
        
        if self.compression == CompressionType.NONE:
            return value, CompressionType.NONE
        
        try:
            # Serializar valor
            serialized = pickle.dumps(value)
            original_size = len(serialized)
            
            # Decidir compresión
            compression_type = self.compression
            if compression_type == CompressionType.AUTO:
                # Usar LZ4 para datos pequeños, GZIP para grandes
                compression_type = CompressionType.LZ4 if original_size < 1024 else CompressionType.GZIP
            
            # Comprimir
            if compression_type == CompressionType.GZIP:
                compressed = gzip.compress(serialized)
            elif compression_type == CompressionType.LZ4:
                compressed = lz4.frame.compress(serialized)
            else:
                compressed = serialized
                compression_type = CompressionType.NONE
            
            # Solo usar compresión si es beneficiosa
            if len(compressed) < original_size * 0.9:  # Al menos 10% de reducción
                return compressed, compression_type
            else:
                return value, CompressionType.NONE
                
        except Exception as e:
            logger.warning(f"Error comprimiendo valor: {e}")
            return value, CompressionType.NONE
    
    def _decompress_value(self, compressed_value: Any, compression: CompressionType) -> Any:
        """Descomprimir valor."""
        
        if compression == CompressionType.NONE:
            return compressed_value
        
        try:
            if compression == CompressionType.GZIP:
                serialized = gzip.decompress(compressed_value)
            elif compression == CompressionType.LZ4:
                serialized = lz4.frame.decompress(compressed_value)
            else:
                return compressed_value
            
            return pickle.loads(serialized)
            
        except Exception as e:
            logger.error(f"Error descomprimiendo valor: {e}")
            return compressed_value
    
    def _calculate_size(self, value: Any) -> int:
        """Calcular tamaño aproximado de un valor."""
        
        try:
            if isinstance(value, (bytes, bytearray)):
                return len(value)
            elif isinstance(value, str):
                return len(value.encode('utf-8'))
            elif isinstance(value, (int, float)):
                return 8
            elif isinstance(value, (list, tuple)):
                return sum(self._calculate_size(item) for item in value)
            elif isinstance(value, dict):
                return sum(
                    self._calculate_size(k) + self._calculate_size(v)
                    for k, v in value.items()
                )
            else:
                # Usar pickle para estimación
                return len(pickle.dumps(value))
        except Exception:
            return 1024  # Estimación por defecto
    
    def _ensure_space(self, required_bytes: int) -> bool:
        """Asegurar espacio disponible en caché."""
        
        max_bytes = self.max_memory_mb * 1024 * 1024
        
        # Verificar límite de entradas
        if len(self._memory_cache) >= self.max_entries:
            self._evict_entries(1)
        
        # Verificar límite de memoria
        while (self._stats.size_bytes + required_bytes > max_bytes and 
               len(self._memory_cache) > 0):
            if not self._evict_entries(1):
                return False
        
        return True
    
    def _evict_entries(self, count: int) -> bool:
        """Expulsar entradas según estrategia."""
        
        if not self._memory_cache:
            return False
        
        evicted = 0
        
        if self.strategy == CacheStrategy.LRU:
            # Remover las menos recientemente usadas
            keys_to_remove = list(self._memory_cache.keys())[:count]
        
        elif self.strategy == CacheStrategy.LFU:
            # Remover las menos frecuentemente usadas
            sorted_entries = sorted(
                self._memory_cache.items(),
                key=lambda x: x[1].access_count
            )
            keys_to_remove = [key for key, _ in sorted_entries[:count]]
        
        elif self.strategy == CacheStrategy.TTL:
            # Remover las que expiran primero
            sorted_entries = sorted(
                self._memory_cache.items(),
                key=lambda x: x[1].created_at.timestamp() + (x[1].ttl or 0)
            )
            keys_to_remove = [key for key, _ in sorted_entries[:count]]
        
        else:  # ADAPTIVE o PREDICTIVE
            # Usar puntuación combinada
            scored_entries = []
            for key, entry in self._memory_cache.items():
                score = self._calculate_eviction_score(key, entry)
                scored_entries.append((key, score))
            
            scored_entries.sort(key=lambda x: x[1])
            keys_to_remove = [key for key, _ in scored_entries[:count]]
        
        # Remover entradas seleccionadas
        for key in keys_to_remove:
            if key in self._memory_cache:
                self._remove_entry(key)
                evicted += 1
                self._stats.evictions += 1
        
        return evicted > 0
    
    def _calculate_eviction_score(self, key: str, entry: CacheEntry) -> float:
        """Calcular puntuación para expulsión (menor = más probable de expulsar)."""
        
        now = time.time()
        
        # Factores base
        age_factor = entry.age_seconds / 3600  # Normalizar a horas
        access_factor = 1.0 / (entry.access_count + 1)
        size_factor = entry.size_bytes / (1024 * 1024)  # MB
        
        # Factor de patrón de acceso
        pattern_factor = 1.0
        if key in self._access_patterns:
            recent_accesses = [
                t for t in self._access_patterns[key]
                if now - t < 3600  # Última hora
            ]
            if recent_accesses:
                pattern_factor = 1.0 / (len(recent_accesses) + 1)
        
        # Combinar factores
        score = (age_factor * 0.3 + 
                access_factor * 0.3 + 
                size_factor * 0.2 + 
                pattern_factor * 0.2)
        
        return score
    
    def _remove_entry(self, key: str):
        """Remover entrada del caché."""
        
        if key in self._memory_cache:
            entry = self._memory_cache[key]
            self._stats.size_bytes -= entry.size_bytes
            del self._memory_cache[key]
            self._stats.entry_count = len(self._memory_cache)
    
    def _record_access_pattern(self, key: str):
        """Registrar patrón de acceso."""
        
        if not self.enable_analytics:
            return
        
        now = time.time()
        self._access_patterns[key].append(now)
        
        # Mantener solo los últimos 100 accesos
        if len(self._access_patterns[key]) > 100:
            self._access_patterns[key] = self._access_patterns[key][-100:]
    
    def _analyze_access_patterns(self) -> Dict[str, Any]:
        """Analizar patrones de acceso."""
        
        if not self.enable_analytics:
            return {}
        
        now = time.time()
        analysis = {
            'total_keys_tracked': len(self._access_patterns),
            'hot_keys': [],
            'cold_keys': [],
            'access_frequency_distribution': {}
        }
        
        # Analizar cada clave
        for key, accesses in self._access_patterns.items():
            recent_accesses = [t for t in accesses if now - t < 3600]  # Última hora
            
            if len(recent_accesses) > 10:
                analysis['hot_keys'].append({
                    'key': key,
                    'recent_accesses': len(recent_accesses),
                    'total_accesses': len(accesses)
                })
            elif len(recent_accesses) == 0 and len(accesses) > 0:
                analysis['cold_keys'].append({
                    'key': key,
                    'last_access_hours_ago': (now - max(accesses)) / 3600,
                    'total_accesses': len(accesses)
                })
        
        # Ordenar por relevancia
        analysis['hot_keys'].sort(key=lambda x: x['recent_accesses'], reverse=True)
        analysis['cold_keys'].sort(key=lambda x: x['last_access_hours_ago'], reverse=True)
        
        # Limitar resultados
        analysis['hot_keys'] = analysis['hot_keys'][:10]
        analysis['cold_keys'] = analysis['cold_keys'][:10]
        
        return analysis
    
    def _optimize_compression(self) -> List[str]:
        """Optimizar compresión de entradas existentes."""
        
        actions = []
        
        for key, entry in list(self._memory_cache.items()):
            if entry.compression == CompressionType.NONE and entry.size_bytes > 1024:
                # Intentar comprimir entrada grande sin compresión
                try:
                    original_value = entry.value
                    compressed_value, compression_type = self._compress_value(original_value)
                    
                    if compression_type != CompressionType.NONE:
                        # Actualizar entrada
                        old_size = entry.size_bytes
                        entry.value = compressed_value
                        entry.compression = compression_type
                        entry.size_bytes = self._calculate_size(compressed_value)
                        
                        # Actualizar estadísticas
                        self._stats.size_bytes += entry.size_bytes - old_size
                        
                        actions.append(f"Comprimida entrada {key}: {old_size} -> {entry.size_bytes} bytes")
                
                except Exception as e:
                    logger.warning(f"Error optimizando compresión para {key}: {e}")
        
        return actions
    
    def _adapt_strategy(self) -> List[str]:
        """Adaptar estrategia basada en patrones de uso."""
        
        actions = []
        stats = self.get_stats()
        
        # Analizar rendimiento actual
        if stats.hit_rate < 0.5:  # Baja tasa de aciertos
            # Considerar cambiar a estrategia más agresiva
            actions.append("Detectada baja tasa de aciertos - considerando optimizaciones")
        
        if stats.entry_count > self.max_entries * 0.9:  # Cerca del límite
            # Ser más agresivo con expulsiones
            actions.append("Caché cerca del límite - aumentando agresividad de expulsión")
        
        return actions
    
    def _predict_access_patterns(self) -> Dict[str, Any]:
        """Predecir patrones de acceso futuros."""
        
        # Implementación simplificada de predicción
        predictions = {
            'likely_hot_keys': [],
            'likely_cold_keys': [],
            'recommended_preloads': []
        }
        
        now = time.time()
        
        for key, accesses in self._access_patterns.items():
            if len(accesses) < 3:
                continue
            
            # Calcular tendencia de acceso
            recent_accesses = [t for t in accesses if now - t < 7200]  # Últimas 2 horas
            older_accesses = [t for t in accesses if 7200 <= now - t < 14400]  # 2-4 horas atrás
            
            if len(recent_accesses) > len(older_accesses) * 1.5:
                predictions['likely_hot_keys'].append(key)
            elif len(recent_accesses) < len(older_accesses) * 0.5:
                predictions['likely_cold_keys'].append(key)
        
        return predictions
    
    def _start_cleanup_thread(self):
        """Iniciar hilo de limpieza automática."""
        
        def cleanup_worker():
            while self._running:
                try:
                    time.sleep(self.cleanup_interval)
                    if self._running:
                        self._cleanup_expired()
                except Exception as e:
                    logger.error(f"Error en hilo de limpieza: {e}")
        
        self._running = True
        self._cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        self._cleanup_thread.start()
    
    def _cleanup_expired(self):
        """Limpiar entradas expiradas."""
        
        with self._lock:
            expired_keys = [
                key for key, entry in self._memory_cache.items()
                if entry.is_expired
            ]
            
            for key in expired_keys:
                self._remove_entry(key)
            
            if expired_keys:
                logger.debug(f"Limpiadas {len(expired_keys)} entradas expiradas")
    
    def _get_from_disk(self, key: str) -> Any:
        """Obtener valor del caché en disco."""
        
        try:
            cache_file = self.disk_cache_dir / f"{hashlib.md5(key.encode()).hexdigest()}.cache"
            if cache_file.exists():
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                    
                # Verificar expiración
                if data.get('expires_at') and datetime.now() > data['expires_at']:
                    cache_file.unlink()
                    return None
                
                return data['value']
        except Exception as e:
            logger.warning(f"Error leyendo caché de disco para {key}: {e}")
        
        return None
    
    def _save_to_disk(self, key: str, value: Any, entry: CacheEntry):
        """Guardar valor en caché de disco."""
        
        try:
            cache_file = self.disk_cache_dir / f"{hashlib.md5(key.encode()).hexdigest()}.cache"
            
            data = {
                'key': key,
                'value': value,
                'created_at': entry.created_at,
                'expires_at': entry.created_at + timedelta(seconds=entry.ttl) if entry.ttl else None,
                'metadata': entry.metadata
            }
            
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
                
        except Exception as e:
            logger.warning(f"Error guardando caché de disco para {key}: {e}")
    
    def _remove_from_disk(self, key: str) -> bool:
        """Remover valor del caché de disco."""
        
        try:
            cache_file = self.disk_cache_dir / f"{hashlib.md5(key.encode()).hexdigest()}.cache"
            if cache_file.exists():
                cache_file.unlink()
                return True
        except Exception as e:
            logger.warning(f"Error removiendo caché de disco para {key}: {e}")
        
        return False
    
    def _load_disk_cache(self):
        """Cargar entradas existentes del caché de disco."""
        
        try:
            cache_files = list(self.disk_cache_dir.glob("*.cache"))
            loaded = 0
            
            for cache_file in cache_files:
                try:
                    with open(cache_file, 'rb') as f:
                        data = pickle.load(f)
                    
                    # Verificar expiración
                    if data.get('expires_at') and datetime.now() > data['expires_at']:
                        cache_file.unlink()
                        continue
                    
                    loaded += 1
                    
                except Exception as e:
                    logger.warning(f"Error cargando {cache_file}: {e}")
                    cache_file.unlink()  # Remover archivo corrupto
            
            if loaded > 0:
                logger.info(f"Cargadas {loaded} entradas del caché de disco")
                
        except Exception as e:
            logger.error(f"Error cargando caché de disco: {e}")
    
    def _clear_disk_cache(self):
        """Limpiar caché de disco."""
        
        try:
            cache_files = list(self.disk_cache_dir.glob("*.cache"))
            for cache_file in cache_files:
                cache_file.unlink()
            logger.info(f"Limpiados {len(cache_files)} archivos de caché de disco")
        except Exception as e:
            logger.error(f"Error limpiando caché de disco: {e}")
    
    def _promote_to_memory(self, key: str, value: Any):
        """Promover entrada de disco a memoria."""
        
        # Usar TTL por defecto para entradas promovidas
        self.set(key, value, ttl=self.default_ttl)
    
    def shutdown(self):
        """Cerrar sistema de caché."""
        
        self._running = False
        
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=5)
        
        # Guardar estadísticas finales
        if self.enable_disk_cache:
            try:
                stats_file = self.disk_cache_dir / 'cache_stats.json'
                with open(stats_file, 'w') as f:
                    json.dump(self.get_analytics(), f, indent=2, default=str)
            except Exception as e:
                logger.error(f"Error guardando estadísticas finales: {e}")
        
        logger.info("Sistema de caché cerrado")

# Instancia global
_cache_instance: Optional[IntelligentCache] = None

def get_cache() -> IntelligentCache:
    """Obtener instancia global del caché."""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = IntelligentCache()
    return _cache_instance

def initialize_cache(config: Dict[str, Any] = None) -> IntelligentCache:
    """Inicializar sistema de caché."""
    global _cache_instance
    _cache_instance = IntelligentCache(config)
    return _cache_instance

# Decoradores de caché

def cached(ttl: Optional[float] = None, key_prefix: str = "",
          dependencies: Optional[Set[str]] = None):
    """Decorador para cachear resultados de funciones."""
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generar clave de caché
            cache_key = f"{key_prefix}{func.__name__}:{hashlib.md5(str(args + tuple(sorted(kwargs.items()))).encode()).hexdigest()}"
            
            # Intentar obtener del caché
            cache = get_cache()
            result = cache.get(cache_key)
            
            if result is not None:
                return result
            
            # Ejecutar función y cachear resultado
            result = func(*args, **kwargs)
            cache.set(cache_key, result, ttl=ttl, dependencies=dependencies)
            
            return result
        
        return wrapper
    return decorator

def cache_invalidate(pattern: str):
    """Decorador para invalidar caché después de ejecutar función."""
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            
            # Invalidar caché
            cache = get_cache()
            cache.invalidate_pattern(pattern)
            
            return result
        
        return wrapper
    return decorator

# ============================================================================
# COMPATIBILIDAD CON MEMORY_CACHE.PY
# ============================================================================

@dataclass
class CacheStats:
    """Estadísticas del caché (compatibilidad con memory_cache)"""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    memory_usage_mb: float = 0.0
    total_items: int = 0
    
    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

class MemoryCache:
    """
    Clase de compatibilidad que envuelve IntelligentCache
    para mantener la API simple de memory_cache.py
    """
    
    def __init__(self, 
                 max_size: int = 1000,
                 max_memory_mb: float = 500.0,
                 default_ttl: Optional[float] = 3600.0,
                 cleanup_interval: float = 300.0,
                 memory_threshold: float = 0.8):
        """Inicializa el caché con compatibilidad memory_cache"""
        
        config = {
            'max_entries': max_size,
            'max_memory_mb': max_memory_mb,
            'default_ttl': default_ttl,
            'cleanup_interval': cleanup_interval,
            'strategy': 'lru',  # Usar LRU para compatibilidad
            'compression': 'none',  # Sin compresión por defecto
            'enable_disk_cache': False  # Solo memoria por defecto
        }
        
        self._intelligent_cache = IntelligentCache(config)
        self.max_size = max_size
        self.max_memory_mb = max_memory_mb
        self.default_ttl = default_ttl
    
    def put(self, key: Union[str, tuple, Any], value: Any, ttl: Optional[float] = None) -> bool:
        """Almacenar valor (compatibilidad)"""
        key_str = self._generate_key(key)
        return self._intelligent_cache.set(key_str, value, ttl=ttl)
    
    def get(self, key: Union[str, tuple, Any]) -> Optional[Any]:
        """Obtener valor (compatibilidad)"""
        key_str = self._generate_key(key)
        return self._intelligent_cache.get(key_str)
    
    def get_or_compute(self, key: Union[str, tuple, Any], 
                      compute_func: Callable[[], Any],
                      ttl: Optional[float] = None) -> Any:
        """Obtener o computar valor (compatibilidad)"""
        key_str = self._generate_key(key)
        result = self._intelligent_cache.get(key_str)
        
        if result is None:
            result = compute_func()
            self._intelligent_cache.set(key_str, result, ttl=ttl)
        
        return result
    
    def invalidate(self, key: Union[str, tuple, Any]) -> bool:
        """Invalidar clave (compatibilidad)"""
        key_str = self._generate_key(key)
        return self._intelligent_cache.invalidate(key_str)
    
    def clear(self):
        """Limpiar caché (compatibilidad)"""
        self._intelligent_cache.clear()
    
    def size(self) -> int:
        """Obtener tamaño del caché"""
        stats = self._intelligent_cache.get_stats()
        return stats.entry_count
    
    def get_stats(self) -> CacheStats:
        """Obtener estadísticas (compatibilidad)"""
        intelligent_stats = self._intelligent_cache.get_stats()
        
        return CacheStats(
            hits=intelligent_stats.hits,
            misses=intelligent_stats.misses,
            evictions=intelligent_stats.evictions,
            memory_usage_mb=intelligent_stats.size_bytes / (1024 * 1024),
            total_items=intelligent_stats.entry_count
        )
    
    def shutdown(self):
        """Cerrar caché (compatibilidad)"""
        self._intelligent_cache.shutdown()
    
    def _generate_key(self, key: Union[str, tuple, Any]) -> str:
        """Generar clave string (compatibilidad)"""
        if isinstance(key, str):
            return key
        elif isinstance(key, (tuple, list)):
            return hashlib.md5(str(key).encode()).hexdigest()
        else:
            return hashlib.md5(str(key).encode()).hexdigest()

# Variables globales para compatibilidad
_global_cache: Optional[MemoryCache] = None
_cache_lock = threading.Lock()

def get_global_cache() -> MemoryCache:
    """Obtener instancia global del caché (compatibilidad)"""
    global _global_cache
    if _global_cache is None:
        with _cache_lock:
            if _global_cache is None:
                _global_cache = MemoryCache()
    return _global_cache

def cache_features(func: Callable) -> Callable:
    """Decorador para cachear características (compatibilidad)"""
    def wrapper(*args, **kwargs):
        cache_key = (func.__name__, args, tuple(sorted(kwargs.items())))
        
        cache = get_global_cache()
        return cache.get_or_compute(
            cache_key,
            lambda: func(*args, **kwargs),
            ttl=1800  # 30 minutos para características
        )
    
    return wrapper

def cache_matches(func: Callable) -> Callable:
    """Decorador para cachear resultados de matching (compatibilidad)"""
    def wrapper(*args, **kwargs):
        cache_key = (func.__name__, args, tuple(sorted(kwargs.items())))
        
        cache = get_global_cache()
        return cache.get_or_compute(
            cache_key,
            lambda: func(*args, **kwargs),
            ttl=900  # 15 minutos para matches
        )
    
    return wrapper

def clear_cache():
    """Limpiar caché global (compatibilidad)"""
    cache = get_global_cache()
    cache.clear()

def get_cache_stats() -> CacheStats:
    """Obtener estadísticas del caché global (compatibilidad)"""
    cache = get_global_cache()
    return cache.get_stats()

def configure_cache(max_size: int = 1000, max_memory_mb: float = 500.0):
    """Configurar caché global (compatibilidad)"""
    global _global_cache
    with _cache_lock:
        if _global_cache is not None:
            _global_cache.shutdown()
        _global_cache = MemoryCache(
            max_size=max_size,
            max_memory_mb=max_memory_mb
        )

# ============================================================================

if __name__ == "__main__":
    # Ejemplo de uso
    cache = initialize_cache({
        'max_memory_mb': 256,
        'strategy': 'adaptive',
        'compression': 'auto',
        'enable_disk_cache': True
    })
    
    # Cachear algunos valores
    cache.set("test_key", {"data": "test_value", "number": 42})
    cache.set("large_data", list(range(10000)), ttl=3600)
    
    # Obtener valores
    value = cache.get("test_key")
    print(f"Valor cacheado: {value}")
    
    # Obtener estadísticas
    stats = cache.get_stats()
    print(f"Estadísticas: Hit rate: {stats.hit_rate:.2%}, Entradas: {stats.entry_count}")
    
    # Obtener análisis
    analytics = cache.get_analytics()
    print(f"Análisis: {json.dumps(analytics, indent=2, default=str)}")
    
    # Optimizar
    optimization = cache.optimize()
    print(f"Optimización: {optimization}")
    
    # Cerrar
    cache.shutdown()
    
    # Ejemplo de compatibilidad con memory_cache
    print("\n--- Prueba de compatibilidad ---")
    memory_cache = MemoryCache(max_size=100)
    memory_cache.put("test", "value")
    print(f"Valor recuperado: {memory_cache.get('test')}")
    print(f"Estadísticas: {memory_cache.get_stats()}")
    memory_cache.shutdown()