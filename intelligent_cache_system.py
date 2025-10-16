#!/usr/bin/env python3
"""
Sistema de Caché Inteligente para SIGeC-Balistica
Sistema avanzado de caché con predicción, compresión y gestión automática.

Este módulo implementa un sistema de caché inteligente que:
- Predice qué datos serán necesarios
- Comprime automáticamente los datos
- Gestiona la memoria de forma eficiente
- Proporciona diferentes estrategias de caché
- Monitorea el rendimiento en tiempo real

Autor: Sistema SIGeC-Balistica
Fecha: 2024
"""

import os
import sys
import time
import threading
import asyncio
import logging
import hashlib
import pickle
import gzip
import lz4.frame
import json
from typing import Dict, List, Optional, Any, Callable, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import psutil
from collections import defaultdict, OrderedDict
import weakref

# Configurar logging
logger = logging.getLogger(__name__)

class CacheStrategy(Enum):
    """Estrategias de caché"""
    LRU = "lru"              # Least Recently Used
    LFU = "lfu"              # Least Frequently Used
    FIFO = "fifo"            # First In, First Out
    ADAPTIVE = "adaptive"     # Adaptativo basado en patrones
    PREDICTIVE = "predictive" # Predictivo con ML

class CompressionType(Enum):
    """Tipos de compresión"""
    NONE = "none"
    GZIP = "gzip"
    LZ4 = "lz4"
    ADAPTIVE = "adaptive"    # Selección automática

class CachePriority(Enum):
    """Prioridades de caché"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class CacheConfig:
    """Configuración del sistema de caché"""
    max_memory_mb: int = 512
    max_items: int = 1000
    default_ttl_seconds: int = 3600
    cleanup_interval_seconds: int = 300
    
    # Estrategias
    strategy: CacheStrategy = CacheStrategy.ADAPTIVE
    compression: CompressionType = CompressionType.ADAPTIVE
    
    # Configuración de compresión
    compression_threshold_bytes: int = 1024  # Comprimir si > 1KB
    compression_level: int = 6
    
    # Configuración predictiva
    enable_prediction: bool = True
    prediction_window_minutes: int = 30
    min_access_count_for_prediction: int = 3
    
    # Configuración de persistencia
    enable_persistence: bool = True
    persistence_file: str = "cache_data.pkl"
    save_interval_seconds: int = 600
    
    # Configuración de monitoreo
    enable_monitoring: bool = True
    monitoring_interval_seconds: int = 60

@dataclass
class CacheItem:
    """Elemento del caché"""
    key: str
    value: Any
    timestamp: datetime
    last_access: datetime
    access_count: int
    size_bytes: int
    priority: CachePriority
    ttl_seconds: Optional[int]
    compressed: bool = False
    compression_type: Optional[CompressionType] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CacheStats:
    """Estadísticas del caché"""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    compressions: int = 0
    decompressions: int = 0
    total_size_bytes: int = 0
    item_count: int = 0
    memory_usage_mb: float = 0.0
    hit_rate: float = 0.0
    compression_ratio: float = 0.0

class CompressionManager:
    """Gestor de compresión inteligente"""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.compression_stats = defaultdict(lambda: {"count": 0, "total_ratio": 0.0})
    
    def should_compress(self, data: Any, size_bytes: int) -> bool:
        """Determinar si se debe comprimir los datos"""
        if self.config.compression == CompressionType.NONE:
            return False
        
        if size_bytes < self.config.compression_threshold_bytes:
            return False
        
        # Para datos numéricos grandes, siempre comprimir
        if isinstance(data, np.ndarray) and data.nbytes > 10240:  # > 10KB
            return True
        
        return True
    
    def get_best_compression_type(self, data: Any) -> CompressionType:
        """Seleccionar el mejor tipo de compresión"""
        if self.config.compression != CompressionType.ADAPTIVE:
            return self.config.compression
        
        # Heurísticas para selección automática
        if isinstance(data, np.ndarray):
            return CompressionType.LZ4  # Mejor para datos numéricos
        elif isinstance(data, (str, dict, list)):
            return CompressionType.GZIP  # Mejor para texto/JSON
        else:
            return CompressionType.LZ4  # Por defecto
    
    def compress(self, data: Any, compression_type: CompressionType) -> Tuple[bytes, float]:
        """Comprimir datos"""
        try:
            # Serializar datos
            serialized = pickle.dumps(data)
            original_size = len(serialized)
            
            # Comprimir según el tipo
            if compression_type == CompressionType.GZIP:
                compressed = gzip.compress(serialized, compresslevel=self.config.compression_level)
            elif compression_type == CompressionType.LZ4:
                compressed = lz4.frame.compress(serialized, compression_level=self.config.compression_level)
            else:
                return serialized, 1.0
            
            # Calcular ratio de compresión
            compression_ratio = len(compressed) / original_size
            
            # Actualizar estadísticas
            self.compression_stats[compression_type.value]["count"] += 1
            self.compression_stats[compression_type.value]["total_ratio"] += compression_ratio
            
            return compressed, compression_ratio
            
        except Exception as e:
            logger.warning(f"Error comprimiendo datos: {e}")
            return pickle.dumps(data), 1.0
    
    def decompress(self, compressed_data: bytes, compression_type: CompressionType) -> Any:
        """Descomprimir datos"""
        try:
            if compression_type == CompressionType.GZIP:
                decompressed = gzip.decompress(compressed_data)
            elif compression_type == CompressionType.LZ4:
                decompressed = lz4.frame.decompress(compressed_data)
            else:
                decompressed = compressed_data
            
            return pickle.loads(decompressed)
            
        except Exception as e:
            logger.error(f"Error descomprimiendo datos: {e}")
            raise

class AccessPredictor:
    """Predictor de accesos futuros"""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.access_patterns = defaultdict(list)
        self.pattern_weights = defaultdict(float)
        self.lock = threading.Lock()
    
    def record_access(self, key: str, timestamp: datetime):
        """Registrar acceso para análisis de patrones"""
        with self.lock:
            self.access_patterns[key].append(timestamp)
            
            # Mantener solo accesos recientes
            cutoff = timestamp - timedelta(minutes=self.config.prediction_window_minutes)
            self.access_patterns[key] = [
                t for t in self.access_patterns[key] if t > cutoff
            ]
    
    def predict_next_accesses(self, current_time: datetime) -> List[Tuple[str, float]]:
        """Predecir próximos accesos"""
        predictions = []
        
        with self.lock:
            for key, accesses in self.access_patterns.items():
                if len(accesses) < self.config.min_access_count_for_prediction:
                    continue
                
                # Calcular probabilidad basada en frecuencia y recencia
                recent_accesses = [
                    a for a in accesses 
                    if (current_time - a).total_seconds() < 1800  # Últimos 30 min
                ]
                
                if not recent_accesses:
                    continue
                
                frequency_score = len(recent_accesses) / len(accesses)
                recency_score = 1.0 / (1.0 + (current_time - max(recent_accesses)).total_seconds() / 3600)
                
                probability = (frequency_score * 0.6) + (recency_score * 0.4)
                predictions.append((key, probability))
        
        # Ordenar por probabilidad descendente
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:10]  # Top 10 predicciones

class IntelligentCacheSystem:
    """Sistema de caché inteligente"""
    
    def __init__(self, config: CacheConfig = None):
        self.config = config or CacheConfig()
        self.cache: OrderedDict[str, CacheItem] = OrderedDict()
        self.stats = CacheStats()
        self.compression_manager = CompressionManager(self.config)
        self.access_predictor = AccessPredictor(self.config) if self.config.enable_prediction else None
        
        # Locks para thread safety
        self.cache_lock = threading.RLock()
        self.stats_lock = threading.Lock()
        
        # Tareas de mantenimiento
        self.running = False
        self.maintenance_tasks = []
        
        # Cargar caché persistente si está habilitado
        if self.config.enable_persistence:
            self._load_persistent_cache()
        
        # Iniciar tareas de mantenimiento
        self._start_maintenance_tasks()
    
    def _start_maintenance_tasks(self):
        """Iniciar tareas de mantenimiento"""
        self.running = True
        
        # Tarea de limpieza
        cleanup_task = threading.Thread(target=self._cleanup_loop, daemon=True)
        cleanup_task.start()
        self.maintenance_tasks.append(cleanup_task)
        
        # Tarea de persistencia
        if self.config.enable_persistence:
            persistence_task = threading.Thread(target=self._persistence_loop, daemon=True)
            persistence_task.start()
            self.maintenance_tasks.append(persistence_task)
        
        # Tarea de monitoreo
        if self.config.enable_monitoring:
            monitoring_task = threading.Thread(target=self._monitoring_loop, daemon=True)
            monitoring_task.start()
            self.maintenance_tasks.append(monitoring_task)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Obtener elemento del caché"""
        with self.cache_lock:
            if key in self.cache:
                item = self.cache[key]
                
                # Verificar TTL
                if self._is_expired(item):
                    del self.cache[key]
                    with self.stats_lock:
                        self.stats.misses += 1
                    return default
                
                # Actualizar estadísticas de acceso
                item.last_access = datetime.now()
                item.access_count += 1
                
                # Mover al final (LRU)
                self.cache.move_to_end(key)
                
                # Registrar acceso para predicción
                if self.access_predictor:
                    self.access_predictor.record_access(key, item.last_access)
                
                # Descomprimir si es necesario
                value = self._decompress_if_needed(item)
                
                with self.stats_lock:
                    self.stats.hits += 1
                    self._update_hit_rate()
                
                return value
            else:
                with self.stats_lock:
                    self.stats.misses += 1
                    self._update_hit_rate()
                return default
    
    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None, 
            priority: CachePriority = CachePriority.MEDIUM, metadata: Dict[str, Any] = None):
        """Almacenar elemento en el caché"""
        with self.cache_lock:
            now = datetime.now()
            
            # Calcular tamaño
            size_bytes = self._calculate_size(value)
            
            # Determinar si comprimir
            should_compress = self.compression_manager.should_compress(value, size_bytes)
            compressed_value = value
            compression_type = None
            
            if should_compress:
                compression_type = self.compression_manager.get_best_compression_type(value)
                compressed_data, compression_ratio = self.compression_manager.compress(value, compression_type)
                compressed_value = compressed_data
                size_bytes = len(compressed_data)
                
                with self.stats_lock:
                    self.stats.compressions += 1
            
            # Crear item del caché
            item = CacheItem(
                key=key,
                value=compressed_value,
                timestamp=now,
                last_access=now,
                access_count=1,
                size_bytes=size_bytes,
                priority=priority,
                ttl_seconds=ttl_seconds or self.config.default_ttl_seconds,
                compressed=should_compress,
                compression_type=compression_type,
                metadata=metadata or {}
            )
            
            # Verificar límites antes de agregar
            self._ensure_capacity(size_bytes)
            
            # Agregar al caché
            if key in self.cache:
                old_item = self.cache[key]
                with self.stats_lock:
                    self.stats.total_size_bytes -= old_item.size_bytes
            
            self.cache[key] = item
            
            with self.stats_lock:
                self.stats.total_size_bytes += size_bytes
                self.stats.item_count = len(self.cache)
                self._update_memory_usage()
    
    def _ensure_capacity(self, new_item_size: int):
        """Asegurar capacidad suficiente en el caché"""
        max_size_bytes = self.config.max_memory_mb * 1024 * 1024
        
        # Verificar límite de memoria
        while (self.stats.total_size_bytes + new_item_size > max_size_bytes or 
               len(self.cache) >= self.config.max_items):
            
            if not self.cache:
                break
            
            # Seleccionar elemento para evicción según estrategia
            key_to_evict = self._select_eviction_candidate()
            if key_to_evict:
                self._evict_item(key_to_evict)
            else:
                break
    
    def _select_eviction_candidate(self) -> Optional[str]:
        """Seleccionar candidato para evicción"""
        if not self.cache:
            return None
        
        if self.config.strategy == CacheStrategy.LRU:
            return next(iter(self.cache))  # Primer elemento (más antiguo)
        
        elif self.config.strategy == CacheStrategy.LFU:
            return min(self.cache.keys(), key=lambda k: self.cache[k].access_count)
        
        elif self.config.strategy == CacheStrategy.FIFO:
            return min(self.cache.keys(), key=lambda k: self.cache[k].timestamp)
        
        elif self.config.strategy == CacheStrategy.ADAPTIVE:
            # Combinar LRU y LFU con prioridades
            candidates = []
            for key, item in self.cache.items():
                score = (
                    (datetime.now() - item.last_access).total_seconds() / 3600 +  # Recencia
                    (1.0 / max(item.access_count, 1)) +  # Frecuencia inversa
                    (1.0 / item.priority.value)  # Prioridad inversa
                )
                candidates.append((key, score))
            
            return max(candidates, key=lambda x: x[1])[0]
        
        else:
            return next(iter(self.cache))
    
    def _evict_item(self, key: str):
        """Evictar elemento del caché"""
        if key in self.cache:
            item = self.cache[key]
            del self.cache[key]
            
            with self.stats_lock:
                self.stats.total_size_bytes -= item.size_bytes
                self.stats.evictions += 1
                self.stats.item_count = len(self.cache)
    
    def _is_expired(self, item: CacheItem) -> bool:
        """Verificar si un elemento ha expirado"""
        if item.ttl_seconds is None:
            return False
        
        age_seconds = (datetime.now() - item.timestamp).total_seconds()
        return age_seconds > item.ttl_seconds
    
    def _decompress_if_needed(self, item: CacheItem) -> Any:
        """Descomprimir elemento si es necesario"""
        if not item.compressed:
            return item.value
        
        try:
            decompressed = self.compression_manager.decompress(item.value, item.compression_type)
            with self.stats_lock:
                self.stats.decompressions += 1
            return decompressed
        except Exception as e:
            logger.error(f"Error descomprimiendo item {item.key}: {e}")
            return None
    
    def _calculate_size(self, value: Any) -> int:
        """Calcular tamaño aproximado de un valor"""
        try:
            if isinstance(value, np.ndarray):
                return value.nbytes
            elif isinstance(value, (str, bytes)):
                return len(value)
            else:
                return len(pickle.dumps(value))
        except Exception:
            return 1024  # Estimación por defecto
    
    def _cleanup_loop(self):
        """Bucle de limpieza periódica"""
        while self.running:
            try:
                self._cleanup_expired_items()
                time.sleep(self.config.cleanup_interval_seconds)
            except Exception as e:
                logger.error(f"Error en limpieza de caché: {e}")
                time.sleep(60)
    
    def _cleanup_expired_items(self):
        """Limpiar elementos expirados"""
        with self.cache_lock:
            expired_keys = []
            for key, item in self.cache.items():
                if self._is_expired(item):
                    expired_keys.append(key)
            
            for key in expired_keys:
                self._evict_item(key)
            
            if expired_keys:
                logger.debug(f"Limpiados {len(expired_keys)} elementos expirados del caché")
    
    def _persistence_loop(self):
        """Bucle de persistencia periódica"""
        while self.running:
            try:
                self._save_persistent_cache()
                time.sleep(self.config.save_interval_seconds)
            except Exception as e:
                logger.error(f"Error guardando caché persistente: {e}")
                time.sleep(300)
    
    def _save_persistent_cache(self):
        """Guardar caché en disco"""
        try:
            cache_data = {
                "items": dict(self.cache),
                "stats": self.stats,
                "timestamp": datetime.now()
            }
            
            with open(self.config.persistence_file, 'wb') as f:
                pickle.dump(cache_data, f)
                
        except Exception as e:
            logger.error(f"Error guardando caché persistente: {e}")
    
    def _load_persistent_cache(self):
        """Cargar caché desde disco"""
        try:
            if os.path.exists(self.config.persistence_file):
                with open(self.config.persistence_file, 'rb') as f:
                    cache_data = pickle.load(f)
                
                self.cache = OrderedDict(cache_data.get("items", {}))
                self.stats = cache_data.get("stats", CacheStats())
                
                logger.info(f"Caché persistente cargado: {len(self.cache)} elementos")
                
        except Exception as e:
            logger.error(f"Error cargando caché persistente: {e}")
    
    def _monitoring_loop(self):
        """Bucle de monitoreo"""
        while self.running:
            try:
                self._update_stats()
                time.sleep(self.config.monitoring_interval_seconds)
            except Exception as e:
                logger.error(f"Error en monitoreo de caché: {e}")
                time.sleep(60)
    
    def _update_stats(self):
        """Actualizar estadísticas"""
        with self.stats_lock:
            self.stats.item_count = len(self.cache)
            self._update_memory_usage()
            self._update_hit_rate()
    
    def _update_hit_rate(self):
        """Actualizar tasa de aciertos"""
        total = self.stats.hits + self.stats.misses
        self.stats.hit_rate = self.stats.hits / total if total > 0 else 0.0
    
    def _update_memory_usage(self):
        """Actualizar uso de memoria"""
        self.stats.memory_usage_mb = self.stats.total_size_bytes / (1024 * 1024)
    
    def get_stats(self) -> CacheStats:
        """Obtener estadísticas del caché"""
        with self.stats_lock:
            return CacheStats(
                hits=self.stats.hits,
                misses=self.stats.misses,
                evictions=self.stats.evictions,
                compressions=self.stats.compressions,
                decompressions=self.stats.decompressions,
                total_size_bytes=self.stats.total_size_bytes,
                item_count=self.stats.item_count,
                memory_usage_mb=self.stats.memory_usage_mb,
                hit_rate=self.stats.hit_rate,
                compression_ratio=self._calculate_compression_ratio()
            )
    
    def _calculate_compression_ratio(self) -> float:
        """Calcular ratio de compresión promedio"""
        total_ratio = 0.0
        count = 0
        
        for comp_type, stats in self.compression_manager.compression_stats.items():
            if stats["count"] > 0:
                total_ratio += stats["total_ratio"]
                count += stats["count"]
        
        return total_ratio / count if count > 0 else 1.0
    
    def clear(self):
        """Limpiar todo el caché"""
        with self.cache_lock:
            self.cache.clear()
            with self.stats_lock:
                self.stats = CacheStats()
    
    def stop(self):
        """Detener sistema de caché"""
        self.running = False
        
        # Guardar caché antes de cerrar
        if self.config.enable_persistence:
            self._save_persistent_cache()
        
        # Esperar a que terminen las tareas
        for task in self.maintenance_tasks:
            if task.is_alive():
                task.join(timeout=5)

# Función de utilidad para crear caché con configuración predeterminada
def create_intelligent_cache(strategy: CacheStrategy = CacheStrategy.ADAPTIVE, 
                           max_memory_mb: int = 512) -> IntelligentCacheSystem:
    """Crear sistema de caché inteligente con configuración predeterminada"""
    config = CacheConfig(
        strategy=strategy,
        max_memory_mb=max_memory_mb,
        enable_prediction=True,
        enable_persistence=True
    )
    return IntelligentCacheSystem(config)

# Función de prueba
def test_intelligent_cache():
    """Probar sistema de caché inteligente"""
    print("🧠 Probando sistema de caché inteligente...")
    
    cache = create_intelligent_cache(CacheStrategy.ADAPTIVE, 128)
    
    # Probar operaciones básicas
    test_data = {
        "small_data": "Hello World",
        "medium_data": list(range(1000)),
        "large_data": np.random.rand(1000, 1000),
        "json_data": {"key": "value", "numbers": list(range(100))}
    }
    
    # Almacenar datos
    for key, value in test_data.items():
        cache.set(key, value, priority=CachePriority.HIGH)
        print(f"✅ Almacenado: {key}")
    
    # Recuperar datos
    for key in test_data.keys():
        retrieved = cache.get(key)
        if retrieved is not None:
            print(f"✅ Recuperado: {key}")
        else:
            print(f"❌ No encontrado: {key}")
    
    # Mostrar estadísticas
    stats = cache.get_stats()
    print(f"\n📊 Estadísticas del caché:")
    print(f"  - Elementos: {stats.item_count}")
    print(f"  - Memoria: {stats.memory_usage_mb:.2f} MB")
    print(f"  - Tasa de aciertos: {stats.hit_rate:.2%}")
    print(f"  - Compresiones: {stats.compressions}")
    print(f"  - Ratio de compresión: {stats.compression_ratio:.2f}")
    
    cache.stop()

if __name__ == "__main__":
    test_intelligent_cache()