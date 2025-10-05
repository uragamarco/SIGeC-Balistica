#!/usr/bin/env python3
"""
Sistema de Lazy Loading Avanzado - SIGeC-Balistica
===========================================

Sistema inteligente de carga perezosa (lazy loading) para imágenes, diseñado para
optimizar el uso de memoria y mejorar el rendimiento en aplicaciones que manejan
grandes conjuntos de imágenes balísticas.

Características principales:
- Carga bajo demanda con predicción inteligente
- Gestión automática de memoria con limpieza proactiva
- Caché multinivel con diferentes estrategias
- Precarga inteligente basada en patrones de uso
- Virtualización de datasets grandes
- Integración con sistemas de archivos y bases de datos
- Monitoreo de rendimiento y uso de memoria

Autor: SIGeC-BalisticaTeam
Versión: 1.0.0
"""

import cv2
import numpy as np
import os
import gc
import psutil
import threading
import time
import weakref
from typing import Dict, List, Tuple, Optional, Any, Union, Callable, Iterator, Set
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
from collections import defaultdict, deque
import logging
import hashlib
from concurrent.futures import ThreadPoolExecutor, Future
import sqlite3
import json

# Importaciones locales
try:
    from utils.logger import LoggerMixin
    LOGGER_MIXIN_AVAILABLE = True
except ImportError:
    LOGGER_MIXIN_AVAILABLE = False
    class LoggerMixin:
        def __init__(self, *args, **kwargs): 
            self.logger = logging.getLogger(self.__class__.__name__)

try:
    from image_processing.optimized_loader import OptimizedImageLoader, LoadingConfig, QualityLevel, LoadingResult
    OPTIMIZED_LOADER_AVAILABLE = True
except ImportError:
    OPTIMIZED_LOADER_AVAILABLE = False


class LoadingPriority(Enum):
    """Prioridades de carga"""
    IMMEDIATE = "immediate"    # Carga inmediata
    HIGH = "high"             # Alta prioridad
    NORMAL = "normal"         # Prioridad normal
    LOW = "low"              # Baja prioridad
    BACKGROUND = "background" # Carga en segundo plano


class PredictionStrategy(Enum):
    """Estrategias de predicción para precarga"""
    SEQUENTIAL = "sequential"      # Predicción secuencial
    PATTERN_BASED = "pattern_based" # Basada en patrones de acceso
    ML_BASED = "ml_based"         # Basada en machine learning
    HYBRID = "hybrid"             # Combinación de estrategias


class MemoryStrategy(Enum):
    """Estrategias de gestión de memoria"""
    LRU = "lru"                   # Least Recently Used
    LFU = "lfu"                   # Least Frequently Used
    ADAPTIVE = "adaptive"         # Adaptativa basada en uso
    PRIORITY_BASED = "priority_based" # Basada en prioridades


@dataclass
class LazyLoadingConfig:
    """Configuración para lazy loading"""
    # Estrategias principales
    prediction_strategy: PredictionStrategy = PredictionStrategy.PATTERN_BASED
    memory_strategy: MemoryStrategy = MemoryStrategy.ADAPTIVE
    
    # Gestión de memoria
    max_memory_mb: float = 1024.0
    memory_threshold: float = 0.8  # Umbral para limpieza automática
    cleanup_percentage: float = 0.3  # Porcentaje a limpiar cuando se alcanza el umbral
    
    # Predicción y precarga
    enable_prediction: bool = True
    prediction_window: int = 5  # Número de imágenes a predecir
    preload_count: int = 3     # Número de imágenes a precargar
    pattern_history_size: int = 100  # Tamaño del historial de patrones
    
    # Caché multinivel
    enable_thumbnail_cache: bool = True
    enable_metadata_cache: bool = True
    enable_persistent_cache: bool = False
    cache_directory: Optional[str] = None
    
    # Rendimiento
    enable_background_loading: bool = True
    max_background_workers: int = 2
    loading_timeout: float = 30.0
    
    # Monitoreo
    enable_monitoring: bool = True
    log_access_patterns: bool = False
    debug_mode: bool = False


@dataclass
class ImageProxy:
    """Proxy para imagen con carga lazy"""
    path: str
    priority: LoadingPriority = LoadingPriority.NORMAL
    quality: QualityLevel = QualityLevel.FULL
    
    # Estado de carga
    is_loaded: bool = False
    is_loading: bool = False
    load_requested: bool = False
    
    # Datos de imagen
    _image: Optional[np.ndarray] = None
    _metadata: Optional[Dict] = None
    _thumbnail: Optional[np.ndarray] = None
    
    # Estadísticas de acceso
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    first_accessed: float = field(default_factory=time.time)
    total_access_time: float = 0.0
    
    # Referencias débiles para limpieza automática
    _weak_refs: Set[weakref.ref] = field(default_factory=set)
    
    def __post_init__(self):
        self.creation_time = time.time()
    
    @property
    def image(self) -> Optional[np.ndarray]:
        """Obtiene la imagen, cargándola si es necesario"""
        self.access_count += 1
        self.last_accessed = time.time()
        
        if not self.is_loaded and not self.is_loading:
            # Marcar como solicitada para carga
            self.load_requested = True
        
        return self._image
    
    @property
    def metadata(self) -> Optional[Dict]:
        """Obtiene metadatos de la imagen"""
        return self._metadata
    
    @property
    def thumbnail(self) -> Optional[np.ndarray]:
        """Obtiene thumbnail de la imagen"""
        return self._thumbnail
    
    def set_image(self, image: np.ndarray, metadata: Optional[Dict] = None):
        """Establece la imagen cargada"""
        self._image = image
        self._metadata = metadata
        self.is_loaded = True
        self.is_loading = False
    
    def set_thumbnail(self, thumbnail: np.ndarray):
        """Establece thumbnail"""
        self._thumbnail = thumbnail
    
    def unload(self):
        """Descarga la imagen de memoria"""
        if self._image is not None:
            del self._image
            self._image = None
        
        self.is_loaded = False
        self.is_loading = False
        gc.collect()
    
    def get_memory_usage(self) -> float:
        """Obtiene uso de memoria en MB"""
        usage = 0.0
        
        if self._image is not None:
            usage += self._image.nbytes / 1024 / 1024
        
        if self._thumbnail is not None:
            usage += self._thumbnail.nbytes / 1024 / 1024
        
        return usage
    
    def get_access_frequency(self) -> float:
        """Calcula frecuencia de acceso"""
        if self.access_count == 0:
            return 0.0
        
        time_span = time.time() - self.first_accessed
        if time_span == 0:
            return float('inf')
        
        return self.access_count / time_span


@dataclass
class AccessPattern:
    """Patrón de acceso a imágenes"""
    sequence: List[str]  # Secuencia de rutas accedidas
    frequency: int = 1
    last_seen: float = field(default_factory=time.time)
    confidence: float = 1.0


class PatternPredictor:
    """Predictor de patrones de acceso"""
    
    def __init__(self, config: LazyLoadingConfig):
        self.config = config
        self.access_history = deque(maxlen=config.pattern_history_size)
        self.patterns = defaultdict(lambda: AccessPattern([]))
        self.lock = threading.RLock()
    
    def record_access(self, image_path: str):
        """Registra acceso a imagen"""
        with self.lock:
            self.access_history.append((image_path, time.time()))
            self._update_patterns()
    
    def predict_next(self, current_path: str, count: int = 3) -> List[str]:
        """Predice próximas imágenes a acceder"""
        with self.lock:
            if self.config.prediction_strategy == PredictionStrategy.SEQUENTIAL:
                return self._predict_sequential(current_path, count)
            elif self.config.prediction_strategy == PredictionStrategy.PATTERN_BASED:
                return self._predict_pattern_based(current_path, count)
            else:
                return self._predict_hybrid(current_path, count)
    
    def _update_patterns(self):
        """Actualiza patrones basado en historial reciente"""
        if len(self.access_history) < 2:
            return
        
        # Extraer secuencias de longitud variable
        for window_size in range(2, min(6, len(self.access_history) + 1)):
            for i in range(len(self.access_history) - window_size + 1):
                sequence = [item[0] for item in list(self.access_history)[i:i + window_size]]
                pattern_key = "->".join(sequence[:-1])
                
                if pattern_key in self.patterns:
                    pattern = self.patterns[pattern_key]
                    if sequence[-1] not in pattern.sequence:
                        pattern.sequence.append(sequence[-1])
                    pattern.frequency += 1
                    pattern.last_seen = time.time()
                else:
                    self.patterns[pattern_key] = AccessPattern(
                        sequence=[sequence[-1]],
                        frequency=1
                    )
    
    def _predict_sequential(self, current_path: str, count: int) -> List[str]:
        """Predicción secuencial simple"""
        # Buscar el archivo actual en el historial
        current_dir = os.path.dirname(current_path)
        current_name = os.path.basename(current_path)
        
        try:
            # Obtener lista de archivos en el directorio
            files = sorted([f for f in os.listdir(current_dir) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tiff', '.bmp'))])
            
            if current_name in files:
                current_idx = files.index(current_name)
                predictions = []
                
                for i in range(1, count + 1):
                    next_idx = current_idx + i
                    if next_idx < len(files):
                        predictions.append(os.path.join(current_dir, files[next_idx]))
                
                return predictions
        except:
            pass
        
        return []
    
    def _predict_pattern_based(self, current_path: str, count: int) -> List[str]:
        """Predicción basada en patrones históricos"""
        predictions = []
        
        # Buscar patrones que terminen con la imagen actual
        for pattern_key, pattern in self.patterns.items():
            if pattern_key.endswith(current_path):
                # Ordenar por frecuencia y recencia
                candidates = [(path, pattern.frequency, pattern.last_seen) 
                            for path in pattern.sequence]
                candidates.sort(key=lambda x: (x[1], x[2]), reverse=True)
                
                for path, freq, last_seen in candidates[:count]:
                    if path not in predictions and path != current_path:
                        predictions.append(path)
        
        return predictions[:count]
    
    def _predict_hybrid(self, current_path: str, count: int) -> List[str]:
        """Predicción híbrida combinando estrategias"""
        pattern_predictions = self._predict_pattern_based(current_path, count // 2 + 1)
        sequential_predictions = self._predict_sequential(current_path, count // 2 + 1)
        
        # Combinar y deduplicar
        combined = []
        for pred in pattern_predictions + sequential_predictions:
            if pred not in combined:
                combined.append(pred)
        
        return combined[:count]


class MemoryManager:
    """Gestor de memoria para lazy loading"""
    
    def __init__(self, config: LazyLoadingConfig):
        self.config = config
        self.max_memory_bytes = int(config.max_memory_mb * 1024 * 1024)
        self.current_usage = 0
        self.proxies = weakref.WeakValueDictionary()
        self.lock = threading.RLock()
        
        # Estadísticas
        self.cleanup_count = 0
        self.total_freed = 0
    
    def register_proxy(self, proxy: ImageProxy):
        """Registra un proxy para gestión de memoria"""
        with self.lock:
            self.proxies[proxy.path] = proxy
    
    def update_usage(self):
        """Actualiza el uso actual de memoria"""
        with self.lock:
            self.current_usage = sum(proxy.get_memory_usage() 
                                   for proxy in self.proxies.values() 
                                   if proxy.is_loaded)
    
    def needs_cleanup(self) -> bool:
        """Verifica si se necesita limpieza de memoria"""
        self.update_usage()
        return self.current_usage > self.max_memory_bytes * self.config.memory_threshold
    
    def cleanup_memory(self) -> int:
        """Limpia memoria según la estrategia configurada"""
        with self.lock:
            if not self.needs_cleanup():
                return 0
            
            target_free = int(self.max_memory_bytes * self.config.cleanup_percentage)
            freed = 0
            
            # Obtener candidatos para limpieza
            candidates = []
            for proxy in self.proxies.values():
                if proxy.is_loaded:
                    score = self._calculate_cleanup_score(proxy)
                    candidates.append((score, proxy))
            
            # Ordenar por score (menor score = mejor candidato para limpieza)
            candidates.sort(key=lambda x: x[0])
            
            # Limpiar hasta alcanzar el objetivo
            for score, proxy in candidates:
                if freed >= target_free:
                    break
                
                memory_usage = proxy.get_memory_usage()
                proxy.unload()
                freed += memory_usage
            
            self.cleanup_count += 1
            self.total_freed += freed
            self.update_usage()
            
            return len([proxy for score, proxy in candidates if not proxy.is_loaded])
    
    def _calculate_cleanup_score(self, proxy: ImageProxy) -> float:
        """Calcula score para limpieza (menor = mejor candidato)"""
        now = time.time()
        
        if self.config.memory_strategy == MemoryStrategy.LRU:
            # Least Recently Used
            return proxy.last_accessed
        
        elif self.config.memory_strategy == MemoryStrategy.LFU:
            # Least Frequently Used
            return proxy.access_count
        
        elif self.config.memory_strategy == MemoryStrategy.PRIORITY_BASED:
            # Basado en prioridad
            priority_weights = {
                LoadingPriority.IMMEDIATE: 1000,
                LoadingPriority.HIGH: 100,
                LoadingPriority.NORMAL: 10,
                LoadingPriority.LOW: 1,
                LoadingPriority.BACKGROUND: 0.1
            }
            return 1.0 / priority_weights.get(proxy.priority, 1)
        
        else:  # ADAPTIVE
            # Combinación adaptativa
            recency_score = now - proxy.last_accessed
            frequency_score = 1.0 / (proxy.access_count + 1)
            memory_score = proxy.get_memory_usage()
            
            return recency_score * 0.4 + frequency_score * 0.3 + memory_score * 0.3
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del gestor de memoria"""
        self.update_usage()
        
        return {
            'current_usage_mb': self.current_usage / 1024 / 1024,
            'max_memory_mb': self.max_memory_bytes / 1024 / 1024,
            'usage_percentage': (self.current_usage / self.max_memory_bytes) * 100,
            'loaded_images': len([p for p in self.proxies.values() if p.is_loaded]),
            'total_proxies': len(self.proxies),
            'cleanup_count': self.cleanup_count,
            'total_freed_mb': self.total_freed / 1024 / 1024
        }


class LazyImageManager(LoggerMixin):
    """
    Gestor principal de lazy loading para imágenes
    """
    
    def __init__(self, config: Optional[LazyLoadingConfig] = None):
        super().__init__()
        self.config = config or LazyLoadingConfig()
        
        # Componentes principales
        self.memory_manager = MemoryManager(self.config)
        self.pattern_predictor = PatternPredictor(self.config)
        self.proxies = {}
        
        # Cargador de imágenes
        if OPTIMIZED_LOADER_AVAILABLE:
            loader_config = LoadingConfig(
                max_cache_size_mb=self.config.max_memory_mb * 0.3,
                enable_progressive=True,
                enable_multithreading=True
            )
            self.image_loader = OptimizedImageLoader(loader_config)
        else:
            self.image_loader = None
        
        # Thread pool para carga en segundo plano
        if self.config.enable_background_loading:
            self.executor = ThreadPoolExecutor(
                max_workers=self.config.max_background_workers,
                thread_name_prefix="LazyLoader"
            )
        else:
            self.executor = None
        
        # Estado interno
        self.loading_futures = {}
        self.lock = threading.RLock()
        
        # Estadísticas
        self.stats = {
            'proxies_created': 0,
            'images_loaded': 0,
            'cache_hits': 0,
            'predictions_made': 0,
            'background_loads': 0
        }
        
        self.log_info(f"LazyImageManager inicializado con {self.config.prediction_strategy.value}")
    
    def create_proxy(self, 
                    image_path: str, 
                    priority: LoadingPriority = LoadingPriority.NORMAL,
                    quality: QualityLevel = QualityLevel.FULL) -> ImageProxy:
        """
        Crea un proxy para imagen con lazy loading
        
        Args:
            image_path: Ruta a la imagen
            priority: Prioridad de carga
            quality: Calidad deseada
        
        Returns:
            ImageProxy configurado
        """
        with self.lock:
            # Verificar si ya existe proxy
            if image_path in self.proxies:
                existing_proxy = self.proxies[image_path]
                # Actualizar prioridad si es mayor
                if priority.value > existing_proxy.priority.value:
                    existing_proxy.priority = priority
                return existing_proxy
            
            # Crear nuevo proxy
            proxy = ImageProxy(
                path=image_path,
                priority=priority,
                quality=quality
            )
            
            self.proxies[image_path] = proxy
            self.memory_manager.register_proxy(proxy)
            self.stats['proxies_created'] += 1
            
            # Cargar thumbnail si está habilitado
            if self.config.enable_thumbnail_cache:
                self._load_thumbnail_async(proxy)
            
            return proxy
    
    def get_image(self, image_path: str, 
                  priority: LoadingPriority = LoadingPriority.NORMAL) -> Optional[np.ndarray]:
        """
        Obtiene imagen, cargándola si es necesario
        
        Args:
            image_path: Ruta a la imagen
            priority: Prioridad de carga
        
        Returns:
            Array numpy con la imagen o None si no se pudo cargar
        """
        # Crear o obtener proxy
        proxy = self.create_proxy(image_path, priority)
        
        # Registrar acceso para predicción
        self.pattern_predictor.record_access(image_path)
        
        # Si la imagen ya está cargada, devolverla
        if proxy.is_loaded and proxy.image is not None:
            self.stats['cache_hits'] += 1
            return proxy.image
        
        # Si no está cargada, cargar ahora
        if not proxy.is_loading:
            self._load_image_sync(proxy)
        
        # Predecir y precargar próximas imágenes
        if self.config.enable_prediction:
            self._predict_and_preload(image_path)
        
        return proxy.image
    
    def get_image_async(self, image_path: str, 
                       priority: LoadingPriority = LoadingPriority.NORMAL) -> Future[Optional[np.ndarray]]:
        """Obtiene imagen de forma asíncrona"""
        if not self.executor:
            raise RuntimeError("Carga asíncrona no habilitada")
        
        return self.executor.submit(self.get_image, image_path, priority)
    
    def preload_images(self, image_paths: List[str], 
                      priority: LoadingPriority = LoadingPriority.BACKGROUND):
        """Precarga lista de imágenes en segundo plano"""
        if not self.executor:
            self.log_warning("Carga en segundo plano no habilitada")
            return
        
        for path in image_paths:
            proxy = self.create_proxy(path, priority)
            if not proxy.is_loaded and not proxy.is_loading:
                future = self.executor.submit(self._load_image_sync, proxy)
                self.loading_futures[path] = future
                self.stats['background_loads'] += 1
    
    def _load_image_sync(self, proxy: ImageProxy) -> bool:
        """Carga imagen de forma síncrona"""
        if proxy.is_loading or proxy.is_loaded:
            return proxy.is_loaded
        
        proxy.is_loading = True
        
        try:
            # Verificar memoria antes de cargar
            if self.memory_manager.needs_cleanup():
                cleaned = self.memory_manager.cleanup_memory()
                self.log_info(f"Limpieza de memoria: {cleaned} imágenes descargadas")
            
            # Cargar imagen
            if self.image_loader:
                result = self.image_loader.load_image(proxy.path, proxy.quality)
                if result.success and result.image is not None:
                    proxy.set_image(result.image, result.metadata.__dict__ if result.metadata else None)
                    self.stats['images_loaded'] += 1
                    return True
            else:
                # Fallback a OpenCV
                image = cv2.imread(proxy.path, cv2.IMREAD_COLOR)
                if image is not None:
                    proxy.set_image(image)
                    self.stats['images_loaded'] += 1
                    return True
            
            proxy.is_loading = False
            return False
            
        except Exception as e:
            self.log_error(f"Error cargando {proxy.path}: {str(e)}")
            proxy.is_loading = False
            return False
    
    def _load_thumbnail_async(self, proxy: ImageProxy):
        """Carga thumbnail de forma asíncrona"""
        if not self.executor or proxy.thumbnail is not None:
            return
        
        def load_thumbnail():
            try:
                if self.image_loader:
                    result = self.image_loader.load_image(proxy.path, QualityLevel.THUMBNAIL)
                    if result.success and result.image is not None:
                        proxy.set_thumbnail(result.image)
                else:
                    # Fallback simple
                    image = cv2.imread(proxy.path, cv2.IMREAD_COLOR)
                    if image is not None:
                        thumbnail = cv2.resize(image, (128, 128), interpolation=cv2.INTER_AREA)
                        proxy.set_thumbnail(thumbnail)
                        del image
            except Exception as e:
                self.log_error(f"Error cargando thumbnail {proxy.path}: {str(e)}")
        
        self.executor.submit(load_thumbnail)
    
    def _predict_and_preload(self, current_path: str):
        """Predice y precarga próximas imágenes"""
        if not self.config.enable_prediction:
            return
        
        try:
            predictions = self.pattern_predictor.predict_next(
                current_path, 
                self.config.preload_count
            )
            
            if predictions:
                self.stats['predictions_made'] += 1
                self.preload_images(predictions, LoadingPriority.BACKGROUND)
                
                if self.config.log_access_patterns:
                    self.log_info(f"Predicciones para {current_path}: {predictions}")
        
        except Exception as e:
            self.log_error(f"Error en predicción: {str(e)}")
    
    def get_proxy_info(self, image_path: str) -> Optional[Dict[str, Any]]:
        """Obtiene información detallada de un proxy"""
        if image_path not in self.proxies:
            return None
        
        proxy = self.proxies[image_path]
        
        return {
            'path': proxy.path,
            'is_loaded': proxy.is_loaded,
            'is_loading': proxy.is_loading,
            'priority': proxy.priority.value,
            'quality': proxy.quality.value,
            'access_count': proxy.access_count,
            'last_accessed': proxy.last_accessed,
            'memory_usage_mb': proxy.get_memory_usage(),
            'access_frequency': proxy.get_access_frequency(),
            'has_thumbnail': proxy.thumbnail is not None
        }
    
    def cleanup_unused_proxies(self, max_age_hours: float = 24.0):
        """Limpia proxies no utilizados recientemente"""
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        to_remove = []
        
        with self.lock:
            for path, proxy in self.proxies.items():
                age = current_time - proxy.last_accessed
                if age > max_age_seconds and not proxy.is_loading:
                    to_remove.append(path)
            
            for path in to_remove:
                proxy = self.proxies.pop(path)
                proxy.unload()
        
        if to_remove:
            self.log_info(f"Limpiados {len(to_remove)} proxies no utilizados")
        
        return len(to_remove)
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas completas del sistema"""
        stats = self.stats.copy()
        
        # Estadísticas de memoria
        stats['memory'] = self.memory_manager.get_stats()
        
        # Estadísticas de proxies
        loaded_proxies = sum(1 for p in self.proxies.values() if p.is_loaded)
        loading_proxies = sum(1 for p in self.proxies.values() if p.is_loading)
        
        stats['proxies'] = {
            'total': len(self.proxies),
            'loaded': loaded_proxies,
            'loading': loading_proxies,
            'unloaded': len(self.proxies) - loaded_proxies - loading_proxies
        }
        
        # Estadísticas del cargador de imágenes
        if self.image_loader:
            stats['image_loader'] = self.image_loader.get_stats()
        
        return stats
    
    def __del__(self):
        """Limpieza al destruir el objeto"""
        if hasattr(self, 'executor') and self.executor:
            self.executor.shutdown(wait=True)


# Funciones de utilidad
def create_lazy_config(max_memory_mb: float = 512, 
                      enable_prediction: bool = True,
                      enable_background: bool = True) -> LazyLoadingConfig:
    """Crea configuración optimizada para lazy loading"""
    return LazyLoadingConfig(
        max_memory_mb=max_memory_mb,
        memory_threshold=0.8,
        enable_prediction=enable_prediction,
        prediction_strategy=PredictionStrategy.HYBRID,
        memory_strategy=MemoryStrategy.ADAPTIVE,
        enable_background_loading=enable_background,
        enable_thumbnail_cache=True,
        enable_monitoring=True
    )


class LazyImageDataset:
    """Dataset lazy para grandes colecciones de imágenes"""
    
    def __init__(self, image_paths: List[str], manager: LazyImageManager):
        self.image_paths = image_paths
        self.manager = manager
        self.current_index = 0
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, index: int) -> Optional[np.ndarray]:
        if 0 <= index < len(self.image_paths):
            self.current_index = index
            return self.manager.get_image(self.image_paths[index])
        return None
    
    def __iter__(self):
        self.current_index = 0
        return self
    
    def __next__(self) -> np.ndarray:
        if self.current_index >= len(self.image_paths):
            raise StopIteration
        
        image = self[self.current_index]
        self.current_index += 1
        
        if image is None:
            return self.__next__()  # Saltar imágenes que no se pudieron cargar
        
        return image


if __name__ == "__main__":
    # Ejemplo de uso
    logging.basicConfig(level=logging.INFO)
    
    # Crear configuración
    config = create_lazy_config(max_memory_mb=256, enable_prediction=True)
    
    # Crear manager
    manager = LazyImageManager(config)
    
    # Ejemplo de uso
    # proxy = manager.create_proxy("path/to/image.jpg", LoadingPriority.HIGH)
    # image = manager.get_image("path/to/image.jpg")
    
    print("LazyImageManager listo para usar")
    print(f"Configuración: {config}")
    print(f"Estadísticas: {manager.get_stats()}")