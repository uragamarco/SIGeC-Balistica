#!/usr/bin/env python3
"""
Cargador Optimizado de Imágenes - SIGeC-Balistica
==========================================

Sistema avanzado de carga de imágenes optimizado para imágenes grandes y gestión
eficiente de memoria. Incluye múltiples estrategias de carga, caché inteligente,
y optimizaciones específicas para imágenes balísticas de alta resolución.

Características principales:
- Múltiples estrategias de carga (lazy, progressive, streaming)
- Caché inteligente con LRU y gestión automática de memoria
- Carga progresiva con diferentes niveles de calidad
- Streaming de imágenes para datasets grandes
- Optimizaciones específicas para formatos de imagen
- Monitoreo de rendimiento y memoria
- Compatibilidad con múltiples formatos (JPEG, PNG, TIFF, RAW)

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
from typing import Dict, List, Tuple, Optional, Any, Union, Callable, Iterator, Generator
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
from collections import OrderedDict
import logging
import hashlib
import pickle
import json
import gzip
import lzma
from concurrent.futures import ThreadPoolExecutor, Future
import warnings

# Importaciones opcionales
try:
    from PIL import Image, ImageFile
    PIL_AVAILABLE = True
    # Permitir carga de imágenes truncadas
    ImageFile.LOAD_TRUNCATED_IMAGES = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import rawpy
    RAWPY_AVAILABLE = True
except ImportError:
    RAWPY_AVAILABLE = False

try:
    from skimage import io as skio
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False

# Importaciones locales
try:
    from utils.logger import LoggerMixin
    LOGGER_MIXIN_AVAILABLE = True
except ImportError:
    LOGGER_MIXIN_AVAILABLE = False
    class LoggerMixin:
        def __init__(self, *args, **kwargs): 
            self.logger = logging.getLogger(self.__class__.__name__)


class LoadingStrategy(Enum):
    """Estrategias de carga de imágenes"""
    IMMEDIATE = "immediate"          # Carga inmediata completa
    LAZY = "lazy"                   # Carga bajo demanda
    PROGRESSIVE = "progressive"      # Carga progresiva por niveles
    STREAMING = "streaming"         # Streaming para datasets grandes
    MEMORY_MAPPED = "memory_mapped" # Mapeo en memoria
    CHUNKED = "chunked"            # Carga por chunks


class ImageFormat(Enum):
    """Formatos de imagen soportados"""
    JPEG = "jpeg"
    PNG = "png"
    TIFF = "tiff"
    BMP = "bmp"
    RAW = "raw"
    UNKNOWN = "unknown"


class QualityLevel(Enum):
    """Niveles de calidad para carga progresiva"""
    THUMBNAIL = "thumbnail"    # 128x128 máximo
    LOW = "low"               # 512x512 máximo
    MEDIUM = "medium"         # 1024x1024 máximo
    HIGH = "high"            # 2048x2048 máximo
    FULL = "full"            # Resolución completa


@dataclass
class LoadingConfig:
    """Configuración para el cargador optimizado de imágenes"""
    
    # Estrategia de carga
    strategy: LoadingStrategy = LoadingStrategy.LAZY
    
    # Configuración de memoria y caché
    max_cache_size_mb: float = 512.0  # Tamaño máximo de caché en MB
    max_memory_usage: float = 0.6     # Porcentaje máximo de RAM
    enable_memory_mapping: bool = True
    
    # Configuración progresiva
    enable_progressive: bool = True
    initial_quality: QualityLevel = QualityLevel.LOW
    progressive_levels: List[QualityLevel] = field(default_factory=lambda: [
        QualityLevel.THUMBNAIL, QualityLevel.LOW, QualityLevel.MEDIUM, QualityLevel.FULL
    ])
    
    # Configuración de threading
    enable_multithreading: bool = True
    max_workers: Optional[int] = None
    preload_next: bool = True
    enable_compression: bool = True
    
    # Configuración de caché
    enable_cache: bool = True
    cache_thumbnails: bool = True
    cache_metadata: bool = True
    persistent_cache: bool = False
    cache_directory: Optional[str] = None
    
    # Configuración de formatos
    jpeg_quality: int = 95
    png_compression: int = 6
    tiff_compression: str = "lzw"
    
    # Configuración RAW avanzada
    raw_auto_brightness: bool = True
    raw_auto_white_balance: bool = True
    raw_demosaic_algorithm: str = "AHD"  # AHD, VNG, PPG, AAHD
    raw_output_color: int = 1  # 0=raw, 1=sRGB, 2=Adobe RGB, 3=Wide Gamut RGB, 4=ProPhoto RGB
    raw_gamma: Tuple[float, float] = (2.222, 4.5)  # Gamma curve
    raw_brightness: float = 1.0
    raw_highlight_mode: int = 0  # 0=clip, 1=unclip, 2=blend, 3+=rebuild
    raw_noise_threshold: float = 100.0
    raw_use_camera_wb: bool = True
    raw_use_auto_wb: bool = False
    raw_user_wb: Optional[Tuple[float, float, float, float]] = None  # R, G, B, G multipliers
    
    # Configuración de caché persistente
    persistent_cache_config: Optional[PersistentCacheConfig] = None
    
    # Configuración de monitoreo
    enable_monitoring: bool = True
    log_loading_times: bool = False
    debug_mode: bool = False


@dataclass
class ImageMetadata:
    """Metadatos de imagen"""
    path: str
    format: ImageFormat
    size: Tuple[int, int]  # (width, height)
    channels: int
    dtype: str
    file_size: int
    last_modified: float
    hash: Optional[str] = None
    
    # Metadatos específicos
    dpi: Optional[Tuple[float, float]] = None
    color_space: Optional[str] = None
    compression: Optional[str] = None
    
    # Estadísticas de carga
    load_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    average_load_time: float = 0.0


@dataclass
class LoadingResult:
    """Resultado de carga de imagen"""
    success: bool
    image: Optional[np.ndarray] = None
    metadata: Optional[ImageMetadata] = None
    quality_level: QualityLevel = QualityLevel.FULL
    loading_time: float = 0.0
    memory_usage: float = 0.0
    cache_hit: bool = False
    error_message: str = ""


class LRUCache:
    """Caché LRU optimizado para imágenes"""
    
    def __init__(self, max_size_mb: float):
        self.max_size_bytes = int(max_size_mb * 1024 * 1024)
        self.cache = OrderedDict()
        self.current_size = 0
        self.lock = threading.RLock()
        
        # Estadísticas
        self.hits = 0
        self.misses = 0
        self.evictions = 0
    
    def get(self, key: str) -> Optional[np.ndarray]:
        """Obtiene imagen del caché"""
        with self.lock:
            if key in self.cache:
                # Mover al final (más reciente)
                image = self.cache.pop(key)
                self.cache[key] = image
                self.hits += 1
                return image.copy()  # Retornar copia para evitar modificaciones
            else:
                self.misses += 1
                return None
    
    def put(self, key: str, image: np.ndarray):
        """Almacena imagen en caché"""
        with self.lock:
            image_size = image.nbytes
            
            # Si la imagen es muy grande, no la almacenamos
            if image_size > self.max_size_bytes * 0.5:
                return
            
            # Remover entrada existente si existe
            if key in self.cache:
                old_image = self.cache.pop(key)
                self.current_size -= old_image.nbytes
            
            # Liberar espacio si es necesario
            while self.current_size + image_size > self.max_size_bytes and self.cache:
                oldest_key, oldest_image = self.cache.popitem(last=False)
                self.current_size -= oldest_image.nbytes
                self.evictions += 1
            
            # Almacenar nueva imagen
            self.cache[key] = image.copy()
            self.current_size += image_size
    
    def clear(self):
        """Limpia el caché"""
        with self.lock:
            self.cache.clear()
            self.current_size = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del caché"""
        with self.lock:
            total_requests = self.hits + self.misses
            hit_rate = self.hits / total_requests if total_requests > 0 else 0
            
            return {
                'size_mb': self.current_size / 1024 / 1024,
                'max_size_mb': self.max_size_bytes / 1024 / 1024,
                'items': len(self.cache),
                'hits': self.hits,
                'misses': self.misses,
                'evictions': self.evictions,
                'hit_rate': hit_rate
            }


class ImageFormatDetector:
    """Detector de formato de imagen"""
    
    @staticmethod
    def detect_format(file_path: str) -> ImageFormat:
        """Detecta el formato de imagen por extensión y magic bytes"""
        path = Path(file_path)
        extension = path.suffix.lower()
        
        # Detección por extensión
        format_map = {
            '.jpg': ImageFormat.JPEG,
            '.jpeg': ImageFormat.JPEG,
            '.png': ImageFormat.PNG,
            '.tiff': ImageFormat.TIFF,
            '.tif': ImageFormat.TIFF,
            '.bmp': ImageFormat.BMP,
            '.raw': ImageFormat.RAW,
            '.cr2': ImageFormat.RAW,
            '.nef': ImageFormat.RAW,
            '.arw': ImageFormat.RAW,
        }
        
        if extension in format_map:
            return format_map[extension]
        
        # Detección por magic bytes
        try:
            with open(file_path, 'rb') as f:
                header = f.read(16)
                
                if header.startswith(b'\xff\xd8\xff'):
                    return ImageFormat.JPEG
                elif header.startswith(b'\x89PNG\r\n\x1a\n'):
                    return ImageFormat.PNG
                elif header.startswith(b'II*\x00') or header.startswith(b'MM\x00*'):
                    return ImageFormat.TIFF
                elif header.startswith(b'BM'):
                    return ImageFormat.BMP
        except:
            pass
        
        return ImageFormat.UNKNOWN


class ProgressiveLoader:
    """Cargador progresivo de imágenes"""
    
    def __init__(self, config: LoadingConfig):
        self.config = config
    
    def load_progressive(self, file_path: str, target_quality: QualityLevel) -> LoadingResult:
        """Carga imagen con calidad progresiva"""
        try:
            # Obtener dimensiones originales
            original_size = self._get_image_size(file_path)
            if original_size is None:
                return LoadingResult(success=False, error_message="No se pudieron obtener dimensiones")
            
            # Calcular tamaño objetivo
            target_size = self._calculate_target_size(original_size, target_quality)
            
            # Cargar imagen
            start_time = time.time()
            
            if target_quality == QualityLevel.FULL:
                # Carga completa
                image = cv2.imread(file_path, cv2.IMREAD_COLOR)
            else:
                # Carga redimensionada
                image = self._load_resized(file_path, target_size)
            
            loading_time = time.time() - start_time
            
            if image is None:
                return LoadingResult(success=False, error_message="Error cargando imagen")
            
            # Crear metadatos
            metadata = self._create_metadata(file_path, image)
            
            return LoadingResult(
                success=True,
                image=image,
                metadata=metadata,
                quality_level=target_quality,
                loading_time=loading_time,
                memory_usage=image.nbytes / 1024 / 1024
            )
            
        except Exception as e:
            return LoadingResult(success=False, error_message=str(e))
    
    def _get_image_size(self, file_path: str) -> Optional[Tuple[int, int]]:
        """Obtiene dimensiones de imagen sin cargarla completamente"""
        try:
            if PIL_AVAILABLE:
                with Image.open(file_path) as img:
                    return img.size  # (width, height)
            else:
                # Usar OpenCV para obtener dimensiones
                img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
                if img is not None:
                    h, w = img.shape[:2]
                    del img
                    return (w, h)
        except:
            pass
        return None
    
    def _calculate_target_size(self, original_size: Tuple[int, int], quality: QualityLevel) -> Tuple[int, int]:
        """Calcula tamaño objetivo basado en nivel de calidad"""
        width, height = original_size
        
        max_sizes = {
            QualityLevel.THUMBNAIL: 128,
            QualityLevel.LOW: 512,
            QualityLevel.MEDIUM: 1024,
            QualityLevel.HIGH: 2048,
            QualityLevel.FULL: max(width, height)
        }
        
        max_size = max_sizes[quality]
        
        if max(width, height) <= max_size:
            return original_size
        
        # Mantener aspect ratio
        if width > height:
            new_width = max_size
            new_height = int(height * max_size / width)
        else:
            new_height = max_size
            new_width = int(width * max_size / height)
        
        return (new_width, new_height)
    
    def _load_resized(self, file_path: str, target_size: Tuple[int, int]) -> Optional[np.ndarray]:
        """Carga imagen redimensionada eficientemente"""
        try:
            if PIL_AVAILABLE:
                # Usar PIL para carga eficiente con thumbnail
                with Image.open(file_path) as img:
                    img.thumbnail(target_size, Image.Resampling.LANCZOS)
                    # Convertir a array numpy
                    img_array = np.array(img)
                    # Convertir RGB a BGR para OpenCV
                    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                    return img_array
            else:
                # Usar OpenCV
                img = cv2.imread(file_path, cv2.IMREAD_COLOR)
                if img is not None:
                    return cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
        except:
            pass
        return None
    
    def _create_metadata(self, file_path: str, image: np.ndarray) -> ImageMetadata:
        """Crea metadatos de imagen"""
        path_obj = Path(file_path)
        stat = path_obj.stat()
        
        return ImageMetadata(
            path=file_path,
            format=ImageFormatDetector.detect_format(file_path),
            size=(image.shape[1], image.shape[0]),  # (width, height)
            channels=image.shape[2] if len(image.shape) == 3 else 1,
            dtype=str(image.dtype),
            file_size=stat.st_size,
            last_modified=stat.st_mtime
        )


class OptimizedImageLoader(LoggerMixin):
    """
    Cargador optimizado de imágenes con múltiples estrategias de carga,
    caché inteligente y soporte para caché persistente con compresión.
    """
    
    def __init__(self, config: Optional[LoadingConfig] = None):
        super().__init__()
        self.config = config or LoadingConfig()
        
        # Inicializar caché en memoria
        self.cache = LRUCache(self.config.max_cache_size_mb) if self.config.enable_cache else None
        
        # Inicializar caché persistente si está habilitado
        self.persistent_cache = None
        if self.config.persistent_cache and self.config.persistent_cache_config:
            try:
                self.persistent_cache = PersistentImageCache(self.config.persistent_cache_config)
                self.log_info("Caché persistente habilitado")
            except Exception as e:
                self.log_error(f"Error inicializando caché persistente: {e}")
        
        # Inicializar cargador progresivo
        self.progressive_loader = ProgressiveLoader(self.config)
        
        # Pool de threads para carga asíncrona
        self.executor = ThreadPoolExecutor(
            max_workers=self.config.max_workers or min(4, os.cpu_count() or 1)
        ) if self.config.enable_multithreading else None
        
        # Estadísticas
        self.stats = {
            'images_loaded': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'persistent_cache_hits': 0,
            'persistent_cache_misses': 0,
            'total_loading_time': 0.0,
            'average_loading_time': 0.0,
            'memory_usage_mb': 0.0,
            'compression_ratio': 0.0
        }
    
    def load_image(self, 
                   file_path: str, 
                   quality: QualityLevel = QualityLevel.FULL,
                   force_reload: bool = False) -> LoadingResult:
        """
        Carga una imagen con la estrategia y calidad especificadas.
        Utiliza caché en memoria y persistente para optimizar rendimiento.
        """
        start_time = time.time()
        
        try:
            # Generar clave de caché
            cache_key = self._generate_cache_key(file_path, quality)
            
            # Intentar cargar desde caché en memoria primero
            if not force_reload and self.cache:
                cached_image = self.cache.get(cache_key)
                if cached_image is not None:
                    self.stats['cache_hits'] += 1
                    metadata = self.get_image_info(file_path)
                    return LoadingResult(
                        success=True,
                        image=cached_image,
                        metadata=metadata,
                        quality_level=quality,
                        loading_time=time.time() - start_time,
                        cache_hit=True
                    )
            
            # Intentar cargar desde caché persistente
            if not force_reload and self.persistent_cache:
                cached_result = self.persistent_cache.get(cache_key)
                if cached_result is not None:
                    image, metadata = cached_result
                    self.stats['persistent_cache_hits'] += 1
                    
                    # Almacenar en caché en memoria para acceso rápido
                    if self.cache:
                        self.cache.put(cache_key, image)
                    
                    return LoadingResult(
                        success=True,
                        image=image,
                        metadata=metadata,
                        quality_level=quality,
                        loading_time=time.time() - start_time,
                        cache_hit=True
                    )
                else:
                    self.stats['persistent_cache_misses'] += 1
            
            # Cargar imagen desde disco
            if self.cache:
                self.stats['cache_misses'] += 1
            
            result = self._load_standard(file_path, quality)
            
            if result.success and result.image is not None:
                # Almacenar en caché en memoria
                if self.cache:
                    self.cache.put(cache_key, result.image)
                
                # Almacenar en caché persistente
                if self.persistent_cache and result.metadata:
                    self.persistent_cache.put(
                        cache_key, result.image, result.metadata, file_path, quality
                    )
                
                # Actualizar estadísticas
                self.stats['images_loaded'] += 1
                loading_time = time.time() - start_time
                self.stats['total_loading_time'] += loading_time
                self.stats['average_loading_time'] = (
                    self.stats['total_loading_time'] / self.stats['images_loaded']
                )
                
                result.loading_time = loading_time
            
            return result
            
        except Exception as e:
            self.log_error(f"Error cargando imagen {file_path}: {e}")
            return LoadingResult(
                success=False,
                error_message=str(e),
                loading_time=time.time() - start_time
            )


    def load_image_async(self, 
                        file_path: str, 
                        quality: QualityLevel = QualityLevel.FULL) -> Future[LoadingResult]:
        """Carga imagen de forma asíncrona"""
        if not self.executor:
            raise RuntimeError("Multithreading no habilitado")
        
        return self.executor.submit(self.load_image, file_path, quality)
    
    def load_batch(self, 
                   file_paths: List[str], 
                   quality: QualityLevel = QualityLevel.FULL,
                   max_concurrent: int = 4) -> Iterator[Tuple[str, LoadingResult]]:
        """
        Carga un lote de imágenes de forma eficiente
        
        Args:
            file_paths: Lista de rutas de imágenes
            quality: Nivel de calidad
            max_concurrent: Máximo número de cargas concurrentes
        
        Yields:
            Tuplas (file_path, LoadingResult)
        """
        if not self.executor:
            # Carga secuencial
            for file_path in file_paths:
                result = self.load_image(file_path, quality)
                yield (file_path, result)
        else:
            # Carga concurrente
            futures = {}
            
            # Enviar trabajos en lotes
            for i in range(0, len(file_paths), max_concurrent):
                batch = file_paths[i:i + max_concurrent]
                
                # Enviar lote actual
                for file_path in batch:
                    future = self.load_image_async(file_path, quality)
                    futures[future] = file_path
                
                # Recoger resultados del lote
                for future in futures:
                    file_path = futures[future]
                    try:
                        result = future.result()
                        yield (file_path, result)
                    except Exception as e:
                        yield (file_path, LoadingResult(success=False, error_message=str(e)))
                
                futures.clear()
    
    def preload_images(self, file_paths: List[str], quality: QualityLevel = QualityLevel.LOW):
        """Precarga imágenes en caché"""
        if not self.cache:
            return
        
        self.log_info(f"Precargando {len(file_paths)} imágenes")
        
        for file_path, result in self.load_batch(file_paths, quality):
            if not result.success:
                self.log_warning(f"Error precargando {file_path}: {result.error_message}")
    
    def _load_standard(self, file_path: str, quality: QualityLevel) -> LoadingResult:
        """Carga estándar de imagen"""
        start_time = time.time()
        
        try:
            # Detectar formato usando método mejorado
            image_format = ImageFormatDetector.detect_format(file_path)
            
            # Usar detección RAW mejorada si el formato no está claro
            if image_format == ImageFormat.UNKNOWN and self._detect_raw_format(file_path):
                image_format = ImageFormat.RAW
            
            # Cargar según formato con prioridad RAW
            if image_format == ImageFormat.RAW or self._detect_raw_format(file_path):
                if RAWPY_AVAILABLE:
                    image = self._load_raw(file_path)
                    if image is not None:
                        self.log_info(f"Imagen RAW cargada exitosamente: {file_path}")
                    else:
                        self.log_warning(f"Fallo carga RAW, intentando PIL: {file_path}")
                        image = self._load_with_pil(file_path) if PIL_AVAILABLE else None
                else:
                    self.log_warning(f"rawpy no disponible para archivo RAW: {file_path}")
                    image = self._load_with_pil(file_path) if PIL_AVAILABLE else None
            elif PIL_AVAILABLE and image_format in [ImageFormat.JPEG, ImageFormat.PNG, ImageFormat.TIFF]:
                image = self._load_with_pil(file_path)
            else:
                # Fallback a OpenCV
                image = cv2.imread(file_path, cv2.IMREAD_COLOR)
            
            if image is None:
                return LoadingResult(success=False, error_message="No se pudo cargar la imagen")
            
            # Redimensionar si es necesario
            if quality != QualityLevel.FULL:
                original_size = (image.shape[1], image.shape[0])
                target_size = self.progressive_loader._calculate_target_size(original_size, quality)
                if target_size != original_size:
                    image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
            
            # Crear metadatos
            metadata = self.progressive_loader._create_metadata(file_path, image)
            metadata.format = image_format  # Actualizar formato detectado
            
            return LoadingResult(
                success=True,
                image=image,
                metadata=metadata,
                quality_level=quality,
                loading_time=time.time() - start_time,
                memory_usage=image.nbytes / 1024 / 1024
            )
            
        except Exception as e:
            return LoadingResult(success=False, error_message=str(e))
    
    def _load_raw(self, file_path: str) -> Optional[np.ndarray]:
        """Carga imagen RAW con configuraciones avanzadas"""
        if not RAWPY_AVAILABLE:
            self.log_warning("rawpy no disponible, no se puede cargar imagen RAW")
            return None
            
        try:
            with rawpy.imread(file_path) as raw:
                # Configurar parámetros de procesamiento
                params = rawpy.Params()
                
                # Configuraciones básicas
                params.use_camera_wb = self.config.raw_use_camera_wb
                params.use_auto_wb = self.config.raw_use_auto_wb
                
                # Balance de blancos personalizado
                if self.config.raw_user_wb:
                    params.user_wb = self.config.raw_user_wb
                
                # Algoritmo de demosaicing
                demosaic_map = {
                    "AHD": rawpy.DemosaicAlgorithm.AHD,
                    "VNG": rawpy.DemosaicAlgorithm.VNG,
                    "PPG": rawpy.DemosaicAlgorithm.PPG,
                    "AAHD": rawpy.DemosaicAlgorithm.AAHD
                }
                if self.config.raw_demosaic_algorithm in demosaic_map:
                    params.demosaic_algorithm = demosaic_map[self.config.raw_demosaic_algorithm]
                
                # Espacio de color de salida
                params.output_color = self.config.raw_output_color
                
                # Curva gamma
                params.gamma = self.config.raw_gamma
                
                # Brillo
                params.bright = self.config.raw_brightness
                
                # Modo de highlights
                params.highlight_mode = self.config.raw_highlight_mode
                
                # Reducción de ruido
                params.noise_thr = self.config.raw_noise_threshold
                
                # Auto-ajustes
                if self.config.raw_auto_brightness:
                    params.auto_bright_thr = 0.01
                
                # Procesar imagen RAW
                rgb = raw.postprocess(params=params)
                
                # Convertir RGB a BGR para OpenCV
                bgr_image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                
                # Log información de procesamiento si está habilitado
                if self.config.debug_mode:
                    self.log_info(f"RAW procesado: {file_path}")
                    self.log_info(f"  - Tamaño: {bgr_image.shape}")
                    self.log_info(f"  - Demosaic: {self.config.raw_demosaic_algorithm}")
                    self.log_info(f"  - Espacio color: {self.config.raw_output_color}")
                    self.log_info(f"  - Balance blancos: camera={self.config.raw_use_camera_wb}, auto={self.config.raw_use_auto_wb}")
                
                return bgr_image
                
        except Exception as e:
            self.log_error(f"Error cargando imagen RAW {file_path}: {e}")
            return None
    
    def get_raw_info(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Obtiene información detallada de archivo RAW sin procesarlo"""
        if not RAWPY_AVAILABLE:
            return None
            
        try:
            with rawpy.imread(file_path) as raw:
                info = {
                    'camera_make': getattr(raw, 'camera_make', 'Unknown'),
                    'camera_model': getattr(raw, 'camera_model', 'Unknown'),
                    'raw_size': getattr(raw, 'raw_image_visible', raw.raw_image).shape,
                    'color_desc': getattr(raw, 'color_desc', 'Unknown'),
                    'num_colors': getattr(raw, 'num_colors', 0),
                    'daylight_whitebalance': getattr(raw, 'daylight_whitebalance', None),
                    'camera_whitebalance': getattr(raw, 'camera_whitebalance', None),
                    'black_level_per_channel': getattr(raw, 'black_level_per_channel', None),
                    'white_level': getattr(raw, 'white_level', None),
                    'color_matrix': getattr(raw, 'color_matrix', None),
                    'rgb_xyz_matrix': getattr(raw, 'rgb_xyz_matrix', None),
                    'tone_curve': getattr(raw, 'tone_curve', None)
                }
                return info
        except Exception as e:
            self.log_error(f"Error obteniendo info RAW {file_path}: {e}")
            return None
    
    def _detect_raw_format(self, file_path: str) -> bool:
        """Detecta si un archivo es formato RAW soportado"""
        raw_extensions = {
            '.cr2', '.cr3',  # Canon
            '.nef', '.nrw',  # Nikon
            '.arw', '.srf', '.sr2',  # Sony
            '.orf',  # Olympus
            '.pef', '.ptx',  # Pentax
            '.raf',  # Fujifilm
            '.rw2',  # Panasonic
            '.dng',  # Adobe DNG
            '.3fr',  # Hasselblad
            '.fff',  # Imacon
            '.iiq',  # Phase One
            '.k25', '.kdc',  # Kodak
            '.mef',  # Mamiya
            '.mos',  # Leaf
            '.mrw',  # Minolta
            '.raw', '.rwl',  # Leica
            '.x3f'   # Sigma
        }
        
        return Path(file_path).suffix.lower() in raw_extensions
    
    def _load_with_pil(self, file_path: str) -> Optional[np.ndarray]:
        """Carga imagen con PIL"""
        try:
            with Image.open(file_path) as img:
                # Convertir a RGB si es necesario
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Convertir a array numpy
                img_array = np.array(img)
                
                # Convertir RGB a BGR para OpenCV
                return cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        except:
            return None
    
    def _generate_cache_key(self, file_path: str, quality: QualityLevel) -> str:
        """Genera clave única para caché"""
        # Incluir ruta, calidad y timestamp de modificación
        try:
            mtime = os.path.getmtime(file_path)
            key_data = f"{file_path}_{quality.value}_{mtime}"
            return hashlib.md5(key_data.encode()).hexdigest()
        except:
            # Fallback sin timestamp
            key_data = f"{file_path}_{quality.value}"
            return hashlib.md5(key_data.encode()).hexdigest()
    
    def get_image_info(self, file_path: str) -> Optional[ImageMetadata]:
        """Obtiene información de imagen sin cargarla"""
        try:
            # Verificar caché de metadatos
            if file_path in self.metadata_cache:
                return self.metadata_cache[file_path]
            
            # Obtener información básica
            path_obj = Path(file_path)
            if not path_obj.exists():
                return None
            
            stat = path_obj.stat()
            image_format = ImageFormatDetector.detect_format(file_path)
            
            # Obtener dimensiones
            size = self.progressive_loader._get_image_size(file_path)
            if size is None:
                return None
            
            # Crear metadatos básicos
            metadata = ImageMetadata(
                path=file_path,
                format=image_format,
                size=size,
                channels=3,  # Asumimos RGB por defecto
                dtype="uint8",
                file_size=stat.st_size,
                last_modified=stat.st_mtime
            )
            
            # Almacenar en caché
            self.metadata_cache[file_path] = metadata
            
            return metadata
            
        except Exception as e:
            self.log_error(f"Error obteniendo info de {file_path}: {str(e)}")
            return None
    
    def clear_cache(self):
        """Limpia todos los cachés (memoria y persistente)"""
        if self.cache:
            self.cache.clear()
            self.log_info("Caché en memoria limpiado")
        
        if self.persistent_cache:
            self.persistent_cache.clear_cache()
            self.log_info("Caché persistente limpiado")
        
        # Reiniciar estadísticas de caché
        self.stats['cache_hits'] = 0
        self.stats['cache_misses'] = 0
        self.stats['persistent_cache_hits'] = 0
        self.stats['persistent_cache_misses'] = 0


class PersistentCacheConfig:
    """Configuración para caché persistente"""
    enabled: bool = True
    cache_directory: str = "cache/images"
    max_disk_size_gb: float = 5.0  # Tamaño máximo en disco
    compression_algorithm: str = "lzma"  # lzma, gzip, none
    compression_level: int = 6  # Nivel de compresión (1-9)
    cache_metadata: bool = True
    cache_thumbnails: bool = True
    cache_full_images: bool = False  # Solo para imágenes pequeñas
    max_file_size_mb: float = 50.0  # Máximo tamaño de archivo para cachear
    cleanup_interval_hours: int = 24  # Intervalo de limpieza automática
    max_age_days: int = 30  # Edad máxima de archivos de caché
    enable_integrity_check: bool = True  # Verificación de integridad
    use_memory_cache: bool = True  # Usar caché en memoria además del persistente

class CacheEntry:
    """Entrada de caché con metadatos"""
    key: str
    file_path: str
    original_path: str
    quality_level: QualityLevel
    created_at: float
    last_accessed: float
    access_count: int
    file_size: int
    compressed: bool
    compression_algorithm: str
    checksum: str
    metadata: Optional[Dict[str, Any]] = None

class PersistentImageCache:
    """Sistema de caché persistente para imágenes con compresión avanzada"""
    
    def __init__(self, config: PersistentCacheConfig):
        self.config = config
        self.cache_dir = Path(config.cache_directory)
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        self.lock = threading.RLock()
        
        # Crear directorio de caché
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Cargar metadatos existentes
        self.entries: Dict[str, CacheEntry] = {}
        self._load_metadata()
        
        # Estadísticas
        self.stats = {
            'hits': 0,
            'misses': 0,
            'disk_reads': 0,
            'disk_writes': 0,
            'compression_saves': 0,
            'total_size_mb': 0.0,
            'cleanup_runs': 0
        }
        
        # Programar limpieza automática
        self._schedule_cleanup()
        
        logging.info(f"Caché persistente inicializado en {self.cache_dir}")
    
    def get(self, key: str) -> Optional[Tuple[np.ndarray, ImageMetadata]]:
        """Obtiene una imagen del caché persistente"""
        with self.lock:
            if key not in self.entries:
                self.stats['misses'] += 1
                return None
            
            entry = self.entries[key]
            cache_file = self.cache_dir / entry.file_path
            
            if not cache_file.exists():
                # Archivo de caché perdido, eliminar entrada
                del self.entries[key]
                self._save_metadata()
                self.stats['misses'] += 1
                return None
            
            try:
                # Verificar integridad si está habilitada
                if self.config.enable_integrity_check:
                    if not self._verify_integrity(cache_file, entry.checksum):
                        logging.warning(f"Integridad comprometida para {key}, eliminando entrada")
                        self._remove_cache_file(entry)
                        del self.entries[key]
                        self._save_metadata()
                        self.stats['misses'] += 1
                        return None
                
                # Cargar imagen desde caché
                image, metadata = self._load_from_cache(cache_file, entry)
                
                # Actualizar estadísticas de acceso
                entry.last_accessed = time.time()
                entry.access_count += 1
                self.stats['hits'] += 1
                self.stats['disk_reads'] += 1
                
                self._save_metadata()
                return image, metadata
                
            except Exception as e:
                logging.error(f"Error cargando desde caché {key}: {e}")
                self._remove_cache_file(entry)
                del self.entries[key]
                self._save_metadata()
                self.stats['misses'] += 1
                return None
    
    def put(self, key: str, image: np.ndarray, metadata: ImageMetadata, 
            original_path: str, quality_level: QualityLevel):
        """Almacena una imagen en el caché persistente"""
        with self.lock:
            try:
                # Verificar si el archivo es demasiado grande
                estimated_size = image.nbytes / (1024 * 1024)  # MB
                if estimated_size > self.config.max_file_size_mb:
                    return
                
                # Generar nombre de archivo único
                timestamp = int(time.time() * 1000)
                cache_filename = f"{key}_{timestamp}.cache"
                cache_file = self.cache_dir / cache_filename
                
                # Comprimir y guardar
                compressed, compression_algo = self._save_to_cache(
                    cache_file, image, metadata
                )
                
                # Calcular checksum
                checksum = self._calculate_checksum(cache_file)
                
                # Crear entrada de caché
                entry = CacheEntry(
                    key=key,
                    file_path=cache_filename,
                    original_path=original_path,
                    quality_level=quality_level,
                    created_at=time.time(),
                    last_accessed=time.time(),
                    access_count=1,
                    file_size=cache_file.stat().st_size,
                    compressed=compressed,
                    compression_algorithm=compression_algo,
                    checksum=checksum,
                    metadata=metadata.__dict__ if metadata else None
                )
                
                # Eliminar entrada anterior si existe
                if key in self.entries:
                    self._remove_cache_file(self.entries[key])
                
                self.entries[key] = entry
                self.stats['disk_writes'] += 1
                
                if compressed:
                    self.stats['compression_saves'] += 1
                
                # Verificar límites de tamaño
                self._check_size_limits()
                
                self._save_metadata()
                
            except Exception as e:
                logging.error(f"Error guardando en caché {key}: {e}")
    
    def _save_to_cache(self, cache_file: Path, image: np.ndarray, 
                      metadata: ImageMetadata) -> Tuple[bool, str]:
        """Guarda imagen en caché con compresión opcional"""
        data = {
            'image': image,
            'metadata': metadata.__dict__ if metadata else None,
            'dtype': str(image.dtype),
            'shape': image.shape
        }
        
        # Serializar con pickle
        serialized = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Aplicar compresión si está configurada
        if self.config.compression_algorithm == "lzma":
            compressed_data = lzma.compress(
                serialized, 
                preset=self.config.compression_level
            )
            compression_algo = "lzma"
            compressed = True
        elif self.config.compression_algorithm == "gzip":
            compressed_data = gzip.compress(
                serialized, 
                compresslevel=self.config.compression_level
            )
            compression_algo = "gzip"
            compressed = True
        else:
            compressed_data = serialized
            compression_algo = "none"
            compressed = False
        
        # Escribir al archivo
        with open(cache_file, 'wb') as f:
            f.write(compressed_data)
        
        return compressed, compression_algo
    
    def _load_from_cache(self, cache_file: Path, entry: CacheEntry) -> Tuple[np.ndarray, ImageMetadata]:
        """Carga imagen desde caché con descompresión"""
        with open(cache_file, 'rb') as f:
            compressed_data = f.read()
        
        # Descomprimir si es necesario
        if entry.compressed:
            if entry.compression_algorithm == "lzma":
                serialized = lzma.decompress(compressed_data)
            elif entry.compression_algorithm == "gzip":
                serialized = gzip.decompress(compressed_data)
            else:
                serialized = compressed_data
        else:
            serialized = compressed_data
        
        # Deserializar
        data = pickle.loads(serialized)
        
        # Reconstruir imagen
        image = data['image']
        
        # Reconstruir metadata
        if data['metadata']:
            metadata = ImageMetadata(**data['metadata'])
        else:
            metadata = None
        
        return image, metadata
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calcula checksum SHA-256 de un archivo"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    
    def _verify_integrity(self, file_path: Path, expected_checksum: str) -> bool:
        """Verifica la integridad de un archivo de caché"""
        try:
            actual_checksum = self._calculate_checksum(file_path)
            return actual_checksum == expected_checksum
        except Exception:
            return False
    
    def _load_metadata(self):
        """Carga metadatos del caché desde disco"""
        if not self.metadata_file.exists():
            return
        
        try:
            with open(self.metadata_file, 'r') as f:
                data = json.load(f)
            
            for key, entry_data in data.get('entries', {}).items():
                self.entries[key] = CacheEntry(**entry_data)
            
            self.stats.update(data.get('stats', {}))
            
        except Exception as e:
            logging.error(f"Error cargando metadatos de caché: {e}")
    
    def _save_metadata(self):
        """Guarda metadatos del caché a disco"""
        try:
            data = {
                'entries': {k: v.__dict__ for k, v in self.entries.items()},
                'stats': self.stats,
                'last_updated': time.time()
            }
            
            with open(self.metadata_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logging.error(f"Error guardando metadatos de caché: {e}")
    
    def _remove_cache_file(self, entry: CacheEntry):
        """Elimina archivo de caché del disco"""
        try:
            cache_file = self.cache_dir / entry.file_path
            if cache_file.exists():
                cache_file.unlink()
        except Exception as e:
            logging.error(f"Error eliminando archivo de caché {entry.file_path}: {e}")
    
    def _check_size_limits(self):
        """Verifica y aplica límites de tamaño del caché"""
        total_size = sum(entry.file_size for entry in self.entries.values())
        max_size_bytes = self.config.max_disk_size_gb * 1024 * 1024 * 1024
        
        if total_size > max_size_bytes:
            # Ordenar por último acceso (LRU)
            sorted_entries = sorted(
                self.entries.items(),
                key=lambda x: x[1].last_accessed
            )
            
            # Eliminar entradas hasta estar bajo el límite
            for key, entry in sorted_entries:
                if total_size <= max_size_bytes * 0.8:  # Dejar 20% de margen
                    break
                
                self._remove_cache_file(entry)
                total_size -= entry.file_size
                del self.entries[key]
            
            logging.info(f"Limpieza de caché: tamaño reducido a {total_size / (1024**3):.2f} GB")
    
    def _schedule_cleanup(self):
        """Programa limpieza automática del caché"""
        def cleanup_worker():
            while True:
                time.sleep(self.config.cleanup_interval_hours * 3600)
                self.cleanup_old_entries()
        
        cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        cleanup_thread.start()
    
    def cleanup_old_entries(self):
        """Limpia entradas antiguas del caché"""
        with self.lock:
            current_time = time.time()
            max_age_seconds = self.config.max_age_days * 24 * 3600
            
            keys_to_remove = []
            for key, entry in self.entries.items():
                if current_time - entry.created_at > max_age_seconds:
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                entry = self.entries[key]
                self._remove_cache_file(entry)
                del self.entries[key]
            
            if keys_to_remove:
                self.stats['cleanup_runs'] += 1
                self._save_metadata()
                logging.info(f"Limpieza automática: eliminadas {len(keys_to_remove)} entradas antiguas")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del caché"""
        with self.lock:
            total_size = sum(entry.file_size for entry in self.entries.values())
            
            return {
                **self.stats,
                'total_entries': len(self.entries),
                'total_size_mb': total_size / (1024 * 1024),
                'hit_rate': self.stats['hits'] / (self.stats['hits'] + self.stats['misses']) 
                          if (self.stats['hits'] + self.stats['misses']) > 0 else 0.0,
                'compression_ratio': self.stats['compression_saves'] / self.stats['disk_writes']
                                   if self.stats['disk_writes'] > 0 else 0.0
            }
    
    def clear_cache(self):
        """Limpia completamente el caché"""
        with self.lock:
            for entry in self.entries.values():
                self._remove_cache_file(entry)
            
            self.entries.clear()
            self.stats = {
                'hits': 0, 'misses': 0, 'disk_reads': 0, 'disk_writes': 0,
                'compression_saves': 0, 'total_size_mb': 0.0, 'cleanup_runs': 0
            }
            
            if self.metadata_file.exists():
                self.metadata_file.unlink()
            
            logging.info("Caché persistente limpiado completamente")


# Funciones de utilidad
def create_optimized_config(target_memory_mb: float = 512, 
                          enable_progressive: bool = True) -> LoadingConfig:
    """Crea configuración optimizada basada en memoria disponible"""
    return LoadingConfig(
        strategy=LoadingStrategy.PROGRESSIVE if enable_progressive else LoadingStrategy.LAZY,
        max_cache_size_mb=target_memory_mb,
        max_memory_usage=0.6,
        enable_progressive=enable_progressive,
        enable_multithreading=True,
        enable_cache=True,
        cache_thumbnails=True
    )


def estimate_memory_usage(image_paths: List[str], quality: QualityLevel = QualityLevel.FULL) -> float:
    """Estima uso de memoria para un conjunto de imágenes"""
    total_mb = 0.0
    
    for path in image_paths:
        try:
            # Obtener dimensiones sin cargar
            if PIL_AVAILABLE:
                with Image.open(path) as img:
                    width, height = img.size
                    channels = len(img.getbands())
            else:
                continue
            
            # Ajustar por calidad
            if quality != QualityLevel.FULL:
                max_sizes = {
                    QualityLevel.THUMBNAIL: 128,
                    QualityLevel.LOW: 512,
                    QualityLevel.MEDIUM: 1024,
                    QualityLevel.HIGH: 2048
                }
                max_size = max_sizes[quality]
                if max(width, height) > max_size:
                    scale = max_size / max(width, height)
                    width = int(width * scale)
                    height = int(height * scale)
            
            # Calcular memoria (asumiendo uint8)
            pixels = width * height * channels
            mb = pixels / 1024 / 1024
            total_mb += mb
            
        except:
            continue
    
    return total_mb


if __name__ == "__main__":
    # Ejemplo de uso
    logging.basicConfig(level=logging.INFO)
    
    # Crear configuración optimizada
    config = create_optimized_config(target_memory_mb=256, enable_progressive=True)
    
    # Crear cargador
    loader = OptimizedImageLoader(config)
    
    # Ejemplo de carga
    # result = loader.load_image("path/to/image.jpg", QualityLevel.MEDIUM)
    
    print("OptimizedImageLoader listo para usar")
    print(f"Configuración: {config}")
    print(f"Estadísticas: {loader.get_stats()}")