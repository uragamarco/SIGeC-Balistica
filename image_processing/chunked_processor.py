#!/usr/bin/env python3
"""
Procesador de Imágenes por Chunks - SEACABAr
===========================================

Sistema optimizado para procesamiento de imágenes grandes mediante fragmentación
y gestión eficiente de memoria. Diseñado específicamente para imágenes balísticas
de alta resolución que pueden exceder la memoria disponible.

Características principales:
- Procesamiento por chunks con solapamiento configurable
- Lazy loading de imágenes
- Gestión automática de memoria
- Reconstrucción seamless de resultados
- Monitoreo de uso de memoria en tiempo real
- Compatibilidad con pipelines existentes

Autor: SEACABAr Team
Versión: 1.0.0
"""

import cv2
import numpy as np
import gc
import psutil
import os
from typing import Dict, List, Tuple, Optional, Any, Union, Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import logging
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

# Importaciones adicionales para carga parcial
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import tifffile
    TIFFFILE_AVAILABLE = True
except ImportError:
    TIFFFILE_AVAILABLE = False

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
    from image_processing.unified_preprocessor import UnifiedPreprocessor, PreprocessingConfig, PreprocessingResult
    PREPROCESSOR_AVAILABLE = True
except ImportError:
    PREPROCESSOR_AVAILABLE = False


class ChunkStrategy(Enum):
    """Estrategias de fragmentación de imágenes"""
    GRID = "grid"                    # División en cuadrícula regular
    ADAPTIVE = "adaptive"            # División adaptativa basada en contenido
    SLIDING_WINDOW = "sliding_window" # Ventana deslizante con solapamiento
    MEMORY_BASED = "memory_based"    # División basada en memoria disponible


class MemoryMode(Enum):
    """Modos de gestión de memoria"""
    CONSERVATIVE = "conservative"    # Uso mínimo de memoria
    BALANCED = "balanced"           # Balance entre memoria y velocidad
    AGGRESSIVE = "aggressive"       # Máximo rendimiento


@dataclass
class ChunkConfig:
    """Configuración para procesamiento por chunks"""
    # Estrategia de fragmentación
    strategy: ChunkStrategy = ChunkStrategy.ADAPTIVE
    
    # Tamaño de chunks
    chunk_size: Tuple[int, int] = (1024, 1024)  # (height, width)
    overlap: int = 64  # Píxeles de solapamiento entre chunks
    
    # Gestión de memoria
    memory_mode: MemoryMode = MemoryMode.BALANCED
    max_memory_usage: float = 0.8  # Porcentaje máximo de RAM a usar
    memory_check_interval: int = 10  # Chunks entre verificaciones de memoria
    
    # Paralelización
    max_workers: Optional[int] = None  # None = auto-detect
    enable_threading: bool = True
    
    # Optimizaciones
    enable_lazy_loading: bool = True
    cache_chunks: bool = False
    preload_next_chunk: bool = True
    
    # Calidad y precisión
    blend_overlap: bool = True  # Mezclar regiones de solapamiento
    blend_method: str = "linear"  # linear, gaussian, feather
    quality_check: bool = True
    
    # Debug y monitoreo
    enable_monitoring: bool = True
    save_chunk_info: bool = False
    debug_mode: bool = False


@dataclass
class ChunkInfo:
    """Información de un chunk individual"""
    id: int
    position: Tuple[int, int]  # (y, x) posición en imagen original
    size: Tuple[int, int]      # (height, width) del chunk
    overlap_regions: Dict[str, Tuple[int, int, int, int]] = field(default_factory=dict)
    processing_time: float = 0.0
    memory_usage: float = 0.0
    success: bool = False
    error_message: str = ""


@dataclass
class ChunkedProcessingResult:
    """Resultado del procesamiento por chunks"""
    success: bool
    processed_image: Optional[np.ndarray] = None
    original_shape: Tuple[int, int] = (0, 0)
    chunks_processed: int = 0
    total_chunks: int = 0
    total_processing_time: float = 0.0
    peak_memory_usage: float = 0.0
    average_chunk_time: float = 0.0
    chunk_info: List[ChunkInfo] = field(default_factory=list)
    error_message: str = ""
    quality_metrics: Dict[str, float] = field(default_factory=dict)


class MemoryMonitor:
    """Monitor de uso de memoria en tiempo real"""
    
    def __init__(self, max_usage: float = 0.8):
        self.max_usage = max_usage
        self.process = psutil.Process()
        self.peak_usage = 0.0
        self.current_usage = 0.0
        self._lock = threading.Lock()
    
    def get_memory_info(self) -> Dict[str, float]:
        """Obtiene información actual de memoria"""
        with self._lock:
            memory_info = self.process.memory_info()
            virtual_memory = psutil.virtual_memory()
            
            # Memoria del proceso en MB
            process_memory = memory_info.rss / 1024 / 1024
            
            # Porcentaje de memoria total del sistema
            system_usage = virtual_memory.percent / 100.0
            
            # Actualizar pico
            self.current_usage = system_usage
            if system_usage > self.peak_usage:
                self.peak_usage = system_usage
            
            return {
                'process_memory_mb': process_memory,
                'system_usage_percent': system_usage * 100,
                'available_memory_mb': virtual_memory.available / 1024 / 1024,
                'peak_usage_percent': self.peak_usage * 100
            }
    
    def is_memory_available(self) -> bool:
        """Verifica si hay memoria disponible"""
        return self.current_usage < self.max_usage
    
    def force_cleanup(self):
        """Fuerza limpieza de memoria"""
        gc.collect()
        time.sleep(0.1)  # Pequeña pausa para permitir limpieza


class LazyImageLoader:
    """Cargador lazy de imágenes para optimizar memoria"""
    
    def __init__(self, image_path: str, chunk_config: ChunkConfig):
        self.image_path = Path(image_path)
        self.config = chunk_config
        self._image_shape = None
        self._image_dtype = None
        self._cached_image = None
        self._load_lock = threading.Lock()
    
    @property
    def shape(self) -> Tuple[int, int]:
        """Obtiene las dimensiones de la imagen sin cargarla completamente"""
        if self._image_shape is None:
            # Leer solo los metadatos
            img = cv2.imread(str(self.image_path), cv2.IMREAD_UNCHANGED)
            if img is not None:
                self._image_shape = img.shape[:2]  # (height, width)
                self._image_dtype = img.dtype
                del img  # Liberar inmediatamente
                gc.collect()
            else:
                raise ValueError(f"No se pudo leer la imagen: {self.image_path}")
        
        return self._image_shape
    
    @property
    def dtype(self) -> np.dtype:
        """Obtiene el tipo de datos sin cargar la imagen"""
        if self._image_dtype is None:
            _ = self.shape  # Esto cargará los metadatos
        return self._image_dtype
    
    def load_chunk(self, y_start: int, y_end: int, x_start: int, x_end: int) -> np.ndarray:
        """Carga un chunk específico de la imagen"""
        with self._load_lock:
            if self.config.enable_lazy_loading:
                # Cargar solo la región necesaria (si es posible)
                return self._load_region(y_start, y_end, x_start, x_end)
            else:
                # Cargar imagen completa y extraer región
                if self._cached_image is None:
                    self._cached_image = cv2.imread(str(self.image_path), cv2.IMREAD_COLOR)
                    if self._cached_image is None:
                        raise ValueError(f"No se pudo cargar la imagen: {self.image_path}")
                
                return self._cached_image[y_start:y_end, x_start:x_end].copy()
    
    def _load_region(self, y_start: int, y_end: int, x_start: int, x_end: int) -> np.ndarray:
        """Carga una región específica de la imagen usando métodos optimizados"""
        
        # Intentar carga parcial con tifffile para archivos TIFF
        if TIFFFILE_AVAILABLE and str(self.image_path).lower().endswith(('.tiff', '.tif')):
            try:
                # tifffile permite carga de regiones específicas
                with tifffile.TiffFile(str(self.image_path)) as tif:
                    # Leer solo la región especificada
                    region = tif.pages[0].asarray()[y_start:y_end, x_start:x_end]
                    
                    # Convertir a formato BGR si es necesario
                    if len(region.shape) == 3 and region.shape[2] == 3:
                        region = cv2.cvtColor(region, cv2.COLOR_RGB2BGR)
                    elif len(region.shape) == 2:
                        region = cv2.cvtColor(region, cv2.COLOR_GRAY2BGR)
                    
                    return region.copy()
            except Exception as e:
                self.log_warning(f"Error en carga parcial TIFF: {e}, usando fallback")
        
        # Intentar carga optimizada con PIL para imágenes grandes
        if PIL_AVAILABLE:
            try:
                with Image.open(str(self.image_path)) as img:
                    # Para imágenes muy grandes, usar crop para reducir memoria
                    if img.size[0] * img.size[1] > 4000 * 4000:  # > 16MP
                        # Crop la región específica
                        box = (x_start, y_start, x_end, y_end)
                        region_pil = img.crop(box)
                        
                        # Convertir a array numpy
                        region_array = np.array(region_pil)
                        
                        # Convertir RGB a BGR para OpenCV
                        if len(region_array.shape) == 3 and region_array.shape[2] == 3:
                            region_array = cv2.cvtColor(region_array, cv2.COLOR_RGB2BGR)
                        elif len(region_array.shape) == 2:
                            region_array = cv2.cvtColor(region_array, cv2.COLOR_GRAY2BGR)
                        
                        return region_array.copy()
            except Exception as e:
                self.log_warning(f"Error en carga parcial PIL: {e}, usando fallback")
        
        # Fallback: carga completa con OpenCV (método original)
        img = cv2.imread(str(self.image_path), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"No se pudo cargar la imagen: {self.image_path}")
        
        # Extraer región
        region = img[y_start:y_end, x_start:x_end].copy()
        del img  # Liberar imagen completa inmediatamente
        gc.collect()
        
        return region
    
    def cleanup(self):
        """Limpia la caché de imagen"""
        with self._load_lock:
            if self._cached_image is not None:
                del self._cached_image
                self._cached_image = None
                gc.collect()


class ChunkedImageProcessor(LoggerMixin):
    """
    Procesador de imágenes por chunks para manejo eficiente de memoria
    """
    
    def __init__(self, config: Optional[ChunkConfig] = None):
        super().__init__()
        self.config = config or ChunkConfig()
        self.memory_monitor = MemoryMonitor(self.config.max_memory_usage)
        self.current_image_path = None  # Para uso en métodos adaptativos
        
        # Estadísticas
        self.processing_stats = {
            'images_processed': 0,
            'total_chunks': 0,
            'total_processing_time': 0.0,
            'average_memory_usage': 0.0,
            'errors': 0
        }
        
        # Configurar número de workers
        if self.config.max_workers is None:
            self.config.max_workers = min(4, os.cpu_count() or 1)
        
        self.log_info(f"ChunkedImageProcessor inicializado con {self.config.max_workers} workers")
    
    def process_image(self, 
                     image_path: str, 
                     processing_function: Callable[[np.ndarray], np.ndarray],
                     **kwargs) -> ChunkedProcessingResult:
        """
        Procesa una imagen usando la función especificada con fragmentación automática
        
        Args:
            image_path: Ruta a la imagen
            processing_function: Función que procesa cada chunk
            **kwargs: Argumentos adicionales para la función de procesamiento
        
        Returns:
            ChunkedProcessingResult con el resultado del procesamiento
        """
        start_time = time.time()
        
        try:
            # Guardar ruta para uso en métodos adaptativos
            self.current_image_path = image_path
            
            # Inicializar cargador lazy
            loader = LazyImageLoader(image_path, self.config)
            original_shape = loader.shape
            
            self.log_info(f"Procesando imagen {image_path} de tamaño {original_shape}")
            
            # Generar chunks
            chunks = self._generate_chunks(original_shape)
            total_chunks = len(chunks)
            
            self.log_info(f"Imagen dividida en {total_chunks} chunks")
            
            # Procesar chunks
            if self.config.enable_threading and total_chunks > 1:
                processed_chunks = self._process_chunks_parallel(
                    loader, chunks, processing_function, **kwargs
                )
            else:
                processed_chunks = self._process_chunks_sequential(
                    loader, chunks, processing_function, **kwargs
                )
            
            # Reconstruir imagen
            reconstructed_image = self._reconstruct_image(
                processed_chunks, original_shape, chunks
            )
            
            # Calcular métricas
            processing_time = time.time() - start_time
            peak_memory = self.memory_monitor.peak_usage
            
            # Limpiar
            loader.cleanup()
            
            # Crear resultado
            result = ChunkedProcessingResult(
                success=True,
                processed_image=reconstructed_image,
                original_shape=original_shape,
                chunks_processed=len([c for c in chunks if c.success]),
                total_chunks=total_chunks,
                total_processing_time=processing_time,
                peak_memory_usage=peak_memory,
                average_chunk_time=processing_time / total_chunks if total_chunks > 0 else 0,
                chunk_info=chunks
            )
            
            # Actualizar estadísticas
            self._update_stats(result)
            
            self.log_info(f"Procesamiento completado en {processing_time:.2f}s")
            
            return result
            
        except Exception as e:
            error_msg = f"Error en procesamiento por chunks: {str(e)}"
            self.log_error(error_msg)
            
            return ChunkedProcessingResult(
                success=False,
                error_message=error_msg,
                total_processing_time=time.time() - start_time
            )
    
    def _generate_chunks(self, image_shape: Tuple[int, int]) -> List[ChunkInfo]:
        """Genera la lista de chunks basada en la estrategia configurada"""
        height, width = image_shape
        
        if self.config.strategy == ChunkStrategy.GRID:
            return self._generate_chunks_grid(image_shape)
        elif self.config.strategy == ChunkStrategy.ADAPTIVE:
            return self._generate_chunks_adaptive(image_shape)
        elif self.config.strategy == ChunkStrategy.MEMORY_BASED:
            return self._generate_chunks_memory_based(image_shape)
        elif self.config.strategy == ChunkStrategy.SLIDING_WINDOW:
            return self._generate_chunks_sliding_window(image_shape)
        else:
            # Fallback a GRID
            return self._generate_chunks_grid(image_shape)
    
    def _generate_chunks_adaptive(self, image_shape: Tuple[int, int]) -> List[ChunkInfo]:
        """Genera chunks usando división adaptativa basada en contenido"""
        height, width = image_shape
        chunk_height, chunk_width = self.config.chunk_size
        overlap = self.config.overlap
        
        chunks = []
        chunk_id = 0
        
        # Cargar una versión reducida de la imagen para análisis
        try:
            # Usar el loader para obtener una muestra de la imagen
            sample_loader = LazyImageLoader(self.current_image_path, self.config)
            
            # Cargar imagen completa a resolución reducida para análisis
            sample_height = min(height, 2048)
            sample_width = min(width, 2048)
            scale_y = height / sample_height
            scale_x = width / sample_width
            
            # Cargar muestra central para análisis
            center_y = height // 2
            center_x = width // 2
            sample_y_start = max(0, center_y - sample_height // 2)
            sample_x_start = max(0, center_x - sample_width // 2)
            sample_y_end = min(height, sample_y_start + sample_height)
            sample_x_end = min(width, sample_x_start + sample_width)
            
            sample_image = sample_loader._load_region(
                sample_y_start, sample_y_end, sample_x_start, sample_x_end
            )
            
            # Análisis de contenido para determinar regiones de interés
            complexity_map = self._analyze_image_complexity(sample_image)
            
            # Generar chunks adaptativos basados en complejidad
            for y in range(0, height, chunk_height - overlap):
                for x in range(0, width, chunk_width - overlap):
                    y_end = min(y + chunk_height, height)
                    x_end = min(x + chunk_width, width)
                    
                    # Mapear coordenadas a la muestra
                    sample_y = int((y + y_end) / 2 / scale_y) - sample_y_start
                    sample_x = int((x + x_end) / 2 / scale_x) - sample_x_start
                    
                    # Obtener complejidad de la región
                    if (0 <= sample_y < complexity_map.shape[0] and 
                        0 <= sample_x < complexity_map.shape[1]):
                        complexity = complexity_map[sample_y, sample_x]
                    else:
                        complexity = 0.5  # Complejidad media por defecto
                    
                    # Ajustar tamaño de chunk basado en complejidad
                    if complexity > 0.7:  # Alta complejidad - chunks más pequeños
                        adjusted_chunk_height = int(chunk_height * 0.7)
                        adjusted_chunk_width = int(chunk_width * 0.7)
                        adjusted_overlap = int(overlap * 1.5)
                    elif complexity < 0.3:  # Baja complejidad - chunks más grandes
                        adjusted_chunk_height = int(chunk_height * 1.3)
                        adjusted_chunk_width = int(chunk_width * 1.3)
                        adjusted_overlap = overlap
                    else:  # Complejidad media - tamaño estándar
                        adjusted_chunk_height = chunk_height
                        adjusted_chunk_width = chunk_width
                        adjusted_overlap = overlap
                    
                    # Recalcular límites con tamaño ajustado
                    y_end = min(y + adjusted_chunk_height, height)
                    x_end = min(x + adjusted_chunk_width, width)
                    
                    chunk = ChunkInfo(
                        id=chunk_id,
                        position=(y, x),
                        size=(y_end - y, x_end - x)
                    )
                    
                    # Agregar información de complejidad
                    chunk.overlap_regions['complexity'] = complexity
                    
                    chunks.append(chunk)
                    chunk_id += 1
            
            sample_loader.cleanup()
            
        except Exception as e:
            self.log_warning(f"Error en análisis adaptativo: {e}, usando división grid")
            return self._generate_chunks_grid(image_shape)
        
        self.log_info(f"Generados {len(chunks)} chunks adaptativos")
        return chunks
    
    def _generate_chunks_memory_based(self, image_shape: Tuple[int, int]) -> List[ChunkInfo]:
        """Genera chunks basados en memoria disponible y uso dinámico"""
        height, width = image_shape
        overlap = self.config.overlap
        
        chunks = []
        chunk_id = 0
        
        # Obtener información de memoria actual
        memory_info = self.memory_monitor.get_memory_info()
        available_mb = memory_info['available_memory_mb']
        
        # Calcular tamaño óptimo de chunk basado en memoria
        bytes_per_pixel = 3  # RGB
        safety_factor = 0.1 if self.config.memory_mode == MemoryMode.CONSERVATIVE else 0.2
        
        # Reservar memoria para procesamiento y overhead
        usable_memory_mb = available_mb * safety_factor
        max_pixels_per_chunk = int((usable_memory_mb * 1024 * 1024) / bytes_per_pixel)
        
        # Calcular dimensiones óptimas manteniendo aspect ratio
        aspect_ratio = width / height
        optimal_height = int(np.sqrt(max_pixels_per_chunk / aspect_ratio))
        optimal_width = int(optimal_height * aspect_ratio)
        
        # Aplicar límites mínimos y máximos
        min_chunk_size = 256
        max_chunk_size = 2048
        
        chunk_height = max(min_chunk_size, min(optimal_height, max_chunk_size))
        chunk_width = max(min_chunk_size, min(optimal_width, max_chunk_size))
        
        self.log_info(f"Tamaño de chunk optimizado para memoria: {chunk_height}x{chunk_width}")
        self.log_info(f"Memoria disponible: {available_mb:.1f}MB, usando: {usable_memory_mb:.1f}MB por chunk")
        
        # Generar chunks con tamaño optimizado
        y = 0
        while y < height:
            x = 0
            while x < width:
                # Verificar memoria antes de cada fila de chunks
                if not self.memory_monitor.is_memory_available():
                    self.log_warning("Memoria insuficiente, reduciendo tamaño de chunks")
                    chunk_height = int(chunk_height * 0.8)
                    chunk_width = int(chunk_width * 0.8)
                
                y_end = min(y + chunk_height, height)
                x_end = min(x + chunk_width, width)
                
                chunk = ChunkInfo(
                    id=chunk_id,
                    position=(y, x),
                    size=(y_end - y, x_end - x)
                )
                
                # Agregar información de memoria
                chunk.overlap_regions['memory_optimized'] = True
                chunk.overlap_regions['target_memory_mb'] = usable_memory_mb
                
                chunks.append(chunk)
                chunk_id += 1
                
                x += chunk_width - overlap
            
            y += chunk_height - overlap
        
        self.log_info(f"Generados {len(chunks)} chunks optimizados para memoria")
        return chunks
    
    def _generate_chunks_sliding_window(self, image_shape: Tuple[int, int]) -> List[ChunkInfo]:
        """Genera chunks usando ventana deslizante con solapamiento inteligente"""
        height, width = image_shape
        chunk_height, chunk_width = self.config.chunk_size
        overlap = self.config.overlap
        
        chunks = []
        chunk_id = 0
        
        # Calcular paso óptimo para ventana deslizante
        step_y = max(chunk_height // 4, chunk_height - overlap)
        step_x = max(chunk_width // 4, chunk_width - overlap)
        
        for y in range(0, height - chunk_height + 1, step_y):
            for x in range(0, width - chunk_width + 1, step_x):
                y_end = min(y + chunk_height, height)
                x_end = min(x + chunk_width, width)
                
                chunk = ChunkInfo(
                    id=chunk_id,
                    position=(y, x),
                    size=(y_end - y, x_end - x)
                )
                
                # Calcular regiones de solapamiento con chunks anteriores
                overlap_regions = {}
                for prev_chunk in chunks[-10:]:  # Verificar últimos 10 chunks
                    prev_y, prev_x = prev_chunk.position
                    prev_h, prev_w = prev_chunk.size
                    
                    # Calcular intersección
                    intersect_y1 = max(y, prev_y)
                    intersect_x1 = max(x, prev_x)
                    intersect_y2 = min(y_end, prev_y + prev_h)
                    intersect_x2 = min(x_end, prev_x + prev_w)
                    
                    if intersect_y1 < intersect_y2 and intersect_x1 < intersect_x2:
                        overlap_key = f"overlap_{prev_chunk.id}"
                        overlap_regions[overlap_key] = (
                            intersect_y1 - y, intersect_x1 - x,
                            intersect_y2 - y, intersect_x2 - x
                        )
                
                chunk.overlap_regions = overlap_regions
                chunks.append(chunk)
                chunk_id += 1
        
        # Agregar chunks de borde si es necesario
        if height % step_y != 0:
            y = height - chunk_height
            for x in range(0, width - chunk_width + 1, step_x):
                x_end = min(x + chunk_width, width)
                chunk = ChunkInfo(
                    id=chunk_id,
                    position=(y, x),
                    size=(chunk_height, x_end - x)
                )
                chunks.append(chunk)
                chunk_id += 1
        
        if width % step_x != 0:
            x = width - chunk_width
            for y in range(0, height - chunk_height + 1, step_y):
                y_end = min(y + chunk_height, height)
                chunk = ChunkInfo(
                    id=chunk_id,
                    position=(y, x),
                    size=(y_end - y, chunk_width)
                )
                chunks.append(chunk)
                chunk_id += 1
        
        self.log_info(f"Generados {len(chunks)} chunks con ventana deslizante")
        return chunks
    
    def _analyze_image_complexity(self, image: np.ndarray) -> np.ndarray:
        """Analiza la complejidad de contenido de la imagen"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Calcular gradientes para detectar bordes y texturas
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Calcular varianza local para detectar texturas
        kernel = np.ones((15, 15), np.float32) / 225
        local_mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        local_variance = cv2.filter2D((gray.astype(np.float32) - local_mean)**2, -1, kernel)
        
        # Combinar métricas de complejidad
        complexity = (gradient_magnitude / 255.0) * 0.6 + (local_variance / 10000.0) * 0.4
        
        # Normalizar a rango [0, 1]
        complexity = np.clip(complexity, 0, 1)
        
        # Suavizar mapa de complejidad
        complexity = cv2.GaussianBlur(complexity, (21, 21), 0)
        
        return complexity
    
    def _generate_chunks_grid(self, image_shape: Tuple[int, int]) -> List[ChunkInfo]:
        """Genera chunks usando división en cuadrícula"""
        height, width = image_shape
        chunk_height, chunk_width = self.config.chunk_size
        overlap = self.config.overlap
        
        chunks = []
        chunk_id = 0
        
        for y in range(0, height, chunk_height - overlap):
            for x in range(0, width, chunk_width - overlap):
                y_end = min(y + chunk_height, height)
                x_end = min(x + chunk_width, width)
                
                chunk = ChunkInfo(
                    id=chunk_id,
                    position=(y, x),
                    size=(y_end - y, x_end - x)
                )
                chunks.append(chunk)
                chunk_id += 1
        
        return chunks
    
    def _process_chunks_sequential(self, 
                                 loader: LazyImageLoader,
                                 chunks: List[ChunkInfo],
                                 processing_function: Callable,
                                 **kwargs) -> Dict[int, np.ndarray]:
        """Procesa chunks secuencialmente"""
        processed_chunks = {}
        
        for i, chunk in enumerate(chunks):
            if self.config.enable_monitoring and i % self.config.memory_check_interval == 0:
                if not self.memory_monitor.is_memory_available():
                    self.memory_monitor.force_cleanup()
                    self.log_warning("Memoria baja, forzando limpieza")
            
            try:
                # Cargar chunk
                y, x = chunk.position
                h, w = chunk.size
                chunk_image = loader.load_chunk(y, y + h, x, x + w)
                
                # Procesar chunk
                start_time = time.time()
                processed_chunk = processing_function(chunk_image, **kwargs)
                processing_time = time.time() - start_time
                
                # Actualizar información del chunk
                chunk.processing_time = processing_time
                chunk.success = True
                
                processed_chunks[chunk.id] = processed_chunk
                
                # Liberar memoria del chunk original
                del chunk_image
                
            except Exception as e:
                chunk.error_message = str(e)
                chunk.success = False
                self.log_error(f"Error procesando chunk {chunk.id}: {str(e)}")
        
        return processed_chunks
    
    def _process_chunks_parallel(self, 
                               loader: LazyImageLoader,
                               chunks: List[ChunkInfo],
                               processing_function: Callable,
                               **kwargs) -> Dict[int, np.ndarray]:
        """Procesa chunks en paralelo con gestión optimizada de memoria y GPU"""
        processed_chunks = {}
        
        # Configurar número de workers basado en memoria y GPU disponible
        max_workers = min(self.config.max_workers, len(chunks))
        
        # Ajustar workers basado en memoria disponible
        memory_info = self.memory_monitor.get_memory_info()
        if memory_info['system_usage_percent'] > 70:
            max_workers = max(1, max_workers // 2)
            self.log_warning(f"Reduciendo workers a {max_workers} debido a alta utilización de memoria")
        
        # Procesar en lotes para evitar sobrecarga de memoria
        batch_size = max(1, max_workers * 2)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for batch_start in range(0, len(chunks), batch_size):
                batch_end = min(batch_start + batch_size, len(chunks))
                batch_chunks = chunks[batch_start:batch_end]
                
                # Enviar lote de chunks para procesamiento
                future_to_chunk = {
                    executor.submit(
                        self._process_single_chunk_optimized, 
                        loader, chunk, processing_function, **kwargs
                    ): chunk for chunk in batch_chunks
                }
                
                # Recopilar resultados del lote
                for future in as_completed(future_to_chunk):
                    chunk = future_to_chunk[future]
                    try:
                        result = future.result()
                        if result is not None:
                            processed_chunks[chunk.id] = result
                            chunk.success = True
                        else:
                            chunk.success = False
                            chunk.error_message = "Procesamiento falló"
                    except Exception as e:
                        chunk.success = False
                        chunk.error_message = str(e)
                        self.log_error(f"Error procesando chunk {chunk.id}: {e}")
                
                # Verificar memoria después de cada lote
                if not self.memory_monitor.is_memory_available():
                    self.log_warning("Memoria insuficiente, forzando limpieza")
                    self.memory_monitor.force_cleanup()
                    
                    # Si aún hay problemas de memoria, reducir workers
                    if not self.memory_monitor.is_memory_available():
                        max_workers = max(1, max_workers // 2)
                        self.log_warning(f"Reduciendo workers a {max_workers}")
        
        return processed_chunks
    
    def _process_single_chunk_optimized(self, 
                                      loader: LazyImageLoader,
                                      chunk: ChunkInfo,
                                      processing_function: Callable,
                                      **kwargs) -> Optional[np.ndarray]:
        """Procesa un chunk individual con optimizaciones de memoria y GPU"""
        start_time = time.time()
        
        try:
            # Verificar memoria antes de procesar
            if not self.memory_monitor.is_memory_available():
                self.log_warning(f"Memoria insuficiente para chunk {chunk.id}, saltando")
                return None
            
            # Cargar chunk
            y_start, x_start = chunk.position
            chunk_height, chunk_width = chunk.size
            y_end = y_start + chunk_height
            x_end = x_start + chunk_width
            
            chunk_image = loader.load_chunk(y_start, y_end, x_start, x_end)
            
            if chunk_image is None or chunk_image.size == 0:
                self.log_warning(f"Chunk {chunk.id} vacío o inválido")
                return None
            
            # Verificar si hay aceleración GPU disponible
            try:
                # Intentar usar GPU si está disponible
                if hasattr(processing_function, '__self__') and hasattr(processing_function.__self__, 'gpu_accelerator'):
                    gpu_accelerator = processing_function.__self__.gpu_accelerator
                    if gpu_accelerator and gpu_accelerator.is_gpu_enabled():
                        # Procesar con GPU
                        processed_chunk = processing_function(chunk_image, use_gpu=True, **kwargs)
                    else:
                        # Procesar con CPU
                        processed_chunk = processing_function(chunk_image, **kwargs)
                else:
                    # Procesamiento estándar
                    processed_chunk = processing_function(chunk_image, **kwargs)
                    
            except Exception as gpu_error:
                self.log_warning(f"Error en procesamiento GPU para chunk {chunk.id}, usando CPU: {gpu_error}")
                processed_chunk = processing_function(chunk_image, **kwargs)
            
            # Limpiar memoria del chunk original
            del chunk_image
            
            # Verificar resultado
            if processed_chunk is None:
                self.log_warning(f"Procesamiento de chunk {chunk.id} retornó None")
                return None
            
            # Actualizar estadísticas del chunk
            chunk.processing_time = time.time() - start_time
            chunk.memory_usage = self.memory_monitor.get_memory_info()['process_memory_mb']
            
            # Forzar limpieza de memoria si es necesario
            if chunk.id % 10 == 0:  # Cada 10 chunks
                gc.collect()
            
            return processed_chunk
            
        except Exception as e:
            chunk.processing_time = time.time() - start_time
            chunk.error_message = str(e)
            self.log_error(f"Error procesando chunk {chunk.id}: {e}")
            return None
    
    def _reconstruct_image(self, 
                         processed_chunks: Dict[int, np.ndarray],
                         original_shape: Tuple[int, int],
                         chunks: List[ChunkInfo]) -> np.ndarray:
        """Reconstruye la imagen a partir de los chunks procesados"""
        height, width = original_shape
        
        # Determinar el número de canales del primer chunk válido
        sample_chunk = next(iter(processed_chunks.values()))
        if len(sample_chunk.shape) == 3:
            channels = sample_chunk.shape[2]
            reconstructed = np.zeros((height, width, channels), dtype=sample_chunk.dtype)
        else:
            reconstructed = np.zeros((height, width), dtype=sample_chunk.dtype)
        
        # Matriz de pesos para blending
        if self.config.blend_overlap:
            weight_matrix = np.zeros((height, width), dtype=np.float32)
        
        # Colocar chunks en la imagen reconstruida
        for chunk in chunks:
            if not chunk.success or chunk.id not in processed_chunks:
                continue
            
            y, x = chunk.position
            h, w = chunk.size
            chunk_data = processed_chunks[chunk.id]
            
            if self.config.blend_overlap:
                # Crear máscara de peso para este chunk
                chunk_weight = self._create_chunk_weight_mask((h, w))
                
                # Actualizar imagen y pesos
                if len(reconstructed.shape) == 3:
                    for c in range(channels):
                        reconstructed[y:y+h, x:x+w, c] += chunk_data[:, :, c] * chunk_weight
                else:
                    reconstructed[y:y+h, x:x+w] += chunk_data * chunk_weight
                
                weight_matrix[y:y+h, x:x+w] += chunk_weight
            else:
                # Colocación simple sin blending
                reconstructed[y:y+h, x:x+w] = chunk_data
        
        # Normalizar por pesos si se usa blending
        if self.config.blend_overlap:
            # Evitar división por cero
            weight_matrix[weight_matrix == 0] = 1
            
            if len(reconstructed.shape) == 3:
                for c in range(channels):
                    reconstructed[:, :, c] /= weight_matrix
            else:
                reconstructed /= weight_matrix
        
        return reconstructed.astype(sample_chunk.dtype)
    
    def _create_chunk_weight_mask(self, chunk_shape: Tuple[int, int]) -> np.ndarray:
        """Crea una máscara de peso para blending suave"""
        h, w = chunk_shape
        
        if self.config.blend_method == "linear":
            # Peso lineal desde los bordes
            weight = np.ones((h, w), dtype=np.float32)
            
            # Reducir peso en los bordes
            fade_size = min(self.config.overlap // 2, min(h, w) // 4)
            if fade_size > 0:
                for i in range(fade_size):
                    alpha = (i + 1) / fade_size
                    weight[i, :] *= alpha
                    weight[-(i+1), :] *= alpha
                    weight[:, i] *= alpha
                    weight[:, -(i+1)] *= alpha
            
            return weight
        
        elif self.config.blend_method == "gaussian":
            # Peso gaussiano
            center_y, center_x = h // 2, w // 2
            y, x = np.ogrid[:h, :w]
            
            sigma = min(h, w) / 4
            weight = np.exp(-((y - center_y)**2 + (x - center_x)**2) / (2 * sigma**2))
            
            return weight.astype(np.float32)
        
        else:  # "feather" o por defecto
            return np.ones((h, w), dtype=np.float32)
    
    def _update_stats(self, result: ChunkedProcessingResult):
        """Actualiza estadísticas de procesamiento"""
        self.processing_stats['images_processed'] += 1
        self.processing_stats['total_chunks'] += result.total_chunks
        self.processing_stats['total_processing_time'] += result.total_processing_time
        
        if not result.success:
            self.processing_stats['errors'] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas de procesamiento"""
        stats = self.processing_stats.copy()
        
        if stats['images_processed'] > 0:
            stats['average_processing_time'] = stats['total_processing_time'] / stats['images_processed']
            stats['average_chunks_per_image'] = stats['total_chunks'] / stats['images_processed']
        
        stats['memory_info'] = self.memory_monitor.get_memory_info()
        
        return stats
    
    def process_with_preprocessor(self, 
                                image_path: str,
                                preprocessor_config: Optional[Dict] = None) -> ChunkedProcessingResult:
        """
        Procesa una imagen usando el UnifiedPreprocessor con fragmentación
        """
        if not PREPROCESSOR_AVAILABLE:
            raise ImportError("UnifiedPreprocessor no disponible")
        
        # Crear función de procesamiento que usa el preprocessor
        def preprocess_chunk(chunk_image: np.ndarray) -> np.ndarray:
            # Crear preprocessor temporal para el chunk
            preprocessor = UnifiedPreprocessor(preprocessor_config)
            
            # Procesar chunk como si fuera una imagen completa
            # Nota: Esto es una simplificación, en una implementación completa
            # se podría optimizar para chunks específicos
            result = preprocessor.preprocess_ballistic_image(chunk_image)
            
            return result
        
        return self.process_image(image_path, preprocess_chunk)


# Funciones de utilidad
def estimate_chunk_size(image_shape: Tuple[int, int], 
                       available_memory_mb: float,
                       safety_factor: float = 0.1) -> Tuple[int, int]:
    """
    Estima el tamaño óptimo de chunk basado en memoria disponible
    
    Args:
        image_shape: (height, width) de la imagen
        available_memory_mb: Memoria disponible en MB
        safety_factor: Factor de seguridad (0.1 = usar solo 10% de memoria)
    
    Returns:
        (chunk_height, chunk_width) óptimo
    """
    height, width = image_shape
    
    # Estimar bytes por píxel (asumiendo RGB)
    bytes_per_pixel = 3
    
    # Calcular píxeles máximos por chunk
    max_bytes = available_memory_mb * 1024 * 1024 * safety_factor
    max_pixels = int(max_bytes / bytes_per_pixel)
    
    # Calcular dimensiones cuadradas óptimas
    optimal_side = int(np.sqrt(max_pixels))
    
    # Ajustar a las dimensiones de la imagen
    chunk_height = min(optimal_side, height)
    chunk_width = min(optimal_side, width)
    
    return (chunk_height, chunk_width)


def create_memory_efficient_config(image_path: str) -> ChunkConfig:
    """
    Crea una configuración optimizada para memoria basada en la imagen
    """
    # Obtener información de la imagen
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"No se pudo leer la imagen: {image_path}")
    
    image_shape = img.shape[:2]
    del img
    gc.collect()
    
    # Obtener información de memoria
    memory_info = psutil.virtual_memory()
    available_mb = memory_info.available / 1024 / 1024
    
    # Estimar tamaño de chunk
    chunk_size = estimate_chunk_size(image_shape, available_mb)
    
    # Crear configuración
    config = ChunkConfig(
        strategy=ChunkStrategy.MEMORY_BASED,
        chunk_size=chunk_size,
        overlap=64,
        memory_mode=MemoryMode.CONSERVATIVE,
        max_memory_usage=0.7,
        enable_lazy_loading=True,
        enable_threading=True,
        blend_overlap=True
    )
    
    return config


if __name__ == "__main__":
    # Ejemplo de uso
    logging.basicConfig(level=logging.INFO)
    
    # Crear configuración
    config = ChunkConfig(
        strategy=ChunkStrategy.GRID,
        chunk_size=(512, 512),
        overlap=32,
        enable_threading=True,
        blend_overlap=True
    )
    
    # Crear procesador
    processor = ChunkedImageProcessor(config)
    
    # Función de ejemplo para procesamiento
    def example_processing(image: np.ndarray) -> np.ndarray:
        # Ejemplo: aplicar filtro gaussiano
        return cv2.GaussianBlur(image, (5, 5), 1.0)
    
    # Procesar imagen (ejemplo)
    # result = processor.process_image("path/to/large/image.jpg", example_processing)
    
    print("ChunkedImageProcessor listo para usar")
    print(f"Configuración: {config}")
    print(f"Estadísticas: {processor.get_stats()}")