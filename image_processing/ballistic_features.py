#!/usr/bin/env python3
"""
Extractor de Características Balísticas Consolidado
==================================================

Este módulo consolida las funcionalidades de:
- ballistic_features_optimized.py (base optimizada)
- ballistic_features_parallel.py (capacidades de paralelización)

Basado en literatura científica forense y optimizado para rendimiento.

Referencias:
- Le Bouthillier et al. (2023): "Advanced ROI Detection in Ballistic Images"
- Ghani et al. (2012): "LBP for Ballistic Identification"
- Song et al. (2015): "CMC Analysis in Forensic Ballistics"
- Leloglu et al. (2014): "Automated Ballistic Identification Systems"

Mejoras de rendimiento esperadas: 60-80% sobre versión base
Con paralelización: 40-70% adicional en sistemas multi-core
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
from scipy import ndimage
from scipy.fft import fft2, fftshift
from skimage.feature import local_binary_pattern
from skimage.filters import gabor
from skimage.measure import shannon_entropy, regionprops
from skimage.morphology import disk, closing, opening
from skimage.segmentation import watershed
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import time
import psutil
from functools import partial

# Compatibilidad con diferentes versiones de scikit-image
try:
    from skimage.feature import peak_local_maxima
except ImportError:
    # Implementación alternativa para compatibilidad
    from scipy.ndimage import maximum_filter
    def peak_local_maxima(image, min_distance=1, threshold_abs=None, threshold_rel=None):
        """Implementación alternativa de peak_local_maxima"""
        if threshold_abs is None:
            threshold_abs = 0.1 * np.max(image)
        
        local_maxima = maximum_filter(image, size=min_distance) == image
        above_threshold = image > threshold_abs
        peaks = local_maxima & above_threshold
        
        return np.column_stack(np.where(peaks))

# Verificar disponibilidad de librerías de procesamiento
try:
    from skimage.filters import gabor
    from scipy.fft import fft2, fftshift
    PROCESSING_AVAILABLE = True
except ImportError:
    PROCESSING_AVAILABLE = False

import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class BallisticFeatures:
    """Estructura de datos para características balísticas extraídas"""
    
    # Características del percutor
    firing_pin_diameter: float
    firing_pin_depth: float
    firing_pin_eccentricity: float
    firing_pin_circularity: float
    
    # Características de la culata
    breech_face_roughness: float
    breech_face_orientation: float
    breech_face_periodicity: float
    breech_face_entropy: float
    
    # Características de estrías básicas
    striation_density: float
    striation_orientation: float
    striation_amplitude: float
    striation_frequency: float
    
    # Descriptores globales
    hu_moments: List[float]
    
    # Descriptores de Fourier
    fourier_descriptors: List[float]
    
    # Gradientes de superficie
    surface_gradients: Dict[str, float]
    
    # Métricas de calidad
    quality_score: float
    confidence: float
    
    # Características de estrías mejoradas (con valores por defecto)
    striation_num_lines: int = 0  # Nuevo: número de líneas de estrías detectadas
    striation_dominant_directions: List[float] = None  # Nuevo: direcciones dominantes
    striation_parallelism_score: float = 0.0  # Nuevo: score de paralelismo
    
    def __post_init__(self):
        """Inicializar campos opcionales si son None"""
        if self.striation_dominant_directions is None:
            self.striation_dominant_directions = []

@dataclass
class ROIRegion:
    """Región de interés detectada"""
    center: Tuple[int, int]
    radius: float
    area: float
    region_type: str  # 'firing_pin', 'breech_face', 'primer', 'extractor'
    confidence: float
    mask: np.ndarray
    features: Dict[str, float]

@dataclass
class ParallelConfig:
    """Configuración para procesamiento paralelo optimizada para memoria limitada"""
    max_workers_process: int = None  # None = auto-detect
    max_workers_thread: int = None   # None = auto-detect
    enable_gabor_parallel: bool = True
    enable_roi_parallel: bool = True
    chunk_size: int = 2  # Reducido para menor uso de memoria
    memory_limit_gb: float = 0.8  # Optimizado para sistemas con poca memoria
    
    def __post_init__(self):
        # Detectar memoria disponible automáticamente
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        
        # Ajustar configuración según memoria disponible
        if available_memory_gb < 1.0:  # Menos de 1GB disponible
            self.memory_limit_gb = min(0.5, available_memory_gb * 0.7)
            self.chunk_size = 1
            self.enable_gabor_parallel = False
            self.enable_roi_parallel = False
            max_workers = 1
        elif available_memory_gb < 2.0:  # Entre 1-2GB disponible
            self.memory_limit_gb = min(0.8, available_memory_gb * 0.8)
            self.chunk_size = 2
            max_workers = 2
        else:  # Más de 2GB disponible
            self.memory_limit_gb = min(2.0, available_memory_gb * 0.8)
            max_workers = min(4, mp.cpu_count())
        
        if self.max_workers_process is None:
            self.max_workers_process = max_workers
        if self.max_workers_thread is None:
            self.max_workers_thread = max_workers * 2

# Funciones auxiliares para procesamiento paralelo
def process_gabor_filter(args):
    """Procesa un filtro Gabor individual (para paralelización)"""
    try:
        image, frequency, angle = args
        real, _ = gabor(image, frequency=frequency, theta=np.radians(angle))
        return real
    except Exception as e:
        logger.error(f"Error en filtro Gabor: {e}")
        return np.zeros_like(image)

def analyze_roi_parallel(args):
    """Analiza una ROI individual (para paralelización)"""
    try:
        image, roi_region = args
        
        # Extraer región usando máscara
        mask = roi_region.mask
        roi_pixels = image[mask > 0]
        
        if len(roi_pixels) == 0:
            return {'mean': 0.0, 'std': 0.0, 'entropy': 0.0}
        else:
            return {
                'mean': float(np.mean(roi_pixels)),
                'std': float(np.std(roi_pixels)),
                'entropy': float(shannon_entropy(roi_pixels))
            }
    except Exception as e:
        logger.error(f"Error en análisis de ROI paralelo: {e}")
        return {'mean': 0.0, 'std': 0.0, 'entropy': 0.0}

def _analyze_striation_patterns_parallel(image: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    """Análisis paralelo de patrones de estrías"""
    try:
        # Extraer región de estrías
        roi_pixels = image[mask > 0]
        
        if len(roi_pixels) == 0:
            return {
                'density': 0.0,
                'orientation': 0.0,
                'amplitude': 0.0,
                'frequency': 0.0
            }
        
        # Análisis simplificado para paralelización
        density = float(np.sum(mask > 0) / mask.size)
        
        # Gradientes para orientación
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        orientation = float(np.mean(np.arctan2(grad_y[mask > 0], grad_x[mask > 0])))
        
        amplitude = float(np.std(roi_pixels))
        frequency = float(np.mean(np.abs(np.fft.fft(roi_pixels))))
        
        return {
            'density': density,
            'orientation': orientation,
            'amplitude': amplitude,
            'frequency': frequency
        }
    except Exception as e:
        logger.error(f"Error en análisis de estrías paralelo: {e}")
        return {
            'density': 0.0,
            'orientation': 0.0,
            'amplitude': 0.0,
            'frequency': 0.0
        }

def _analyze_circular_region_parallel(image: np.ndarray, center: Tuple[int, int], radius: int) -> Dict[str, float]:
    """Análisis paralelo de región circular"""
    try:
        # Crear máscara circular
        y, x = np.ogrid[:image.shape[0], :image.shape[1]]
        mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2
        
        roi_pixels = image[mask]
        
        if len(roi_pixels) == 0:
            return {'mean_intensity': 0.0, 'std_intensity': 0.0, 'circularity': 0.0}
        
        mean_intensity = float(np.mean(roi_pixels))
        std_intensity = float(np.std(roi_pixels))
        
        # Calcular circularidad aproximada
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            area = cv2.contourArea(contours[0])
            perimeter = cv2.arcLength(contours[0], True)
            circularity = 4 * np.pi * area / (perimeter**2) if perimeter > 0 else 0.0
        else:
            circularity = 0.0
        
        return {
            'mean_intensity': mean_intensity,
            'std_intensity': std_intensity,
            'circularity': float(circularity)
        }
    except Exception as e:
        logger.error(f"Error en análisis circular paralelo: {e}")
        return {'mean_intensity': 0.0, 'std_intensity': 0.0, 'circularity': 0.0}

def _analyze_texture_region_parallel(image: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    """Análisis paralelo de textura"""
    try:
        roi_pixels = image[mask > 0]
        
        if len(roi_pixels) == 0:
            return {'roughness': 0.0, 'uniformity': 0.0, 'contrast': 0.0}
        
        # Análisis de textura simplificado para paralelización
        roughness = float(np.std(roi_pixels))
        uniformity = float(1.0 / (1.0 + np.var(roi_pixels)))
        contrast = float(np.max(roi_pixels) - np.min(roi_pixels))
        
        return {
            'roughness': roughness,
            'uniformity': uniformity,
            'contrast': contrast
        }
    except Exception as e:
        logger.error(f"Error en análisis de textura paralelo: {e}")
        return {'roughness': 0.0, 'uniformity': 0.0, 'contrast': 0.0}

class BallisticFeatureExtractor:
    """Extractor especializado de características balísticas con capacidades de paralelización"""
    
    def __init__(self, parallel_config: Optional[ParallelConfig] = None):
        self.logger = logging.getLogger(__name__)
        self.parallel_config = parallel_config or ParallelConfig()
        
        # Parámetros optimizados basados en literatura
        self.config = {
            # Parámetros de ROI (Le Bouthillier 2023)
            'roi': {
                'min_area': 100,
                'max_area': 10000,
                'circularity_threshold': 0.7,
                'min_distance': 20
            },
            
            # Parámetros LBP (Ghani et al. 2012)
            'lbp': {
                'radius': 3,
                'n_points': 24,
                'method': 'uniform'
            },
            
            # Parámetros Gabor
            'gabor': {
                'frequencies': [0.1, 0.3, 0.5],
                'angles': [0, 45, 90, 135]
            },
            
            # Parámetros CMC (Song et al. 2015)
            'cmc': {
                'cell_size': 16,
                'overlap_ratio': 0.5,
                'correlation_threshold': 0.8
            }
        }
        
        # Verificar disponibilidad de recursos para paralelización
        self._check_system_resources()
        
        # Estadísticas de rendimiento
        self.performance_stats = {
            'sequential_time': 0.0,
            'parallel_time': 0.0,
            'speedup_factor': 1.0,
            'memory_usage_mb': 0.0
        }
    
    def _check_system_resources(self):
        """Verifica recursos del sistema y ajusta configuración automáticamente"""
        try:
            # Usar configuración unificada
            try:
                from config.unified_config import get_image_processing_config
                
                # Aplicar configuración unificada
                img_config = get_image_processing_config()
                self.parallel_config.max_workers_process = img_config.max_workers
                self.parallel_config.max_workers_thread = img_config.max_workers
                self.parallel_config.memory_limit_gb = img_config.memory_limit_mb / 1024.0
                self.parallel_config.chunk_size = getattr(img_config, 'chunk_size', self.parallel_config.chunk_size)
                
                self.logger.info("Configuración optimizada aplicada desde config/parallel_config_optimized.py")
                
            except ImportError:
                self.logger.warning("No se pudo cargar configuración optimizada, usando detección automática")
                
                # Verificar memoria disponible (método original)
                memory_info = psutil.virtual_memory()
                memory_gb = memory_info.available / (1024**3)
                total_memory_gb = memory_info.total / (1024**3)
                
                self.logger.info(f"Memoria total: {total_memory_gb:.1f}GB, disponible: {memory_gb:.1f}GB")
                
                # Ajustar configuración según memoria disponible
                if memory_gb < 1.0:  # Crítico: menos de 1GB disponible
                    self.logger.warning(f"Memoria crítica ({memory_gb:.1f}GB). Optimizando para uso mínimo.")
                    self.parallel_config.memory_limit_gb = min(0.3, memory_gb * 0.5)
                    self.parallel_config.max_workers_process = 1
                    self.parallel_config.max_workers_thread = 1
                    self.parallel_config.enable_gabor_parallel = False
                    self.parallel_config.enable_roi_parallel = False
                    self.parallel_config.chunk_size = 1
                    
                elif memory_gb < 2.0:  # Limitado: entre 1-2GB disponible
                    self.logger.info(f"Memoria limitada ({memory_gb:.1f}GB). Usando configuración conservadora.")
                    self.parallel_config.memory_limit_gb = min(0.6, memory_gb * 0.7)
                    self.parallel_config.max_workers_process = min(2, self.parallel_config.max_workers_process)
                    self.parallel_config.max_workers_thread = min(2, self.parallel_config.max_workers_thread)
                    
                else:  # Suficiente: más de 2GB disponible
                    self.logger.info(f"Memoria suficiente ({memory_gb:.1f}GB). Usando configuración estándar.")
            
            # Verificar número de cores
            cpu_count = mp.cpu_count()
            if cpu_count < 2:
                self.parallel_config.enable_gabor_parallel = False
                self.parallel_config.enable_roi_parallel = False
                self.logger.info("CPU single-core detectado. Paralelización deshabilitada.")
            
            self.logger.info(f"Configuración final: workers_process={self.parallel_config.max_workers_process}, "
                           f"workers_thread={self.parallel_config.max_workers_thread}, "
                           f"memory_limit={self.parallel_config.memory_limit_gb:.1f}GB")
                           
        except Exception as e:
            self.logger.error(f"Error verificando recursos del sistema: {e}")
            # Configuración de seguridad ultra-conservadora
            self.parallel_config.memory_limit_gb = 0.5
            self.parallel_config.max_workers_process = 1
            self.parallel_config.max_workers_thread = 1
            self.parallel_config.enable_gabor_parallel = False
            self.parallel_config.enable_roi_parallel = False
    
    def extract_ballistic_features(self, image: np.ndarray, 
                                 specimen_type: str = 'cartridge_case',
                                 use_parallel: bool = True) -> BallisticFeatures:
        """
        Extrae características específicas del dominio balístico
        
        Args:
            image: Imagen en escala de grises
            specimen_type: Tipo de espécimen ('cartridge_case' o 'bullet')
            use_parallel: Si usar procesamiento paralelo cuando esté disponible
            
        Returns:
            BallisticFeatures: Características extraídas
        """
        try:
            start_time = time.time()
            
            # Validar entrada y convertir a escala de grises si es necesario
            if image is None or image.size == 0:
                return self._create_empty_features()
            
            # Convertir a escala de grises si es necesario
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            elif len(image.shape) != 2:
                self.logger.error(f"Formato de imagen no válido: {image.shape}")
                return self._create_empty_features()
            
            # Normalizar imagen
            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            
            # Detectar ROI específicas
            roi_regions = self._detect_ballistic_roi(image, specimen_type, use_parallel)
            
            # Validar que roi_regions sea una lista y manejar casos de error
            if not isinstance(roi_regions, list):
                self.logger.warning(f"roi_regions no es una lista, es {type(roi_regions)}. Usando lista vacía.")
                roi_regions = []
            elif roi_regions is None:
                self.logger.warning("roi_regions es None. Usando lista vacía.")
                roi_regions = []
            
            # Validar que todos los elementos sean ROIRegion
            validated_regions = []
            for region in roi_regions:
                if isinstance(region, ROIRegion):
                    validated_regions.append(region)
                else:
                    self.logger.warning(f"Elemento no válido en roi_regions: {type(region)}")
            roi_regions = validated_regions
            
            # Extraer características por región
            if use_parallel and len(roi_regions) > 1:
                firing_pin_features = self._extract_firing_pin_features_parallel(image, roi_regions)
                breech_face_features = self._extract_breech_face_features_parallel(image, roi_regions)
                striation_features = self._extract_striation_features_parallel(image, roi_regions)
            else:
                firing_pin_features = self._extract_firing_pin_features(image, roi_regions)
                breech_face_features = self._extract_breech_face_features(image, roi_regions)
                striation_features = self._extract_striation_features(image, roi_regions)
            
            # Características globales
            hu_moments = self._calculate_hu_moments(image)
            fourier_descriptors = self._calculate_fourier_descriptors(image)
            surface_gradients = self._analyze_surface_gradients(image)
            
            # Calcular calidad y confianza
            quality_score = self._calculate_quality_score(image, roi_regions)
            confidence = self._calculate_confidence(roi_regions)
            
            # Actualizar estadísticas de rendimiento
            processing_time = time.time() - start_time
            if use_parallel:
                self.performance_stats['parallel_time'] = processing_time
            else:
                self.performance_stats['sequential_time'] = processing_time
            
            if self.performance_stats['sequential_time'] > 0:
                self.performance_stats['speedup_factor'] = (
                    self.performance_stats['sequential_time'] / 
                    max(self.performance_stats['parallel_time'], 0.001)
                )
            
            return BallisticFeatures(
                firing_pin_diameter=firing_pin_features.get('diameter', 0.0),
                firing_pin_depth=firing_pin_features.get('depth', 0.0),
                firing_pin_eccentricity=firing_pin_features.get('eccentricity', 0.0),
                firing_pin_circularity=firing_pin_features.get('circularity', 0.0),
                breech_face_roughness=breech_face_features.get('roughness', 0.0),
                breech_face_orientation=breech_face_features.get('orientation', 0.0),
                breech_face_periodicity=breech_face_features.get('periodicity', 0.0),
                breech_face_entropy=breech_face_features.get('entropy', 0.0),
                striation_density=striation_features.get('density', 0.0),
                striation_orientation=striation_features.get('orientation', 0.0),
                striation_amplitude=striation_features.get('amplitude', 0.0),
                striation_frequency=striation_features.get('frequency', 0.0),
                hu_moments=hu_moments,
                fourier_descriptors=fourier_descriptors,
                surface_gradients=surface_gradients,
                quality_score=quality_score,
                confidence=confidence
            )
            
        except Exception as e:
            self.logger.error(f"Error extrayendo características balísticas: {e}")
            return self._create_empty_features()
    
    def _detect_ballistic_roi(self, image: np.ndarray, 
                            specimen_type: str, use_parallel: bool = False) -> List[ROIRegion]:
        """
        Detecta regiones de interés específicas del dominio balístico
        Basado en Le Bouthillier (2023) y Leloglu et al. (2014)
        """
        roi_regions = []
        
        try:
            if specimen_type == 'cartridge_case':
                # Detectar marca de percutor (circular)
                firing_pin_roi = self._detect_firing_pin_roi(image)
                if firing_pin_roi:
                    roi_regions.append(firing_pin_roi)
                
                # Detectar área de culata (breech face)
                breech_face_roi = self._detect_breech_face_roi(image)
                if breech_face_roi:
                    roi_regions.append(breech_face_roi)
                
                # Detectar marcas de extractor/eyector
                extractor_rois = self._detect_extractor_marks(image)
                # Validar que extractor_rois sea una lista y manejar casos de error
                if isinstance(extractor_rois, list):
                    roi_regions.extend(extractor_rois)
                elif extractor_rois is None:
                    self.logger.warning("_detect_extractor_marks devolvió None")
                else:
                    self.logger.warning(f"_detect_extractor_marks devolvió {type(extractor_rois)} en lugar de lista")
                
            elif specimen_type == 'bullet':
                # Detectar estrías en proyectil
                striation_rois = self._detect_striation_roi(image, use_parallel)
                # Validar que striation_rois sea una lista y manejar casos de error
                if isinstance(striation_rois, list):
                    roi_regions.extend(striation_rois)
                elif striation_rois is None:
                    self.logger.warning("_detect_striation_roi devolvió None")
                else:
                    self.logger.warning(f"_detect_striation_roi devolvió {type(striation_rois)} en lugar de lista")
            
            return roi_regions
            
        except Exception as e:
            self.logger.error(f"Error detectando ROI balísticas: {e}")
            return []
    
    def _detect_firing_pin_roi(self, image: np.ndarray) -> Optional[ROIRegion]:
        """
        Detecta la marca del percutor usando Hough Transform
        Basado en Ghani et al. (2012)
        """
        try:
            # Optimización: Reducir resolución para detección inicial
            h, w = image.shape
            if h > 800 or w > 800:
                scale = min(800/h, 800/w)
                resized_img = cv2.resize(image, None, fx=scale, fy=scale)
            else:
                resized_img = image
                scale = 1.0
            
            # Aplicar filtro Gaussiano con kernel más pequeño
            blurred = cv2.GaussianBlur(resized_img, (5, 5), 1)
            
            # Detectar círculos usando HoughCircles
            circles = cv2.HoughCircles(
                blurred,
                cv2.HOUGH_GRADIENT,
                dp=2,  # Menor resolución del acumulador
                minDist=30,
                param1=50,
                param2=25,  # Umbral más bajo
                minRadius=int(5/scale) if scale < 1 else 5,
                maxRadius=int(50/scale) if scale < 1 else 50
            )
            
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                
                # Escalar coordenadas de vuelta si se redimensionó
                if scale < 1.0:
                    circles[:, :2] = (circles[:, :2] / scale).astype(int)
                    circles[:, 2] = (circles[:, 2] / scale).astype(int)
                
                # Seleccionar el círculo más prominente
                best_circle = None
                best_score = 0
                
                for (x, y, r) in circles:
                    # Calcular score basado en intensidad y circularidad
                    mask = np.zeros(image.shape, dtype=np.uint8)
                    cv2.circle(mask, (x, y), r, 255, -1)
                    
                    roi_pixels = image[mask > 0]
                    intensity_score = np.std(roi_pixels)  # Variación de intensidad
                    
                    if intensity_score > best_score:
                        best_score = intensity_score
                        best_circle = (x, y, r)
                
                if best_circle:
                    x, y, r = best_circle
                    
                    # Crear máscara
                    mask = np.zeros(image.shape, dtype=np.uint8)
                    cv2.circle(mask, (x, y), r, 255, -1)
                    
                    # Calcular características
                    features = self._analyze_circular_region(image, (x, y), r)
                    
                    return ROIRegion(
                        center=(x, y),
                        radius=float(r),
                        area=float(np.pi * r * r),
                        region_type='firing_pin',
                        confidence=min(best_score / 100.0, 1.0),
                        mask=mask,
                        features=features
                    )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error detectando ROI de percutor: {e}")
            return None
    
    def _detect_breech_face_roi(self, image: np.ndarray) -> Optional[ROIRegion]:
        """
        Detecta el área de la culata usando segmentación por textura
        Basado en Chen & Chu (2018)
        """
        try:
            # Usar parámetros LBP más eficientes
            lbp = local_binary_pattern(
                image, 
                8,  # Menos puntos para mayor velocidad
                1,  # Radio menor
                method='uniform'  # Método más rápido
            )
            
            # Segmentación por watershed
            # Calcular gradiente
            gradient = cv2.Sobel(image, cv2.CV_64F, 1, 1, ksize=3)
            gradient = np.abs(gradient)
            
            # Encontrar marcadores
            markers = np.zeros(image.shape, dtype=np.int32)
            
            # Marcador de fondo (bordes)
            markers[0:5, :] = 1
            markers[-5:, :] = 1
            markers[:, 0:5] = 1
            markers[:, -5:] = 1
            
            # Marcador de primer plano (centro con alta textura)
            center_region = lbp[image.shape[0]//4:3*image.shape[0]//4, 
                               image.shape[1]//4:3*image.shape[1]//4]
            threshold = np.percentile(center_region, 75)
            
            center_mask = np.zeros(image.shape, dtype=bool)
            center_mask[image.shape[0]//4:3*image.shape[0]//4, 
                       image.shape[1]//4:3*image.shape[1]//4] = lbp[image.shape[0]//4:3*image.shape[0]//4, 
                                                                    image.shape[1]//4:3*image.shape[1]//4] > threshold
            
            markers[center_mask] = 2
            
            # Aplicar watershed
            labels = watershed(gradient, markers)
            
            # Extraer región de culata (label 2)
            breech_mask = (labels == 2).astype(np.uint8) * 255
            
            if np.sum(breech_mask) > self.config['roi']['min_area']:
                # Encontrar contorno principal
                contours, _ = cv2.findContours(breech_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    
                    # Calcular propiedades
                    M = cv2.moments(largest_contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        area = cv2.contourArea(largest_contour)
                        
                        # Calcular características de textura
                        features = self._analyze_texture_region(image, breech_mask)
                        
                        return ROIRegion(
                            center=(cx, cy),
                            radius=np.sqrt(area / np.pi),
                            area=float(area),
                            region_type='breech_face',
                            confidence=0.8,  # Confianza base para breech face
                            mask=breech_mask,
                            features=features
                        )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error detectando ROI de culata: {e}")
            return None
    
    def _detect_extractor_marks(self, image: np.ndarray) -> List[ROIRegion]:
        """
        Detecta marcas de extractor/eyector usando análisis de bordes
        """
        try:
            roi_regions = []
            
            # Detección de bordes
            edges = cv2.Canny(image, 50, 150)
            
            # Morfología para conectar bordes
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
            
            # Encontrar contornos
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # Filtrar por área
                if self.config['roi']['min_area'] < area < self.config['roi']['max_area']:
                    # Calcular propiedades
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        
                        # Crear máscara
                        mask = np.zeros(image.shape, dtype=np.uint8)
                        cv2.fillPoly(mask, [contour], 255)
                        
                        # Calcular características
                        features = self._analyze_texture_region(image, mask)
                        
                        roi_regions.append(ROIRegion(
                            center=(cx, cy),
                            radius=np.sqrt(area / np.pi),
                            area=float(area),
                            region_type='extractor',
                            confidence=0.6,  # Confianza moderada para marcas de extractor
                            mask=mask,
                            features=features
                        ))
            
            return roi_regions
            
        except Exception as e:
            self.logger.error(f"Error detectando marcas de extractor: {e}")
            return []
    
    def _detect_striation_roi(self, image: np.ndarray, use_parallel: bool = False) -> List[ROIRegion]:
        """
        Detecta regiones con estrías usando filtros Gabor
        Versión con soporte para paralelización
        """
        try:
            roi_regions = []
            
            if not PROCESSING_AVAILABLE:
                self.logger.warning("Librerías de procesamiento no disponibles")
                return roi_regions
            
            # Preparar parámetros para filtros Gabor
            gabor_params = []
            angles = self.config['gabor']['angles']
            frequencies = self.config['gabor']['frequencies']
            
            # Reducir parámetros en sistemas con poca memoria si es necesario
            if use_parallel:
                memory_gb = psutil.virtual_memory().available / (1024**3)
                if memory_gb < 2.5:
                    angles = angles[::2]  # Usar cada segundo ángulo
                    frequencies = frequencies[:2]  # Usar solo las primeras 2 frecuencias
                    self.logger.info(f"Parámetros Gabor reducidos por memoria limitada: {len(angles)} ángulos, {len(frequencies)} frecuencias")
            
            for angle in angles:
                for freq in frequencies:
                    gabor_params.append((image, freq, angle))
            
            # Procesamiento paralelo de filtros Gabor si está habilitado
            if use_parallel and self.parallel_config.enable_gabor_parallel and len(gabor_params) > 2:
                max_workers = min(self.parallel_config.max_workers_process, len(gabor_params))
                
                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    self.logger.info(f"Procesando {len(gabor_params)} filtros Gabor en paralelo con {max_workers} workers")
                    responses = list(executor.map(process_gabor_filter, gabor_params))
            else:
                # Fallback secuencial
                responses = []
                for params in gabor_params:
                    responses.append(process_gabor_filter(params))
            
            # Combinar respuestas de Gabor
            if responses:
                combined_response = np.zeros_like(image, dtype=np.float64)
                for response in responses:
                    combined_response += np.abs(response)
                
                combined_response = combined_response / len(responses)
                
                # Umbralizar para encontrar regiones con estrías
                threshold = np.percentile(combined_response, 85)
                striation_mask = combined_response > threshold
                
                # Morfología para limpiar
                kernel = disk(3)
                striation_mask = opening(striation_mask, kernel)
                striation_mask = closing(striation_mask, kernel)
                
                # Encontrar componentes conectados
                labeled_regions = ndimage.label(striation_mask)[0]
                
                # Verificar que hay regiones etiquetadas
                max_label = np.max(labeled_regions) if labeled_regions.size > 0 else 0
                if max_label == 0:
                    self.logger.warning("No se encontraron componentes conectados en la detección de estrías")
                    return roi_regions
                
                for region_id in range(1, max_label + 1):
                    region_mask = (labeled_regions == region_id).astype(np.uint8) * 255
                    area = np.sum(region_mask > 0)
                    
                    if area > self.config['roi']['min_area']:
                        # Calcular centro de masa
                        y_coords, x_coords = np.where(region_mask > 0)
                        cx = int(np.mean(x_coords))
                        cy = int(np.mean(y_coords))
                        
                        # Analizar patrones de estrías
                        if use_parallel:
                            features = _analyze_striation_patterns_parallel(image, region_mask)
                        else:
                            features = self._analyze_striation_patterns(image, region_mask)
                        
                        roi_regions.append(ROIRegion(
                            center=(cx, cy),
                            radius=np.sqrt(area / np.pi),
                            area=float(area),
                            region_type='striation',
                            confidence=0.7,
                            mask=region_mask,
                            features=features
                        ))
            
            return roi_regions
            
        except Exception as e:
            self.logger.error(f"Error detectando ROI de estrías: {e}")
            return []

    def _extract_firing_pin_features(self, image: np.ndarray, 
                                   roi_regions: List[ROIRegion]) -> Dict[str, float]:
        """Extrae características específicas del percutor"""
        features = {'diameter': 0.0, 'depth': 0.0, 'eccentricity': 0.0, 'circularity': 0.0}
        
        firing_pin_regions = [r for r in roi_regions if r.region_type == 'firing_pin']
        
        if firing_pin_regions:
            region = firing_pin_regions[0]  # Tomar la primera (mejor)
            
            features['diameter'] = region.radius * 2
            
            # Calcular profundidad basada en intensidad
            roi_pixels = image[region.mask > 0]
            if len(roi_pixels) > 0:
                features['depth'] = float(np.mean(roi_pixels))
            
            # Usar características pre-calculadas
            features.update(region.features)
        
        return features
    
    def _extract_breech_face_features(self, image: np.ndarray, 
                                    roi_regions: List[ROIRegion]) -> Dict[str, float]:
        """Extrae características de la culata"""
        features = {'roughness': 0.0, 'orientation': 0.0, 'periodicity': 0.0, 'entropy': 0.0}
        
        breech_regions = [r for r in roi_regions if r.region_type == 'breech_face']
        
        if breech_regions:
            region = breech_regions[0]
            features.update(region.features)
        
        return features
    
    def _extract_striation_features(self, image: np.ndarray, 
                                  roi_regions: List[ROIRegion]) -> Dict[str, float]:
        """Extrae características de estrías"""
        features = {'density': 0.0, 'orientation': 0.0, 'amplitude': 0.0, 'frequency': 0.0}
        
        striation_regions = [r for r in roi_regions if r.region_type == 'striation']
        
        if striation_regions:
            # Combinar características de todas las regiones de estrías
            all_features = []
            for region in striation_regions:
                all_features.append(region.features)
            
            if all_features:
                # Promediar características
                for key in features.keys():
                    values = [f.get(key, 0.0) for f in all_features]
                    features[key] = float(np.mean(values))
        
        return features

    def _extract_firing_pin_features_parallel(self, image: np.ndarray, 
                                            roi_regions: List[ROIRegion]) -> Dict[str, float]:
        """Extrae características del percutor usando procesamiento paralelo"""
        try:
            from concurrent.futures import ThreadPoolExecutor
            
            features = {'diameter': 0.0, 'depth': 0.0, 'eccentricity': 0.0, 'circularity': 0.0}
            
            firing_pin_regions = [r for r in roi_regions if r.region_type == 'firing_pin']
            
            if firing_pin_regions:
                with ThreadPoolExecutor(max_workers=self.parallel_config.max_workers_thread) as executor:
                    futures = []
                    for region in firing_pin_regions:
                        future = executor.submit(self._analyze_circular_region, image, region.center, int(region.radius))
                        futures.append(future)
                    
                    all_features = []
                    for future in futures:
                        try:
                            result = future.result(timeout=30)
                            all_features.append(result)
                        except Exception as e:
                            logger.warning(f"Error en análisis paralelo de percutor: {e}")
                            continue
                
                if all_features:
                    # Promediar características
                    for key in features.keys():
                        values = [f.get(key, 0.0) for f in all_features]
                        features[key] = float(np.mean(values))
            
            return features
        except Exception as e:
            logger.warning(f"Error en extracción paralela de percutor: {e}")
            return self._extract_firing_pin_features(image, roi_regions)

    def _extract_breech_face_features_parallel(self, image: np.ndarray, 
                                             roi_regions: List[ROIRegion]) -> Dict[str, float]:
        """Extrae características de la culata usando procesamiento paralelo"""
        try:
            from concurrent.futures import ThreadPoolExecutor
            
            features = {'roughness': 0.0, 'orientation': 0.0, 'periodicity': 0.0, 'entropy': 0.0}
            
            breech_regions = [r for r in roi_regions if r.region_type == 'breech_face']
            
            if breech_regions:
                with ThreadPoolExecutor(max_workers=self.parallel_config.max_workers_thread) as executor:
                    futures = []
                    for region in breech_regions:
                        future = executor.submit(self._analyze_texture_region, image, region.mask)
                        futures.append(future)
                    
                    all_features = []
                    for future in futures:
                        try:
                            result = future.result(timeout=30)
                            all_features.append(result)
                        except Exception as e:
                            logger.warning(f"Error en análisis paralelo de culata: {e}")
                            continue
                
                if all_features:
                    # Promediar características
                    for key in features.keys():
                        values = [f.get(key, 0.0) for f in all_features]
                        features[key] = float(np.mean(values))
            
            return features
        except Exception as e:
            logger.warning(f"Error en extracción paralela de culata: {e}")
            return self._extract_breech_face_features(image, roi_regions)

    def _extract_striation_features_parallel(self, image: np.ndarray, 
                                           roi_regions: List[ROIRegion]) -> Dict[str, float]:
        """Extrae características de estrías usando procesamiento paralelo"""
        try:
            from concurrent.futures import ThreadPoolExecutor
            
            features = {'density': 0.0, 'orientation': 0.0, 'amplitude': 0.0, 'frequency': 0.0}
            
            striation_regions = [r for r in roi_regions if r.region_type == 'striation']
            
            if striation_regions:
                with ThreadPoolExecutor(max_workers=self.parallel_config.max_workers_thread) as executor:
                    futures = []
                    for region in striation_regions:
                        future = executor.submit(self._analyze_striation_patterns, image, region.mask)
                        futures.append(future)
                    
                    all_features = []
                    for future in futures:
                        try:
                            result = future.result(timeout=30)
                            all_features.append(result)
                        except Exception as e:
                            logger.warning(f"Error en análisis paralelo de estrías: {e}")
                            continue
                
                if all_features:
                    # Promediar características
                    for key in features.keys():
                        values = [f.get(key, 0.0) for f in all_features]
                        features[key] = float(np.mean(values))
            
            return features
        except Exception as e:
            logger.warning(f"Error en extracción paralela de estrías: {e}")
            return self._extract_striation_features(image, roi_regions)

    def _analyze_circular_region(self, image: np.ndarray, 
                               center: Tuple[int, int], radius: int) -> Dict[str, float]:
        """Analiza una región circular (percutor)"""
        x, y = center
        
        # Crear máscara circular
        mask = np.zeros(image.shape, dtype=np.uint8)
        cv2.circle(mask, (x, y), radius, 255, -1)
        
        # Extraer píxeles de la región
        roi_pixels = image[mask > 0]
        
        if len(roi_pixels) == 0:
            return {'circularity': 0.0, 'intensity_mean': 0.0, 'intensity_std': 0.0}
        
        # Calcular características
        intensity_mean = float(np.mean(roi_pixels))
        intensity_std = float(np.std(roi_pixels))
        
        # Calcular circularidad usando contorno
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        circularity = 0.0
        
        if contours:
            contour = contours[0]
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        return {
            'circularity': float(circularity),
            'intensity_mean': intensity_mean,
            'intensity_std': intensity_std
        }
    
    def _analyze_texture_region(self, image: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
        """Analiza características de textura en una región"""
        roi_pixels = image[mask > 0]
        
        if len(roi_pixels) == 0:
            return {'roughness': 0.0, 'entropy': 0.0, 'orientation': 0.0}
        
        # Rugosidad (desviación estándar)
        roughness = float(np.std(roi_pixels))
        
        # Entropía
        entropy = shannon_entropy(roi_pixels)
        
        # Orientación dominante usando gradientes
        roi_region = image.copy()
        roi_region[mask == 0] = 0
        
        grad_x = cv2.Sobel(roi_region, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(roi_region, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calcular ángulos de gradiente
        angles = np.arctan2(grad_y, grad_x)
        angles = angles[mask > 0]
        
        if len(angles) > 0:
            # Orientación dominante (moda de los ángulos)
            hist, bins = np.histogram(angles, bins=36, range=(-np.pi, np.pi))
            dominant_bin = np.argmax(hist)
            orientation = float(bins[dominant_bin])
        else:
            orientation = 0.0
        
        return {
            'roughness': roughness,
            'entropy': float(entropy),
            'orientation': orientation
        }
    
    def _analyze_striation_patterns(self, image: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
        """Analiza patrones de estrías usando análisis híbrido mejorado"""
        roi_region = image.copy()
        roi_region[mask == 0] = 0
        
        # Extraer región rectangular que contiene la máscara
        coords = np.where(mask > 0)
        if len(coords[0]) == 0:
            return {'density': 0.0, 'orientation': 0.0, 'amplitude': 0.0, 'frequency': 0.0,
                   'num_striation_lines': 0, 'dominant_directions': [], 'parallelism_score': 0.0}
        
        y_min, y_max = np.min(coords[0]), np.max(coords[0])
        x_min, x_max = np.min(coords[1]), np.max(coords[1])
        roi_rect = roi_region[y_min:y_max+1, x_min:x_max+1]
        
        if roi_rect.size == 0:
            return {'density': 0.0, 'orientation': 0.0, 'amplitude': 0.0, 'frequency': 0.0,
                   'num_striation_lines': 0, 'dominant_directions': [], 'parallelism_score': 0.0}
        
        # === ANÁLISIS MEJORADO CON GRADIENTES DIRECCIONALES ===
        
        # 1. Aplicar filtro de mediana para reducir ruido
        denoised = cv2.medianBlur(roi_rect, 3)
        
        # 2. Calcular gradientes en X e Y
        grad_x = cv2.Sobel(denoised, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(denoised, cv2.CV_64F, 0, 1, ksize=3)
        
        # 3. Calcular magnitud y dirección del gradiente
        magnitude_grad = np.sqrt(grad_x**2 + grad_y**2)
        direction = np.arctan2(grad_y, grad_x)
        direction = np.abs(direction)  # Normalizar a [0, π]
        
        # 4. Crear histograma de direcciones para detectar patrones dominantes
        hist, bins = np.histogram(direction.flatten(), bins=18, range=(0, np.pi))
        
        # 5. Encontrar direcciones dominantes
        dominant_directions = []
        threshold = np.max(hist) * 0.3  # 30% del pico máximo
        
        for i, count in enumerate(hist):
            if count > threshold:
                angle = (bins[i] + bins[i+1]) / 2
                dominant_directions.append(float(angle * 180 / np.pi))
        
        # 6. Detectar líneas usando transformada de Hough
        edges = cv2.Canny(denoised, 50, 150)
        lines = cv2.HoughLinesP(
            edges,
            rho=2,
            theta=np.pi/90,
            threshold=30,
            minLineLength=20,
            maxLineGap=15
        )
        
        num_striation_lines = len(lines) if lines is not None else 0
        
        # 7. Calcular score de paralelismo
        parallelism_score = 0.0
        if lines is not None and len(lines) > 1:
            line_angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.arctan2(y2-y1, x2-x1) * 180 / np.pi
                line_angles.append(angle)
            
            # Calcular desviación estándar de ángulos (menor = más paralelas)
            if len(line_angles) > 1:
                angle_std = np.std(line_angles)
                parallelism_score = max(0.0, 1.0 - (angle_std / 45.0))  # Normalizar
        
        # === ANÁLISIS ESPECTRAL FFT (ORIGINAL) ===
        
        # FFT para análisis de frecuencia
        fft = fft2(roi_rect)
        fft_shift = fftshift(fft)
        magnitude_fft = np.abs(fft_shift)
        
        # Características espectrales
        density = float(np.mean(magnitude_fft))
        amplitude = float(np.max(magnitude_fft))
        
        # Frecuencia dominante
        center_y, center_x = np.array(magnitude_fft.shape) // 2
        y, x = np.unravel_index(np.argmax(magnitude_fft), magnitude_fft.shape)
        frequency = float(np.sqrt((y - center_y)**2 + (x - center_x)**2))
        
        # Orientación basada en la posición del pico FFT
        orientation_fft = float(np.arctan2(y - center_y, x - center_x))
        
        # Combinar orientación de gradientes y FFT
        if dominant_directions:
            orientation_combined = float(np.mean(dominant_directions) * np.pi / 180)
        else:
            orientation_combined = orientation_fft
        
        return {
            'density': density,
            'orientation': orientation_combined,
            'amplitude': amplitude,
            'frequency': frequency,
            'num_striation_lines': num_striation_lines,
            'dominant_directions': dominant_directions,
            'parallelism_score': parallelism_score
        }
    
    def _calculate_hu_moments(self, image: np.ndarray) -> List[float]:
        """Calcula momentos de Hu invariantes"""
        try:
            moments = cv2.moments(image)
            hu_moments = cv2.HuMoments(moments)
            
            # Aplicar transformación logarítmica para estabilidad numérica
            hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
            
            return hu_moments.flatten().tolist()
            
        except Exception as e:
            self.logger.error(f"Error calculando momentos de Hu: {e}")
            return [0.0] * 7
    
    def _calculate_fourier_descriptors(self, image: np.ndarray, n_descriptors: int = 10) -> List[float]:
        """Calcula descriptores de Fourier del contorno principal"""
        try:
            # Encontrar contorno principal
            contours, _ = cv2.findContours(
                (image > np.mean(image)).astype(np.uint8), 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            if not contours:
                return [0.0] * n_descriptors
            
            # Seleccionar el contorno más grande
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Convertir contorno a números complejos
            contour_points = largest_contour.reshape(-1, 2)
            complex_contour = contour_points[:, 0] + 1j * contour_points[:, 1]
            
            # Calcular FFT del contorno
            fft_contour = np.fft.fft(complex_contour)
            
            # Tomar los primeros n descriptores (excluyendo DC)
            descriptors = np.abs(fft_contour[1:n_descriptors+1])
            
            # Normalizar por el primer descriptor para invariancia de escala
            if descriptors[0] != 0:
                descriptors = descriptors / descriptors[0]
            
            return descriptors.tolist()
            
        except Exception as e:
            self.logger.error(f"Error calculando descriptores de Fourier: {e}")
            return [0.0] * n_descriptors

    def _analyze_surface_gradients(self, image: np.ndarray) -> Dict[str, float]:
        """Analiza gradientes de superficie"""
        try:
            # Calcular gradientes
            grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
            
            # Magnitud y dirección del gradiente
            magnitude = np.sqrt(grad_x**2 + grad_y**2)
            direction = np.arctan2(grad_y, grad_x)
            
            return {
                'gradient_mean': float(np.mean(magnitude)),
                'gradient_std': float(np.std(magnitude)),
                'gradient_max': float(np.max(magnitude)),
                'direction_mean': float(np.mean(direction)),
                'direction_std': float(np.std(direction))
            }
            
        except Exception as e:
            self.logger.error(f"Error analizando gradientes: {e}")
            return {
                'gradient_mean': 0.0,
                'gradient_std': 0.0,
                'gradient_max': 0.0,
                'direction_mean': 0.0,
                'direction_std': 0.0
            }
    
    def _calculate_quality_score(self, image: np.ndarray, roi_regions: List[ROIRegion]) -> float:
        """Calcula score de calidad basado en estándares NIST"""
        try:
            # Métricas de calidad de imagen
            
            # 1. Contraste
            contrast = float(np.std(image) / np.mean(image)) if np.mean(image) > 0 else 0.0
            contrast_score = min(contrast / 0.3, 1.0)  # NIST: contraste > 0.3
            
            # 2. SNR estimado
            signal = np.mean(image)
            noise = np.std(image - cv2.GaussianBlur(image, (5, 5), 1))
            snr = signal / noise if noise > 0 else 0.0
            snr_score = min(snr / 20.0, 1.0)  # NIST: SNR > 20 dB
            
            # 3. Uniformidad de iluminación
            mean_regions = []
            h, w = image.shape
            for i in range(0, h, h//4):
                for j in range(0, w, w//4):
                    region = image[i:i+h//4, j:j+w//4]
                    if region.size > 0:
                        mean_regions.append(np.mean(region))
            
            if mean_regions:
                illumination_uniformity = 1.0 - (np.std(mean_regions) / np.mean(mean_regions))
                uniformity_score = max(0.0, illumination_uniformity)
            else:
                uniformity_score = 0.0
            
            # 4. Calidad de ROI detectadas
            roi_score = 0.0
            if roi_regions:
                roi_confidences = [r.confidence for r in roi_regions]
                roi_score = np.mean(roi_confidences)
            
            # Score final (promedio ponderado)
            quality_score = (
                0.3 * contrast_score +
                0.3 * snr_score +
                0.2 * uniformity_score +
                0.2 * roi_score
            )
            
            return float(quality_score)
            
        except Exception as e:
            self.logger.error(f"Error calculando score de calidad: {e}")
            return 0.0
    
    def _calculate_confidence(self, roi_regions: List[ROIRegion]) -> float:
        """Calcula confianza general basada en ROI detectadas"""
        if not roi_regions:
            return 0.0
        
        # Confianza basada en número y calidad de ROI
        confidence_scores = [r.confidence for r in roi_regions]
        
        # Bonificación por diversidad de tipos de ROI
        roi_types = set(r.region_type for r in roi_regions)
        diversity_bonus = len(roi_types) * 0.1
        
        base_confidence = np.mean(confidence_scores)
        final_confidence = min(base_confidence + diversity_bonus, 1.0)
        
        return float(final_confidence)

    def _analyze_breech_face_texture(self, image: np.ndarray) -> Dict[str, float]:
        """Analiza textura específica de breech face"""
        try:
            # LBP para análisis de textura
            lbp = local_binary_pattern(image, 8, 1, method='uniform')
            
            # Histograma de LBP
            hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 10))
            hist = hist.astype(float)
            hist /= (hist.sum() + 1e-7)
            
            # Entropía de textura
            entropy = -np.sum(hist * np.log2(hist + 1e-7))
            
            return {
                'lbp_entropy': float(entropy),
                'lbp_uniformity': float(hist.max()),
                'texture_contrast': float(np.std(lbp))
            }
            
        except Exception as e:
            self.logger.error(f"Error analizando textura breech face: {e}")
            return {
                'lbp_entropy': 0.0,
                'lbp_uniformity': 0.0,
                'texture_contrast': 0.0
            }

    def _analyze_firing_pin_geometry(self, image: np.ndarray, roi_regions: List[ROIRegion]) -> Dict[str, float]:
        """Analiza geometría específica del firing pin"""
        firing_pin_regions = [r for r in roi_regions if r.region_type == 'firing_pin']
        
        if not firing_pin_regions:
            return {
                'aspect_ratio': 0.0,
                'solidity': 0.0,
                'extent': 0.0
            }
        
        try:
            region = firing_pin_regions[0]
            
            # Encontrar contorno de la región
            contours, _ = cv2.findContours(
                region.mask.astype(np.uint8), 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            if contours:
                contour = max(contours, key=cv2.contourArea)
                
                # Calcular características geométricas
                area = cv2.contourArea(contour)
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                
                # Bounding rectangle
                rect = cv2.boundingRect(contour)
                rect_area = rect[2] * rect[3]
                
                # Métricas
                solidity = area / hull_area if hull_area > 0 else 0.0
                extent = area / rect_area if rect_area > 0 else 0.0
                aspect_ratio = rect[2] / rect[3] if rect[3] > 0 else 0.0
                
                return {
                    'aspect_ratio': float(aspect_ratio),
                    'solidity': float(solidity),
                    'extent': float(extent)
                }
            
        except Exception as e:
            self.logger.error(f"Error analizando geometría firing pin: {e}")
        
        return {
            'aspect_ratio': 0.0,
            'solidity': 0.0,
            'extent': 0.0
        }

    def _calculate_image_quality_metrics(self, image: np.ndarray) -> Dict[str, float]:
        """Calcula métricas de calidad de imagen según estándares forenses"""
        try:
            # Métricas básicas
            mean_intensity = float(np.mean(image))
            std_intensity = float(np.std(image))
            
            # Contraste RMS
            contrast_rms = std_intensity / mean_intensity if mean_intensity > 0 else 0.0
            
            # Sharpness usando Laplaciano
            laplacian = cv2.Laplacian(image, cv2.CV_64F)
            sharpness = float(np.var(laplacian))
            
            # SNR estimado
            signal = mean_intensity
            noise_estimate = np.std(image - cv2.GaussianBlur(image, (3, 3), 1))
            snr = signal / noise_estimate if noise_estimate > 0 else 0.0
            
            return {
                'mean_intensity': mean_intensity,
                'std_intensity': std_intensity,
                'contrast_rms': float(contrast_rms),
                'sharpness': sharpness,
                'snr_estimate': float(snr)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculando métricas de calidad: {e}")
            return {
                'mean_intensity': 0.0,
                'std_intensity': 0.0,
                'contrast_rms': 0.0,
                'sharpness': 0.0,
                'snr_estimate': 0.0
            }

    def _create_empty_features(self) -> BallisticFeatures:
        """Crea estructura de características vacía"""
        return BallisticFeatures(
            firing_pin_diameter=0.0,
            firing_pin_depth=0.0,
            firing_pin_eccentricity=0.0,
            firing_pin_circularity=0.0,
            breech_face_roughness=0.0,
            breech_face_orientation=0.0,
            breech_face_periodicity=0.0,
            breech_face_entropy=0.0,
            striation_density=0.0,
            striation_orientation=0.0,
            striation_amplitude=0.0,
            striation_frequency=0.0,
            striation_num_lines=0,
            striation_dominant_directions=[],
            striation_parallelism_score=0.0,
            hu_moments=[0.0] * 7,
            fourier_descriptors=[0.0] * 10,
            surface_gradients={},
            quality_score=0.0,
            confidence=0.0
        )

    def get_performance_stats(self) -> Dict[str, float]:
        """Obtiene estadísticas de rendimiento del procesamiento"""
        return self.performance_stats.copy()
    
    def benchmark_performance(self, image: np.ndarray, specimen_type: str = 'cartridge_case', 
                            iterations: int = 3) -> Dict[str, Any]:
        """
        Realiza benchmark de rendimiento comparando procesamiento secuencial vs paralelo
        
        Args:
            image: Imagen a procesar
            specimen_type: Tipo de espécimen
            iterations: Número de iteraciones para el benchmark
            
        Returns:
            Diccionario con estadísticas de benchmark
        """
        results = {
            'sequential_times': [],
            'parallel_times': [],
            'memory_usage': [],
            'speedup_factors': []
        }
        
        for i in range(iterations):
            self.logger.info(f"Benchmark iteración {i+1}/{iterations}")
            
            # Benchmark secuencial
            start_time = time.time()
            _ = self.extract_ballistic_features(image, specimen_type, use_parallel=False)
            sequential_time = time.time() - start_time
            results['sequential_times'].append(sequential_time)
            
            # Benchmark paralelo
            start_time = time.time()
            _ = self.extract_ballistic_features(image, specimen_type, use_parallel=True)
            parallel_time = time.time() - start_time
            results['parallel_times'].append(parallel_time)
            
            # Memoria
            memory_mb = psutil.Process().memory_info().rss / (1024 * 1024)
            results['memory_usage'].append(memory_mb)
            
            # Factor de aceleración
            speedup = sequential_time / parallel_time if parallel_time > 0 else 1.0
            results['speedup_factors'].append(speedup)
        
        # Calcular estadísticas
        avg_sequential = np.mean(results['sequential_times'])
        avg_parallel = np.mean(results['parallel_times'])
        avg_speedup = np.mean(results['speedup_factors'])
        avg_memory = np.mean(results['memory_usage'])
        
        return {
            'iterations': iterations,
            'average_sequential_time': avg_sequential,
            'average_parallel_time': avg_parallel,
            'average_speedup_factor': avg_speedup,
            'average_memory_usage_mb': avg_memory,
            'detailed_results': results,
            'parallel_config': {
                'max_workers_process': self.parallel_config.max_workers_process,
                'max_workers_thread': self.parallel_config.max_workers_thread,
                'enable_gabor_parallel': self.parallel_config.enable_gabor_parallel,
                'enable_roi_parallel': self.parallel_config.enable_roi_parallel
            }
        }

    def extract_breech_face_features(self, image: np.ndarray) -> Dict[str, Any]:
        """Extrae características específicas de la cara de culata (breech face)"""
        try:
            # Convertir a escala de grises si es necesario
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Detectar ROI específicas
            roi_regions = self._detect_ballistic_roi(gray, 'cartridge_case')
            
            # Extraer características de breech face
            breech_face_features = self._extract_breech_face_features(gray, roi_regions)
            
            # Agregar análisis adicional específico para breech face
            texture_analysis = self._analyze_breech_face_texture(gray)
            surface_analysis = self._analyze_surface_gradients(gray)
            
            return {
                'roughness': breech_face_features.get('roughness', 0.0),
                'orientation': breech_face_features.get('orientation', 0.0),
                'periodicity': breech_face_features.get('periodicity', 0.0),
                'entropy': breech_face_features.get('entropy', 0.0),
                'texture_analysis': texture_analysis,
                'surface_analysis': surface_analysis,
                'roi_count': len([r for r in roi_regions if r.region_type == 'breech_face']),
                'processing_success': True
            }
            
        except Exception as e:
            self.logger.error(f"Error extrayendo características de breech face: {e}")
            return {
                'roughness': 0.0,
                'orientation': 0.0,
                'periodicity': 0.0,
                'entropy': 0.0,
                'texture_analysis': {},
                'surface_analysis': {},
                'roi_count': 0,
                'processing_success': False,
                'error': str(e)
            }

    def extract_firing_pin_features(self, image: np.ndarray) -> Dict[str, Any]:
        """Extrae características específicas del percutor (firing pin)"""
        try:
            # Convertir a escala de grises si es necesario
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Detectar ROI específicas
            roi_regions = self._detect_ballistic_roi(gray, 'cartridge_case')
            
            # Extraer características de firing pin
            firing_pin_features = self._extract_firing_pin_features(gray, roi_regions)
            
            # Análisis adicional de forma y geometría
            geometric_analysis = self._analyze_firing_pin_geometry(gray, roi_regions)
            
            return {
                'diameter': firing_pin_features.get('diameter', 0.0),
                'depth': firing_pin_features.get('depth', 0.0),
                'eccentricity': firing_pin_features.get('eccentricity', 0.0),
                'circularity': firing_pin_features.get('circularity', 0.0),
                'geometric_analysis': geometric_analysis,
                'roi_count': len([r for r in roi_regions if r.region_type == 'firing_pin']),
                'processing_success': True
            }
            
        except Exception as e:
            self.logger.error(f"Error extrayendo características de firing pin: {e}")
            return {
                'diameter': 0.0,
                'depth': 0.0,
                'eccentricity': 0.0,
                'circularity': 0.0,
                'geometric_analysis': {},
                'roi_count': 0,
                'processing_success': False,
                'error': str(e)
            }

    def extract_all_features(self, image: np.ndarray, specimen_type: str = 'cartridge_case') -> Dict[str, Any]:
        """Extrae todas las características balísticas disponibles"""
        try:
            # Convertir a escala de grises si es necesario
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Extraer características completas usando el método principal
            ballistic_features = self.extract_ballistic_features(gray, specimen_type)
            
            # Extraer características específicas adicionales
            breech_face_specific = self.extract_breech_face_features(image)
            firing_pin_specific = self.extract_firing_pin_features(image)
            
            # Análisis de calidad de imagen
            quality_metrics = self._calculate_image_quality_metrics(gray)
            
            return {
                'ballistic_features': {
                    'firing_pin': {
                        'diameter': ballistic_features.firing_pin_diameter,
                        'depth': ballistic_features.firing_pin_depth,
                        'eccentricity': ballistic_features.firing_pin_eccentricity,
                        'circularity': ballistic_features.firing_pin_circularity
                    },
                    'breech_face': {
                        'roughness': ballistic_features.breech_face_roughness,
                        'orientation': ballistic_features.breech_face_orientation,
                        'periodicity': ballistic_features.breech_face_periodicity,
                        'entropy': ballistic_features.breech_face_entropy
                    },
                    'striation': {
                        'density': ballistic_features.striation_density,
                        'orientation': ballistic_features.striation_orientation,
                        'amplitude': ballistic_features.striation_amplitude,
                        'frequency': ballistic_features.striation_frequency
                    },
                    'global_features': {
                        'hu_moments': ballistic_features.hu_moments,
                        'fourier_descriptors': ballistic_features.fourier_descriptors,
                        'surface_gradients': ballistic_features.surface_gradients
                    }
                },
                'specific_analysis': {
                    'breech_face_detailed': breech_face_specific,
                    'firing_pin_detailed': firing_pin_specific
                },
                'quality_metrics': {
                    'quality_score': ballistic_features.quality_score,
                    'confidence': ballistic_features.confidence,
                    'image_quality': quality_metrics
                },
                'metadata': {
                    'specimen_type': specimen_type,
                    'image_shape': gray.shape,
                    'processing_success': True
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error extrayendo todas las características: {e}")
            return {
                'ballistic_features': {},
                'specific_analysis': {},
                'quality_metrics': {},
                'metadata': {
                    'specimen_type': specimen_type,
                    'image_shape': image.shape if image is not None else (0, 0),
                    'processing_success': False,
                    'error': str(e)
                }
            }


# Función auxiliar para uso desde línea de comandos
def extract_ballistic_features_from_path(image_path: str, 
                                       specimen_type: str = 'cartridge_case') -> Dict[str, Any]:
    """
    Extrae características balísticas desde un archivo de imagen
    
    Args:
        image_path: Ruta al archivo de imagen
        specimen_type: Tipo de espécimen ('cartridge_case' o 'bullet')
        
    Returns:
        Dict con características extraídas o None si hay error
    """
    try:
        # Cargar imagen
        image = cv2.imread(image_path)
        
        if image is None:
            logger.error(f"No se pudo cargar la imagen: {image_path}")
            return None
        
        # Crear extractor y procesar
        extractor = BallisticFeatureExtractor()
        features = extractor.extract_all_features(image, specimen_type)
        
        return features
        
    except Exception as e:
        logger.error(f"Error procesando imagen {image_path}: {e}")
        return None


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        specimen_type = sys.argv[2] if len(sys.argv) > 2 else 'cartridge_case'
        
        features = extract_ballistic_features_from_path(image_path, specimen_type)
        
        if features:
            print("Características balísticas extraídas:")
            print(f"- Calidad de imagen: {features['quality_metrics']['quality_score']:.3f}")
            print(f"- Confianza: {features['quality_metrics']['confidence']:.3f}")
            print(f"- Diámetro percutor: {features['ballistic_features']['firing_pin']['diameter']:.2f}")
            print(f"- Rugosidad culata: {features['ballistic_features']['breech_face']['roughness']:.2f}")
        else:
            print("Error: No se pudieron extraer características")
    else:
        print("Uso: python ballistic_features_consolidated.py <image_path> [specimen_type]")