"""
Detector de ROI Unificado para Análisis Balístico
Sistema Balístico Forense MVP

Módulo consolidado para detección automática de regiones de interés (ROI) en imágenes balísticas
que combina las funcionalidades de roi_detector.py, enhanced_roi_detector.py y roi_detector_simple.py

Implementa técnicas avanzadas de segmentación:
- Watershed para segmentación automática
- Análisis de textura para identificación de regiones
- Detección específica de breach face y firing pin
- Validación basada en estándares forenses

Basado en:
- Le Bouthillier (2023) - ROI automática
- Chen & Chu (2018) - Segmentación de culata
- Leloglu et al. (2014) - Detección de percutor
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
import logging
from pathlib import Path
from enum import Enum
import json
import time
import os

# Importaciones opcionales con manejo de errores
try:
    from scipy import ndimage
    from scipy.spatial.distance import cdist
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Manejo de skimage con fallbacks
SKIMAGE_AVAILABLE = False
try:
    from skimage.feature import local_binary_pattern
    from skimage.filters import gaussian, sobel, threshold_otsu
    from skimage.measure import label, regionprops
    from skimage.morphology import disk, watershed, remove_small_objects
    from skimage.segmentation import clear_border
    try:
        from lbp_cache import cached_local_binary_pattern, get_lbp_cache_stats
        LBP_CACHE_AVAILABLE = True
    except ImportError:
        LBP_CACHE_AVAILABLE = False
    
    # Intentar importar peak_local_maxima (nombre correcto en versiones recientes)
    try:
        from skimage.feature import peak_local_max as peak_local_maxima
        SKIMAGE_AVAILABLE = True
    except ImportError:
        # Intentar importar peak_local_maxima (nombre alternativo)
        try:
            from skimage.feature import peak_local_maxima
            SKIMAGE_AVAILABLE = True
        except ImportError:
            # Si ninguno está disponible, crear un fallback
            if SCIPY_AVAILABLE:
                def peak_local_maxima(image, min_distance=1, threshold_abs=None, threshold_rel=None):
                    """Implementación alternativa de peak_local_maxima usando scipy"""
                    if threshold_abs is None:
                        threshold_abs = 0
                    
                    # Usar maximum_filter para encontrar máximos locales
                    local_maxima = ndimage.maximum_filter(image, size=min_distance*2+1) == image
                    local_maxima &= image > threshold_abs
                    
                    # Retornar coordenadas de los máximos
                    coords = np.where(local_maxima)
                    return np.column_stack(coords)
                
                SKIMAGE_AVAILABLE = True
            else:
                SKIMAGE_AVAILABLE = False
except ImportError:
    SKIMAGE_AVAILABLE = False
    # Fallback para peak_local_maxima si scipy está disponible
    if SCIPY_AVAILABLE:
        def peak_local_maxima(image, min_distance=1, threshold_abs=None, threshold_rel=None):
            """Implementación alternativa de peak_local_maxima usando scipy"""
            if threshold_abs is None:
                threshold_abs = 0
            
            # Usar maximum_filter para encontrar máximos locales
            local_maxima = ndimage.maximum_filter(image, size=min_distance*2+1) == image
            local_maxima &= image > threshold_abs
            
            # Retornar coordenadas de los máximos
            coords = np.where(local_maxima)
            return np.column_stack(coords)

# Verificar disponibilidad de torch (opcional)
try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from utils.logger import LoggerMixin
    LOGGER_MIXIN_AVAILABLE = True
except ImportError:
    LOGGER_MIXIN_AVAILABLE = False
    # Crear clase dummy para mantener compatibilidad
    class LoggerMixin:
        def __init__(self, *args, **kwargs): 
            self.logger = logging.getLogger(self.__class__.__name__)
        
        def log_info(self, message): 
            self.logger.info(message)
        
        def log_error(self, message): 
            self.logger.error(message)
        
        def log_warning(self, message): 
            self.logger.warning(message)

# Importar el nuevo módulo de Watershed mejorado
try:
    from enhanced_watershed_roi import (
        EnhancedWatershedROI, 
        WatershedConfig, 
        WatershedMethod,
        WatershedRegion
    )
    ENHANCED_WATERSHED_AVAILABLE = True
except ImportError:
    ENHANCED_WATERSHED_AVAILABLE = False


class DetectionLevel(Enum):
    """Niveles de detección de ROI"""
    SIMPLE = "simple"
    STANDARD = "standard"
    ADVANCED = "advanced"

@dataclass
class ROIRegion:
    """Región de interés básica"""
    # Información básica
    center: Tuple[int, int]
    radius: float
    contour: np.ndarray
    confidence: float
    region_type: str  # 'firing_pin', 'breech_face', 'primer', 'base', etc.
    bounding_box: Tuple[int, int, int, int]  # x, y, width, height
    
    # Características adicionales (opcionales)
    area: float = 0.0
    perimeter: float = 0.0
    quality_score: float = 0.0
    
    # Características geométricas (opcionales)
    circularity: float = 0.0
    eccentricity: float = 0.0
    solidity: float = 0.0
    aspect_ratio: float = 0.0
    
    # Características de textura (opcionales)
    texture_features: Dict[str, float] = field(default_factory=dict)
    
    # Máscara de la región (opcional)
    mask: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte la región a diccionario serializable"""
        # Convertir contorno a lista de coordenadas [x, y]
        contour_coords = []
        if self.contour is not None and len(self.contour) > 0:
            # Simplificar contorno para evitar demasiados puntos
            epsilon = 0.02 * cv2.arcLength(self.contour, True)
            simplified_contour = cv2.approxPolyDP(self.contour, epsilon, True)
            contour_coords = [[int(point[0][0]), int(point[0][1])] for point in simplified_contour]
        
        result = {
            'center': [int(self.center[0]), int(self.center[1])],
            'radius': float(self.radius),
            'confidence': float(self.confidence),
            'region_type': self.region_type,
            'bounding_box': [int(self.bounding_box[0]), int(self.bounding_box[1]), 
                           int(self.bounding_box[2]), int(self.bounding_box[3])],
            'contour_points': contour_coords,
            'area': float(self.area),
            'quality_score': float(self.quality_score)
        }
        
        # Añadir características opcionales si están disponibles
        if self.perimeter > 0:
            result['perimeter'] = float(self.perimeter)
        
        if self.circularity > 0:
            result['circularity'] = float(self.circularity)
            result['eccentricity'] = float(self.eccentricity)
            result['solidity'] = float(self.solidity)
            result['aspect_ratio'] = float(self.aspect_ratio)
        
        if self.texture_features:
            result['texture_features'] = {k: float(v) for k, v in self.texture_features.items()}
        
        return result

@dataclass
class ROIDetectionConfig:
    """Configuración para detección de ROI"""
    # Configuración general
    level: DetectionLevel = DetectionLevel.STANDARD
    
    # Parámetros de área
    min_area: int = 100
    max_area: int = 10000
    
    # Parámetros de forma
    min_circularity: float = 0.3
    max_eccentricity: float = 0.9
    
    # Parámetros de watershed
    watershed_markers_distance: int = 20
    watershed_compactness: float = 0.001
    
    # Parámetros de filtrado
    gaussian_sigma: float = 2.0
    morphology_kernel_size: int = 5
    
    # Parámetros de detección de bordes
    edge_detection_low: int = 50
    edge_detection_high: int = 150
    
    # Parámetros de confianza
    min_confidence: float = 0.5
    max_regions: int = 10
    overlap_threshold: float = 0.3
    
    # Parámetros específicos para tipos de regiones
    firing_pin_diameter_range: Tuple[int, int] = (10, 100)
    breech_face_min_area: int = 1000
    extractor_aspect_ratio_range: Tuple[float, float] = (0.2, 0.8)
    
    # Parámetros del Watershed mejorado
    use_enhanced_watershed: bool = True
    watershed_method: str = "ballistic_optimized"  # classic, distance, marker_controlled, hybrid, ballistic_optimized
    watershed_gaussian_sigma: float = 1.5
    watershed_marker_min_distance: int = 15
    watershed_marker_threshold_rel: float = 0.3
    watershed_min_region_area: int = 50
    watershed_max_region_area: int = 15000
    watershed_min_circularity: float = 0.2
    watershed_firing_pin_enhancement: bool = True
    watershed_breech_face_enhancement: bool = True

class UnifiedROIDetector(LoggerMixin):
    """Detector unificado de regiones de interés para imágenes balísticas"""
    
    def __init__(self, config: Optional[Union[Dict, ROIDetectionConfig]] = None):
        """
        Inicializa el detector de ROI
        
        Args:
            config: Configuración del detector (opcional)
                   Puede ser un diccionario o un objeto ROIDetectionConfig
        """
        super().__init__()
        
        # Configuraciones predefinidas
        self.default_configs = {
            DetectionLevel.SIMPLE: ROIDetectionConfig(
                level=DetectionLevel.SIMPLE,
                min_area=100,
                max_area=10000,
                min_circularity=0.3,
                max_eccentricity=0.9,
                watershed_markers_distance=20,
                watershed_compactness=0.001,
                gaussian_sigma=2.0,
                morphology_kernel_size=5,
                edge_detection_low=50,
                edge_detection_high=150,
                min_confidence=0.5,
                max_regions=10,
                overlap_threshold=0.3,
                firing_pin_diameter_range=(10, 100),
                breech_face_min_area=1000,
                extractor_aspect_ratio_range=(0.2, 0.8)
            ),
            DetectionLevel.STANDARD: ROIDetectionConfig(
                level=DetectionLevel.STANDARD,
                min_area=100,
                max_area=20000,
                min_circularity=0.3,
                max_eccentricity=0.9,
                watershed_markers_distance=20,
                watershed_compactness=0.001,
                gaussian_sigma=2.0,
                morphology_kernel_size=5,
                edge_detection_low=50,
                edge_detection_high=150,
                min_confidence=0.7,
                max_regions=15,
                overlap_threshold=0.3,
                firing_pin_diameter_range=(10, 100),
                breech_face_min_area=1000,
                extractor_aspect_ratio_range=(0.2, 0.8)
            ),
            DetectionLevel.ADVANCED: ROIDetectionConfig(
                level=DetectionLevel.ADVANCED,
                min_area=50,
                max_area=50000,
                min_circularity=0.2,
                max_eccentricity=0.95,
                watershed_markers_distance=15,
                watershed_compactness=0.001,
                gaussian_sigma=1.5,
                morphology_kernel_size=3,
                edge_detection_low=30,
                edge_detection_high=200,
                min_confidence=0.6,
                max_regions=20,
                overlap_threshold=0.2,
                firing_pin_diameter_range=(5, 150),
                breech_face_min_area=500,
                extractor_aspect_ratio_range=(0.1, 0.9)
            )
        }
        
        # Procesar configuración
        if config is None:
            self.config = self.default_configs[DetectionLevel.STANDARD]
        elif isinstance(config, dict):
            # Convertir diccionario a ROIDetectionConfig
            level_str = config.get('level', 'standard').lower()
            level = next((l for l in DetectionLevel if l.value == level_str), 
                         DetectionLevel.STANDARD)
            
            # Partir de la configuración predeterminada y actualizar
            self.config = self.default_configs[level]
            
            # Actualizar con valores del diccionario
            for key, value in config.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
        else:
            # Usar configuración proporcionada directamente
            self.config = config
        
        # Verificar dependencias opcionales
        self._check_dependencies()
        
        # Inicializar detector Watershed mejorado si está disponible
        self.enhanced_watershed = None
        if ENHANCED_WATERSHED_AVAILABLE and self.config.use_enhanced_watershed:
            try:
                watershed_config = WatershedConfig(
                    method=WatershedMethod(self.config.watershed_method),
                    gaussian_sigma=self.config.watershed_gaussian_sigma,
                    marker_min_distance=self.config.watershed_marker_min_distance,
                    marker_threshold_rel=self.config.watershed_marker_threshold_rel,
                    min_region_area=self.config.watershed_min_region_area,
                    max_region_area=self.config.watershed_max_region_area,
                    min_circularity=self.config.watershed_min_circularity,
                    firing_pin_enhancement=self.config.watershed_firing_pin_enhancement,
                    breech_face_enhancement=self.config.watershed_breech_face_enhancement
                )
                self.enhanced_watershed = EnhancedWatershedROI(watershed_config)
                self.logger.info(f"Detector Watershed mejorado inicializado (método: {self.config.watershed_method})")
            except Exception as e:
                self.logger.warning(f"Error inicializando detector Watershed mejorado: {e}")
                self.enhanced_watershed = None
        
        self.logger.info(f"Detector de ROI unificado inicializado (nivel: {self.config.level.value})")
    
    def _check_dependencies(self):
        """Verifica la disponibilidad de dependencias opcionales"""
        if not SCIPY_AVAILABLE:
            self.logger.warning("scipy no disponible. Algunas funciones avanzadas estarán limitadas.")
        
        if not SKIMAGE_AVAILABLE:
            self.logger.warning("scikit-image no disponible. Algunas funciones avanzadas estarán limitadas.")
        
        if not TORCH_AVAILABLE and self.config.level == DetectionLevel.ADVANCED:
            self.logger.warning("PyTorch no disponible. Algunas funciones avanzadas estarán limitadas.")
    
    def load_image(self, image_path: str) -> Optional[np.ndarray]:
        """
        Carga una imagen desde archivo
        
        Args:
            image_path: Ruta de la imagen
            
        Returns:
            Imagen cargada o None si hay error
        """
        try:
            path = Path(image_path)
            if not path.exists():
                self.logger.error(f"Archivo no encontrado: {image_path}")
                return None
            
            # Cargar imagen
            image = cv2.imread(str(path))
            if image is None:
                self.logger.error(f"No se pudo cargar la imagen: {image_path}")
                return None
            
            # Convertir de BGR a RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            self.logger.info(f"Imagen cargada: {path.name} ({image.shape})")
            return image
            
        except Exception as e:
            self.logger.error(f"Error cargando imagen {image_path}: {e}")
            return None
    
    def detect_roi_regions(self, image: Union[str, np.ndarray], 
                          specimen_type: str = 'cartridge_case',
                          level: Optional[str] = None) -> List[ROIRegion]:
        """
        Método alias para detect_roi para compatibilidad con tests
        """
        return self.detect_roi(image, specimen_type, level)
    
    def detect_roi(self, image: Union[str, np.ndarray], 
                   specimen_type: str = 'cartridge_case',
                   level: Optional[str] = None) -> List[ROIRegion]:
        """
        Detecta regiones de interés en una imagen balística
        
        Args:
            image: Imagen o ruta de imagen
            specimen_type: Tipo de espécimen ('cartridge_case', 'bullet', 'unknown')
            level: Nivel de detección (simple, standard, advanced)
            
        Returns:
            Lista de regiones de interés detectadas
        """
        start_time = time.time()
        
        try:
            # Configurar nivel si se especifica
            config = self.config
            if level is not None:
                level_enum = next((l for l in DetectionLevel if l.value == level.lower()), None)
                if level_enum and level_enum in self.default_configs:
                    config = self.default_configs[level_enum]
                else:
                    self.logger.warning(f"Nivel de detección no válido: {level}. Usando configuración actual.")
            
            # Cargar imagen si es necesario
            if isinstance(image, str):
                loaded_image = self.load_image(image)
                if loaded_image is None:
                    return []
                image = loaded_image
            
            # Preprocesar imagen
            preprocessed = self._preprocess_image(image)
            
            # Detectar ROI según tipo de espécimen
            if specimen_type.lower() == 'cartridge_case':
                if config.level == DetectionLevel.SIMPLE:
                    regions = self._detect_cartridge_simple(preprocessed)
                else:
                    regions = self._detect_cartridge_case_roi(preprocessed)
            elif specimen_type.lower() == 'bullet':
                if config.level == DetectionLevel.SIMPLE:
                    regions = self._detect_bullet_simple(preprocessed)
                else:
                    regions = self._detect_bullet_roi(preprocessed)
            else:
                # Tipo desconocido, usar detección genérica
                regions = self._detect_generic_roi(preprocessed)
            
            # Filtrar regiones por confianza
            regions = [r for r in regions if r.confidence >= config.min_confidence]
            
            # Filtrar regiones superpuestas
            regions = self._filter_overlapping_regions(regions)
            
            # Limitar número de regiones
            regions = regions[:config.max_regions]
            
            # Calcular métricas de calidad para regiones avanzadas
            if config.level != DetectionLevel.SIMPLE:
                regions = self._calculate_quality_metrics(image, regions)
            
            processing_time = time.time() - start_time
            self.logger.info(f"Detección de ROI completada en {processing_time:.3f}s. Regiones encontradas: {len(regions)}")
            
            return regions
            
        except Exception as e:
            self.logger.error(f"Error en detección de ROI: {e}")
            return []
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocesa la imagen para detección de ROI
        
        Args:
            image: Imagen a procesar
            
        Returns:
            Imagen preprocesada
        """
        try:
            # Convertir a escala de grises si es necesario
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image.copy()
            
            # Aplicar filtro gaussiano para reducir ruido
            blurred = cv2.GaussianBlur(gray, (5, 5), self.config.gaussian_sigma)
            
            # Normalizar contraste
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(blurred)
            
            # Aplicar filtro morfológico para eliminar ruido pequeño
            kernel_size = self.config.morphology_kernel_size
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            morphed = cv2.morphologyEx(enhanced, cv2.MORPH_OPEN, kernel)
            
            return morphed
            
        except Exception as e:
            self.logger.error(f"Error en preprocesamiento: {e}")
            return image
    
    def _detect_cartridge_simple(self, image: np.ndarray) -> List[ROIRegion]:
        """
        Detecta ROI en casquillos usando método simple (basado en círculos)
        
        Args:
            image: Imagen preprocesada
            
        Returns:
            Lista de regiones detectadas
        """
        try:
            # Detectar círculos usando transformada de Hough
            circles = cv2.HoughCircles(
                image, 
                cv2.HOUGH_GRADIENT, 
                dp=1, 
                minDist=50,
                param1=50,
                param2=30,
                minRadius=10,
                maxRadius=100
            )
            
            if circles is None:
                return []
            
            # Convertir a enteros
            circles = np.uint16(np.around(circles))
            
            # Crear regiones
            regions = []
            for i in circles[0, :]:
                x, y, r = i
                
                # Clasificar región por tamaño
                region_type = self._classify_by_size(r)
                
                # Calcular confianza
                confidence = self._calculate_simple_confidence(image, x, y, r)
                
                # Crear contorno circular
                contour = self._create_circle_contour(x, y, r)
                
                # Crear bounding box
                bbox = (x - r, y - r, 2 * r, 2 * r)
                
                # Crear región
                region = ROIRegion(
                    center=(x, y),
                    radius=r,
                    contour=contour,
                    confidence=confidence,
                    region_type=region_type,
                    bounding_box=bbox,
                    area=np.pi * r * r
                )
                
                regions.append(region)
            
            # Ordenar por confianza
            regions.sort(key=lambda r: r.confidence, reverse=True)
            
            return regions
            
        except Exception as e:
            self.logger.error(f"Error en detección simple de casquillos: {e}")
            return []
    
    def _detect_bullet_simple(self, image: np.ndarray) -> List[ROIRegion]:
        """
        Detecta ROI en proyectiles usando método simple
        
        Args:
            image: Imagen preprocesada
            
        Returns:
            Lista de regiones detectadas
        """
        try:
            # Detectar bordes
            edges = cv2.Canny(image, 50, 150)
            
            # Encontrar contornos
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filtrar contornos por área
            filtered_contours = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if self.config.min_area <= area <= self.config.max_area:
                    filtered_contours.append(contour)
            
            # Crear regiones
            regions = []
            for contour in filtered_contours:
                # Calcular centro y radio aproximado
                M = cv2.moments(contour)
                if M["m00"] == 0:
                    continue
                
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # Calcular radio aproximado como la distancia promedio al centro
                dists = []
                for point in contour:
                    x, y = point[0]
                    dist = np.sqrt((x - cx)**2 + (y - cy)**2)
                    dists.append(dist)
                
                radius = np.mean(dists)
                
                # Calcular bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # Calcular confianza
                confidence = 0.7  # Valor por defecto para método simple
                
                # Crear región
                region = ROIRegion(
                    center=(cx, cy),
                    radius=radius,
                    contour=contour,
                    confidence=confidence,
                    region_type="striation",
                    bounding_box=(x, y, w, h),
                    area=cv2.contourArea(contour)
                )
                
                regions.append(region)
            
            # Ordenar por área
            regions.sort(key=lambda r: r.area, reverse=True)
            
            return regions
            
        except Exception as e:
            self.logger.error(f"Error en detección simple de proyectiles: {e}")
            return []
    
    def _detect_cartridge_case_roi(self, image: np.ndarray) -> List[ROIRegion]:
        """
        Detecta ROI en casquillos usando método avanzado
        
        Args:
            image: Imagen preprocesada
            
        Returns:
            Lista de regiones detectadas
        """
        try:
            # Método 1: Detección de círculos para percutor y culote
            circles = self._detect_circles(image)
            
            # Método 2: Segmentación watershed mejorada si está disponible
            if self.enhanced_watershed is not None:
                try:
                    watershed_regions = self.enhanced_watershed.segment_roi(image, 'cartridge_case')
                    
                    # Convertir WatershedRegion a ROIRegion
                    for wr in watershed_regions:
                        # Crear contorno desde la máscara
                        contours, _ = cv2.findContours(wr.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        if not contours:
                            continue
                        
                        contour = contours[0]
                        
                        # Crear región ROI
                        region = ROIRegion(
                            center=wr.center,
                            radius=wr.radius,
                            contour=contour,
                            confidence=wr.confidence,
                            region_type=wr.region_type,
                            bounding_box=wr.bounding_box,
                            area=wr.area,
                            perimeter=wr.perimeter,
                            circularity=wr.circularity,
                            eccentricity=wr.eccentricity,
                            solidity=wr.solidity,
                            aspect_ratio=wr.aspect_ratio,
                            mask=wr.mask
                        )
                        
                        regions.append(region)
                        
                except Exception as e:
                    self.logger.warning(f"Error en segmentación Watershed mejorada: {e}")
                    # Fallback a método original
                    if SKIMAGE_AVAILABLE:
                        segments = self._watershed_segmentation(image)
                    else:
                        segments = None
            else:
                # Método original: Segmentación watershed básica
                if SKIMAGE_AVAILABLE:
                    segments = self._watershed_segmentation(image)
                else:
                    segments = None
            
            # Crear regiones a partir de círculos
            regions = []
            for x, y, r in circles:
                # Clasificar región por tamaño
                region_type = self._classify_cartridge_region(r)
                
                # Extraer contorno circular
                contour = self._extract_circular_contour(image, (x, y), r)
                
                # Calcular confianza
                confidence = self._calculate_confidence(image, contour, region_type)
                
                # Crear bounding box
                bbox = (x - r, y - r, 2 * r, 2 * r)
                
                # Crear región
                region = ROIRegion(
                    center=(x, y),
                    radius=r,
                    contour=contour,
                    confidence=confidence,
                    region_type=region_type,
                    bounding_box=bbox,
                    area=np.pi * r * r
                )
                
                regions.append(region)
            
            # Añadir regiones de segmentación watershed si están disponibles
            if segments is not None and SKIMAGE_AVAILABLE:
                try:
                    props = regionprops(segments)
                    
                    for prop in props:
                        # Filtrar por área
                        if not (self.config.min_area <= prop.area <= self.config.max_area):
                            continue
                        
                        # Calcular centro
                        y, x = prop.centroid
                        x, y = int(x), int(y)
                        
                        # Calcular radio equivalente
                        radius = np.sqrt(prop.area / np.pi)
                        
                        # Extraer contorno
                        mask = segments == prop.label
                        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        
                        if not contours:
                            continue
                        
                        contour = contours[0]
                        
                        # Calcular bounding box
                        bbox = cv2.boundingRect(contour)
                        
                        # Determinar tipo de región
                        if prop.eccentricity > 0.7:
                            region_type = "extractor"
                        else:
                            region_type = "unknown"
                        
                        # Calcular confianza
                        confidence = 0.6  # Valor por defecto para segmentación
                        
                        # Crear región
                        region = ROIRegion(
                            center=(x, y),
                            radius=radius,
                            contour=contour,
                            confidence=confidence,
                            region_type=region_type,
                            bounding_box=bbox,
                            area=prop.area,
                            perimeter=prop.perimeter,
                            circularity=4 * np.pi * prop.area / (prop.perimeter * prop.perimeter) if prop.perimeter > 0 else 0,
                            eccentricity=prop.eccentricity,
                            solidity=prop.solidity,
                            aspect_ratio=prop.major_axis_length / prop.minor_axis_length if prop.minor_axis_length > 0 else 0
                        )
                        
                        regions.append(region)
                except Exception as e:
                    self.logger.error(f"Error procesando regiones de watershed: {e}")
            
            # Filtrar regiones superpuestas
            regions = self._filter_overlapping_regions(regions)
            
            # Ordenar por confianza
            regions.sort(key=lambda r: r.confidence, reverse=True)
            
            return regions
            
        except Exception as e:
            self.logger.error(f"Error en detección avanzada de casquillos: {e}")
            return []
    
    def _detect_bullet_roi(self, image: np.ndarray) -> List[ROIRegion]:
        """
        Detecta ROI en proyectiles usando método avanzado
        
        Args:
            image: Imagen preprocesada
            
        Returns:
            Lista de regiones detectadas
        """
        try:
            regions = []
            
            # Método 1: Segmentación watershed mejorada si está disponible
            if self.enhanced_watershed is not None:
                try:
                    watershed_regions = self.enhanced_watershed.segment_roi(image, 'bullet')
                    
                    # Convertir WatershedRegion a ROIRegion
                    for wr in watershed_regions:
                        # Crear contorno desde la máscara
                        contours, _ = cv2.findContours(wr.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        if not contours:
                            continue
                        
                        contour = contours[0]
                        
                        # Crear región ROI
                        region = ROIRegion(
                            center=wr.center,
                            radius=wr.radius,
                            contour=contour,
                            confidence=wr.confidence,
                            region_type=wr.region_type,
                            bounding_box=wr.bounding_box,
                            area=wr.area,
                            perimeter=wr.perimeter,
                            circularity=wr.circularity,
                            eccentricity=wr.eccentricity,
                            solidity=wr.solidity,
                            aspect_ratio=wr.aspect_ratio,
                            mask=wr.mask
                        )
                        
                        regions.append(region)
                        
                except Exception as e:
                    self.logger.warning(f"Error en segmentación Watershed mejorada para proyectiles: {e}")
            
            # Método 2: Detección de bordes para estrías (fallback o complementario)
            edges = cv2.Canny(image, self.config.edge_detection_low, self.config.edge_detection_high)
            
            # Encontrar contornos
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filtrar contornos por área
            filtered_contours = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if self.config.min_area <= area <= self.config.max_area:
                    filtered_contours.append(contour)
            
            # Crear regiones adicionales desde contornos
            for contour in filtered_contours:
                # Calcular centro y radio aproximado
                M = cv2.moments(contour)
                if M["m00"] == 0:
                    continue
                
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # Calcular radio aproximado como la distancia promedio al centro
                dists = []
                for point in contour:
                    x, y = point[0]
                    dist = np.sqrt((x - cx)**2 + (y - cy)**2)
                    dists.append(dist)
                
                radius = np.mean(dists)
                
                # Calcular bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # Calcular características geométricas
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)
                circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                
                # Determinar tipo de región
                if w > h * 2:
                    region_type = "striation_horizontal"
                elif h > w * 2:
                    region_type = "striation_vertical"
                else:
                    region_type = "striation"
                
                # Calcular confianza basada en características
                if SKIMAGE_AVAILABLE:
                    # Crear máscara para la región
                    mask = np.zeros(image.shape, dtype=np.uint8)
                    cv2.drawContours(mask, [contour], 0, 255, -1)
                    
                    # Calcular características de textura
                    texture_features = self._calculate_texture_features(image, mask)
                    
                    # Calcular confianza basada en textura
                    confidence = 0.5 + 0.5 * texture_features.get('contrast', 0)
                else:
                    texture_features = {}
                    confidence = 0.7  # Valor por defecto
                
                # Crear región
                region = ROIRegion(
                    center=(cx, cy),
                    radius=radius,
                    contour=contour,
                    confidence=confidence,
                    region_type=region_type,
                    bounding_box=(x, y, w, h),
                    area=area,
                    perimeter=perimeter,
                    circularity=circularity,
                    aspect_ratio=w/h if h > 0 else 0,
                    texture_features=texture_features
                )
                
                regions.append(region)
            
            # Ordenar por confianza
            regions.sort(key=lambda r: r.confidence, reverse=True)
            
            return regions
            
        except Exception as e:
            self.logger.error(f"Error en detección avanzada de proyectiles: {e}")
            return []
    
    def _detect_generic_roi(self, image: np.ndarray) -> List[ROIRegion]:
        """
        Detecta ROI genéricas en imágenes balísticas
        
        Args:
            image: Imagen preprocesada
            
        Returns:
            Lista de regiones detectadas
        """
        try:
            # Combinar métodos de detección de casquillos y proyectiles
            cartridge_regions = self._detect_cartridge_simple(image)
            bullet_regions = self._detect_bullet_simple(image)
            
            # Combinar regiones
            regions = cartridge_regions + bullet_regions
            
            # Filtrar regiones superpuestas
            regions = self._filter_overlapping_regions(regions)
            
            # Ordenar por confianza
            regions.sort(key=lambda r: r.confidence, reverse=True)
            
            return regions
            
        except Exception as e:
            self.logger.error(f"Error en detección genérica: {e}")
            return []
    
    def _detect_circles(self, image: np.ndarray) -> List[Tuple[int, int, int]]:
        """
        Detecta círculos en la imagen usando transformada de Hough
        
        Args:
            image: Imagen preprocesada
            
        Returns:
            Lista de círculos detectados (x, y, r)
        """
        try:
            # Detectar círculos usando transformada de Hough
            circles = cv2.HoughCircles(
                image, 
                cv2.HOUGH_GRADIENT, 
                dp=1, 
                minDist=50,
                param1=50,
                param2=30,
                minRadius=10,
                maxRadius=100
            )
            
            if circles is None:
                return []
            
            # Convertir a enteros
            circles = np.uint16(np.around(circles))
            
            # Convertir a lista de tuplas
            return [(c[0], c[1], c[2]) for c in circles[0, :]]
            
        except Exception as e:
            self.logger.error(f"Error en detección de círculos: {e}")
            return []
    
    def _watershed_segmentation(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Segmenta la imagen usando watershed
        
        Args:
            image: Imagen preprocesada
            
        Returns:
            Imagen segmentada con etiquetas
        """
        if not SKIMAGE_AVAILABLE:
            return None
            
        try:
            # Calcular gradiente
            grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
            gradient = np.sqrt(grad_x**2 + grad_y**2)
            gradient = cv2.normalize(gradient, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            
            # Encontrar máximos locales para marcadores
            markers = np.zeros_like(image)
            
            # Usar peak_local_maxima si está disponible
            try:
                peaks = peak_local_maxima(image, min_distance=self.config.watershed_markers_distance)
                markers[tuple(peaks.T)] = 1
            except Exception as e:
                self.logger.warning(f"Error usando peak_local_maxima: {e}")
                # Fallback: usar threshold para encontrar marcadores
                _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                markers = cv2.erode(thresh, None, iterations=2)
            
            # Etiquetar marcadores
            markers = label(markers)
            
            # Aplicar watershed
            segments = watershed(-gradient, markers, compactness=self.config.watershed_compactness)
            
            # Limpiar bordes
            segments = clear_border(segments)
            
            return segments
            
        except Exception as e:
            self.logger.error(f"Error en segmentación watershed: {e}")
            return None
    
    def _classify_by_size(self, radius: float) -> str:
        """
        Clasifica una región por su tamaño
        
        Args:
            radius: Radio de la región
            
        Returns:
            Tipo de región
        """
        if radius < 20:
            return "firing_pin"
        elif radius < 50:
            return "primer"
        else:
            return "breech_face"
    
    def _classify_cartridge_region(self, radius: float) -> str:
        """
        Clasifica una región de casquillo por su tamaño
        
        Args:
            radius: Radio de la región
            
        Returns:
            Tipo de región
        """
        # Clasificar por tamaño
        if self.config.firing_pin_diameter_range[0] <= radius <= self.config.firing_pin_diameter_range[1]:
            return "firing_pin"
        elif radius > self.config.firing_pin_diameter_range[1]:
            return "breech_face"
        else:
            return "unknown"
    
    def _extract_circular_contour(self, image: np.ndarray, center: Tuple[int, int], radius: float) -> np.ndarray:
        """
        Extrae un contorno circular
        
        Args:
            image: Imagen
            center: Centro del círculo (x, y)
            radius: Radio del círculo
            
        Returns:
            Contorno del círculo
        """
        try:
            # Crear contorno circular
            angles = np.linspace(0, 2*np.pi, 100)
            x = center[0] + radius * np.cos(angles)
            y = center[1] + radius * np.sin(angles)
            
            # Convertir a puntos de contorno
            points = np.column_stack((x, y)).astype(np.int32)
            contour = points.reshape((-1, 1, 2))
            
            return contour
            
        except Exception as e:
            self.logger.error(f"Error extrayendo contorno circular: {e}")
            # Retornar contorno vacío
            return np.array([[[0, 0]]], dtype=np.int32)
    
    def _create_circle_contour(self, x: int, y: int, r: int) -> np.ndarray:
        """
        Crea un contorno circular
        
        Args:
            x: Coordenada x del centro
            y: Coordenada y del centro
            r: Radio
            
        Returns:
            Contorno del círculo
        """
        try:
            # Crear contorno circular
            angles = np.linspace(0, 2*np.pi, 100)
            cx = x + r * np.cos(angles)
            cy = y + r * np.sin(angles)
            
            # Convertir a puntos de contorno
            points = np.column_stack((cx, cy)).astype(np.int32)
            contour = points.reshape((-1, 1, 2))
            
            return contour
            
        except Exception as e:
            self.logger.error(f"Error creando contorno circular: {e}")
            # Retornar contorno vacío
            return np.array([[[0, 0]]], dtype=np.int32)
    
    def _calculate_simple_confidence(self, image: np.ndarray, x: int, y: int, r: int) -> float:
        """
        Calcula la confianza de una región simple
        
        Args:
            image: Imagen
            x: Coordenada x del centro
            y: Coordenada y del centro
            r: Radio
            
        Returns:
            Confianza (0-1)
        """
        try:
            # Crear máscara circular
            mask = np.zeros(image.shape, dtype=np.uint8)
            cv2.circle(mask, (x, y), r, 255, -1)
            
            # Calcular contraste dentro de la máscara
            mean_inside = cv2.mean(image, mask=mask)[0]
            
            # Crear máscara para anillo exterior
            outer_mask = np.zeros(image.shape, dtype=np.uint8)
            cv2.circle(outer_mask, (x, y), r + 5, 255, -1)
            cv2.circle(outer_mask, (x, y), r, 0, -1)
            
            # Calcular contraste en anillo exterior
            mean_outside = cv2.mean(image, mask=outer_mask)[0]
            
            # Calcular diferencia de contraste
            contrast_diff = abs(mean_inside - mean_outside) / 255.0
            
            # Calcular confianza basada en contraste
            confidence = min(contrast_diff * 2.0, 1.0)
            
            return confidence
            
        except Exception as e:
            self.logger.error(f"Error calculando confianza simple: {e}")
            return 0.5  # Valor por defecto
    
    def _calculate_confidence(self, image: np.ndarray, contour: np.ndarray, region_type: str) -> float:
        """
        Calcula la confianza de una región
        
        Args:
            image: Imagen
            contour: Contorno de la región
            region_type: Tipo de región
            
        Returns:
            Confianza (0-1)
        """
        try:
            # Crear máscara para la región
            mask = np.zeros(image.shape, dtype=np.uint8)
            cv2.drawContours(mask, [contour], 0, 255, -1)
            
            # Calcular estadísticas dentro de la máscara
            mean_inside = cv2.mean(image, mask=mask)[0]
            
            # Dilatar contorno para crear anillo exterior
            dilated_contour = cv2.dilate(mask, np.ones((5, 5), np.uint8))
            outer_mask = dilated_contour - mask
            
            # Calcular estadísticas en anillo exterior
            mean_outside = cv2.mean(image, mask=outer_mask)[0]
            
            # Calcular diferencia de contraste
            contrast_diff = abs(mean_inside - mean_outside) / 255.0
            
            # Calcular varianza dentro de la región
            roi = cv2.bitwise_and(image, image, mask=mask)
            roi_flat = roi[mask > 0]
            if len(roi_flat) > 0:
                variance = np.var(roi_flat) / (255.0 * 255.0)
            else:
                variance = 0
            
            # Calcular confianza basada en tipo de región
            if region_type == "firing_pin":
                # Para percutor, alto contraste es bueno
                confidence = 0.5 + 0.5 * contrast_diff - 0.3 * variance
            elif region_type == "breech_face":
                # Para culote, textura es importante
                confidence = 0.4 + 0.3 * contrast_diff + 0.3 * variance
            else:
                # Para otros tipos, usar valor por defecto
                confidence = 0.5 + 0.5 * contrast_diff
            
            # Limitar a rango [0, 1]
            confidence = max(0.0, min(1.0, confidence))
            
            return confidence
            
        except Exception as e:
            self.logger.error(f"Error calculando confianza: {e}")
            return 0.5  # Valor por defecto
    
    def _calculate_texture_features(self, image: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
        """
        Calcula características de textura para una región
        
        Args:
            image: Imagen
            mask: Máscara de la región
            
        Returns:
            Diccionario con características de textura
        """
        if not SKIMAGE_AVAILABLE:
            return {}
            
        try:
            # Extraer región de interés
            roi = cv2.bitwise_and(image, image, mask=mask)
            roi_flat = roi[mask > 0]
            
            if len(roi_flat) == 0:
                return {}
            
            # Calcular estadísticas básicas
            mean = np.mean(roi_flat) / 255.0
            std = np.std(roi_flat) / 255.0
            
            # Calcular LBP si está disponible
            try:
                # Crear imagen temporal para LBP
                temp_roi = np.zeros_like(image)
                temp_roi[mask > 0] = image[mask > 0]
                
                # Calcular LBP usando cache
                lbp = cached_local_binary_pattern(temp_roi, 8, 1, method='uniform')
                lbp_roi = lbp[mask > 0]
                
                # Calcular histograma LBP
                hist, _ = np.histogram(lbp_roi, bins=10, range=(0, 10), density=True)
                
                # Calcular uniformidad y entropía
                uniformity = np.sum(hist * hist)
                entropy = -np.sum(hist * np.log2(hist + 1e-10))
                
                return {
                    'mean': float(mean),
                    'std': float(std),
                    'contrast': float(std * 2),  # Contraste normalizado
                    'uniformity': float(uniformity),
                    'entropy': float(entropy / 4)  # Normalizado a [0, 1]
                }
            except Exception:
                # Fallback si LBP falla
                return {
                    'mean': float(mean),
                    'std': float(std),
                    'contrast': float(std * 2)
                }
            
        except Exception as e:
            self.logger.error(f"Error calculando características de textura: {e}")
            return {}
    
    def _filter_overlapping_regions(self, regions: List[ROIRegion]) -> List[ROIRegion]:
        """
        Filtra regiones superpuestas
        
        Args:
            regions: Lista de regiones
            
        Returns:
            Lista de regiones filtradas
        """
        if not regions:
            return []
            
        try:
            # Ordenar por confianza
            sorted_regions = sorted(regions, key=lambda r: r.confidence, reverse=True)
            
            # Lista de regiones filtradas
            filtered_regions = [sorted_regions[0]]
            
            # Filtrar regiones superpuestas
            for region in sorted_regions[1:]:
                overlap = False
                
                for filtered_region in filtered_regions:
                    # Calcular superposición
                    overlap_ratio = self._calculate_overlap(region, filtered_region)
                    
                    if overlap_ratio > self.config.overlap_threshold:
                        overlap = True
                        break
                
                if not overlap:
                    filtered_regions.append(region)
            
            return filtered_regions
            
        except Exception as e:
            self.logger.error(f"Error filtrando regiones superpuestas: {e}")
            return regions
    
    def _calculate_overlap(self, region1: ROIRegion, region2: ROIRegion) -> float:
        """
        Calcula la superposición entre dos regiones
        
        Args:
            region1: Primera región
            region2: Segunda región
            
        Returns:
            Ratio de superposición (0-1)
        """
        try:
            # Calcular distancia entre centros
            cx1, cy1 = region1.center
            cx2, cy2 = region2.center
            
            distance = np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
            
            # Calcular superposición basada en radios
            overlap_distance = region1.radius + region2.radius - distance
            
            if overlap_distance <= 0:
                return 0.0
            
            # Calcular ratio de superposición
            min_radius = min(region1.radius, region2.radius)
            overlap_ratio = overlap_distance / (2 * min_radius)
            
            return min(1.0, max(0.0, overlap_ratio))
            
        except Exception as e:
            self.logger.error(f"Error calculando superposición: {e}")
            return 0.0
    
    def _calculate_quality_metrics(self, image: np.ndarray, regions: List[ROIRegion]) -> List[ROIRegion]:
        """
        Calcula métricas de calidad para las regiones
        
        Args:
            image: Imagen original
            regions: Lista de regiones
            
        Returns:
            Lista de regiones con métricas de calidad
        """
        try:
            for i, region in enumerate(regions):
                # Crear máscara para la región
                mask = np.zeros(image.shape[:2], dtype=np.uint8)
                cv2.drawContours(mask, [region.contour], 0, 255, -1)
                
                # Calcular características de textura
                if len(image.shape) == 3:
                    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                else:
                    gray = image
                
                texture_features = self._calculate_texture_features(gray, mask)
                
                # Calcular calidad basada en características
                contrast = texture_features.get('contrast', 0)
                uniformity = texture_features.get('uniformity', 0)
                
                # Calcular score de calidad
                quality_score = 0.4 + 0.3 * contrast + 0.3 * uniformity
                quality_score = max(0.0, min(1.0, quality_score))
                
                # Actualizar región
                regions[i].quality_score = quality_score
                regions[i].texture_features = texture_features
            
            return regions
            
        except Exception as e:
            self.logger.error(f"Error calculando métricas de calidad: {e}")
            return regions
    
    def visualize_roi(self, image: np.ndarray, regions: List[ROIRegion]) -> np.ndarray:
        """
        Visualiza las regiones de interés en la imagen
        
        Args:
            image: Imagen original
            regions: Lista de regiones
            
        Returns:
            Imagen con regiones visualizadas
        """
        try:
            # Validar entrada
            if image is None:
                raise ValueError("La imagen no puede ser None")
                
            # Crear copia de la imagen
            if len(image.shape) == 2:
                vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                vis_image = image.copy()
            
            # Si no hay regiones, agregar texto informativo
            if not regions or len(regions) == 0:
                height, width = vis_image.shape[:2]
                cv2.putText(vis_image, "No ROI regions detected", 
                           (width//4, height//2), cv2.FONT_HERSHEY_SIMPLEX, 
                           1, (255, 0, 0), 2)
                return vis_image
            
            # Colores para diferentes tipos de regiones
            colors = {
                "firing_pin": (0, 255, 0),      # Verde
                "breech_face": (0, 0, 255),     # Azul
                "primer": (255, 0, 0),          # Rojo
                "extractor": (255, 255, 0),     # Amarillo
                "striation": (255, 0, 255),     # Magenta
                "striation_horizontal": (0, 255, 255),  # Cian
                "striation_vertical": (128, 0, 255),    # Púrpura
                "unknown": (128, 128, 128)      # Gris
            }
            
            # Dibujar regiones
            for region in regions:
                # Validar región
                if region is None or region.contour is None:
                    continue
                    
                # Obtener color
                color = colors.get(region.region_type, (128, 128, 128))
                
                # Dibujar contorno
                cv2.drawContours(vis_image, [region.contour], 0, color, 2)
                
                # Dibujar centro
                cv2.circle(vis_image, region.center, 3, color, -1)
                
                # Dibujar etiqueta
                x, y = region.center
                label = f"{region.region_type} ({region.confidence:.2f})"
                cv2.putText(vis_image, label, (x - 10, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            return vis_image
            
        except Exception as e:
            self.logger.error(f"Error visualizando ROI: {e}")
            return image if image is not None else np.zeros((100, 100, 3), dtype=np.uint8)

def detect_roi_from_path(image_path: str, 
                       specimen_type: str = 'cartridge_case',
                       level: str = 'standard') -> List[Dict[str, Any]]:
    """
    Detecta regiones de interés en una imagen desde archivo
    
    Args:
        image_path: Ruta de la imagen
        specimen_type: Tipo de espécimen ('cartridge_case', 'bullet', 'unknown')
        level: Nivel de detección ('simple', 'standard', 'advanced')
        
    Returns:
        Lista de regiones detectadas como diccionarios
    """
    try:
        # Crear detector
        detector = UnifiedROIDetector()
        
        # Detectar regiones
        regions = detector.detect_roi(image_path, specimen_type, level)
        
        # Convertir a diccionarios
        return [region.to_dict() for region in regions]
        
    except Exception as e:
        logging.error(f"Error en detección de ROI: {e}")
        return []

# Punto de entrada para uso como script
if __name__ == "__main__":
    import argparse
    
    # Configurar logging
    logging.basicConfig(level=logging.INFO)
    
    # Crear parser de argumentos
    parser = argparse.ArgumentParser(description="Detector unificado de ROI para imágenes balísticas")
    parser.add_argument("input", help="Imagen de entrada")
    parser.add_argument("--output", "-o", help="Imagen de salida con visualización", default=None)
    parser.add_argument("--level", "-l", help="Nivel de detección", 
                       choices=["simple", "standard", "advanced"], default="standard")
    parser.add_argument("--type", "-t", help="Tipo de espécimen", 
                       choices=["cartridge_case", "bullet", "unknown"], default="cartridge_case")
    
    args = parser.parse_args()
    
    # Crear detector
    detector = UnifiedROIDetector()
    
    # Cargar imagen
    image = detector.load_image(args.input)
    
    if image is None:
        print(f"Error: No se pudo cargar la imagen {args.input}")
        exit(1)
    
    # Detectar regiones
    regions = detector.detect_roi(image, args.type, args.level)
    
    # Mostrar resultados
    print(f"Detectadas {len(regions)} regiones de interés:")
    for i, region in enumerate(regions):
        print(f"\nRegión {i+1}:")
        print(f"  Tipo: {region.region_type}")
        print(f"  Centro: {region.center}")
        print(f"  Radio: {region.radius:.1f}")
        print(f"  Área: {region.area:.1f}")
        print(f"  Confianza: {region.confidence:.3f}")
        print(f"  Calidad: {region.quality_score:.3f}")
    
    # Visualizar regiones
    if len(regions) > 0:
        vis_image = detector.visualize_roi(image, regions)
        
        # Guardar imagen si se especifica
        if args.output:
            # Convertir a BGR para OpenCV
            vis_image_bgr = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(args.output, vis_image_bgr)
            print(f"Imagen con ROI guardada en: {args.output}")
    else:
        print("No se detectaron regiones de interés")