"""
Módulo de Segmentación Watershed Mejorada para ROI Balísticas
Sistema Balístico Forense MVP

Implementación avanzada de segmentación Watershed optimizada para:
- Detección automática de regiones de interés en vainas y proyectiles
- Análisis específico de características balísticas
- Mejora de precisión en identificación de percutor, culote y estrías
- Integración con pipeline de preprocesamiento existente

Basado en:
- Vincent & Soille (1991) - Algoritmo Watershed original
- Meyer (1994) - Watershed por inmersión
- Beucher & Lantuéjoul (1979) - Transformada de distancia
- Roerdink & Meijster (2000) - Implementación computacional eficiente

Autor: Sistema Balístico Forense MVP
Fecha: 2024
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
import logging
from pathlib import Path
from enum import Enum
import time

# Importaciones opcionales con manejo de errores
try:
    from scipy import ndimage
    from scipy.spatial.distance import cdist
    from scipy.ndimage import distance_transform_edt
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from skimage.feature import peak_local_maxima
    from skimage.filters import gaussian, sobel, threshold_otsu, rank
    from skimage.measure import label, regionprops
    from skimage.morphology import disk, watershed, remove_small_objects, opening, closing
    from skimage.segmentation import clear_border
    from skimage.util import img_as_ubyte
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False

from utils.logger import LoggerMixin


class WatershedMethod(Enum):
    """Métodos de segmentación Watershed disponibles"""
    CLASSIC = "classic"           # Watershed clásico con gradiente
    DISTANCE = "distance"         # Basado en transformada de distancia
    MARKER_CONTROLLED = "marker_controlled"  # Con marcadores específicos
    HYBRID = "hybrid"            # Combinación de métodos
    BALLISTIC_OPTIMIZED = "ballistic_optimized"  # Optimizado para balística


@dataclass
class WatershedConfig:
    """Configuración para segmentación Watershed mejorada"""
    # Método de segmentación
    method: WatershedMethod = WatershedMethod.BALLISTIC_OPTIMIZED
    
    # Parámetros de preprocesamiento
    gaussian_sigma: float = 1.5
    morphology_kernel_size: int = 3
    noise_reduction_iterations: int = 2
    
    # Parámetros de gradiente
    gradient_method: str = "sobel"  # "sobel", "scharr", "laplacian"
    gradient_threshold: float = 0.1
    
    # Parámetros de marcadores
    marker_min_distance: int = 15
    marker_threshold_rel: float = 0.3
    marker_footprint_size: int = 3
    
    # Parámetros de watershed
    compactness: float = 0.001
    watershed_line: bool = True
    
    # Parámetros de filtrado post-procesamiento
    min_region_area: int = 50
    max_region_area: int = 15000
    min_circularity: float = 0.2
    max_eccentricity: float = 0.95
    
    # Parámetros específicos para balística
    firing_pin_enhancement: bool = True
    breech_face_enhancement: bool = True
    edge_preservation: bool = True
    
    # Parámetros de confianza
    confidence_weights: Dict[str, float] = field(default_factory=lambda: {
        'area': 0.25,
        'circularity': 0.20,
        'contrast': 0.20,
        'edge_strength': 0.15,
        'texture': 0.10,
        'position': 0.10
    })


@dataclass
class WatershedRegion:
    """Región detectada por segmentación Watershed"""
    label: int
    center: Tuple[int, int]
    area: float
    perimeter: float
    bounding_box: Tuple[int, int, int, int]  # x, y, width, height
    contour: np.ndarray
    mask: np.ndarray
    
    # Propiedades geométricas
    circularity: float = 0.0
    eccentricity: float = 0.0
    solidity: float = 0.0
    aspect_ratio: float = 0.0
    extent: float = 0.0
    
    # Propiedades de intensidad
    mean_intensity: float = 0.0
    std_intensity: float = 0.0
    min_intensity: float = 0.0
    max_intensity: float = 0.0
    
    # Métricas de calidad
    confidence: float = 0.0
    edge_strength: float = 0.0
    texture_score: float = 0.0
    
    # Clasificación
    region_type: str = "unknown"
    is_ballistic_feature: bool = False


class EnhancedWatershedROI(LoggerMixin):
    """
    Detector de ROI mejorado usando segmentación Watershed avanzada
    
    Implementa múltiples variantes de Watershed optimizadas para
    características balísticas específicas.
    """
    
    def __init__(self, config: Optional[WatershedConfig] = None):
        """
        Inicializa el detector Watershed mejorado
        
        Args:
            config: Configuración personalizada
        """
        super().__init__()
        self.config = config or Watershedget_unified_config()
        self._check_dependencies()
        
        # Cache para optimización
        self._gradient_cache = {}
        self._marker_cache = {}
        
        self.logger.info(f"Inicializado EnhancedWatershedROI con método: {self.config.method.value}")
    
    def _check_dependencies(self):
        """Verifica dependencias disponibles"""
        if not SKIMAGE_AVAILABLE:
            self.logger.warning("scikit-image no disponible. Funcionalidad limitada.")
        if not SCIPY_AVAILABLE:
            self.logger.warning("scipy no disponible. Algunas funciones no estarán disponibles.")
    
    def segment_roi(self, image: np.ndarray, 
                   specimen_type: str = "cartridge_case") -> List[WatershedRegion]:
        """
        Segmenta ROI usando Watershed mejorado
        
        Args:
            image: Imagen a segmentar (escala de grises)
            specimen_type: Tipo de espécimen ("cartridge_case", "bullet")
            
        Returns:
            Lista de regiones segmentadas
        """
        start_time = time.time()
        
        try:
            # Preprocesar imagen
            preprocessed = self._preprocess_for_watershed(image, specimen_type)
            
            # Aplicar método de segmentación seleccionado
            if self.config.method == WatershedMethod.CLASSIC:
                segments = self._watershed_classic(preprocessed)
            elif self.config.method == WatershedMethod.DISTANCE:
                segments = self._watershed_distance(preprocessed)
            elif self.config.method == WatershedMethod.MARKER_CONTROLLED:
                segments = self._watershed_marker_controlled(preprocessed)
            elif self.config.method == WatershedMethod.HYBRID:
                segments = self._watershed_hybrid(preprocessed)
            else:  # BALLISTIC_OPTIMIZED
                segments = self._watershed_ballistic_optimized(preprocessed, specimen_type)
            
            if segments is None:
                return []
            
            # Extraer regiones
            regions = self._extract_regions(segments, image, preprocessed)
            
            # Filtrar y clasificar regiones
            regions = self._filter_regions(regions, specimen_type)
            regions = self._classify_regions(regions, specimen_type)
            
            # Calcular métricas de confianza
            regions = self._calculate_confidence_scores(regions, image)
            
            # Ordenar por confianza
            regions.sort(key=lambda r: r.confidence, reverse=True)
            
            processing_time = time.time() - start_time
            self.logger.info(f"Segmentación Watershed completada en {processing_time:.3f}s. "
                           f"Regiones encontradas: {len(regions)}")
            
            return regions
            
        except Exception as e:
            self.logger.error(f"Error en segmentación Watershed: {e}")
            return []
    
    def _preprocess_for_watershed(self, image: np.ndarray, specimen_type: str) -> np.ndarray:
        """
        Preprocesa imagen específicamente para Watershed
        
        Args:
            image: Imagen original
            specimen_type: Tipo de espécimen
            
        Returns:
            Imagen preprocesada
        """
        # Asegurar escala de grises
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Reducción de ruido adaptativa
        if self.config.noise_reduction_iterations > 0:
            for _ in range(self.config.noise_reduction_iterations):
                gray = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Suavizado gaussiano
        if self.config.gaussian_sigma > 0:
            gray = cv2.GaussianBlur(gray, (5, 5), self.config.gaussian_sigma)
        
        # Mejoras específicas por tipo de espécimen
        if specimen_type == "cartridge_case":
            if self.config.firing_pin_enhancement:
                gray = self._enhance_firing_pin_features(gray)
            if self.config.breech_face_enhancement:
                gray = self._enhance_breech_face_features(gray)
        elif specimen_type == "bullet":
            gray = self._enhance_striation_features(gray)
        
        # Operaciones morfológicas
        if self.config.morphology_kernel_size > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, 
                (self.config.morphology_kernel_size, self.config.morphology_kernel_size)
            )
            gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        
        return gray
    
    def _watershed_ballistic_optimized(self, image: np.ndarray, specimen_type: str) -> Optional[np.ndarray]:
        """
        Watershed optimizado específicamente para características balísticas
        
        Args:
            image: Imagen preprocesada
            specimen_type: Tipo de espécimen
            
        Returns:
            Imagen segmentada con etiquetas
        """
        if not SKIMAGE_AVAILABLE:
            return self._watershed_opencv_fallback(image)
        
        try:
            # Calcular gradiente multi-escala
            gradient = self._compute_multiscale_gradient(image)
            
            # Generar marcadores adaptativos
            markers = self._generate_adaptive_markers(image, specimen_type)
            
            # Aplicar watershed con parámetros optimizados
            segments = watershed(
                gradient, 
                markers, 
                compactness=self.config.compactness,
                watershed_line=self.config.watershed_line
            )
            
            # Post-procesamiento específico para balística
            segments = self._postprocess_ballistic_segments(segments, image, specimen_type)
            
            return segments
            
        except Exception as e:
            self.logger.error(f"Error en watershed balístico optimizado: {e}")
            return self._watershed_opencv_fallback(image)
    
    def _compute_multiscale_gradient(self, image: np.ndarray) -> np.ndarray:
        """
        Calcula gradiente multi-escala para mejor detección de bordes
        
        Args:
            image: Imagen de entrada
            
        Returns:
            Gradiente combinado
        """
        gradients = []
        scales = [1.0, 1.5, 2.0]
        
        for scale in scales:
            # Suavizar a diferentes escalas
            sigma = self.config.gaussian_sigma * scale
            smoothed = gaussian(image, sigma=sigma)
            
            # Calcular gradiente
            if self.config.gradient_method == "sobel":
                grad = sobel(smoothed)
            elif self.config.gradient_method == "scharr":
                grad_x = cv2.Scharr(smoothed.astype(np.float32), cv2.CV_64F, 1, 0)
                grad_y = cv2.Scharr(smoothed.astype(np.float32), cv2.CV_64F, 0, 1)
                grad = np.sqrt(grad_x**2 + grad_y**2)
            else:  # laplacian
                grad = cv2.Laplacian(smoothed.astype(np.float32), cv2.CV_64F)
                grad = np.abs(grad)
            
            gradients.append(grad)
        
        # Combinar gradientes con pesos
        weights = [0.5, 0.3, 0.2]
        combined_gradient = np.zeros_like(gradients[0])
        
        for grad, weight in zip(gradients, weights):
            combined_gradient += weight * grad
        
        # Normalizar
        combined_gradient = (combined_gradient - combined_gradient.min()) / \
                          (combined_gradient.max() - combined_gradient.min() + 1e-8)
        
        return combined_gradient
    
    def _generate_adaptive_markers(self, image: np.ndarray, specimen_type: str) -> np.ndarray:
        """
        Genera marcadores adaptativos basados en el tipo de espécimen
        
        Args:
            image: Imagen preprocesada
            specimen_type: Tipo de espécimen
            
        Returns:
            Imagen de marcadores
        """
        if not SKIMAGE_AVAILABLE:
            return self._generate_simple_markers(image)
        
        try:
            # Encontrar máximos locales
            local_maxima = peak_local_maxima(
                image,
                min_distance=self.config.marker_min_distance,
                threshold_rel=self.config.marker_threshold_rel,
                footprint=disk(self.config.marker_footprint_size)
            )
            
            # Crear imagen de marcadores
            markers = np.zeros_like(image, dtype=np.int32)
            
            # Añadir marcadores de máximos locales
            for i, (y, x) in enumerate(local_maxima):
                markers[y, x] = i + 1
            
            # Añadir marcadores específicos por tipo
            if specimen_type == "cartridge_case":
                markers = self._add_cartridge_specific_markers(markers, image)
            elif specimen_type == "bullet":
                markers = self._add_bullet_specific_markers(markers, image)
            
            # Etiquetar marcadores conectados
            markers = label(markers > 0).astype(np.int32)
            
            return markers
            
        except Exception as e:
            self.logger.error(f"Error generando marcadores adaptativos: {e}")
            return self._generate_simple_markers(image)
    
    def _add_cartridge_specific_markers(self, markers: np.ndarray, image: np.ndarray) -> np.ndarray:
        """
        Añade marcadores específicos para casquillos
        
        Args:
            markers: Marcadores existentes
            image: Imagen original
            
        Returns:
            Marcadores mejorados
        """
        # Detectar centro aproximado (percutor)
        center_y, center_x = np.array(image.shape) // 2
        
        # Buscar región circular central (percutor)
        y_coords, x_coords = np.ogrid[:image.shape[0], :image.shape[1]]
        center_mask = (x_coords - center_x)**2 + (y_coords - center_y)**2 < (min(image.shape) // 8)**2
        
        if np.any(center_mask):
            center_region = image[center_mask]
            if len(center_region) > 0:
                # Añadir marcador en el punto más brillante del centro
                center_max_idx = np.unravel_index(
                    np.argmax(image * center_mask), 
                    image.shape
                )
                next_label = markers.max() + 1
                markers[center_max_idx] = next_label
        
        return markers
    
    def _add_bullet_specific_markers(self, markers: np.ndarray, image: np.ndarray) -> np.ndarray:
        """
        Añade marcadores específicos para proyectiles
        
        Args:
            markers: Marcadores existentes
            image: Imagen original
            
        Returns:
            Marcadores mejorados
        """
        # Para proyectiles, buscar líneas de estrías
        # Aplicar filtro direccional para detectar estrías
        kernel_horizontal = np.array([[-1, -1, -1], [2, 2, 2], [-1, -1, -1]])
        kernel_vertical = np.array([[-1, 2, -1], [-1, 2, -1], [-1, 2, -1]])
        
        response_h = cv2.filter2D(image.astype(np.float32), -1, kernel_horizontal)
        response_v = cv2.filter2D(image.astype(np.float32), -1, kernel_vertical)
        
        # Combinar respuestas
        striation_response = np.maximum(np.abs(response_h), np.abs(response_v))
        
        # Encontrar picos en respuesta de estrías
        threshold = np.percentile(striation_response, 90)
        striation_peaks = striation_response > threshold
        
        # Añadir marcadores en picos de estrías
        peak_coords = np.where(striation_peaks)
        next_label = markers.max() + 1
        
        for y, x in zip(peak_coords[0][::10], peak_coords[1][::10]):  # Submuestrear
            if markers[y, x] == 0:  # Solo si no hay marcador existente
                markers[y, x] = next_label
                next_label += 1
        
        return markers
    
    def _enhance_firing_pin_features(self, image: np.ndarray) -> np.ndarray:
        """
        Mejora características del percutor
        
        Args:
            image: Imagen original
            
        Returns:
            Imagen con percutor mejorado
        """
        # Filtro circular para realzar formas circulares pequeñas
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        tophat = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
        
        # Combinar con imagen original
        enhanced = cv2.addWeighted(image, 0.7, tophat, 0.3, 0)
        
        return enhanced
    
    def _enhance_breech_face_features(self, image: np.ndarray) -> np.ndarray:
        """
        Mejora características de la cara de recámara
        
        Args:
            image: Imagen original
            
        Returns:
            Imagen con cara de recámara mejorada
        """
        # Filtro para realzar texturas finas
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(image, -1, kernel)
        
        # Combinar con imagen original
        enhanced = cv2.addWeighted(image, 0.6, sharpened, 0.4, 0)
        
        return enhanced
    
    def _enhance_striation_features(self, image: np.ndarray) -> np.ndarray:
        """
        Mejora características de estrías en proyectiles
        
        Args:
            image: Imagen original
            
        Returns:
            Imagen con estrías mejoradas
        """
        # Filtros direccionales para estrías
        kernel_h = np.array([[-1, -1, -1], [2, 2, 2], [-1, -1, -1]]) / 3
        kernel_v = np.array([[-1, 2, -1], [-1, 2, -1], [-1, 2, -1]]) / 3
        
        response_h = cv2.filter2D(image.astype(np.float32), -1, kernel_h)
        response_v = cv2.filter2D(image.astype(np.float32), -1, kernel_v)
        
        # Combinar respuestas
        enhanced = np.maximum(np.abs(response_h), np.abs(response_v))
        enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
        
        # Combinar con imagen original
        result = cv2.addWeighted(image, 0.5, enhanced, 0.5, 0)
        
        return result
    
    def _watershed_opencv_fallback(self, image: np.ndarray) -> np.ndarray:
        """
        Implementación de fallback usando OpenCV cuando scikit-image no está disponible
        
        Args:
            image: Imagen preprocesada
            
        Returns:
            Imagen segmentada
        """
        # Calcular gradiente
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        gradient = np.sqrt(grad_x**2 + grad_y**2)
        gradient = cv2.normalize(gradient, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Crear marcadores simples
        _, markers = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        markers = cv2.erode(markers, None, iterations=2)
        markers = cv2.dilate(markers, None, iterations=1)
        
        # Etiquetar marcadores
        _, markers = cv2.connectedComponents(markers)
        
        # Aplicar watershed de OpenCV
        markers = cv2.watershed(cv2.cvtColor(gradient, cv2.COLOR_GRAY2BGR), markers)
        
        return markers
    
    def _extract_regions(self, segments: np.ndarray, 
                        original_image: np.ndarray, 
                        preprocessed_image: np.ndarray) -> List[WatershedRegion]:
        """
        Extrae regiones de la imagen segmentada
        
        Args:
            segments: Imagen segmentada
            original_image: Imagen original
            preprocessed_image: Imagen preprocesada
            
        Returns:
            Lista de regiones extraídas
        """
        regions = []
        
        if not SKIMAGE_AVAILABLE:
            return self._extract_regions_opencv(segments, original_image)
        
        try:
            # Obtener propiedades de regiones
            props = regionprops(segments, intensity_image=original_image)
            
            for prop in props:
                if prop.area < self.config.min_region_area:
                    continue
                
                # Crear máscara de región
                mask = (segments == prop.label).astype(np.uint8)
                
                # Encontrar contorno
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not contours:
                    continue
                
                contour = max(contours, key=cv2.contourArea)
                
                # Calcular propiedades adicionales
                circularity = 4 * np.pi * prop.area / (prop.perimeter * prop.perimeter) if prop.perimeter > 0 else 0
                
                # Crear región
                region = WatershedRegion(
                    label=prop.label,
                    center=(int(prop.centroid[1]), int(prop.centroid[0])),
                    area=prop.area,
                    perimeter=prop.perimeter,
                    bounding_box=prop.bbox,
                    contour=contour,
                    mask=mask,
                    circularity=circularity,
                    eccentricity=prop.eccentricity,
                    solidity=prop.solidity,
                    aspect_ratio=prop.major_axis_length / prop.minor_axis_length if prop.minor_axis_length > 0 else 0,
                    extent=prop.extent,
                    mean_intensity=prop.mean_intensity,
                    std_intensity=np.std(original_image[mask > 0]),
                    min_intensity=prop.min_intensity,
                    max_intensity=prop.max_intensity
                )
                
                regions.append(region)
                
        except Exception as e:
            self.logger.error(f"Error extrayendo regiones: {e}")
        
        return regions
    
    def _filter_regions(self, regions: List[WatershedRegion], specimen_type: str) -> List[WatershedRegion]:
        """
        Filtra regiones basado en criterios geométricos y de calidad
        
        Args:
            regions: Lista de regiones
            specimen_type: Tipo de espécimen
            
        Returns:
            Lista de regiones filtradas
        """
        filtered = []
        
        for region in regions:
            # Filtro por área
            if not (self.config.min_region_area <= region.area <= self.config.max_region_area):
                continue
            
            # Filtro por circularidad
            if region.circularity < self.config.min_circularity:
                continue
            
            # Filtro por excentricidad
            if region.eccentricity > self.config.max_eccentricity:
                continue
            
            # Filtros específicos por tipo
            if specimen_type == "cartridge_case":
                # Para casquillos, preferir regiones más circulares
                if region.circularity < 0.4 and region.area < 500:
                    continue
            elif specimen_type == "bullet":
                # Para proyectiles, permitir regiones más alargadas
                if region.aspect_ratio > 5.0:
                    continue
            
            filtered.append(region)
        
        return filtered
    
    def _classify_regions(self, regions: List[WatershedRegion], specimen_type: str) -> List[WatershedRegion]:
        """
        Clasifica regiones según su tipo probable
        
        Args:
            regions: Lista de regiones
            specimen_type: Tipo de espécimen
            
        Returns:
            Lista de regiones clasificadas
        """
        for region in regions:
            if specimen_type == "cartridge_case":
                region.region_type = self._classify_cartridge_region(region)
            elif specimen_type == "bullet":
                region.region_type = self._classify_bullet_region(region)
            else:
                region.region_type = "unknown"
            
            # Marcar como característica balística si es relevante
            region.is_ballistic_feature = region.region_type in [
                "firing_pin", "breech_face", "extractor", "ejector", "striation"
            ]
        
        return regions
    
    def _classify_cartridge_region(self, region: WatershedRegion) -> str:
        """
        Clasifica región de casquillo
        
        Args:
            region: Región a clasificar
            
        Returns:
            Tipo de región
        """
        # Clasificación basada en tamaño y forma
        if region.area < 200 and region.circularity > 0.7:
            return "firing_pin"
        elif region.area > 1000 and region.circularity > 0.5:
            return "breech_face"
        elif region.eccentricity > 0.8 and region.aspect_ratio > 2.0:
            return "extractor"
        elif region.area < 500 and region.circularity > 0.6:
            return "ejector"
        else:
            return "primer"
    
    def _classify_bullet_region(self, region: WatershedRegion) -> str:
        """
        Clasifica región de proyectil
        
        Args:
            region: Región a clasificar
            
        Returns:
            Tipo de región
        """
        # Clasificación basada en forma y orientación
        if region.aspect_ratio > 3.0 and region.eccentricity > 0.8:
            return "striation"
        elif region.circularity > 0.6 and region.area < 300:
            return "land_mark"
        elif region.area > 500:
            return "groove_mark"
        else:
            return "surface_mark"
    
    def _calculate_confidence_scores(self, regions: List[WatershedRegion], 
                                   original_image: np.ndarray) -> List[WatershedRegion]:
        """
        Calcula puntuaciones de confianza para cada región
        
        Args:
            regions: Lista de regiones
            original_image: Imagen original
            
        Returns:
            Lista de regiones con confianza calculada
        """
        for region in regions:
            # Calcular componentes de confianza
            area_score = self._calculate_area_score(region)
            circularity_score = region.circularity
            contrast_score = self._calculate_contrast_score(region, original_image)
            edge_score = self._calculate_edge_strength(region, original_image)
            texture_score = self._calculate_texture_score(region, original_image)
            position_score = self._calculate_position_score(region, original_image.shape)
            
            # Combinar puntuaciones con pesos
            weights = self.config.confidence_weights
            confidence = (
                weights['area'] * area_score +
                weights['circularity'] * circularity_score +
                weights['contrast'] * contrast_score +
                weights['edge_strength'] * edge_score +
                weights['texture'] * texture_score +
                weights['position'] * position_score
            )
            
            region.confidence = np.clip(confidence, 0.0, 1.0)
            region.edge_strength = edge_score
            region.texture_score = texture_score
        
        return regions
    
    def _calculate_area_score(self, region: WatershedRegion) -> float:
        """Calcula puntuación basada en área"""
        # Normalizar área a rango [0, 1]
        area_normalized = (region.area - self.config.min_region_area) / \
                         (self.config.max_region_area - self.config.min_region_area)
        
        # Función gaussiana centrada en área óptima
        optimal_area = 0.3  # 30% del rango
        score = np.exp(-((area_normalized - optimal_area) ** 2) / (2 * 0.2 ** 2))
        
        return np.clip(score, 0.0, 1.0)
    
    def _calculate_contrast_score(self, region: WatershedRegion, image: np.ndarray) -> float:
        """Calcula puntuación basada en contraste"""
        if region.std_intensity == 0:
            return 0.0
        
        # Contraste local normalizado
        contrast = region.std_intensity / (region.mean_intensity + 1e-8)
        
        # Normalizar a [0, 1]
        contrast_normalized = np.clip(contrast / 0.5, 0.0, 1.0)
        
        return contrast_normalized
    
    def _calculate_edge_strength(self, region: WatershedRegion, image: np.ndarray) -> float:
        """Calcula fuerza de bordes en la región"""
        # Calcular gradiente en la región
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Extraer gradiente en la región
        region_gradient = gradient_magnitude[region.mask > 0]
        
        if len(region_gradient) == 0:
            return 0.0
        
        # Promedio de magnitud de gradiente normalizado
        edge_strength = np.mean(region_gradient) / 255.0
        
        return np.clip(edge_strength, 0.0, 1.0)
    
    def _calculate_texture_score(self, region: WatershedRegion, image: np.ndarray) -> float:
        """Calcula puntuación de textura"""
        # Extraer región de imagen
        region_pixels = image[region.mask > 0]
        
        if len(region_pixels) < 10:
            return 0.0
        
        # Calcular varianza local como medida de textura
        texture_variance = np.var(region_pixels)
        
        # Normalizar
        texture_score = np.clip(texture_variance / 1000.0, 0.0, 1.0)
        
        return texture_score
    
    def _calculate_position_score(self, region: WatershedRegion, image_shape: Tuple[int, int]) -> float:
        """Calcula puntuación basada en posición"""
        height, width = image_shape
        center_x, center_y = region.center
        
        # Distancia al centro de la imagen
        image_center_x, image_center_y = width // 2, height // 2
        distance_to_center = np.sqrt((center_x - image_center_x)**2 + (center_y - image_center_y)**2)
        
        # Normalizar por diagonal de imagen
        max_distance = np.sqrt(width**2 + height**2) / 2
        normalized_distance = distance_to_center / max_distance
        
        # Puntuación más alta para regiones centrales
        position_score = 1.0 - normalized_distance
        
        return np.clip(position_score, 0.0, 1.0)
    
    def _generate_simple_markers(self, image: np.ndarray) -> np.ndarray:
        """
        Genera marcadores simples cuando scikit-image no está disponible
        
        Args:
            image: Imagen de entrada
            
        Returns:
            Imagen de marcadores
        """
        # Threshold adaptativo
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Operaciones morfológicas para limpiar
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Etiquetar componentes conectados
        _, markers = cv2.connectedComponents(binary)
        
        return markers
    
    def _extract_regions_opencv(self, segments: np.ndarray, 
                               original_image: np.ndarray) -> List[WatershedRegion]:
        """
        Extrae regiones usando solo OpenCV
        
        Args:
            segments: Imagen segmentada
            original_image: Imagen original
            
        Returns:
            Lista de regiones
        """
        regions = []
        
        # Obtener etiquetas únicas
        unique_labels = np.unique(segments)
        
        for label in unique_labels:
            if label <= 0:  # Ignorar fondo y bordes
                continue
            
            # Crear máscara
            mask = (segments == label).astype(np.uint8)
            
            # Calcular área
            area = np.sum(mask)
            if area < self.config.min_region_area:
                continue
            
            # Encontrar contorno
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            
            contour = max(contours, key=cv2.contourArea)
            
            # Calcular propiedades básicas
            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
            
            center_x = int(M["m10"] / M["m00"])
            center_y = int(M["m01"] / M["m00"])
            
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            
            # Bounding box
            bbox = cv2.boundingRect(contour)
            
            # Intensidades
            region_pixels = original_image[mask > 0]
            mean_intensity = np.mean(region_pixels)
            std_intensity = np.std(region_pixels)
            min_intensity = np.min(region_pixels)
            max_intensity = np.max(region_pixels)
            
            # Crear región
            region = WatershedRegion(
                label=int(label),
                center=(center_x, center_y),
                area=float(area),
                perimeter=float(perimeter),
                bounding_box=bbox,
                contour=contour,
                mask=mask,
                circularity=float(circularity),
                mean_intensity=float(mean_intensity),
                std_intensity=float(std_intensity),
                min_intensity=float(min_intensity),
                max_intensity=float(max_intensity)
            )
            
            regions.append(region)
        
        return regions
    
    def _watershed_classic(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Watershed clásico basado en gradiente"""
        if not SKIMAGE_AVAILABLE:
            return self._watershed_opencv_fallback(image)
        
        gradient = sobel(image)
        markers = self._generate_adaptive_markers(image, "unknown")
        segments = watershed(gradient, markers)
        return segments
    
    def _watershed_distance(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Watershed basado en transformada de distancia"""
        if not SKIMAGE_AVAILABLE:
            return self._watershed_opencv_fallback(image)
        
        # Binarizar imagen
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Transformada de distancia
        if SCIPY_AVAILABLE:
            distance = distance_transform_edt(binary)
        else:
            distance = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        
        # Encontrar máximos locales
        local_maxima = peak_local_maxima(distance, min_distance=20)
        markers = np.zeros_like(image, dtype=np.int32)
        for i, (y, x) in enumerate(local_maxima):
            markers[y, x] = i + 1
        
        # Aplicar watershed
        segments = watershed(-distance, markers)
        return segments
    
    def _watershed_marker_controlled(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Watershed controlado por marcadores específicos"""
        if not SKIMAGE_AVAILABLE:
            return self._watershed_opencv_fallback(image)
        
        # Generar marcadores específicos
        markers = self._generate_adaptive_markers(image, "unknown")
        
        # Calcular gradiente
        gradient = sobel(image)
        
        # Aplicar watershed
        segments = watershed(gradient, markers, compactness=self.config.compactness)
        return segments
    
    def _watershed_hybrid(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Método híbrido combinando múltiples enfoques"""
        if not SKIMAGE_AVAILABLE:
            return self._watershed_opencv_fallback(image)
        
        # Combinar gradiente y distancia
        gradient = sobel(image)
        
        # Binarizar para distancia
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if SCIPY_AVAILABLE:
            distance = distance_transform_edt(binary)
        else:
            distance = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        
        # Combinar métricas
        combined = 0.6 * gradient + 0.4 * (distance / distance.max())
        
        # Generar marcadores
        markers = self._generate_adaptive_markers(image, "unknown")
        
        # Aplicar watershed
        segments = watershed(combined, markers, compactness=self.config.compactness)
        return segments
    
    def _postprocess_ballistic_segments(self, segments: np.ndarray, 
                                      image: np.ndarray, 
                                      specimen_type: str) -> np.ndarray:
        """
        Post-procesa segmentos específicamente para características balísticas
        
        Args:
            segments: Segmentos iniciales
            image: Imagen original
            specimen_type: Tipo de espécimen
            
        Returns:
            Segmentos post-procesados
        """
        if not SKIMAGE_AVAILABLE:
            return segments
        
        # Remover regiones pequeñas
        segments = remove_small_objects(segments, min_size=self.config.min_region_area)
        
        # Limpiar bordes
        segments = clear_border(segments)
        
        # Post-procesamiento específico por tipo
        if specimen_type == "cartridge_case":
            segments = self._postprocess_cartridge_segments(segments, image)
        elif specimen_type == "bullet":
            segments = self._postprocess_bullet_segments(segments, image)
        
        return segments
    
    def _postprocess_cartridge_segments(self, segments: np.ndarray, image: np.ndarray) -> np.ndarray:
        """Post-procesa segmentos de casquillos"""
        # Preservar regiones circulares centrales (percutor)
        center_y, center_x = np.array(image.shape) // 2
        
        # Crear máscara circular central
        y_coords, x_coords = np.ogrid[:image.shape[0], :image.shape[1]]
        center_mask = (x_coords - center_x)**2 + (y_coords - center_y)**2 < (min(image.shape) // 6)**2
        
        # Preservar segmentos en región central
        central_segments = segments * center_mask
        
        return segments
    
    def _postprocess_bullet_segments(self, segments: np.ndarray, image: np.ndarray) -> np.ndarray:
        """Post-procesa segmentos de proyectiles"""
        # Para proyectiles, preservar regiones alargadas (estrías)
        # Esto se maneja en el filtrado posterior
        return segments


def create_enhanced_watershed_detector(config_dict: Optional[Dict] = None) -> EnhancedWatershedROI:
    """
    Función de conveniencia para crear detector Watershed mejorado
    
    Args:
        config_dict: Diccionario de configuración
        
    Returns:
        Detector configurado
    """
    if config_dict:
        config = WatershedConfig(**config_dict)
    else:
        config = Watershedget_unified_config()
    
    return EnhancedWatershedROI(config)


if __name__ == "__main__":
    import argparse
    
    # Configurar logging
    logging.basicConfig(level=logging.INFO)
    
    # Parser de argumentos
    parser = argparse.ArgumentParser(description="Segmentación Watershed mejorada para ROI balísticas")
    parser.add_argument("input", help="Imagen de entrada")
    parser.add_argument("--output", "-o", help="Imagen de salida", default=None)
    parser.add_argument("--method", "-m", help="Método de watershed", 
                       choices=["classic", "distance", "marker_controlled", "hybrid", "ballistic_optimized"], 
                       default="ballistic_optimized")
    parser.add_argument("--type", "-t", help="Tipo de espécimen", 
                       choices=["cartridge_case", "bullet"], default="cartridge_case")
    
    args = parser.parse_args()
    
    # Crear detector
    config = WatershedConfig(method=WatershedMethod(args.method))
    detector = EnhancedWatershedROI(config)
    
    # Cargar imagen
    image = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: No se pudo cargar la imagen {args.input}")
        exit(1)
    
    # Segmentar ROI
    regions = detector.segment_roi(image, args.type)
    
    # Mostrar resultados
    print(f"Detectadas {len(regions)} regiones:")
    for i, region in enumerate(regions):
        print(f"\nRegión {i+1}:")
        print(f"  Tipo: {region.region_type}")
        print(f"  Centro: {region.center}")
        print(f"  Área: {region.area:.1f}")
        print(f"  Confianza: {region.confidence:.3f}")
        print(f"  Circularidad: {region.circularity:.3f}")
    
    # Guardar visualización si se especifica
    if args.output and len(regions) > 0:
        # Crear imagen de visualización
        vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        for region in regions:
            # Dibujar contorno
            cv2.drawContours(vis_image, [region.contour], -1, (0, 255, 0), 2)
            
            # Dibujar centro
            cv2.circle(vis_image, region.center, 3, (0, 0, 255), -1)
            
            # Añadir etiqueta
            label = f"{region.region_type} ({region.confidence:.2f})"
            cv2.putText(vis_image, label, 
                       (region.center[0] + 10, region.center[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imwrite(args.output, vis_image)
        print(f"Visualización guardada en: {args.output}")